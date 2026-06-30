from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import multiprocessing
import os

try:
    from numba import cuda, njit  # type: ignore
    import warnings
    from numba.core.errors import NumbaPerformanceWarning
    warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
except Exception:  # pragma: no cover - optional dependency
    cuda = None
    def njit(*args, **kwargs):
        return lambda f: f

RouteEval = Tuple[bool, float]


@dataclass(frozen=True)
class BackendSnapshot:
    depot: int
    customer_num: int
    capacity: float
    start_time: float
    end_time: float
    dispatch_cost: float
    unit_cost: float
    delivery: np.ndarray
    pickup: np.ndarray
    start: np.ndarray
    end: np.ndarray
    service: np.ndarray
    dist: np.ndarray
    time: np.ndarray

    @classmethod
    def from_data(cls, data) -> "BackendSnapshot":
        size = data.customer_num + 1
        delivery = np.zeros(size, dtype=np.float64)
        pickup = np.zeros(size, dtype=np.float64)
        start = np.zeros(size, dtype=np.float64)
        end = np.zeros(size, dtype=np.float64)
        service = np.zeros(size, dtype=np.float64)

        for idx, point in enumerate(data.node):
            delivery[idx] = float(point.delivery)
            pickup[idx] = float(point.pickup)
            start[idx] = float(point.start)
            end[idx] = float(point.end)
            service[idx] = float(point.s_time)

        return cls(
            depot=int(data.DC),
            customer_num=int(data.customer_num),
            capacity=float(data.vehicle.capacity),
            start_time=float(data.start_time),
            end_time=float(data.end_time),
            dispatch_cost=float(data.vehicle.d_cost),
            unit_cost=float(data.vehicle.unit_cost),
            delivery=np.ascontiguousarray(delivery),
            pickup=np.ascontiguousarray(pickup),
            start=np.ascontiguousarray(start),
            end=np.ascontiguousarray(end),
            service=np.ascontiguousarray(service),
            dist=np.ascontiguousarray(np.asarray(data.dist, dtype=np.float64)),
            time=np.ascontiguousarray(np.asarray(data.time, dtype=np.float64)),
        )


def _normalize_route(route: Sequence[int]) -> List[int]:
    if isinstance(route, list):
        return route
    return [int(node) for node in route]


@njit(nogil=True, cache=True)
def _evaluate_route_cpu_kernel(
    nl, depot, start_time, capacity, dispatch_cost, unit_cost,
    delivery, pickup, start_window, end_window, service, dist, time_matrix
):
    length = len(nl)
    if length < 2:
        return False, 0.0
    if nl[0] != depot or nl[-1] != depot:
        return False, 0.0
    if length == 2:
        return True, 0.0

    load = 0.0
    for i in range(1, length - 1):
        load += delivery[nl[i]]
    if load > capacity:
        return False, 0.0

    distance = 0.0
    time_val = start_time
    prev = nl[0]
    for i in range(1, length):
        node = nl[i]
        load = load - delivery[node] + pickup[node]
        if load < 0 or load > capacity:
            return False, 0.0

        time_val += time_matrix[prev, node]
        if time_val > end_window[node]:
            return False, 0.0
        if time_val < start_window[node]:
            time_val = start_window[node]
        time_val += service[node]

        distance += dist[prev, node]
        prev = node

    return True, dispatch_cost + distance * unit_cost


def _evaluate_route_cpu(route: Sequence[int], snapshot: BackendSnapshot) -> RouteEval:
    nl = np.array(_normalize_route(route), dtype=np.int32)
    return _evaluate_route_cpu_kernel(
        nl,
        int(snapshot.depot), float(snapshot.start_time), float(snapshot.capacity), float(snapshot.dispatch_cost), float(snapshot.unit_cost),
        snapshot.delivery, snapshot.pickup, snapshot.start, snapshot.end, snapshot.service, snapshot.dist, snapshot.time
    )



@njit(nogil=True, cache=True)
def _evaluate_insertions_cpu_kernel(
    route, candidates, depot, start_time, capacity, dispatch_cost, unit_cost,
    delivery, pickup, start, end, service, dist, time_matrix,
    feasible, costs
):
    route_len = len(route)
    for c_idx in range(len(candidates)):
        candidate = candidates[c_idx]
        for pos in range(1, route_len):
            out_idx = c_idx * route_len + pos
            
            # Fast reject capacity
            load = 0.0
            for i in range(1, route_len - 1):
                node = route[i]
                load += delivery[node]
            load += delivery[candidate]
            if load > capacity:
                feasible[out_idx] = 0
                continue

            distance = 0.0
            time_val = start_time
            prev = route[0]
            
            is_feasible = True
            for i in range(1, route_len + 1):
                if i == pos:
                    node = candidate
                elif i < pos:
                    node = route[i]
                else:
                    node = route[i - 1]
                    
                if i == route_len:
                    break
                    
                load = load - delivery[node] + pickup[node]
                if load > capacity:
                    is_feasible = False
                    break

                time_val += time_matrix[prev, node]
                if time_val > end[node]:
                    is_feasible = False
                    break
                if time_val < start[node]:
                    time_val = start[node]
                time_val += service[node]

                distance += dist[prev, node]
                prev = node
                
            if is_feasible:
                feasible[out_idx] = 1
                costs[out_idx] = dispatch_cost + distance * unit_cost
            else:
                feasible[out_idx] = 0

def _evaluate_insertions_cpu(route: Sequence[int], candidates: Sequence[int], snapshot: BackendSnapshot) -> Tuple[np.ndarray, np.ndarray]:
    route_arr = np.asarray(route, dtype=np.int32)
    cand_arr = np.asarray(candidates, dtype=np.int32)
    feasible = np.zeros(len(cand_arr) * len(route_arr), dtype=np.int32)
    costs = np.zeros(len(cand_arr) * len(route_arr), dtype=np.float64)
    
    _evaluate_insertions_cpu_kernel(
        route_arr, cand_arr, snapshot.depot, snapshot.start_time, snapshot.capacity,
        snapshot.dispatch_cost, snapshot.unit_cost, snapshot.delivery, snapshot.pickup,
        snapshot.start, snapshot.end, snapshot.service, snapshot.dist, snapshot.time,
        feasible, costs
    )
    return feasible, costs


def _pack_routes(routes: Sequence[Sequence[int]], depot: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(routes) == 0:
        return np.zeros((0, 0), dtype=np.int32), np.zeros(0, dtype=np.int32)

    lengths = np.asarray([len(route) for route in routes], dtype=np.int32)
    max_len = int(lengths.max()) if len(lengths) > 0 else 0
    packed = np.full((len(routes), max_len), depot, dtype=np.int32)
    for index, route in enumerate(routes):
        packed[index, : len(route)] = np.asarray(route, dtype=np.int32)
    return packed, lengths


if cuda is not None:  # pragma: no cover - optional GPU path

    @cuda.jit
    def _evaluate_route_batch_kernel(
        routes,
        lengths,
        depot,
        start_time,
        capacity,
        dispatch_cost,
        unit_cost,
        delivery,
        pickup,
        start,
        end,
        service,
        dist,
        time_matrix,
        feasible,
        costs,
    ):
        idx = cuda.grid(1)
        if idx >= routes.shape[0]:
            return

        length = lengths[idx]
        if length < 2:
            feasible[idx] = 0
            costs[idx] = 0.0
            return

        if routes[idx, 0] != depot or routes[idx, length - 1] != depot:
            feasible[idx] = 0
            costs[idx] = 0.0
            return

        if length == 2:
            feasible[idx] = 1
            costs[idx] = 0.0
            return

        load = 0.0
        for pos in range(1, length - 1):
            node = routes[idx, pos]
            load += delivery[node]
        if load > capacity:
            feasible[idx] = 0
            costs[idx] = 0.0
            return

        distance = 0.0
        time_val = start_time
        prev = routes[idx, 0]
        for pos in range(1, length):
            node = routes[idx, pos]
            load = load - delivery[node] + pickup[node]
            if load > capacity:
                feasible[idx] = 0
                costs[idx] = 0.0
                return

            time_val += time_matrix[prev, node]
            if time_val > end[node]:
                feasible[idx] = 0
                costs[idx] = 0.0
                return
            if time_val < start[node]:
                time_val = start[node]
            time_val += service[node]

            distance += dist[prev, node]
            prev = node

        feasible[idx] = 1
        costs[idx] = dispatch_cost + distance * unit_cost

    @cuda.jit
    def _evaluate_insertions_cuda_kernel(
        route, candidates, depot, start_time, capacity, dispatch_cost, unit_cost,
        delivery, pickup, start, end, service, dist, time_matrix,
        feasible, costs
    ):
        idx = cuda.grid(1)
        route_len = len(route)
        
        if idx >= len(candidates) * route_len:
            return
            
        c_idx = idx // route_len
        pos = idx % route_len
        
        # Fast reject positions
        if pos == 0:
            feasible[idx] = 0
            return
            
        candidate = candidates[c_idx]

        load = 0.0
        for i in range(1, route_len - 1):
            node = route[i]
            load += delivery[node]
        load += delivery[candidate]
        if load > capacity:
            feasible[idx] = 0
            return

        distance = 0.0
        time_val = start_time
        prev = route[0]

        is_feasible = True
        for i in range(1, route_len + 1):
            if i == pos:
                node = candidate
            elif i < pos:
                node = route[i]
            else:
                node = route[i - 1]
                
            if i == route_len:
                break
                
            load = load - delivery[node] + pickup[node]
            if load > capacity:
                is_feasible = False
                break

            time_val += time_matrix[prev, node]
            if time_val > end[node]:
                is_feasible = False
                break
            if time_val < start[node]:
                time_val = start[node]
            time_val += service[node]

            distance += dist[prev, node]
            prev = node

        if is_feasible:
            feasible[idx] = 1
            costs[idx] = dispatch_cost + distance * unit_cost
        else:
            feasible[idx] = 0

class BaseComputeBackend:
    name = "cpu"
    is_cuda = False
    multi_process_safe = True

    def __init__(self, snapshot: BackendSnapshot) -> None:
        self.snapshot = snapshot

    def evaluate_route(self, route: Sequence[int]) -> RouteEval:
        return _evaluate_route_cpu(route, self.snapshot)

    def evaluate_routes(self, routes: Sequence[Sequence[int]]) -> List[RouteEval]:
        return [_evaluate_route_cpu(route, self.snapshot) for route in routes]

    def evaluate_insertions(self, route_nodes: Sequence[int], candidate_nodes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        return _evaluate_insertions_cpu(route_nodes, candidate_nodes, self.snapshot)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


class CudaComputeBackend(BaseComputeBackend):  # pragma: no cover - exercised only with CUDA
    name = "cuda"
    is_cuda = True
    multi_process_safe = False

    def __init__(self, snapshot: BackendSnapshot) -> None:
        if cuda is None:
            raise RuntimeError("Numba CUDA is not available")
        if not cuda.is_available():
            raise RuntimeError("CUDA runtime is not available")
        super().__init__(snapshot)
        self._device_cache = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_device_cache"] = None
        return state

    def _ensure_device_cache(self):
        if self._device_cache is not None:
            return self._device_cache
        self._device_cache = {
            "delivery": cuda.to_device(self.snapshot.delivery),
            "pickup": cuda.to_device(self.snapshot.pickup),
            "start": cuda.to_device(self.snapshot.start),
            "end": cuda.to_device(self.snapshot.end),
            "service": cuda.to_device(self.snapshot.service),
            "dist": cuda.to_device(self.snapshot.dist),
            "time": cuda.to_device(self.snapshot.time),
        }
        return self._device_cache

    def evaluate_routes(self, routes: Sequence[Sequence[int]]) -> List[RouteEval]:
        if len(routes) == 0:
            return []

        packed, lengths = _pack_routes(routes, self.snapshot.depot)
        device_cache = self._ensure_device_cache()
        routes_d = cuda.to_device(packed)
        lengths_d = cuda.to_device(lengths)
        feasible_d = cuda.device_array(len(routes), dtype=np.int32)
        costs_d = cuda.device_array(len(routes), dtype=np.float64)

        threads_per_block = 128
        blocks = (len(routes) + threads_per_block - 1) // threads_per_block
        _evaluate_route_batch_kernel[blocks, threads_per_block](
            routes_d,
            lengths_d,
            int(self.snapshot.depot),
            float(self.snapshot.start_time),
            float(self.snapshot.capacity),
            float(self.snapshot.dispatch_cost),
            float(self.snapshot.unit_cost),
            device_cache["delivery"],
            device_cache["pickup"],
            device_cache["start"],
            device_cache["end"],
            device_cache["service"],
            device_cache["dist"],
            device_cache["time"],
            feasible_d,
            costs_d,
        )

        feasible = feasible_d.copy_to_host()
        costs = costs_d.copy_to_host()
        return [(bool(f), float(c)) for f, c in zip(feasible, costs)]

    def evaluate_insertions(self, route_nodes: Sequence[int], candidate_nodes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        if len(candidate_nodes) == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

        route_arr = np.asarray(route_nodes, dtype=np.int32)
        cand_arr = np.asarray(candidate_nodes, dtype=np.int32)
        
        device_cache = self._ensure_device_cache()
        route_d = cuda.to_device(route_arr)
        cand_d = cuda.to_device(cand_arr)
        
        total_evals = len(cand_arr) * len(route_arr)
        feasible_d = cuda.device_array(total_evals, dtype=np.int32)
        costs_d = cuda.device_array(total_evals, dtype=np.float64)

        threads_per_block = 256
        blocks = (total_evals + threads_per_block - 1) // threads_per_block
        _evaluate_insertions_cuda_kernel[blocks, threads_per_block](
            route_d, cand_d,
            int(self.snapshot.depot),
            float(self.snapshot.start_time),
            float(self.snapshot.capacity),
            float(self.snapshot.dispatch_cost),
            float(self.snapshot.unit_cost),
            device_cache["delivery"],
            device_cache["pickup"],
            device_cache["start"],
            device_cache["end"],
            device_cache["service"],
            device_cache["dist"],
            device_cache["time"],
            feasible_d,
            costs_d,
        )

        feasible = feasible_d.copy_to_host()
        costs = costs_d.copy_to_host()
        return feasible, costs


_worker_idx = None
_request_queue = None
_response_queues = None


def init_pool_worker(id_queue, request_queue, response_queues):
    global _worker_idx, _request_queue, _response_queues
    _worker_idx = id_queue.get()
    _request_queue = request_queue
    _response_queues = response_queues


class GpuProxyBackend(BaseComputeBackend):
    name = "cuda_proxy"
    is_cuda = True
    multi_process_safe = True

    def __init__(self, snapshot: BackendSnapshot, num_workers: int):
        super().__init__(snapshot)
        self._request_queue = multiprocessing.Queue()
        self._response_queues = [multiprocessing.Queue() for _ in range(num_workers + 1)]
        self._main_process_idx = num_workers

        self.id_queue = multiprocessing.Queue()
        for i in range(num_workers):
            self.id_queue.put(i)

        self._gpu_worker = multiprocessing.Process(
            target=self._gpu_worker_loop,
            args=(snapshot, self._request_queue, self._response_queues),
            daemon=True
        )
        self._gpu_worker.start()

    @staticmethod
    def _gpu_worker_loop(snapshot, request_queue, response_queues):
        try:
            backend = CudaComputeBackend(snapshot)
        except Exception as e:
            for q in response_queues:
                q.put((None, e))
            return

        import queue
        while True:
            requests = []
            try:
                worker_idx, routes = request_queue.get()
                if worker_idx is None:
                    break
                requests.append((worker_idx, routes))
            except Exception:
                break

            while not request_queue.empty():
                try:
                    worker_idx, routes = request_queue.get_nowait()
                    if worker_idx is None:
                        request_queue.put((None, None))
                        break
                    requests.append((worker_idx, routes))
                except queue.Empty:
                    break
                except Exception:
                    break

            try:
                all_routes = []
                slices = []
                start = 0
                
                requests_insert = []
                requests_routes = []
                for w_idx, payload in requests:
                    if isinstance(payload, tuple) and len(payload) == 3 and payload[0] == "insertions":
                        requests_insert.append((w_idx, payload[1], payload[2]))
                    else:
                        requests_routes.append((w_idx, payload))
                
                if requests_routes:
                    for w_idx, r in requests_routes:
                        all_routes.extend(r)
                        slices.append((w_idx, start, start + len(r)))
                        start += len(r)

                    #import sys
                    #print(f"[GPU Worker] Processing mega-batch of {len(all_routes)} routes from {len(requests_routes)} workers", file=sys.stderr)
                    #sys.stderr.flush()
                    results = backend.evaluate_routes(all_routes)
                    #print(f"[GPU Worker] Finished mega-batch", file=sys.stderr)
                    #sys.stderr.flush()

                    for w_idx, start_idx, end_idx in slices:
                        response_queues[w_idx].put((w_idx, results[start_idx:end_idx]))
                        
                for w_idx, r_nodes, c_nodes in requests_insert:
                    results = backend.evaluate_insertions(r_nodes, c_nodes)
                    response_queues[w_idx].put((w_idx, results))

            except Exception as e:
                import sys
                print(f"[GPU Worker] Error processing mega-batch: {e}", file=sys.stderr)
                sys.stderr.flush()
                for w_idx, _ in requests:
                    response_queues[w_idx].put((w_idx, e))

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpickleable queue and process objects before sending over IPC
        state["_request_queue"] = None
        state["_response_queues"] = None
        state["id_queue"] = None
        state["_gpu_worker"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def evaluate_route(self, route: Sequence[int]) -> RouteEval:
        results = self.evaluate_routes([route])
        return results[0]

    def evaluate_routes(self, routes: Sequence[Sequence[int]]) -> List[RouteEval]:
        global _worker_idx, _request_queue, _response_queues
        idx = _worker_idx

        # Use global inherited queues in worker process, or self queues in main process
        req_queue = _request_queue if _request_queue is not None else self._request_queue
        res_queues = _response_queues if _response_queues is not None else self._response_queues

        if idx is None:
            idx = self._main_process_idx

        req_queue.put((idx, routes))
        resp_idx, results = res_queues[idx].get()
        if isinstance(results, Exception):
            raise results
        return results

    def evaluate_insertions(self, route_nodes: Sequence[int], candidate_nodes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        global _worker_idx, _request_queue, _response_queues
        idx = _worker_idx

        req_queue = _request_queue if _request_queue is not None else self._request_queue
        res_queues = _response_queues if _response_queues is not None else self._response_queues

        if idx is None:
            idx = self._main_process_idx

        req_queue.put((idx, ("insertions", route_nodes, candidate_nodes)))
        resp_idx, results = res_queues[idx].get()
        if isinstance(results, Exception):
            raise results
        return results

    def shutdown(self):
        try:
            self._request_queue.put((None, None))
            self._gpu_worker.join(timeout=1.0)
        except Exception:
            pass


def create_backend(data, mode: str = "auto") -> BaseComputeBackend:
    requested = (mode or "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unknown compute backend: %s" % mode)

    snapshot = BackendSnapshot.from_data(data)
    
    # Pre-compile JIT kernels in the main process to avoid multiprocessing cache race condition on Windows
    try:
        dummy_nl = np.array([0, 0], dtype=np.int32)
        dummy_candidates = np.array([0], dtype=np.int32)
        
        # Compile _evaluate_route_cpu_kernel
        _evaluate_route_cpu_kernel(
            dummy_nl,
            int(snapshot.depot), float(snapshot.start_time), float(snapshot.capacity), float(snapshot.dispatch_cost), float(snapshot.unit_cost),
            snapshot.delivery, snapshot.pickup, snapshot.start, snapshot.end, snapshot.service, snapshot.dist, snapshot.time
        )
        
        # Compile _evaluate_insertions_cpu_kernel
        dummy_feasible = np.zeros(2, dtype=np.int32)
        dummy_costs = np.zeros(2, dtype=np.float64)
        _evaluate_insertions_cpu_kernel(
            dummy_nl, dummy_candidates,
            int(snapshot.depot), float(snapshot.start_time), float(snapshot.capacity), float(snapshot.dispatch_cost), float(snapshot.unit_cost),
            snapshot.delivery, snapshot.pickup, snapshot.start, snapshot.end, snapshot.service, snapshot.dist, snapshot.time,
            dummy_feasible, dummy_costs
        )
    except Exception as e:
        print("Warning: JIT pre-compilation failed: %s" % e)

    if requested in {"auto", "cuda"}:
        cuda_available = False
        if cuda is not None:
            try:
                cuda_available = cuda.is_available()
            except Exception:
                pass

        if cuda_available:
            workers = getattr(data, "parallel_workers", 1)
            if workers != 1:
                try:
                    num_workers = workers
                    if num_workers <= 0:
                        num_workers = os.cpu_count() or 1
                    return GpuProxyBackend(snapshot, num_workers)
                except Exception as e:
                    print("Failed to initialize multi-process GPU backend: %s. Falling back to single-process CUDA." % e)
            try:
                return CudaComputeBackend(snapshot)
            except Exception:
                if requested == "cuda":
                    print("CUDA backend requested but unavailable. Falling back to CPU backend.")

    return BaseComputeBackend(snapshot)


def evaluate_route_cpu(route: Sequence[int], data) -> RouteEval:
    return _evaluate_route_cpu(route, BackendSnapshot.from_data(data))


def evaluate_route_batch(routes: Sequence[Sequence[int]], data) -> List[RouteEval]:
    backend = getattr(data, "backend", None)
    if backend is None:
        return [_evaluate_route_cpu(route, BackendSnapshot.from_data(data)) for route in routes]
    return backend.evaluate_routes(routes)
