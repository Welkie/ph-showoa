from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Any

import os
import multiprocessing
import numpy as np

import torch

try:
    from numba import cuda, njit  # type: ignore
    import warnings
    from numba.core.errors import NumbaPerformanceWarning
    warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
except Exception:  # pragma: no cover - optional dependency
    cuda: Any = None
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
        idx = cuda.grid(1)  # type: ignore[call-arg]
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
        idx = cuda.grid(1)  # type: ignore[call-arg]
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


class TorchComputeBackend(BaseComputeBackend):
    name = "torch_cuda"
    is_cuda = True
    multi_process_safe = False

    def __init__(
        self,
        snapshot: BackendSnapshot,
        device_str: str = "auto",
        strict_full_gpu: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not available")
        cuda_avail = torch.cuda.is_available()
        if device_str == "auto":
            device_str = "cuda" if cuda_avail else "cpu"
        elif device_str == "cuda" and not cuda_avail:
            raise RuntimeError("CUDA backend requested but CUDA is not available")
        if strict_full_gpu and (not cuda_avail or device_str != "cuda"):
            raise RuntimeError("strict full_gpu requires a CUDA device")
        super().__init__(snapshot)
        self.device = torch.device(device_str)
        self.is_cuda = (self.device.type == "cuda")
        self.strict_full_gpu = strict_full_gpu
        self.name = "torch_cuda" if self.is_cuda else "torch_cpu"
        self.depot = int(snapshot.depot)
        self.customer_num = int(snapshot.customer_num)
        self.capacity = float(snapshot.capacity)
        self.start_time = float(snapshot.start_time)
        self.dispatch_cost = float(snapshot.dispatch_cost)
        self.unit_cost = float(snapshot.unit_cost)

        self.delivery_t = torch.as_tensor(snapshot.delivery, dtype=torch.float32, device=self.device)
        self.pickup_t = torch.as_tensor(snapshot.pickup, dtype=torch.float32, device=self.device)
        self.start_t = torch.as_tensor(snapshot.start, dtype=torch.float32, device=self.device)
        self.end_t = torch.as_tensor(snapshot.end, dtype=torch.float32, device=self.device)
        self.service_t = torch.as_tensor(snapshot.service, dtype=torch.float32, device=self.device)
        self.dist_t = torch.as_tensor(snapshot.dist, dtype=torch.float32, device=self.device)
        self.time_t = torch.as_tensor(snapshot.time, dtype=torch.float32, device=self.device)

    def evaluate_route(self, route: Sequence[int]) -> RouteEval:
        if self.strict_full_gpu:
            raise RuntimeError("strict full_gpu requires tensor evaluation APIs")
        route_tensor = torch.as_tensor(route, dtype=torch.long, device=self.device).unsqueeze(0)
        length_tensor = torch.tensor([route_tensor.shape[1]], dtype=torch.long, device=self.device)
        feasible, distance = self.evaluate_routes_gpu(route_tensor, length_tensor)
        if bool(feasible[0].item()):
            return True, self.dispatch_cost + float(distance[0].item()) * self.unit_cost
        return False, 0.0

    def evaluate_routes_gpu(self, routes_t: torch.Tensor, lengths_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        100% Pure GPU tensor evaluation of an arbitrary batch of N routes on CUDA VRAM.
        Computes exact capacity feasibility (running delivery & pickup load constraints),
        time-window feasibility, and route distance in fully vectorized PyTorch operations.
        Zero host-device memory transfers.
        """
        if self.strict_full_gpu:
            if routes_t.device != self.device or lengths_t.device != self.device:
                raise RuntimeError("strict full_gpu received a host or foreign-device tensor")
            if routes_t.dtype != torch.long or lengths_t.dtype != torch.long:
                raise RuntimeError("strict full_gpu route tensors must use torch.long")
        N, L = routes_t.shape
        if N == 0 or L < 2:
            return torch.zeros(N, dtype=torch.bool, device=self.device), torch.zeros(N, dtype=torch.float32, device=self.device)

        prev = routes_t[:, :-1]
        curr = routes_t[:, 1:]

        # Endpoint check
        valid_endpoints = (routes_t[:, 0] == self.depot) & (routes_t[torch.arange(N, device=self.device), lengths_t - 1] == self.depot)

        # Distance
        dists = self.dist_t[prev, curr]
        col_indices = torch.arange(L - 1, device=self.device).unsqueeze(0).expand(N, -1)
        valid_mask = col_indices < (lengths_t.unsqueeze(1) - 1)
        dists = torch.where(valid_mask, dists, 0.0)
        total_dist = dists.sum(dim=1)

        # Capacity check
        delivs = self.delivery_t[curr]
        pickups = self.pickup_t[curr]
        delivs = torch.where(valid_mask, delivs, 0.0)
        pickups = torch.where(valid_mask, pickups, 0.0)

        init_deliv = delivs.sum(dim=1)
        load_valid = (init_deliv <= self.capacity + 1e-5)

        running_load = init_deliv.unsqueeze(1) - torch.cumsum(delivs, dim=1) + torch.cumsum(pickups, dim=1)
        load_valid = load_valid & ((running_load >= -1e-5) & (running_load <= self.capacity + 1e-5) | ~valid_mask).all(dim=1)

        # Time Windows check
        travel_times = self.time_t[prev, curr]
        window_start = self.start_t[curr]
        window_end = self.end_t[curr]
        s_times = self.service_t[curr]

        time_val = torch.full((N,), self.start_time, dtype=torch.float32, device=self.device)
        tw_valid = torch.ones((N,), dtype=torch.bool, device=self.device)

        for j in range(L - 1):
            is_active = j < (lengths_t - 1)
            t_travel = travel_times[:, j]
            t_start = window_start[:, j]
            t_end = window_end[:, j]
            t_serv = s_times[:, j]

            arrival = time_val + t_travel
            tw_valid = tw_valid & (~is_active | (arrival <= t_end + 1e-5))
            time_val = torch.where(is_active, torch.max(arrival, t_start) + t_serv, time_val)

        feasible = valid_endpoints & load_valid & tw_valid
        return feasible, total_dist

    def evaluate_routes(self, routes: Sequence[Sequence[int]]) -> List[RouteEval]:
        if self.strict_full_gpu:
            raise RuntimeError("strict full_gpu requires evaluate_routes_gpu()")
        if len(routes) == 0:
            return []
        packed, lengths = _pack_routes(routes, self.depot)
        if packed.shape[1] < 2:
            return [(False, 0.0) for _ in routes]

        routes_t = torch.as_tensor(packed, dtype=torch.long, device=self.device)
        lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        feasible, total_dist = self.evaluate_routes_gpu(routes_t, lengths_t)
        costs = torch.where(feasible, self.dispatch_cost + total_dist * self.unit_cost, 0.0)

        feas_np = feasible.cpu().numpy()
        costs_np = costs.cpu().numpy()
        return [(bool(f), float(c)) for f, c in zip(feas_np, costs_np)]

    def evaluate_population_tensor(
        self,
        pop_routes_t: torch.Tensor,
        pop_lengths_t: torch.Tensor,
        pop_route_counts_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.strict_full_gpu:
            tensors = (pop_routes_t, pop_lengths_t, pop_route_counts_t)
            if any(t.device != self.device for t in tensors):
                raise RuntimeError("strict full_gpu population tensors must stay on CUDA")
            if pop_routes_t.dtype != torch.long or pop_lengths_t.dtype != torch.long or pop_route_counts_t.dtype != torch.long:
                raise RuntimeError("strict full_gpu population indices must use torch.long")
        P, R, L = pop_routes_t.shape
        if L < 2:
            return (
                torch.zeros(P, dtype=torch.bool, device=self.device),
                torch.full((P,), float("inf"), dtype=torch.float32, device=self.device),
                torch.zeros(P, dtype=torch.long, device=self.device),
                torch.zeros(P, dtype=torch.float32, device=self.device),
            )

        prev = pop_routes_t[:, :, :-1]
        curr = pop_routes_t[:, :, 1:]

        dists = self.dist_t[prev, curr]
        col_indices = torch.arange(L - 1, device=self.device).view(1, 1, L - 1)
        valid_mask = col_indices < (pop_lengths_t.unsqueeze(-1) - 1)
        dists = torch.where(valid_mask, dists, 0.0)
        route_dists = dists.sum(dim=-1)

        route_indices = torch.arange(R, device=self.device).unsqueeze(0)
        route_mask = route_indices < pop_route_counts_t.unsqueeze(-1)
        total_distances = torch.where(route_mask, route_dists, 0.0).sum(dim=-1)

        end_indices = (pop_lengths_t - 1).clamp_min(0).unsqueeze(-1)
        end_nodes = pop_routes_t.gather(2, end_indices).squeeze(-1)
        endpoint_valid = (pop_routes_t[:, :, 0] == self.depot) & (end_nodes == self.depot)
        route_endpoint_valid = torch.where(route_mask, endpoint_valid, True).all(dim=-1)

        delivs = torch.where(valid_mask, self.delivery_t[curr], 0.0)
        pickups = torch.where(valid_mask, self.pickup_t[curr], 0.0)

        init_deliv = delivs.sum(dim=-1)
        running_load = init_deliv.unsqueeze(-1) - torch.cumsum(delivs, dim=-1) + torch.cumsum(pickups, dim=-1)
        load_valid = (init_deliv <= self.capacity + 1e-5) & ((running_load >= -1e-5) & (running_load <= self.capacity + 1e-5) | ~valid_mask).all(dim=-1)
        route_load_valid = torch.where(route_mask, load_valid, True).all(dim=-1)

        travel_times = self.time_t[prev, curr]
        window_start = self.start_t[curr]
        window_end = self.end_t[curr]
        s_times = self.service_t[curr]

        time_val = torch.full((P, R), self.start_time, dtype=torch.float32, device=self.device)
        tw_valid = torch.ones((P, R), dtype=torch.bool, device=self.device)

        for j in range(L - 1):
            is_active = j < (pop_lengths_t - 1)
            t_travel = travel_times[:, :, j]
            t_start = window_start[:, :, j]
            t_end = window_end[:, :, j]
            t_serv = s_times[:, :, j]

            arrival = time_val + t_travel
            tw_valid = tw_valid & (~is_active | (arrival <= t_end + 1e-5))
            time_val = torch.where(is_active, torch.max(arrival, t_start) + t_serv, time_val)

        route_tw_valid = torch.where(route_mask, tw_valid, True).all(dim=-1)

        # Vectorized exact 1-to-1 customer occurrence and boundary check on GPU
        active_nodes_mask = valid_mask & route_mask.unsqueeze(-1)
        nodes_in_routes = pop_routes_t[:, :, 1:]
        nodes_in_routes = torch.where(active_nodes_mask, nodes_in_routes, 0)
        is_depot = nodes_in_routes == self.depot
        nodes_in_routes = torch.where(is_depot, 0, nodes_in_routes)

        has_invalid = ((nodes_in_routes < 0) | (nodes_in_routes > self.customer_num)).any(dim=-1).any(dim=-1)
        flat_nodes = nodes_in_routes.reshape(P, -1)
        occ = torch.zeros((P, self.customer_num + 1), dtype=torch.int32, device=self.device)
        safe_flat = flat_nodes.clamp(0, self.customer_num)
        occ.scatter_add_(1, safe_flat, torch.ones_like(safe_flat, dtype=torch.int32))
        coverage_valid = (~has_invalid) & (occ[:, 1:self.customer_num + 1] == 1).all(dim=-1)

        pop_feasible = route_endpoint_valid & route_load_valid & route_tw_valid & coverage_valid

        active_routes = route_mask & (pop_lengths_t > 2)
        vehicle_counts = active_routes.sum(dim=-1).long()
        total_costs = vehicle_counts.float() * self.dispatch_cost + total_distances * self.unit_cost
        total_costs = torch.where(pop_feasible, total_costs, torch.tensor(float("inf"), device=self.device))

        return pop_feasible, total_costs, vehicle_counts, total_distances

    def evaluate_candidate_insertions_tensor(
        self, route_tensor: torch.Tensor, candidate_node: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates inserting candidate_node into all positions 1..L-1 of route_tensor in 1 pure GPU vector operation.
        Returns feasible_mask (L-1,) and costs (L-1,).
        """
        L = int(route_tensor.shape[0])
        if L < 2:
            return torch.zeros(0, dtype=torch.bool, device=self.device), torch.zeros(0, dtype=torch.float32, device=self.device)

        num_cands = L - 1
        positions = torch.arange(1, L, device=self.device)
        output_positions = torch.arange(L + 1, device=self.device).view(1, -1)
        source_positions = torch.where(
            output_positions < positions.view(-1, 1),
            output_positions,
            output_positions - 1,
        ).clamp_min(0)
        cand_routes = route_tensor.expand(num_cands, -1).gather(1, source_positions)
        cand_routes.scatter_(1, positions.view(-1, 1), candidate_node)

        cand_lengths = torch.full((num_cands,), L + 1, dtype=torch.long, device=self.device)
        feas, total_dist = self.evaluate_routes_gpu(cand_routes, cand_lengths)
        costs = torch.where(feas, self.dispatch_cost + total_dist * self.unit_cost, torch.tensor(float('inf'), device=self.device))
        return feas, costs

    def evaluate_candidate_batch_insertions_gpu(
        self, candidate_routes_list: List[torch.Tensor], candidate_lengths_list: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates a batch of arbitrary candidate route variations directly on GPU.
        Zero CPU conversions.
        """
        if not candidate_routes_list:
            return torch.zeros(0, dtype=torch.bool, device=self.device), torch.zeros(0, dtype=torch.float32, device=self.device)
        N = len(candidate_routes_list)
        max_len = max(candidate_lengths_list)
        batch_t = torch.full((N, max_len), self.depot, dtype=torch.long, device=self.device)
        lengths_t = torch.as_tensor(candidate_lengths_list, dtype=torch.long, device=self.device)
        for i in range(N):
            l = candidate_lengths_list[i]
            batch_t[i, :l] = candidate_routes_list[i][:l]
        return self.evaluate_routes_gpu(batch_t, lengths_t)

    def compute_lexicographic_scores(
        self, feas: torch.Tensor, v_counts: torch.Tensor, total_dists: torch.Tensor
    ) -> torch.Tensor:
        """
        Strict lexicographic ordering: NV primary (1e8 multiplier), TD secondary.
        Infeasible solutions receive a huge penalty (1e15).
        """
        scores = v_counts.to(torch.float64) * 1_000_000_000.0 + total_dists.to(torch.float64)
        infeasible = torch.full((), float("inf"), device=self.device, dtype=torch.float64)
        return torch.where(feas, scores, infeasible)

    def evaluate_insertions(self, route_nodes: Sequence[int], candidate_nodes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        if self.strict_full_gpu:
            raise RuntimeError("strict full_gpu requires tensor insertion APIs")
        if len(candidate_nodes) == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

        route_arr = np.asarray(route_nodes, dtype=np.int32)
        cand_arr = np.asarray(candidate_nodes, dtype=np.int32)
        k = len(route_arr)
        m = len(cand_arr)

        candidates_batch = []
        for cand in cand_arr:
            for pos in range(1, k):
                cand_route = np.insert(route_arr, pos, cand)
                candidates_batch.append(cand_route)

        evals = self.evaluate_routes(candidates_batch)
        feasible = np.zeros(m * k, dtype=np.int32)
        costs = np.zeros(m * k, dtype=np.float64)

        idx = 0
        for c_idx in range(m):
            for pos in range(1, k):
                out_idx = c_idx * k + pos
                f, c = evals[idx]
                if f:
                    feasible[out_idx] = 1
                    costs[out_idx] = c
                idx += 1

        return feasible, costs


def create_backend(data, mode: str = "auto") -> BaseComputeBackend:
    requested = (mode or "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unknown compute backend: %s" % mode)

    snapshot = BackendSnapshot.from_data(data)
    
    strict_full_gpu = getattr(data, "architecture", "legacy") == "full_gpu"
    if strict_full_gpu and requested == "cpu":
        raise RuntimeError("architecture=full_gpu requires compute_backend=cuda or auto")

    # Avoid even initializing the CPU JIT path during a strict GPU run.
    if not strict_full_gpu:
        try:
            dummy_nl = np.array([0, 0], dtype=np.int32)
            dummy_candidates = np.array([0], dtype=np.int32)
            _evaluate_route_cpu_kernel(
                dummy_nl,
                int(snapshot.depot), float(snapshot.start_time), float(snapshot.capacity), float(snapshot.dispatch_cost), float(snapshot.unit_cost),
                snapshot.delivery, snapshot.pickup, snapshot.start, snapshot.end, snapshot.service, snapshot.dist, snapshot.time
            )
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
        if torch is not None and torch.cuda.is_available():
            try:
                return TorchComputeBackend(
                    snapshot,
                    device_str="cuda",
                    strict_full_gpu=strict_full_gpu,
                )
            except Exception as e:
                if strict_full_gpu:
                    raise RuntimeError("Failed to initialize strict CUDA tensor backend") from e
                print("Failed to initialize TorchComputeBackend: %s" % e)

        if strict_full_gpu:
            raise RuntimeError("architecture=full_gpu requires a working PyTorch CUDA runtime")

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

    if strict_full_gpu:
        raise RuntimeError("No CUDA backend is available for architecture=full_gpu")
    return BaseComputeBackend(snapshot)


def evaluate_route_cpu(route: Sequence[int], data) -> RouteEval:
    return _evaluate_route_cpu(route, BackendSnapshot.from_data(data))


def evaluate_route_batch(routes: Sequence[Sequence[int]], data) -> List[RouteEval]:
    backend = getattr(data, "backend", None)
    if backend is None:
        return [_evaluate_route_cpu(route, BackendSnapshot.from_data(data)) for route in routes]
    return backend.evaluate_routes(routes)
