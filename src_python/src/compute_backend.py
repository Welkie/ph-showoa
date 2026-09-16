from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

try:
    from numba import cuda  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cuda = None


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


def _evaluate_route_cpu(route: Sequence[int], snapshot: BackendSnapshot) -> RouteEval:
    nl = _normalize_route(route)
    if len(nl) < 2:
        return False, 0.0
    if nl[0] != snapshot.depot or nl[-1] != snapshot.depot:
        return False, 0.0
    if len(nl) == 2:
        return True, 0.0

    load = 0.0
    for node in nl[1:-1]:
        load += snapshot.delivery[node]
    if load > snapshot.capacity:
        return False, 0.0

    distance = 0.0
    time_val = snapshot.start_time
    prev = nl[0]
    for node in nl[1:]:
        load = load - snapshot.delivery[node] + snapshot.pickup[node]
        if load > snapshot.capacity:
            return False, 0.0

        time_val += snapshot.time[prev, node]
        if time_val > snapshot.end[node]:
            return False, 0.0
        if time_val < snapshot.start[node]:
            time_val = snapshot.start[node]
        time_val += snapshot.service[node]

        distance += snapshot.dist[prev, node]
        prev = node

    return True, snapshot.dispatch_cost + distance * snapshot.unit_cost


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
        return [(bool(feasible[i]), float(costs[i])) for i in range(len(routes))]


def create_backend(data, mode: str = "auto") -> BaseComputeBackend:
    requested = (mode or "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unknown compute backend: %s" % mode)

    snapshot = BackendSnapshot.from_data(data)
    if requested in {"auto", "cuda"}:
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
