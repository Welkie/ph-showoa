"""Capture a complete, fixed-generation run and replay it with one host launch.

The generation loop is unrolled once during CPU_PREP, not during a run. CUDA
stream dependencies preserve operator order. Scheduling predicates and the
generation counter live on device. No conditional-graph-node API is required.
"""

import ctypes
import os
from dataclasses import dataclass

import numpy as np
from numba import config, cuda

from .config import HYBRID_MODE_SHO, HYBRID_MODE_WOA
from .gpu_graph_kernels import build_graph_kernels


class _DriverGraph:
    """Small CUDA Driver API bridge, sharing Numba's current CUDA context."""

    def __init__(self):
        loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
        try:
            self.lib = loader("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
        except OSError as exc:
            raise RuntimeError("CUDA Graph requires an NVIDIA CUDA driver") from exc
        ptr = ctypes.c_void_p
        signatures = {
            # The unversioned export takes only a stream; cuda.h maps the
            # two-argument API to _v2. ctypes must use the actual ABI symbol.
            "cuStreamBeginCapture_v2": [ptr, ctypes.c_int],
            "cuStreamEndCapture": [ptr, ctypes.POINTER(ptr)],
            "cuGraphInstantiateWithFlags": [ctypes.POINTER(ptr), ptr, ctypes.c_ulonglong],
            "cuGraphLaunch": [ptr, ptr],
            "cuGraphDestroy": [ptr],
            "cuGraphExecDestroy": [ptr],
            "cuGetErrorString": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
        }
        for name, argtypes in signatures.items():
            try:
                fn = getattr(self.lib, name)
            except AttributeError as exc:
                raise RuntimeError(
                    f"CUDA driver lacks {name}; a driver supporting CUDA Graph "
                    "instantiation with flags is required (CUDA 11.4+). "
                    "No CPU/host-loop fallback will be used."
                ) from exc
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int

    def call(self, name, *args):
        result = getattr(self.lib, name)(*args)
        if result:
            message = ctypes.c_char_p()
            self.lib.cuGetErrorString(result, ctypes.byref(message))
            detail = message.value.decode() if message.value else str(result)
            raise RuntimeError(f"{name} failed: {detail}")


@dataclass(frozen=True)
class _Step:
    name: str
    kernel: object
    blocks: int
    threads: int
    args: tuple


class CudaSearchGraph:
    def __init__(self, engine, buffers):
        self.simulated = bool(config.ENABLE_CUDASIM)
        self.stream = cuda.stream()
        self.driver = None
        self.graph = ctypes.c_void_p()
        self.executable = ctypes.c_void_p()
        self._closed = False
        self._compiled = []  # Keep modules alive until after graph destruction.
        data = engine.data
        self.max_iter = int(getattr(data, "max_iter", 1000))
        ls = int(getattr(data, "local_search_interval", 25))
        stag = int(getattr(data, "stagnation_interval", 50))
        migr = int(getattr(data, "migration_interval", 20))
        if self.max_iter < 0 or min(ls, stag, migr) <= 0:
            raise ValueError("max_iter must be >= 0 and search/migration intervals > 0")
        self.rng = cuda.device_array((engine.P, 4), dtype=np.uint32)
        self.seed = cuda.device_array(1, dtype=np.int64)
        self.host_seed = cuda.pinned_array(1, dtype=np.int64)
        self.generation = cuda.device_array(1, dtype=np.int64)
        self.parameters = cuda.device_array(2, dtype=np.float64)
        self.no_improve = cuda.device_array(1, dtype=np.int64)
        self.previous_best = cuda.device_array(1, dtype=np.float64)
        self.diversify_due = cuda.device_array(1, dtype=np.int32)
        self.history = cuda.device_array((max(1, self.max_iter), 4), dtype=np.float64)
        self.host_history = cuda.pinned_array(self.history.shape, dtype=np.float64)
        self.steps = self._make_steps(engine, buffers, ls, stag, migr)
        if not self.simulated:
            try:
                self._capture()
            except BaseException:
                self.close()
                raise

    def _make_steps(self, engine, buffers, ls, stag, migr):
        pop, nxt, cand = buffers[0:5], buffers[5:10], buffers[10:15]
        ibest, gbest = buffers[15:20], buffers[20:25]
        route, route2, unrouted, flags, scores = buffers[25:30]
        problem = buffers[30]
        k = engine.kernels
        search_blocks = engine.blocks if engine.ls_scope == "population" else engine.isl_blocks if engine.ls_scope == "island" else 1
        search_threads = engine.threads_per_block if engine.ls_scope in {"population", "island"} else 1
        g = build_graph_kernels(k)
        steps = []
        threads = engine.threads_per_block

        def add(name, kernel, blocks, args, block_threads=threads):
            steps.append(_Step(name, kernel, blocks, block_threads, tuple(args)))

        def bests(current):
            add("island_bests", k["update_island_bests"], engine.isl_blocks,
                (*current, *ibest, engine.num_islands, engine.island_size))
            add("global_best", k["update_global_best"], 1,
                (*ibest, *gbest, engine.num_islands), 1)

        add("reset", g["reset"], engine.blocks,
            (self.rng, self.seed, self.generation, self.parameters, ibest[2], gbest[2],
             ibest[3], ibest[4], gbest[3], gbest[4],
             self.no_improve, self.previous_best, self.diversify_due))
        add("init", k["init_population"], engine.blocks,
            (*pop, *nxt, *cand, route, route2, unrouted, flags, scores, problem,
             float(getattr(engine.data, "grasp_alpha_lo", 0.10)),
             float(getattr(engine.data, "grasp_alpha_hi", 0.40)), self.rng))
        bests(pop)
        mode_name = getattr(engine.data, "hybrid_mode", "ph_showoa")
        mode = 1 if mode_name == HYBRID_MODE_SHO else 2 if mode_name == HYBRID_MODE_WOA else 0
        # This loop constructs a DAG in CPU_PREP. No host generation loop is
        # executed during graph replay. Pointer parity is fixed at construction.
        for _ in range(self.max_iter):
            add("parameters", g["parameters"], 1,
                (self.generation, self.parameters, self.max_iter, mode,
                 gbest[4], self.previous_best), 1)
            add("update", g["update"], engine.blocks,
                (*pop, *nxt, *cand, *ibest, route, route2, unrouted, flags,
                 problem, self.parameters, self.generation, self.max_iter,
                 self.rng, engine.island_size))
            pop, nxt = nxt, pop
            # Record every accepted best before a destructive diversification.
            bests(pop)
            search_best = pop if engine.ls_scope == "population" else (ibest if engine.ls_scope == "island" else gbest)
            add("eliminate", g["eliminate"], search_blocks,
                (*search_best, route, unrouted, flags, problem, self.generation, ls), search_threads)
            add("search", g["search"], search_blocks,
                (*search_best, route, route2, problem, self.generation, ls), search_threads)
            bests(pop)
            add("publish_best", k["publish_global_best"], 1,
                (*gbest, *ibest, engine.num_islands), 1)
            add("stagnation", g["stagnation"], 1,
                (gbest[4], self.previous_best, self.no_improve, self.diversify_due, stag), 1)
            add("diversify", g["diversify"], engine.isl_blocks,
                (*pop, *cand, *ibest, route, unrouted, flags, problem, self.rng,
                 engine.num_islands, engine.island_size, self.diversify_due))
            add("migrate", g["migrate"], 1,
                (*pop, *ibest, engine.num_islands, engine.island_size,
                 self.generation, migr), 1)
            bests(pop)
            add("finish", g["finish"], 1,
                (self.generation, self.parameters, gbest[2], gbest[3], self.history), 1)
        return steps

    def _capture(self):
        self.driver = _DriverGraph()
        compiled = {}
        # Compile AND load all kernel modules before capture. No compilation,
        # host transfers, allocations or default-stream work inside capture.
        for step in self.steps:
            if step.name not in compiled:
                specialized = step.kernel.specialize(*step.args)
                for overload in specialized.overloads.values():
                    overload.bind()
                compiled[step.name] = specialized
        self._compiled = list(compiled.values())
        cuda.synchronize()
        handle = self.stream.handle
        self._stream_handle = ctypes.c_void_p(
            handle.value if hasattr(handle, "value") else int(handle)
        )
        self.driver.call("cuStreamBeginCapture_v2", self._stream_handle, 0)
        try:
            for step in self.steps:
                compiled[step.name][step.blocks, step.threads, self.stream](*step.args)
        except BaseException:
            # End an invalidated capture too, without masking the original error.
            abandoned = ctypes.c_void_p()
            self.driver.lib.cuStreamEndCapture(self._stream_handle, ctypes.byref(abandoned))
            if abandoned.value:
                self.driver.lib.cuGraphDestroy(abandoned)
            raise
        self.driver.call("cuStreamEndCapture", self._stream_handle, ctypes.byref(self.graph))
        self.driver.call("cuGraphInstantiateWithFlags", ctypes.byref(self.executable), self.graph, 0)

    def prepare_run(self, seed):
        if self._closed:
            raise RuntimeError("Search graph is closed")
        self.host_seed[0] = int(seed) & 0xFFFFFFFF
        self.seed.copy_to_device(self.host_seed, stream=self.stream)

    def launch(self):
        if self._closed:
            raise RuntimeError("Search graph is closed")
        if self.simulated:
            # Test-only emulation; explicitly reported as CPU simulation.
            for step in self.steps:
                step.kernel[step.blocks, step.threads, self.stream](*step.args)
        else:
            self.driver.call("cuGraphLaunch", self.executable, self._stream_handle)

    def synchronize(self):
        self.stream.synchronize()

    def read_history(self):
        self.history.copy_to_host(self.host_history, stream=self.stream)
        self.stream.synchronize()
        return self.host_history[:self.max_iter]

    def close(self):
        if self._closed:
            return
        # The caller synchronizes successful runs before closing. Destruction of
        # an executable graph is also safe for already-enqueued launches.
        try:
            if self.driver and self.executable.value:
                self.driver.call("cuGraphExecDestroy", self.executable)
        finally:
            try:
                if self.driver and self.graph.value:
                    self.driver.call("cuGraphDestroy", self.graph)
            finally:
                self.executable = ctypes.c_void_p()
                self.graph = ctypes.c_void_p()
                self._closed = True
