"""Pure Numba CUDA / CPU Solver Engine for PH-SHOWOA (SA-RCRS-GRASP).

Enforces strict compliance with:
CPU_PREP run=N
    ↓
RUN_GPU_BEGIN run=N
    ↓
Toàn bộ initialization + SA + RCRS-GRASP +
SHO/WOA + objective + acceptance +
local search + migration trên CUDA
    ↓
RUN_GPU_END run=N
    ↓
CPU_DECODE sau khi hoàn tất toàn bộ runs
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Tuple

import numpy as np

from . import state
from .config import (
    HYBRID_MODE_SHO,
    HYBRID_MODE_WOA,
    PRECISION,
)
from .gpu_kernels import build_kernel_bundle
from .solution import Route, Solution

try:
    from numba import cuda
except ImportError:
    cuda = None


def _dynamic_parameters(iter_idx: int, max_iter: int) -> Tuple[float, float]:
    if max_iter <= 0:
        return 0.0, 0.0
    a = 2.0 - 2.0 * (float(iter_idx) / float(max_iter))
    p_hybrid = 0.5 * (1.0 + math.cos(math.pi * float(iter_idx) / float(max_iter)))
    return a, p_hybrid


def _mode_probability(p_hybrid: float, data: Any) -> float:
    hybrid_mode = getattr(data, "hybrid_mode", "ph_showoa")
    if hybrid_mode == HYBRID_MODE_SHO:
        return 1.0
    if hybrid_mode == HYBRID_MODE_WOA:
        return 0.0
    return p_hybrid


def prepare_problem_data(data: Any):
    customer_num = int(data.customer_num)
    depot = int(data.DC)
    capacity = float(data.vehicle.capacity)
    start_time = float(data.start_time)
    dispatch_cost = float(data.vehicle.d_cost)
    unit_cost = float(data.vehicle.unit_cost)

    delivery = np.zeros(customer_num + 1, dtype=np.float64)
    pickup = np.zeros(customer_num + 1, dtype=np.float64)
    start_tw = np.zeros(customer_num + 1, dtype=np.float64)
    end_tw = np.zeros(customer_num + 1, dtype=np.float64)
    service = np.zeros(customer_num + 1, dtype=np.float64)

    for idx, pt in enumerate(data.node):
        delivery[idx] = float(pt.delivery)
        pickup[idx] = float(pt.pickup)
        start_tw[idx] = float(pt.start)
        end_tw[idx] = float(pt.end)
        service[idx] = float(pt.s_time)

    dist_matrix = np.ascontiguousarray(np.array(data.dist, dtype=np.float64))
    time_matrix = np.ascontiguousarray(np.array(data.time, dtype=np.float64))

    return (
        depot, capacity, start_time, dispatch_cost, unit_cost,
        delivery, pickup, start_tw, end_tw, service, dist_matrix, time_matrix,
        0, 0, customer_num
    )


class GpuEngine:
    def __init__(self, data: Any, is_cuda: bool = False):
        self.data = data
        self.is_cuda = is_cuda and (cuda is not None and (cuda.is_available() or os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"))
        self.prob_data = prepare_problem_data(data)
        self.kernels = build_kernel_bundle(self.is_cuda)

        self.P = int(getattr(data, "p_size", 36))
        self.num_islands = int(getattr(data, "num_islands", 6))
        if self.P % self.num_islands != 0:
            self.num_islands = 1
        self.island_size = self.P // self.num_islands

        self.N = int(data.customer_num)
        self.R = min(self.N + 5, 105)
        self.L = self.N + 2

        self.threads_per_block = 32
        self.blocks = (self.P + self.threads_per_block - 1) // self.threads_per_block
        self.isl_blocks = (self.num_islands + self.threads_per_block - 1) // self.threads_per_block

    def _allocate_buffers(self):
        P, R, L, N, num_islands = self.P, self.R, self.L, self.N, self.num_islands

        if self.is_cuda:
            pop_nodes = cuda.device_array((P, R, L), dtype=np.int32)
            pop_rlen = cuda.device_array((P, R), dtype=np.int32)
            pop_nr = cuda.device_array(P, dtype=np.int32)
            pop_dist = cuda.device_array(P, dtype=np.float64)
            pop_cost = cuda.device_array(P, dtype=np.float64)

            next_nodes = cuda.device_array((P, R, L), dtype=np.int32)
            next_rlen = cuda.device_array((P, R), dtype=np.int32)
            next_nr = cuda.device_array(P, dtype=np.int32)
            next_dist = cuda.device_array(P, dtype=np.float64)
            next_cost = cuda.device_array(P, dtype=np.float64)

            cand_nodes = cuda.device_array((P, R, L), dtype=np.int32)
            cand_rlen = cuda.device_array((P, R), dtype=np.int32)
            cand_nr = cuda.device_array(P, dtype=np.int32)
            cand_dist = cuda.device_array(P, dtype=np.float64)
            cand_cost = cuda.device_array(P, dtype=np.float64)

            ibest_nodes = cuda.device_array((num_islands, R, L), dtype=np.int32)
            ibest_rlen = cuda.device_array((num_islands, R), dtype=np.int32)
            ibest_nr = cuda.device_array(num_islands, dtype=np.int32)
            ibest_dist = cuda.device_array(num_islands, dtype=np.float64)
            ibest_cost = cuda.device_array(num_islands, dtype=np.float64)

            gbest_nodes = cuda.device_array((1, R, L), dtype=np.int32)
            gbest_rlen = cuda.device_array((1, R), dtype=np.int32)
            gbest_nr = cuda.device_array(1, dtype=np.int32)
            gbest_dist = cuda.device_array(1, dtype=np.float64)
            gbest_cost = cuda.device_array(1, dtype=np.float64)

            scratch_route = cuda.device_array((P, L), dtype=np.int32)
            scratch_route2 = cuda.device_array((P, L), dtype=np.int32)
            scratch_unrouted = cuda.device_array((P, N + 1), dtype=np.int32)
            scratch_flags = cuda.device_array((P, N + 1), dtype=np.int32)
            scratch_scores = cuda.device_array((P, N + 1), dtype=np.float64)

            # Upload problem data to device
            prob_device = (
                self.prob_data[0], self.prob_data[1], self.prob_data[2], self.prob_data[3], self.prob_data[4],
                cuda.to_device(self.prob_data[5]), cuda.to_device(self.prob_data[6]),
                cuda.to_device(self.prob_data[7]), cuda.to_device(self.prob_data[8]),
                cuda.to_device(self.prob_data[9]), cuda.to_device(self.prob_data[10]),
                cuda.to_device(self.prob_data[11]), self.prob_data[12], self.prob_data[13], self.prob_data[14]
            )
        else:
            pop_nodes = np.zeros((P, R, L), dtype=np.int32)
            pop_rlen = np.zeros((P, R), dtype=np.int32)
            pop_nr = np.zeros(P, dtype=np.int32)
            pop_dist = np.zeros(P, dtype=np.float64)
            pop_cost = np.zeros(P, dtype=np.float64)

            next_nodes = np.zeros((P, R, L), dtype=np.int32)
            next_rlen = np.zeros((P, R), dtype=np.int32)
            next_nr = np.zeros(P, dtype=np.int32)
            next_dist = np.zeros(P, dtype=np.float64)
            next_cost = np.zeros(P, dtype=np.float64)

            cand_nodes = np.zeros((P, R, L), dtype=np.int32)
            cand_rlen = np.zeros((P, R), dtype=np.int32)
            cand_nr = np.zeros(P, dtype=np.int32)
            cand_dist = np.zeros(P, dtype=np.float64)
            cand_cost = np.zeros(P, dtype=np.float64)

            ibest_nodes = np.zeros((num_islands, R, L), dtype=np.int32)
            ibest_rlen = np.zeros((num_islands, R), dtype=np.int32)
            ibest_nr = np.zeros(num_islands, dtype=np.int32)
            ibest_dist = np.zeros(num_islands, dtype=np.float64)
            ibest_cost = np.zeros(num_islands, dtype=np.float64)

            gbest_nodes = np.zeros((1, R, L), dtype=np.int32)
            gbest_rlen = np.zeros((1, R), dtype=np.int32)
            gbest_nr = np.zeros(1, dtype=np.int32)
            gbest_dist = np.zeros(1, dtype=np.float64)
            gbest_cost = np.zeros(1, dtype=np.float64)

            scratch_route = np.zeros((P, L), dtype=np.int32)
            scratch_route2 = np.zeros((P, L), dtype=np.int32)
            scratch_unrouted = np.zeros((P, N + 1), dtype=np.int32)
            scratch_flags = np.zeros((P, N + 1), dtype=np.int32)
            scratch_scores = np.zeros((P, N + 1), dtype=np.float64)

            prob_device = self.prob_data

        return (
            pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
            next_nodes, next_rlen, next_nr, next_dist, next_cost,
            cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
            ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
            gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost,
            scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
            prob_device
        )

    def _init_rng(self, seed: int):
        P = self.P
        rng = np.zeros((P, 4), dtype=np.uint32)
        for s in range(P):
            rng[s, 0] = np.uint32((seed + s * 1337) & 0xFFFFFFFF) | 1
            rng[s, 1] = np.uint32((seed + s * 2749 + 362436069) & 0xFFFFFFFF) | 1
            rng[s, 2] = np.uint32((seed + s * 5171 + 521288629) & 0xFFFFFFFF) | 1
            rng[s, 3] = np.uint32((seed + s * 7919 + 88675123) & 0xFFFFFFFF) | 1
        if self.is_cuda:
            return cuda.to_device(rng)
        return rng

    def run_solve(self, best_s: Solution):
        total_runs = int(getattr(self.data, "runs", 30))
        max_iter = int(getattr(self.data, "max_iter", 1000))
        ls_interval = int(getattr(self.data, "local_search_interval", 25))
        stag_interval = int(getattr(self.data, "stagnation_interval", 50))
        migr_interval = int(getattr(self.data, "migration_interval", 20))
        alpha_lo = float(getattr(self.data, "grasp_alpha_lo", 0.10))
        alpha_hi = float(getattr(self.data, "grasp_alpha_hi", 0.40))
        base_seed = int(getattr(self.data, "seed", 42))

        backend_label = "Numba CUDA" if self.is_cuda else "Numba CPU Reference"
        print(f"[GpuEngine] Running pure {backend_label} solver (P={self.P}, islands={self.num_islands})", flush=True)

        start_total = time.perf_counter()
        completed_runs = 0

        # Pre-allocate device buffers ONCE (0 re-allocations)
        buffers = self._allocate_buffers()
        (
            pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
            next_nodes, next_rlen, next_nr, next_dist, next_cost,
            cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
            ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
            gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost,
            scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
            prob_device
        ) = buffers

        k = self.kernels

        for run in range(1, total_runs + 1):
            run_seed = base_seed + run * 100003
            rng_states = self._init_rng(run_seed)

            print(f"CPU_PREP run={run}", flush=True)
            print(f"RUN_GPU_BEGIN run={run}", flush=True)

            # Reset ibest and gbest
            if self.is_cuda:
                ibest_nr.copy_to_device(np.zeros(self.num_islands, dtype=np.int32))
                gbest_nr.copy_to_device(np.zeros(1, dtype=np.int32))
            else:
                ibest_nr.fill(0)
                gbest_nr.fill(0)

            # 1. Initialization kernel
            if self.is_cuda:
                k["init_population"][self.blocks, self.threads_per_block](
                    pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                    scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
                    prob_device, alpha_lo, alpha_hi, rng_states
                )
                k["update_island_bests"][self.isl_blocks, self.threads_per_block](
                    pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                    ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                    self.num_islands, self.island_size
                )
                k["update_global_best"][1, 1](
                    ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                    gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost,
                    self.num_islands
                )
            else:
                k["init_population"](
                    pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                    scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
                    prob_device, alpha_lo, alpha_hi, rng_states
                )
                k["update_island_bests"](
                    pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                    ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                    self.num_islands, self.island_size
                )
                k["update_global_best"](
                    ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                    gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost,
                    self.num_islands
                )

            # 2. Main Generation Loop (ZERO CPU SYNC)
            cur_pop = (pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost)
            nxt_pop = (next_nodes, next_rlen, next_nr, next_dist, next_cost)

            for gen in range(1, max_iter + 1):
                iter_idx = gen - 1
                a, p_hybrid = _dynamic_parameters(iter_idx, max_iter)
                p_mode = _mode_probability(p_hybrid, self.data)

                # SHO / WOA update
                if self.is_cuda:
                    k["update_population"][self.blocks, self.threads_per_block](
                        cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                        nxt_pop[0], nxt_pop[1], nxt_pop[2], nxt_pop[3], nxt_pop[4],
                        cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                        scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                        prob_device, a, p_mode, iter_idx, max_iter, rng_states, self.island_size
                    )
                else:
                    k["update_population"](
                        cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                        nxt_pop[0], nxt_pop[1], nxt_pop[2], nxt_pop[3], nxt_pop[4],
                        cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                        scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                        prob_device, a, p_mode, iter_idx, max_iter, rng_states, self.island_size
                    )

                # Ping-pong swap
                cur_pop, nxt_pop = nxt_pop, cur_pop

                # Periodic route elimination and deep local search
                if gen % ls_interval == 0:
                    if self.is_cuda:
                        k["route_elimination"][self.blocks, self.threads_per_block](
                            cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                            scratch_route, scratch_unrouted, scratch_flags, prob_device, 5
                        )
                        k["local_search"][self.blocks, self.threads_per_block](
                            cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                            scratch_route, scratch_route2, prob_device, 2
                        )
                    else:
                        k["route_elimination"](
                            cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                            scratch_route, scratch_unrouted, scratch_flags, prob_device, 5
                        )
                        k["local_search"](
                            cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                            scratch_route, scratch_route2, prob_device, 2
                        )

                # Periodic Stagnation Diversification
                if gen % stag_interval == 0:
                    if self.is_cuda:
                        k["stagnation_diversify"][self.isl_blocks, self.threads_per_block](
                            cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                            scratch_route, scratch_unrouted, scratch_flags,
                            prob_device, rng_states, self.num_islands, self.island_size
                        )
                    else:
                        k["stagnation_diversify"](
                            cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                            scratch_route, scratch_unrouted, scratch_flags,
                            prob_device, rng_states, self.num_islands, self.island_size
                        )

                # Periodic Island Migration
                if self.num_islands > 1 and gen % migr_interval == 0:
                    if self.is_cuda:
                        k["island_migration"][1, 1](
                            cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                            ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                            self.num_islands, self.island_size
                        )
                    else:
                        k["island_migration"](
                            cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                            ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                            self.num_islands, self.island_size
                        )

                # Update best trackers
                if self.is_cuda:
                    k["update_island_bests"][self.isl_blocks, self.threads_per_block](
                        cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                        self.num_islands, self.island_size
                    )
                    k["update_global_best"][1, 1](
                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                        gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost,
                        self.num_islands
                    )
                else:
                    k["update_island_bests"](
                        cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                        self.num_islands, self.island_size
                    )
                    k["update_global_best"](
                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                        gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost,
                        self.num_islands
                    )

            if self.is_cuda:
                cuda.synchronize()

            print(f"RUN_GPU_END run={run}", flush=True)

            # 3. CPU_DECODE (Single sync at end of run)
            if self.is_cuda:
                h_gbest_nodes = gbest_nodes.copy_to_host()
                h_gbest_rlen = gbest_rlen.copy_to_host()
                h_gbest_nr = gbest_nr.copy_to_host()
                h_gbest_dist = gbest_dist.copy_to_host()
                h_gbest_cost = gbest_cost.copy_to_host()
            else:
                h_gbest_nodes = gbest_nodes
                h_gbest_rlen = gbest_rlen
                h_gbest_nr = gbest_nr
                h_gbest_dist = gbest_dist
                h_gbest_cost = gbest_cost

            run_nv = int(h_gbest_nr[0])
            run_dist = float(h_gbest_dist[0])
            run_cost = float(h_gbest_cost[0])

            run_best_sol = Solution(self.data)
            run_best_sol.route_list = []
            for r in range(run_nv):
                l = int(h_gbest_rlen[0, r])
                if l > 2:
                    rt = Route()
                    rt.node_list = [int(h_gbest_nodes[0, r, i]) for i in range(l)]
                    rt.update(self.data)
                    run_best_sol.route_list.append(rt)
            run_best_sol.cal_cost(self.data)

            is_valid = run_best_sol.check(self.data, False)
            if not is_valid:
                print(f"Warning: Run {run} solution check returned False", flush=True)

            # Update best_s
            is_better = False
            if best_s.len() == 0 or best_s.cost == float("inf") or len(best_s.route_list) == 0:
                is_better = True
            elif run_best_sol.len() < best_s.len():
                is_better = True
            elif run_best_sol.len() == best_s.len() and run_best_sol.cost < best_s.cost - PRECISION:
                is_better = True

            if is_better and is_valid:
                best_s.copy_from(run_best_sol)
                state.best_s_cost = best_s.cost
                state.find_best_run = run
                state.find_best_time = int(time.perf_counter() - start_total)
                td = best_s.cost - 2000.0 * best_s.len()
                print(f"Best solution update: {best_s.cost:.4f} (NV={best_s.len()}, TD={td:.4f})", flush=True)

            print(f"Run {run} finishes", flush=True)
            completed_runs += 1

        total_time = int(time.perf_counter() - start_total)
        print("------------Summary-----------", flush=True)
        print(f"Total {completed_runs} runs, total consumed {total_time} sec", flush=True)
        best_s.output(self.data)
        best_s.check(self.data)
        sys.stdout.flush()
        return True


def run_solver(data: Any, best_s: Solution):
    """Entry point for the pure Numba solver."""
    use_cuda = getattr(data, "compute_backend", "auto") == "cuda"
    if getattr(data, "compute_backend", "auto") == "auto":
        use_cuda = (cuda is not None and (cuda.is_available() or os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"))

    engine = GpuEngine(data, is_cuda=use_cuda)
    return engine.run_solve(best_s)
