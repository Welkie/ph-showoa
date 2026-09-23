"""PH-SHOWOA: one CUDA Graph launch per run, then CPU decode/check.

Graph construction and buffer allocation happen once before runs. The CPU
reference path remains available only as a separate reference implementation.
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
    ratio = min(max(float(iter_idx) / float(max_iter), 0.0), 1.0)
    p_hybrid = max(0.15, 0.5 * (1.0 - ratio))
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
    dispatch_cost = 2000.0  # Fixed paper score weighting: 2000.0 * NV + 1.0 * TD
    unit_cost = 1.0
    max_vehicles = int(data.vehicle.max_num)

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
        max_vehicles, 0, customer_num
    )


class GpuEngine:
    def __init__(self, data: Any, is_cuda: bool = False):
        self.data = data
        if int(data.DC) != 0:
            raise ValueError("GPU encoding requires depot 0 and customer IDs 1..N")
        if int(data.customer_num) < 1:
            raise ValueError("At least one customer is required")
        if int(getattr(data, "max_iter", 1000)) < 0:
            raise ValueError("max_iter must be nonnegative")
        for name, default in (("runs", 30), ("local_search_interval", 25),
                              ("stagnation_interval", 50), ("migration_interval", 20),
                              ("output_per_gens", 25)):
            if int(getattr(data, name, default)) <= 0:
                raise ValueError(name + " must be positive")
        self.is_cuda = is_cuda and (cuda is not None and (cuda.is_available() or os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"))
        if is_cuda and not self.is_cuda:
            raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback")
        self.prob_data = prepare_problem_data(data)
        sa_t0 = float(getattr(data, "sa_t0", 100.0))
        sa_alpha = float(getattr(data, "sa_alpha", 0.95))
        sa_tmin = float(getattr(data, "sa_tmin", 0.1))
        sa_itermax = int(getattr(data, "sa_iterations", getattr(data, "sa_itermax", 100)))
        mutation = float(getattr(data, "sho_mutation_prob", 0.35))
        diversify = float(getattr(data, "diversify_ratio", 0.40))
        if not (0 < sa_tmin < sa_t0 and 0 < sa_alpha < 1 and sa_itermax >= 0):
            raise ValueError("Invalid SA initialization parameters")
        if not (0 <= mutation <= 1 and 0 <= diversify <= 1):
            raise ValueError("Mutation/diversification probabilities must be in [0, 1]")
        self.kernels = build_kernel_bundle(self.is_cuda, int(data.customer_num),
                                           sa_t0, sa_alpha, sa_tmin, sa_itermax,
                                           mutation, diversify,
                                           bool(getattr(data, "gpu_2opt_star", True)))

        self.P = int(getattr(data, "p_size", 32))
        self.num_islands = int(getattr(data, "num_islands", 4))
        if self.P <= 0 or self.num_islands <= 0:
            raise ValueError("Population size and number of islands must be positive")
        if self.P % self.num_islands != 0:
            print(f"[GpuEngine] P={self.P} is not divisible by requested islands={self.num_islands}; using one island", flush=True)
            self.num_islands = 1
        self.island_size = self.P // self.num_islands
        self.ls_scope = getattr(data, "gpu_ls_scope", "population")
        if self.ls_scope not in {"population", "island", "global"}:
            raise ValueError("gpu_ls_scope must be population, island or global")

        self.N = int(data.customer_num)
        self.R = max(1, self.N)  # At most one non-empty route per customer.
        self.L = self.N + 2

        self.threads_per_block = 32
        self.blocks = (self.P + self.threads_per_block - 1) // self.threads_per_block
        self.isl_blocks = (self.num_islands + self.threads_per_block - 1) // self.threads_per_block
        self.search_description = (
            f"[GpuEngine] SA_RCRS_GRASP: P={self.P}, islands={self.num_islands}, "
            f"SA iterations={sa_itermax}, local_search={self.ls_scope}, "
            f"2opt_star={bool(getattr(data, 'gpu_2opt_star', True))}; cost=2000*NV+TD")

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

    def _run_cuda_solve(self, best_s: Solution):
        from .gpu_graph import CudaSearchGraph

        total_runs = int(getattr(self.data, "runs", 30))
        out_interval = int(getattr(self.data, "output_per_gens", 25))
        if total_runs <= 0 or out_interval <= 0:
            raise ValueError("runs and output_per_gens must be positive")
        base_seed = int(getattr(self.data, "seed", 42))
        start_total = time.perf_counter()
        print("CPU_PREP: allocating buffers and building full-run CUDA Graph", flush=True)
        buffers = self._allocate_buffers()
        device_best = buffers[20:25]
        host_best = [cuda.pinned_array(buf.shape, dtype=buf.dtype) for buf in device_best]
        graph = CudaSearchGraph(self, buffers)
        label = "CUDA SIMULATOR (CPU, no real graph)" if graph.simulated else "Numba CUDA Graph"
        print(f"[GpuEngine] {label}: P={self.P}, islands={self.num_islands}, "
              f"graph nodes={len(graph.steps)}; logs are printed after each run", flush=True)
        try:
            for run in range(1, total_runs + 1):
                print(f"---------------------------------Run {run} ({label})---------------------------", flush=True)
                graph.prepare_run(base_seed + run * 100003)
                print(f"  Run {run} CPU_PREP: seed uploaded; buffers and graph reused", flush=True)
                marker = "RUN_SIM" if graph.simulated else "RUN_GPU"
                print(f"  Run {run} {marker}_BEGIN", flush=True)
                graph.launch()
                graph.synchronize()
                print(f"  Run {run} {marker}_END", flush=True)

                # All device-to-host copies and Python solution work are after
                # completion of the whole run, never between generations.
                for device, host in zip(device_best, host_best):
                    device.copy_to_host(host, stream=graph.stream)
                history = graph.read_history()  # Also waits for best copies.
                for gen, (a, p_mode, nv, td) in enumerate(history, start=1):
                    if gen == 1 or gen == graph.max_iter or gen % out_interval == 0:
                        print(f"[{label}] Gen: {gen}. a {a:.4f}, p_hybrid {p_mode:.4f}. "
                              f"Best NV {int(nv)}, Best TD {td:.4f}", flush=True)

                print(f"  Run {run} CPU_DECODE", flush=True)
                h_nodes, h_rlen, h_nr, h_dist, _ = host_best
                run_best_sol = Solution(self.data)
                for r in range(int(h_nr[0])):
                    length = int(h_rlen[0, r])
                    if length > 2:
                        route = Route()
                        route.node_list = [int(h_nodes[0, r, i]) for i in range(length)]
                        route.update(self.data)
                        run_best_sol.route_list.append(route)
                run_best_sol.cal_cost(self.data)
                is_valid = run_best_sol.check(self.data, False)
                if not is_valid:
                    print(f"Warning: Run {run} solution check returned False", flush=True)
                is_better = (
                    best_s.len() == 0 or best_s.cost == float("inf")
                    or run_best_sol.cost < best_s.cost - PRECISION
                )
                if is_valid and is_better:
                    best_s.copy_from(run_best_sol)
                    state.best_s_cost = best_s.cost
                    state.find_best_run = run
                    print(f"  Run {run} Best solution update: {best_s.cost:.4f} "
                          f"(NV={best_s.len()}, TD={best_s.cost - best_s.len() * 2000.0:.4f})", flush=True)
                print(f"Run {run} finishes | Run Best: NV={int(h_nr[0])}, TD={h_dist[0]:.4f} "
                      f"| Global Best: NV={best_s.len()}, "
                      f"TD={best_s.cost - best_s.len() * 2000.0:.4f}", flush=True)
        finally:
            graph.close()
        print("------------Summary-----------", flush=True)
        print(f"Total {total_runs} runs, total consumed {int(time.perf_counter() - start_total)} sec", flush=True)
        best_s.output(self.data)
        if not best_s.check(self.data):
            raise RuntimeError("No feasible final solution passed CPU verification")
        sys.stdout.flush()
        return True

    def run_solve(self, best_s: Solution):
        print(self.search_description, flush=True)
        if self.is_cuda:
            return self._run_cuda_solve(best_s)
        return self._run_reference_solve(best_s)

    def _run_reference_solve(self, best_s: Solution):
        """CPU reference only; CUDA runs never enter this host generation loop."""
        total_runs = int(getattr(self.data, "runs", 30))
        max_iter = int(getattr(self.data, "max_iter", 1000))
        ls_interval = int(getattr(self.data, "local_search_interval", 25))
        stag_interval = int(getattr(self.data, "stagnation_interval", 50))
        migr_interval = int(getattr(self.data, "migration_interval", 20))
        alpha_lo = float(getattr(self.data, "grasp_alpha_lo", 0.10))
        alpha_hi = float(getattr(self.data, "grasp_alpha_hi", 0.40))
        base_seed = int(getattr(self.data, "seed", 42))

        backend_label = "Numba CPU Reference"
        print(f"[GpuEngine] Running pure {backend_label} solver (P={self.P}, islands={self.num_islands})", flush=True)

        start_total = time.perf_counter()
        completed_runs = 0

        # Pre-allocate CPU reference buffers once
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

            print(f"---------------------------------Run {run} (100% Pure {backend_label} Engine)---------------------------", flush=True)
            print(f"  Run {run} [CPU_TELEMETRY] Running pure Numba CPU Reference mode (P={self.P}, islands={self.num_islands})", flush=True)

            print(f"  Run {run} CPU_PREP: seed/config ready; launching on {backend_label}", flush=True)
            print(f"  Run {run} RUN_CPU_BEGIN", flush=True)

            # Reset ibest and gbest
            ibest_nr.fill(0)
            gbest_nr.fill(0)
            ibest_dist.fill(math.inf)
            ibest_cost.fill(math.inf)
            gbest_dist.fill(math.inf)
            gbest_cost.fill(math.inf)

            # 1. Initialization kernel
            k["init_population"](
                pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                next_nodes, next_rlen, next_nr, next_dist, next_cost,
                cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
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

            # 2. CPU reference generation loop
            cur_pop = (pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost)
            nxt_pop = (next_nodes, next_rlen, next_nr, next_dist, next_cost)

            no_improve = 0
            for gen in range(1, max_iter + 1):
                previous_best = gbest_cost[0]
                iter_idx = gen - 1
                a, p_hybrid = _dynamic_parameters(iter_idx, max_iter)
                p_mode = _mode_probability(p_hybrid, self.data)

                # SHO / WOA update
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

                # Save accepted improvements before local search/diversification.
                k["update_island_bests"](
                    *cur_pop, ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                    self.num_islands, self.island_size)
                k["update_global_best"](
                    ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                    gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost, self.num_islands)
                if iter_idx % ls_interval == 0:
                    search_best = (cur_pop if self.ls_scope == "population" else
                                   (ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost)
                                   if self.ls_scope == "island" else
                                   (gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost))
                    k["route_elimination"](
                        *search_best,
                        scratch_route, scratch_unrouted, scratch_flags, prob_device, 5)
                    k["local_search"](
                        *search_best,
                        scratch_route, scratch_route2, prob_device, 0)
                    k["update_island_bests"](
                        *cur_pop, ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                        self.num_islands, self.island_size)
                    k["update_global_best"](
                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                        gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost, self.num_islands)
                k["publish_global_best"](
                    gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost,
                    ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, self.num_islands)
                if gbest_cost[0] < previous_best - PRECISION:
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= stag_interval:
                    k["stagnation_diversify"](
                        *cur_pop, cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                        scratch_route, scratch_unrouted, scratch_flags,
                        prob_device, rng_states, self.num_islands, self.island_size)
                    no_improve = 0

                # Periodic Island Migration
                if self.num_islands > 1 and gen % migr_interval == 0:
                    k["island_migration"](
                        cur_pop[0], cur_pop[1], cur_pop[2], cur_pop[3], cur_pop[4],
                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                        self.num_islands, self.island_size
                    )

                # Update best trackers
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

                out_interval = int(getattr(self.data, "output_per_gens", 25))
                if gen % out_interval == 0 or gen == 1 or gen == max_iter:
                    b_nv = int(gbest_nr[0])
                    b_td = float(gbest_dist[0])
                    print(f"[CPU] Gen: {gen}. a {a:.4f}, p_hybrid {p_mode:.4f}. Best NV {b_nv}, Best TD {b_td:.4f}", flush=True)

            print(f"  Run {run} RUN_CPU_END", flush=True)

            # 3. Decode the CPU reference result
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
            elif run_best_sol.cost < best_s.cost - PRECISION:
                is_better = True

            if is_better and is_valid:
                best_s.copy_from(run_best_sol)
                state.best_s_cost = best_s.cost
                state.find_best_run = run
                td = best_s.cost - best_s.len() * 2000.0
                print(f"  Run {run} Best solution update: {best_s.cost:.4f} (NV={best_s.len()}, TD={td:.4f})", flush=True)

            best_nv = best_s.len()
            best_td = best_s.cost - best_s.len() * 2000.0
            print(f"Run {run} finishes | Run Best: NV={run_nv}, TD={run_dist:.4f} | Global Best: NV={best_nv}, TD={best_td:.4f}", flush=True)
            completed_runs += 1

        total_time = int(time.perf_counter() - start_total)
        print("------------Summary-----------", flush=True)
        print(f"Total {completed_runs} runs, total consumed {total_time} sec", flush=True)
        best_s.output(self.data)
        if not best_s.check(self.data):
            raise RuntimeError("No feasible final solution passed CPU verification")
        sys.stdout.flush()
        return True


def run_solver(data: Any, best_s: Solution):
    """Entry point for the pure Numba solver."""
    use_cuda = getattr(data, "compute_backend", "auto") == "cuda"
    if getattr(data, "compute_backend", "auto") == "auto":
        use_cuda = (cuda is not None and (cuda.is_available() or os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"))

    engine = GpuEngine(data, is_cuda=use_cuda)
    return engine.run_solve(best_s)
