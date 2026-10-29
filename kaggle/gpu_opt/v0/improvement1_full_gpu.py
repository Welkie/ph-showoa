"""Improvement 1 (docs/improved.md section 1): 100% GPU-resident execution.

Device-side building blocks (Xorshift128 RNG per individual, solution copy and in-place route edits on
pre-allocated arrays), problem upload, buffer allocation and the kernel bundle that compiles every module
either as CUDA kernels (one thread per individual) or as Numba CPU functions (same code, loops over individuals).
"""

from __future__ import annotations

import math
import os

import numpy as np

try:
    from numba import cuda, njit
except ImportError:  # pragma: no cover - optional dependency
    cuda = None

    def njit(*args, **kwargs):
        return lambda f: f


def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""

    @dev_fn
    def xorshift128(rng_states, s):
        x = rng_states[s, 0]
        y = rng_states[s, 1]
        z = rng_states[s, 2]
        w = rng_states[s, 3]
        t = (x ^ ((x << 11) & 0xFFFFFFFF)) & 0xFFFFFFFF
        x = y
        y = z
        z = w
        w = ((w ^ (w >> 19)) ^ (t ^ (t >> 8))) & 0xFFFFFFFF
        rng_states[s, 0] = x
        rng_states[s, 1] = y
        rng_states[s, 2] = z
        rng_states[s, 3] = w
        return w

    @dev_fn
    def rand_u01(rng_states, s):
        return float(xorshift128(rng_states, s)) / 4294967296.0

    @dev_fn
    def randint(rng_states, s, low, high):
        if low >= high:
            return low
        return low + int(xorshift128(rng_states, s) % (high - low + 1))

    @dev_fn
    def shuffle_ints(arr, count, rng_states, s):
        for i in range(count - 1, 0, -1):
            j = randint(rng_states, s, 0, i)
            tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp

    @dev_fn
    def copy_solution(src_nodes, src_rlen, src_nr, src_dist, src_cost, src_s,
                      dst_nodes, dst_rlen, dst_nr, dst_dist, dst_cost, dst_s):
        n_r = src_nr[src_s]
        dst_nr[dst_s] = n_r
        dst_dist[dst_s] = src_dist[src_s]
        dst_cost[dst_s] = src_cost[src_s]
        for r in range(n_r):
            l = src_rlen[src_s, r]
            dst_rlen[dst_s, r] = l
            for i in range(l):
                dst_nodes[dst_s, r, i] = src_nodes[src_s, r, i]

    @dev_fn
    def is_better_lex(nr_a, dist_a, nr_b, dist_b):
        if nr_a <= 0:
            return False
        if nr_b <= 0:
            return True
        # Scalar objective: Cost = 2000*NV + TD
        return (2000.0 * nr_a + dist_a) < (2000.0 * nr_b + dist_b) - 1e-3

    @dev_fn
    def compact_routes(nodes, rlen, nr, s):
        # Drop empty routes (length <= 2), keeping the order of the others
        w = 0
        n_r = nr[s]
        for r in range(n_r):
            l = rlen[s, r]
            if l > 2:
                if w != r:
                    for k in range(l):
                        nodes[s, w, k] = nodes[s, r, k]
                    rlen[s, w] = l
                w += 1
        for r in range(w, n_r):
            rlen[s, r] = 0
        nr[s] = w

    @dev_fn
    def remove_at(nodes, rlen, s, r, pos):
        l = rlen[s, r]
        for k in range(pos, l - 1):
            nodes[s, r, k] = nodes[s, r, k + 1]
        rlen[s, r] = l - 1

    @dev_fn
    def insert_at(nodes, rlen, s, r, pos, c):
        l = rlen[s, r]
        for k in range(l, pos, -1):
            nodes[s, r, k] = nodes[s, r, k - 1]
        nodes[s, r, pos] = c
        rlen[s, r] = l + 1

    @dev_fn
    def open_route(nodes, rlen, nr, s, c, depot):
        r = nr[s]
        if r >= nodes.shape[1]:
            return False
        nodes[s, r, 0] = depot
        nodes[s, r, 1] = c
        nodes[s, r, 2] = depot
        rlen[s, r] = 3
        nr[s] = r + 1
        return True

    k.update({"xorshift128": xorshift128, "rand_u01": rand_u01, "randint": randint, "shuffle_ints": shuffle_ints, "copy_solution": copy_solution, "is_better_lex": is_better_lex, "compact_routes": compact_routes, "remove_at": remove_at, "insert_at": insert_at, "open_route": open_route})


# ---------------------------------------------------------------------------
# Problem upload and pre-allocated device buffers
# ---------------------------------------------------------------------------
def prepare_problem_data(problem):
    """Pack a src_python_baseline-style Problem into the tuple read by the device code.

    Index: 0 depot, 1 capacity, 2 depot ready time, 3 dispatching cost, 4 unit cost, 5 delivery, 6 pickup,
    7 ready time, 8 due time, 9 service time, 10 distance matrix, 11 travel-time matrix, 12 VEHICLES (fleet limit),
    13 unused, 14 number of customers.
    """
    return (
        0, float(problem.capacity), float(problem.ready_time[0]),
        float(problem.dispatch_cost), float(problem.unit_cost),
        np.ascontiguousarray(np.array(problem.delivery, dtype=np.float64)),
        np.ascontiguousarray(np.array(problem.pickup, dtype=np.float64)),
        np.ascontiguousarray(np.array(problem.ready_time, dtype=np.float64)),
        np.ascontiguousarray(np.array(problem.due_time, dtype=np.float64)),
        np.ascontiguousarray(np.array(problem.service_time, dtype=np.float64)),
        np.ascontiguousarray(np.array(problem.distance, dtype=np.float64)),
        np.ascontiguousarray(np.array(problem.travel_time, dtype=np.float64)),
        int(problem.max_vehicles), 0, int(problem.num_customers),
    )


SOLUTION_SETS = ("pop", "next", "cand", "bak")


def allocate_buffers(is_cuda, P, R, L, N, num_islands, max_iter, prob_data):
    """Allocate every array of a run once (ping-pong population, candidates, island/global bests, scratch)."""
    zeros = (lambda shape, dt: cuda.device_array(shape, dtype=dt)) if is_cuda else np.zeros
    buf = {}
    for name, rows in (("pop", P), ("next", P), ("cand", P), ("bak", P), ("ibest", num_islands), ("gbest", 1)):
        buf[name] = (zeros((rows, R, L), np.int32), zeros((rows, R), np.int32), zeros(rows, np.int32),
                     zeros(rows, np.float64), zeros(rows, np.float64))
    buf["scratch_route"] = zeros((P, L), np.int32)
    buf["scratch_route2"] = zeros((P, L), np.int32)
    buf["scratch_unrouted"] = zeros((P, N + 1), np.int32)
    buf["scratch_flags"] = zeros((P, N + 1), np.int32)
    buf["scratch_scores"] = zeros((P, N + 1), np.float64)
    buf["slot_r"] = zeros((P, N + 2 * R), np.int32)          # improvement 4: (route, position) of every slot
    buf["slot_k"] = zeros((P, N + 2 * R), np.int32)
    buf["rng_states"] = zeros((P, 4), np.uint32)
    buf["run_log"] = zeros((max_iter, 2), np.float64)
    if is_cuda:
        buf["prob"] = tuple(cuda.to_device(x) if isinstance(x, np.ndarray) else x for x in prob_data)
    else:
        buf["prob"] = prob_data
    return buf


# ---------------------------------------------------------------------------
# Kernel bundle
# ---------------------------------------------------------------------------
def build_kernel_bundle(is_cuda: bool = False):
    """Compile all device functions (baseline algorithms + improvements) and the top-level kernels."""
    from . import (algorithm1_solver, algorithm2_initialization, algorithm3_hybrid, algorithm4_woa,
                   algorithm5_sho, algorithm6_crossover, algorithm7_mutation, algorithm8_localsearch,
                   algorithm9_feasibility, algorithm10_repair, improvement2_sa_rcrs_grasp,
                   improvement3_island_model, improvement4_population_local_search)

    if is_cuda:
        dev_fn = lambda f: cuda.jit(device=True)(f)
        k_fn = lambda f: cuda.jit(f)
    else:
        dev_fn = lambda f: njit(fastmath=True, nogil=True)(f)
        k_fn = lambda f: njit(fastmath=True, nogil=True)(f)

    k = {"dev_fn": dev_fn}
    build(dev_fn, k)
    for module in (algorithm9_feasibility, algorithm7_mutation, algorithm10_repair, algorithm2_initialization,
                   improvement4_population_local_search, algorithm8_localsearch, improvement2_sa_rcrs_grasp,
                   algorithm6_crossover, algorithm5_sho, algorithm4_woa, algorithm3_hybrid, algorithm1_solver):
        module.build(dev_fn, k)

    kernels = {}
    _build_population_kernels(is_cuda, k_fn, k, kernels)
    improvement2_sa_rcrs_grasp.build_kernels(is_cuda, k_fn, k, kernels)
    improvement3_island_model.build_kernels(is_cuda, k_fn, k, kernels)
    improvement4_population_local_search.build_kernels(is_cuda, k_fn, k, kernels)
    return kernels


def _build_population_kernels(is_cuda, k_fn, k, kernels):
    rand_u01 = k["rand_u01"]
    construct_rcrs_grasp = k["construct_rcrs_grasp"]
    sa_warmup_single = k["sa_warmup_single"]
    route_elimination_single = k["route_elimination_single"]
    deep_local_search_single = k["deep_local_search_single"]
    init_paper_sa_single = k["init_paper_sa_single"]
    update_individual = k["update_individual"]
    mark_worst_individuals = k["mark_worst_individuals"]
    mutate_single = k["mutate_single"]
    refresh_solution = k["refresh_solution"]

    @k["dev_fn"]
    def init_individual(nodes, rlen, nr, dist, cost, s,
                        cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                        bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                        scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
                        prob_data, init_mode, alpha_lo, alpha_hi, sa_iters, re_passes,
                        t0, alpha_sa, t_min, iter_max, rng_states, init_fi_ls):
        if init_mode == 2:                       # --init mixed: even individuals Algorithm 2, odd ones RCRS-GRASP
            init_mode = 1 if s % 2 == 0 else 0
        if init_mode == 1:
            # Algorithm 2 of the paper (--init sa)
            init_paper_sa_single(nodes, rlen, nr, dist, cost, s,
                                 cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                 bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                                 scratch_route, scratch_unrouted, scratch_scores,
                                 prob_data, rng_states, t0, alpha_sa, t_min, iter_max)
        else:
            # Improvement 2 (--init sa_rcrs_grasp): RCRS-GRASP, SA warm-up, route elimination, local search.
            # With --ls_strategy best (default) the local search is the block-parallel kernel of improvement 4,
            # launched by the host right after this kernel.
            alpha = alpha_lo + rand_u01(rng_states, s) * (alpha_hi - alpha_lo)
            construct_rcrs_grasp(nodes, rlen, nr, dist, cost, s,
                                 scratch_route, scratch_unrouted, scratch_flags, scratch_scores,
                                 prob_data, alpha, rng_states)
            sa_warmup_single(nodes, rlen, nr, dist, cost, s, scratch_route, scratch_route2,
                             prob_data, rng_states, sa_iters)
            route_elimination_single(nodes, rlen, nr, dist, cost, s,
                                     scratch_route, scratch_unrouted, scratch_flags, prob_data, re_passes)
            if init_fi_ls:
                deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_route2, prob_data, 1000000)

    @k["dev_fn"]
    def diversify_population(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                             scratch_route, scratch_route2, scratch_flags, prob_data, rng_states):
        # DIVERSIFY (Algorithm 1 line 45; not defined in the paper): 5 mutations on the worst 20%, as the baseline
        P = pop_nodes.shape[0]
        mark_worst_individuals(pop_cost, scratch_flags, P)
        for s in range(P):
            if scratch_flags[s, 0] == 1:
                for _ in range(5):
                    mutate_single(pop_nodes, pop_rlen, pop_nr, s, scratch_route, scratch_route2,
                                  prob_data, rng_states)
                refresh_solution(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost, s, prob_data)

    @k["dev_fn"]
    def stagnation_step(ibest_nr, ibest_dist, stag_state, num_islands):
        # Algorithm 1 lines 39-44 on the best of the whole population (epsilon = 1e-4 on the cost)
        best_i = -1
        for isl in range(num_islands):
            if ibest_nr[isl] > 0:
                if best_i < 0 or (2000.0 * ibest_nr[isl] + ibest_dist[isl] <
                                  2000.0 * ibest_nr[best_i] + ibest_dist[best_i]):
                    best_i = isl
        if best_i >= 0:
            cur = 2000.0 * ibest_nr[best_i] + ibest_dist[best_i]
            if stag_state[0, 0] <= 0.0 or cur < 2000.0 * stag_state[0, 0] + stag_state[0, 1] - 1e-4:
                stag_state[0, 0] = float(ibest_nr[best_i])
                stag_state[0, 1] = ibest_dist[best_i]
                stag_state[0, 2] = 0.0
            else:
                stag_state[0, 2] += 1.0

    @k["dev_fn"]
    def seed_rng(rng_states, s, seed):
        rng_states[s, 0] = np.uint32((seed + s * 1337) & 0xFFFFFFFF) | np.uint32(1)
        rng_states[s, 1] = np.uint32((seed + s * 2749 + 362436069) & 0xFFFFFFFF) | np.uint32(1)
        rng_states[s, 2] = np.uint32((seed + s * 5171 + 521288629) & 0xFFFFFFFF) | np.uint32(1)
        rng_states[s, 3] = np.uint32((seed + s * 7919 + 88675123) & 0xFFFFFFFF) | np.uint32(1)

    if is_cuda:
        @k_fn
        def init_population_kernel(nodes, rlen, nr, dist, cost,
                                   cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                   bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                                   scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
                                   prob_data, init_mode, alpha_lo, alpha_hi, sa_iters, re_passes,
                                   t0, alpha_sa, t_min, iter_max, rng_states, init_fi_ls):
            s = cuda.grid(1)
            if s < nodes.shape[0]:
                init_individual(nodes, rlen, nr, dist, cost, s,
                                cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                                scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
                                prob_data, init_mode, alpha_lo, alpha_hi, sa_iters, re_passes,
                                t0, alpha_sa, t_min, iter_max, rng_states, init_fi_ls)

        @k_fn
        def update_population_kernel(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                                     next_nodes, next_rlen, next_nr, next_dist, next_cost,
                                     cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                     ibest_nodes, ibest_rlen, ibest_nr,
                                     scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                     prob_data, a_param, p_hybrid, mutation_prob, iteration, max_iter,
                                     rng_states, island_size):
            s = cuda.grid(1)
            if s < cur_nodes.shape[0]:
                update_individual(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                                  next_nodes, next_rlen, next_nr, next_dist, next_cost,
                                  cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                  ibest_nodes, ibest_rlen, ibest_nr,
                                  scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                  s, s // island_size, island_size,
                                  prob_data, a_param, p_hybrid, mutation_prob, iteration, max_iter, rng_states)

        @k_fn
        def stagnation_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                              ibest_nr, ibest_dist, stag_state, num_islands, stag_interval,
                              scratch_route, scratch_route2, scratch_flags, prob_data, rng_states):
            if cuda.grid(1) == 0:
                stagnation_step(ibest_nr, ibest_dist, stag_state, num_islands)
                if stag_state[0, 2] >= stag_interval:
                    diversify_population(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                         scratch_route, scratch_route2, scratch_flags, prob_data, rng_states)
                    stag_state[0, 2] = 0.0

        @k_fn
        def init_device_rng_kernel(rng_states, seed):
            s = cuda.grid(1)
            if s < rng_states.shape[0]:
                seed_rng(rng_states, s, seed)

        @k_fn
        def record_log_kernel(run_log, gbest_nr, gbest_dist, gen):
            if cuda.grid(1) == 0:
                run_log[gen - 1, 0] = float(gbest_nr[0])
                run_log[gen - 1, 1] = float(gbest_dist[0])
    else:
        @k_fn
        def init_population_kernel(nodes, rlen, nr, dist, cost,
                                   cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                   bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                                   scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
                                   prob_data, init_mode, alpha_lo, alpha_hi, sa_iters, re_passes,
                                   t0, alpha_sa, t_min, iter_max, rng_states, init_fi_ls):
            for s in range(nodes.shape[0]):
                init_individual(nodes, rlen, nr, dist, cost, s,
                                cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                                scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
                                prob_data, init_mode, alpha_lo, alpha_hi, sa_iters, re_passes,
                                t0, alpha_sa, t_min, iter_max, rng_states, init_fi_ls)

        @k_fn
        def update_population_kernel(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                                     next_nodes, next_rlen, next_nr, next_dist, next_cost,
                                     cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                     ibest_nodes, ibest_rlen, ibest_nr,
                                     scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                     prob_data, a_param, p_hybrid, mutation_prob, iteration, max_iter,
                                     rng_states, island_size):
            for s in range(cur_nodes.shape[0]):
                update_individual(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                                  next_nodes, next_rlen, next_nr, next_dist, next_cost,
                                  cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                  ibest_nodes, ibest_rlen, ibest_nr,
                                  scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                  s, s // island_size, island_size,
                                  prob_data, a_param, p_hybrid, mutation_prob, iteration, max_iter, rng_states)

        @k_fn
        def stagnation_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                              ibest_nr, ibest_dist, stag_state, num_islands, stag_interval,
                              scratch_route, scratch_route2, scratch_flags, prob_data, rng_states):
            stagnation_step(ibest_nr, ibest_dist, stag_state, num_islands)
            if stag_state[0, 2] >= stag_interval:
                diversify_population(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                     scratch_route, scratch_route2, scratch_flags, prob_data, rng_states)
                stag_state[0, 2] = 0.0

        @k_fn
        def init_device_rng_kernel(rng_states, seed):
            for s in range(rng_states.shape[0]):
                seed_rng(rng_states, s, seed)

        @k_fn
        def record_log_kernel(run_log, gbest_nr, gbest_dist, gen):
            run_log[gen - 1, 0] = float(gbest_nr[0])
            run_log[gen - 1, 1] = float(gbest_dist[0])

    kernels.update({
        "init_population": init_population_kernel,
        "update_population": update_population_kernel,
        "stagnation": stagnation_kernel,
        "init_device_rng": init_device_rng_kernel,
        "record_log": record_log_kernel,
    })
