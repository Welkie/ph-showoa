"""Pure Numba CUDA / CPU Solver Kernels for PH-SHOWOA (SA-RCRS-GRASP).

All computation (initialization, RCRS-GRASP, Simulated Annealing, SHO/WOA,
NV-first route elimination, deep local search, stagnation diversification,
and island ring migration) runs inside device kernels.
Zero CPU-GPU synchronization occurs during a search run.
"""

from __future__ import annotations

import math
import numpy as np

from .gpu_base_operators import build_base_operators


def scalar_sa_probability(new_cost, current_cost, iteration, max_iter):
    """Algorithm 1: Boltzmann acceptance using the complete scalar objective."""
    delta = new_cost - current_cost
    if delta < 0.0:
        return 1.0
    temperature = 1.0 - float(iteration) / float(max_iter) if max_iter > 0 else 0.0
    return math.exp(-delta / (1e-6 + temperature * abs(current_cost)))

try:
    from numba import cuda, njit
except ImportError:
    cuda = None
    def njit(*args, **kwargs):
        return lambda f: f


def build_kernel_bundle(is_cuda: bool = False, customer_count: int = 100,
                        sa_t0: float = 100.0, sa_alpha: float = 0.95,
                        sa_tmin: float = 0.1, sa_itermax: int = 100,
                        mutation_probability: float = 0.35,
                        diversify_ratio: float = 0.40):
    """Builds a bundle of Numba device functions and kernels for CUDA or CPU."""
    seen_size = int(customer_count) + 1
    if is_cuda:
        dev_fn = lambda f: cuda.jit(device=True)(f)
        k_fn = lambda f: cuda.jit(f)
    else:
        # Infeasible solutions use infinity; fastmath may invalidate isfinite
        # and ordered comparisons against that sentinel.
        dev_fn = lambda f: njit(nogil=True)(f)
        k_fn = lambda f: njit(nogil=True)(f)

    sa_probability = dev_fn(scalar_sa_probability)

    # -------------------------------------------------------------------------
    # 1. Device PRNG: Xorshift128 (4 uint32 state words per thread)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2. Sequential Route Evaluation (Matches Route.check() 100%)
    # -------------------------------------------------------------------------
    @dev_fn
    def eval_route(route, length, prob_data):
        depot = int(prob_data[0])
        capacity = prob_data[1]
        start_time = prob_data[2]
        dispatch_cost = prob_data[3]
        unit_cost = prob_data[4]
        delivery = prob_data[5]
        pickup = prob_data[6]
        start_tw = prob_data[7]
        end_tw = prob_data[8]
        service = prob_data[9]
        dist_matrix = prob_data[10]
        time_matrix = prob_data[11]

        if length < 2 or route[0] != depot or route[length - 1] != depot:
            return False, 0.0
        if length == 2:
            return True, 0.0
        for i in range(1, length - 1):
            if route[i] <= 0 or route[i] > int(prob_data[14]):
                return False, 0.0

        load = 0.0
        for i in range(1, length - 1):
            load += delivery[route[i]]
        if load > capacity + 1e-6:
            return False, 0.0

        distance_val = 0.0
        time_val = start_time
        prev = route[0]

        for i in range(1, length):
            node = route[i]
            load = load - delivery[node] + pickup[node]
            if load < -1e-6 or load > capacity + 1e-6:
                return False, 0.0

            time_val += time_matrix[prev, node]
            if time_val > end_tw[node] + 1e-6:
                return False, 0.0
            if time_val < start_tw[node]:
                time_val = start_tw[node]
            time_val += service[node]

            distance_val += dist_matrix[prev, node]
            prev = node

        return True, distance_val

    @dev_fn
    def eval_solution(nodes, rlen, nr, s, prob_data):
        num_routes = nr[s]
        total_dist = 0.0
        active_routes = 0
        dispatch_cost = prob_data[3]
        unit_cost = prob_data[4]
        max_vehicles = int(prob_data[12])

        if num_routes < 0 or num_routes > nodes.shape[1]:
            return False, 0, math.inf, math.inf
        if is_cuda:
            seen = cuda.local.array(seen_size, dtype=np.int32)
        else:
            seen = np.empty(seen_size, dtype=np.int32)
        for c in range(customer_count + 1):
            seen[c] = 0
        visited = 0

        for r in range(num_routes):
            l = rlen[s, r]
            if l < 2 or l > nodes.shape[2]:
                return False, 0, math.inf, math.inf
            if l > 2:
                ok, d = eval_route(nodes[s, r, :l], l, prob_data)
                if not ok:
                    return False, 0, math.inf, math.inf
                for p in range(1, l - 1):
                    c = nodes[s, r, p]
                    if seen[c] != 0:
                        return False, 0, math.inf, math.inf
                    seen[c] = 1
                    visited += 1
                total_dist += d
                active_routes += 1

        if max_vehicles > 0 and active_routes > max_vehicles:
            return False, active_routes, math.inf, math.inf

        if visited != int(prob_data[14]):
            return False, active_routes, math.inf, math.inf

        total_cost = active_routes * dispatch_cost + total_dist * unit_cost
        return True, active_routes, total_dist, total_cost

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
    def is_better_cost(nr_a, cost_a, nr_b, cost_b):
        if nr_a <= 0 or not math.isfinite(cost_a):
            return False
        if nr_b <= 0:
            return True
        return cost_a < cost_b - 0.001

    base = build_base_operators(dev_fn, eval_route, eval_solution, copy_solution,
                                rand_u01, randint, shuffle_ints, sa_t0, sa_alpha,
                                sa_tmin, sa_itermax, mutation_probability, diversify_ratio)
    compact = base['compact']
    refresh = base['refresh']
    repair = base['repair']
    insert_customer = base['insert_customer']
    route_capacity = base['capacity']
    sa_warmup_single = base['warmup']
    deep_local_search_single = base['local_search']
    relink = base['relink']
    perturb = base['perturb']
    light_mutation = base['light_mutation']

    # -------------------------------------------------------------------------
    # 3. RCRS-GRASP Score & Construction
    # -------------------------------------------------------------------------
    @dev_fn
    def rcrs_score(route, length, customer, pos, prob_data, lambda1, lambda2):
        """Retained RCRS-GRASP extension with a residual-capacity proxy.

        score = delta_td + lambda1 * rc_pen - lambda2 * rs
        - delta_td: travel-distance increase (td)
        - rc_pen:   capacity tightness proxy (tc simplified for GPU)
        - rs:       round-trip savings from depot (dist[depot,c] + dist[c,depot])
          sign negative => customers far from depot preferred (saves more distance)
        lambda1, lambda2 sinh ngẫu nhiên per-agent từ RNG (Latin Hypercube spirit).
        """
        depot = int(prob_data[0])
        capacity = prob_data[1]
        delivery = prob_data[5]
        pickup = prob_data[6]
        dist_matrix = prob_data[10]

        prev = route[pos - 1]
        next_node = route[pos]

        # Travel-distance increase (delta_td)
        delta_td = dist_matrix[prev, customer] + dist_matrix[customer, next_node] - dist_matrix[prev, next_node]
        if delta_td < 0.0:
            delta_td = 0.0

        # Capacity tightness proxy: penalise routes nearing capacity (simplified tc)
        load = 0.0
        for i in range(1, length - 1):
            load += delivery[route[i]]
        max_load = load
        for i in range(1, length):
            node = route[i]
            load = load - delivery[node] + pickup[node]
            if load > max_load:
                max_load = load

        cust_dem = delivery[customer] if delivery[customer] > pickup[customer] else pickup[customer]
        c_new = max_load + cust_dem
        rc_pen = c_new - capacity * 0.70
        if rc_pen < 0.0:
            rc_pen = 0.0

        # Route savings: round-trip from depot — bám sát base code criterion()
        # rs = dist[DC, node] + dist[node, DC]; larger rs => farther from depot
        rs = dist_matrix[depot, customer] + dist_matrix[customer, depot]

        # RCRS-GRASP extension: proxy differs from the base's exact tc.
        return delta_td + lambda1 * rc_pen - lambda2 * rs

    @dev_fn
    def construct_rcrs_grasp(nodes, rlen, nr, dist, cost, s,
                             scratch_route, scratch_unrouted, scratch_flags, scratch_scores,
                             prob_data, alpha, rng_states):
        customer_num = int(prob_data[14])
        depot = int(prob_data[0])
        dispatch_cost = prob_data[3]
        unit_cost = prob_data[4]

        # Per-agent lambda parameters (bám sát base code Latin Hypercube sampling)
        # Mỗi agent sinh lambda1, lambda2 ngẫu nhiên riêng => đa dạng hóa init
        lambda1 = rand_u01(rng_states, s)
        lambda2 = rand_u01(rng_states, s)

        unrouted_count = customer_num
        for i in range(customer_num):
            scratch_unrouted[s, i] = i + 1

        shuffle_ints(scratch_unrouted[s], unrouted_count, rng_states, s)

        nr[s] = 0
        dist[s] = 0.0
        cost[s] = 0.0

        while unrouted_count > 0:
            num_r = nr[s]
            if num_r == 0:
                pick = randint(rng_states, s, 0, unrouted_count - 1)
                c = scratch_unrouted[s, pick]
                nodes[s, 0, 0] = depot
                nodes[s, 0, 1] = c
                nodes[s, 0, 2] = depot
                rlen[s, 0] = 3
                nr[s] = 1
                ok, d = eval_route(nodes[s, 0, :3], 3, prob_data)
                dist[s] = d
                cost[s] = dispatch_cost + d * unit_cost
                for k in range(pick, unrouted_count - 1):
                    scratch_unrouted[s, k] = scratch_unrouted[s, k + 1]
                unrouted_count -= 1
                continue

            best_score_global = 1e12
            worst_score_global = -1e12

            for i in range(unrouted_count):
                c = scratch_unrouted[s, i]
                best_c_score = 1e12
                best_r = -1
                best_p = -1

                for r in range(num_r):
                    l = rlen[s, r]
                    for p in range(1, l):
                        for k in range(p):
                            scratch_route[s, k] = nodes[s, r, k]
                        scratch_route[s, p] = c
                        for k in range(p, l):
                            scratch_route[s, k + 1] = nodes[s, r, k]

                        ok, _ = eval_route(scratch_route[s, :l+1], l + 1, prob_data)
                        if ok:
                            sc = rcrs_score(nodes[s, r, :l], l, c, p, prob_data, lambda1, lambda2)
                            if sc < best_c_score:
                                best_c_score = sc
                                best_r = r
                                best_p = p

                scratch_scores[s, i] = best_c_score
                scratch_flags[s, i] = best_r * 10000 + best_p
                if best_r != -1:
                    if best_c_score < best_score_global:
                        best_score_global = best_c_score
                    if best_c_score > worst_score_global:
                        worst_score_global = best_c_score

            if best_score_global >= 1e11:
                pick = randint(rng_states, s, 0, unrouted_count - 1)
                c = scratch_unrouted[s, pick]
                new_r = nr[s]
                nodes[s, new_r, 0] = depot
                nodes[s, new_r, 1] = c
                nodes[s, new_r, 2] = depot
                rlen[s, new_r] = 3
                nr[s] = new_r + 1
                ok, d = eval_route(nodes[s, new_r, :3], 3, prob_data)
                dist[s] += d
                cost[s] += dispatch_cost + d * unit_cost
                for k in range(pick, unrouted_count - 1):
                    scratch_unrouted[s, k] = scratch_unrouted[s, k + 1]
                unrouted_count -= 1
                continue

            threshold = best_score_global + alpha * (worst_score_global - best_score_global)
            rcl_count = 0
            for i in range(unrouted_count):
                if scratch_scores[s, i] <= threshold + 1e-6 and scratch_flags[s, i] >= 0:
                    rcl_count += 1

            if rcl_count == 0:
                pick = randint(rng_states, s, 0, unrouted_count - 1)
                c = scratch_unrouted[s, pick]
                new_r = nr[s]
                nodes[s, new_r, 0] = depot
                nodes[s, new_r, 1] = c
                nodes[s, new_r, 2] = depot
                rlen[s, new_r] = 3
                nr[s] = new_r + 1
                ok, d = eval_route(nodes[s, new_r, :3], 3, prob_data)
                dist[s] += d
                cost[s] += dispatch_cost + d * unit_cost
                for k in range(pick, unrouted_count - 1):
                    scratch_unrouted[s, k] = scratch_unrouted[s, k + 1]
                unrouted_count -= 1
                continue

            chosen_rcl_idx = randint(rng_states, s, 0, rcl_count - 1)
            cur_idx = 0
            actual_i = 0
            for i in range(unrouted_count):
                if scratch_scores[s, i] <= threshold + 1e-6 and scratch_flags[s, i] >= 0:
                    if cur_idx == chosen_rcl_idx:
                        actual_i = i
                        break
                    cur_idx += 1

            chosen_c = scratch_unrouted[s, actual_i]
            encoded = scratch_flags[s, actual_i]
            chosen_r = encoded // 10000
            chosen_p = encoded % 10000

            l = rlen[s, chosen_r]
            for k in range(l, chosen_p, -1):
                nodes[s, chosen_r, k] = nodes[s, chosen_r, k - 1]
            nodes[s, chosen_r, chosen_p] = chosen_c
            rlen[s, chosen_r] = l + 1

            for k in range(actual_i, unrouted_count - 1):
                scratch_unrouted[s, k] = scratch_unrouted[s, k + 1]
            unrouted_count -= 1

        ok, n_act, t_dist, t_cost = eval_solution(nodes, rlen, nr, s, prob_data)
        dist[s] = t_dist
        cost[s] = t_cost

    # -------------------------------------------------------------------------
    # 4. NV-First Route Elimination (Crushes Number of Vehicles)
    # -------------------------------------------------------------------------
    @dev_fn
    def route_elimination_single(nodes, rlen, nr, dist, cost, s,
                                 scratch_route, scratch_unrouted, scratch_flags,
                                 prob_data, passes=10):
        for p_iter in range(passes):
            num_r = nr[s]
            if num_r <= 1:
                break

            min_len = 99999
            min_r = -1
            for r in range(num_r):
                l = rlen[s, r]
                if 2 < l < min_len:
                    min_len = l
                    min_r = r

            max_elim_len = max(18, int(prob_data[14]) // 3)
            if min_r == -1 or min_len > max_elim_len:
                break

            num_ejected = min_len - 2
            _, source_distance = eval_route(nodes[s, min_r], min_len, prob_data)
            added_distance = 0.0
            for i in range(num_ejected):
                scratch_unrouted[s, i] = nodes[s, min_r, i + 1]

            all_inserted = True
            for e_idx in range(num_ejected):
                c = scratch_unrouted[s, e_idx]
                best_delta = 1e12
                best_target_r = -1
                best_target_p = -1

                for r in range(num_r):
                    if r == min_r:
                        continue
                    l = rlen[s, r]
                    ok_old, old_d = eval_route(nodes[s, r, :l], l, prob_data)
                    for pos in range(1, l):
                        for k in range(pos):
                            scratch_route[s, k] = nodes[s, r, k]
                        scratch_route[s, pos] = c
                        for k in range(pos, l):
                            scratch_route[s, k + 1] = nodes[s, r, k]
                        ok_new, new_d = eval_route(scratch_route[s, :l+1], l + 1, prob_data)
                        if ok_new:
                            delta = new_d - old_d
                            if delta < best_delta:
                                best_delta = delta
                                best_target_r = r
                                best_target_p = pos

                if best_target_r != -1:
                    l = rlen[s, best_target_r]
                    for k in range(l, best_target_p, -1):
                        nodes[s, best_target_r, k] = nodes[s, best_target_r, k - 1]
                    nodes[s, best_target_r, best_target_p] = c
                    rlen[s, best_target_r] = l + 1
                    scratch_flags[s, e_idx] = best_target_r
                    added_distance += best_delta
                else:
                    all_inserted = False
                    # Rollback all previously inserted customers from this elimination attempt
                    for undo_idx in range(e_idx):
                        undo_r = scratch_flags[s, undo_idx]
                        undo_c = scratch_unrouted[s, undo_idx]
                        lr = rlen[s, undo_r]
                        found_p = -1
                        for p in range(1, lr - 1):
                            if nodes[s, undo_r, p] == undo_c:
                                found_p = p
                                break
                        if found_p != -1:
                            for p in range(found_p, lr - 1):
                                nodes[s, undo_r, p] = nodes[s, undo_r, p + 1]
                            rlen[s, undo_r] = lr - 1
                    break

            if not all_inserted:
                break
            else:
                # Retained route-elimination improvement must improve scalar TC.
                delta = (added_distance - source_distance) * prob_data[4] - prob_data[3]
                if delta >= -0.001:
                    for undo_idx in range(num_ejected):
                        undo_r = scratch_flags[s, undo_idx]
                        undo_c = scratch_unrouted[s, undo_idx]
                        size = rlen[s, undo_r]
                        for pos in range(1, size - 1):
                            if nodes[s, undo_r, pos] == undo_c:
                                for k in range(pos, size - 1):
                                    nodes[s, undo_r, k] = nodes[s, undo_r, k + 1]
                                rlen[s, undo_r] = size - 1
                                break
                    break
                # Successfully inserted all ejected customers! Remove min_r by shifting
                for r in range(min_r, num_r - 1):
                    rlen[s, r] = rlen[s, r + 1]
                    for i in range(rlen[s, r]):
                        nodes[s, r, i] = nodes[s, r + 1, i]
                rlen[s, num_r - 1] = 0
                nr[s] = num_r - 1
                ok, n_act, t_dist, t_cost = eval_solution(nodes, rlen, nr, s, prob_data)
                dist[s] = t_dist
                cost[s] = t_cost

    # -------------------------------------------------------------------------
    # 5. Simulated Annealing Warmup (5 Neighborhood Operators)
    # -------------------------------------------------------------------------
    @dev_fn
    def guided_crossover_sho_single(cur_nodes, cur_rlen, cur_nr, s,
                                    peer_nodes, peer_rlen, peer_nr, peer_s,
                                    ibest_nodes, ibest_rlen, ibest_nr, island_id,
                                    cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                    scratch_route, scratch_unrouted, scratch_flags,
                                    prob_data, rng_states):
        depot = int(prob_data[0])
        customer_num = int(prob_data[14])
        for i in range(customer_num + 1):
            scratch_flags[s, i] = 0

        cand_nr[s] = 0
        cand_dist[s] = 0.0
        cand_cost[s] = 0.0

        # Step 1: Copy 1-2 elite routes from island best
        best_r_count = ibest_nr[island_id]
        if best_r_count > 0:
            num_elite = 1 if best_r_count == 1 or rand_u01(rng_states, s) < 0.6 else 2
            r1 = randint(rng_states, s, 0, best_r_count - 1)
            l1 = ibest_rlen[island_id, r1]
            if l1 > 2:
                c_idx = cand_nr[s]
                cand_rlen[s, c_idx] = l1
                for k in range(l1):
                    node = ibest_nodes[island_id, r1, k]
                    cand_nodes[s, c_idx, k] = node
                    if node != depot:
                        scratch_flags[s, node] = 1
                cand_nr[s] = c_idx + 1

            if num_elite == 2 and best_r_count > 1:
                r2 = randint(rng_states, s, 0, best_r_count - 2)
                if r2 >= r1:
                    r2 += 1
                l2 = ibest_rlen[island_id, r2]
                if l2 > 2:
                    c_idx = cand_nr[s]
                    cand_rlen[s, c_idx] = l2
                    for k in range(l2):
                        node = ibest_nodes[island_id, r2, k]
                        cand_nodes[s, c_idx, k] = node
                        if node != depot:
                            scratch_flags[s, node] = 1
                    cand_nr[s] = c_idx + 1

        # Step 2: Collect remaining unrouted customers from current and peer
        unrouted_cnt = 0
        # From peer
        for r in range(peer_nr[peer_s]):
            for k in range(1, peer_rlen[peer_s, r] - 1):
                node = peer_nodes[peer_s, r, k]
                if node != depot and scratch_flags[s, node] == 0:
                    scratch_unrouted[s, unrouted_cnt] = node
                    scratch_flags[s, node] = 1
                    unrouted_cnt += 1
        # From current
        for r in range(cur_nr[s]):
            for k in range(1, cur_rlen[s, r] - 1):
                node = cur_nodes[s, r, k]
                if node != depot and scratch_flags[s, node] == 0:
                    scratch_unrouted[s, unrouted_cnt] = node
                    scratch_flags[s, node] = 1
                    unrouted_cnt += 1
        # Fallback to guarantee all customers are present
        for c in range(1, customer_num + 1):
            if scratch_flags[s, c] == 0:
                scratch_unrouted[s, unrouted_cnt] = c
                scratch_flags[s, c] = 1
                unrouted_cnt += 1

        # Base crossover: pack the partner/current sequence by capacity,
        # then run the same detect/repair/recheck pipeline.
        if unrouted_cnt > 0:
            r = cand_nr[s]
            cand_nodes[s, r, 0] = depot
            cand_rlen[s, r] = 2
            cand_nodes[s, r, 1] = depot
            cand_nr[s] += 1
            for i in range(unrouted_cnt):
                c = scratch_unrouted[s, i]
                size = cand_rlen[s, r]
                for k in range(size - 1):
                    scratch_route[s, k] = cand_nodes[s, r, k]
                scratch_route[s, size - 1] = c
                scratch_route[s, size] = depot
                if size > 2 and not route_capacity(scratch_route[s], size + 1, prob_data):
                    r = cand_nr[s]
                    if r >= cand_nodes.shape[1]:
                        cand_cost[s] = math.inf
                        return
                    cand_nr[s] += 1
                    cand_nodes[s, r, 0] = depot
                    cand_rlen[s, r] = 2
                    size = 2
                cand_nodes[s, r, size - 1] = c
                cand_nodes[s, r, size] = depot
                cand_rlen[s, r] = size + 1
        repair(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
               scratch_route, scratch_flags, prob_data)

    @dev_fn
    def woa_intensification_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                                  ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, island_id,
                                  cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                  backup_nodes, backup_rlen, backup_nr, backup_dist, backup_cost,
                                  scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                  a_param, prob_data, rng_states):
        # src_python._woa_intensification with paper_flags=True.
        copy_solution(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                      cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s)
        limit = min(cand_nr[s], ibest_nr[island_id])
        for r in range(limit):
            a_value = (2.0 * rand_u01(rng_states, s) - 1.0) * a_param
            c_value = 2.0 * rand_u01(rng_states, s)
            probability = rand_u01(rng_states, s)
            copy_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                          backup_nodes, backup_rlen, backup_nr, backup_dist, backup_cost, s)
            if probability < 0.5 and abs(a_value) < 1.0:
                if r < cand_nr[s]:
                    relink(cand_nodes[s, r], cand_rlen[s, r],
                           ibest_nodes[island_id, r], ibest_rlen[island_id, r],
                           max(0.0, 1.0 - abs(a_value)) * c_value / 2.0,
                           a_value < 0.0, scratch_unrouted[s])
            elif probability < 0.5:
                if rand_u01(rng_states, s) < 0.5:
                    # Uniformly choose a route with at least two customers.
                    eligible = 0
                    for q in range(cand_nr[s]):
                        if cand_rlen[s, q] >= 4:
                            scratch_unrouted[s, eligible] = q
                            eligible += 1
                    if eligible > 0:
                        q = scratch_unrouted[s, randint(rng_states, s, 0, eligible - 1)]
                        size = cand_rlen[s, q]
                        i = randint(rng_states, s, 1, size - 2)
                        j = randint(rng_states, s, 1, size - 3)
                        if j >= i:
                            j += 1
                        if i > j:
                            i, j = j, i
                        operation = randint(rng_states, s, 0, 2)
                        if operation == 0:
                            cand_nodes[s, q, i], cand_nodes[s, q, j] = cand_nodes[s, q, j], cand_nodes[s, q, i]
                        elif operation == 1:
                            value = cand_nodes[s, q, i]
                            for k in range(i, j):
                                cand_nodes[s, q, k] = cand_nodes[s, q, k + 1]
                            cand_nodes[s, q, j] = value
                        else:
                            while i < j:
                                cand_nodes[s, q, i], cand_nodes[s, q, j] = cand_nodes[s, q, j], cand_nodes[s, q, i]
                                i += 1
                                j -= 1
                else:
                    light_mutation(cand_nodes, cand_rlen, cand_nr, s, scratch_route,
                                   scratch_route2, prob_data, rng_states)
                    repair(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                           scratch_route, scratch_flags, prob_data)
            else:
                value = c_value - 1.0
                spiral = min(1.0, abs(math.exp(value) * math.cos(2.0 * math.pi * value)))
                if r < cand_nr[s]:
                    relink(cand_nodes[s, r], cand_rlen[s, r],
                           ibest_nodes[island_id, r], ibest_rlen[island_id, r],
                           spiral, value < 0.0, scratch_unrouted[s])
            if not refresh(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, prob_data):
                copy_solution(backup_nodes, backup_rlen, backup_nr, backup_dist, backup_cost, s,
                              cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s)

    @dev_fn
    def sho_woa_update_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                              next_nodes, next_rlen, next_nr, next_dist, next_cost,
                              cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                              ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                              scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                              s, island_id, island_size,
                              prob_data, a_param, p_mode, iteration, max_iter, rng_states):
        island_start = island_id * island_size
        # Base tournament: exclude self, sample up to three distinct partners.
        # The island is the retained extension's population boundary.
        best_peer = s
        p1, p2 = -1, -1
        for draw in range(min(3, island_size - 1)):
            peer = island_start + randint(rng_states, s, 0, island_size - 2)
            if peer >= s:
                peer += 1
            while peer == p1 or peer == p2:
                peer = island_start + randint(rng_states, s, 0, island_size - 2)
                if peer >= s:
                    peer += 1
            if draw == 0:
                p1 = peer
                best_peer = peer
            elif draw == 1:
                p2 = peer
            if cur_cost[peer] < cur_cost[best_peer]:
                best_peer = peer

        if rand_u01(rng_states, s) < p_mode:
            guided_crossover_sho_single(cur_nodes, cur_rlen, cur_nr, s,
                                        cur_nodes, cur_rlen, cur_nr, best_peer,
                                        ibest_nodes, ibest_rlen, ibest_nr, island_id,
                                        cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                        scratch_route, scratch_unrouted, scratch_flags,
                                        prob_data, rng_states)
            if refresh(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, prob_data):
                if rand_u01(rng_states, s) < mutation_probability:
                    copy_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                                  next_nodes, next_rlen, next_nr, next_dist, next_cost, s)
                    light_mutation(cand_nodes, cand_rlen, cand_nr, s, scratch_route,
                                   scratch_route2, prob_data, rng_states)
                    if not repair(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                                  scratch_route, scratch_flags, prob_data):
                        copy_solution(next_nodes, next_rlen, next_nr, next_dist, next_cost, s,
                                      cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s)
        else:
            woa_intensification_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                                      ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, island_id,
                                      cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                      next_nodes, next_rlen, next_nr, next_dist, next_cost,
                                      scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                      a_param, prob_data, rng_states)

        compact(cand_nodes, cand_rlen, cand_nr, s)
        # Evaluate candidate validity
        ok, act_r, t_d, t_c = eval_solution(cand_nodes, cand_rlen, cand_nr, s, prob_data)
        if not ok:
            copy_solution(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                          next_nodes, next_rlen, next_nr, next_dist, next_cost, s)
            return False

        cand_nr[s] = act_r
        cand_dist[s] = t_d
        cand_cost[s] = t_c

        accepted = False
        new_cost = cand_cost[s]
        old_cost = cur_cost[s]

        if new_cost < old_cost:
            accepted = True
        else:
            prob = sa_probability(new_cost, old_cost, iteration, max_iter)
            if rand_u01(rng_states, s) < prob:
                accepted = True

        if accepted:
            copy_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                          next_nodes, next_rlen, next_nr, next_dist, next_cost, s)
        else:
            copy_solution(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                          next_nodes, next_rlen, next_nr, next_dist, next_cost, s)

        return accepted

    # -------------------------------------------------------------------------
    # 8. Diversification & Island Ring Migration
    # -------------------------------------------------------------------------
    @dev_fn
    def diversify_island_single(nodes, rlen, nr, dist, cost,
                                cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                island_id, island_size,
                                scratch_route, scratch_unrouted, scratch_flags,
                                prob_data, rng_states):
        if diversify_ratio <= 0.0 or island_size < 2:
            return
        start = island_id * island_size
        elite = start
        for s in range(start + 1, start + island_size):
            if cost[s] < cost[elite]:
                elite = s
        if ibest_nr[island_id] > 0 and math.isfinite(ibest_cost[island_id]):
            copy_solution(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, island_id,
                          nodes, rlen, nr, dist, cost, elite)
        remaining = island_size - 1
        selected = int(round(remaining * diversify_ratio))
        for s in range(start, start + island_size):
            if s == elite:
                continue
            take = selected > 0 and randint(rng_states, start, 0, remaining - 1) < selected
            remaining -= 1
            if not take:
                continue
            selected -= 1
            copy_solution(nodes, rlen, nr, dist, cost, s,
                          cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s)
            customer_num = int(prob_data[14])
            depot = int(prob_data[0])
            for c in range(customer_num + 1):
                scratch_flags[s, c] = 0
            count = 0
            for r in range(cand_nr[s]):
                for p in range(1, cand_rlen[s, r] - 1):
                    scratch_unrouted[s, count] = cand_nodes[s, r, p]
                    count += 1
            shuffle_ints(scratch_unrouted[s], count, rng_states, s)
            ruin_count = max(1, int(round(count * (0.20 + 0.20 * rand_u01(rng_states, s)))))
            for i in range(ruin_count):
                scratch_flags[s, scratch_unrouted[s, i]] = 1
            for r in range(cand_nr[s]):
                write = 1
                for p in range(1, cand_rlen[s, r] - 1):
                    c = cand_nodes[s, r, p]
                    if scratch_flags[s, c] == 0:
                        cand_nodes[s, r, write] = c
                        write += 1
                cand_nodes[s, r, write] = depot
                cand_rlen[s, r] = write + 1
            compact(cand_nodes, cand_rlen, cand_nr, s)
            shuffle_ints(scratch_unrouted[s], ruin_count, rng_states, s)
            ok = True
            for i in range(ruin_count):
                if not insert_customer(cand_nodes, cand_rlen, cand_nr, s,
                                       scratch_unrouted[s, i], scratch_route, prob_data):
                    ok = False
                    break
            if ok and repair(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                             scratch_route, scratch_flags, prob_data):
                copy_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                              nodes, rlen, nr, dist, cost, s)

    # -------------------------------------------------------------------------
    # 9. Top-Level Kernels
    # -------------------------------------------------------------------------
    if is_cuda:
        @k_fn
        def init_population_kernel(nodes, rlen, nr, dist, cost,
                                   best_nodes, best_rlen, best_nr, best_dist, best_cost,
                                   trial_nodes, trial_rlen, trial_nr, trial_dist, trial_cost,
                                   scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
                                   prob_data, alpha_lo, alpha_hi, rng_states):
            s = cuda.grid(1)
            if s < nodes.shape[0]:
                alpha = alpha_lo + rand_u01(rng_states, s) * (alpha_hi - alpha_lo)
                construct_rcrs_grasp(nodes, rlen, nr, dist, cost, s,
                                     scratch_route, scratch_unrouted, scratch_flags, scratch_scores,
                                     prob_data, alpha, rng_states)
                if not repair(nodes, rlen, nr, dist, cost, s,
                              scratch_route, scratch_flags, prob_data):
                    cost[s] = math.inf
                    return
                sa_warmup_single(nodes, rlen, nr, dist, cost, s,
                                 best_nodes, best_rlen, best_nr, best_dist, best_cost,
                                 trial_nodes, trial_rlen, trial_nr, trial_dist, trial_cost,
                                 scratch_route, scratch_route2, prob_data, rng_states)
                route_elimination_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_unrouted, scratch_flags, prob_data, 5)

        @k_fn
        def update_population_kernel(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                                     next_nodes, next_rlen, next_nr, next_dist, next_cost,
                                     cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                     ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                     scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                     prob_data, a_param, p_mode, iteration, max_iter, rng_states, island_size):
            s = cuda.grid(1)
            if s < cur_nodes.shape[0]:
                island_id = s // island_size
                sho_woa_update_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                                      next_nodes, next_rlen, next_nr, next_dist, next_cost,
                                      cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                      ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                      scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                      s, island_id, island_size,
                                      prob_data, a_param, p_mode, iteration, max_iter, rng_states)

        @k_fn
        def route_elimination_kernel(nodes, rlen, nr, dist, cost,
                                     scratch_route, scratch_unrouted, scratch_flags, prob_data, passes):
            s = cuda.grid(1)
            if s < nodes.shape[0]:
                route_elimination_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_unrouted, scratch_flags, prob_data, passes)

        @k_fn
        def local_search_kernel(nodes, rlen, nr, dist, cost,
                                scratch_route, scratch_route2, prob_data, max_passes):
            s = cuda.grid(1)
            if s < nodes.shape[0]:
                deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_route2, prob_data, max_passes)

        @k_fn
        def update_island_bests_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                       ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                       num_islands, island_size):
            isl = cuda.grid(1)
            if isl < num_islands:
                best_s = isl * island_size
                for i in range(1, island_size):
                    cand_s = isl * island_size + i
                    if is_better_cost(pop_nr[cand_s], pop_cost[cand_s], pop_nr[best_s], pop_cost[best_s]):
                        best_s = cand_s
                if is_better_cost(pop_nr[best_s], pop_cost[best_s], ibest_nr[isl], ibest_cost[isl]):
                    copy_solution(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost, best_s,
                                  ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl)

        @k_fn
        def update_global_best_kernel(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                      gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost,
                                      num_islands):
            if cuda.grid(1) == 0:
                best_isl = 0
                for isl in range(1, num_islands):
                    if is_better_cost(ibest_nr[isl], ibest_cost[isl], ibest_nr[best_isl], ibest_cost[best_isl]):
                        best_isl = isl
                if is_better_cost(ibest_nr[best_isl], ibest_cost[best_isl], gbest_nr[0], gbest_cost[0]):
                    copy_solution(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, best_isl,
                                  gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost, 0)

        @k_fn
        def publish_global_best_kernel(g_nodes, g_rlen, g_nr, g_dist, g_cost,
                                       i_nodes, i_rlen, i_nr, i_dist, i_cost, num_islands):
            if cuda.grid(1) == 0:
                best = 0
                for i in range(1, num_islands):
                    if is_better_cost(i_nr[i], i_cost[i], i_nr[best], i_cost[best]):
                        best = i
                if is_better_cost(g_nr[0], g_cost[0], i_nr[best], i_cost[best]):
                    copy_solution(g_nodes, g_rlen, g_nr, g_dist, g_cost, 0,
                                  i_nodes, i_rlen, i_nr, i_dist, i_cost, best)

        @k_fn
        def island_migration_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                    ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                    num_islands, island_size):
            if cuda.grid(1) == 0 and num_islands > 1:
                # Ring migration: copy ibest of island i to worst of island (i + 1) % num_islands
                for isl in range(num_islands):
                    dst_isl = (isl + 1) % num_islands
                    # Find worst in dst_isl
                    worst_s = dst_isl * island_size
                    for i in range(1, island_size):
                        cand_s = dst_isl * island_size + i
                        if is_better_cost(pop_nr[worst_s], pop_cost[worst_s], pop_nr[cand_s], pop_cost[cand_s]):
                            worst_s = cand_s
                    # If ibest of isl is better than worst of dst_isl, migrate
                    if is_better_cost(ibest_nr[isl], ibest_cost[isl], pop_nr[worst_s], pop_cost[worst_s]):
                        copy_solution(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl,
                                      pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost, worst_s)

        @k_fn
        def stagnation_diversify_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                        cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                        scratch_route, scratch_unrouted, scratch_flags,
                                        prob_data, rng_states, num_islands, island_size):
            isl = cuda.grid(1)
            if isl < num_islands:
                diversify_island_single(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                        cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                        isl, island_size,
                                        scratch_route, scratch_unrouted, scratch_flags,
                                        prob_data, rng_states)

    else:
        # CPU Mode
        @k_fn
        def init_population_kernel(nodes, rlen, nr, dist, cost,
                                   best_nodes, best_rlen, best_nr, best_dist, best_cost,
                                   trial_nodes, trial_rlen, trial_nr, trial_dist, trial_cost,
                                   scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
                                   prob_data, alpha_lo, alpha_hi, rng_states):
            for s in range(nodes.shape[0]):
                alpha = alpha_lo + rand_u01(rng_states, s) * (alpha_hi - alpha_lo)
                construct_rcrs_grasp(nodes, rlen, nr, dist, cost, s,
                                     scratch_route, scratch_unrouted, scratch_flags, scratch_scores,
                                     prob_data, alpha, rng_states)
                if not repair(nodes, rlen, nr, dist, cost, s,
                              scratch_route, scratch_flags, prob_data):
                    cost[s] = math.inf
                    continue
                sa_warmup_single(nodes, rlen, nr, dist, cost, s,
                                 best_nodes, best_rlen, best_nr, best_dist, best_cost,
                                 trial_nodes, trial_rlen, trial_nr, trial_dist, trial_cost,
                                 scratch_route, scratch_route2, prob_data, rng_states)
                route_elimination_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_unrouted, scratch_flags, prob_data, 5)

        @k_fn
        def update_population_kernel(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                                     next_nodes, next_rlen, next_nr, next_dist, next_cost,
                                     cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                     ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                     scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                     prob_data, a_param, p_mode, iteration, max_iter, rng_states, island_size):
            for s in range(cur_nodes.shape[0]):
                island_id = s // island_size
                sho_woa_update_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                                      next_nodes, next_rlen, next_nr, next_dist, next_cost,
                                      cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                      ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                      scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                      s, island_id, island_size,
                                      prob_data, a_param, p_mode, iteration, max_iter, rng_states)

        @k_fn
        def route_elimination_kernel(nodes, rlen, nr, dist, cost,
                                     scratch_route, scratch_unrouted, scratch_flags, prob_data, passes):
            for s in range(nodes.shape[0]):
                route_elimination_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_unrouted, scratch_flags, prob_data, passes)

        @k_fn
        def local_search_kernel(nodes, rlen, nr, dist, cost,
                                scratch_route, scratch_route2, prob_data, max_passes):
            for s in range(nodes.shape[0]):
                deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_route2, prob_data, max_passes)

        @k_fn
        def update_island_bests_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                       ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                       num_islands, island_size):
            for isl in range(num_islands):
                best_s = isl * island_size
                for i in range(1, island_size):
                    cand_s = isl * island_size + i
                    if is_better_cost(pop_nr[cand_s], pop_cost[cand_s], pop_nr[best_s], pop_cost[best_s]):
                        best_s = cand_s
                if is_better_cost(pop_nr[best_s], pop_cost[best_s], ibest_nr[isl], ibest_cost[isl]):
                    copy_solution(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost, best_s,
                                  ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl)

        @k_fn
        def update_global_best_kernel(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                      gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost,
                                      num_islands):
            best_isl = 0
            for isl in range(1, num_islands):
                if is_better_cost(ibest_nr[isl], ibest_cost[isl], ibest_nr[best_isl], ibest_cost[best_isl]):
                    best_isl = isl
            if is_better_cost(ibest_nr[best_isl], ibest_cost[best_isl], gbest_nr[0], gbest_cost[0]):
                copy_solution(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, best_isl,
                              gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost, 0)

        @k_fn
        def publish_global_best_kernel(g_nodes, g_rlen, g_nr, g_dist, g_cost,
                                       i_nodes, i_rlen, i_nr, i_dist, i_cost, num_islands):
            if True:
                best = 0
                for i in range(1, num_islands):
                    if is_better_cost(i_nr[i], i_cost[i], i_nr[best], i_cost[best]):
                        best = i
                if is_better_cost(g_nr[0], g_cost[0], i_nr[best], i_cost[best]):
                    copy_solution(g_nodes, g_rlen, g_nr, g_dist, g_cost, 0,
                                  i_nodes, i_rlen, i_nr, i_dist, i_cost, best)

        @k_fn
        def island_migration_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                    ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                    num_islands, island_size):
            if num_islands > 1:
                for isl in range(num_islands):
                    dst_isl = (isl + 1) % num_islands
                    worst_s = dst_isl * island_size
                    for i in range(1, island_size):
                        cand_s = dst_isl * island_size + i
                        if is_better_cost(pop_nr[worst_s], pop_cost[worst_s], pop_nr[cand_s], pop_cost[cand_s]):
                            worst_s = cand_s
                    if is_better_cost(ibest_nr[isl], ibest_cost[isl], pop_nr[worst_s], pop_cost[worst_s]):
                        copy_solution(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl,
                                      pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost, worst_s)

        @k_fn
        def stagnation_diversify_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                        cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                        scratch_route, scratch_unrouted, scratch_flags,
                                        prob_data, rng_states, num_islands, island_size):
            for isl in range(num_islands):
                diversify_island_single(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                        cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                        ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                        isl, island_size,
                                        scratch_route, scratch_unrouted, scratch_flags,
                                        prob_data, rng_states)

    return {
        "device_evaluate": eval_solution,
        "device_refresh": refresh,
        "device_repair": repair,
        "device_relink": relink,
        "device_perturb": perturb,
        "device_local_search": deep_local_search_single,
        "device_crossover": guided_crossover_sho_single,
        "device_warmup": sa_warmup_single,
        "device_woa": woa_intensification_single,
        "init_population": init_population_kernel,
        "update_population": update_population_kernel,
        "route_elimination": route_elimination_kernel,
        "local_search": local_search_kernel,
        "update_island_bests": update_island_bests_kernel,
        "update_global_best": update_global_best_kernel,
        "island_migration": island_migration_kernel,
        "publish_global_best": publish_global_best_kernel,
        "stagnation_diversify": stagnation_diversify_kernel,
    }
