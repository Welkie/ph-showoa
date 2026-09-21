"""Pure Numba CUDA / CPU Solver Kernels for PH-SHOWOA (SA-RCRS-GRASP).

All computation (initialization, RCRS-GRASP, Simulated Annealing, SHO/WOA,
NV-first route elimination, deep local search, stagnation diversification,
and island ring migration) runs inside device kernels.
Zero CPU-GPU synchronization occurs during a search run.
"""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import cuda, njit
except ImportError:
    cuda = None
    def njit(*args, **kwargs):
        return lambda f: f


def build_kernel_bundle(is_cuda: bool = False):
    """Builds a bundle of Numba device functions and kernels for CUDA or CPU."""
    if is_cuda:
        dev_fn = lambda f: cuda.jit(device=True)(f)
        k_fn = lambda f: cuda.jit(f)
    else:
        dev_fn = lambda f: njit(fastmath=True, nogil=True)(f)
        k_fn = lambda f: njit(fastmath=True, nogil=True)(f)

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
            if route[i] == depot:
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

        for r in range(num_routes):
            l = rlen[s, r]
            if l > 2:
                ok, d = eval_route(nodes[s, r, :l], l, prob_data)
                if not ok:
                    return False, 0, 1e12, 1e12
                total_dist += d
                active_routes += 1

        if max_vehicles > 0 and active_routes > max_vehicles:
            return False, active_routes, 1e12, 1e12

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
    def is_better_lex(nr_a, dist_a, nr_b, dist_b):
        if nr_a <= 0:
            return False
        if nr_b <= 0:
            return True
        if nr_a < nr_b:
            return True
        if nr_a == nr_b and dist_a < dist_b - 1e-4:
            return True
        return False

    # -------------------------------------------------------------------------
    # 3. RCRS-GRASP Score & Construction
    # -------------------------------------------------------------------------
    @dev_fn
    def rcrs_score(route, length, customer, pos, prob_data):
        depot = int(prob_data[0])
        capacity = prob_data[1]
        delivery = prob_data[5]
        pickup = prob_data[6]
        dist_matrix = prob_data[10]

        prev = route[pos - 1]
        next_node = route[pos]

        delta_td = dist_matrix[prev, customer] + dist_matrix[customer, next_node] - dist_matrix[prev, next_node]
        if delta_td < 0.0:
            delta_td = 0.0

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

        dx_prev = dist_matrix[depot, prev]
        dx_c = dist_matrix[depot, customer]
        rs_pen = abs(dx_prev + dist_matrix[prev, customer] - dx_c)

        return delta_td + 0.5 * rc_pen + 0.3 * rs_pen

    @dev_fn
    def construct_rcrs_grasp(nodes, rlen, nr, dist, cost, s,
                             scratch_route, scratch_unrouted, scratch_flags, scratch_scores,
                             prob_data, alpha, rng_states):
        customer_num = int(prob_data[14])
        depot = int(prob_data[0])
        dispatch_cost = prob_data[3]
        unit_cost = prob_data[4]

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
                            sc = rcrs_score(nodes[s, r, :l], l, c, p, prob_data)
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
    def sa_warmup_single(nodes, rlen, nr, dist, cost, s,
                         scratch_route, scratch_route2,
                         prob_data, rng_states, sa_iters=25):
        temp = 100.0
        cooling = 0.85
        num_r = nr[s]
        if num_r == 0:
            return

        while temp > 0.5:
            for _ in range(sa_iters):
                num_r = nr[s]
                if num_r == 0:
                    break
                move_type = randint(rng_states, s, 1, 5)

                if move_type == 1 and num_r >= 1:
                    # Intra 2-opt
                    r = randint(rng_states, s, 0, num_r - 1)
                    l = rlen[s, r]
                    if l >= 5:
                        i = randint(rng_states, s, 1, l - 3)
                        j = randint(rng_states, s, i + 1, l - 2)
                        for k in range(l):
                            scratch_route[s, k] = nodes[s, r, k]
                        p1, p2 = i, j
                        while p1 < p2:
                            tmp = scratch_route[s, p1]
                            scratch_route[s, p1] = scratch_route[s, p2]
                            scratch_route[s, p2] = tmp
                            p1 += 1
                            p2 -= 1
                        ok, new_d = eval_route(scratch_route[s, :l], l, prob_data)
                        if ok:
                            _, old_d = eval_route(nodes[s, r, :l], l, prob_data)
                            delta = new_d - old_d
                            if delta < 0.0 or rand_u01(rng_states, s) < math.exp(-delta / (temp + 1e-4)):
                                for k in range(l):
                                    nodes[s, r, k] = scratch_route[s, k]
                                dist[s] += delta
                                cost[s] += delta * prob_data[4]

                elif move_type == 2 and num_r >= 1:
                    # Intra relocate
                    r = randint(rng_states, s, 0, num_r - 1)
                    l = rlen[s, r]
                    if l >= 4:
                        i = randint(rng_states, s, 1, l - 2)
                        j = randint(rng_states, s, 1, l - 2)
                        if i != j:
                            for k in range(l):
                                scratch_route[s, k] = nodes[s, r, k]
                            val = scratch_route[s, i]
                            if i < j:
                                for k in range(i, j):
                                    scratch_route[s, k] = scratch_route[s, k + 1]
                                scratch_route[s, j] = val
                            else:
                                for k in range(i, j, -1):
                                    scratch_route[s, k] = scratch_route[s, k - 1]
                                scratch_route[s, j] = val
                            ok, new_d = eval_route(scratch_route[s, :l], l, prob_data)
                            if ok:
                                _, old_d = eval_route(nodes[s, r, :l], l, prob_data)
                                delta = new_d - old_d
                                if delta < 0.0 or rand_u01(rng_states, s) < math.exp(-delta / (temp + 1e-4)):
                                    for k in range(l):
                                        nodes[s, r, k] = scratch_route[s, k]
                                    dist[s] += delta
                                    cost[s] += delta * prob_data[4]

                elif move_type == 3 and num_r >= 2:
                    # Inter relocate
                    r1 = randint(rng_states, s, 0, num_r - 1)
                    r2 = randint(rng_states, s, 0, num_r - 1)
                    if r1 != r2:
                        l1 = rlen[s, r1]
                        l2 = rlen[s, r2]
                        if l1 >= 4 and l2 >= 3:
                            i = randint(rng_states, s, 1, l1 - 2)
                            j = randint(rng_states, s, 1, l2 - 1)
                            c = nodes[s, r1, i]
                            idx = 0
                            for k in range(l1):
                                if k != i:
                                    scratch_route[s, idx] = nodes[s, r1, k]
                                    idx += 1
                            for k in range(j):
                                scratch_route2[s, k] = nodes[s, r2, k]
                            scratch_route2[s, j] = c
                            for k in range(j, l2):
                                scratch_route2[s, k + 1] = nodes[s, r2, k]

                            ok1, new_d1 = eval_route(scratch_route[s, :l1-1], l1 - 1, prob_data)
                            ok2, new_d2 = eval_route(scratch_route2[s, :l2+1], l2 + 1, prob_data)
                            if ok1 and ok2:
                                _, old_d1 = eval_route(nodes[s, r1, :l1], l1, prob_data)
                                _, old_d2 = eval_route(nodes[s, r2, :l2], l2, prob_data)
                                delta = (new_d1 + new_d2) - (old_d1 + old_d2)
                                if delta < 0.0 or rand_u01(rng_states, s) < math.exp(-delta / (temp + 1e-4)):
                                    for k in range(l1 - 1):
                                        nodes[s, r1, k] = scratch_route[s, k]
                                    nodes[s, r1, l1 - 1] = 0
                                    rlen[s, r1] = l1 - 1
                                    for k in range(l2 + 1):
                                        nodes[s, r2, k] = scratch_route2[s, k]
                                    rlen[s, r2] = l2 + 1
                                    dist[s] += delta
                                    cost[s] += delta * prob_data[4]

                elif move_type == 4 and num_r >= 2:
                    # Inter swap
                    r1 = randint(rng_states, s, 0, num_r - 1)
                    r2 = randint(rng_states, s, 0, num_r - 1)
                    if r1 != r2:
                        l1 = rlen[s, r1]
                        l2 = rlen[s, r2]
                        if l1 >= 3 and l2 >= 3:
                            i = randint(rng_states, s, 1, l1 - 2)
                            j = randint(rng_states, s, 1, l2 - 2)
                            for k in range(l1):
                                scratch_route[s, k] = nodes[s, r1, k]
                            for k in range(l2):
                                scratch_route2[s, k] = nodes[s, r2, k]
                            scratch_route[s, i] = nodes[s, r2, j]
                            scratch_route2[s, j] = nodes[s, r1, i]

                            ok1, new_d1 = eval_route(scratch_route[s, :l1], l1, prob_data)
                            ok2, new_d2 = eval_route(scratch_route2[s, :l2], l2, prob_data)
                            if ok1 and ok2:
                                _, old_d1 = eval_route(nodes[s, r1, :l1], l1, prob_data)
                                _, old_d2 = eval_route(nodes[s, r2, :l2], l2, prob_data)
                                delta = (new_d1 + new_d2) - (old_d1 + old_d2)
                                if delta < 0.0 or rand_u01(rng_states, s) < math.exp(-delta / (temp + 1e-4)):
                                    nodes[s, r1, i] = scratch_route[s, i]
                                    nodes[s, r2, j] = scratch_route2[s, j]
                                    dist[s] += delta
                                    cost[s] += delta * prob_data[4]

            temp *= cooling

    # -------------------------------------------------------------------------
    # 6. Deep Local Search (Systematic 2-Opt & Relocate)
    # -------------------------------------------------------------------------
    @dev_fn
    def deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                 scratch_route, scratch_route2,
                                 prob_data, max_passes=2):
        num_r = nr[s]
        if num_r == 0:
            return

        for _ in range(max_passes):
            improved = False
            for r in range(num_r):
                l = rlen[s, r]
                if l < 5:
                    continue
                for i in range(1, l - 3):
                    for j in range(i + 1, l - 2):
                        for k in range(l):
                            scratch_route[s, k] = nodes[s, r, k]
                        p1, p2 = i, j
                        while p1 < p2:
                            tmp = scratch_route[s, p1]
                            scratch_route[s, p1] = scratch_route[s, p2]
                            scratch_route[s, p2] = tmp
                            p1 += 1
                            p2 -= 1
                        ok, new_d = eval_route(scratch_route[s, :l], l, prob_data)
                        if ok:
                            _, old_d = eval_route(nodes[s, r, :l], l, prob_data)
                            if new_d < old_d - 1e-4:
                                for k in range(l):
                                    nodes[s, r, k] = scratch_route[s, k]
                                dist[s] += (new_d - old_d)
                                cost[s] += (new_d - old_d) * prob_data[4]
                                improved = True

            for r in range(num_r):
                l = rlen[s, r]
                if l < 4:
                    continue
                for i in range(1, l - 1):
                    for j in range(1, l - 1):
                        if i == j:
                            continue
                        for k in range(l):
                            scratch_route[s, k] = nodes[s, r, k]
                        val = scratch_route[s, i]
                        if i < j:
                            for k in range(i, j):
                                scratch_route[s, k] = scratch_route[s, k + 1]
                            scratch_route[s, j] = val
                        else:
                            for k in range(i, j, -1):
                                scratch_route[s, k] = scratch_route[s, k - 1]
                            scratch_route[s, j] = val
                        ok, new_d = eval_route(scratch_route[s, :l], l, prob_data)
                        if ok:
                            _, old_d = eval_route(nodes[s, r, :l], l, prob_data)
                            if new_d < old_d - 1e-4:
                                for k in range(l):
                                    nodes[s, r, k] = scratch_route[s, k]
                                dist[s] += (new_d - old_d)
                                cost[s] += (new_d - old_d) * prob_data[4]
                                improved = True

            # 3. Inter-route 2-opt* (swap tails of two routes)
            L = scratch_route.shape[1]
            dist_matrix = prob_data[10]
            for r1 in range(num_r - 1):
                len1 = rlen[s, r1]
                if len1 < 4:
                    continue
                found_tail_swap = False
                for r2 in range(r1 + 1, num_r):
                    len2 = rlen[s, r2]
                    if len2 < 4:
                        continue
                    for p1 in range(2, len1 - 1):
                        u1 = nodes[s, r1, p1 - 1]
                        v1 = nodes[s, r1, p1]
                        for p2 in range(2, len2 - 1):
                            u2 = nodes[s, r2, p2 - 1]
                            v2 = nodes[s, r2, p2]
                            new_len1 = p1 + len2 - p2
                            new_len2 = p2 + len1 - p1
                            if new_len1 >= L or new_len2 >= L:
                                continue
                            delta_d = (dist_matrix[u1, v2] + dist_matrix[u2, v1] -
                                       dist_matrix[u1, v1] - dist_matrix[u2, v2])
                            if delta_d < -1e-4:
                                for k in range(p1):
                                    scratch_route[s, k] = nodes[s, r1, k]
                                for k in range(p2, len2):
                                    scratch_route[s, p1 + (k - p2)] = nodes[s, r2, k]
                                ok1, d1 = eval_route(scratch_route[s, :new_len1], new_len1, prob_data)
                                if ok1:
                                    for k in range(p2):
                                        scratch_route2[s, k] = nodes[s, r2, k]
                                    for k in range(p1, len1):
                                        scratch_route2[s, p2 + (k - p1)] = nodes[s, r1, k]
                                    ok2, d2 = eval_route(scratch_route2[s, :new_len2], new_len2, prob_data)
                                    if ok2:
                                        _, old_d1 = eval_route(nodes[s, r1, :len1], len1, prob_data)
                                        _, old_d2 = eval_route(nodes[s, r2, :len2], len2, prob_data)
                                        if (d1 + d2) < (old_d1 + old_d2) - 1e-4:
                                            for k in range(new_len1):
                                                nodes[s, r1, k] = scratch_route[s, k]
                                            for k in range(new_len1, len1):
                                                nodes[s, r1, k] = 0
                                            for k in range(new_len2):
                                                nodes[s, r2, k] = scratch_route2[s, k]
                                            for k in range(new_len2, len2):
                                                nodes[s, r2, k] = 0
                                            rlen[s, r1] = new_len1
                                            rlen[s, r2] = new_len2
                                            change = (d1 + d2) - (old_d1 + old_d2)
                                            dist[s] += change
                                            cost[s] += change * prob_data[4]
                                            improved = True
                                            found_tail_swap = True
                                            break
                        if found_tail_swap:
                            break
                    if found_tail_swap:
                        break
                if found_tail_swap:
                    break

            # 4. Inter-route Relocate (move single customer between routes)
            for r1 in range(num_r):
                len1 = rlen[s, r1]
                if len1 < 4:
                    continue
                found_reloc = False
                for r2 in range(num_r):
                    if r1 == r2:
                        continue
                    len2 = rlen[s, r2]
                    if len2 + 1 >= L:
                        continue
                    for i in range(1, len1 - 1):
                        u = nodes[s, r1, i]
                        prev_u = nodes[s, r1, i - 1]
                        next_u = nodes[s, r1, i + 1]
                        cost_rem = dist_matrix[prev_u, next_u] - (dist_matrix[prev_u, u] + dist_matrix[u, next_u])
                        for j in range(1, len2):
                            prev_v = nodes[s, r2, j - 1]
                            next_v = nodes[s, r2, j]
                            cost_ins = (dist_matrix[prev_v, u] + dist_matrix[u, next_v]) - dist_matrix[prev_v, next_v]
                            if cost_rem + cost_ins < -1e-4:
                                k_idx = 0
                                for k in range(len1):
                                    if k != i:
                                        scratch_route[s, k_idx] = nodes[s, r1, k]
                                        k_idx += 1
                                ok1, d1 = eval_route(scratch_route[s, :len1 - 1], len1 - 1, prob_data)
                                if ok1:
                                    for k in range(j):
                                        scratch_route2[s, k] = nodes[s, r2, k]
                                    scratch_route2[s, j] = u
                                    for k in range(j, len2):
                                        scratch_route2[s, k + 1] = nodes[s, r2, k]
                                    ok2, d2 = eval_route(scratch_route2[s, :len2 + 1], len2 + 1, prob_data)
                                    if ok2:
                                        _, old_d1 = eval_route(nodes[s, r1, :len1], len1, prob_data)
                                        _, old_d2 = eval_route(nodes[s, r2, :len2], len2, prob_data)
                                        if (d1 + d2) < (old_d1 + old_d2) - 1e-4:
                                            for k in range(len1 - 1):
                                                nodes[s, r1, k] = scratch_route[s, k]
                                            nodes[s, r1, len1 - 1] = 0
                                            for k in range(len2 + 1):
                                                nodes[s, r2, k] = scratch_route2[s, k]
                                            rlen[s, r1] = len1 - 1
                                            rlen[s, r2] = len2 + 1
                                            change = (d1 + d2) - (old_d1 + old_d2)
                                            dist[s] += change
                                            cost[s] += change * prob_data[4]
                                            improved = True
                                            found_reloc = True
                                            break
                        if found_reloc:
                            break
                    if found_reloc:
                        break
                if found_reloc:
                    break

            if not improved:
                break

    # -------------------------------------------------------------------------
    # 7. SHO / WOA Offspring Generation (Guided Crossover & Intensification)
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
                r2 = randint(rng_states, s, 0, best_r_count - 1)
                if r2 == r1:
                    r2 = (r1 + 1) % best_r_count
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
        # From current
        for r in range(cur_nr[s]):
            for k in range(1, cur_rlen[s, r] - 1):
                node = cur_nodes[s, r, k]
                if node != depot and scratch_flags[s, node] == 0:
                    scratch_unrouted[s, unrouted_cnt] = node
                    scratch_flags[s, node] = 1
                    unrouted_cnt += 1
        # From peer
        for r in range(peer_nr[peer_s]):
            for k in range(1, peer_rlen[peer_s, r] - 1):
                node = peer_nodes[peer_s, r, k]
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

        # Insert unrouted customers into candidate routes with min delta
        for i in range(unrouted_cnt):
            node = scratch_unrouted[s, i]
            best_delta = 1e12
            best_r = -1
            best_p = -1

            for r in range(cand_nr[s]):
                l = cand_rlen[s, r]
                ok_o, old_d = eval_route(cand_nodes[s, r, :l], l, prob_data)
                for p in range(1, l):
                    for k in range(p):
                        scratch_route[s, k] = cand_nodes[s, r, k]
                    scratch_route[s, p] = node
                    for k in range(p, l):
                        scratch_route[s, k + 1] = cand_nodes[s, r, k]
                    ok_n, new_d = eval_route(scratch_route[s, :l+1], l + 1, prob_data)
                    if ok_n:
                        delta = new_d - old_d
                        if delta < best_delta:
                            best_delta = delta
                            best_r = r
                            best_p = p

            if best_r != -1:
                l = cand_rlen[s, best_r]
                for k in range(l, best_p, -1):
                    cand_nodes[s, best_r, k] = cand_nodes[s, best_r, k - 1]
                cand_nodes[s, best_r, best_p] = node
                cand_rlen[s, best_r] = l + 1
            else:
                # Open new route
                new_r = cand_nr[s]
                cand_nodes[s, new_r, 0] = depot
                cand_nodes[s, new_r, 1] = node
                cand_nodes[s, new_r, 2] = depot
                cand_rlen[s, new_r] = 3
                cand_nr[s] = new_r + 1

        eval_solution(cand_nodes, cand_rlen, cand_nr, s, prob_data)

    @dev_fn
    def woa_intensification_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                                  ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, island_id,
                                  cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                  scratch_route, scratch_unrouted, scratch_flags,
                                  a_param, prob_data, rng_states):
        depot = int(prob_data[0])
        customer_num = int(prob_data[14])
        r1 = rand_u01(rng_states, s)
        a_vec = 2.0 * a_param * r1 - a_param

        if abs(a_vec) < 1.0 and ibest_nr[island_id] > 0:
            # Encirclement: start from best, copy segments, repair
            copy_solution(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, island_id,
                          cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s)
            # Perturb 1-2 random nodes
            if cand_nr[s] > 0:
                r = randint(rng_states, s, 0, cand_nr[s] - 1)
                l = cand_rlen[s, r]
                if l >= 5:
                    p1 = randint(rng_states, s, 1, l - 3)
                    p2 = randint(rng_states, s, p1 + 1, l - 2)
                    tmp = cand_nodes[s, r, p1]
                    cand_nodes[s, r, p1] = cand_nodes[s, r, p2]
                    cand_nodes[s, r, p2] = tmp
            eval_solution(cand_nodes, cand_rlen, cand_nr, s, prob_data)
        else:
            # Exploration: start from current, apply random 2-opt or relocate
            copy_solution(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                          cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s)
            if cand_nr[s] > 0:
                r = randint(rng_states, s, 0, cand_nr[s] - 1)
                l = cand_rlen[s, r]
                if l >= 5:
                    p1 = randint(rng_states, s, 1, l - 3)
                    p2 = randint(rng_states, s, p1 + 1, l - 2)
                    while p1 < p2:
                        tmp = cand_nodes[s, r, p1]
                        cand_nodes[s, r, p1] = cand_nodes[s, r, p2]
                        cand_nodes[s, r, p2] = tmp
                        p1 += 1
                        p2 -= 1
            eval_solution(cand_nodes, cand_rlen, cand_nr, s, prob_data)

    @dev_fn
    def sho_woa_update_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                              next_nodes, next_rlen, next_nr, next_dist, next_cost,
                              cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                              ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                              scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                              s, island_id, island_size,
                              prob_data, a_param, p_mode, iteration, max_iter, rng_states):
        island_start = island_id * island_size
        p1 = island_start + randint(rng_states, s, 0, island_size - 1)
        p2 = island_start + randint(rng_states, s, 0, island_size - 1)
        p3 = island_start + randint(rng_states, s, 0, island_size - 1)

        best_peer = p1
        if is_better_lex(cur_nr[p2], cur_dist[p2], cur_nr[best_peer], cur_dist[best_peer]):
            best_peer = p2
        if is_better_lex(cur_nr[p3], cur_dist[p3], cur_nr[best_peer], cur_dist[best_peer]):
            best_peer = p3

        if rand_u01(rng_states, s) < p_mode:
            guided_crossover_sho_single(cur_nodes, cur_rlen, cur_nr, s,
                                        cur_nodes, cur_rlen, cur_nr, best_peer,
                                        ibest_nodes, ibest_rlen, ibest_nr, island_id,
                                        cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                        scratch_route, scratch_unrouted, scratch_flags,
                                        prob_data, rng_states)
        else:
            woa_intensification_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                                      ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, island_id,
                                      cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                      scratch_route, scratch_unrouted, scratch_flags,
                                      a_param, prob_data, rng_states)

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
        new_nr = cand_nr[s]
        old_nr = cur_nr[s]
        new_dist = cand_dist[s]
        old_dist = cur_dist[s]

        if new_nr < old_nr:
            accepted = True
        elif new_nr == old_nr:
            delta = new_dist - old_dist
            if delta <= 1e-3:
                accepted = True
            else:
                temp = 1.0 - float(iteration) / float(max_iter) if max_iter > 0 else 0.0
                prob = math.exp(-delta / (1e-6 + temp * abs(old_dist)))
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
                                island_id, island_size,
                                scratch_route, scratch_unrouted, scratch_flags,
                                prob_data, rng_states):
        island_start = island_id * island_size
        diversify_count = max(1, int(island_size * 0.40))
        # Diversify the worst individuals in this island
        for idx in range(diversify_count):
            s = island_start + island_size - 1 - idx
            # Ruin 20-40% customers
            customer_num = int(prob_data[14])
            depot = int(prob_data[0])
            for i in range(customer_num + 1):
                scratch_flags[s, i] = 0

            # Pick random customers to remove
            ruin_cnt = randint(rng_states, s, int(customer_num * 0.20), int(customer_num * 0.40))
            for i in range(customer_num):
                scratch_unrouted[s, i] = i + 1
            shuffle_ints(scratch_unrouted[s], customer_num, rng_states, s)
            for i in range(ruin_cnt):
                c = scratch_unrouted[s, i]
                scratch_flags[s, c] = 1

            # Remove flagged customers from routes
            for r in range(nr[s]):
                l = rlen[s, r]
                write_p = 1
                for p in range(1, l - 1):
                    node = nodes[s, r, p]
                    if scratch_flags[s, node] == 0:
                        nodes[s, r, write_p] = node
                        write_p += 1
                nodes[s, r, write_p] = depot
                rlen[s, r] = write_p + 1

            # Recreate: insert removed customers back with min delta
            for i in range(ruin_cnt):
                c = scratch_unrouted[s, i]
                best_delta = 1e12
                best_r = -1
                best_p = -1
                for r in range(nr[s]):
                    l = rlen[s, r]
                    if l <= 2:
                        continue
                    ok_o, old_d = eval_route(nodes[s, r, :l], l, prob_data)
                    for p in range(1, l):
                        for k in range(p):
                            scratch_route[s, k] = nodes[s, r, k]
                        scratch_route[s, p] = c
                        for k in range(p, l):
                            scratch_route[s, k + 1] = nodes[s, r, k]
                        ok_n, new_d = eval_route(scratch_route[s, :l+1], l + 1, prob_data)
                        if ok_n:
                            delta = new_d - old_d
                            if delta < best_delta:
                                best_delta = delta
                                best_r = r
                                best_p = p

                if best_r != -1:
                    l = rlen[s, best_r]
                    for k in range(l, best_p, -1):
                        nodes[s, best_r, k] = nodes[s, best_r, k - 1]
                    nodes[s, best_r, best_p] = c
                    rlen[s, best_r] = l + 1
                else:
                    new_r = nr[s]
                    nodes[s, new_r, 0] = depot
                    nodes[s, new_r, 1] = c
                    nodes[s, new_r, 2] = depot
                    rlen[s, new_r] = 3
                    nr[s] = new_r + 1

            eval_solution(nodes, rlen, nr, s, prob_data)

    # -------------------------------------------------------------------------
    # 9. Top-Level Kernels
    # -------------------------------------------------------------------------
    if is_cuda:
        @k_fn
        def init_population_kernel(nodes, rlen, nr, dist, cost,
                                   scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
                                   prob_data, alpha_lo, alpha_hi, rng_states):
            s = cuda.grid(1)
            if s < nodes.shape[0]:
                alpha = alpha_lo + rand_u01(rng_states, s) * (alpha_hi - alpha_lo)
                construct_rcrs_grasp(nodes, rlen, nr, dist, cost, s,
                                     scratch_route, scratch_unrouted, scratch_flags, scratch_scores,
                                     prob_data, alpha, rng_states)
                sa_warmup_single(nodes, rlen, nr, dist, cost, s,
                                 scratch_route, scratch_route2,
                                 prob_data, rng_states, sa_iters=20)
                route_elimination_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_unrouted, scratch_flags,
                                         prob_data, passes=5)
                deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_route2,
                                         prob_data, max_passes=2)

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
                    if is_better_lex(pop_nr[cand_s], pop_dist[cand_s], pop_nr[best_s], pop_dist[best_s]):
                        best_s = cand_s
                if is_better_lex(pop_nr[best_s], pop_dist[best_s], ibest_nr[isl], ibest_dist[isl]) or ibest_nr[isl] == 0:
                    copy_solution(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost, best_s,
                                  ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl)

        @k_fn
        def update_global_best_kernel(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                      gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost,
                                      num_islands):
            if cuda.grid(1) == 0:
                best_isl = 0
                for isl in range(1, num_islands):
                    if is_better_lex(ibest_nr[isl], ibest_dist[isl], ibest_nr[best_isl], ibest_dist[best_isl]):
                        best_isl = isl
                if is_better_lex(ibest_nr[best_isl], ibest_dist[best_isl], gbest_nr[0], gbest_dist[0]) or gbest_nr[0] == 0:
                    copy_solution(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, best_isl,
                                  gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost, 0)

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
                        if is_better_lex(pop_nr[worst_s], pop_dist[worst_s], pop_nr[cand_s], pop_dist[cand_s]):
                            worst_s = cand_s
                    # If ibest of isl is better than worst of dst_isl, migrate
                    if is_better_lex(ibest_nr[isl], ibest_dist[isl], pop_nr[worst_s], pop_dist[worst_s]):
                        copy_solution(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl,
                                      pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost, worst_s)

        @k_fn
        def stagnation_diversify_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                        scratch_route, scratch_unrouted, scratch_flags,
                                        prob_data, rng_states, num_islands, island_size):
            isl = cuda.grid(1)
            if isl < num_islands:
                diversify_island_single(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                        isl, island_size,
                                        scratch_route, scratch_unrouted, scratch_flags,
                                        prob_data, rng_states)

    else:
        # CPU Mode
        @k_fn
        def init_population_kernel(nodes, rlen, nr, dist, cost,
                                   scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
                                   prob_data, alpha_lo, alpha_hi, rng_states):
            for s in range(nodes.shape[0]):
                alpha = alpha_lo + rand_u01(rng_states, s) * (alpha_hi - alpha_lo)
                construct_rcrs_grasp(nodes, rlen, nr, dist, cost, s,
                                     scratch_route, scratch_unrouted, scratch_flags, scratch_scores,
                                     prob_data, alpha, rng_states)
                sa_warmup_single(nodes, rlen, nr, dist, cost, s,
                                 scratch_route, scratch_route2,
                                 prob_data, rng_states, sa_iters=20)
                route_elimination_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_unrouted, scratch_flags,
                                         prob_data, passes=5)
                deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_route2,
                                         prob_data, max_passes=2)

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
                    if is_better_lex(pop_nr[cand_s], pop_dist[cand_s], pop_nr[best_s], pop_dist[best_s]):
                        best_s = cand_s
                if is_better_lex(pop_nr[best_s], pop_dist[best_s], ibest_nr[isl], ibest_dist[isl]) or ibest_nr[isl] == 0:
                    copy_solution(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost, best_s,
                                  ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl)

        @k_fn
        def update_global_best_kernel(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                      gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost,
                                      num_islands):
            best_isl = 0
            for isl in range(1, num_islands):
                if is_better_lex(ibest_nr[isl], ibest_dist[isl], ibest_nr[best_isl], ibest_dist[best_isl]):
                    best_isl = isl
            if is_better_lex(ibest_nr[best_isl], ibest_dist[best_isl], gbest_nr[0], gbest_dist[0]) or gbest_nr[0] == 0:
                copy_solution(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, best_isl,
                              gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost, 0)

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
                        if is_better_lex(pop_nr[worst_s], pop_dist[worst_s], pop_nr[cand_s], pop_dist[cand_s]):
                            worst_s = cand_s
                    if is_better_lex(ibest_nr[isl], ibest_dist[isl], pop_nr[worst_s], pop_dist[worst_s]):
                        copy_solution(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl,
                                      pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost, worst_s)

        @k_fn
        def stagnation_diversify_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                        scratch_route, scratch_unrouted, scratch_flags,
                                        prob_data, rng_states, num_islands, island_size):
            for isl in range(num_islands):
                diversify_island_single(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                        isl, island_size,
                                        scratch_route, scratch_unrouted, scratch_flags,
                                        prob_data, rng_states)

    return {
        "init_population": init_population_kernel,
        "update_population": update_population_kernel,
        "route_elimination": route_elimination_kernel,
        "local_search": local_search_kernel,
        "update_island_bests": update_island_bests_kernel,
        "update_global_best": update_global_best_kernel,
        "island_migration": island_migration_kernel,
        "stagnation_diversify": stagnation_diversify_kernel,
    }
