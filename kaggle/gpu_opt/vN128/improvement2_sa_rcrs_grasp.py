"""Improvement 2 (docs/improved.md section 2): SA_RCRS_GRASP initialization with route elimination.

Replaces Algorithm 2 by four stages run per individual on the GPU: RCRS-GRASP construction, a short SA warm-up,
route elimination (also re-applied to the population every 25 iterations) and Algorithm 8.
"""

from __future__ import annotations

try:
    from numba import cuda
except ImportError:  # pragma: no cover - optional dependency
    cuda = None

import math

# RCRS insertion score = delta_TD + RC_WEIGHT * max(0, peak load - RC_THRESHOLD * Q) + RS_WEIGHT * radial deviation.
# SEED_RULE: customer that opens a new route (0 random, 1 earliest due time, 2 farthest from the depot).
# Read when the kernels are compiled.
RC_WEIGHT = 0.5
RC_THRESHOLD = 0.70
RS_WEIGHT = 0.3
SEED_RULE = 0


def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""
    eval_route = k["eval_route"]
    eval_solution = k["eval_solution"]
    rand_u01 = k["rand_u01"]
    randint = k["randint"]
    shuffle_ints = k["shuffle_ints"]
    rc_weight = RC_WEIGHT
    rc_threshold = RC_THRESHOLD
    rs_weight = RS_WEIGHT
    seed_rule = SEED_RULE

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
        rc_pen = c_new - capacity * rc_threshold
        if rc_pen < 0.0:
            rc_pen = 0.0

        dx_prev = dist_matrix[depot, prev]
        dx_c = dist_matrix[depot, customer]
        rs_pen = abs(dx_prev + dist_matrix[prev, customer] - dx_c)

        return delta_td + rc_weight * rc_pen + rs_weight * rs_pen

    @dev_fn
    def pick_seed(unrouted, count, s, prob_data, rng_states):
        # Index (in unrouted) of the customer that opens a new route
        if seed_rule == 0:
            return randint(rng_states, s, 0, count - 1)
        end_tw = prob_data[8]
        dist_matrix = prob_data[10]
        depot = int(prob_data[0])
        best = 0
        for i in range(1, count):
            c = unrouted[s, i]
            b = unrouted[s, best]
            if seed_rule == 1:
                if end_tw[c] < end_tw[b]:
                    best = i
            elif dist_matrix[depot, c] > dist_matrix[depot, b]:
                best = i
        return best

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
                pick = pick_seed(scratch_unrouted, unrouted_count, s, prob_data, rng_states)
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
                pick = pick_seed(scratch_unrouted, unrouted_count, s, prob_data, rng_states)
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
                pick = pick_seed(scratch_unrouted, unrouted_count, s, prob_data, rng_states)
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
                            if delta < 0.0 or rand_u01(rng_states, s) < math.exp(-delta / (temp + 1e-3)):
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
                                if delta < 0.0 or rand_u01(rng_states, s) < math.exp(-delta / (temp + 1e-3)):
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
                                if delta < 0.0 or rand_u01(rng_states, s) < math.exp(-delta / (temp + 1e-3)):
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
                                if delta < 0.0 or rand_u01(rng_states, s) < math.exp(-delta / (temp + 1e-3)):
                                    nodes[s, r1, i] = scratch_route[s, i]
                                    nodes[s, r2, j] = scratch_route2[s, j]
                                    dist[s] += delta
                                    cost[s] += delta * prob_data[4]

            temp *= cooling

    k.update({"rcrs_score": rcrs_score, "pick_seed": pick_seed, "construct_rcrs_grasp": construct_rcrs_grasp, "route_elimination_single": route_elimination_single, "sa_warmup_single": sa_warmup_single})


def build_kernels(is_cuda, k_fn, k, kernels):
    """Route elimination for the whole population (run every local_search_interval iterations)."""
    route_elimination_single = k["route_elimination_single"]
    if is_cuda:
        @k_fn
        def route_elimination_kernel(nodes, rlen, nr, dist, cost,
                                     scratch_route, scratch_unrouted, scratch_flags, prob_data, passes):
            s = cuda.grid(1)
            if s < nodes.shape[0]:
                route_elimination_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_unrouted, scratch_flags, prob_data, passes)
    else:
        @k_fn
        def route_elimination_kernel(nodes, rlen, nr, dist, cost,
                                     scratch_route, scratch_unrouted, scratch_flags, prob_data, passes):
            for s in range(nodes.shape[0]):
                route_elimination_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_unrouted, scratch_flags, prob_data, passes)
    kernels["route_elimination"] = route_elimination_kernel
