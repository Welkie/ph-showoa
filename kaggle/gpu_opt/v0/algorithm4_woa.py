"""Algorithm 4. BasedOnWOAUpdate (GPU device code).

Per route: encircling (|A| < 1) or a random intra / multi-route operation (|A| >= 1) with probability 0.5, otherwise
the spiral movement; route-level rollback; an infeasible result is repaired (Algorithm 10) and, if still infeasible,
the current whale is returned. Otherwise the candidate is returned even when it is worse (Direction B): Algorithm 1's
SA acceptance decides, as in src_python_baseline/algorithm4_woa.py. The best is the island best.
"""

from __future__ import annotations

import math


def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""
    all_routes_feasible = k["all_routes_feasible"]
    apply_inter_op = k["apply_inter_op"]
    apply_intra_op = k["apply_intra_op"]
    copy_solution = k["copy_solution"]
    eval_route = k["eval_route"]
    pick_two_routes = k["pick_two_routes"]
    rand_u01 = k["rand_u01"]
    randint = k["randint"]
    refresh_solution = k["refresh_solution"]
    repair_single = k["repair_single"]

    @dev_fn
    def relink_route(nodes, rlen, s, r, t_nodes, t_rlen, t_idx, tr, fraction, forward,
                     scratch_unrouted, scratch_flags):
        # ENCIRCLING_PREY / SPIRAL_MOVEMENT (same choice as src_python_baseline): reorder the first
        # max(2, round(fraction * |common|)) shared customers of route r into their order in the target route
        l = rlen[s, r]
        tl = t_rlen[t_idx, tr]
        if l - 2 < 2 or tl <= 2:
            return
        for k in range(1, tl - 1):
            scratch_flags[s, t_nodes[t_idx, tr, k]] = k
        cnt = 0
        for k in range(1, l - 1):
            c = nodes[s, r, k]
            if scratch_flags[s, c] > 0:
                scratch_unrouted[s, cnt] = c
                cnt += 1
        if cnt >= 2:
            f = fraction
            if f < 0.1:
                f = 0.1
            if f > 1.0:
                f = 1.0
            num = int(math.floor(cnt * f + 0.5))
            if num < 2:
                num = 2
            if num > cnt:
                num = cnt
            for a in range(1, num):
                key = scratch_unrouted[s, a]
                b = a - 1
                while b >= 0 and ((forward and scratch_flags[s, scratch_unrouted[s, b]] > scratch_flags[s, key]) or
                                  ((not forward) and scratch_flags[s, scratch_unrouted[s, b]] < scratch_flags[s, key])):
                    scratch_unrouted[s, b + 1] = scratch_unrouted[s, b]
                    b -= 1
                scratch_unrouted[s, b + 1] = key
            used = 0
            for k in range(1, l - 1):
                if used >= num:
                    break
                if scratch_flags[s, nodes[s, r, k]] > 0:
                    nodes[s, r, k] = scratch_unrouted[s, used]
                    used += 1
        for k in range(1, tl - 1):
            scratch_flags[s, t_nodes[t_idx, tr, k]] = 0

    @dev_fn
    def woa_update_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                          ibest_nodes, ibest_rlen, ibest_nr, island_id,
                          cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                          bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                          scratch_route, scratch_unrouted, scratch_flags,
                          a_param, prob_data, rng_states):
        # Algorithm 4. Returns True when a feasible candidate is left in cand (False: current whale returned).
        customer_num = int(prob_data[14])
        copy_solution(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                      cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s)
        for i in range(customer_num + 1):
            scratch_flags[s, i] = 0
        dim = cand_nr[s]
        best_dim = ibest_nr[island_id]
        limit = dim if dim < best_dim else best_dim
        for i in range(limit):
            p = rand_u01(rng_states, s)
            ai = 2.0 * a_param * rand_u01(rng_states, s) - a_param
            ci = 2.0 * rand_u01(rng_states, s)
            copy_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                          bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost, s)
            if p < 0.5:
                if abs(ai) < 1.0:
                    relink_route(cand_nodes, cand_rlen, s, i, ibest_nodes, ibest_rlen, island_id, i,
                                 (1.0 - abs(ai)) * (ci / 2.0), ai >= 0.0, scratch_unrouted, scratch_flags)
                else:
                    if rand_u01(rng_states, s) < 0.5:
                        op = randint(rng_states, s, 0, 2)
                        apply_intra_op(cand_nodes, cand_rlen, s, i, op, rng_states)
                    else:
                        op = randint(rng_states, s, 0, 1)
                        changed = False
                        if cand_nr[s] >= 2:
                            r1, r2 = pick_two_routes(cand_nr, s, rng_states)
                            changed = apply_inter_op(cand_nodes, cand_rlen, s, r1, r2, op, rng_states)
                        if changed and not all_routes_feasible(cand_nodes, cand_rlen, cand_nr, s, prob_data):
                            copy_solution(bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost, s,
                                          cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s)
            else:
                l_val = ci - 1.0
                frac = abs(math.exp(l_val) * math.cos(2.0 * math.pi * l_val))
                if frac > 1.0:
                    frac = 1.0
                relink_route(cand_nodes, cand_rlen, s, i, ibest_nodes, ibest_rlen, island_id, i,
                             frac, l_val >= 0.0, scratch_unrouted, scratch_flags)
            li = cand_rlen[s, i]
            ok_i, d_i = eval_route(cand_nodes[s, i, :li], li, prob_data)
            if not ok_i:
                copy_solution(bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost, s,
                              cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s)

        ok = refresh_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, prob_data)
        if not ok:
            # Feasibility repair; if still infeasible the current whale is returned
            repair_single(cand_nodes, cand_rlen, cand_nr, s, scratch_route, prob_data)
            ok = refresh_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, prob_data)
        # Direction B: the feasible candidate is returned even when nf > cf; Algorithm 1 decides acceptance
        return ok

    k.update({"relink_route": relink_route, "woa_update_single": woa_update_single})
