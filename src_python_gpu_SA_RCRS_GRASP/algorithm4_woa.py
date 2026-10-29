"""Algorithm 4. BasedOnWOAUpdate (GPU device code).

Per route: encircling (|A| < 1) or a random intra / multi-route operation (|A| >= 1) with probability 0.5, otherwise
the spiral movement; route-level rollback (only the routes the iteration changed are saved and restored); an infeasible result is repaired (Algorithm 10) and, if still infeasible,
the current whale is returned. Otherwise the candidate is returned even when it is worse (Direction B): Algorithm 1's
SA acceptance decides, as in src_python_baseline/algorithm4_woa.py. The best is the global best.
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
    def save_route(nodes, rlen, s, r, bak_nodes, bak_rlen, slot):
        l = rlen[s, r]
        bak_rlen[s, slot] = l
        for kk in range(l):
            bak_nodes[s, slot, kk] = nodes[s, r, kk]

    @dev_fn
    def restore_route(bak_nodes, bak_rlen, s, slot, nodes, rlen, r):
        l = bak_rlen[s, slot]
        rlen[s, r] = l
        for kk in range(l):
            nodes[s, r, kk] = bak_nodes[s, slot, kk]

    @dev_fn
    def woa_generate(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                     best_nodes, best_rlen, best_nr,
                     cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                     bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                     scratch_unrouted, scratch_flags, a_param, prob_data, rng_states):
        # Algorithm 4 up to the candidate (route updates with rollback); the caller refreshes and repairs it
        customer_num = int(prob_data[14])
        copy_solution(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                      cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s)
        for i in range(customer_num + 1):
            scratch_flags[s, i] = 0
        dim = cand_nr[s]
        best_dim = best_nr[0]
        limit = dim if dim < best_dim else best_dim
        for i in range(limit):
            p = rand_u01(rng_states, s)
            ai = 2.0 * a_param * rand_u01(rng_states, s) - a_param
            ci = 2.0 * rand_u01(rng_states, s)
            # Route-level rollback: an iteration changes route i, or routes r1 and r2 (multi-route operation);
            # only those are saved (slots 0, 1, 2 of bak) instead of the whole solution
            save_route(cand_nodes, cand_rlen, s, i, bak_nodes, bak_rlen, 0)
            inter = False
            r1 = 0
            r2 = 0
            if p < 0.5:
                if abs(ai) < 1.0:
                    relink_route(cand_nodes, cand_rlen, s, i, best_nodes, best_rlen, 0, i,
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
                            save_route(cand_nodes, cand_rlen, s, r1, bak_nodes, bak_rlen, 1)
                            save_route(cand_nodes, cand_rlen, s, r2, bak_nodes, bak_rlen, 2)
                            inter = True
                            changed = apply_inter_op(cand_nodes, cand_rlen, s, r1, r2, op, rng_states)
                        if changed and not all_routes_feasible(cand_nodes, cand_rlen, cand_nr, s, prob_data):
                            restore_route(bak_nodes, bak_rlen, s, 1, cand_nodes, cand_rlen, r1)
                            restore_route(bak_nodes, bak_rlen, s, 2, cand_nodes, cand_rlen, r2)
            else:
                l_val = ci - 1.0
                frac = abs(math.exp(l_val) * math.cos(2.0 * math.pi * l_val))
                if frac > 1.0:
                    frac = 1.0
                relink_route(cand_nodes, cand_rlen, s, i, best_nodes, best_rlen, 0, i,
                             frac, l_val >= 0.0, scratch_unrouted, scratch_flags)
            li = cand_rlen[s, i]
            ok_i, d_i = eval_route(cand_nodes[s, i, :li], li, prob_data)
            if not ok_i:
                if inter:
                    restore_route(bak_nodes, bak_rlen, s, 1, cand_nodes, cand_rlen, r1)
                    restore_route(bak_nodes, bak_rlen, s, 2, cand_nodes, cand_rlen, r2)
                restore_route(bak_nodes, bak_rlen, s, 0, cand_nodes, cand_rlen, i)

    @dev_fn
    def woa_update_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                          best_nodes, best_rlen, best_nr,
                          cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                          bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                          scratch_route, scratch_unrouted, scratch_flags,
                          a_param, prob_data, rng_states):
        # Algorithm 4. Returns True when a feasible candidate is left in cand (False: current whale returned).
        woa_generate(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                     best_nodes, best_rlen, best_nr,
                     cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                     bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                     scratch_unrouted, scratch_flags, a_param, prob_data, rng_states)
        ok = refresh_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, prob_data)
        if not ok:
            # Feasibility repair; if still infeasible the current whale is returned
            repair_single(cand_nodes, cand_rlen, cand_nr, s, scratch_route, prob_data)
            ok = refresh_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, prob_data)
        # Direction B: the feasible candidate is returned even when nf > cf; Algorithm 1 decides acceptance
        return ok

    k.update({"relink_route": relink_route, "woa_generate": woa_generate, "woa_update_single": woa_update_single})
