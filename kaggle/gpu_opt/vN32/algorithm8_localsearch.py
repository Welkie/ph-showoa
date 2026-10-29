"""Algorithm 8. CombinedLocalSearch (GPU device code).

As src_python_baseline/algorithm8_localsearch.py: first improvement on the hierarchical objective (NV, then TD),
the scan restarts from 2-opt after every improving move and stops when no move improves; Relocate may empty a
route (one vehicle less). Step (d), Inter 2-opt*, is improvement 4.
"""

from __future__ import annotations


def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""
    compact_routes = k["compact_routes"]
    eval_route = k["eval_route"]
    inter_2opt_star_first_improvement = k["inter_2opt_star_first_improvement"]
    refresh_solution = k["refresh_solution"]

    @dev_fn
    def deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                 scratch_route, scratch_route2,
                                 prob_data, max_moves):
        compact_routes(nodes, rlen, nr, s)
        if nr[s] == 0:
            return
        L = scratch_route.shape[1]
        dist_matrix = prob_data[10]
        tol = 1e-4
        moves = 0
        while moves < max_moves:
            num_r = nr[s]
            found = False

            # (a) 2-opt: intra-route segment reversal
            for r in range(num_r):
                l = rlen[s, r]
                if l < 4:
                    continue
                ok0, old_d = eval_route(nodes[s, r, :l], l, prob_data)
                for p in range(1, l - 2):
                    for q in range(p + 1, l - 1):
                        for k in range(l):
                            scratch_route[s, k] = nodes[s, r, k]
                        a = p
                        b = q
                        while a < b:
                            tmp = scratch_route[s, a]
                            scratch_route[s, a] = scratch_route[s, b]
                            scratch_route[s, b] = tmp
                            a += 1
                            b -= 1
                        ok, nd = eval_route(scratch_route[s, :l], l, prob_data)
                        if ok and nd < old_d - tol:
                            for k in range(l):
                                nodes[s, r, k] = scratch_route[s, k]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                moves += 1
                continue

            # (b) Relocate: move one customer to another position of its route or to another route
            for rs in range(num_r):
                ls = rlen[s, rs]
                ok_s, old_s = eval_route(nodes[s, rs, :ls], ls, prob_data)
                for ps in range(1, ls - 1):
                    c = nodes[s, rs, ps]
                    prev_c = nodes[s, rs, ps - 1]
                    next_c = nodes[s, rs, ps + 1]
                    cost_rem = dist_matrix[prev_c, next_c] - dist_matrix[prev_c, c] - dist_matrix[c, next_c]
                    for rd in range(num_r):
                        if rd == rs:
                            n = ls - 2
                            i = ps - 1
                            for pd in range(n):
                                if pd == i or pd == i + 1:
                                    continue
                                ins = pd if pd < i else pd - 1
                                # route without c, then c inserted at customer index ins
                                w = 0
                                for k in range(ls):
                                    if k != ps:
                                        scratch_route[s, w] = nodes[s, rs, k]
                                        w += 1
                                for k in range(ls - 1, ins + 1, -1):
                                    scratch_route[s, k] = scratch_route[s, k - 1]
                                scratch_route[s, ins + 1] = c
                                ok, nd = eval_route(scratch_route[s, :ls], ls, prob_data)
                                if ok and nd < old_s - tol:
                                    for k in range(ls):
                                        nodes[s, rs, k] = scratch_route[s, k]
                                    found = True
                                    break
                        else:
                            ld = rlen[s, rd]
                            if ld + 1 > L:
                                continue
                            empties = ls == 3
                            ok_d0, old_dd = eval_route(nodes[s, rd, :ld], ld, prob_data)
                            for pdst in range(1, ld):
                                prev_v = nodes[s, rd, pdst - 1]
                                next_v = nodes[s, rd, pdst]
                                cost_ins = dist_matrix[prev_v, c] + dist_matrix[c, next_v] - dist_matrix[prev_v, next_v]
                                if not empties and cost_rem + cost_ins >= -tol:
                                    continue
                                w = 0
                                for k in range(ls):
                                    if k != ps:
                                        scratch_route[s, w] = nodes[s, rs, k]
                                        w += 1
                                ok1, d1 = eval_route(scratch_route[s, :ls - 1], ls - 1, prob_data)
                                if not ok1:
                                    continue
                                for k in range(pdst):
                                    scratch_route2[s, k] = nodes[s, rd, k]
                                scratch_route2[s, pdst] = c
                                for k in range(pdst, ld):
                                    scratch_route2[s, k + 1] = nodes[s, rd, k]
                                ok2, d2 = eval_route(scratch_route2[s, :ld + 1], ld + 1, prob_data)
                                if not ok2:
                                    continue
                                # Hierarchical: emptying the source route lowers NV; otherwise TD must drop
                                if empties or (d1 + d2) < (old_s + old_dd) - tol:
                                    for k in range(ls - 1):
                                        nodes[s, rs, k] = scratch_route[s, k]
                                    rlen[s, rs] = ls - 1
                                    for k in range(ld + 1):
                                        nodes[s, rd, k] = scratch_route2[s, k]
                                    rlen[s, rd] = ld + 1
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                compact_routes(nodes, rlen, nr, s)
                moves += 1
                continue

            # (c) Swap: exchange two customers of distinct routes
            for r1 in range(num_r - 1):
                len1 = rlen[s, r1]
                for r2 in range(r1 + 1, num_r):
                    len2 = rlen[s, r2]
                    for i in range(1, len1 - 1):
                        u = nodes[s, r1, i]
                        prev_u = nodes[s, r1, i - 1]
                        next_u = nodes[s, r1, i + 1]
                        for j in range(1, len2 - 1):
                            v = nodes[s, r2, j]
                            prev_v = nodes[s, r2, j - 1]
                            next_v = nodes[s, r2, j + 1]
                            delta_d = (dist_matrix[prev_u, v] + dist_matrix[v, next_u] +
                                       dist_matrix[prev_v, u] + dist_matrix[u, next_v] -
                                       dist_matrix[prev_u, u] - dist_matrix[u, next_u] -
                                       dist_matrix[prev_v, v] - dist_matrix[v, next_v])
                            if delta_d >= -tol:
                                continue
                            for k in range(len1):
                                scratch_route[s, k] = nodes[s, r1, k]
                            scratch_route[s, i] = v
                            ok1, d1 = eval_route(scratch_route[s, :len1], len1, prob_data)
                            if not ok1:
                                continue
                            for k in range(len2):
                                scratch_route2[s, k] = nodes[s, r2, k]
                            scratch_route2[s, j] = u
                            ok2, d2 = eval_route(scratch_route2[s, :len2], len2, prob_data)
                            if not ok2:
                                continue
                            _, old_d1 = eval_route(nodes[s, r1, :len1], len1, prob_data)
                            _, old_d2 = eval_route(nodes[s, r2, :len2], len2, prob_data)
                            if (d1 + d2) < (old_d1 + old_d2) - tol:
                                for k in range(len1):
                                    nodes[s, r1, k] = scratch_route[s, k]
                                for k in range(len2):
                                    nodes[s, r2, k] = scratch_route2[s, k]
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                moves += 1
                continue

            # (d) Inter 2-opt* (improvement 4, improvement4_population_local_search.py)
            if inter_2opt_star_first_improvement(nodes, rlen, nr, s, scratch_route, scratch_route2, prob_data, tol):
                moves += 1
                continue
            break

        refresh_solution(nodes, rlen, nr, dist, cost, s, prob_data)

    k.update({"deep_local_search_single": deep_local_search_single})
