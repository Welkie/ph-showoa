"""Improvement 2 (docs/improved.md section 2), CUDA only: route minimization by an ejection pool after the
initialization (Nagata & Braysy 2009, without the squeeze and perturbation steps).

Repeated until ep_budget customers have been taken from the pool: a random route is removed and its customers are
put in the ejection pool (a LIFO stack). The top customer c is inserted at its feasible position of smallest extra
distance; when there is none, c is inserted into a route while one customer of that route (two customers only when no
single ejection is feasible and EP_KMAX = 2) is ejected and pushed onto the stack, choosing the feasible (route,
position, ejected set) of smallest sum of penalty counters, then smallest extra distance, and the counter of c is
incremented (the counters are reset to 1 for every removed route, so that customers which are hard to insert are not
ejected again and again). The route is eliminated when the pool is empty; when the budget runs out first, the solution before the
removal is restored, also when EP_ATTEMPT_POPS customers were taken for this route; the procedure stops after
EP_MAX_FAILS such failures in a row.
The (route, position, ejected set) choices are spread over the threads of the block (block minimum, smallest code on
ties); thread 0 does the random draws and the writes.
"""

from __future__ import annotations

try:
    from numba import cuda
except ImportError:  # pragma: no cover - optional dependency
    cuda = None

PEN_SCALE = 1e6     # key = penalty sum * PEN_SCALE + extra distance (the distances are far below PEN_SCALE)
EP_ATTEMPT_POPS = 300   # a route removal fails after this many customers taken from the pool (read at compile time)
EP_MAX_FAILS = 10       # stop after this many failed route removals in a row (read at compile time)
EP_KMAX = 2             # at most this many customers ejected for one insertion, 1 or 2 (read at compile time)


def build(dev_fn, k):
    """Device functions of the ejection-pool route minimization (CUDA only)."""
    eval_route = k["eval_route"]
    eval_solution = k["eval_solution"]
    eval_virtual = k["eval_virtual"]
    insert_at = k["insert_at"]
    block_min = k["block_min"]
    copy_block = k["copy_block"]
    randint = k["randint"]
    max_fails = EP_MAX_FAILS
    attempt_pops = EP_ATTEMPT_POPS
    kmax = EP_KMAX

    @dev_fn
    def eval_eject(nodes, s, r, l, p, c, e1, e2, prob_data):
        # eval_route of route r with c inserted before position p and the customers at e1 and e2 removed (e2 < 0:
        # only e1), same checks in the same order as eval_route
        capacity = prob_data[1]
        delivery = prob_data[5]
        pickup = prob_data[6]
        start_tw = prob_data[7]
        end_tw = prob_data[8]
        service = prob_data[9]
        dist_matrix = prob_data[10]
        time_matrix = prob_data[11]
        load = delivery[c]
        for i in range(1, l - 1):
            if i != e1 and i != e2:
                load += delivery[nodes[s, r, i]]
        if load > capacity + 1e-9:
            return False, 0.0
        distance_val = 0.0
        time_val = prob_data[2]
        prev = nodes[s, r, 0]
        for i in range(1, l):
            for step in range(2):
                if step == 0:
                    if i != p:
                        continue
                    node = c
                else:
                    if i == e1 or i == e2:
                        continue
                    node = nodes[s, r, i]
                load = load - delivery[node] + pickup[node]
                if load < -1e-9 or load > capacity + 1e-9:
                    return False, 0.0
                time_val += time_matrix[prev, node]
                if time_val > end_tw[node] + 1e-9:
                    return False, 0.0
                if time_val < start_tw[node]:
                    time_val = start_tw[node]
                time_val += service[node]
                distance_val += dist_matrix[prev, node]
                prev = node
        return True, distance_val

    @dev_fn
    def ejection_pool_block(nodes, rlen, nr, dist, cost, s, bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                            stack, pen, prob_data, rng_states, budget, t, nt, sv, si, sh):
        customer_num = int(prob_data[14])
        L = nodes.shape[2]
        left = budget
        fails = 0
        while left > 0 and nr[s] > 1 and fails < max_fails:
            copy_block(nodes, rlen, nr, dist, cost, s, bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost, s, t, nt)
            if t == 0:
                num_r = nr[s]
                rm = randint(rng_states, s, 0, num_r - 1)
                n = 0
                for i in range(1, rlen[s, rm] - 1):
                    stack[s, n] = nodes[s, rm, i]
                    n += 1
                for r in range(rm, num_r - 1):
                    rlen[s, r] = rlen[s, r + 1]
                    for i in range(rlen[s, r]):
                        nodes[s, r, i] = nodes[s, r + 1, i]
                rlen[s, num_r - 1] = 0
                nr[s] = num_r - 1
                for cc in range(1, customer_num + 1):
                    pen[s, cc] = 1.0
                sh[0] = n
            cuda.syncthreads()
            pops = 0
            while sh[0] > 0 and left > 0 and pops < attempt_pops:
                left -= 1
                pops += 1
                n = sh[0]
                c = stack[s, n - 1]
                num_r = nr[s]
                # Feasible insertion of smallest extra distance
                best = 1e12
                code = -1
                cnt = 0
                for r in range(num_r):
                    l = rlen[s, r]
                    have_old = False
                    old_d = 0.0
                    for pos in range(1, l):
                        if cnt % nt == t:
                            ok, nd = eval_virtual(nodes, s, r, l + 1, 0, pos, 0, c, prob_data)
                            if ok:
                                if not have_old:
                                    ok_old, old_d = eval_route(nodes[s, r, :l], l, prob_data)
                                    have_old = True
                                if nd - old_d < best:
                                    best = nd - old_d
                                    code = r * L + pos
                        cnt += 1
                v, code = block_min(sv, si, t, nt, best, code)
                if code >= 0:
                    if t == 0:
                        tr = code // L
                        insert_at(nodes, rlen, s, tr, code - tr * L, c)
                        sh[0] = n - 1
                    cuda.syncthreads()
                    continue
                # Insertion with the ejection of one customer of the route (e2 == e1); two customers (e1 < e2) only
                # when no single ejection is feasible
                code = -1
                for k_ej in range(kmax):
                    best = 1e300
                    code = -1
                    cnt = 0
                    for r in range(num_r):
                        l = rlen[s, r]
                        have_old = False
                        old_d = 0.0
                        for e1 in range(1, l - 1):
                            for pos in range(1, l):
                                if cnt % nt == t:
                                    p1 = pen[s, nodes[s, r, e1]]
                                    e2_lo = e1 if k_ej == 0 else e1 + 1
                                    e2_hi = e1 + 1 if k_ej == 0 else l - 1
                                    for e2 in range(e2_lo, e2_hi):
                                        pe = p1 if e2 == e1 else p1 + pen[s, nodes[s, r, e2]]
                                        if pe * PEN_SCALE - 0.5 * PEN_SCALE > best:
                                            continue
                                        ok, nd = eval_eject(nodes, s, r, l, pos, c, e1, e2 if e2 != e1 else -1,
                                                            prob_data)
                                        if ok:
                                            if not have_old:
                                                ok_old, old_d = eval_route(nodes[s, r, :l], l, prob_data)
                                                have_old = True
                                            key = pe * PEN_SCALE + (nd - old_d)
                                            if key < best:
                                                best = key
                                                code = ((r * L + pos) * L + e1) * L + e2
                                cnt += 1
                    v, code = block_min(sv, si, t, nt, best, code)
                    if code >= 0:
                        break
                if code < 0:
                    break
                if t == 0:
                    e2 = code % L
                    rest = code // L
                    e1 = rest % L
                    rest = rest // L
                    pos = rest % L
                    r = rest // L
                    pen[s, c] += 1.0
                    l = rlen[s, r]
                    u1 = nodes[s, r, e1]
                    u2 = nodes[s, r, e2]
                    # kept customers before pos, then compact the kept customers to positions 1..m and insert c
                    before = 0
                    for i in range(1, pos):
                        if i != e1 and i != e2:
                            before += 1
                    w = 1
                    for i in range(1, l - 1):
                        if i != e1 and i != e2:
                            nodes[s, r, w] = nodes[s, r, i]
                            w += 1
                    nodes[s, r, w] = nodes[s, r, l - 1]
                    rlen[s, r] = w + 1
                    insert_at(nodes, rlen, s, r, before + 1, c)
                    stack[s, n - 1] = u1
                    if e2 != e1:
                        stack[s, n] = u2
                        sh[0] = n + 1
                    else:
                        sh[0] = n
                cuda.syncthreads()
            if sh[0] == 0:
                fails = 0
                if t == 0:
                    ok, n_act, t_dist, t_cost = eval_solution(nodes, rlen, nr, s, prob_data)
                    dist[s] = t_dist
                    cost[s] = t_cost
                cuda.syncthreads()
            else:
                fails += 1
                copy_block(bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost, s, nodes, rlen, nr, dist, cost, s, t, nt)

    k.update({"eval_eject": eval_eject, "ejection_pool_block": ejection_pool_block})
