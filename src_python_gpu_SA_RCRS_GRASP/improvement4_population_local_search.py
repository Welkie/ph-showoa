"""Improvement 4 (docs/improved.md section 4): local search for the whole population, executed block-parallel.

Every local_search_interval iterations (and at the end of the sa_rcrs_grasp initialization) every individual is
improved, not only the global best as in Algorithm 8 of the paper, and Inter 2-opt* (tail exchange between two
routes) is added to 2-opt, Relocate and Swap. One CUDA block of NT threads works on one individual:
  type 0  2-opt inside a route (reverse positions ka..kb)
  type 1  Relocate the customer at slot a to just after slot b (same or another route; emptying a route = NV - 1)
  type 2  Swap two customers of different routes
  type 3  Inter 2-opt* (exchange the tails after ka / kb; no emptying)
A slot is a (route, position) pair; thread t evaluates the pair indices t, t + NT, ... The block keeps the move with
the smallest key (delta distance, minus BIG when a route is emptied: NV first, then TD), ties broken by the smallest
move code, applies it and scans again until no move improves (best improvement). The result does not depend on the
number of threads, so the CPU backend runs the same scan with one thread and returns the same solution. Distance
deltas are O(1); feasibility (Section 3: load and time windows) is checked only for a move that improves and beats
the thread's best so far, by walking the new route segment by segment without copying it.

--ls_strategy first keeps the previous execution (Algorithm 8, first improvement, one thread per individual, with
inter_2opt_star_first_improvement as step (d)) for the ablation of docs/improved.md section 4.
"""

from __future__ import annotations

try:
    from numba import cuda, float64, int64   # module level: the CUDA simulator patches module globals
except ImportError:  # pragma: no cover - optional dependency
    cuda = None

NT = 128           # threads per block
TOL = 1e-4         # same improvement tolerance as algorithm8_localsearch.py
BIG = 1.0e7        # key bonus of a move that empties a route


def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""
    eval_route = k["eval_route"]
    compact_routes = k["compact_routes"]

    @dev_fn
    def inter_2opt_star_first_improvement(nodes, rlen, nr, s, scratch_route, scratch_route2, prob_data, tol):
        # Inter 2-opt*: swap the tails of two routes; applies the first improving move and returns True
        num_r = nr[s]
        L = scratch_route.shape[1]
        dist_matrix = prob_data[10]
        found = False
        for r1 in range(num_r - 1):
            len1 = rlen[s, r1]
            if len1 < 4:
                continue
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
                        if delta_d >= -tol:
                            continue
                        for k in range(p1):
                            scratch_route[s, k] = nodes[s, r1, k]
                        for k in range(p2, len2):
                            scratch_route[s, p1 + (k - p2)] = nodes[s, r2, k]
                        ok1, d1 = eval_route(scratch_route[s, :new_len1], new_len1, prob_data)
                        if not ok1:
                            continue
                        for k in range(p2):
                            scratch_route2[s, k] = nodes[s, r2, k]
                        for k in range(p1, len1):
                            scratch_route2[s, p2 + (k - p1)] = nodes[s, r1, k]
                        ok2, d2 = eval_route(scratch_route2[s, :new_len2], new_len2, prob_data)
                        if not ok2:
                            continue
                        _, old_d1 = eval_route(nodes[s, r1, :len1], len1, prob_data)
                        _, old_d2 = eval_route(nodes[s, r2, :len2], len2, prob_data)
                        if (d1 + d2) < (old_d1 + old_d2) - tol:
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
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        return found


    @dev_fn
    def seg_node(nodes, s, r, a, b, rev, t):
        return nodes[s, r, b - t] if rev else nodes[s, r, a + t]

    @dev_fn
    def eval_segments(nodes, s, prob_data, r0, a0, b0, v0, r1, a1, b1, v1, r2, a2, b2, v2, r3, a3, b3, v3):
        # Feasibility of the route made of up to 4 segments (positions a..b of route r, reversed if v; empty if
        # a > b), with the same checks and tolerances as eval_route: start load = sum of deliveries <= Q, then
        # load - d + p in [0, Q], arrival <= due time (depot included), wait until the ready time, add service.
        capacity = prob_data[1]
        start_time = prob_data[2]
        delivery = prob_data[5]
        pickup = prob_data[6]
        start_tw = prob_data[7]
        end_tw = prob_data[8]
        service = prob_data[9]
        time_matrix = prob_data[11]
        load = 0.0
        for g in range(4):
            if g == 0:
                r, a, b, v = r0, a0, b0, v0
            elif g == 1:
                r, a, b, v = r1, a1, b1, v1
            elif g == 2:
                r, a, b, v = r2, a2, b2, v2
            else:
                r, a, b, v = r3, a3, b3, v3
            for t in range(b - a + 1):
                load += delivery[seg_node(nodes, s, r, a, b, v, t)]
        if load > capacity + 1e-9:
            return False
        time_val = start_time
        prev = 0
        first = True
        for g in range(4):
            if g == 0:
                r, a, b, v = r0, a0, b0, v0
            elif g == 1:
                r, a, b, v = r1, a1, b1, v1
            elif g == 2:
                r, a, b, v = r2, a2, b2, v2
            else:
                r, a, b, v = r3, a3, b3, v3
            for t in range(b - a + 1):
                node = seg_node(nodes, s, r, a, b, v, t)
                if first:                       # leading depot
                    first = False
                    prev = node
                    continue
                load = load - delivery[node] + pickup[node]
                if load < -1e-9 or load > capacity + 1e-9:
                    return False
                time_val += time_matrix[prev, node]
                if time_val > end_tw[node] + 1e-9:
                    return False
                if time_val < start_tw[node]:
                    time_val = start_tw[node]
                time_val += service[node]
                prev = node
        return True

    @dev_fn
    def scan_moves(nodes, rlen, s, slot_r, slot_k, n_slots, start, step, prob_data, use_2opt_star):
        # Best (key, code) over the pair indices start, start + step, ...; code = pair index * 4 + move type
        dm = prob_data[10]
        best_key = 0.0
        best_code = -1
        for idx in range(start, n_slots * n_slots, step):
            sa = idx // n_slots
            sb = idx - sa * n_slots
            ra = slot_r[s, sa]
            ka = slot_k[s, sa]
            rb = slot_r[s, sb]
            kb = slot_k[s, sb]
            la = rlen[s, ra]
            lb = rlen[s, rb]
            # type 0: 2-opt inside route ra (as Algorithm 8: p in 1..l-3, q in p+1..l-2, l >= 4)
            if ra == rb and la >= 4 and ka >= 1 and kb > ka and kb <= la - 2:
                x0 = nodes[s, ra, ka - 1]
                x1 = nodes[s, ra, ka]
                y0 = nodes[s, ra, kb]
                y1 = nodes[s, ra, kb + 1]
                key = dm[x0, y0] + dm[x1, y1] - dm[x0, x1] - dm[y0, y1]
                if key < -TOL and (best_code < 0 or key < best_key):
                    if eval_segments(nodes, s, prob_data, ra, 0, ka - 1, False, ra, ka, kb, True,
                                     ra, kb + 1, la - 1, False, ra, 1, 0, False):
                        best_key = key
                        best_code = idx * 4
            # type 1: relocate the customer at (ra, ka) to just after position kb of route rb
            if ka >= 1 and ka <= la - 2 and kb <= lb - 2 and not (ra == rb and (kb == ka or kb == ka - 1)):
                c = nodes[s, ra, ka]
                p = nodes[s, ra, ka - 1]
                n = nodes[s, ra, ka + 1]
                x = nodes[s, rb, kb]
                y = nodes[s, rb, kb + 1]
                key = dm[p, n] - dm[p, c] - dm[c, n] + dm[x, c] + dm[c, y] - dm[x, y]
                if ra != rb and la == 3:
                    key -= BIG
                if key < -TOL and (best_code < 0 or key < best_key):
                    if ra == rb:
                        if kb < ka:
                            ok = eval_segments(nodes, s, prob_data, ra, 0, kb, False, ra, ka, ka, False,
                                               ra, kb + 1, ka - 1, False, ra, ka + 1, la - 1, False)
                        else:
                            ok = eval_segments(nodes, s, prob_data, ra, 0, ka - 1, False, ra, ka + 1, kb, False,
                                               ra, ka, ka, False, ra, kb + 1, la - 1, False)
                    else:
                        ok = eval_segments(nodes, s, prob_data, ra, 0, ka - 1, False, ra, ka + 1, la - 1, False,
                                           ra, 1, 0, False, ra, 1, 0, False)
                        if ok:
                            ok = eval_segments(nodes, s, prob_data, rb, 0, kb, False, ra, ka, ka, False,
                                               rb, kb + 1, lb - 1, False, ra, 1, 0, False)
                    if ok:
                        best_key = key
                        best_code = idx * 4 + 1
            if ra < rb:
                # type 2: swap the customers (ra, ka) and (rb, kb)
                if ka >= 1 and ka <= la - 2 and kb >= 1 and kb <= lb - 2:
                    u = nodes[s, ra, ka]
                    pu = nodes[s, ra, ka - 1]
                    nu = nodes[s, ra, ka + 1]
                    v = nodes[s, rb, kb]
                    pv = nodes[s, rb, kb - 1]
                    nv = nodes[s, rb, kb + 1]
                    key = (dm[pu, v] + dm[v, nu] + dm[pv, u] + dm[u, nv]
                           - dm[pu, u] - dm[u, nu] - dm[pv, v] - dm[v, nv])
                    if key < -TOL and (best_code < 0 or key < best_key):
                        if eval_segments(nodes, s, prob_data, ra, 0, ka - 1, False, rb, kb, kb, False,
                                         ra, ka + 1, la - 1, False, ra, 1, 0, False):
                            if eval_segments(nodes, s, prob_data, rb, 0, kb - 1, False, ra, ka, ka, False,
                                             rb, kb + 1, lb - 1, False, ra, 1, 0, False):
                                best_key = key
                                best_code = idx * 4 + 2
                # type 3: Inter 2-opt* cutting after ka and kb (as improvement 4: cut in 1..l-3, l >= 4)
                if (use_2opt_star and la >= 4 and lb >= 4 and ka >= 1 and ka <= la - 3
                        and kb >= 1 and kb <= lb - 3):
                    u1 = nodes[s, ra, ka]
                    v1 = nodes[s, ra, ka + 1]
                    u2 = nodes[s, rb, kb]
                    v2 = nodes[s, rb, kb + 1]
                    key = dm[u1, v2] + dm[u2, v1] - dm[u1, v1] - dm[u2, v2]
                    if key < -TOL and (best_code < 0 or key < best_key):
                        if eval_segments(nodes, s, prob_data, ra, 0, ka, False, rb, kb + 1, lb - 1, False,
                                         ra, 1, 0, False, ra, 1, 0, False):
                            if eval_segments(nodes, s, prob_data, rb, 0, kb, False, ra, ka + 1, la - 1, False,
                                             ra, 1, 0, False, ra, 1, 0, False):
                                best_key = key
                                best_code = idx * 4 + 3
        return best_key, best_code

    @dev_fn
    def build_slots(rlen, nr, s, slot_r, slot_k):
        n_slots = 0
        for r in range(nr[s]):
            for kk in range(rlen[s, r]):
                slot_r[s, n_slots] = r
                slot_k[s, n_slots] = kk
                n_slots += 1
        return n_slots

    @dev_fn
    def write_segments(nodes, s, out, r0, a0, b0, v0, r1, a1, b1, v1, r2, a2, b2, v2, r3, a3, b3, v3):
        w = 0
        for g in range(4):
            if g == 0:
                r, a, b, v = r0, a0, b0, v0
            elif g == 1:
                r, a, b, v = r1, a1, b1, v1
            elif g == 2:
                r, a, b, v = r2, a2, b2, v2
            else:
                r, a, b, v = r3, a3, b3, v3
            for t in range(b - a + 1):
                out[s, w] = seg_node(nodes, s, r, a, b, v, t)
                w += 1
        return w

    @dev_fn
    def apply_move(nodes, rlen, s, slot_r, slot_k, n_slots, code, scratch_route, scratch_route2):
        idx = code // 4
        typ = code - idx * 4
        sa = idx // n_slots
        sb = idx - sa * n_slots
        ra = slot_r[s, sa]
        ka = slot_k[s, sa]
        rb = slot_r[s, sb]
        kb = slot_k[s, sb]
        la = rlen[s, ra]
        lb = rlen[s, rb]
        n1 = 0
        n2 = 0
        if typ == 0:
            n1 = write_segments(nodes, s, scratch_route, ra, 0, ka - 1, False, ra, ka, kb, True,
                                ra, kb + 1, la - 1, False, ra, 1, 0, False)
        elif typ == 1:
            if ra == rb:
                if kb < ka:
                    n1 = write_segments(nodes, s, scratch_route, ra, 0, kb, False, ra, ka, ka, False,
                                        ra, kb + 1, ka - 1, False, ra, ka + 1, la - 1, False)
                else:
                    n1 = write_segments(nodes, s, scratch_route, ra, 0, ka - 1, False, ra, ka + 1, kb, False,
                                        ra, ka, ka, False, ra, kb + 1, la - 1, False)
            else:
                n1 = write_segments(nodes, s, scratch_route, ra, 0, ka - 1, False, ra, ka + 1, la - 1, False,
                                    ra, 1, 0, False, ra, 1, 0, False)
                n2 = write_segments(nodes, s, scratch_route2, rb, 0, kb, False, ra, ka, ka, False,
                                    rb, kb + 1, lb - 1, False, ra, 1, 0, False)
        elif typ == 2:
            n1 = write_segments(nodes, s, scratch_route, ra, 0, ka - 1, False, rb, kb, kb, False,
                                ra, ka + 1, la - 1, False, ra, 1, 0, False)
            n2 = write_segments(nodes, s, scratch_route2, rb, 0, kb - 1, False, ra, ka, ka, False,
                                rb, kb + 1, lb - 1, False, ra, 1, 0, False)
        else:
            n1 = write_segments(nodes, s, scratch_route, ra, 0, ka, False, rb, kb + 1, lb - 1, False,
                                ra, 1, 0, False, ra, 1, 0, False)
            n2 = write_segments(nodes, s, scratch_route2, rb, 0, kb, False, ra, ka + 1, la - 1, False,
                                ra, 1, 0, False, ra, 1, 0, False)
        for kk in range(n1):
            nodes[s, ra, kk] = scratch_route[s, kk]
        for kk in range(n1, la):
            nodes[s, ra, kk] = 0
        rlen[s, ra] = n1
        if n2 > 0:
            for kk in range(n2):
                nodes[s, rb, kk] = scratch_route2[s, kk]
            for kk in range(n2, lb):
                nodes[s, rb, kk] = 0
            rlen[s, rb] = n2

    @dev_fn
    def best_individual(nr, dist, P):
        # Hierarchical best of the population (NV, then TD), used for --ls_scope best
        best_s = 0
        for s in range(1, P):
            if nr[s] < nr[best_s] or (nr[s] == nr[best_s] and dist[s] < dist[best_s]):
                best_s = s
        return best_s

    @dev_fn
    def block_ls_cpu_single(nodes, rlen, nr, s, slot_r, slot_k, scratch_route, scratch_route2,
                            prob_data, max_moves, use_2opt_star):
        # CPU backend: the same best-improvement steps with one thread
        moves = 0
        while True:
            compact_routes(nodes, rlen, nr, s)
            n_slots = build_slots(rlen, nr, s, slot_r, slot_k)
            key, code = scan_moves(nodes, rlen, s, slot_r, slot_k, n_slots, 0, 1, prob_data, use_2opt_star)
            if code < 0 or moves >= max_moves:
                break
            apply_move(nodes, rlen, s, slot_r, slot_k, n_slots, code, scratch_route, scratch_route2)
            moves += 1
        compact_routes(nodes, rlen, nr, s)

    k.update({"eval_segments": eval_segments, "scan_moves": scan_moves, "build_slots": build_slots,
              "apply_move": apply_move, "best_individual": best_individual,
              "block_ls_cpu_single": block_ls_cpu_single})

    k.update({"inter_2opt_star_first_improvement": inter_2opt_star_first_improvement})


def build_kernels(is_cuda, k_fn, k, kernels):
    """local_search_bi (default): block-parallel best improvement (only_best = 1: only the best individual).
    local_search / local_search_best_fi: first improvement (Algorithm 8), one thread per individual / only
    the best individual; launched on one thread in the CUDA case (ablation).
    """
    deep_local_search_single = k["deep_local_search_single"]
    compact_routes = k["compact_routes"]
    build_slots = k["build_slots"]
    scan_moves = k["scan_moves"]
    apply_move = k["apply_move"]
    best_individual = k["best_individual"]
    refresh_solution = k["refresh_solution"]
    block_ls_cpu_single = k["block_ls_cpu_single"]

    if is_cuda:
        @k_fn
        def local_search_bi_kernel(nodes, rlen, nr, dist, cost, slot_r, slot_k, scratch_route, scratch_route2,
                                   prob_data, max_moves, use_2opt_star, only_best):
            t = cuda.threadIdx.x
            sk = cuda.shared.array(NT, float64)
            scode = cuda.shared.array(NT, int64)
            sh = cuda.shared.array(3, int64)
            if t == 0:
                sh[2] = best_individual(nr, dist, nodes.shape[0]) if only_best else cuda.blockIdx.x
            cuda.syncthreads()
            s = sh[2]
            moves = 0
            while True:
                if t == 0:
                    compact_routes(nodes, rlen, nr, s)
                    sh[0] = build_slots(rlen, nr, s, slot_r, slot_k)
                cuda.syncthreads()
                n_slots = sh[0]
                key, code = scan_moves(nodes, rlen, s, slot_r, slot_k, n_slots, t, NT, prob_data, use_2opt_star)
                sk[t] = key
                scode[t] = code
                cuda.syncthreads()
                stride = NT // 2
                while stride > 0:
                    if t < stride:
                        c2 = scode[t + stride]
                        if c2 >= 0:
                            c1 = scode[t]
                            k2 = sk[t + stride]
                            if c1 < 0 or k2 < sk[t] or (k2 == sk[t] and c2 < c1):
                                sk[t] = k2
                                scode[t] = c2
                    cuda.syncthreads()
                    stride //= 2
                if t == 0:
                    if scode[0] >= 0 and moves < max_moves:
                        apply_move(nodes, rlen, s, slot_r, slot_k, n_slots, scode[0], scratch_route, scratch_route2)
                        sh[1] = 0
                    else:
                        sh[1] = 1
                cuda.syncthreads()
                if sh[1] == 1:
                    break
                moves += 1
            if t == 0:
                refresh_solution(nodes, rlen, nr, dist, cost, s, prob_data)
    else:
        @k_fn
        def local_search_bi_kernel(nodes, rlen, nr, dist, cost, slot_r, slot_k, scratch_route, scratch_route2,
                                   prob_data, max_moves, use_2opt_star, only_best):
            if only_best:
                s = best_individual(nr, dist, nodes.shape[0])
                block_ls_cpu_single(nodes, rlen, nr, s, slot_r, slot_k, scratch_route, scratch_route2,
                                    prob_data, max_moves, use_2opt_star)
                refresh_solution(nodes, rlen, nr, dist, cost, s, prob_data)
            else:
                for s in range(nodes.shape[0]):
                    block_ls_cpu_single(nodes, rlen, nr, s, slot_r, slot_k, scratch_route, scratch_route2,
                                        prob_data, max_moves, use_2opt_star)
                    refresh_solution(nodes, rlen, nr, dist, cost, s, prob_data)
    kernels["local_search_bi"] = local_search_bi_kernel
    if is_cuda:
        @k_fn
        def local_search_kernel(nodes, rlen, nr, dist, cost, scratch_route, scratch_route2, prob_data, max_moves):
            s = cuda.grid(1)
            if s < nodes.shape[0]:
                deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_route2, prob_data, max_moves)
    else:
        @k_fn
        def local_search_kernel(nodes, rlen, nr, dist, cost, scratch_route, scratch_route2, prob_data, max_moves):
            for s in range(nodes.shape[0]):
                deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_route2, prob_data, max_moves)

    @k_fn
    def local_search_best_fi_kernel(nodes, rlen, nr, dist, cost, scratch_route, scratch_route2, prob_data, max_moves):
        # --ls_strategy first --ls_scope best: Algorithm 8 only for the best individual of the population
        s = best_individual(nr, dist, nodes.shape[0])
        deep_local_search_single(nodes, rlen, nr, dist, cost, s, scratch_route, scratch_route2, prob_data, max_moves)

    kernels["local_search"] = local_search_kernel
    kernels["local_search_best_fi"] = local_search_best_fi_kernel
