"""Improvement 1 (docs/improved.md section 1), CUDA only: one block of NTU threads per individual.

With one thread per individual a population of 30 keeps one block of 30 threads busy on a GPU with 40 SMs, and each
thread runs the scans of Algorithms 4-10 alone. Here the update kernel gives every individual a block:
  - thread 0 runs the random part (WOA route updates, SHO tournament / crossover / mutation, SA acceptance), which
    must consume the random numbers in the same order as the CPU code;
  - the whole block runs the scans of the repair (Algorithm 10): the best insertion position of a customer
    (reinsert, every (route, position) evaluated by a different thread, block minimum) and the intra-route 2-opt
    (every pair (p, q) by a different thread, smallest improving pair = first improvement);
  - the block copies the result with one thread per position.
Candidate routes are evaluated without being written: eval_virtual walks route r as if c were inserted at p (or as
if positions p..q were reversed) with the same operations, in the same order, as eval_route on the written route,
so the floating-point results are identical. Ties are broken by the smallest (route, position) / pair index, the
order of the sequential loops. The solution is therefore exactly the one of the CPU backend for the same seed.

Prefix tables (opt bit 2, set by the host when every delivery, pickup and the capacity are integers): before a
reinsertion every route is walked once (one thread per route) and the time, distance and load after each position are
kept in shared memory. Inserting c at p leaves positions 0..p-1 unchanged, so the evaluation starts from the state
after p - 1 (the same floating-point values the full walk would produce) and only walks c and the rest of the route.
The start load gains delivery[c], so the loads of the prefix are shifted by delivery[c]: exact for integer demands,
hence the flag. The intra-route 2-opt of the repair uses the same table of its route (the prefix before p is not
changed by reversing p..q). Without the flag (or when the routes do not fit the tables) eval_virtual is used.
"""

from __future__ import annotations

import math

try:
    from numba import cuda, float64, int32, int64   # module level: the CUDA simulator patches module globals
except ImportError:  # pragma: no cover - optional dependency
    cuda = None

NTU = 64           # threads per block of the update kernel
MAXS = 320         # prefix tables: positions of all routes (N + 2 NV <= 100 + 2 * 105)
RM = 106           # prefix tables: routes
# pf (float64): time, distance, load after each position, max / min load over positions 1..k, start load and
# distance of each route; pfi (int32): time windows met up to position k, route offset, invalid route, feasible route
T_OFF, D_OFF, LD_OFF, MX_OFF, MN_OFF = 0, MAXS, 2 * MAXS, 3 * MAXS, 4 * MAXS
L0_OFF, RD_OFF = 5 * MAXS, 5 * MAXS + RM
PF_SIZE = 5 * MAXS + 2 * RM
OK_OFF, OFF_OFF, BAD_OFF, FEAS_OFF = 0, MAXS, MAXS + RM, MAXS + 2 * RM
PFI_SIZE = MAXS + 3 * RM


def build(dev_fn, k):
    """Device functions of the block-parallel update (CUDA only)."""
    eval_route = k["eval_route"]
    insert_at = k["insert_at"]
    open_route = k["open_route"]
    remove_at = k["remove_at"]
    compact_routes = k["compact_routes"]
    route_capacity_ok = k["route_capacity_ok"]
    first_late_position = k["first_late_position"]
    eval_solution = k["eval_solution"]

    @dev_fn
    def vnode(nodes, s, r, i, mode, p, q, c):
        # Node at position i of route r after inserting c at p (mode 0) or reversing positions p..q (mode 1)
        if mode == 0:
            if i < p:
                return nodes[s, r, i]
            if i == p:
                return c
            return nodes[s, r, i - 1]
        if i >= p and i <= q:
            return nodes[s, r, p + q - i]
        return nodes[s, r, i]

    @dev_fn
    def eval_virtual(nodes, s, r, length, mode, p, q, c, prob_data):
        # eval_route of the modified route (length = its length), same operations in the same order
        depot = int(prob_data[0])
        capacity = prob_data[1]
        start_time = prob_data[2]
        delivery = prob_data[5]
        pickup = prob_data[6]
        start_tw = prob_data[7]
        end_tw = prob_data[8]
        service = prob_data[9]
        dist_matrix = prob_data[10]
        time_matrix = prob_data[11]
        if length < 2 or vnode(nodes, s, r, 0, mode, p, q, c) != depot or \
                vnode(nodes, s, r, length - 1, mode, p, q, c) != depot:
            return False, 0.0
        if length == 2:
            return True, 0.0
        for i in range(1, length - 1):
            if vnode(nodes, s, r, i, mode, p, q, c) == depot:
                return False, 0.0
        load = 0.0
        for i in range(1, length - 1):
            load += delivery[vnode(nodes, s, r, i, mode, p, q, c)]
        if load > capacity + 1e-9:
            return False, 0.0
        distance_val = 0.0
        time_val = start_time
        prev = vnode(nodes, s, r, 0, mode, p, q, c)
        for i in range(1, length):
            node = vnode(nodes, s, r, i, mode, p, q, c)
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
    def arrival_virtual(nodes, s, r, pos, p, c, prob_data):
        # arrival_at(route with c inserted at p, pos)
        start_tw = prob_data[7]
        service = prob_data[9]
        time_matrix = prob_data[11]
        t = prob_data[2]
        for i in range(1, pos + 1):
            node = vnode(nodes, s, r, i, 0, p, 0, c)
            t += time_matrix[vnode(nodes, s, r, i - 1, 0, p, 0, c), node]
            if t < start_tw[node]:
                t = start_tw[node]
            if i == pos:
                return t
            t += service[node]
        return t

    @dev_fn
    def prefix_route(nodes, rlen, s, r, base, prob_data, pf, pfi):
        # Prefix tables of route r at positions base .. base + rlen - 1 (one thread)
        depot = int(prob_data[0])
        capacity = prob_data[1]
        delivery = prob_data[5]
        pickup = prob_data[6]
        start_tw = prob_data[7]
        end_tw = prob_data[8]
        service = prob_data[9]
        dist_matrix = prob_data[10]
        time_matrix = prob_data[11]
        l = rlen[s, r]
        pfi[OFF_OFF + r] = base
        bad = l < 2 or nodes[s, r, 0] != depot or nodes[s, r, l - 1] != depot
        if not bad:
            for i in range(1, l - 1):
                if nodes[s, r, i] == depot:
                    bad = True
                    break
        pfi[BAD_OFF + r] = 1 if bad else 0
        if bad:
            pfi[FEAS_OFF + r] = 0
            pf[RD_OFF + r] = 0.0
        else:
            load = 0.0
            for i in range(1, l - 1):
                load += delivery[nodes[s, r, i]]
            pf[L0_OFF + r] = load
            time_val = prob_data[2]
            dist_val = 0.0
            ok = 1
            mx = -1e300
            mn = 1e300
            prev = nodes[s, r, 0]
            pf[T_OFF + base] = time_val
            pf[D_OFF + base] = 0.0
            pf[LD_OFF + base] = load
            pf[MX_OFF + base] = mx
            pf[MN_OFF + base] = mn
            pfi[OK_OFF + base] = 1
            for kk in range(1, l):
                node = nodes[s, r, kk]
                load = load - delivery[node] + pickup[node]
                if load > mx:
                    mx = load
                if load < mn:
                    mn = load
                if ok == 1:
                    time_val += time_matrix[prev, node]
                    if time_val > end_tw[node] + 1e-9:
                        ok = 0
                    else:
                        if time_val < start_tw[node]:
                            time_val = start_tw[node]
                        time_val += service[node]
                        dist_val += dist_matrix[prev, node]
                pf[T_OFF + base + kk] = time_val
                pf[D_OFF + base + kk] = dist_val
                pf[LD_OFF + base + kk] = load
                pf[MX_OFF + base + kk] = mx
                pf[MN_OFF + base + kk] = mn
                pfi[OK_OFF + base + kk] = ok
                prev = node
            feas = (pf[L0_OFF + r] <= capacity + 1e-9 and ok == 1 and mx <= capacity + 1e-9
                    and mn >= -1e-9)
            pfi[FEAS_OFF + r] = 1 if feas else 0
            pf[RD_OFF + r] = dist_val if feas else 0.0     # eval_route returns 0.0 when infeasible

    @dev_fn
    def build_prefix(nodes, rlen, nr, s, prob_data, t, nt, pf, pfi):
        # Prefix tables of every route (route r by thread r mod nt); False (same on every thread) when they do not fit
        n_r = nr[s]
        total = 0
        for r in range(n_r):
            total += rlen[s, r]
        if n_r > RM or total > MAXS:
            return False
        base = 0
        for r in range(n_r):
            if r % nt == t:
                prefix_route(nodes, rlen, s, r, base, prob_data, pf, pfi)
            base += rlen[s, r]
        cuda.syncthreads()
        return True

    @dev_fn
    def eval_insert_pref(nodes, s, r, l, p, c, prob_data, pf, pfi):
        # eval_virtual(route r with c inserted at p) from the state after position p - 1
        capacity = prob_data[1]
        delivery = prob_data[5]
        pickup = prob_data[6]
        start_tw = prob_data[7]
        end_tw = prob_data[8]
        service = prob_data[9]
        dist_matrix = prob_data[10]
        time_matrix = prob_data[11]
        if pfi[BAD_OFF + r] == 1:
            return False, 0.0
        base = pfi[OFF_OFF + r]
        dc = delivery[c]
        if pf[L0_OFF + r] + dc > capacity + 1e-9:
            return False, 0.0
        kp = base + p - 1
        if p >= 2:
            if pfi[OK_OFF + kp] == 0:
                return False, 0.0
            if pf[MX_OFF + kp] + dc > capacity + 1e-9 or pf[MN_OFF + kp] + dc < -1e-9:
                return False, 0.0
        time_val = pf[T_OFF + kp]
        distance_val = pf[D_OFF + kp]
        load = pf[LD_OFF + kp] + dc
        prev = nodes[s, r, p - 1]
        for i in range(p - 1, l):
            node = c if i == p - 1 else nodes[s, r, i]
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
    def eval_reverse_pref(nodes, s, r, l, p, q, prob_data, pf, pfi):
        # eval_virtual(route r with positions p..q reversed) from the state after position p - 1. The start load is
        # the sum of the same deliveries in another order: exact for integer demands (opt bit 2)
        capacity = prob_data[1]
        delivery = prob_data[5]
        pickup = prob_data[6]
        start_tw = prob_data[7]
        end_tw = prob_data[8]
        service = prob_data[9]
        dist_matrix = prob_data[10]
        time_matrix = prob_data[11]
        if pfi[BAD_OFF + r] == 1 or pf[L0_OFF + r] > capacity + 1e-9:
            return False, 0.0
        kp = pfi[OFF_OFF + r] + p - 1
        if p >= 2:
            if pfi[OK_OFF + kp] == 0 or pf[MX_OFF + kp] > capacity + 1e-9 or pf[MN_OFF + kp] < -1e-9:
                return False, 0.0
        time_val = pf[T_OFF + kp]
        distance_val = pf[D_OFF + kp]
        load = pf[LD_OFF + kp]
        prev = nodes[s, r, p - 1]
        for i in range(p, l):
            node = nodes[s, r, p + q - i] if i <= q else nodes[s, r, i]
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
    def arrival_pref(nodes, s, r, p, c, prob_data, pf, pfi):
        # arrival_at(route r with c inserted at p, p) for a feasible insertion
        t = pf[T_OFF + pfi[OFF_OFF + r] + p - 1] + prob_data[11][nodes[s, r, p - 1], c]
        if t < prob_data[7][c]:
            t = prob_data[7][c]
        return t

    @dev_fn
    def block_min(sv, si, t, nt, val, code):
        # Block minimum of (val, code) with code >= 0 (smallest code on ties); every thread gets the result
        sv[t] = val
        si[t] = code
        cuda.syncthreads()
        stride = nt // 2
        while stride > 0:
            if t < stride:
                c2 = si[t + stride]
                if c2 >= 0:
                    c1 = si[t]
                    v2 = sv[t + stride]
                    if c1 < 0 or v2 < sv[t] or (v2 == sv[t] and c2 < c1):
                        sv[t] = v2
                        si[t] = c2
            cuda.syncthreads()
            stride //= 2
        rv = sv[0]
        rc = si[0]
        cuda.syncthreads()
        return rv, rc

    @dev_fn
    def reinsert_block(nodes, rlen, nr, s, c, exclude_r, by_arrival, prob_data, t, nt, sv, si, pf, pfi, opt):
        # reinsert_customer (Algorithm 10) with the (route, position) pairs spread over the block
        depot = int(prob_data[0])
        L = nodes.shape[2]
        best_val = 1e18
        best_code = -1
        cnt = 0
        pref = False
        if opt & 2:
            pref = build_prefix(nodes, rlen, nr, s, prob_data, t, nt, pf, pfi)
        for r in range(nr[s]):
            if r == exclude_r:
                continue
            l = rlen[s, r]
            if l + 1 > L:
                continue
            have_old = False
            old_d = 0.0
            for p in range(1, l):
                if cnt % nt == t:
                    if pref:
                        ok_n, new_d = eval_insert_pref(nodes, s, r, l, p, c, prob_data, pf, pfi)
                    else:
                        ok_n, new_d = eval_virtual(nodes, s, r, l + 1, 0, p, 0, c, prob_data)
                    if ok_n:
                        if by_arrival:
                            if pref:
                                val = arrival_pref(nodes, s, r, p, c, prob_data, pf, pfi)
                            else:
                                val = arrival_virtual(nodes, s, r, p, p, c, prob_data)
                        else:
                            if not have_old:
                                if pref:
                                    old_d = pf[RD_OFF + r]
                                else:
                                    ok_o, old_d = eval_route(nodes[s, r, :l], l, prob_data)
                                have_old = True
                            val = new_d - old_d
                        if val < best_val:
                            best_val = val
                            best_code = r * L + p
                cnt += 1
        v, code = block_min(sv, si, t, nt, best_val, best_code)
        if t == 0:
            if code >= 0:
                insert_at(nodes, rlen, s, code // L, code - (code // L) * L, c)
            else:
                open_route(nodes, rlen, nr, s, c, depot)
        cuda.syncthreads()

    @dev_fn
    def two_opt_block(nodes, rlen, s, r, prob_data, tol, t, nt, sv, si, pf, pfi, opt):
        # two_opt_route_first_improvement: the pairs (p, q) in loop order are scanned nt at a time and the smallest
        # improving one is applied, then the scan restarts
        l = rlen[s, r]
        if l < 4:
            return
        ok0, best_d = eval_route(nodes[s, r, :l], l, prob_data)
        n_pairs = (l - 3) * (l - 2) // 2
        pref = (opt & 2) != 0
        while True:
            if pref:                            # prefix table of route r, rebuilt after every applied reversal
                if t == 0:
                    prefix_route(nodes, rlen, s, r, 0, prob_data, pf, pfi)
                cuda.syncthreads()
            win = -1
            base = 0
            while base < n_pairs:
                code = -1
                idx = base + t
                if idx < n_pairs:
                    p = 1
                    rem = idx
                    cnt_p = l - 3
                    while rem >= cnt_p:
                        rem -= cnt_p
                        p += 1
                        cnt_p -= 1
                    if pref:
                        ok, d = eval_reverse_pref(nodes, s, r, l, p, p + 1 + rem, prob_data, pf, pfi)
                    else:
                        ok, d = eval_virtual(nodes, s, r, l, 1, p, p + 1 + rem, 0, prob_data)
                    if ok and d < best_d - tol:
                        code = idx
                v, win = block_min(sv, si, t, nt, 0.0, code)
                if win >= 0:
                    break
                base += nt
            if win < 0:
                return
            p = 1
            rem = win
            cnt_p = l - 3
            while rem >= cnt_p:
                rem -= cnt_p
                p += 1
                cnt_p -= 1
            q = p + 1 + rem
            if pref:
                ok, best_d = eval_reverse_pref(nodes, s, r, l, p, q, prob_data, pf, pfi)
            else:
                ok, best_d = eval_virtual(nodes, s, r, l, 1, p, q, 0, prob_data)
            cuda.syncthreads()                  # every thread has read the route before thread 0 reverses it
            if t == 0:
                a = p
                b = q
                while a < b:
                    tmp = nodes[s, r, a]
                    nodes[s, r, a] = nodes[s, r, b]
                    nodes[s, r, b] = tmp
                    a += 1
                    b -= 1
            cuda.syncthreads()

    @dev_fn
    def repair_block(nodes, rlen, nr, s, prob_data, t, nt, sv, si, pf, pfi, opt, do_2opt):
        # repair_single (Algorithm 10, steps 2-4). Every thread evaluates the loop conditions on the same data;
        # only thread 0 writes, between two barriers.
        delivery = prob_data[5]
        pickup = prob_data[6]
        n0 = nr[s]
        for r in range(n0):
            attempts = 0
            while attempts < rlen[s, r] - 2 and not route_capacity_ok(nodes[s, r, :rlen[s, r]], rlen[s, r], prob_data):
                attempts += 1
                best_k = 1
                best_v = -1.0
                for kk in range(1, rlen[s, r] - 1):
                    c = nodes[s, r, kk]
                    v = abs(delivery[c] - pickup[c])
                    if v > best_v:
                        best_v = v
                        best_k = kk
                c_sel = nodes[s, r, best_k]
                cuda.syncthreads()
                if t == 0:
                    remove_at(nodes, rlen, s, r, best_k)
                cuda.syncthreads()
                reinsert_block(nodes, rlen, nr, s, c_sel, r, False, prob_data, t, nt, sv, si, pf, pfi, opt)
        cuda.syncthreads()
        if t == 0:
            compact_routes(nodes, rlen, nr, s)
        cuda.syncthreads()
        n0 = nr[s]
        for r in range(n0):
            attempts = 0
            while attempts < rlen[s, r] - 2:
                late = first_late_position(nodes[s, r, :rlen[s, r]], rlen[s, r], prob_data)
                if late < 0:
                    break
                attempts += 1
                c_late = nodes[s, r, late]
                cuda.syncthreads()
                if t == 0:
                    remove_at(nodes, rlen, s, r, late)
                cuda.syncthreads()
                reinsert_block(nodes, rlen, nr, s, c_late, -1, True, prob_data, t, nt, sv, si, pf, pfi, opt)
        cuda.syncthreads()
        if t == 0:
            compact_routes(nodes, rlen, nr, s)
        cuda.syncthreads()
        if do_2opt:                     # 0 only in the profiling probe
            for r in range(nr[s]):
                two_opt_block(nodes, rlen, s, r, prob_data, 1e-6, t, nt, sv, si, pf, pfi, opt)

    @dev_fn
    def copy_block(src_nodes, src_rlen, src_nr, src_dist, src_cost, src_s,
                   dst_nodes, dst_rlen, dst_nr, dst_dist, dst_cost, dst_s, t, nt):
        # copy_solution with one thread per position
        n_r = src_nr[src_s]
        for r in range(n_r):
            l = src_rlen[src_s, r]
            for i in range(t, l, nt):
                dst_nodes[dst_s, r, i] = src_nodes[src_s, r, i]
        cuda.syncthreads()
        if t == 0:
            for r in range(n_r):
                dst_rlen[dst_s, r] = src_rlen[src_s, r]
            dst_nr[dst_s] = n_r
            dst_dist[dst_s] = src_dist[src_s]
            dst_cost[dst_s] = src_cost[src_s]
        cuda.syncthreads()

    @dev_fn
    def route_elimination_block(nodes, rlen, nr, dist, cost, s, scratch_unrouted, scratch_flags, prob_data, passes,
                                t, nt, sv, si, pf, pfi, opt):
        # route_elimination_single (improvement 2) with the insertion positions spread over the block
        L = nodes.shape[2]
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
            if t == 0:
                for i in range(num_ejected):
                    scratch_unrouted[s, i] = nodes[s, min_r, i + 1]
            all_inserted = True
            for e_idx in range(num_ejected):
                c = nodes[s, min_r, e_idx + 1]          # route min_r is not changed before it is removed
                best_delta = 1e12
                best_code = -1
                cnt = 0
                pref = False
                if opt & 2:
                    pref = build_prefix(nodes, rlen, nr, s, prob_data, t, nt, pf, pfi)
                for r in range(num_r):
                    if r == min_r:
                        continue
                    l = rlen[s, r]
                    have_old = False
                    old_d = 0.0
                    for pos in range(1, l):
                        if cnt % nt == t:
                            if pref:
                                ok_new, new_d = eval_insert_pref(nodes, s, r, l, pos, c, prob_data, pf, pfi)
                            else:
                                ok_new, new_d = eval_virtual(nodes, s, r, l + 1, 0, pos, 0, c, prob_data)
                            if ok_new:
                                if not have_old:
                                    if pref:
                                        old_d = pf[RD_OFF + r]
                                    else:
                                        ok_old, old_d = eval_route(nodes[s, r, :l], l, prob_data)
                                    have_old = True
                                delta = new_d - old_d
                                if delta < best_delta:
                                    best_delta = delta
                                    best_code = r * L + pos
                        cnt += 1
                v, code = block_min(sv, si, t, nt, best_delta, best_code)
                if code >= 0:
                    if t == 0:
                        tr = code // L
                        insert_at(nodes, rlen, s, tr, code - tr * L, c)
                        scratch_flags[s, e_idx] = tr
                    cuda.syncthreads()
                else:
                    all_inserted = False
                    if t == 0:
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
                    cuda.syncthreads()
                    break
            if not all_inserted:
                break
            if t == 0:
                for r in range(min_r, num_r - 1):
                    rlen[s, r] = rlen[s, r + 1]
                    for i in range(rlen[s, r]):
                        nodes[s, r, i] = nodes[s, r + 1, i]
                rlen[s, num_r - 1] = 0
                nr[s] = num_r - 1
                ok, n_act, t_dist, t_cost = eval_solution(nodes, rlen, nr, s, prob_data)
                dist[s] = t_dist
                cost[s] = t_cost
            cuda.syncthreads()

    rcrs_score = k["rcrs_score"]
    pick_seed = k["pick_seed"]
    randint = k["randint"]
    shuffle_ints = k["shuffle_ints"]

    @dev_fn
    def grasp_block(nodes, rlen, nr, dist, cost, s, scratch_unrouted, scratch_flags, scratch_scores,
                    prob_data, alpha, rng_states, t, nt, sv, si, sh, pf, pfi, opt):
        # construct_rcrs_grasp (improvement 2): the best insertion of every unrouted customer is computed by a
        # different thread (customer i by thread i mod nt); thread 0 draws the random numbers and inserts
        customer_num = int(prob_data[14])
        depot = int(prob_data[0])
        dispatch_cost = prob_data[3]
        unit_cost = prob_data[4]
        if t == 0:
            for i in range(customer_num):
                scratch_unrouted[s, i] = i + 1
            shuffle_ints(scratch_unrouted[s], customer_num, rng_states, s)
            nr[s] = 0
            dist[s] = 0.0
            cost[s] = 0.0
            sh[0] = customer_num
        cuda.syncthreads()
        while True:
            unrouted_count = sh[0]
            if unrouted_count <= 0:
                break
            num_r = nr[s]
            if num_r == 0:
                cuda.syncthreads()
                if t == 0:
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
                    for kk in range(pick, unrouted_count - 1):
                        scratch_unrouted[s, kk] = scratch_unrouted[s, kk + 1]
                    sh[0] = unrouted_count - 1
                cuda.syncthreads()
                continue

            my_best = 1e12
            my_worst = -1e12
            pref = False
            if opt & 2:
                pref = build_prefix(nodes, rlen, nr, s, prob_data, t, nt, pf, pfi)
            for i in range(t, unrouted_count, nt):
                c = scratch_unrouted[s, i]
                best_c_score = 1e12
                best_r = -1
                best_p = -1
                for r in range(num_r):
                    l = rlen[s, r]
                    for p in range(1, l):
                        if pref:
                            ok, d = eval_insert_pref(nodes, s, r, l, p, c, prob_data, pf, pfi)
                        else:
                            ok, d = eval_virtual(nodes, s, r, l + 1, 0, p, 0, c, prob_data)
                        if ok:
                            sc = rcrs_score(nodes[s, r, :l], l, c, p, prob_data)
                            if sc < best_c_score:
                                best_c_score = sc
                                best_r = r
                                best_p = p
                scratch_scores[s, i] = best_c_score
                scratch_flags[s, i] = best_r * 10000 + best_p
                if best_r != -1:
                    if best_c_score < my_best:
                        my_best = best_c_score
                    if best_c_score > my_worst:
                        my_worst = best_c_score
            best_score_global, z = block_min(sv, si, t, nt, my_best, 0)
            neg_worst, z = block_min(sv, si, t, nt, -my_worst, 0)
            worst_score_global = -neg_worst

            if t == 0:
                new_route = False
                if best_score_global >= 1e11:
                    new_route = True
                else:
                    threshold = best_score_global + alpha * (worst_score_global - best_score_global)
                    rcl_count = 0
                    for i in range(unrouted_count):
                        if scratch_scores[s, i] <= threshold + 1e-6 and scratch_flags[s, i] >= 0:
                            rcl_count += 1
                    if rcl_count == 0:
                        new_route = True
                    else:
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
                        for kk in range(l, chosen_p, -1):
                            nodes[s, chosen_r, kk] = nodes[s, chosen_r, kk - 1]
                        nodes[s, chosen_r, chosen_p] = chosen_c
                        rlen[s, chosen_r] = l + 1
                        for kk in range(actual_i, unrouted_count - 1):
                            scratch_unrouted[s, kk] = scratch_unrouted[s, kk + 1]
                if new_route:
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
                    for kk in range(pick, unrouted_count - 1):
                        scratch_unrouted[s, kk] = scratch_unrouted[s, kk + 1]
                sh[0] = unrouted_count - 1
            cuda.syncthreads()

        if t == 0:
            ok, n_act, t_dist, t_cost = eval_solution(nodes, rlen, nr, s, prob_data)
            dist[s] = t_dist
            cost[s] = t_cost
        cuda.syncthreads()

    rand_u01 = k["rand_u01"]
    refresh_solution = k["refresh_solution"]
    copy_solution = k["copy_solution"]
    apply_intra_op = k["apply_intra_op"]
    apply_inter_op = k["apply_inter_op"]
    pick_two_routes = k["pick_two_routes"]

    @dev_fn
    def init_sa_block(nodes, rlen, nr, dist, cost, s,
                      cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                      best_nodes, best_rlen, best_nr, best_dist, best_cost,
                      scratch_unrouted, scratch_scores, prob_data, rng_states, t0, alpha, t_min, iter_max,
                      t, nt, sv, si, sh, pf, pfi, opt):
        # init_paper_sa_single (Algorithm 2): insertions and repair block-parallel, SA moves on thread 0
        customer_num = int(prob_data[14])
        due = prob_data[8]
        if t == 0:
            for c in range(1, customer_num + 1):
                scratch_scores[s, c] = due[c] + (rand_u01(rng_states, s) * 20.0 - 10.0)
            for i in range(customer_num):
                c = i + 1
                j = i - 1
                while j >= 0 and scratch_scores[s, scratch_unrouted[s, j]] > scratch_scores[s, c]:
                    scratch_unrouted[s, j + 1] = scratch_unrouted[s, j]
                    j -= 1
                scratch_unrouted[s, j + 1] = c
            nr[s] = 0
        cuda.syncthreads()
        for i in range(customer_num):
            reinsert_block(nodes, rlen, nr, s, scratch_unrouted[s, i], -1, False, prob_data, t, nt, sv, si,
                           pf, pfi, opt)
        repair_block(nodes, rlen, nr, s, prob_data, t, nt, sv, si, pf, pfi, opt, 1)
        if t == 0:
            refresh_solution(nodes, rlen, nr, dist, cost, s, prob_data)
        cuda.syncthreads()
        copy_block(nodes, rlen, nr, dist, cost, s, best_nodes, best_rlen, best_nr, best_dist, best_cost, s, t, nt)
        temp = t0
        while temp > t_min:
            for _ in range(iter_max):
                copy_block(nodes, rlen, nr, dist, cost, s, cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                           t, nt)
                if t == 0:
                    if rand_u01(rng_states, s) < 0.5 or cand_nr[s] < 2:
                        if cand_nr[s] > 0:
                            r = randint(rng_states, s, 0, cand_nr[s] - 1)
                            op = randint(rng_states, s, 0, 2)
                            apply_intra_op(cand_nodes, cand_rlen, s, r, op, rng_states)
                    else:
                        op = randint(rng_states, s, 0, 1)
                        r1, r2 = pick_two_routes(cand_nr, s, rng_states)
                        apply_inter_op(cand_nodes, cand_rlen, s, r1, r2, op, rng_states)
                    acc = 0
                    if refresh_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, prob_data):
                        cf = cost[s]
                        delta = cand_cost[s] - cf
                        if delta < 0.0 or rand_u01(rng_states, s) < math.exp(-delta / (temp * abs(cf) + 1e-6)):
                            acc = 1
                    sh[1] = acc
                cuda.syncthreads()
                if sh[1] == 1:
                    copy_block(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, nodes, rlen, nr, dist, cost, s,
                               t, nt)
                    if t == 0:
                        sh[2] = 1 if cost[s] < best_cost[s] else 0
                    cuda.syncthreads()
                    if sh[2] == 1:
                        copy_block(nodes, rlen, nr, dist, cost, s, best_nodes, best_rlen, best_nr, best_dist,
                                   best_cost, s, t, nt)
            temp = alpha * temp
        copy_block(best_nodes, best_rlen, best_nr, best_dist, best_cost, s, nodes, rlen, nr, dist, cost, s, t, nt)

    k.update({"eval_virtual": eval_virtual, "block_min": block_min, "reinsert_block": reinsert_block,
              "grasp_block": grasp_block, "init_sa_block": init_sa_block,
              "two_opt_block": two_opt_block, "repair_block": repair_block, "copy_block": copy_block,
              "route_elimination_block": route_elimination_block})


def build_kernels(k_fn, k, kernels):
    """update_population as one block per individual (replaces the one-thread-per-individual CUDA kernel)."""
    rand_u01 = k["rand_u01"]
    woa_generate = k["woa_generate"]
    sho_generate = k["sho_generate"]
    refresh_solution = k["refresh_solution"]
    eval_solution = k["eval_solution"]
    repair_block = k["repair_block"]
    copy_block = k["copy_block"]

    @k_fn
    def update_population_block_kernel(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                                       next_nodes, next_rlen, next_nr, next_dist, next_cost,
                                       cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                       ibest_nodes, ibest_rlen, ibest_nr,
                                       scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                       prob_data, a_param, p_hybrid, mutation_prob, iteration, max_iter,
                                       rng_states, island_size, opt):
        # Algorithm 1, lines 11-31 for the individual of this block (same steps as update_individual)
        s = cuda.blockIdx.x
        t = cuda.threadIdx.x
        sv = cuda.shared.array(NTU, float64)
        si = cuda.shared.array(NTU, int64)
        sh = cuda.shared.array(4, int64)
        pf = cuda.shared.array(PF_SIZE, float64)
        pfi = cuda.shared.array(PFI_SIZE, int32)
        island_id = s // island_size
        if t == 0:
            # Algorithm 3: SHO (r < pHybrid) or WOA; next doubles as WOA's rollback copy
            if rand_u01(rng_states, s) < p_hybrid:
                sho_generate(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                             ibest_nodes, ibest_rlen, ibest_nr, island_id, island_size,
                             cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                             scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                             mutation_prob, prob_data, rng_states)
            else:
                woa_generate(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                             ibest_nodes, ibest_rlen, ibest_nr, island_id,
                             cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                             next_nodes, next_rlen, next_nr, next_dist, next_cost,
                             scratch_unrouted, scratch_flags, a_param, prob_data, rng_states)
            sh[0] = 1 if refresh_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, prob_data) else 0
        cuda.syncthreads()
        if sh[0] == 0:
            # Algorithm 10 on the infeasible candidate (WOA and SHO alike), block-parallel
            repair_block(cand_nodes, cand_rlen, cand_nr, s, prob_data, t, NTU, sv, si, pf, pfi, opt, 1)
            if t == 0:
                sh[0] = 1 if refresh_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                                              prob_data) else 0
            cuda.syncthreads()
        if t == 0:
            # Lines 12-27: infeasible -> keep w; nf <= cf accepted, otherwise SA (Metropolis) acceptance
            accepted = False
            if sh[0] == 1:
                ok, act_r, t_d, t_c = eval_solution(cand_nodes, cand_rlen, cand_nr, s, prob_data)
                if ok:
                    nf = t_c
                    cf = cur_cost[s]
                    if nf <= cf:
                        accepted = True
                    else:
                        temp = 1.0 - float(iteration) / float(max_iter) if max_iter > 0 else 0.0
                        prob = math.exp(-(nf - cf) / (1e-6 + temp * abs(cf)))
                        if rand_u01(rng_states, s) < prob:
                            accepted = True
            sh[1] = 1 if accepted else 0
        cuda.syncthreads()
        if sh[1] == 1:
            copy_block(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                       next_nodes, next_rlen, next_nr, next_dist, next_cost, s, t, NTU)
        else:
            copy_block(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                       next_nodes, next_rlen, next_nr, next_dist, next_cost, s, t, NTU)

    route_elimination_block = k["route_elimination_block"]

    @k_fn
    def route_elimination_block_kernel(nodes, rlen, nr, dist, cost, scratch_unrouted, scratch_flags, prob_data,
                                       passes, opt):
        s = cuda.blockIdx.x
        t = cuda.threadIdx.x
        sv = cuda.shared.array(NTU, float64)
        si = cuda.shared.array(NTU, int64)
        pf = cuda.shared.array(PF_SIZE, float64)
        pfi = cuda.shared.array(PFI_SIZE, int32)
        route_elimination_block(nodes, rlen, nr, dist, cost, s, scratch_unrouted, scratch_flags, prob_data, passes,
                                t, NTU, sv, si, pf, pfi, opt)

    grasp_block = k["grasp_block"]
    init_sa_block = k["init_sa_block"]
    sa_warmup_single = k["sa_warmup_single"]
    deep_local_search_single = k["deep_local_search_single"]

    @k_fn
    def init_population_block_kernel(nodes, rlen, nr, dist, cost,
                                     cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                     bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                                     scratch_route, scratch_route2, scratch_unrouted, scratch_flags, scratch_scores,
                                     prob_data, init_mode, alpha_lo, alpha_hi, sa_iters, re_passes,
                                     t0, alpha_sa, t_min, iter_max, rng_states, init_fi_ls, opt):
        # init_individual for the individual of this block
        s = cuda.blockIdx.x
        t = cuda.threadIdx.x
        sv = cuda.shared.array(NTU, float64)
        si = cuda.shared.array(NTU, int64)
        sh = cuda.shared.array(4, int64)
        sa = cuda.shared.array(1, float64)
        pf = cuda.shared.array(PF_SIZE, float64)
        pfi = cuda.shared.array(PFI_SIZE, int32)
        mode = init_mode
        if mode == 2:                       # --init mixed: even individuals Algorithm 2, odd ones RCRS-GRASP
            mode = 1 if s % 2 == 0 else 0
        if mode == 1:
            init_sa_block(nodes, rlen, nr, dist, cost, s,
                          cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                          bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                          scratch_unrouted, scratch_scores, prob_data, rng_states, t0, alpha_sa, t_min, iter_max,
                          t, NTU, sv, si, sh, pf, pfi, opt)
        else:
            if t == 0:
                sa[0] = alpha_lo + rand_u01(rng_states, s) * (alpha_hi - alpha_lo)
            cuda.syncthreads()
            grasp_block(nodes, rlen, nr, dist, cost, s, scratch_unrouted, scratch_flags, scratch_scores,
                        prob_data, sa[0], rng_states, t, NTU, sv, si, sh, pf, pfi, opt)
            if t == 0:
                sa_warmup_single(nodes, rlen, nr, dist, cost, s, scratch_route, scratch_route2,
                                 prob_data, rng_states, sa_iters)
            cuda.syncthreads()
            route_elimination_block(nodes, rlen, nr, dist, cost, s, scratch_unrouted, scratch_flags, prob_data,
                                    re_passes, t, NTU, sv, si, pf, pfi, opt)
            if init_fi_ls and t == 0:
                deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_route2, prob_data, 1000000)

    @k_fn
    def update_probe_kernel(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                            bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                            cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                            ibest_nodes, ibest_rlen, ibest_nr,
                            scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                            prob_data, a_param, p_hybrid, mutation_prob, iteration, max_iter,
                            rng_states, island_size, opt, stage):
        # Profiling only (kaggle/gpu_opt/profile_kernels.py, on copies of cand / next / rng_states): the steps of
        # update_population_block_kernel up to stage 0 = candidate + refresh, 1 = + repair steps 2-3, 2 = + 2-opt
        s = cuda.blockIdx.x
        t = cuda.threadIdx.x
        sv = cuda.shared.array(NTU, float64)
        si = cuda.shared.array(NTU, int64)
        sh = cuda.shared.array(4, int64)
        pf = cuda.shared.array(PF_SIZE, float64)
        pfi = cuda.shared.array(PFI_SIZE, int32)
        island_id = s // island_size
        if t == 0:
            if rand_u01(rng_states, s) < p_hybrid:
                sho_generate(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                             ibest_nodes, ibest_rlen, ibest_nr, island_id, island_size,
                             cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                             scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                             mutation_prob, prob_data, rng_states)
            else:
                woa_generate(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                             ibest_nodes, ibest_rlen, ibest_nr, island_id,
                             cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                             bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                             scratch_unrouted, scratch_flags, a_param, prob_data, rng_states)
            sh[0] = 1 if refresh_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, prob_data) else 0
        cuda.syncthreads()
        if stage >= 1 and sh[0] == 0:
            repair_block(cand_nodes, cand_rlen, cand_nr, s, prob_data, t, NTU, sv, si, pf, pfi, opt,
                         1 if stage >= 2 else 0)

    kernels["update_probe"] = update_probe_kernel
    kernels["init_population"] = init_population_block_kernel
    kernels["update_population"] = update_population_block_kernel
    kernels["route_elimination"] = route_elimination_block_kernel
