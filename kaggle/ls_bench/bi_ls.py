"""Prototype: parallel best-improvement local search (one CUDA block per solution) for the LS benchmark.

Same neighbourhoods as src_python_gpu_SA_RCRS_GRASP/algorithm8_localsearch.py (+ improvement 4):
  type 0  2-opt inside a route (reverse positions ka..kb)
  type 1  relocate the customer at slot a to just after slot b (same or another route; emptying a route = NV - 1)
  type 2  swap two customers of different routes
  type 3  inter 2-opt* (exchange the tails after ka / kb; same restrictions as the current code, no emptying)
Each step scans the whole neighbourhood (pairs of slots, a slot = (route, position)) and applies the best move:
key = delta_distance, minus BIG when a route is emptied (hierarchical objective: NV first). Ties go to the smallest
move code, so the result does not depend on the number of threads (the CPU version is the same scan with 1 thread).
Distance deltas are O(1); feasibility (load + time windows, Section 3) is checked in O(L) only for a candidate that
improves and beats the thread's best so far, by walking the new route segment by segment without copying it.
"""

import math

import numpy as np

try:
    from numba import cuda, float64, int64   # module level: the CUDA simulator patches module globals
except ImportError:  # pragma: no cover
    cuda = None

NT = 128           # threads per block (CUDA version)
TOL = 1e-4
BIG = 1.0e7


def build(is_cuda):
    if is_cuda:
        dev = lambda f: cuda.jit(device=True)(f)
    else:
        from numba import njit
        dev = lambda f: njit(fastmath=True, nogil=True)(f)

    @dev
    def seg_node(nodes, s, r, a, b, rev, t):
        return nodes[s, r, b - t] if rev else nodes[s, r, a + t]

    @dev
    def eval_segs(nodes, s, prob, r0, a0, b0, v0, r1, a1, b1, v1, r2, a2, b2, v2, r3, a3, b3, v3):
        # Feasibility of the route made of up to 4 segments (positions a..b of route r, reversed if v); empty if a > b
        capacity = prob[1]
        t0 = prob[2]
        delivery = prob[5]
        pickup = prob[6]
        ready = prob[7]
        due = prob[8]
        service = prob[9]
        tmat = prob[11]
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
        time_val = t0
        prev = 0
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
                if node == 0 and prev == 0 and time_val == t0:
                    continue                                  # leading depot
                load = load - delivery[node] + pickup[node]
                if load < -1e-9 or load > capacity + 1e-9:
                    return False
                time_val += tmat[prev, node]
                if time_val > due[node] + 1e-9:
                    return False
                if time_val < ready[node]:
                    time_val = ready[node]
                time_val += service[node]
                prev = node
        return True

    @dev
    def scan(nodes, rlen, s, slot_r, slot_k, S, start, step, prob):
        # Best (key, code) over pair indices start, start+step, ... ; code = idx * 4 + type
        dm = prob[10]
        best_key = 0.0
        best_code = -1
        for idx in range(start, S * S, step):
            sa = idx // S
            sb = idx - sa * S
            ra = slot_r[s, sa]
            ka = slot_k[s, sa]
            rb = slot_r[s, sb]
            kb = slot_k[s, sb]
            la = rlen[s, ra]
            lb = rlen[s, rb]
            # type 0: 2-opt inside route ra (current code: p in 1..l-3, q in p+1..l-2, l >= 4)
            if ra == rb and la >= 4 and ka >= 1 and kb > ka and kb <= la - 2:
                x0 = nodes[s, ra, ka - 1]
                x1 = nodes[s, ra, ka]
                y0 = nodes[s, ra, kb]
                y1 = nodes[s, ra, kb + 1]
                key = dm[x0, y0] + dm[x1, y1] - dm[x0, x1] - dm[y0, y1]
                if key < -TOL and (best_code < 0 or key < best_key):
                    if eval_segs(nodes, s, prob, ra, 0, ka - 1, False, ra, ka, kb, True, ra, kb + 1, la - 1, False,
                                 ra, 1, 0, False):
                        best_key = key
                        best_code = idx * 4
            # type 1: relocate customer at (ra, ka) to just after position kb of route rb
            if ka >= 1 and ka <= la - 2 and kb <= lb - 2 and not (ra == rb and (kb == ka or kb == ka - 1)):
                c = nodes[s, ra, ka]
                p = nodes[s, ra, ka - 1]
                n = nodes[s, ra, ka + 1]
                x = nodes[s, rb, kb]
                y = nodes[s, rb, kb + 1]
                d = dm[p, n] - dm[p, c] - dm[c, n] + dm[x, c] + dm[c, y] - dm[x, y]
                empties = ra != rb and la == 3
                key = d - BIG if empties else d
                if key < -TOL and (best_code < 0 or key < best_key):
                    if ra == rb:
                        if kb < ka:
                            ok = eval_segs(nodes, s, prob, ra, 0, kb, False, ra, ka, ka, False, ra, kb + 1, ka - 1, False,
                                           ra, ka + 1, la - 1, False)
                        else:
                            ok = eval_segs(nodes, s, prob, ra, 0, ka - 1, False, ra, ka + 1, kb, False, ra, ka, ka, False,
                                           ra, kb + 1, la - 1, False)
                    else:
                        ok = eval_segs(nodes, s, prob, ra, 0, ka - 1, False, ra, ka + 1, la - 1, False, ra, 1, 0, False,
                                       ra, 1, 0, False)
                        if ok:
                            ok = eval_segs(nodes, s, prob, rb, 0, kb, False, ra, ka, ka, False, rb, kb + 1, lb - 1, False,
                                           ra, 1, 0, False)
                    if ok:
                        best_key = key
                        best_code = idx * 4 + 1
            if ra < rb:
                # type 2: swap customers (ra, ka) and (rb, kb)
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
                        if eval_segs(nodes, s, prob, ra, 0, ka - 1, False, rb, kb, kb, False, ra, ka + 1, la - 1, False,
                                     ra, 1, 0, False):
                            if eval_segs(nodes, s, prob, rb, 0, kb - 1, False, ra, ka, ka, False, rb, kb + 1, lb - 1, False,
                                         ra, 1, 0, False):
                                best_key = key
                                best_code = idx * 4 + 2
                # type 3: 2-opt* cutting after ka (route ra) and kb (route rb); current code: cut in 1..l-3, l >= 4
                if la >= 4 and lb >= 4 and ka >= 1 and ka <= la - 3 and kb >= 1 and kb <= lb - 3:
                    u1 = nodes[s, ra, ka]
                    v1 = nodes[s, ra, ka + 1]
                    u2 = nodes[s, rb, kb]
                    v2 = nodes[s, rb, kb + 1]
                    key = dm[u1, v2] + dm[u2, v1] - dm[u1, v1] - dm[u2, v2]
                    if key < -TOL and (best_code < 0 or key < best_key):
                        if eval_segs(nodes, s, prob, ra, 0, ka, False, rb, kb + 1, lb - 1, False, ra, 1, 0, False,
                                     ra, 1, 0, False):
                            if eval_segs(nodes, s, prob, rb, 0, kb, False, ra, ka + 1, la - 1, False, ra, 1, 0, False,
                                         ra, 1, 0, False):
                                best_key = key
                                best_code = idx * 4 + 3
        return best_key, best_code

    @dev
    def build_slots(rlen, nr, s, slot_r, slot_k):
        S = 0
        for r in range(nr[s]):
            for k in range(rlen[s, r]):
                slot_r[s, S] = r
                slot_k[s, S] = k
                S += 1
        return S

    @dev
    def compact(nodes, rlen, nr, s):
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

    @dev
    def write_segs(nodes, s, out, r0, a0, b0, v0, r1, a1, b1, v1, r2, a2, b2, v2, r3, a3, b3, v3):
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

    @dev
    def apply_move(nodes, rlen, s, slot_r, slot_k, S, code, sc1, sc2):
        idx = code // 4
        typ = code - idx * 4
        sa = idx // S
        sb = idx - sa * S
        ra = slot_r[s, sa]
        ka = slot_k[s, sa]
        rb = slot_r[s, sb]
        kb = slot_k[s, sb]
        la = rlen[s, ra]
        lb = rlen[s, rb]
        n1 = 0
        n2 = 0
        if typ == 0:
            n1 = write_segs(nodes, s, sc1, ra, 0, ka - 1, False, ra, ka, kb, True, ra, kb + 1, la - 1, False, ra, 1, 0, False)
        elif typ == 1:
            if ra == rb:
                if kb < ka:
                    n1 = write_segs(nodes, s, sc1, ra, 0, kb, False, ra, ka, ka, False, ra, kb + 1, ka - 1, False,
                                    ra, ka + 1, la - 1, False)
                else:
                    n1 = write_segs(nodes, s, sc1, ra, 0, ka - 1, False, ra, ka + 1, kb, False, ra, ka, ka, False,
                                    ra, kb + 1, la - 1, False)
            else:
                n1 = write_segs(nodes, s, sc1, ra, 0, ka - 1, False, ra, ka + 1, la - 1, False, ra, 1, 0, False,
                                ra, 1, 0, False)
                n2 = write_segs(nodes, s, sc2, rb, 0, kb, False, ra, ka, ka, False, rb, kb + 1, lb - 1, False,
                                ra, 1, 0, False)
        elif typ == 2:
            n1 = write_segs(nodes, s, sc1, ra, 0, ka - 1, False, rb, kb, kb, False, ra, ka + 1, la - 1, False, ra, 1, 0, False)
            n2 = write_segs(nodes, s, sc2, rb, 0, kb - 1, False, ra, ka, ka, False, rb, kb + 1, lb - 1, False, ra, 1, 0, False)
        else:
            n1 = write_segs(nodes, s, sc1, ra, 0, ka, False, rb, kb + 1, lb - 1, False, ra, 1, 0, False, ra, 1, 0, False)
            n2 = write_segs(nodes, s, sc2, rb, 0, kb, False, ra, ka + 1, la - 1, False, ra, 1, 0, False, ra, 1, 0, False)
        for k in range(n1):
            nodes[s, ra, k] = sc1[s, k]
        for k in range(n1, la):
            nodes[s, ra, k] = 0
        rlen[s, ra] = n1
        if n2 > 0:
            for k in range(n2):
                nodes[s, rb, k] = sc2[s, k]
            for k in range(n2, lb):
                nodes[s, rb, k] = 0
            rlen[s, rb] = n2

    if is_cuda:
        @cuda.jit
        def ls_kernel(nodes, rlen, nr, slot_r, slot_k, sc1, sc2, prob, max_moves, moves_out):
            s = cuda.blockIdx.x
            t = cuda.threadIdx.x
            sk = cuda.shared.array(NT, float64)
            scode = cuda.shared.array(NT, int64)
            sh = cuda.shared.array(2, int64)
            moves = 0
            while True:
                if t == 0:
                    compact(nodes, rlen, nr, s)
                    sh[0] = build_slots(rlen, nr, s, slot_r, slot_k)
                cuda.syncthreads()
                S = sh[0]
                key, code = scan(nodes, rlen, s, slot_r, slot_k, S, t, NT, prob)
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
                        apply_move(nodes, rlen, s, slot_r, slot_k, S, scode[0], sc1, sc2)
                        sh[1] = 0
                    else:
                        sh[1] = 1
                cuda.syncthreads()
                if sh[1] == 1:
                    break
                moves += 1
            if t == 0:
                compact(nodes, rlen, nr, s)
                moves_out[s] = moves
        return ls_kernel
    else:
        @njit(fastmath=True, nogil=True)
        def ls_cpu(nodes, rlen, nr, slot_r, slot_k, sc1, sc2, prob, max_moves, moves_out):
            for s in range(nodes.shape[0]):
                moves = 0
                while True:
                    compact(nodes, rlen, nr, s)
                    S = build_slots(rlen, nr, s, slot_r, slot_k)
                    key, code = scan(nodes, rlen, s, slot_r, slot_k, S, 0, 1, prob)
                    if code < 0 or moves >= max_moves:
                        break
                    apply_move(nodes, rlen, s, slot_r, slot_k, S, code, sc1, sc2)
                    moves += 1
                compact(nodes, rlen, nr, s)
                moves_out[s] = moves
        return ls_cpu
