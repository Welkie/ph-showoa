"""Algorithm 6. CrossoverGuided for VRPSPDTW (GPU device code).

Keeps 1 (p = 0.6) or 2 routes of the best, then packs the remaining customers (partner first, then current) into new
routes, opening a route when the vehicle capacity would be exceeded (line 41, with the load model of Section 3).
Feasibility repair happens in Algorithm 5. The best is the global best.
"""

from __future__ import annotations


def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""
    open_route = k["open_route"]
    rand_u01 = k["rand_u01"]
    randint = k["randint"]
    route_capacity_ok = k["route_capacity_ok"]

    @dev_fn
    def guided_crossover_sho_single(cur_nodes, cur_rlen, cur_nr, s,
                                    peer_nodes, peer_rlen, peer_nr, peer_s,
                                    best_nodes, best_rlen, best_nr,
                                    cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                    scratch_route, scratch_unrouted, scratch_flags,
                                    prob_data, rng_states):
        # Algorithm 6: keep 1 (p = 0.6) or 2 routes of the best, then pack the remaining customers
        # (partner first, then current) into new routes, opening a route when capacity would be exceeded
        depot = int(prob_data[0])
        customer_num = int(prob_data[14])
        for i in range(customer_num + 1):
            scratch_flags[s, i] = 0
        cand_nr[s] = 0

        best_r_count = best_nr[0]
        if best_r_count > 0:
            take = 1 if best_r_count == 1 or rand_u01(rng_states, s) < 0.6 else 2
            r1 = randint(rng_states, s, 0, best_r_count - 1)
            r2 = -1
            if take == 2:
                r2 = randint(rng_states, s, 0, best_r_count - 2)
                if r2 >= r1:
                    r2 += 1
            for t in range(take):
                rb = r1 if t == 0 else r2
                lb = best_rlen[0, rb]
                if lb > 2:
                    c_idx = cand_nr[s]
                    cand_rlen[s, c_idx] = lb
                    for k in range(lb):
                        node = best_nodes[0, rb, k]
                        cand_nodes[s, c_idx, k] = node
                        if node != depot:
                            scratch_flags[s, node] = 1
                    cand_nr[s] = c_idx + 1

        unrouted_cnt = 0
        for r in range(peer_nr[peer_s]):
            for k in range(1, peer_rlen[peer_s, r] - 1):
                node = peer_nodes[peer_s, r, k]
                if node != depot and scratch_flags[s, node] == 0:
                    scratch_unrouted[s, unrouted_cnt] = node
                    scratch_flags[s, node] = 1
                    unrouted_cnt += 1
        for r in range(cur_nr[s]):
            for k in range(1, cur_rlen[s, r] - 1):
                node = cur_nodes[s, r, k]
                if node != depot and scratch_flags[s, node] == 0:
                    scratch_unrouted[s, unrouted_cnt] = node
                    scratch_flags[s, node] = 1
                    unrouted_cnt += 1
        # Safety net only: parents always cover every customer
        for c in range(1, customer_num + 1):
            if scratch_flags[s, c] == 0:
                scratch_unrouted[s, unrouted_cnt] = c
                scratch_flags[s, c] = 1
                unrouted_cnt += 1

        cur_r = -1
        for i in range(unrouted_cnt):
            c = scratch_unrouted[s, i]
            if cur_r >= 0:
                l = cand_rlen[s, cur_r]
                # try current route + c (kept closed by a depot at the end)
                cand_nodes[s, cur_r, l - 1] = c
                cand_nodes[s, cur_r, l] = depot
                if route_capacity_ok(cand_nodes[s, cur_r, :l + 1], l + 1, prob_data):
                    cand_rlen[s, cur_r] = l + 1
                    continue
                cand_nodes[s, cur_r, l - 1] = depot
            cur_r = cand_nr[s]
            open_route(cand_nodes, cand_rlen, cand_nr, s, c, depot)

    k.update({"guided_crossover_sho_single": guided_crossover_sho_single})
