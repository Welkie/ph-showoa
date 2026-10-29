"""Algorithm 5. BasedOnSHOUpdate (GPU device code).

Tournament partner (k = 3, without replacement, inside the island: improvement 3), CrossoverGuided, mutation with
probability 0.35, Feasible_Or_Repair; the feasible offspring is returned even when it is worse (Direction B), so
Algorithm 1's SA acceptance decides, as in src_python_baseline/algorithm5_sho.py.
"""

from __future__ import annotations


def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""
    guided_crossover_sho_single = k["guided_crossover_sho_single"]
    is_better_lex = k["is_better_lex"]
    mutate_single = k["mutate_single"]
    rand_u01 = k["rand_u01"]
    randint = k["randint"]
    refresh_solution = k["refresh_solution"]
    repair_single = k["repair_single"]

    @dev_fn
    def sho_generate(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                     ibest_nodes, ibest_rlen, ibest_nr, island_id, island_size,
                     cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                     scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                     mutation_prob, prob_data, rng_states):
        # Algorithm 5 up to the offspring (tournament, crossover, mutation); the caller refreshes and repairs it
        island_start = island_id * island_size
        k = 3 if island_size >= 3 else island_size
        p1 = island_start + randint(rng_states, s, 0, island_size - 1)
        best_peer = p1
        if k >= 2:
            p2 = island_start + randint(rng_states, s, 0, island_size - 2)
            if p2 >= p1:
                p2 += 1
            if is_better_lex(cur_nr[p2], cur_dist[p2], cur_nr[best_peer], cur_dist[best_peer]):
                best_peer = p2
            if k >= 3:
                lo = p1 if p1 < p2 else p2
                hi = p2 if p1 < p2 else p1
                p3 = island_start + randint(rng_states, s, 0, island_size - 3)
                if p3 >= lo:
                    p3 += 1
                if p3 >= hi:
                    p3 += 1
                if is_better_lex(cur_nr[p3], cur_dist[p3], cur_nr[best_peer], cur_dist[best_peer]):
                    best_peer = p3

        guided_crossover_sho_single(cur_nodes, cur_rlen, cur_nr, s,
                                    cur_nodes, cur_rlen, cur_nr, best_peer,
                                    ibest_nodes, ibest_rlen, ibest_nr, island_id,
                                    cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                    scratch_route, scratch_unrouted, scratch_flags,
                                    prob_data, rng_states)
        if rand_u01(rng_states, s) < mutation_prob:
            mutate_single(cand_nodes, cand_rlen, cand_nr, s, scratch_route, scratch_route2, prob_data, rng_states)

    @dev_fn
    def sho_update_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                          ibest_nodes, ibest_rlen, ibest_nr, island_id, island_size,
                          cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                          scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                          mutation_prob, prob_data, rng_states):
        # Algorithm 5. Returns True when a feasible offspring is left in cand (False: current hyena returned).
        sho_generate(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                     ibest_nodes, ibest_rlen, ibest_nr, island_id, island_size,
                     cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                     scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                     mutation_prob, prob_data, rng_states)
        ok = refresh_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, prob_data)
        if not ok:
            repair_single(cand_nodes, cand_rlen, cand_nr, s, scratch_route, prob_data)
            ok = refresh_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, prob_data)
            if not ok:
                return False
        # Direction B: the feasible offspring is returned even when nf > cf; Algorithm 1 decides acceptance
        return True

    k.update({"sho_generate": sho_generate, "sho_update_single": sho_update_single})
