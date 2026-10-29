"""Algorithm 3. UPDATE_POSITION_HYBRID (GPU device code): r < pHybrid -> SHO, otherwise WOA."""

from __future__ import annotations


def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""
    rand_u01 = k["rand_u01"]
    sho_update_single = k["sho_update_single"]
    woa_update_single = k["woa_update_single"]

    @dev_fn
    def update_position_hybrid(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                               best_nodes, best_rlen, best_nr,
                               cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                               bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                               scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                               a_param, p_hybrid, mutation_prob, prob_data, rng_states):
        # Algorithm 3: r < pHybrid -> BasedOnSHOUpdate (lines 3-6), otherwise BasedOnWOAUpdate (lines 7-9).
        # Returns True when a feasible candidate is left in cand (False: the current solution is returned).
        if rand_u01(rng_states, s) < p_hybrid:
            return sho_update_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                                     best_nodes, best_rlen, best_nr,
                                     cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                     scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                     mutation_prob, prob_data, rng_states)
        return woa_update_single(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                                 best_nodes, best_rlen, best_nr,
                                 cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                 bak_nodes, bak_rlen, bak_nr, bak_dist, bak_cost,
                                 scratch_route, scratch_unrouted, scratch_flags,
                                 a_param, prob_data, rng_states)

    k.update({"update_position_hybrid": update_position_hybrid})
