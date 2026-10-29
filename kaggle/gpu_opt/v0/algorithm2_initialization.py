"""Algorithm 2. The Initialization Procedure for VRPSPDTW (GPU device code, used by --init sa).

Same choices as src_python_baseline/algorithm2_initialization.py: S0 by cheapest insertion in due-time order with
U(-10, 10) noise, then Algorithm 10; simulated annealing on the total cost (pseudo-code lines 2, 7, 10) with
T0 = 1.0, alpha = 0.9, Tmin = 0.01, itermax = 20 (Priority Level 4: Table 2 gives no SA parameters).
The default initialization of this package is improvement 2 (improvement2_sa_rcrs_grasp.py).
"""

from __future__ import annotations

import math


def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""
    apply_inter_op = k["apply_inter_op"]
    apply_intra_op = k["apply_intra_op"]
    copy_solution = k["copy_solution"]
    pick_two_routes = k["pick_two_routes"]
    rand_u01 = k["rand_u01"]
    randint = k["randint"]
    refresh_solution = k["refresh_solution"]
    reinsert_customer = k["reinsert_customer"]
    repair_single = k["repair_single"]

    @dev_fn
    def init_paper_sa_single(nodes, rlen, nr, dist, cost, s,
                             cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                             best_nodes, best_rlen, best_nr, best_dist, best_cost,
                             scratch_route, scratch_unrouted, scratch_scores,
                             prob_data, rng_states, t0, alpha, t_min, iter_max):
        # S0 (Priority Level 4, as src_python_baseline): customers by due time + U(-10, 10), each inserted at its
        # cheapest feasible position (new route otherwise), then Algorithm 10
        customer_num = int(prob_data[14])
        due = prob_data[8]
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
        for i in range(customer_num):
            reinsert_customer(nodes, rlen, nr, s, scratch_unrouted[s, i], -1, False, scratch_route, prob_data)
        repair_single(nodes, rlen, nr, s, scratch_route, prob_data)
        refresh_solution(nodes, rlen, nr, dist, cost, s, prob_data)

        # Algorithm 2: simulated annealing on the total cost (pseudo-code lines 2, 7, 10)
        copy_solution(nodes, rlen, nr, dist, cost, s, best_nodes, best_rlen, best_nr, best_dist, best_cost, s)
        t = t0
        while t > t_min:
            for _ in range(iter_max):
                copy_solution(nodes, rlen, nr, dist, cost, s, cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s)
                # PERTURB (line 5): swap, insert, reverse within a route, or Pd-shift, Pd-exchange
                if rand_u01(rng_states, s) < 0.5 or cand_nr[s] < 2:
                    if cand_nr[s] > 0:
                        r = randint(rng_states, s, 0, cand_nr[s] - 1)
                        op = randint(rng_states, s, 0, 2)
                        apply_intra_op(cand_nodes, cand_rlen, s, r, op, rng_states)
                else:
                    op = randint(rng_states, s, 0, 1)
                    r1, r2 = pick_two_routes(cand_nr, s, rng_states)
                    apply_inter_op(cand_nodes, cand_rlen, s, r1, r2, op, rng_states)
                if refresh_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s, prob_data):
                    cf = cost[s]
                    delta = cand_cost[s] - cf
                    if delta < 0.0 or rand_u01(rng_states, s) < math.exp(-delta / (t * abs(cf) + 1e-6)):
                        copy_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                                      nodes, rlen, nr, dist, cost, s)
                        if cost[s] < best_cost[s]:
                            copy_solution(nodes, rlen, nr, dist, cost, s,
                                          best_nodes, best_rlen, best_nr, best_dist, best_cost, s)
            t = alpha * t
        copy_solution(best_nodes, best_rlen, best_nr, best_dist, best_cost, s, nodes, rlen, nr, dist, cost, s)

    k.update({"init_paper_sa_single": init_paper_sa_single})
