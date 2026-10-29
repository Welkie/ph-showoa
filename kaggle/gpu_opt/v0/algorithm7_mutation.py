"""Algorithm 7. MutationOperators (GPU device code).

SWAP: swap, insert or reverse inside one route; RELOCATE: Pd-shift or Pd-exchange between two routes.
The move is rolled back when a constraint is violated. Same operators as src_python_baseline/algorithm7_mutation.py.
"""

from __future__ import annotations


def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""
    all_routes_feasible = k["all_routes_feasible"]
    compact_routes = k["compact_routes"]
    eval_route = k["eval_route"]
    insert_at = k["insert_at"]
    rand_u01 = k["rand_u01"]
    randint = k["randint"]
    remove_at = k["remove_at"]

    @dev_fn
    def apply_intra_op(nodes, rlen, s, r, op, rng_states):
        # Li & Lim random route operation on customers 1..n of route r: 0 swap, 1 insert, 2 reverse
        n = rlen[s, r] - 2
        if n < 2:
            return
        if op == 0:
            i = randint(rng_states, s, 0, n - 1)
            j = randint(rng_states, s, 0, n - 1)
            while j == i:
                j = randint(rng_states, s, 0, n - 1)
            tmp = nodes[s, r, i + 1]
            nodes[s, r, i + 1] = nodes[s, r, j + 1]
            nodes[s, r, j + 1] = tmp
        elif op == 1:
            i = randint(rng_states, s, 0, n - 1)
            c = nodes[s, r, i + 1]
            remove_at(nodes, rlen, s, r, i + 1)
            new_pos = randint(rng_states, s, 0, n - 1)
            insert_at(nodes, rlen, s, r, new_pos + 1, c)
        else:
            i = randint(rng_states, s, 0, n - 2)
            j = randint(rng_states, s, i + 1, n - 1)
            p1 = i + 1
            p2 = j + 1
            while p1 < p2:
                tmp = nodes[s, r, p1]
                nodes[s, r, p1] = nodes[s, r, p2]
                nodes[s, r, p2] = tmp
                p1 += 1
                p2 -= 1

    @dev_fn
    def pick_two_routes(nr, s, rng_states):
        num_r = nr[s]
        r1 = randint(rng_states, s, 0, num_r - 1)
        r2 = randint(rng_states, s, 0, num_r - 2)
        if r2 >= r1:
            r2 += 1
        return r1, r2

    @dev_fn
    def apply_inter_op(nodes, rlen, s, r1, r2, op, rng_states):
        # Li & Lim multi-route operation on routes r1, r2: 0 Pd-shift (r1 -> r2), 1 Pd-exchange.
        # Returns False when nothing was changed.
        n1 = rlen[s, r1] - 2
        n2 = rlen[s, r2] - 2
        if n1 <= 0:
            return False
        if op == 0:
            pos1 = randint(rng_states, s, 0, n1 - 1)
            c = nodes[s, r1, pos1 + 1]
            remove_at(nodes, rlen, s, r1, pos1 + 1)
            pos2 = randint(rng_states, s, 0, n2)
            insert_at(nodes, rlen, s, r2, pos2 + 1, c)
            return True
        if n2 <= 0:
            return False
        pos1 = randint(rng_states, s, 0, n1 - 1)
        pos2 = randint(rng_states, s, 0, n2 - 1)
        tmp = nodes[s, r1, pos1 + 1]
        nodes[s, r1, pos1 + 1] = nodes[s, r2, pos2 + 1]
        nodes[s, r2, pos2 + 1] = tmp
        return True

    @dev_fn
    def mutate_single(nodes, rlen, nr, s, scratch_route, scratch_route2, prob_data, rng_states):
        # Algorithm 7: SWAP (intra: swap, insert, reverse) or RELOCATE (inter: Pd-shift, Pd-exchange), rollback if infeasible
        num_r = nr[s]
        if num_r == 0:
            return
        use_intra = rand_u01(rng_states, s) < 0.5 or num_r < 2
        if use_intra:
            r = randint(rng_states, s, 0, num_r - 1)
            l = rlen[s, r]
            if l - 2 >= 2:
                for k in range(l):
                    scratch_route[s, k] = nodes[s, r, k]
                op = randint(rng_states, s, 0, 2)
                apply_intra_op(nodes, rlen, s, r, op, rng_states)
                ok, d = eval_route(nodes[s, r, :l], l, prob_data)
                if not ok:
                    for k in range(l):
                        nodes[s, r, k] = scratch_route[s, k]
                    rlen[s, r] = l
        else:
            r1, r2 = pick_two_routes(nr, s, rng_states)
            l1 = rlen[s, r1]
            l2 = rlen[s, r2]
            for k in range(l1):
                scratch_route[s, k] = nodes[s, r1, k]
            for k in range(l2):
                scratch_route2[s, k] = nodes[s, r2, k]
            op = randint(rng_states, s, 0, 1)
            if apply_inter_op(nodes, rlen, s, r1, r2, op, rng_states):
                if not all_routes_feasible(nodes, rlen, nr, s, prob_data):
                    for k in range(l1):
                        nodes[s, r1, k] = scratch_route[s, k]
                    rlen[s, r1] = l1
                    for k in range(l2):
                        nodes[s, r2, k] = scratch_route2[s, k]
                    rlen[s, r2] = l2
        compact_routes(nodes, rlen, nr, s)

    k.update({"apply_intra_op": apply_intra_op, "pick_two_routes": pick_two_routes, "apply_inter_op": apply_inter_op, "mutate_single": mutate_single})
