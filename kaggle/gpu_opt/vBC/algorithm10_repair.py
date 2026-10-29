"""Algorithm 10. RepairProcedure (GPU device code).

Steps 2-4 of src_python_baseline/algorithm10_repair.py: capacity repair (move the customer with the largest
|delivery - pickup| to its cheapest feasible position), time-window repair (reinsert the first late customer at the
feasible position with the earliest arrival), then intra-route 2-opt. Step 1 (duplicate / missing customers) is not
needed here: every caller passes solutions that visit each customer exactly once.
"""

from __future__ import annotations


def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""
    compact_routes = k["compact_routes"]
    eval_route = k["eval_route"]
    insert_at = k["insert_at"]
    open_route = k["open_route"]
    remove_at = k["remove_at"]
    route_capacity_ok = k["route_capacity_ok"]

    @dev_fn
    def arrival_at(route, length, pos, prob_data):
        # Start-of-service time at position pos (with waiting), as in src_python_baseline Algorithm 10
        start_tw = prob_data[7]
        service = prob_data[9]
        time_matrix = prob_data[11]
        t = prob_data[2]
        for i in range(1, pos + 1):
            t += time_matrix[route[i - 1], route[i]]
            if t < start_tw[route[i]]:
                t = start_tw[route[i]]
            if i == pos:
                return t
            t += service[route[i]]
        return t

    @dev_fn
    def first_late_position(route, length, prob_data):
        # Position of the first customer whose time window is violated; last customer if the depot return is late
        start_tw = prob_data[7]
        end_tw = prob_data[8]
        service = prob_data[9]
        time_matrix = prob_data[11]
        t = prob_data[2]
        for i in range(1, length):
            node = route[i]
            t += time_matrix[route[i - 1], node]
            if i == length - 1:
                if t > end_tw[node] + 1e-9:
                    return length - 2
                return -1
            if t < start_tw[node]:
                t = start_tw[node]
            if t > end_tw[node] + 1e-9:
                return i
            t += service[node]
        return -1

    @dev_fn
    def reinsert_customer(nodes, rlen, nr, s, c, exclude_r, by_arrival, scratch_route, prob_data):
        # Best feasible insertion of c (min distance increase, or earliest arrival); opens a new route otherwise
        depot = int(prob_data[0])
        best_val = 1e18
        best_r = -1
        best_p = -1
        L = nodes.shape[2]
        for r in range(nr[s]):
            if r == exclude_r:
                continue
            l = rlen[s, r]
            if l + 1 > L:
                continue
            ok_o, old_d = eval_route(nodes[s, r, :l], l, prob_data)
            for p in range(1, l):
                for k in range(p):
                    scratch_route[s, k] = nodes[s, r, k]
                scratch_route[s, p] = c
                for k in range(p, l):
                    scratch_route[s, k + 1] = nodes[s, r, k]
                ok_n, new_d = eval_route(scratch_route[s, :l + 1], l + 1, prob_data)
                if ok_n:
                    if by_arrival:
                        val = arrival_at(scratch_route[s, :l + 1], l + 1, p, prob_data)
                    else:
                        val = new_d - old_d
                    if val < best_val:
                        best_val = val
                        best_r = r
                        best_p = p
        if best_r != -1:
            insert_at(nodes, rlen, s, best_r, best_p, c)
        else:
            open_route(nodes, rlen, nr, s, c, depot)

    @dev_fn
    def two_opt_route_first_improvement(nodes, rlen, s, r, scratch_route, prob_data, tol):
        # Intra-route 2-opt, first improvement with restart until no improving reversal remains
        l = rlen[s, r]
        if l < 4:
            return
        ok0, best_d = eval_route(nodes[s, r, :l], l, prob_data)
        improved = True
        while improved:
            improved = False
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
                    ok, d = eval_route(scratch_route[s, :l], l, prob_data)
                    if ok and d < best_d - tol:
                        for k in range(l):
                            nodes[s, r, k] = scratch_route[s, k]
                        best_d = d
                        improved = True
                        break
                if improved:
                    break

    @dev_fn
    def repair_single(nodes, rlen, nr, s, scratch_route, prob_data):
        # Algorithm 10, steps 2-4 (step 1 is not needed: offspring here never duplicate or miss customers)
        delivery = prob_data[5]
        pickup = prob_data[6]
        # Step 2: capacity violations
        n0 = nr[s]
        for r in range(n0):
            attempts = 0
            while attempts < rlen[s, r] - 2 and not route_capacity_ok(nodes[s, r, :rlen[s, r]], rlen[s, r], prob_data):
                attempts += 1
                best_k = 1
                best_v = -1.0
                for k in range(1, rlen[s, r] - 1):
                    c = nodes[s, r, k]
                    v = abs(delivery[c] - pickup[c])
                    if v > best_v:
                        best_v = v
                        best_k = k
                c_sel = nodes[s, r, best_k]
                remove_at(nodes, rlen, s, r, best_k)
                reinsert_customer(nodes, rlen, nr, s, c_sel, r, False, scratch_route, prob_data)
        compact_routes(nodes, rlen, nr, s)
        # Step 3: time-window violations, reinsert at the feasible position with the earliest arrival
        n0 = nr[s]
        for r in range(n0):
            attempts = 0
            while attempts < rlen[s, r] - 2:
                late = first_late_position(nodes[s, r, :rlen[s, r]], rlen[s, r], prob_data)
                if late < 0:
                    break
                attempts += 1
                c_late = nodes[s, r, late]
                remove_at(nodes, rlen, s, r, late)
                reinsert_customer(nodes, rlen, nr, s, c_late, -1, True, scratch_route, prob_data)
        compact_routes(nodes, rlen, nr, s)
        # Step 4: intra-route 2-opt
        for r in range(nr[s]):
            two_opt_route_first_improvement(nodes, rlen, s, r, scratch_route, prob_data, 1e-6)

    k.update({"arrival_at": arrival_at, "first_late_position": first_late_position, "reinsert_customer": reinsert_customer, "two_opt_route_first_improvement": two_opt_route_first_improvement, "repair_single": repair_single})
