"""Device implementations of src_python's paper-mode search operators.

RCRS-GRASP construction and island migration remain in gpu_kernels. These
operators use caller-owned buffers; no allocation or host work occurs in a run.
"""

import math


def build_base_operators(dev_fn, evaluate_route, evaluate_solution, copy_solution,
                         uniform, randint, shuffle, sa_t0, sa_alpha, sa_tmin,
                         sa_itermax, mutation_probability, diversify_ratio,
                         enable_two_opt_star=True):
    @dev_fn
    def compact(nodes, lengths, counts, s):
        write = 0
        old_count = counts[s]
        for r in range(old_count):
            size = lengths[s, r]
            if size > 2:
                if write != r:
                    for p in range(size):
                        nodes[s, write, p] = nodes[s, r, p]
                lengths[s, write] = size
                write += 1
        for r in range(write, old_count):
            lengths[s, r] = 0
        counts[s] = write

    @dev_fn
    def refresh(nodes, lengths, counts, distances, costs, s, problem):
        compact(nodes, lengths, counts, s)
        ok, count, distance, cost = evaluate_solution(nodes, lengths, counts, s, problem)
        distances[s] = distance
        costs[s] = cost
        return ok

    @dev_fn
    def route_capacity(route, size, problem):
        load = 0.0
        for p in range(1, size - 1):
            load += problem[5][route[p]]
        if load > problem[1] + 1e-6:
            return False
        for p in range(1, size):
            load += problem[6][route[p]] - problem[5][route[p]]
            if load < -1e-6 or load > problem[1] + 1e-6:
                return False
        return True

    @dev_fn
    def remove_at(nodes, lengths, s, r, p):
        for k in range(p, lengths[s, r] - 1):
            nodes[s, r, k] = nodes[s, r, k + 1]
        lengths[s, r] -= 1

    @dev_fn
    def insert_at(nodes, lengths, s, r, p, customer):
        for k in range(lengths[s, r], p, -1):
            nodes[s, r, k] = nodes[s, r, k - 1]
        nodes[s, r, p] = customer
        lengths[s, r] += 1

    @dev_fn
    def insert_customer(nodes, lengths, counts, s, customer, scratch, problem,
                        mode=0, excluded=-1):
        # mode 0: base _append_customer_to_best_position (scalar delta).
        # mode 1: repair missing customer; new route only as a fallback.
        # mode 2: capacity repair; minimize total target-route cost.
        # mode 3: time repair; minimize arrival at the reinserted customer.
        depot = int(problem[0])
        best = math.inf
        best_r = -1
        best_p = 1
        scratch[s, 0], scratch[s, 1], scratch[s, 2] = depot, customer, depot
        single_ok, single_d = evaluate_route(scratch[s], 3, problem)
        can_open = counts[s] < min(nodes.shape[1], int(problem[12]))
        if mode == 0 and single_ok and can_open:
            best = problem[3] + problem[4] * single_d
        for r in range(counts[s]):
            if r == excluded:
                continue
            size = lengths[s, r]
            if size + 1 > nodes.shape[2]:
                continue
            old_d = 0.0
            for k in range(1, size):
                old_d += problem[10][nodes[s, r, k - 1], nodes[s, r, k]]
            for p in range(1, size):
                for k in range(p):
                    scratch[s, k] = nodes[s, r, k]
                scratch[s, p] = customer
                for k in range(p, size):
                    scratch[s, k + 1] = nodes[s, r, k]
                ok, distance = evaluate_route(scratch[s], size + 1, problem)
                if not ok:
                    continue
                value = (distance - old_d) * problem[4]
                if size <= 2:
                    value += problem[3]
                if mode == 2:
                    value = problem[3] + distance * problem[4]
                elif mode == 3:
                    arrival = problem[2]
                    for k in range(1, p + 1):
                        node = scratch[s, k]
                        arrival += problem[11][scratch[s, k - 1], node]
                        if k < p:
                            arrival = max(arrival, problem[7][node]) + problem[9][node]
                    value = arrival
                tolerance = 0.001 if mode == 0 else 0.0
                if value < best - tolerance:
                    best, best_r, best_p = value, r, p
        if best_r >= 0:
            insert_at(nodes, lengths, s, best_r, best_p, customer)
            return True
        if single_ok and can_open:
            r = counts[s]
            nodes[s, r, 0], nodes[s, r, 1], nodes[s, r, 2] = depot, customer, depot
            lengths[s, r] = 3
            counts[s] += 1
            return True
        return False

    @dev_fn
    def repair(nodes, lengths, counts, distances, costs, s, scratch, flags, problem):
        if refresh(nodes, lengths, counts, distances, costs, s, problem):
            return True
        n = int(problem[14])
        depot = int(problem[0])
        for c in range(n + 1):
            flags[s, c] = 0
        # Keep the first occurrence, as base feasible_or_repair_algorithm_10.
        for r in range(counts[s]):
            write = 1
            size = lengths[s, r]
            for p in range(1, size - 1):
                c = nodes[s, r, p]
                if c > 0 and c <= n and flags[s, c] == 0:
                    nodes[s, r, write] = c
                    write += 1
                    flags[s, c] = 1
            nodes[s, r, 0] = depot
            nodes[s, r, write] = depot
            lengths[s, r] = write + 1
        compact(nodes, lengths, counts, s)
        for c in range(1, n + 1):
            if flags[s, c] == 0:
                if not insert_customer(nodes, lengths, counts, s, c, scratch, problem, 1):
                    return False
        r = 0
        while r < counts[s]:
            while not route_capacity(nodes[s, r], lengths[s, r], problem):
                size = lengths[s, r]
                if size <= 2:
                    return False
                position = 1
                largest = -1.0
                for p in range(1, size - 1):
                    c = nodes[s, r, p]
                    value = abs(problem[5][c] - problem[6][c])
                    if value > largest:
                        largest, position = value, p
                c = nodes[s, r, position]
                remove_at(nodes, lengths, s, r, position)
                if not insert_customer(nodes, lengths, counts, s, c, scratch, problem, 2, r):
                    return False
            r += 1
        compact(nodes, lengths, counts, s)
        r = 0
        while r < counts[s]:
            # A successful reinsertion goes into a feasible route. Bound attempts
            # defensively for inconsistent input time windows.
            for attempt in range(n + 1):
                time = problem[2]
                violation = -1
                size = lengths[s, r]
                for p in range(1, size):
                    c = nodes[s, r, p]
                    time += problem[11][nodes[s, r, p - 1], c]
                    if time > problem[8][c] + 1e-6:
                        violation = p if c != depot else size - 2
                        break
                    time = max(time, problem[7][c]) + problem[9][c]
                if violation < 0:
                    break
                c = nodes[s, r, violation]
                remove_at(nodes, lengths, s, r, violation)
                if not insert_customer(nodes, lengths, counts, s, c, scratch, problem, 3):
                    return False
            r += 1
        compact(nodes, lengths, counts, s)
        # The final intra-route 2-opt repair in the base.
        for r in range(counts[s]):
            while True:
                improved = False
                size = lengths[s, r]
                ok, best = evaluate_route(nodes[s, r], size, problem)
                if not ok:
                    best = math.inf
                for i in range(1, size - 2):
                    for j in range(i + 1, size - 1):
                        for k in range(size):
                            scratch[s, k] = nodes[s, r, k]
                        for k in range(i, j + 1):
                            scratch[s, k] = nodes[s, r, j + i - k]
                        valid, d = evaluate_route(scratch[s], size, problem)
                        if valid and d < best - 1e-9:
                            for k in range(size):
                                nodes[s, r, k] = scratch[s, k]
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
        return refresh(nodes, lengths, counts, distances, costs, s, problem)

    @dev_fn
    def perturb(nodes, lengths, counts, s, scratch, scratch2, problem, rng, move):
        # Algorithm 2 / base: swap, insert, reverse, inter-relocate, inter-swap.
        count = counts[s]
        if count == 0 or (move >= 4 and count < 2):
            return False
        r1 = randint(rng, s, 0, count - 1)
        l1 = lengths[s, r1]
        if move <= 3:
            if l1 < 4:
                return False
            i = randint(rng, s, 1, l1 - 2)
            j = randint(rng, s, 1, l1 - 2)
            if move == 1:
                while i == j:
                    j = randint(rng, s, 1, l1 - 2)
                value = nodes[s, r1, i]
                nodes[s, r1, i] = nodes[s, r1, j]
                nodes[s, r1, j] = value
            elif move == 2:
                value = nodes[s, r1, i]
                remove_at(nodes, lengths, s, r1, i)
                insert_at(nodes, lengths, s, r1, j, value)
            else:
                if i > j:
                    i, j = j, i
                while i < j:
                    value = nodes[s, r1, i]
                    nodes[s, r1, i] = nodes[s, r1, j]
                    nodes[s, r1, j] = value
                    i += 1
                    j -= 1
        elif count >= 2:
            r2 = randint(rng, s, 0, count - 1)
            while r1 == r2:
                r2 = randint(rng, s, 0, count - 1)
            i = randint(rng, s, 1, l1 - 2)
            l2 = lengths[s, r2]
            if move == 4:
                if l2 >= nodes.shape[2]:
                    return False
                j = randint(rng, s, 1, l2 - 1)
                value = nodes[s, r1, i]
                remove_at(nodes, lengths, s, r1, i)
                insert_at(nodes, lengths, s, r2, j, value)
            else:
                j = randint(rng, s, 1, l2 - 2)
                value = nodes[s, r1, i]
                nodes[s, r1, i] = nodes[s, r2, j]
                nodes[s, r2, j] = value
        compact(nodes, lengths, counts, s)
        return True

    @dev_fn
    def warmup(nodes, lengths, counts, distances, costs, s,
               best_nodes, best_lengths, best_counts, best_distances, best_costs,
               trial_nodes, trial_lengths, trial_counts, trial_distances, trial_costs,
               scratch, scratch2, problem, rng):
        if not refresh(nodes, lengths, counts, distances, costs, s, problem):
            return
        copy_solution(nodes, lengths, counts, distances, costs, s,
                      best_nodes, best_lengths, best_counts, best_distances, best_costs, s)
        temperature = sa_t0
        while temperature > sa_tmin:
            for iteration in range(sa_itermax):
                copy_solution(nodes, lengths, counts, distances, costs, s,
                              trial_nodes, trial_lengths, trial_counts, trial_distances, trial_costs, s)
                if not perturb(trial_nodes, trial_lengths, trial_counts, s, scratch, scratch2,
                               problem, rng, randint(rng, s, 1, 5)):
                    continue
                if refresh(trial_nodes, trial_lengths, trial_counts, trial_distances, trial_costs, s, problem):
                    delta = trial_distances[s] - distances[s]
                    if delta < 0.0 or uniform(rng, s) < math.exp(-delta / (1e-6 + temperature * abs(distances[s]))):
                        copy_solution(trial_nodes, trial_lengths, trial_counts, trial_distances, trial_costs, s,
                                      nodes, lengths, counts, distances, costs, s)
                        if distances[s] < best_distances[s]:
                            copy_solution(nodes, lengths, counts, distances, costs, s,
                                          best_nodes, best_lengths, best_counts, best_distances, best_costs, s)
            temperature *= sa_alpha
        copy_solution(best_nodes, best_lengths, best_counts, best_distances, best_costs, s,
                      nodes, lengths, counts, distances, costs, s)

    @dev_fn
    def local_search(nodes, lengths, counts, distances, costs, s, scratch, scratch2,
                     problem, unused_passes=0):
        # Ordered first improvement, restarting after every improving move.
        if not refresh(nodes, lengths, counts, distances, costs, s, problem):
            return
        while True:
            improved = False
            for r in range(counts[s]):
                size = lengths[s, r]
                _, old = evaluate_route(nodes[s, r], size, problem)
                for i in range(1, size - 2):
                    for j in range(i + 1, size - 1):
                        for k in range(size):
                            scratch[s, k] = nodes[s, r, k]
                        for k in range(i, j + 1):
                            scratch[s, k] = nodes[s, r, j + i - k]
                        ok, new = evaluate_route(scratch[s], size, problem)
                        if ok and (new - old) * problem[4] < -0.001:
                            for k in range(size):
                                nodes[s, r, k] = scratch[s, k]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            for r1 in range(counts[s]):
                l1 = lengths[s, r1]
                _, old1 = evaluate_route(nodes[s, r1], l1, problem)
                for i in range(1, l1 - 1):
                    for r2 in range(counts[s]):
                        l2 = lengths[s, r2]
                        _, old2 = evaluate_route(nodes[s, r2], l2, problem)
                        for j in range(1, l2):
                            if r1 == r2 and (j == i or j == i + 1):
                                continue
                            if r1 != r2 and l2 + 1 > nodes.shape[2]:
                                continue
                            c = nodes[s, r1, i]
                            write = 0
                            for k in range(l1):
                                if k != i:
                                    scratch[s, write] = nodes[s, r1, k]
                                    write += 1
                            if r1 == r2:
                                pos = j - 1 if j > i else j
                                for k in range(l1 - 1, pos, -1):
                                    scratch[s, k] = scratch[s, k - 1]
                                scratch[s, pos] = c
                                ok, new = evaluate_route(scratch[s], l1, problem)
                                delta = (new - old1) * problem[4]
                            else:
                                for k in range(j):
                                    scratch2[s, k] = nodes[s, r2, k]
                                scratch2[s, j] = c
                                for k in range(j, l2):
                                    scratch2[s, k + 1] = nodes[s, r2, k]
                                ok1, new1 = evaluate_route(scratch[s], l1 - 1, problem)
                                ok2, new2 = evaluate_route(scratch2[s], l2 + 1, problem)
                                ok = ok1 and ok2
                                delta = (new1 + new2 - old1 - old2) * problem[4]
                                if l1 == 3:
                                    delta -= problem[3]
                            if ok and delta < -0.001:
                                size = l1 if r1 == r2 else l1 - 1
                                for k in range(size):
                                    nodes[s, r1, k] = scratch[s, k]
                                lengths[s, r1] = size
                                if r1 != r2:
                                    for k in range(l2 + 1):
                                        nodes[s, r2, k] = scratch2[s, k]
                                    lengths[s, r2] = l2 + 1
                                compact(nodes, lengths, counts, s)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            for r1 in range(counts[s]):
                l1 = lengths[s, r1]
                _, old1 = evaluate_route(nodes[s, r1], l1, problem)
                for r2 in range(r1 + 1, counts[s]):
                    l2 = lengths[s, r2]
                    _, old2 = evaluate_route(nodes[s, r2], l2, problem)
                    for i in range(1, l1 - 1):
                        for j in range(1, l2 - 1):
                            for k in range(l1):
                                scratch[s, k] = nodes[s, r1, k]
                            for k in range(l2):
                                scratch2[s, k] = nodes[s, r2, k]
                            scratch[s, i], scratch2[s, j] = nodes[s, r2, j], nodes[s, r1, i]
                            ok1, new1 = evaluate_route(scratch[s], l1, problem)
                            ok2, new2 = evaluate_route(scratch2[s], l2, problem)
                            if ok1 and ok2 and (new1 + new2 - old1 - old2) * problem[4] < -0.001:
                                nodes[s, r1, i], nodes[s, r2, j] = scratch[s, i], scratch2[s, j]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # GPU extension: exchange non-empty tails after exhausting the
            # base neighborhoods. Keep both vehicles; validate both new routes.
            if enable_two_opt_star:
                for r1 in range(counts[s]):
                    l1 = lengths[s, r1]
                    _, old1 = evaluate_route(nodes[s, r1], l1, problem)
                    for r2 in range(r1 + 1, counts[s]):
                        l2 = lengths[s, r2]
                        _, old2 = evaluate_route(nodes[s, r2], l2, problem)
                        for i in range(2, l1 - 1):
                            for j in range(2, l2 - 1):
                                n1, n2 = i + l2 - j, j + l1 - i
                                if n1 > nodes.shape[2] or n2 > nodes.shape[2]:
                                    continue
                                dm = problem[10]
                                delta = (dm[nodes[s, r1, i - 1], nodes[s, r2, j]]
                                         + dm[nodes[s, r2, j - 1], nodes[s, r1, i]]
                                         - dm[nodes[s, r1, i - 1], nodes[s, r1, i]]
                                         - dm[nodes[s, r2, j - 1], nodes[s, r2, j]])
                                if delta * problem[4] >= -0.001:
                                    continue
                                for k in range(i):
                                    scratch[s, k] = nodes[s, r1, k]
                                for k in range(j, l2):
                                    scratch[s, i + k - j] = nodes[s, r2, k]
                                for k in range(j):
                                    scratch2[s, k] = nodes[s, r2, k]
                                for k in range(i, l1):
                                    scratch2[s, j + k - i] = nodes[s, r1, k]
                                ok1, new1 = evaluate_route(scratch[s], n1, problem)
                                ok2, new2 = evaluate_route(scratch2[s], n2, problem)
                                if ok1 and ok2 and (new1 + new2 - old1 - old2) * problem[4] < -0.001:
                                    for k in range(n1):
                                        nodes[s, r1, k] = scratch[s, k]
                                    for k in range(n2):
                                        nodes[s, r2, k] = scratch2[s, k]
                                    lengths[s, r1], lengths[s, r2] = n1, n2
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                continue
            # Inter-route Relocate: move a single customer from r1 to r2
            for r1 in range(counts[s]):
                l1 = lengths[s, r1]
                if l1 < 4:
                    continue
                for r2 in range(counts[s]):
                    if r1 == r2:
                        continue
                    l2 = lengths[s, r2]
                    if l2 + 1 > nodes.shape[2]:
                        continue
                    _, old1 = evaluate_route(nodes[s, r1], l1, problem)
                    _, old2 = evaluate_route(nodes[s, r2], l2, problem)
                    dm = problem[10]
                    for i in range(1, l1 - 1):
                        u = nodes[s, r1, i]
                        prev_u = nodes[s, r1, i - 1]
                        next_u = nodes[s, r1, i + 1]
                        cost_rem = dm[prev_u, next_u] - (dm[prev_u, u] + dm[u, next_u])
                        for j in range(1, l2):
                            prev_v = nodes[s, r2, j - 1]
                            next_v = nodes[s, r2, j]
                            cost_ins = (dm[prev_v, u] + dm[u, next_v]) - dm[prev_v, next_v]
                            if (cost_rem + cost_ins) * problem[4] >= -0.001:
                                continue
                            k_idx = 0
                            for k in range(l1):
                                if k != i:
                                    scratch[s, k_idx] = nodes[s, r1, k]
                                    k_idx += 1
                            ok1, new1 = evaluate_route(scratch[s], l1 - 1, problem)
                            if ok1:
                                for k in range(j):
                                    scratch2[s, k] = nodes[s, r2, k]
                                scratch2[s, j] = u
                                for k in range(j, l2):
                                    scratch2[s, k + 1] = nodes[s, r2, k]
                                ok2, new2 = evaluate_route(scratch2[s], l2 + 1, problem)
                                if ok2 and (new1 + new2 - old1 - old2) * problem[4] < -0.001:
                                    for k in range(l1 - 1):
                                        nodes[s, r1, k] = scratch[s, k]
                                    for k in range(l1 - 1, l1):
                                        nodes[s, r1, k] = 0
                                    for k in range(l2 + 1):
                                        nodes[s, r2, k] = scratch2[s, k]
                                    lengths[s, r1] = l1 - 1
                                    lengths[s, r2] = l2 + 1
                                    compact(nodes, lengths, counts, s)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        refresh(nodes, lengths, counts, distances, costs, s, problem)

    @dev_fn
    def relink(route, size, guide, guide_size, amount, reverse, desired):
        common_count = 0
        for q in range(1, guide_size - 1):
            c = guide[q]
            for p in range(1, size - 1):
                if route[p] == c:
                    desired[common_count] = c
                    common_count += 1
                    break
        if common_count < 2:
            return
        if reverse:
            for i in range(common_count // 2):
                j = common_count - i - 1
                desired[i], desired[j] = desired[j], desired[i]
        moves = max(1, min(common_count - 1, int(math.ceil(common_count * amount))))
        for index in range(moves):
            wanted = desired[index]
            for p in range(1, size - 1):
                if route[p] == wanted:
                    route[index + 1], route[p] = route[p], route[index + 1]
                    break

    @dev_fn
    def light_mutation(nodes, lengths, counts, s, scratch, scratch2, problem, rng):
        n = int(problem[14])
        if uniform(rng, s) < 0.5:
            if n < 2:
                return
            # Uniform sampling over customer positions, not over routes.
            first = randint(rng, s, 0, n - 1)
            second = randint(rng, s, 0, n - 2)
            if second >= first:
                second += 1
            r1, p1, r2, p2 = -1, -1, -1, -1
            offset = 0
            for r in range(counts[s]):
                for p in range(1, lengths[s, r] - 1):
                    if offset == first:
                        r1, p1 = r, p
                    if offset == second:
                        r2, p2 = r, p
                    offset += 1
            if r1 >= 0 and r2 >= 0:
                nodes[s, r1, p1], nodes[s, r2, p2] = nodes[s, r2, p2], nodes[s, r1, p1]
        elif counts[s] >= 2:
            index = randint(rng, s, 0, n - 1)
            source, position = -1, -1
            offset = 0
            for r in range(counts[s]):
                for p in range(1, lengths[s, r] - 1):
                    if offset == index:
                        source, position = r, p
                    offset += 1
            if source < 0:
                return
            target = randint(rng, s, 0, counts[s] - 2)
            if target >= source:
                target += 1
            pos = randint(rng, s, 1, lengths[s, target] - 1)
            c = nodes[s, source, position]
            remove_at(nodes, lengths, s, source, position)
            insert_at(nodes, lengths, s, target, pos, c)
        compact(nodes, lengths, counts, s)

    return {
        'compact': compact, 'refresh': refresh, 'repair': repair,
        'insert_customer': insert_customer, 'capacity': route_capacity,
        'warmup': warmup, 'local_search': local_search, 'relink': relink,
        'perturb': perturb, 'light_mutation': light_mutation,
    }
