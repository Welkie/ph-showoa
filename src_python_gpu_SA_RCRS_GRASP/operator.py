from __future__ import annotations

import random
from typing import List, Tuple, Set

import torch

from .eval import _chk_route_list, evaluate_route_batch
from .solution import Route, Solution


# =============================================================================
# Helper Utilities: Route Distance, Route Creation & Feasible Repair
# =============================================================================

def _route_distance(nl: List[int], data) -> float:
    """Calculates total distance of a route node list."""
    dist_mat = data.dist
    return sum(dist_mat[nl[i]][nl[i+1]] for i in range(len(nl) - 1))


def optimize_route_nodes_2opt(node_list: List[int], data) -> List[int]:
    """
    Applies intra-route 2-opt distance trimming to eliminate route crossings
    and drastically reduce Total Distance (TD).
    """
    if len(node_list) <= 3:
        return node_list

    best_nl = list(node_list)
    improved = True
    dist_mat = data.dist

    def calc_dist(nl):
        return sum(dist_mat[nl[i]][nl[i+1]] for i in range(len(nl) - 1))

    best_dist = calc_dist(best_nl)

    while improved:
        improved = False
        length = len(best_nl)
        for i in range(1, length - 2):
            for j in range(i + 1, length - 1):
                a, b = best_nl[i-1], best_nl[i]
                c, d = best_nl[j], best_nl[j+1]
                old_d = dist_mat[a][b] + dist_mat[c][d]
                new_d = dist_mat[a][c] + dist_mat[b][d]
                if new_d < old_d - 1e-6:
                    new_nl = best_nl[:i] + list(reversed(best_nl[i:j+1])) + best_nl[j+1:]
                    flag, _ = _chk_route_list(new_nl, data)
                    if flag:
                        best_nl = new_nl
                        best_dist = calc_dist(best_nl)
                        improved = True
                        break
            if improved:
                break
    return best_nl


def _insert_customer_best_position_routes(routes: List[List[int]], customer: int, data) -> bool:
    """
    Inserts customer into the existing route and position that minimizes incremental distance
    while strictly satisfying capacity and time-window constraints.
    If no existing route can feasibly absorb it, opens a new route [depot, customer, depot].
    """
    depot = data.DC
    best_r_idx = -1
    best_pos = -1
    best_delta = float('inf')

    for r_idx, r_nodes in enumerate(routes):
        if len(r_nodes) < 2:
            continue
        for pos in range(1, len(r_nodes)):
            prev, nxt = r_nodes[pos - 1], r_nodes[pos]
            delta = data.dist[prev][customer] + data.dist[customer][nxt] - data.dist[prev][nxt]
            if delta < best_delta:
                cand_nl = r_nodes[:pos] + [customer] + r_nodes[pos:]
                flag, _ = _chk_route_list(cand_nl, data)
                if flag:
                    best_delta = delta
                    best_r_idx = r_idx
                    best_pos = pos

    if best_r_idx != -1:
        routes[best_r_idx].insert(best_pos, customer)
        return True
    else:
        routes.append([depot, customer, depot])
        return True


def feasible_or_repair_algorithm_10_routes(routes: List[List[int]], data) -> List[List[int]]:
    """
    Algorithm 10 Feasibility Repair:
    Ensures that every customer 1..customer_num is visited exactly once,
    and all routes are feasible according to simultaneous pickup/delivery
    capacity and time windows.
    """
    depot = data.DC
    num_customers = data.customer_num

    # 1. Clean empty routes and invalid structures
    clean_routes: List[List[int]] = []
    for r in routes:
        custs = [n for n in r if n != depot and 1 <= n <= num_customers]
        if custs:
            clean_routes.append([depot] + custs + [depot])

    if not clean_routes:
        clean_routes = [[depot, c, depot] for c in range(1, num_customers + 1)]
        return clean_routes

    # 2. Count customer occurrences and remove duplicates
    occurrences: List[List[Tuple[int, int]]] = [[] for _ in range(num_customers + 1)]
    for r_idx, r in enumerate(clean_routes):
        for pos, node in enumerate(r):
            if node != depot and 1 <= node <= num_customers:
                occurrences[node].append((r_idx, pos))

    for c in range(1, num_customers + 1):
        occs = occurrences[c]
        if len(occs) > 1:
            best_occ_idx = 0
            best_detour = float('inf')
            for occ_i, (r_i, p_i) in enumerate(occs):
                r_nodes = clean_routes[r_i]
                prev = r_nodes[p_i - 1]
                nxt = r_nodes[p_i + 1] if p_i + 1 < len(r_nodes) else depot
                detour = data.dist[prev][c] + data.dist[c][nxt] - data.dist[prev][nxt]
                if detour < best_detour:
                    best_detour = detour
                    best_occ_idx = occ_i

            for occ_i, (r_i, p_i) in enumerate(occs):
                if occ_i != best_occ_idx:
                    clean_routes[r_i][p_i] = -1

    for r_idx in range(len(clean_routes)):
        clean_routes[r_idx] = [n for n in clean_routes[r_idx] if n != -1]
    clean_routes = [r for r in clean_routes if len(r) > 2]

    # 3. Find missing customers and insert feasibly
    visited: Set[int] = set()
    for r in clean_routes:
        for node in r:
            if node != depot:
                visited.add(node)

    for c in range(1, num_customers + 1):
        if c not in visited:
            _insert_customer_best_position_routes(clean_routes, c, data)
            visited.add(c)

    # 4. Repair infeasible routes (capacity or time windows)
    final_routes: List[List[int]] = []
    infeasible_customers: List[int] = []
    for r in clean_routes:
        flag, _ = _chk_route_list(r, data)
        if flag:
            final_routes.append(r)
        else:
            custs = [n for n in r if n != depot]
            infeasible_customers.extend(custs)

    if not final_routes and infeasible_customers:
        final_routes.append([depot, infeasible_customers.pop(0), depot])

    for c in infeasible_customers:
        _insert_customer_best_position_routes(final_routes, c, data)

    return final_routes


# =============================================================================
# Multi-Operator Deep Local Search on Solution
# =============================================================================

def do_local_search(s: Solution, data, backend=None, max_passes: int = 3):
    """
    Multi-operator variable neighborhood descent (VND):
    - Intra-route 2-opt
    - Inter-route Or-opt (relocate 1 customer from route A to route B; vehicle elimination if singleton)
    - Inter-route 2-Exchange (swap 1 customer from route A with 1 from route B)
    - Inter-route 2-Opt* (cross exchange of route tails)
    """
    depot = data.DC
    routes = [list(r.node_list) for r in s.route_list if len(r.node_list) > 2]
    if not routes:
        return

    improved = True
    pass_cnt = 0

    while improved and pass_cnt < max_passes:
        improved = False
        pass_cnt += 1

        # 1. Inter-route Or-opt / Relocate (Move 1 customer from r1 to r2)
        num_r = len(routes)
        relocate_done = False
        for r1 in range(num_r):
            if relocate_done:
                break
            # Check if r1 has only 1 customer: moving it will eliminate a vehicle!
            if len(routes[r1]) == 3:
                c = routes[r1][1]
                for r2 in range(num_r):
                    if r1 == r2:
                        continue
                    for pos2 in range(1, len(routes[r2])):
                        cand_r2 = routes[r2][:pos2] + [c] + routes[r2][pos2:]
                        flag2, _ = _chk_route_list(cand_r2, data)
                        if flag2:
                            routes[r2] = cand_r2
                            routes.pop(r1)
                            improved = True
                            relocate_done = True
                            break
                    if relocate_done:
                        break
            else:
                for pos1 in range(1, len(routes[r1]) - 1):
                    c = routes[r1][pos1]
                    cand_r1 = routes[r1][:pos1] + routes[r1][pos1+1:]
                    flag1, _ = _chk_route_list(cand_r1, data)
                    if not flag1:
                        continue
                    for r2 in range(num_r):
                        if r1 == r2:
                            continue
                        for pos2 in range(1, len(routes[r2])):
                            cand_r2 = routes[r2][:pos2] + [c] + routes[r2][pos2:]
                            flag2, _ = _chk_route_list(cand_r2, data)
                            if flag2:
                                old_d = _route_distance(routes[r1], data) + _route_distance(routes[r2], data)
                                new_d = _route_distance(cand_r1, data) + _route_distance(cand_r2, data)
                                if new_d < old_d - 1e-6:
                                    routes[r1] = cand_r1
                                    routes[r2] = cand_r2
                                    improved = True
                                    relocate_done = True
                                    break
                        if relocate_done:
                            break
                    if relocate_done:
                        break
        if improved:
            continue

        # 2. Inter-route 2-Exchange / Swap (swap customer from r1 with customer from r2)
        swap_done = False
        num_r = len(routes)
        for r1 in range(num_r):
            if swap_done:
                break
            for r2 in range(r1 + 1, num_r):
                if swap_done:
                    break
                old_d = _route_distance(routes[r1], data) + _route_distance(routes[r2], data)
                for p1 in range(1, len(routes[r1]) - 1):
                    for p2 in range(1, len(routes[r2]) - 1):
                        u, v = routes[r1][p1], routes[r2][p2]
                        cand_r1 = routes[r1][:p1] + [v] + routes[r1][p1+1:]
                        cand_r2 = routes[r2][:p2] + [u] + routes[r2][p2+1:]
                        flag1, _ = _chk_route_list(cand_r1, data)
                        if flag1:
                            flag2, _ = _chk_route_list(cand_r2, data)
                            if flag2:
                                new_d = _route_distance(cand_r1, data) + _route_distance(cand_r2, data)
                                if new_d < old_d - 1e-6:
                                    routes[r1] = cand_r1
                                    routes[r2] = cand_r2
                                    improved = True
                                    swap_done = True
                                    break
                    if swap_done:
                        break
        if improved:
            continue

        # 3. Inter-route 2-Opt* (swap route tails)
        star_done = False
        num_r = len(routes)
        for r1 in range(num_r):
            if star_done:
                break
            for r2 in range(r1 + 1, num_r):
                if star_done:
                    break
                old_d = _route_distance(routes[r1], data) + _route_distance(routes[r2], data)
                len1, len2 = len(routes[r1]), len(routes[r2])
                for i in range(1, len1 - 1):
                    for j in range(1, len2 - 1):
                        cand_r1 = routes[r1][:i+1] + routes[r2][j+1:]
                        cand_r2 = routes[r2][:j+1] + routes[r1][i+1:]
                        flag1, _ = _chk_route_list(cand_r1, data)
                        if flag1:
                            flag2, _ = _chk_route_list(cand_r2, data)
                            if flag2:
                                new_d = _route_distance(cand_r1, data) + _route_distance(cand_r2, data)
                                if new_d < old_d - 1e-6:
                                    routes[r1] = cand_r1
                                    routes[r2] = cand_r2
                                    improved = True
                                    star_done = True
                                    break
                    if star_done:
                        break

    # Final intra-route 2-opt on all routes
    for r_i in range(len(routes)):
        routes[r_i] = optimize_route_nodes_2opt(routes[r_i], data)

    # Rebuild Solution
    s.route_list = []
    for r in routes:
        if len(r) > 2:
            rt = Route(data)
            rt.node_list = r
            rt.update(data)
            s.append(rt)
    s.update(data)
    s.cal_cost(data)


def new_route_insertion(s: Solution, data, backend=None, rng=None, initial_node=-1):
    pass


# =============================================================================
# Pure PyTorch GPU Tensorized Population Initialization & SA Warmup
# =============================================================================

def tensor_rcrs_grasp_init(
    P: int,
    data,
    backend,
    alpha_lo: float = 0.10,
    alpha_hi: float = 0.40,
    sa_iters: int = 150
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs an initial population tensor of shape (P, R, L) on CUDA VRAM
    using RCRS-GRASP initialization + 2-opt distance trimming + SA warm-up.
    """
    device = backend.device
    num_customers = data.customer_num
    depot = data.DC
    max_routes = min(num_customers, 30)
    max_nodes = num_customers + 2

    pop_routes = torch.full((P, max_routes, max_nodes), depot, dtype=torch.long, device=device)
    pop_lengths = torch.full((P, max_routes), 2, dtype=torch.long, device=device)
    pop_route_counts = torch.ones(P, dtype=torch.long, device=device)

    for p in range(P):
        alpha = random.uniform(alpha_lo, alpha_hi)
        unrouted = list(range(1, num_customers + 1))
        random.shuffle(unrouted)

        routes = [[depot, depot]]

        while unrouted:
            best_candidates = []
            global_best_score = float('inf')
            max_score = float('-inf')

            for c in unrouted:
                best_c = {"customer": c, "r_idx": -1, "pos": -1, "score": float('inf')}
                for r_idx, r_nodes in enumerate(routes):
                    for pos in range(1, len(r_nodes)):
                        cand_nl = r_nodes[:pos] + [c] + r_nodes[pos:]
                        flag, _ = _chk_route_list(cand_nl, data)
                        if not flag:
                            continue

                        prev, nxt = r_nodes[pos - 1], r_nodes[pos]
                        delta_td = max(0.0, data.dist[prev][c] + data.dist[c][nxt] - data.dist[prev][nxt])
                        load = sum(data.node[n].delivery for n in r_nodes[1:-1])
                        c_load = load + max(data.node[c].delivery, data.node[c].pickup)
                        rc_penalty = max(0.0, c_load - data.vehicle.capacity * 0.70)
                        rs_penalty = abs(data.dist[depot][prev] + data.dist[prev][c] - data.dist[depot][c])

                        score = delta_td + 0.5 * rc_penalty + 0.3 * rs_penalty
                        if score < best_c["score"]:
                            best_c["score"] = score
                            best_c["r_idx"] = r_idx
                            best_c["pos"] = pos

                if best_c["r_idx"] != -1:
                    if best_c["score"] < global_best_score:
                        global_best_score = best_c["score"]
                    if best_c["score"] > max_score:
                        max_score = best_c["score"]
                best_candidates.append(best_c)

            threshold = (
                float('inf')
                if (global_best_score == float('inf') or max_score == float('-inf'))
                else global_best_score + alpha * (max_score - global_best_score)
            )

            rcl_indices = [idx for idx, cand in enumerate(best_candidates) if cand["r_idx"] != -1 and cand["score"] <= threshold + 1e-9]
            forced_indices = [idx for idx, cand in enumerate(best_candidates) if cand["r_idx"] == -1]

            if not rcl_indices:
                if not forced_indices:
                    break
                pick_c = best_candidates[random.choice(forced_indices)]["customer"]
                routes.append([depot, pick_c, depot])
                unrouted.remove(pick_c)
            else:
                chosen = best_candidates[random.choice(rcl_indices)]
                routes[chosen["r_idx"]].insert(chosen["pos"], chosen["customer"])
                unrouted.remove(chosen["customer"])

        routes = feasible_or_repair_algorithm_10_routes(routes, data)

        num_r = min(len(routes), max_routes)
        pop_route_counts[p] = num_r
        for r_i in range(num_r):
            r_nodes = routes[r_i]
            r_len = min(len(r_nodes), max_nodes)
            pop_lengths[p, r_i] = r_len
            pop_routes[p, r_i, :r_len] = torch.tensor(r_nodes[:r_len], dtype=torch.long, device=device)

    if sa_iters > 0:
        pop_routes, pop_lengths, pop_route_counts = tensor_sa_warmup(
            pop_routes, pop_lengths, pop_route_counts, backend, data, sa_iters=sa_iters
        )

    # Post-warmup pure-GPU vehicle elimination and per-route 2-opt across the entire population
    feas, costs, v_cnts, dists = backend.evaluate_population_tensor(pop_routes, pop_lengths, pop_route_counts)
    all_mask = torch.ones(P, dtype=torch.bool, device=device)
    pop_routes, pop_lengths, pop_route_counts, feas, costs, v_cnts, dists = tensor_gpu_vehicle_elimination(
        pop_routes, pop_lengths, pop_route_counts, feas, costs, v_cnts, dists,
        all_mask, backend, data
    )
    pop_routes, pop_lengths, pop_route_counts, feas, costs, v_cnts, dists = tensor_gpu_intra_2opt(
        pop_routes, pop_lengths, pop_route_counts, feas, costs, v_cnts, dists,
        all_mask, backend, data, max_passes=5
    )

    return pop_routes, pop_lengths, pop_route_counts


def tensor_sa_warmup(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    backend,
    data,
    sa_iters: int = 150,
    temp_init: float = 100.0,
    cooling: float = 0.965
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Simulated Annealing warm-up with rich move set:
    - Move 1: Intra-route Swap
    - Move 2: Intra-route Insert
    - Move 3: Intra-route 2-opt (Reverse)
    - Move 4: Inter-route Relocate / Pd-Shift (vehicle elimination when singleton!)
    - Move 5: Inter-route 2-Exchange (swap between routes)
    - Move 6: Inter-route 2-Opt* (tail exchange)
    - Move 7: Intra & Inter Or-Opt (relocate 2-3 consecutive customers)
    Vectorized Metropolis acceptance across all P solutions in batch. Zero CPU transfers.
    """
    P, R, L = pop_routes.shape
    device = backend.device
    depot = data.DC

    feas, costs, v_cnts, dists = backend.evaluate_population_tensor(pop_routes, pop_lengths, pop_route_counts)

    temp = temp_init
    cooling_steps = max(sa_iters, 100)

    for _ in range(cooling_steps):
        cand_routes = pop_routes.clone()
        cand_lengths = pop_lengths.clone()
        cand_counts = pop_route_counts.clone()

        for p in range(P):
            r_cnt = int(cand_counts[p].item())
            if r_cnt == 0:
                continue

            move_type = random.randint(1, 7)

            if move_type == 1:  # Intra-route Swap
                r_idx = random.randint(0, r_cnt - 1)
                r_len = int(cand_lengths[p, r_idx].item())
                if r_len >= 4:
                    i1, i2 = random.randint(1, r_len - 2), random.randint(1, r_len - 2)
                    if i1 != i2:
                        val1 = cand_routes[p, r_idx, i1].clone()
                        val2 = cand_routes[p, r_idx, i2].clone()
                        cand_routes[p, r_idx, i1] = val2
                        cand_routes[p, r_idx, i2] = val1

            elif move_type == 2:  # Intra-route Insert
                r_idx = random.randint(0, r_cnt - 1)
                r_len = int(cand_lengths[p, r_idx].item())
                if r_len >= 4:
                    i1, i2 = random.randint(1, r_len - 2), random.randint(1, r_len - 2)
                    if i1 != i2:
                        cust = cand_routes[p, r_idx, i1].clone()
                        if i1 < i2:
                            cand_routes[p, r_idx, i1:i2] = cand_routes[p, r_idx, i1+1:i2+1].clone()
                        else:
                            cand_routes[p, r_idx, i2+1:i1+1] = cand_routes[p, r_idx, i2:i1].clone()
                        cand_routes[p, r_idx, i2] = cust

            elif move_type == 3:  # Intra-route 2-opt / Reverse
                r_idx = random.randint(0, r_cnt - 1)
                r_len = int(cand_lengths[p, r_idx].item())
                if r_len >= 4:
                    i1, i2 = random.randint(1, r_len - 2), random.randint(1, r_len - 2)
                    if i1 > i2:
                        i1, i2 = i2, i1
                    if i1 != i2:
                        sub = cand_routes[p, r_idx, i1:i2+1].clone()
                        cand_routes[p, r_idx, i1:i2+1] = torch.flip(sub, dims=[0])

            elif move_type == 4:  # Inter-route Relocate (Route Elimination if singleton!)
                if r_cnt >= 2:
                    r1 = random.randint(0, r_cnt - 1)
                    r2 = random.randint(0, r_cnt - 1)
                    while r1 == r2:
                        r2 = random.randint(0, r_cnt - 1)

                    len1 = int(cand_lengths[p, r1].item())
                    len2 = int(cand_lengths[p, r2].item())

                    if len1 >= 3 and len2 < L - 1:
                        i1 = random.randint(1, len1 - 2)
                        cust = cand_routes[p, r1, i1].clone()
                        i2 = random.randint(1, len2 - 1)

                        # Remove from r1 on GPU
                        cand_routes[p, r1, i1:len1-1] = cand_routes[p, r1, i1+1:len1].clone()
                        cand_routes[p, r1, len1-1] = depot
                        cand_lengths[p, r1] = len1 - 1

                        # Insert into r2 on GPU
                        cand_routes[p, r2, i2+1:len2+1] = cand_routes[p, r2, i2:len2].clone()
                        cand_routes[p, r2, i2] = cust
                        cand_lengths[p, r2] = len2 + 1

                        # If r1 is now empty [depot, depot], compact routes to eliminate vehicle!
                        if len1 - 1 <= 2:
                            for r_shift in range(r1, r_cnt - 1):
                                cand_routes[p, r_shift] = cand_routes[p, r_shift + 1].clone()
                                cand_lengths[p, r_shift] = cand_lengths[p, r_shift + 1]
                            cand_routes[p, r_cnt - 1].fill_(depot)
                            cand_lengths[p, r_cnt - 1] = 2
                            cand_counts[p] = r_cnt - 1

            elif move_type == 5:  # Inter-route 2-Exchange (Swap between routes)
                if r_cnt >= 2:
                    r1 = random.randint(0, r_cnt - 1)
                    r2 = random.randint(0, r_cnt - 1)
                    while r1 == r2:
                        r2 = random.randint(0, r_cnt - 1)

                    len1 = int(cand_lengths[p, r1].item())
                    len2 = int(cand_lengths[p, r2].item())

                    if len1 >= 3 and len2 >= 3:
                        i1 = random.randint(1, len1 - 2)
                        i2 = random.randint(1, len2 - 2)
                        val1 = cand_routes[p, r1, i1].clone()
                        val2 = cand_routes[p, r2, i2].clone()
                        cand_routes[p, r1, i1] = val2
                        cand_routes[p, r2, i2] = val1

            elif move_type == 6:  # Inter-route 2-Opt* (Tail Exchange)
                if r_cnt >= 2:
                    r1 = random.randint(0, r_cnt - 1)
                    r2 = random.randint(0, r_cnt - 1)
                    while r1 == r2:
                        r2 = random.randint(0, r_cnt - 1)

                    len1 = int(cand_lengths[p, r1].item())
                    len2 = int(cand_lengths[p, r2].item())

                    if len1 >= 4 and len2 >= 4:
                        i = random.randint(1, len1 - 2)
                        j = random.randint(1, len2 - 2)
                        new_len1 = i + 1 + (len2 - 1 - j)
                        new_len2 = j + 1 + (len1 - 1 - i)
                        if new_len1 < L and new_len2 < L:
                            tail1 = cand_routes[p, r1, i+1:len1].clone()
                            tail2 = cand_routes[p, r2, j+1:len2].clone()
                            cand_routes[p, r1, i+1:new_len1] = tail2
                            cand_routes[p, r1, new_len1:] = depot
                            cand_lengths[p, r1] = new_len1

                            cand_routes[p, r2, j+1:new_len2] = tail1
                            cand_routes[p, r2, new_len2:] = depot
                            cand_lengths[p, r2] = new_len2

            elif move_type == 7:  # Intra & Inter Or-Opt (Block relocation of 2-3 customers)
                r1 = random.randint(0, r_cnt - 1)
                len1 = int(cand_lengths[p, r1].item())
                if len1 >= 5:
                    k = 3 if (len1 >= 6 and random.random() < 0.5) else 2
                    i1 = random.randint(1, len1 - 1 - k)
                    block = cand_routes[p, r1, i1:i1+k].clone()

                    is_inter = (r_cnt >= 2) and (random.random() < 0.5)
                    if is_inter:
                        r2 = random.randint(0, r_cnt - 1)
                        while r2 == r1:
                            r2 = random.randint(0, r_cnt - 1)
                        len2 = int(cand_lengths[p, r2].item())
                        if len2 + k < L:
                            # Remove block from r1
                            cand_routes[p, r1, i1:len1-k] = cand_routes[p, r1, i1+k:len1].clone()
                            cand_routes[p, r1, len1-k:len1] = depot
                            cand_lengths[p, r1] = len1 - k

                            # Insert block into r2
                            i2 = random.randint(1, len2 - 1)
                            cand_routes[p, r2, i2+k:len2+k] = cand_routes[p, r2, i2:len2].clone()
                            cand_routes[p, r2, i2:i2+k] = block
                            cand_lengths[p, r2] = len2 + k

                            # If r1 is now empty, compact
                            if len1 - k <= 2:
                                for r_shift in range(r1, r_cnt - 1):
                                    cand_routes[p, r_shift] = cand_routes[p, r_shift + 1].clone()
                                    cand_lengths[p, r_shift] = cand_lengths[p, r_shift + 1]
                                cand_routes[p, r_cnt - 1].fill_(depot)
                                cand_lengths[p, r_cnt - 1] = 2
                                cand_counts[p] = r_cnt - 1
                    else:
                        # Intra-route Or-opt
                        rem_len = len1 - k
                        if rem_len >= 3:
                            cand_routes[p, r1, i1:rem_len] = cand_routes[p, r1, i1+k:len1].clone()
                            cand_routes[p, r1, rem_len:len1] = depot
                            i2 = random.randint(1, rem_len - 1)
                            cand_routes[p, r1, i2+k:len1] = cand_routes[p, r1, i2:rem_len].clone()
                            cand_routes[p, r1, i2:i2+k] = block

        cand_feas, cand_costs, cand_v_cnts, cand_dists = backend.evaluate_population_tensor(
            cand_routes, cand_lengths, cand_counts
        )

        # Pure Vectorized Lexicographic Metropolis Acceptance
        c_nv, r_nv = cand_v_cnts, v_cnts
        c_d, r_d = cand_dists, dists

        better_nv = c_nv < r_nv
        worse_nv = c_nv > r_nv
        same_nv = c_nv == r_nv
        delta = c_d - r_d
        better_d = same_nv & (delta <= 0.001)

        denom = 1e-6 + temp * r_d.abs()
        sa_prob = torch.exp(-delta.clamp(min=0.0) / denom)
        sa_acc = same_nv & (torch.rand(P, device=device) < sa_prob)

        accept = cand_feas & (better_nv | better_d | sa_acc) & ~worse_nv

        pop_routes[accept] = cand_routes[accept]
        pop_lengths[accept] = cand_lengths[accept]
        pop_route_counts[accept] = cand_counts[accept]
        costs[accept] = cand_costs[accept]
        v_cnts[accept] = cand_v_cnts[accept]
        dists[accept] = cand_dists[accept]

        temp *= cooling

    return pop_routes, pop_lengths, pop_route_counts


# =============================================================================
# PyTorch GPU Tensorized Genetic Crossover & WOA Operators
# =============================================================================

def tensor_gpu_elite_crossover(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    best_indices,
    p_hybrid_mask: torch.Tensor,
    backend,
    data
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Tensorized Elite Route Crossover (Non-increasing Vehicle Count).
    Replaces the route in child p with highest customer overlap by the elite route from best_p.
    Removes duplicated customers from other routes, and re-inserts displaced customers feasibly.
    Guarantees vehicle count NV_cand <= NV_parent so offspring are not rejected by lexicographic filter.
    """
    P, R, L = pop_routes.shape
    device = backend.device
    depot = data.DC

    cand_routes = pop_routes.clone()
    cand_lengths = pop_lengths.clone()
    cand_counts = pop_route_counts.clone()

    active_p = torch.nonzero(p_hybrid_mask, as_tuple=True)[0]
    if len(active_p) == 0:
        return cand_routes, cand_lengths, cand_counts

    for p in active_p:
        if isinstance(best_indices, int):
            best_p = best_indices
        elif hasattr(best_indices, "__getitem__"):
            best_p = int(best_indices[p].item())
        else:
            best_p = int(best_indices)

        best_r_cnt = int(pop_route_counts[best_p].item())
        curr_r_cnt = int(cand_counts[p].item())
        if best_r_cnt == 0 or curr_r_cnt == 0:
            continue

        elite_r = torch.randint(0, best_r_cnt, (1,), device=device).item()
        elite_len = int(pop_lengths[best_p, elite_r].item())
        if elite_len <= 2:
            continue

        elite_custs = pop_routes[best_p, elite_r, 1:elite_len-1]

        # Find which route in p has maximum overlap with elite_custs
        max_overlap = -1
        best_target_r = 0
        for r in range(curr_r_cnt):
            r_len = int(cand_lengths[p, r].item())
            if r_len <= 2:
                continue
            r_custs = cand_routes[p, r, 1:r_len-1]
            overlap = torch.isin(r_custs, elite_custs).sum().item()
            if overlap > max_overlap:
                max_overlap = overlap
                best_target_r = r

        # Old customers in best_target_r that are NOT in elite_custs must be re-inserted
        old_target_len = int(cand_lengths[p, best_target_r].item())
        old_custs = cand_routes[p, best_target_r, 1:old_target_len-1]
        orphans_mask = ~torch.isin(old_custs, elite_custs)
        orphans = old_custs[orphans_mask]

        # Replace best_target_r with the elite route
        cand_routes[p, best_target_r].fill_(depot)
        cand_routes[p, best_target_r, :elite_len] = pop_routes[best_p, elite_r, :elite_len]
        cand_lengths[p, best_target_r] = elite_len

        # In all other routes of p, remove any customers that are in elite_custs
        for r in range(curr_r_cnt):
            if r == best_target_r:
                continue
            r_len = int(cand_lengths[p, r].item())
            if r_len <= 2:
                continue
            nodes = cand_routes[p, r, 1:r_len-1]
            keep_mask = ~torch.isin(nodes, elite_custs)
            kept = nodes[keep_mask]
            cand_routes[p, r].fill_(depot)
            new_len = len(kept) + 2
            if len(kept) > 0:
                cand_routes[p, r, 1:new_len-1] = kept
            cand_lengths[p, r] = new_len

        # Re-insert orphans feasibly into existing routes without opening new routes
        all_orphans_inserted = True
        if len(orphans) > 0:
            for orphan_cust in orphans:
                c_val = orphan_cust.item()
                best_r = -1
                best_pos = -1
                best_cost_inc = float('inf')

                for r in range(curr_r_cnt):
                    if r == best_target_r:
                        continue
                    r_len = int(cand_lengths[p, r].item())
                    if r_len >= L - 1 or r_len <= 2:
                        continue

                    r_nodes = cand_routes[p, r, :r_len]
                    N_cand = r_len - 1
                    cand_batch_padded = torch.full((N_cand, r_len + 1), depot, dtype=torch.long, device=device)
                    for pos_i in range(1, r_len):
                        cand_batch_padded[pos_i - 1, :pos_i] = r_nodes[:pos_i]
                        cand_batch_padded[pos_i - 1, pos_i] = c_val
                        cand_batch_padded[pos_i - 1, pos_i+1:] = r_nodes[pos_i:]
                    cand_lens = torch.full((N_cand,), r_len + 1, dtype=torch.long, device=device)

                    feas_batch, dist_batch = backend.evaluate_routes_gpu(cand_batch_padded, cand_lens)
                    if feas_batch.any():
                        valid_dist = torch.where(feas_batch, dist_batch, torch.tensor(float('inf'), device=device))
                        min_d, min_idx = torch.min(valid_dist, dim=0)
                        if min_d.item() < best_cost_inc:
                            best_cost_inc = min_d.item()
                            best_r = r
                            best_pos = min_idx.item() + 1

                # If no other route can absorb c_val, try inserting into best_target_r
                if best_r == -1:
                    r_len = int(cand_lengths[p, best_target_r].item())
                    if r_len < L - 1:
                        r_nodes = cand_routes[p, best_target_r, :r_len]
                        N_cand = r_len - 1
                        cand_batch_padded = torch.full((N_cand, r_len + 1), depot, dtype=torch.long, device=device)
                        for pos_i in range(1, r_len):
                            cand_batch_padded[pos_i - 1, :pos_i] = r_nodes[:pos_i]
                            cand_batch_padded[pos_i - 1, pos_i] = c_val
                            cand_batch_padded[pos_i - 1, pos_i+1:] = r_nodes[pos_i:]
                        cand_lens = torch.full((N_cand,), r_len + 1, dtype=torch.long, device=device)
                        feas_batch, dist_batch = backend.evaluate_routes_gpu(cand_batch_padded, cand_lens)
                        if feas_batch.any():
                            valid_dist = torch.where(feas_batch, dist_batch, torch.tensor(float('inf'), device=device))
                            min_d, min_idx = torch.min(valid_dist, dim=0)
                            best_r = best_target_r
                            best_pos = min_idx.item() + 1

                if best_r != -1:
                    cur_len = int(cand_lengths[p, best_r].item())
                    cand_routes[p, best_r, best_pos+1:cur_len+1] = cand_routes[p, best_r, best_pos:cur_len].clone()
                    cand_routes[p, best_r, best_pos] = c_val
                    cand_lengths[p, best_r] = cur_len + 1
                else:
                    all_orphans_inserted = False
                    break

        if not all_orphans_inserted:
            # If an orphan cannot be feasibly placed, abandon crossover to strictly preserve valid solution
            cand_routes[p] = pop_routes[p].clone()
            cand_lengths[p] = pop_lengths[p].clone()
            cand_counts[p] = pop_route_counts[p].clone()
            continue

        # Compact empty routes (length <= 2) to eliminate vehicles if any
        write_idx = 0
        for r in range(curr_r_cnt):
            if cand_lengths[p, r] > 2:
                if write_idx != r:
                    cand_routes[p, write_idx] = cand_routes[p, r].clone()
                    cand_lengths[p, write_idx] = cand_lengths[p, r]
                write_idx += 1
        for r in range(write_idx, R):
            cand_routes[p, r].fill_(depot)
            cand_lengths[p, r] = 2
        cand_counts[p] = max(1, write_idx)

    return cand_routes, cand_lengths, cand_counts


def tensor_gpu_intra_2opt(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    feas: torch.Tensor,
    costs: torch.Tensor,
    v_counts: torch.Tensor,
    total_dists: torch.Tensor,
    active_mask: torch.Tensor,
    backend,
    data,
    max_passes: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Tensorized Intra-Route 2-Opt Downhill Optimizer (Per-Route Acceptance).
    Evaluates all (i, j) subsegment reversal pairs per route directly on CUDA VRAM.
    Accepts moves per-route using evaluate_routes_gpu without whole-individual rejection.
    Zero host-device memory transfers.
    """
    P, R, L = pop_routes.shape
    device = backend.device
    dist_t = backend.dist_t

    active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
    if len(active_indices) == 0:
        return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists

    for p in active_indices:
        p_idx = int(p.item())
        r_cnt = int(pop_route_counts[p_idx].item())
        for r in range(r_cnt):
            r_len = int(pop_lengths[p_idx, r].item())
            if r_len < 5:
                continue

            for _ in range(max_passes):
                nodes = pop_routes[p_idx, r, :r_len]
                N = r_len - 2
                prev_i = nodes[0:N]
                curr_i = nodes[1:N+1]
                curr_j = nodes[1:N+1]
                next_j = nodes[2:N+2]

                A = prev_i.unsqueeze(1)
                B = curr_i.unsqueeze(1)
                C = curr_j.unsqueeze(0)
                D = next_j.unsqueeze(0)

                delta_mat = dist_t[A, C] + dist_t[B, D] - (dist_t[A, B] + dist_t[C, D])

                u_idx = torch.arange(N, device=device).unsqueeze(1)
                v_idx = torch.arange(N, device=device).unsqueeze(0)
                valid_mask = v_idx > u_idx

                delta_mat = torch.where(valid_mask, delta_mat, torch.tensor(float('inf'), device=device))
                improving_indices = torch.nonzero(delta_mat < -1e-4, as_tuple=False)
                if len(improving_indices) == 0:
                    break

                improving_deltas = delta_mat[improving_indices[:, 0], improving_indices[:, 1]]
                sort_order = torch.argsort(improving_deltas)
                top_k = min(len(sort_order), 8)
                best_moves = improving_indices[sort_order[:top_k]]

                batch_routes = nodes.unsqueeze(0).repeat(top_k, 1)
                for m_i in range(top_k):
                    u = best_moves[m_i, 0].item()
                    v = best_moves[m_i, 1].item()
                    i_pos = u + 1
                    j_pos = v + 1
                    batch_routes[m_i, i_pos:j_pos+1] = torch.flip(nodes[i_pos:j_pos+1], dims=[0])

                batch_lens = torch.full((top_k,), r_len, dtype=torch.long, device=device)
                cand_feas, cand_dists = backend.evaluate_routes_gpu(batch_routes, batch_lens)

                if cand_feas.any():
                    valid_dists = torch.where(cand_feas, cand_dists, torch.tensor(float('inf'), device=device))
                    best_cand_d, best_cand_idx = torch.min(valid_dists, dim=0)

                    curr_route = pop_routes[p_idx, r:r+1, :r_len]
                    curr_len_t = pop_lengths[p_idx, r:r+1]
                    _, curr_d = backend.evaluate_routes_gpu(curr_route, curr_len_t)
                    if best_cand_d.item() < curr_d.item() - 1e-4:
                        pop_routes[p_idx, r, :r_len] = batch_routes[best_cand_idx]
                    else:
                        break
                else:
                    break

    # Re-evaluate population tensor on GPU to refresh solution-level scores
    c_feas, c_costs, c_v_cnts, c_dists = backend.evaluate_population_tensor(
        pop_routes, pop_lengths, pop_route_counts
    )
    pop_mask = torch.zeros(P, dtype=torch.bool, device=device)
    pop_mask[active_indices] = True
    feas[pop_mask] = c_feas[pop_mask]
    costs[pop_mask] = c_costs[pop_mask]
    v_counts[pop_mask] = c_v_cnts[pop_mask]
    total_dists[pop_mask] = c_dists[pop_mask]

    return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists


def tensor_gpu_vehicle_elimination(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    feas: torch.Tensor,
    costs: torch.Tensor,
    v_counts: torch.Tensor,
    total_dists: torch.Tensor,
    active_mask: torch.Tensor,
    backend,
    data,
    max_customers_in_route: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dedicated Pure-GPU Vehicle Elimination Routine.
    Targets shortest routes (1-4 customers) in each active individual.
    Attempts to absorb ALL customers of the target route into the remaining routes
    using GPU-verified feasible insertions. If successful, the route is deleted,
    reducing vehicle count (NV -> NV - 1) and cutting dispatching cost by 2000!
    """
    P, R, L = pop_routes.shape
    device = backend.device
    depot = data.DC

    active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
    if len(active_indices) == 0:
        return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists

    for p in active_indices:
        p_idx = int(p.item())
        r_cnt = int(pop_route_counts[p_idx].item())
        if r_cnt <= 1:
            continue

        short_routes = []
        for r in range(r_cnt):
            l = int(pop_lengths[p_idx, r].item())
            if 3 <= l <= max_customers_in_route + 2:
                short_routes.append((l - 2, r))

        if not short_routes:
            continue

        short_routes.sort()
        for num_c, r_victim in short_routes:
            len_victim = int(pop_lengths[p_idx, r_victim].item())
            if len_victim <= 2:
                continue

            victim_custs = pop_routes[p_idx, r_victim, 1:len_victim-1].clone()
            cand_routes = pop_routes[p_idx].clone()
            cand_lengths = pop_lengths[p_idx].clone()

            all_absorbed = True
            for c_tensor in victim_custs:
                c = c_tensor.item()
                best_dst_r = -1
                best_dst_pos = -1
                min_cost = float('inf')

                for r_dst in range(r_cnt):
                    if r_dst == r_victim:
                        continue
                    len_dst = int(cand_lengths[r_dst].item())
                    if len_dst >= L - 1 or len_dst <= 2:
                        continue

                    N_cand = len_dst - 1
                    r_nodes = cand_routes[r_dst, :len_dst]
                    batch_cand = torch.full((N_cand, len_dst + 1), depot, dtype=torch.long, device=device)
                    for pos_i in range(1, len_dst):
                        batch_cand[pos_i - 1, :pos_i] = r_nodes[:pos_i]
                        batch_cand[pos_i - 1, pos_i] = c
                        batch_cand[pos_i - 1, pos_i+1:] = r_nodes[pos_i:]
                    cand_lens = torch.full((N_cand,), len_dst + 1, dtype=torch.long, device=device)

                    cand_feas, cand_dists = backend.evaluate_routes_gpu(batch_cand, cand_lens)
                    if cand_feas.any():
                        valid_dists = torch.where(cand_feas, cand_dists, torch.tensor(float('inf'), device=device))
                        min_d, min_idx = torch.min(valid_dists, dim=0)
                        if min_d.item() < min_cost:
                            min_cost = min_d.item()
                            best_dst_r = r_dst
                            best_dst_pos = min_idx.item() + 1

                if best_dst_r != -1:
                    cur_len = int(cand_lengths[best_dst_r].item())
                    cand_routes[best_dst_r, best_dst_pos+1:cur_len+1] = cand_routes[best_dst_r, best_dst_pos:cur_len].clone()
                    cand_routes[best_dst_r, best_dst_pos] = c
                    cand_lengths[best_dst_r] = cur_len + 1
                else:
                    all_absorbed = False
                    break

            if all_absorbed:
                cand_routes[r_victim].fill_(depot)
                cand_lengths[r_victim] = 2

                write_idx = 0
                for r in range(r_cnt):
                    if cand_lengths[r] > 2:
                        pop_routes[p_idx, write_idx] = cand_routes[r].clone()
                        pop_lengths[p_idx, write_idx] = cand_lengths[r]
                        write_idx += 1
                for r in range(write_idx, R):
                    pop_routes[p_idx, r].fill_(depot)
                    pop_lengths[p_idx, r] = 2
                pop_route_counts[p_idx] = write_idx
                break

    c_feas, c_costs, c_v_cnts, c_dists = backend.evaluate_population_tensor(
        pop_routes, pop_lengths, pop_route_counts
    )
    pop_mask = torch.zeros(P, dtype=torch.bool, device=device)
    pop_mask[active_indices] = True
    feas[pop_mask] = c_feas[pop_mask]
    costs[pop_mask] = c_costs[pop_mask]
    v_counts[pop_mask] = c_v_cnts[pop_mask]
    total_dists[pop_mask] = c_dists[pop_mask]

    return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists


def tensor_gpu_inter_2opt_star(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    feas: torch.Tensor,
    costs: torch.Tensor,
    v_counts: torch.Tensor,
    total_dists: torch.Tensor,
    active_mask: torch.Tensor,
    backend,
    data,
    max_pairs: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Tensorized Inter-Route 2-Opt* (Tail Exchange).
    Tests exchanging route tails between pairs of routes (r1, r2):
       r1' = r1[:i+1] + r2[j+1:]
       r2' = r2[:j+1] + r1[i+1:]
    Evaluates both candidate routes with evaluate_routes_gpu.
    Accepts moves where both routes are feasible and total distance decreases.
    """
    P, R, L = pop_routes.shape
    device = backend.device
    depot = data.DC

    active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
    if len(active_indices) == 0:
        return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists

    for p in active_indices:
        p_idx = int(p.item())
        r_cnt = int(pop_route_counts[p_idx].item())
        if r_cnt < 2:
            continue

        pairs_tried = 0
        for r1 in range(r_cnt):
            if pairs_tried >= max_pairs:
                break
            len1 = int(pop_lengths[p_idx, r1].item())
            if len1 < 4:
                continue

            for r2 in range(r1 + 1, r_cnt):
                if pairs_tried >= max_pairs:
                    break
                len2 = int(pop_lengths[p_idx, r2].item())
                if len2 < 4:
                    continue

                pairs_tried += 1

                cur_batch = torch.stack([pop_routes[p_idx, r1], pop_routes[p_idx, r2]])
                cur_lens = torch.tensor([len1, len2], dtype=torch.long, device=device)
                _, cur_dists = backend.evaluate_routes_gpu(cur_batch, cur_lens)
                cur_total_d = cur_dists.sum().item()

                nodes1 = pop_routes[p_idx, r1, :len1]
                nodes2 = pop_routes[p_idx, r2, :len2]

                i_vals = torch.arange(1, len1 - 2, device=device)
                j_vals = torch.arange(1, len2 - 2, device=device)
                if len(i_vals) == 0 or len(j_vals) == 0:
                    continue

                u_i = nodes1[i_vals]
                u_next = nodes1[i_vals + 1]
                v_j = nodes2[j_vals]
                v_next = nodes2[j_vals + 1]

                cost_orig = backend.dist_t[u_i, u_next].unsqueeze(1) + backend.dist_t[v_j, v_next].unsqueeze(0)
                cost_new = backend.dist_t[u_i.unsqueeze(1), v_next.unsqueeze(0)] + backend.dist_t[v_j.unsqueeze(0), u_next.unsqueeze(1)]
                delta_mat = cost_new - cost_orig

                new_len1_mat = len2 + i_vals.unsqueeze(1) - j_vals.unsqueeze(0)
                new_len2_mat = len1 + j_vals.unsqueeze(0) - i_vals.unsqueeze(1)

                neg_mask = (delta_mat < -1e-4) & (new_len1_mat < L) & (new_len2_mat < L)
                if not neg_mask.any():
                    continue

                neg_coords = torch.nonzero(neg_mask, as_tuple=False)
                neg_vals = delta_mat[neg_coords[:, 0], neg_coords[:, 1]]
                topk = min(len(neg_vals), 8)
                topk_idx = torch.argsort(neg_vals)[:topk]
                selected_cuts = neg_coords[topk_idx]

                M = len(selected_cuts)
                cand_routes_batch = torch.full((2 * M, L), depot, dtype=torch.long, device=device)
                cand_lens_batch = torch.empty((2 * M,), dtype=torch.long, device=device)

                for m in range(M):
                    i = int(i_vals[selected_cuts[m, 0]].item())
                    j = int(j_vals[selected_cuts[m, 1]].item())
                    nlen1 = len2 + i - j
                    nlen2 = len1 + j - i

                    cand_routes_batch[2 * m, :i + 1] = nodes1[:i + 1]
                    cand_routes_batch[2 * m, i + 1:nlen1] = nodes2[j + 1:]
                    cand_lens_batch[2 * m] = nlen1

                    cand_routes_batch[2 * m + 1, :j + 1] = nodes2[:j + 1]
                    cand_routes_batch[2 * m + 1, j + 1:nlen2] = nodes1[i + 1:]
                    cand_lens_batch[2 * m + 1] = nlen2

                pair_feas, pair_dists = backend.evaluate_routes_gpu(cand_routes_batch, cand_lens_batch)

                best_m = -1
                best_d = cur_total_d
                for m in range(M):
                    if pair_feas[2 * m] and pair_feas[2 * m + 1]:
                        pair_d = (pair_dists[2 * m] + pair_dists[2 * m + 1]).item()
                        if pair_d < best_d - 1e-4:
                            best_d = pair_d
                            best_m = m

                if best_m != -1:
                    pop_routes[p_idx, r1] = cand_routes_batch[2 * best_m]
                    pop_lengths[p_idx, r1] = cand_lens_batch[2 * best_m]
                    pop_routes[p_idx, r2] = cand_routes_batch[2 * best_m + 1]
                    pop_lengths[p_idx, r2] = cand_lens_batch[2 * best_m + 1]

    c_feas, c_costs, c_v_cnts, c_dists = backend.evaluate_population_tensor(
        pop_routes, pop_lengths, pop_route_counts
    )
    pop_mask = torch.zeros(P, dtype=torch.bool, device=device)
    pop_mask[active_indices] = True
    feas[pop_mask] = c_feas[pop_mask]
    costs[pop_mask] = c_costs[pop_mask]
    v_counts[pop_mask] = c_v_cnts[pop_mask]
    total_dists[pop_mask] = c_dists[pop_mask]

    return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists


def tensor_gpu_smart_relocate(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    feas: torch.Tensor,
    costs: torch.Tensor,
    v_counts: torch.Tensor,
    total_dists: torch.Tensor,
    active_mask: torch.Tensor,
    backend,
    data,
    max_attempts: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Tensorized Smart Inter-Route Relocate & Vehicle Elimination.
    Evaluates candidate insertion positions across all destination routes via evaluate_routes_gpu
    on CUDA VRAM.
    """
    P, R, L = pop_routes.shape
    device = backend.device
    depot = data.DC

    active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
    if len(active_indices) == 0:
        return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists

    for p in active_indices:
        p_idx = int(p.item())
        r_cnt = int(pop_route_counts[p_idx].item())
        if r_cnt <= 1:
            continue

        for _ in range(max_attempts):
            src_candidates = []
            for r in range(r_cnt):
                if int(pop_lengths[p_idx, r].item()) <= 4:
                    src_candidates.append(r)
            if src_candidates:
                r_src = random.choice(src_candidates)
            else:
                r_src = torch.randint(0, r_cnt, (1,), device=device).item()

            len_src = int(pop_lengths[p_idx, r_src].item())
            if len_src <= 2:
                continue

            pos_src = torch.randint(1, len_src - 1, (1,), device=device).item()
            cust = pop_routes[p_idx, r_src, pos_src].item()

            # Test source route feasibility and distance without cust
            cand_src = pop_routes[p_idx, r_src, :len_src].clone()
            cand_src_rem = torch.full((L,), depot, dtype=torch.long, device=device)
            cand_src_rem[:pos_src] = cand_src[:pos_src]
            cand_src_rem[pos_src:len_src-1] = cand_src[pos_src+1:]
            new_len_src = len_src - 1
            src_feas, src_dist = backend.evaluate_routes_gpu(cand_src_rem.unsqueeze(0), torch.tensor([new_len_src], dtype=torch.long, device=device))
            if not src_feas[0] and new_len_src > 2:
                continue

            _, cur_src_dist = backend.evaluate_routes_gpu(cand_src.unsqueeze(0), torch.tensor([len_src], dtype=torch.long, device=device))
            delta_src = src_dist[0].item() - cur_src_dist[0].item()

            best_dst = -1
            best_pos = -1
            best_net_delta = 0.0

            for r_dst in range(r_cnt):
                if r_dst == r_src:
                    continue
                len_dst = int(pop_lengths[p_idx, r_dst].item())
                if len_dst >= L - 1 or len_dst <= 2:
                    continue

                r_nodes = pop_routes[p_idx, r_dst, :len_dst]
                _, cur_dst_dist = backend.evaluate_routes_gpu(r_nodes.unsqueeze(0), torch.tensor([len_dst], dtype=torch.long, device=device))

                N_cand = len_dst - 1
                batch_cand = torch.full((N_cand, len_dst + 1), depot, dtype=torch.long, device=device)
                for pos_i in range(1, len_dst):
                    batch_cand[pos_i - 1, :pos_i] = r_nodes[:pos_i]
                    batch_cand[pos_i - 1, pos_i] = cust
                    batch_cand[pos_i - 1, pos_i+1:] = r_nodes[pos_i:]
                cand_lens = torch.full((N_cand,), len_dst + 1, dtype=torch.long, device=device)

                cand_feas, cand_dists = backend.evaluate_routes_gpu(batch_cand, cand_lens)
                if cand_feas.any():
                    valid_dists = torch.where(cand_feas, cand_dists, torch.tensor(float('inf'), device=device))
                    min_d, min_idx = torch.min(valid_dists, dim=0)
                    delta_dst = min_d.item() - cur_dst_dist[0].item()
                    net_delta = delta_src + delta_dst
                    if net_delta < best_net_delta - 1e-4:
                        best_net_delta = net_delta
                        best_dst = r_dst
                        best_pos = min_idx.item() + 1

            if best_dst != -1:
                # Apply move
                len_dst = int(pop_lengths[p_idx, best_dst].item())
                pop_routes[p_idx, best_dst, best_pos+1:len_dst+1] = pop_routes[p_idx, best_dst, best_pos:len_dst].clone()
                pop_routes[p_idx, best_dst, best_pos] = cust
                pop_lengths[p_idx, best_dst] = len_dst + 1

                pop_routes[p_idx, r_src, :new_len_src] = cand_src_rem[:new_len_src]
                pop_routes[p_idx, r_src, new_len_src:] = depot
                pop_lengths[p_idx, r_src] = new_len_src

                # Eliminate route if empty
                if new_len_src <= 2:
                    for r_shift in range(r_src, r_cnt - 1):
                        pop_routes[p_idx, r_shift] = pop_routes[p_idx, r_shift + 1].clone()
                        pop_lengths[p_idx, r_shift] = pop_lengths[p_idx, r_shift + 1]
                    pop_routes[p_idx, r_cnt - 1].fill_(depot)
                    pop_lengths[p_idx, r_cnt - 1] = 2
                    pop_route_counts[p_idx] = r_cnt - 1
                    r_cnt -= 1

    c_feas, c_costs, c_v_cnts, c_dists = backend.evaluate_population_tensor(
        pop_routes, pop_lengths, pop_route_counts
    )
    pop_mask = torch.zeros(P, dtype=torch.bool, device=device)
    pop_mask[active_indices] = True
    feas[pop_mask] = c_feas[pop_mask]
    costs[pop_mask] = c_costs[pop_mask]
    v_counts[pop_mask] = c_v_cnts[pop_mask]
    total_dists[pop_mask] = c_dists[pop_mask]

    return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists


def tensor_gpu_ruin_and_recreate(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    feas: torch.Tensor,
    costs: torch.Tensor,
    v_counts: torch.Tensor,
    total_dists: torch.Tensor,
    target_mask: torch.Tensor,
    backend,
    data,
    removal_fraction: float = 0.30
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Tensorized Stagnation Diversification (Ruin & Recreate).
    Removes a fraction of customers and re-inserts them via vectorized minimum-detour
    insertion on GPU VRAM, followed by 2-opt trimming.
    """
    P, R, L = pop_routes.shape
    device = backend.device
    dist_t = backend.dist_t
    depot = data.DC

    target_indices = torch.nonzero(target_mask, as_tuple=True)[0]
    if len(target_indices) == 0:
        return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists

    cand_routes = pop_routes.clone()
    cand_lengths = pop_lengths.clone()
    cand_counts = pop_route_counts.clone()
    cand_modified = torch.zeros(P, dtype=torch.bool, device=device)

    for p in target_indices:
        p_idx = int(p.item())
        r_cnt = int(cand_counts[p_idx].item())
        all_custs = []
        for r in range(r_cnt):
            r_len = int(cand_lengths[p_idx, r].item())
            if r_len > 2:
                all_custs.extend(cand_routes[p_idx, r, 1:r_len-1].tolist())

        if len(all_custs) < 6:
            continue

        rem_cnt = max(2, int(len(all_custs) * removal_fraction))
        random.shuffle(all_custs)
        unrouted = all_custs[:rem_cnt]
        unrouted_set = set(unrouted)
        unrouted_tensor = torch.tensor(unrouted, dtype=torch.long, device=device)

        # Ruin: filter out unrouted customers
        active_r = 0
        for r in range(r_cnt):
            r_len = int(cand_lengths[p_idx, r].item())
            nodes = cand_routes[p_idx, r, :r_len]
            keep_mask = ~torch.isin(nodes[1:-1], unrouted_tensor)
            kept = nodes[1:-1][keep_mask]
            if len(kept) > 0:
                new_len = len(kept) + 2
                cand_routes[p_idx, active_r].fill_(depot)
                cand_routes[p_idx, active_r, 1:new_len-1] = kept
                cand_lengths[p_idx, active_r] = new_len
                active_r += 1

        for r in range(active_r, R):
            cand_routes[p_idx, r].fill_(depot)
            cand_lengths[p_idx, r] = 2
        cand_counts[p_idx] = max(active_r, 1)

        # Recreate: vectorized greedy insertion for unrouted customers
        for c in unrouted:
            best_r = -1
            best_pos = -1
            best_delta = float('inf')
            curr_cnt = int(cand_counts[p_idx].item())

            for r in range(curr_cnt):
                r_len = int(cand_lengths[p_idx, r].item())
                if r_len >= L - 1:
                    continue
                r_nodes = cand_routes[p_idx, r, :r_len]
                prev = r_nodes[:-1]
                nxt = r_nodes[1:]
                deltas = dist_t[prev, c] + dist_t[c, nxt] - dist_t[prev, nxt]
                min_d, min_p = torch.min(deltas, dim=0)
                if min_d.item() < best_delta:
                    best_delta = min_d.item()
                    best_r = r
                    best_pos = int(min_p.item()) + 1

            if best_r != -1:
                r_len = int(cand_lengths[p_idx, best_r].item())
                cand_routes[p_idx, best_r, best_pos+1:r_len+1] = cand_routes[p_idx, best_r, best_pos:r_len].clone()
                cand_routes[p_idx, best_r, best_pos] = c
                cand_lengths[p_idx, best_r] = r_len + 1
            elif curr_cnt < R:
                cand_routes[p_idx, curr_cnt].fill_(depot)
                cand_routes[p_idx, curr_cnt, 1] = c
                cand_lengths[p_idx, curr_cnt] = 3
                cand_counts[p_idx] = curr_cnt + 1

        cand_modified[p_idx] = True

    if not cand_modified.any():
        return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists

    c_feas, c_costs, c_v_cnts, c_dists = backend.evaluate_population_tensor(
        cand_routes, cand_lengths, cand_counts
    )

    accepted = target_mask & cand_modified & c_feas
    if accepted.any():
        pop_routes[accepted] = cand_routes[accepted]
        pop_lengths[accepted] = cand_lengths[accepted]
        pop_route_counts[accepted] = cand_counts[accepted]
        feas[accepted] = c_feas[accepted]
        costs[accepted] = c_costs[accepted]
        v_counts[accepted] = c_v_cnts[accepted]
        total_dists[accepted] = c_dists[accepted]

        # Immediate fast 2-opt pass on accepted solutions
        pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists = tensor_gpu_intra_2opt(
            pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists,
            accepted, backend, data, max_passes=2
        )

    return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists


def tensor_gpu_neighborhood_moves(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    p_woa_mask: torch.Tensor,
    backend,
    data
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Tensorized Multi-Operator Local Search & Mutation (TGA).
    Applies Intelligent Intra-2opt, Intra-swap, Inter-relocate (Best Insertion), Inter-swap
    directly on GPU VRAM tensors.
    """
    P, R, L = pop_routes.shape
    device = backend.device
    dist_t = backend.dist_t
    depot = data.DC

    cand_routes = pop_routes.clone()
    cand_lengths = pop_lengths.clone()
    cand_counts = pop_route_counts.clone()

    active_p = torch.nonzero(p_woa_mask, as_tuple=True)[0]
    if len(active_p) == 0:
        return cand_routes, cand_lengths, cand_counts

    for p in active_p:
        r_cnt = int(cand_counts[p].item())
        if r_cnt == 0:
            continue

        move_type = torch.randint(0, 5, (1,), device=device).item()

        if move_type == 0:
            # 1. Intra-route 2-Opt via Vectorized Delta on GPU
            r = torch.randint(0, r_cnt, (1,), device=device).item()
            r_len = int(cand_lengths[p, r].item())
            if r_len >= 5:
                nodes = cand_routes[p, r, :r_len]
                N = r_len - 2
                prev_i = nodes[0:N]
                curr_i = nodes[1:N+1]
                curr_j = nodes[1:N+1]
                next_j = nodes[2:N+2]

                A = prev_i.unsqueeze(1)
                B = curr_i.unsqueeze(1)
                C = curr_j.unsqueeze(0)
                D = next_j.unsqueeze(0)

                delta_mat = dist_t[A, C] + dist_t[B, D] - (dist_t[A, B] + dist_t[C, D])
                u_idx = torch.arange(N, device=device).unsqueeze(1)
                v_idx = torch.arange(N, device=device).unsqueeze(0)
                delta_mat = torch.where(v_idx > u_idx, delta_mat, torch.tensor(float('inf'), device=device))

                min_val, min_flat = torch.min(delta_mat.view(-1), dim=0)
                if min_val.item() < -1e-4:
                    i1 = int((min_flat // N).item()) + 1
                    i2 = int((min_flat % N).item()) + 1
                    sub = cand_routes[p, r, i1:i2+1].clone()
                    cand_routes[p, r, i1:i2+1] = torch.flip(sub, dims=[0])
                else:
                    # Perturbation flip if no negative 2-opt found
                    i1 = torch.randint(1, r_len - 3, (1,), device=device).item()
                    i2 = torch.randint(i1 + 1, r_len - 1, (1,), device=device).item()
                    sub = cand_routes[p, r, i1:i2+1].clone()
                    cand_routes[p, r, i1:i2+1] = torch.flip(sub, dims=[0])

        elif move_type == 1:
            # 2. Intra-route Swap (Swap 2 nodes on GPU)
            r = torch.randint(0, r_cnt, (1,), device=device).item()
            r_len = int(cand_lengths[p, r].item())
            if r_len >= 4:
                i1 = torch.randint(1, r_len - 1, (1,), device=device).item()
                i2 = torch.randint(1, r_len - 1, (1,), device=device).item()
                if i1 != i2:
                    val1 = cand_routes[p, r, i1].clone()
                    cand_routes[p, r, i1] = cand_routes[p, r, i2]
                    cand_routes[p, r, i2] = val1

        elif move_type == 2:
            # 3. Inter-route Relocate with Best-Insertion Slot
            if r_cnt >= 2:
                r_src = torch.randint(0, r_cnt, (1,), device=device).item()
                r_dst = torch.randint(0, r_cnt, (1,), device=device).item()
                while r_src == r_dst:
                    r_dst = torch.randint(0, r_cnt, (1,), device=device).item()

                len_src = int(cand_lengths[p, r_src].item())
                len_dst = int(cand_lengths[p, r_dst].item())

                if len_src >= 3 and len_dst < L - 1:
                    pos_src = torch.randint(1, len_src - 1, (1,), device=device).item()
                    cust = cand_routes[p, r_src, pos_src].item()

                    # Find minimum detour position in dst via tensor
                    r_nodes = cand_routes[p, r_dst, :len_dst]
                    prev = r_nodes[:-1]
                    nxt = r_nodes[1:]
                    deltas = dist_t[prev, cust] + dist_t[cust, nxt] - dist_t[prev, nxt]
                    pos_dst = int(torch.argmin(deltas).item()) + 1

                    # Shift dst right & insert
                    cand_routes[p, r_dst, pos_dst+1:len_dst+1] = cand_routes[p, r_dst, pos_dst:len_dst].clone()
                    cand_routes[p, r_dst, pos_dst] = cust
                    cand_lengths[p, r_dst] = len_dst + 1

                    # Remove from src
                    cand_routes[p, r_src, pos_src:len_src-1] = cand_routes[p, r_src, pos_src+1:len_src].clone()
                    cand_routes[p, r_src, len_src-1] = depot
                    cand_lengths[p, r_src] = len_src - 1

                    # If r_src became empty (len <= 2), eliminate route!
                    if cand_lengths[p, r_src] <= 2:
                        for r_shift in range(r_src, r_cnt - 1):
                            cand_routes[p, r_shift] = cand_routes[p, r_shift + 1].clone()
                            cand_lengths[p, r_shift] = cand_lengths[p, r_shift + 1]
                        cand_routes[p, r_cnt - 1].fill_(depot)
                        cand_lengths[p, r_cnt - 1] = 2
                        cand_counts[p] = r_cnt - 1

        elif move_type == 3:
            # 4. Inter-route Swap
            if r_cnt >= 2:
                r1 = torch.randint(0, r_cnt, (1,), device=device).item()
                r2 = torch.randint(0, r_cnt, (1,), device=device).item()
                while r1 == r2:
                    r2 = torch.randint(0, r_cnt, (1,), device=device).item()
                len1 = int(cand_lengths[p, r1].item())
                len2 = int(cand_lengths[p, r2].item())
                if len1 >= 3 and len2 >= 3:
                    p1 = torch.randint(1, len1 - 1, (1,), device=device).item()
                    p2 = torch.randint(1, len2 - 1, (1,), device=device).item()
                    v1 = cand_routes[p, r1, p1].clone()
                    v2 = cand_routes[p, r2, p2].clone()
                    cand_routes[p, r1, p1] = v2
                    cand_routes[p, r2, p2] = v1

        elif move_type == 4:
            # 5. Intra-route Relocate with Best-Insertion Slot
            r = torch.randint(0, r_cnt, (1,), device=device).item()
            r_len = int(cand_lengths[p, r].item())
            if r_len >= 5:
                pos_src = torch.randint(1, r_len - 1, (1,), device=device).item()
                cust = cand_routes[p, r, pos_src].item()

                # Remove cust from r
                tmp = torch.cat([cand_routes[p, r, :pos_src], cand_routes[p, r, pos_src+1:r_len]])
                prev = tmp[:-1]
                nxt = tmp[1:]
                deltas = dist_t[prev, cust] + dist_t[cust, nxt] - dist_t[prev, nxt]
                pos_dst = int(torch.argmin(deltas).item()) + 1

                if pos_src < pos_dst:
                    cand_routes[p, r, pos_src:pos_dst] = cand_routes[p, r, pos_src+1:pos_dst+1].clone()
                else:
                    cand_routes[p, r, pos_dst+1:pos_src+1] = cand_routes[p, r, pos_dst:pos_src].clone()
                cand_routes[p, r, pos_dst] = cust

    return cand_routes, cand_lengths, cand_counts


def tensor_guided_crossover(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    best_indices,
    peer_indices,
    p_hybrid_mask: torch.Tensor,
    backend,
    data
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """100% Pure GPU Tensorized Crossover alias."""
    return tensor_gpu_elite_crossover(
        pop_routes, pop_lengths, pop_route_counts, best_indices, p_hybrid_mask, backend, data
    )


def tensor_woa_intensification(
    cand_routes: torch.Tensor,
    cand_lengths: torch.Tensor,
    cand_counts: torch.Tensor,
    best_indices,
    a_param: float,
    p_woa_mask: torch.Tensor,
    backend,
    data
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """100% Pure GPU Tensorized Multi-Operator Mutation alias."""
    return tensor_gpu_neighborhood_moves(
        cand_routes, cand_lengths, cand_counts, p_woa_mask, backend, data
    )
