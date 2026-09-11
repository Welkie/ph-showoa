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
    sa_iters: int = 25
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

    return pop_routes, pop_lengths, pop_route_counts


def tensor_sa_warmup(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    backend,
    data,
    sa_iters: int = 50,
    temp_init: float = 100.0,
    cooling: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Parallel GPU Simulated Annealing warm-up with inter-route moves (Pd-Shift & Pd-Exchange):
    - Move 1: Intra-route Swap
    - Move 2: Intra-route Insert
    - Move 3: Intra-route 2-opt (Reverse)
    - Move 4: Inter-route Pd-Shift (moves customer from r1 to r2; ELIMINATES route if singleton!)
    - Move 5: Inter-route Pd-Exchange (swaps customers between r1 and r2)
    Vectorized Metropolis acceptance across all P solutions in batch.
    """
    P, R, L = pop_routes.shape
    device = backend.device
    depot = data.DC

    feas, costs, v_cnts, dists = backend.evaluate_population_tensor(pop_routes, pop_lengths, pop_route_counts)

    temp = temp_init
    # Scale iterations by problem size
    cooling_steps = max(sa_iters, 40)

    for _ in range(cooling_steps):
        cand_routes = pop_routes.clone()
        cand_lengths = pop_lengths.clone()
        cand_counts = pop_route_counts.clone()

        for p in range(P):
            r_cnt = int(cand_counts[p].item())
            if r_cnt == 0:
                continue

            move_type = random.randint(1, 5)

            if move_type == 1:  # Intra-route Swap
                r_idx = random.randint(0, r_cnt - 1)
                r_len = int(cand_lengths[p, r_idx].item())
                if r_len >= 4:
                    i1, i2 = random.randint(1, r_len - 2), random.randint(1, r_len - 2)
                    if i1 != i2:
                        val1 = cand_routes[p, r_idx, i1].item()
                        val2 = cand_routes[p, r_idx, i2].item()
                        cand_routes[p, r_idx, i1] = val2
                        cand_routes[p, r_idx, i2] = val1

            elif move_type == 2:  # Intra-route Insert
                r_idx = random.randint(0, r_cnt - 1)
                r_len = int(cand_lengths[p, r_idx].item())
                if r_len >= 4:
                    i1, i2 = random.randint(1, r_len - 2), random.randint(1, r_len - 2)
                    if i1 != i2:
                        cust = cand_routes[p, r_idx, i1].item()
                        nodes = cand_routes[p, r_idx, :r_len].cpu().tolist()
                        nodes.pop(i1)
                        nodes.insert(i2, cust)
                        cand_routes[p, r_idx, :r_len] = torch.tensor(nodes, dtype=torch.long, device=device)

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

            elif move_type == 4:  # Inter-route Pd-Shift (Route Elimination)
                if r_cnt >= 2:
                    r1 = random.randint(0, r_cnt - 1)
                    r2 = random.randint(0, r_cnt - 1)
                    while r1 == r2:
                        r2 = random.randint(0, r_cnt - 1)

                    len1 = int(cand_lengths[p, r1].item())
                    len2 = int(cand_lengths[p, r2].item())

                    if len1 >= 3 and len2 < L - 1:
                        i1 = random.randint(1, len1 - 2)
                        cust = cand_routes[p, r1, i1].item()
                        i2 = random.randint(1, len2 - 1)

                        # Remove from r1
                        nodes1 = cand_routes[p, r1, :len1].cpu().tolist()
                        nodes1.pop(i1)

                        # Insert into r2
                        nodes2 = cand_routes[p, r2, :len2].cpu().tolist()
                        nodes2.insert(i2, cust)

                        len1_new = len(nodes1)
                        len2_new = len(nodes2)

                        cand_lengths[p, r1] = len1_new
                        cand_routes[p, r1, :len1_new] = torch.tensor(nodes1, dtype=torch.long, device=device)
                        cand_routes[p, r1, len1_new:] = depot

                        cand_lengths[p, r2] = len2_new
                        cand_routes[p, r2, :len2_new] = torch.tensor(nodes2, dtype=torch.long, device=device)

                        # If r1 is now empty [depot, depot], compact routes and eliminate vehicle!
                        if len1_new == 2:
                            cand_routes[p, r1:r_cnt-1] = cand_routes[p, r1+1:r_cnt].clone()
                            cand_lengths[p, r1:r_cnt-1] = cand_lengths[p, r1+1:r_cnt].clone()
                            cand_routes[p, r_cnt-1].fill_(depot)
                            cand_lengths[p, r_cnt-1] = 2
                            cand_counts[p] = r_cnt - 1

            elif move_type == 5:  # Inter-route Pd-Exchange (Swap between routes)
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
                        val1 = cand_routes[p, r1, i1].item()
                        val2 = cand_routes[p, r2, i2].item()
                        cand_routes[p, r1, i1] = val2
                        cand_routes[p, r2, i2] = val1

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
    100% Pure GPU Tensorized Elite Route Crossover.
    Operates strictly on GPU VRAM tensors without any CPU list conversions.
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
        if best_r_cnt == 0:
            continue

        elite_r = torch.randint(0, best_r_cnt, (1,), device=device).item()
        elite_len = int(pop_lengths[best_p, elite_r].item())
        if elite_len <= 2:
            continue

        elite_custs = pop_routes[best_p, elite_r, 1:elite_len-1]

        # In individual p, remove any customer in elite_custs
        curr_r_cnt = int(cand_counts[p].item())
        active_r = 0
        for r in range(curr_r_cnt):
            r_len = int(cand_lengths[p, r].item())
            nodes = cand_routes[p, r, :r_len]
            keep_mask = ~torch.isin(nodes[1:-1], elite_custs)
            kept = nodes[1:-1][keep_mask]
            if len(kept) > 0:
                new_len = len(kept) + 2
                cand_routes[p, active_r].fill_(depot)
                cand_routes[p, active_r, 1:new_len-1] = kept
                cand_lengths[p, active_r] = new_len
                active_r += 1

        for r in range(active_r, R):
            cand_routes[p, r].fill_(depot)
            cand_lengths[p, r] = 2

        if active_r < R:
            cand_routes[p, active_r, :elite_len] = pop_routes[best_p, elite_r, :elite_len]
            cand_lengths[p, active_r] = elite_len
            cand_counts[p] = active_r + 1
        else:
            cand_counts[p] = active_r

    return cand_routes, cand_lengths, cand_counts


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
    Applies Intra-2opt, Intra-swap, Inter-relocate (Vehicle Elimination), Inter-swap
    directly on GPU VRAM tensors.
    """
    P, R, L = pop_routes.shape
    device = backend.device
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
            # 1. Intra-route 2-Opt (Subsegment Reverse on GPU)
            r = torch.randint(0, r_cnt, (1,), device=device).item()
            r_len = int(cand_lengths[p, r].item())
            if r_len >= 5:
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
            # 3. Inter-route Relocate (Vehicle Elimination)
            if r_cnt >= 2:
                r_src = torch.randint(0, r_cnt, (1,), device=device).item()
                r_dst = torch.randint(0, r_cnt, (1,), device=device).item()
                while r_src == r_dst:
                    r_dst = torch.randint(0, r_cnt, (1,), device=device).item()

                len_src = int(cand_lengths[p, r_src].item())
                len_dst = int(cand_lengths[p, r_dst].item())

                if len_src >= 3 and len_dst < L - 1:
                    pos_src = torch.randint(1, len_src - 1, (1,), device=device).item()
                    cust = cand_routes[p, r_src, pos_src].clone()
                    pos_dst = torch.randint(1, len_dst, (1,), device=device).item()

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
            # 5. Intra-route Relocate (Single customer relocation within route)
            r = torch.randint(0, r_cnt, (1,), device=device).item()
            r_len = int(cand_lengths[p, r].item())
            if r_len >= 4:
                pos_src = torch.randint(1, r_len - 1, (1,), device=device).item()
                pos_dst = torch.randint(1, r_len - 1, (1,), device=device).item()
                if pos_src != pos_dst:
                    cust = cand_routes[p, r, pos_src].clone()
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
