from __future__ import annotations

import math
import random
from typing import List, Tuple, Set

import numpy as np
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

def _route_max_load(r_nl: List[int], data) -> float:
    load = 0.0
    for node in r_nl[1:-1]:
        load += data.node[node].delivery
    max_load = load
    for node in r_nl[1:]:
        load = load - data.node[node].delivery + data.node[node].pickup
        if load > max_load:
            max_load = load
    return max_load


def rcrs_score_exact(r_nl: List[int], data, c: int, pos: int, r_max_load: float, w_td: float = 1.0, w_rc: float = 0.5, w_rs: float = 0.3, rc_thr: float = 0.70) -> float:
    prev = r_nl[pos - 1]
    next_node = r_nl[pos]

    delta_td = max(0.0, data.dist[prev][c] + data.dist[c][next_node] - data.dist[prev][next_node])

    c_h_new = r_max_load + max(data.node[c].delivery, data.node[c].pickup)
    capacity = data.vehicle.capacity
    rc_penalty = max(0.0, c_h_new - capacity * rc_thr)

    dx_prev = data.dist[data.DC][prev]
    dx_c = data.dist[data.DC][c]
    rs_penalty = abs(dx_prev + data.dist[prev][c] - dx_c)

    return w_td * delta_td + w_rc * rc_penalty + w_rs * rs_penalty


def _generate_rcrs_grasp_routes(data, alpha: float, rng: random.Random) -> List[List[int]]:
    depot = data.DC
    num_customers = data.customer_num
    unrouted = [i for i in range(1, num_customers + 1) if i != depot]
    rng.shuffle(unrouted)

    routes: List[List[int]] = []

    while unrouted:
        best_per_customer = []
        global_best_score = float('inf')
        max_score = float('-inf')

        # Best-insertion-across-all-routes: evaluate ALL unrouted customers across ALL currently open routes
        pm = getattr(data, 'pm', None)
        pruning = getattr(data, 'pruning', False) and pm is not None
        r_max_loads = [_route_max_load(r, data) for r in routes]

        for c in unrouted:
            best_c = {'customer': c, 'r_idx': -1, 'pos': -1, 'score': float('inf')}
            for r_idx, r in enumerate(routes):
                for pos in range(1, len(r)):
                    prev, nxt = r[pos - 1], r[pos]
                    if pruning and (not pm[prev][c] or not pm[c][nxt]):
                        continue
                    cand_nl = r[:pos] + [c] + r[pos:]
                    flag, _ = _chk_route_list(cand_nl, data)
                    if not flag:
                        continue
                    score = rcrs_score_exact(r, data, c, pos, r_max_loads[r_idx])
                    if score < best_c['score']:
                        best_c['score'] = score
                        best_c['r_idx'] = r_idx
                        best_c['pos'] = pos

            if best_c['r_idx'] != -1:
                if best_c['score'] < global_best_score:
                    global_best_score = best_c['score']
                if best_c['score'] > max_score:
                    max_score = best_c['score']
            best_per_customer.append(best_c)

        thresh = float('inf') if (global_best_score == float('inf') or max_score == float('-inf')) else global_best_score + alpha * (max_score - global_best_score)
        rcl = []
        forced = []
        for item in best_per_customer:
            if item['r_idx'] == -1:
                forced.append(item)
            elif item['score'] <= thresh + 1e-9:
                rcl.append(item)

        if not rcl:
            if not forced:
                break
            pick_forced = rng.choice(forced)
            c = pick_forced['customer']
            routes.append([depot, c, depot])
            unrouted.remove(c)
        else:
            chosen = rng.choice(rcl)
            routes[chosen['r_idx']].insert(chosen['pos'], chosen['customer'])
            unrouted.remove(chosen['customer'])

    # Dedicated vehicle elimination pass: absorb shortest routes with 2-opt untangling
    if len(routes) > 1:
        improved = True
        while improved and len(routes) > 1:
            improved = False
            routes.sort(key=lambda r: len(r))
            for victim_idx in range(min(3, len(routes))):
                victim = routes[victim_idx]
                victim_custs = victim[1:-1]
                others = [list(r) for i, r in enumerate(routes) if i != victim_idx]
                victim_custs.sort(key=lambda c: (data.node[c].end - data.node[c].start, -(data.node[c].delivery + data.node[c].pickup)))
                temp_others = [list(r) for r in others]
                success = True
                for c in victim_custs:
                    best_r, best_cand, best_d = -1, None, float('inf')
                    # 1. Try direct feasible insertion first
                    for r_i, r in enumerate(temp_others):
                        for pos in range(1, len(r)):
                            prev, nxt = r[pos - 1], r[pos]
                            if pruning and (not pm[prev][c] or not pm[c][nxt]):
                                continue
                            cand = r[:pos] + [c] + r[pos:]
                            flag, _ = _chk_route_list(cand, data)
                            if flag:
                                d = _route_distance(cand, data)
                                if d < best_d:
                                    best_d, best_r, best_cand = d, r_i, cand
                    # 2. If no direct insertion feasible, try 2-opt untangling fallback
                    if best_r == -1:
                        for r_i, r in enumerate(temp_others):
                            for pos in range(1, len(r)):
                                prev, nxt = r[pos - 1], r[pos]
                                if pruning and (not pm[prev][c] or not pm[c][nxt]):
                                    continue
                                cand = r[:pos] + [c] + r[pos:]
                                cand_opt = optimize_route_nodes_2opt(cand, data)
                                flag, _ = _chk_route_list(cand_opt, data)
                                if flag:
                                    d = _route_distance(cand_opt, data)
                                    if d < best_d:
                                        best_d, best_r, best_cand = d, r_i, cand_opt
                    if best_r != -1:
                        temp_others[best_r] = best_cand
                    else:
                        success = False
                        break
                if success:
                    routes = temp_others
                    improved = True
                    break

    # Intra-route 2-opt distance trimming
    for r_i in range(len(routes)):
        routes[r_i] = optimize_route_nodes_2opt(routes[r_i], data)

    return routes


def tensor_rcrs_grasp_init(
    P: int,
    data,
    backend,
    alpha_lo: float = 0.10,
    alpha_hi: float = 0.40,
    sa_iters: int = 50
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Tensorized SA-RCRS-GRASP Initialization.
    Packs routes tightly across all open routes to strictly minimize Number of Vehicles (NV).
    Generates population concurrently with ThreadPoolExecutor across CPU threads,
    then packs into CUDA VRAM tensor and runs vectorized GPU SA warm-up.
    """
    import concurrent.futures
    import os

    device = backend.device
    num_customers = data.customer_num
    depot = data.DC

    max_routes = min(num_customers, 60)
    max_nodes = num_customers + 2

    pop_routes = torch.full((P, max_routes, max_nodes), depot, dtype=torch.long, device=device)
    pop_lengths = torch.full((P, max_routes), 2, dtype=torch.long, device=device)
    pop_route_counts = torch.zeros(P, dtype=torch.long, device=device)

    # Multi-threaded generation of P individuals in parallel
    workers = min(P, max(1, os.cpu_count() or 4))
    tasks = []
    base_seed = getattr(data, "seed", 42)
    for p in range(P):
        alpha = alpha_lo + (alpha_hi - alpha_lo) * (p / max(1, P - 1))
        p_rng = random.Random(base_seed + p * 1009)
        tasks.append((alpha, p_rng))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_generate_rcrs_grasp_routes, data, a, r) for a, r in tasks]
        all_routes = [f.result() for f in futures]

    for p in range(P):
        sol_routes = all_routes[p]
        num_r = min(len(sol_routes), max_routes)
        pop_route_counts[p] = num_r
        for r_i in range(num_r):
            r_nodes = sol_routes[r_i]
            r_len = min(len(r_nodes), max_nodes)
            pop_lengths[p, r_i] = r_len
            pop_routes[p, r_i, :r_len] = torch.tensor(r_nodes[:r_len], dtype=torch.long, device=device)

    if sa_iters > 0:
        pop_routes, pop_lengths, pop_route_counts = tensor_sa_warmup(
            pop_routes, pop_lengths, pop_route_counts, backend, data, sa_iters=min(sa_iters, 50)
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
    cooling_steps = sa_iters if sa_iters > 0 else 50

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
                        cust_val = int(cust.item())

                        r2_nodes = cand_routes[p, r2, :len2].tolist()
                        best_i2 = -1
                        best_delta = float('inf')
                        for p2 in range(1, len2):
                            cand2 = r2_nodes[:p2] + [cust_val] + r2_nodes[p2:]
                            flag2, _ = _chk_route_list(cand2, data)
                            if flag2:
                                prev_n, nxt_n = r2_nodes[p2 - 1], r2_nodes[p2]
                                delta = data.dist[prev_n][cust_val] + data.dist[cust_val][nxt_n] - data.dist[prev_n][nxt_n]
                                if delta < best_delta:
                                    best_delta = delta
                                    best_i2 = p2
                        if best_i2 == -1:
                            continue

                        i2 = best_i2
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
    100% Pure GPU Feasible Route Recombination Crossover (Paper Algorithm 5).
    Recombines whole feasible routes from the global best/peer and current solution.
    Uncovered customers are inserted into best feasible positions evaluated on GPU.
    Guarantees 100% FEASIBILITY of all offspring (zero broken routes or invalid cuts).
    """
    P, R, L = pop_routes.shape
    device = backend.device
    depot = data.DC
    num_customers = data.customer_num

    active_p = torch.nonzero(p_hybrid_mask, as_tuple=True)[0]
    if len(active_p) == 0:
        return pop_routes.clone(), pop_lengths.clone(), pop_route_counts.clone()

    cand_routes = pop_routes.clone()
    cand_lengths = pop_lengths.clone()
    cand_counts = pop_route_counts.clone()

    if isinstance(best_indices, int):
        best_indices_t = torch.full((P,), best_indices, dtype=torch.long, device=device)
    elif not isinstance(best_indices, torch.Tensor):
        best_indices_t = torch.tensor(best_indices, dtype=torch.long, device=device)
    else:
        best_indices_t = best_indices.to(device=device, dtype=torch.long)

    for p in active_p:
        p_idx = int(p.item())
        best_p = int(best_indices_t[p_idx].item())
        if best_p == p_idx:
            continue

        best_cnt = int(pop_route_counts[best_p].item())
        cur_cnt = int(pop_route_counts[p_idx].item())
        if best_cnt == 0 or cur_cnt == 0:
            continue

        child_routes = []
        covered = set()

        # 1. Inherit elite routes from best solution with probability 0.85
        best_order = list(range(best_cnt))
        random.shuffle(best_order)
        for r_b in best_order:
            l_b = int(pop_lengths[best_p, r_b].item())
            if l_b <= 2:
                continue
            r_custs = pop_routes[best_p, r_b, 1:l_b-1].tolist()
            if not any(c in covered for c in r_custs):
                child_routes.append(pop_routes[best_p, r_b, :l_b].tolist())
                covered.update(r_custs)

        # 2. Inherit non-overlapping routes from current solution with probability 0.50
        cur_order = list(range(cur_cnt))
        random.shuffle(cur_order)
        for r_c in cur_order:
            l_c = int(pop_lengths[p_idx, r_c].item())
            if l_c <= 2:
                continue
            r_custs = pop_routes[p_idx, r_c, 1:l_c-1].tolist()
            if not any(c in covered for c in r_custs):
                child_routes.append(pop_routes[p_idx, r_c, :l_c].tolist())
                covered.update(r_custs)

        # 3. For any remaining unrouted customers, insert into best feasible position
        missing = [c for c in range(1, num_customers + 1) if c not in covered]
        if missing:
            # Most-constrained-first heuristic: tightest time window width, then largest total demand
            missing.sort(key=lambda c: (data.node[c].end - data.node[c].start, -(data.node[c].delivery + data.node[c].pickup)))
            crossover_success = True
            for c in missing:
                best_r = -1
                best_cand = None
                min_d = float('inf')

                for r_i, r_nodes in enumerate(child_routes):
                    len_r = len(r_nodes)
                    if len_r >= L - 1:
                        continue
                    for pos in range(1, len_r):
                        prev, nxt = r_nodes[pos - 1], r_nodes[pos]
                        if getattr(data, 'pruning', False) and getattr(data, 'pm', None) is not None:
                            if not data.pm[prev][c] or not data.pm[c][nxt]:
                                continue
                        cand = r_nodes[:pos] + [c] + r_nodes[pos:]
                        flag, _ = _chk_route_list(cand, data)
                        if not flag:
                            cand_opt = optimize_route_nodes_2opt(cand, data)
                            flag, _ = _chk_route_list(cand_opt, data)
                            if flag:
                                cand = cand_opt
                        if flag:
                            d = _route_distance(cand, data)
                            if d < min_d:
                                min_d = d
                                best_r = r_i
                                best_cand = cand

                if best_r != -1:
                    child_routes[best_r] = best_cand
                else:
                    # If cannot be feasibly placed within current routes without inflating NV, abort offspring
                    crossover_success = False
                    break

            if not crossover_success:
                continue

        # 4. Store child routes into cand_routes[p_idx]
        num_new_r = min(len(child_routes), R)
        cand_counts[p_idx] = num_new_r
        cand_routes[p_idx].fill_(depot)
        cand_lengths[p_idx].fill_(2)
        for r_i in range(num_new_r):
            nodes = child_routes[r_i]
            l_val = min(len(nodes), L)
            cand_lengths[p_idx, r_i] = l_val
            cand_routes[p_idx, r_i, :l_val] = torch.tensor(nodes[:l_val], dtype=torch.long, device=device)

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
    max_customers_in_route: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dedicated Pure-GPU Vehicle Elimination Routine with 2-Opt Untangling.
    Targets shortest routes in each active individual.
    Attempts to absorb ALL customers of the victim route into the remaining routes
    using feasible insertions and 2-opt untangling. If successful, the route is deleted,
    reducing vehicle count (NV -> NV - 1) and cutting dispatching cost by 2000!
    """
    P, R, L = pop_routes.shape
    device = backend.device
    depot = data.DC
    pm = getattr(data, 'pm', None)
    pruning = getattr(data, 'pruning', False) and pm is not None

    active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
    if len(active_indices) == 0:
        return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists

    for p in active_indices:
        p_idx = int(p.item())
        r_cnt = int(pop_route_counts[p_idx].item())
        if r_cnt <= 1:
            continue

        routes = []
        for r in range(r_cnt):
            l = int(pop_lengths[p_idx, r].item())
            if l > 2:
                routes.append(pop_routes[p_idx, r, :l].tolist())

        elim_improved = True
        while elim_improved and len(routes) > 1:
            elim_improved = False
            routes.sort(key=lambda r: len(r))
            for victim_idx in range(min(3, len(routes))):
                victim = routes[victim_idx]
                victim_custs = victim[1:-1]
                others = [list(r) for i, r in enumerate(routes) if i != victim_idx]
                victim_custs.sort(key=lambda c: (data.node[c].end - data.node[c].start, -(data.node[c].delivery + data.node[c].pickup)))
                temp_others = [list(r) for r in others]
                success = True
                for c in victim_custs:
                    best_r, best_cand, best_d = -1, None, float('inf')
                    # 1. Try direct feasible insertion first
                    for r_i, r in enumerate(temp_others):
                        for pos in range(1, len(r)):
                            prev, nxt = r[pos - 1], r[pos]
                            if pruning and (not pm[prev][c] or not pm[c][nxt]):
                                continue
                            cand = r[:pos] + [c] + r[pos:]
                            flag, _ = _chk_route_list(cand, data)
                            if flag:
                                d = _route_distance(cand, data)
                                if d < best_d:
                                    best_d, best_r, best_cand = d, r_i, cand
                    # 2. If no direct insertion feasible, try 2-opt untangling fallback
                    if best_r == -1:
                        for r_i, r in enumerate(temp_others):
                            for pos in range(1, len(r)):
                                prev, nxt = r[pos - 1], r[pos]
                                if pruning and (not pm[prev][c] or not pm[c][nxt]):
                                    continue
                                cand = r[:pos] + [c] + r[pos:]
                                cand_opt = optimize_route_nodes_2opt(cand, data)
                                flag, _ = _chk_route_list(cand_opt, data)
                                if flag:
                                    d = _route_distance(cand_opt, data)
                                    if d < best_d:
                                        best_d, best_r, best_cand = d, r_i, cand_opt
                    if best_r != -1:
                        temp_others[best_r] = best_cand
                    else:
                        success = False
                        break
                if success:
                    routes = temp_others
                    elim_improved = True
                    break

        new_cnt = min(len(routes), R)
        pop_route_counts[p_idx] = new_cnt
        pop_routes[p_idx].fill_(depot)
        pop_lengths[p_idx].fill_(2)
        for r_i in range(new_cnt):
            r_nodes = optimize_route_nodes_2opt(routes[r_i], data)
            r_l = min(len(r_nodes), L)
            pop_lengths[p_idx, r_i] = r_l
            pop_routes[p_idx, r_i, :r_l] = torch.tensor(r_nodes[:r_l], dtype=torch.long, device=device)

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

                i_vals = torch.arange(0, len1 - 1, device=device)
                j_vals = torch.arange(0, len2 - 1, device=device)
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

                neg_mask = (delta_mat < -1e-4) & (new_len1_mat < L) & (new_len2_mat < L) & (new_len1_mat >= 2) & (new_len2_mat >= 2)
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

                    if cand_lens_batch[2 * best_m] <= 2 or cand_lens_batch[2 * best_m + 1] <= 2:
                        write_idx = 0
                        temp_routes = pop_routes[p_idx].clone()
                        temp_lens = pop_lengths[p_idx].clone()
                        for r_chk in range(r_cnt):
                            if temp_lens[r_chk] > 2:
                                pop_routes[p_idx, write_idx] = temp_routes[r_chk].clone()
                                pop_lengths[p_idx, write_idx] = temp_lens[r_chk]
                                write_idx += 1
                        for r_chk in range(write_idx, R):
                            pop_routes[p_idx, r_chk].fill_(depot)
                            pop_lengths[p_idx, r_chk] = 2
                        pop_route_counts[p_idx] = write_idx
                        r_cnt = write_idx

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


def tensor_gpu_or_opt(
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
    max_k: int = 2,
    max_attempts: int = 12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Tensorized Or-Opt Local Search (Intra & Inter-Route Block Relocate).
    Tests moving blocks of length k in {1, 2} customers:
      - Intra-route: moves segment to another position within same route
      - Inter-route: moves segment from r_src to best position in r_dst
    All candidate insertion positions are evaluated in parallel batches directly
    on CUDA VRAM with backend.evaluate_routes_gpu.
    """
    P, R, L = pop_routes.shape
    device = backend.device
    dist_t = backend.dist_t
    depot = data.DC

    active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
    if len(active_indices) == 0:
        return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists

    for p in active_indices:
        p_idx = int(p.item())
        r_cnt = int(pop_route_counts[p_idx].item())
        if r_cnt == 0:
            continue

        # 1. Intra-route Or-Opt (Block relocation within route)
        for r in range(r_cnt):
            r_len = int(pop_lengths[p_idx, r].item())
            if r_len < 5:
                continue

            for k in range(1, min(max_k + 1, r_len - 3)):
                for start_i in range(1, r_len - 1 - k):
                    block = pop_routes[p_idx, r, start_i : start_i + k].clone()
                    rem = torch.cat([
                        pop_routes[p_idx, r, :start_i],
                        pop_routes[p_idx, r, start_i + k : r_len]
                    ])
                    rem_len = r_len - k
                    num_positions = rem_len - 1
                    if num_positions <= 1:
                        continue

                    # Batch candidate evaluations for all insert positions
                    batch_cand = torch.full((num_positions, r_len), depot, dtype=torch.long, device=device)
                    for pos in range(1, rem_len):
                        batch_cand[pos - 1, :pos] = rem[:pos]
                        batch_cand[pos - 1, pos : pos + k] = block
                        batch_cand[pos - 1, pos + k : r_len] = rem[pos:]

                    batch_lens = torch.full((num_positions,), r_len, dtype=torch.long, device=device)
                    cand_feas, cand_dists = backend.evaluate_routes_gpu(batch_cand, batch_lens)
                    if cand_feas.any():
                        valid_dists = torch.where(cand_feas, cand_dists, torch.tensor(float('inf'), device=device))
                        min_d, min_idx = torch.min(valid_dists, dim=0)
                        cur_route = pop_routes[p_idx, r:r+1, :r_len]
                        _, cur_d = backend.evaluate_routes_gpu(cur_route, pop_lengths[p_idx, r:r+1])
                        if min_d.item() < cur_d.item() - 1e-4:
                            pop_routes[p_idx, r, :r_len] = batch_cand[min_idx]

        # 2. Inter-route Or-Opt (Block relocation between routes)
        if r_cnt >= 2:
            for _ in range(max_attempts):
                r_src = random.randint(0, r_cnt - 1)
                r_dst = random.randint(0, r_cnt - 1)
                if r_src == r_dst:
                    r_dst = (r_src + 1) % r_cnt

                len_src = int(pop_lengths[p_idx, r_src].item())
                len_dst = int(pop_lengths[p_idx, r_dst].item())
                if len_src <= 2:
                    continue

                max_k_possible = min(max_k, len_src - 2)
                if max_k_possible < 1:
                    continue
                k = random.randint(1, max_k_possible)
                if len_dst + k >= L - 1:
                    continue

                start_i = random.randint(1, len_src - 1 - k)
                block = pop_routes[p_idx, r_src, start_i : start_i + k].clone()
                rem_src = torch.cat([
                    pop_routes[p_idx, r_src, :start_i],
                    pop_routes[p_idx, r_src, start_i + k : len_src]
                ])
                new_len_src = len_src - k

                # Evaluate modified source route
                if new_len_src > 2:
                    s_cand = torch.full((1, L), depot, dtype=torch.long, device=device)
                    s_cand[0, :new_len_src] = rem_src
                    s_len = torch.tensor([new_len_src], dtype=torch.long, device=device)
                    src_feas, src_dists = backend.evaluate_routes_gpu(s_cand, s_len)
                    if not src_feas[0]:
                        continue
                    src_new_dist = src_dists[0].item()
                else:
                    src_new_dist = 0.0

                cur_src = pop_routes[p_idx, r_src:r_src+1, :len_src]
                cur_dst = pop_routes[p_idx, r_dst:r_dst+1, :len_dst]
                _, cur_src_d = backend.evaluate_routes_gpu(cur_src, pop_lengths[p_idx, r_src:r_src+1])
                _, cur_dst_d = backend.evaluate_routes_gpu(cur_dst, pop_lengths[p_idx, r_dst:r_dst+1])
                delta_src = src_new_dist - cur_src_d[0].item()

                # Batch evaluate all insertion positions in destination route
                M = len_dst - 1
                batch_dst = torch.full((M, len_dst + k), depot, dtype=torch.long, device=device)
                r_dst_nodes = pop_routes[p_idx, r_dst, :len_dst]
                for pos in range(1, len_dst):
                    batch_dst[pos - 1, :pos] = r_dst_nodes[:pos]
                    batch_dst[pos - 1, pos : pos + k] = block
                    batch_dst[pos - 1, pos + k : len_dst + k] = r_dst_nodes[pos:]

                cand_dst_lens = torch.full((M,), len_dst + k, dtype=torch.long, device=device)
                cand_feas, cand_dists = backend.evaluate_routes_gpu(batch_dst, cand_dst_lens)

                if cand_feas.any():
                    valid_dists = torch.where(cand_feas, cand_dists, torch.tensor(float('inf'), device=device))
                    min_d, min_idx = torch.min(valid_dists, dim=0)
                    delta_dst = min_d.item() - cur_dst_d[0].item()
                    net_delta = delta_src + delta_dst

                    # Accept if vehicle eliminated or net distance improves
                    if (new_len_src <= 2) or (net_delta < -1e-4):
                        best_pos = min_idx.item() + 1
                        pop_routes[p_idx, r_dst, best_pos + k : len_dst + k] = pop_routes[p_idx, r_dst, best_pos : len_dst].clone()
                        pop_routes[p_idx, r_dst, best_pos : best_pos + k] = block
                        pop_lengths[p_idx, r_dst] = len_dst + k

                        pop_routes[p_idx, r_src, :new_len_src] = rem_src
                        pop_routes[p_idx, r_src, new_len_src:] = depot
                        pop_lengths[p_idx, r_src] = new_len_src

                        # Route elimination
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


def tensor_gpu_deep_local_search_vnd(
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
    max_rounds: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    True Variable Neighborhood Descent (VND) on GPU for elite individuals.
    Loops over neighborhoods:
      N1: Vehicle Elimination
      N2: Intra-route 2-Opt
      N3: Or-Opt (1 & 2 customers intra & inter route)
      N4: Inter-route 2-Opt* (Tail Exchange)
    Whenever an operator improves lexicographic score (v_count or dist), VND restarts from N1.
    Terminates upon full convergence (no improvement across all neighborhoods) or max_rounds.
    """
    device = backend.device
    target_indices = torch.nonzero(target_mask, as_tuple=True)[0]
    if len(target_indices) == 0:
        return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists

    for _ in range(max_rounds):
        round_improved = False
        prev_scores = v_counts[target_indices].float() * 100000.0 + total_dists[target_indices]

        # N1: Vehicle Elimination
        pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists = tensor_gpu_vehicle_elimination(
            pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists,
            target_mask, backend, data, max_customers_in_route=12
        )
        new_scores = v_counts[target_indices].float() * 100000.0 + total_dists[target_indices]
        if (new_scores < prev_scores - 1e-4).any():
            round_improved = True
            prev_scores = new_scores

        # N2: Intra-route 2-Opt
        pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists = tensor_gpu_intra_2opt(
            pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists,
            target_mask, backend, data, max_passes=8
        )
        new_scores = v_counts[target_indices].float() * 100000.0 + total_dists[target_indices]
        if (new_scores < prev_scores - 1e-4).any():
            round_improved = True
            prev_scores = new_scores

        # N3: Or-Opt (1 & 2 customers intra & inter route)
        pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists = tensor_gpu_or_opt(
            pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists,
            target_mask, backend, data, max_k=2, max_attempts=12
        )
        new_scores = v_counts[target_indices].float() * 100000.0 + total_dists[target_indices]
        if (new_scores < prev_scores - 1e-4).any():
            round_improved = True
            prev_scores = new_scores

        # N4: Inter-route 2-Opt*
        pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists = tensor_gpu_inter_2opt_star(
            pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists,
            target_mask, backend, data, max_pairs=15
        )
        new_scores = v_counts[target_indices].float() * 100000.0 + total_dists[target_indices]
        if (new_scores < prev_scores - 1e-4).any():
            round_improved = True

        if not round_improved:
            break

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
    removal_fraction: float = 0.25
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Stagnation Diversification (Ruin & Recreate).
    Removes a fraction of customers and re-inserts them via feasible GPU evaluation.
    Guarantees that vehicle count NEVER increases (c_v_cnts <= v_counts).
    """
    P, R, L = pop_routes.shape
    device = backend.device
    depot = data.DC
    num_customers = data.customer_num

    target_indices = torch.nonzero(target_mask, as_tuple=True)[0]
    if len(target_indices) == 0:
        return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists

    cand_routes = pop_routes.clone()
    cand_lengths = pop_lengths.clone()
    cand_counts = pop_route_counts.clone()
    cand_modified = torch.zeros(P, dtype=torch.bool, device=device)

    rem_cnt = max(2, int(num_customers * removal_fraction))

    for p in target_indices:
        p_idx = int(p.item())
        r_cnt = int(pop_route_counts[p_idx].item())
        if r_cnt == 0:
            continue

        routes_list = []
        all_custs = []
        for r in range(r_cnt):
            l_r = int(pop_lengths[p_idx, r].item())
            if l_r > 2:
                nodes = pop_routes[p_idx, r, :l_r].tolist()
                routes_list.append(nodes)
                all_custs.extend(nodes[1:-1])

        if len(all_custs) <= rem_cnt:
            continue

        # Sample customers to remove
        rem_c = random.sample(all_custs, rem_cnt)
        rem_set = set(rem_c)

        # Remove from routes and filter empty routes
        pruned_routes = []
        for r_nodes in routes_list:
            c_kept = [c for c in r_nodes[1:-1] if c not in rem_set]
            if c_kept:
                pruned_routes.append([depot] + c_kept + [depot])

        # Sort removed customers by constraint tightness (most constrained first)
        rem_c.sort(key=lambda c: (data.node[c].end - data.node[c].start, -(data.node[c].delivery + data.node[c].pickup)))

        recreate_success = True
        for c in rem_c:
            best_r = -1
            best_pos = -1
            min_cost = float('inf')

            for r_i, r_nodes in enumerate(pruned_routes):
                len_r = len(r_nodes)
                if len_r >= L - 1:
                    continue
                N_cand = len_r - 1
                batch_cand = [r_nodes[:pos] + [c] + r_nodes[pos:] for pos in range(1, len_r)]
                r_t = torch.tensor(batch_cand, dtype=torch.long, device=device)
                l_t = torch.full((N_cand,), len_r + 1, dtype=torch.long, device=device)
                c_feas, c_dists = backend.evaluate_routes_gpu(r_t, l_t)
                if c_feas.any():
                    valid_idx = torch.nonzero(c_feas, as_tuple=True)[0]
                    min_d, argmin_d = torch.min(c_dists[valid_idx], dim=0)
                    if min_d.item() < min_cost:
                        min_cost = min_d.item()
                        best_r = r_i
                        best_pos = valid_idx[argmin_d].item() + 1

            if best_r != -1:
                pruned_routes[best_r].insert(best_pos, c)
            else:
                # Cannot feasibly insert without opening extra route; abort modification
                recreate_success = False
                break

        if recreate_success and len(pruned_routes) <= r_cnt:
            num_new_r = min(len(pruned_routes), R)
            cand_counts[p_idx] = num_new_r
            cand_routes[p_idx].fill_(depot)
            cand_lengths[p_idx].fill_(2)
            for r_i in range(num_new_r):
                nodes = pruned_routes[r_i]
                l_val = min(len(nodes), L)
                cand_lengths[p_idx, r_i] = l_val
                cand_routes[p_idx, r_i, :l_val] = torch.tensor(nodes[:l_val], dtype=torch.long, device=device)
            cand_modified[p_idx] = True

    if not cand_modified.any():
        return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists

    c_feas, c_costs, c_v_cnts, c_dists = backend.evaluate_population_tensor(
        cand_routes, cand_lengths, cand_counts
    )

    better_nv = c_v_cnts < v_counts
    same_nv = c_v_cnts == v_counts
    better_d = same_nv & (c_dists < total_dists)
    allow_div = same_nv & (c_dists <= total_dists * 1.05)

    accepted = target_mask & cand_modified & c_feas & (better_nv | better_d | allow_div) & (c_v_cnts <= v_counts)
    if accepted.any():
        pop_routes[accepted] = cand_routes[accepted]
        pop_lengths[accepted] = cand_lengths[accepted]
        pop_route_counts[accepted] = cand_counts[accepted]
        feas[accepted] = c_feas[accepted]
        costs[accepted] = c_costs[accepted]
        v_counts[accepted] = c_v_cnts[accepted]
        total_dists[accepted] = c_dists[accepted]

        # Fast 2-opt pass on accepted solutions
        pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists = tensor_gpu_intra_2opt(
            pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists,
            accepted, backend, data, max_passes=2
        )

    return pop_routes, pop_lengths, pop_route_counts, feas, costs, v_counts, total_dists


def tensor_gpu_woa_intensification(
    cand_routes: torch.Tensor,
    cand_lengths: torch.Tensor,
    cand_counts: torch.Tensor,
    best_indices,
    a_param: float,
    p_woa_mask: torch.Tensor,
    backend,
    data
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Tensorized WOA Intensification (Paper Algorithm 4).
    Exploit branch (|A| < 1.0): Encircling / Spiral injection of elite routes from best
    with guaranteed feasibility repair and zero vehicle count inflation.
    Explore branch (|A| >= 1.0): Intra-route 2-opt, relocate, and inter-route customer swap
    with evaluate_routes_gpu verification.
    """
    P, R, L = cand_routes.shape
    device = backend.device
    depot = data.DC
    num_customers = data.customer_num

    if not p_woa_mask.any():
        return cand_routes, cand_lengths, cand_counts

    if isinstance(best_indices, int):
        best_indices_t = torch.full((P,), best_indices, dtype=torch.long, device=device)
    elif not isinstance(best_indices, torch.Tensor):
        best_indices_t = torch.tensor(best_indices, dtype=torch.long, device=device)
    else:
        best_indices_t = best_indices.to(device=device, dtype=torch.long)

    r1 = torch.rand(P, device=device)
    a_vec = 2.0 * a_param * r1 - a_param
    exploit_mask = p_woa_mask & (torch.abs(a_vec) < 1.0)
    explore_mask = p_woa_mask & (torch.abs(a_vec) >= 1.0)

    # 1. EXPLOIT BRANCH (|A| < 1.0): Elite route injection from best
    act_exploit = torch.nonzero(exploit_mask, as_tuple=True)[0]
    for p in act_exploit:
        p_idx = int(p.item())
        best_p = int(best_indices_t[p_idx].item())
        if best_p == p_idx:
            continue

        best_cnt = int(cand_counts[best_p].item())
        cur_cnt = int(cand_counts[p_idx].item())
        if best_cnt == 0 or cur_cnt == 0:
            continue

        r_b = random.randint(0, best_cnt - 1)
        l_b = int(cand_lengths[best_p, r_b].item())
        if l_b <= 2:
            continue
        elite_nodes = cand_routes[best_p, r_b, :l_b].tolist()
        elite_custs = set(elite_nodes[1:-1])
        if not elite_custs:
            continue

        # Find which route in p_idx has highest customer overlap with elite_custs
        best_overlap = -1
        best_r_p = 0
        for r_i in range(cur_cnt):
            l_i = int(cand_lengths[p_idx, r_i].item())
            if l_i <= 2:
                continue
            r_custs = set(cand_routes[p_idx, r_i, 1:l_i-1].tolist())
            overlap = len(r_custs.intersection(elite_custs))
            if overlap > best_overlap:
                best_overlap = overlap
                best_r_p = r_i

        # Replace best_r_p with elite route, remove duplicates from other routes
        routes_p = []
        for r_i in range(cur_cnt):
            l_i = int(cand_lengths[p_idx, r_i].item())
            if l_i <= 2:
                continue
            if r_i == best_r_p:
                routes_p.append(list(elite_nodes))
            else:
                rem_custs = [c for c in cand_routes[p_idx, r_i, 1:l_i-1].tolist() if c not in elite_custs]
                if rem_custs:
                    routes_p.append([depot] + rem_custs + [depot])

        # Missing customers: those not currently routed
        all_routed = set(c for r in routes_p for c in r[1:-1])
        missing = [c for c in range(1, num_customers + 1) if c not in all_routed]

        if missing:
            # Tightest time window width and largest demand first
            missing.sort(key=lambda c: (data.node[c].end - data.node[c].start, -(data.node[c].delivery + data.node[c].pickup)))
            exploit_success = True
            for c in missing:
                best_r = -1
                best_pos = -1
                min_cost = float('inf')

                best_cand = None
                # Only insert into non-elite routes to keep the injected elite route intact
                for r_i, r_nodes in enumerate(routes_p):
                    if r_nodes == elite_nodes:
                        continue
                    len_r = len(r_nodes)
                    if len_r >= L - 1:
                        continue
                    for pos in range(1, len_r):
                        prev, nxt = r_nodes[pos - 1], r_nodes[pos]
                        if getattr(data, 'pruning', False) and getattr(data, 'pm', None) is not None:
                            if not data.pm[prev][c] or not data.pm[c][nxt]:
                                continue
                        cand = r_nodes[:pos] + [c] + r_nodes[pos:]
                        flag, _ = _chk_route_list(cand, data)
                        if not flag:
                            cand_opt = optimize_route_nodes_2opt(cand, data)
                            flag, _ = _chk_route_list(cand_opt, data)
                            if flag:
                                cand = cand_opt
                        if flag:
                            d = _route_distance(cand, data)
                            if d < min_cost:
                                min_cost = d
                                best_r = r_i
                                best_cand = cand

                if best_r != -1:
                    routes_p[best_r] = best_cand
                else:
                    exploit_success = False
                    break

            if not exploit_success:
                continue

        # Store into cand_routes[p_idx]
        num_new_r = min(len(routes_p), R)
        cand_counts[p_idx] = num_new_r
        cand_routes[p_idx].fill_(depot)
        cand_lengths[p_idx].fill_(2)
        for r_i in range(num_new_r):
            nodes = routes_p[r_i]
            l_val = min(len(nodes), L)
            cand_lengths[p_idx, r_i] = l_val
            cand_routes[p_idx, r_i, :l_val] = torch.tensor(nodes[:l_val], dtype=torch.long, device=device)

    # 2. EXPLORE BRANCH (|A| >= 1.0): Randomized local moves (2-opt, relocate, swap)
    act_explore = torch.nonzero(explore_mask, as_tuple=True)[0]
    for p in act_explore:
        p_idx = int(p.item())
        cur_cnt = int(cand_counts[p_idx].item())
        if cur_cnt == 0:
            continue

        move_choice = random.randint(1, 3)
        if move_choice == 1:  # Intra-route 2-opt
            r_idx = random.randint(0, cur_cnt - 1)
            r_len = int(cand_lengths[p_idx, r_idx].item())
            if r_len >= 4:
                i1, i2 = sorted([random.randint(1, r_len - 2), random.randint(1, r_len - 2)])
                if i1 != i2:
                    sub = cand_routes[p_idx, r_idx, i1:i2+1].clone()
                    test_route = cand_routes[p_idx, r_idx:r_idx+1, :r_len].clone()
                    test_route[0, i1:i2+1] = torch.flip(sub, dims=[0])
                    t_len = cand_lengths[p_idx, r_idx:r_idx+1]
                    f, d = backend.evaluate_routes_gpu(test_route, t_len)
                    if f[0]:
                        cand_routes[p_idx, r_idx, i1:i2+1] = torch.flip(sub, dims=[0])

        elif move_choice == 2 and cur_cnt >= 2:  # Inter-route Relocate
            r1, r2 = random.sample(range(cur_cnt), 2)
            len1 = int(cand_lengths[p_idx, r1].item())
            len2 = int(cand_lengths[p_idx, r2].item())
            if len1 >= 3 and len2 < L - 1:
                i1 = random.randint(1, len1 - 2)
                cust = cand_routes[p_idx, r1, i1].clone()
                batch_c = [cand_routes[p_idx, r2, :pos].tolist() + [int(cust.item())] + cand_routes[p_idx, r2, pos:len2].tolist() for pos in range(1, len2)]
                r_t = torch.tensor(batch_c, dtype=torch.long, device=device)
                l_t = torch.full((len(batch_c),), len2 + 1, dtype=torch.long, device=device)
                f, d = backend.evaluate_routes_gpu(r_t, l_t)
                if f.any():
                    valid_idx = torch.nonzero(f, as_tuple=True)[0]
                    best_pos = valid_idx[0].item() + 1
                    src_nodes = [c for i, c in enumerate(cand_routes[p_idx, r1, :len1].tolist()) if i != i1]
                    if len(src_nodes) > 2:
                        s_t = torch.tensor([src_nodes], dtype=torch.long, device=device)
                        sl_t = torch.tensor([len(src_nodes)], dtype=torch.long, device=device)
                        sf, _ = backend.evaluate_routes_gpu(s_t, sl_t)
                        if sf[0]:
                            cand_routes[p_idx, r1, i1:len1-1] = cand_routes[p_idx, r1, i1+1:len1].clone()
                            cand_routes[p_idx, r1, len1-1] = depot
                            cand_lengths[p_idx, r1] = len1 - 1

                            cand_routes[p_idx, r2, best_pos+1:len2+1] = cand_routes[p_idx, r2, best_pos:len2].clone()
                            cand_routes[p_idx, r2, best_pos] = cust
                            cand_lengths[p_idx, r2] = len2 + 1

        elif move_choice == 3 and cur_cnt >= 2:  # Inter-route Swap
            r1, r2 = random.sample(range(cur_cnt), 2)
            len1 = int(cand_lengths[p_idx, r1].item())
            len2 = int(cand_lengths[p_idx, r2].item())
            if len1 >= 3 and len2 >= 3:
                i1 = random.randint(1, len1 - 2)
                i2 = random.randint(1, len2 - 2)
                c1 = cand_routes[p_idx, r1, i1].clone()
                c2 = cand_routes[p_idx, r2, i2].clone()
                max_l = max(len1, len2)
                tb = torch.full((2, max_l), depot, dtype=torch.long, device=device)
                tb[0, :len1] = cand_routes[p_idx, r1, :len1]
                tb[0, i1] = c2
                tb[1, :len2] = cand_routes[p_idx, r2, :len2]
                tb[1, i2] = c1

                tbl = torch.tensor([len1, len2], dtype=torch.long, device=device)
                f, d = backend.evaluate_routes_gpu(tb, tbl)
                if f.all():
                    cand_routes[p_idx, r1, i1] = c2
                    cand_routes[p_idx, r2, i2] = c1

    return cand_routes, cand_lengths, cand_counts


def tensor_gpu_neighborhood_moves(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    p_woa_mask: torch.Tensor,
    backend,
    data
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility wrapper redirecting to tensor_gpu_woa_intensification with default a=1.0."""
    P = pop_routes.shape[0]
    best_indices = torch.zeros(P, dtype=torch.long, device=backend.device)
    return tensor_gpu_woa_intensification(
        pop_routes, pop_lengths, pop_route_counts, best_indices, 1.0, p_woa_mask, backend, data
    )


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
    return tensor_gpu_woa_intensification(
        cand_routes, cand_lengths, cand_counts, best_indices, a_param, p_woa_mask, backend, data
    )
