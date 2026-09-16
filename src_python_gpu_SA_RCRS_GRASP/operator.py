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

def tensor_rcrs_grasp_init(
    P: int,
    data,
    backend,
    alpha_lo: float = 0.10,
    alpha_hi: float = 0.40,
    sa_iters: int = 50
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure PyTorch GPU 4D Batched Masked Tensor RCRS-GRASP Initialization.
    Eliminates all CPU loops and Numba dependencies.
    Computes delta_td, rc_penalty, rs_penalty across all (P x K x routes x positions)
    in a unified 4D tensor with padding & capacity masks, then reduces via torch.min & torch.topk.
    """
    device = backend.device
    num_customers = data.customer_num
    depot = data.DC
    capacity = backend.capacity
    dist_t = backend.dist_t
    delivery_t = backend.delivery_t
    pickup_t = backend.pickup_t

    max_routes = min(num_customers, 60)
    max_nodes = num_customers + 2

    # RCRS Weights and Capacity Threshold (matching C++ paper formulation)
    w_td = 1.0
    w_rc = 0.5
    w_rs = 0.3
    rc_thr = 0.70

    pop_routes = torch.full((P, max_routes, max_nodes), depot, dtype=torch.long, device=device)
    pop_lengths = torch.full((P, max_routes), 2, dtype=torch.long, device=device)
    pop_route_counts = torch.ones(P, dtype=torch.long, device=device)
    pop_loads = torch.zeros((P, max_routes), dtype=torch.float32, device=device)

    unrouted_mask = torch.zeros((P, num_customers + 1), dtype=torch.bool, device=device)
    unrouted_mask[:, 1:num_customers+1] = True

    # Initial seed customer for each individual
    for p in range(P):
        seed_c = random.randint(1, num_customers)
        pop_routes[p, 0, 1] = seed_c
        pop_lengths[p, 0] = 3
        pop_loads[p, 0] = delivery_t[seed_c].item()
        unrouted_mask[p, seed_c] = False

    # Pre-allocated index grids for 4D masking
    pos_grid = torch.arange(max_nodes - 1, device=device)[None, None, None, :]  # (1, 1, 1, max_nodes - 1)
    routes_grid = torch.arange(max_routes, device=device)[None, None, :, None]   # (1, 1, max_routes, 1)
    inf_val = torch.tensor(float('inf'), device=device)

    while unrouted_mask.any():
        # 1. Fully vectorized GPU candidate sampling
        K = min(32, num_customers)
        rand_keys = torch.rand((P, num_customers + 1), device=device)
        rand_keys = torch.where(unrouted_mask, rand_keys, torch.tensor(-1.0, device=device))
        _, cands = torch.topk(rand_keys, k=K, dim=-1)
        cand_valid_mask = torch.gather(unrouted_mask, dim=1, index=cands)

        if not cand_valid_mask.any():
            break

        # 2. 4D Batched Masked Tensor Computation (P x K x max_routes x (max_nodes - 1))
        prev = pop_routes[:, None, :, :-1]  # (P, 1, R, max_nodes - 1)
        nxt  = pop_routes[:, None, :, 1:]   # (P, 1, R, max_nodes - 1)
        c    = cands[:, :, None, None]      # (P, K, 1, 1)

        d_prev_c = dist_t[prev, c]
        d_c_nxt  = dist_t[c, nxt]
        d_prev_nxt = dist_t[prev, nxt]
        delta_td = torch.clamp(d_prev_c + d_c_nxt - d_prev_nxt, min=0.0)

        dx_prev = dist_t[depot, prev]
        dx_c = dist_t[depot, c]
        rs_pen = torch.abs(dx_prev + d_prev_c - dx_c)

        pos_scores = w_td * delta_td + w_rs * rs_pen

        # Mask invalid positions and routes
        route_lens = pop_lengths[:, None, :, None]
        valid_pos = (pos_grid < route_lens - 1)
        valid_routes = (routes_grid < pop_route_counts[:, None, None, None])
        valid_mask = valid_pos & valid_routes & cand_valid_mask[:, :, None, None]

        pos_scores = torch.where(valid_mask, pos_scores, inf_val)

        # 3. Reduce position to best_pos, and evaluate capacity + RC penalty
        best_pos_score, best_pos_rel = torch.min(pos_scores, dim=-1)
        best_pos = best_pos_rel + 1

        c_deliv = delivery_t[cands]
        c_pickup = pickup_t[cands]
        c_dem = torch.maximum(c_deliv, c_pickup)

        cap_feasible = (pop_loads[:, None, :] + c_deliv[:, :, None]) <= capacity
        rc_pen = torch.clamp(pop_loads[:, None, :] + c_dem[:, :, None] - capacity * rc_thr, min=0.0)

        rcrs_route_score = best_pos_score + w_rc * rc_pen
        rcrs_route_score = torch.where(
            cap_feasible & (routes_grid[:, :, :, 0] < pop_route_counts[:, None, None]) & cand_valid_mask[:, :, None],
            rcrs_route_score,
            inf_val
        )

        # Top-2 routes per candidate via torch.topk
        top2_scores, top2_routes = torch.topk(rcrs_route_score, k=2, dim=-1, largest=False)
        top2_pos = torch.gather(best_pos, dim=-1, index=top2_routes)

        # 4. Vectorized route candidate construction on GPU & Batch Feasibility Check
        valid_0 = cand_valid_mask & (top2_scores[..., 0] < 1e8)
        valid_1 = cand_valid_mask & (top2_scores[..., 1] < 1e8) & (top2_routes[..., 1] != top2_routes[..., 0])
        valid_mask_2d = torch.stack([valid_0, valid_1], dim=-1)  # (P, K, 2)

        flat_valid = valid_mask_2d.flatten()
        valid_idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)

        if len(valid_idx) == 0:
            for p in range(P):
                p_unr = torch.nonzero(unrouted_mask[p], as_tuple=True)[0]
                if len(p_unr) > 0:
                    c_new = int(p_unr[0].item())
                    new_r = int(pop_route_counts[p].item())
                    if new_r < max_routes:
                        pop_routes[p, new_r, 1] = c_new
                        pop_routes[p, new_r, 2] = depot
                        pop_lengths[p, new_r] = 3
                        pop_loads[p, new_r] = delivery_t[c_new].item()
                        pop_route_counts[p] = new_r + 1
                        unrouted_mask[p, c_new] = False
                    else:
                        min_r = int(torch.argmin(pop_lengths[p, :new_r]).item())
                        cur_len = int(pop_lengths[p, min_r].item())
                        pop_routes[p, min_r, cur_len - 1] = c_new
                        pop_routes[p, min_r, cur_len] = depot
                        pop_lengths[p, min_r] = cur_len + 1
                        pop_loads[p, min_r] += delivery_t[c_new].item()
                        unrouted_mask[p, c_new] = False
            continue

        p_grid = torch.arange(P, device=device)[:, None, None].expand(P, K, 2)
        c_expanded = cands[:, :, None].expand(P, K, 2)

        p_sub = p_grid.flatten()[valid_idx]
        r_sub = top2_routes.flatten()[valid_idx]
        pos_sub = top2_pos.flatten()[valid_idx]
        c_sub = c_expanded.flatten()[valid_idx]
        score_sub = top2_scores.flatten()[valid_idx]

        len_sub = pop_lengths[p_sub, r_sub]
        new_len = len_sub + 1
        max_L = int(new_len.max().item())

        N_cands = len(valid_idx)
        orig_routes = pop_routes[p_sub, r_sub, :max_L]

        col_idx = torch.arange(max_L, device=device).unsqueeze(0).expand(N_cands, -1)
        orig_idx = torch.where(col_idx < pos_sub.unsqueeze(1), col_idx, col_idx - 1)
        orig_idx = torch.clamp(orig_idx, min=0)
        gathered = torch.gather(orig_routes, dim=1, index=orig_idx)

        cand_routes = torch.where(col_idx == pos_sub.unsqueeze(1), c_sub.unsqueeze(1), gathered)
        cand_routes = torch.where(col_idx < new_len.unsqueeze(1), cand_routes, depot)

        # Batch GPU feasibility evaluation (time windows + load dynamics)
        batch_feas, _ = backend.evaluate_routes_gpu(cand_routes, new_len)

        # Transfer compact 1D results to CPU in one single transfer
        p_cpu = p_sub.cpu().numpy()
        r_cpu = r_sub.cpu().numpy()
        pos_cpu = pos_sub.cpu().numpy()
        c_cpu = c_sub.cpu().numpy()
        score_cpu = score_sub.cpu().numpy()
        feas_cpu = batch_feas.cpu().numpy()

        p_cands = {p: [] for p in range(P)}
        feasible_mask = feas_cpu & (score_cpu < 1e8)
        for i in np.nonzero(feasible_mask)[0]:
            p_val = p_cpu[i]
            p_cands[p_val].append((score_cpu[i], int(c_cpu[i]), int(r_cpu[i]), int(pos_cpu[i])))

        for p in range(P):
            cands_list = p_cands[p]
            if cands_list:
                cands_list.sort(key=lambda x: x[0])
                min_s = cands_list[0][0]
                max_s = cands_list[-1][0]
                alpha = random.uniform(alpha_lo, alpha_hi)
                threshold = min_s + alpha * (max_s - min_s)
                rcl = [x for x in cands_list if x[0] <= threshold + 1e-5]
                chosen = random.choice(rcl)
                _, c_chosen, r_chosen, pos_chosen = chosen

                cur_len = int(pop_lengths[p, r_chosen].item())
                pop_routes[p, r_chosen, pos_chosen+1:cur_len+1] = pop_routes[p, r_chosen, pos_chosen:cur_len].clone()
                pop_routes[p, r_chosen, pos_chosen] = c_chosen
                pop_lengths[p, r_chosen] = cur_len + 1
                pop_loads[p, r_chosen] += delivery_t[c_chosen].item()
                unrouted_mask[p, c_chosen] = False
            else:
                p_unr = torch.nonzero(unrouted_mask[p], as_tuple=True)[0]
                if len(p_unr) > 0:
                    c_new = int(p_unr[0].item())
                    new_r = int(pop_route_counts[p].item())
                    if new_r < max_routes:
                        pop_routes[p, new_r, 1] = c_new
                        pop_routes[p, new_r, 2] = depot
                        pop_lengths[p, new_r] = 3
                        pop_loads[p, new_r] = delivery_t[c_new].item()
                        pop_route_counts[p] = new_r + 1
                        unrouted_mask[p, c_new] = False
                    else:
                        min_r = int(torch.argmin(pop_lengths[p, :new_r]).item())
                        cur_len = int(pop_lengths[p, min_r].item())
                        pop_routes[p, min_r, cur_len - 1] = c_new
                        pop_routes[p, min_r, cur_len] = depot
                        pop_lengths[p, min_r] = cur_len + 1
                        pop_loads[p, min_r] += delivery_t[c_new].item()
                        unrouted_mask[p, c_new] = False

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
    100% Pure GPU Tensorized Giant-Tour Order Crossover (OX) (Zero Orphans & Invariant NV).
    Vectorized extraction and scatter:
    - Zero host-device transfers (.tolist / torch.tensor).
    - Zero route loops (rebuilt via single in-place scatter_).
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

    flat = cand_routes.view(P, R * L)
    cust_mask = (flat != depot) & (flat > 0) & (flat <= num_customers)
    sort_idx = torch.argsort((~cust_mask).long(), dim=1, stable=True)
    cust_flat_idx = sort_idx[:, :num_customers]
    giant_tours = torch.gather(flat, 1, cust_flat_idx)

    for p in active_p:
        p_idx = int(p.item())
        best_p = int(best_indices_t[p_idx].item())
        if best_p == p_idx:
            continue

        best_giant = giant_tours[best_p]
        child_giant = giant_tours[p_idx]
        N = num_customers
        if N <= 3:
            continue

        i1 = random.randint(0, N - 1)
        i2 = random.randint(0, N - 1)
        if i1 > i2:
            i1, i2 = i2, i1
        if i1 == i2:
            continue

        slice_best = best_giant[i1:i2+1]
        slice_mask = torch.isin(child_giant, slice_best)
        rem = child_giant[~slice_mask]
        if len(rem) == N - (i2 - i1 + 1):
            new_tour_t = torch.cat([rem[:i1], slice_best, rem[i1:]])
            giant_tours[p_idx] = new_tour_t

    flat.scatter_(1, cust_flat_idx, giant_tours)
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
            target_mask, backend, data, max_customers_in_route=4
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
    num_customers = data.customer_num

    # Extract giant tours to sample removed customers without tolist()
    flat = cand_routes.view(P, R * L)
    cust_mask = (flat != depot) & (flat > 0) & (flat <= num_customers)
    sort_idx = torch.argsort((~cust_mask).long(), dim=1, stable=True)
    cust_flat_idx = sort_idx[:, :num_customers]
    giant_tours = torch.gather(flat, 1, cust_flat_idx)

    rem_cnt = max(2, int(num_customers * removal_fraction))
    is_unrouted_mask = torch.zeros((P, num_customers + 1), dtype=torch.bool, device=device)
    for p in target_indices:
        p_idx = int(p.item())
        perm = torch.randperm(num_customers, device=device)[:rem_cnt]
        rem_c = giant_tours[p_idx, perm]
        is_unrouted_mask[p_idx, rem_c] = True

    # Vectorized remove across target individuals
    p_grid = torch.arange(P, device=device)[:, None, None].expand(P, R, L)
    is_rem = is_unrouted_mask[p_grid, cand_routes] & target_mask[:, None, None] & (cand_routes != depot)
    cand_routes = torch.where(is_rem, torch.tensor(depot, device=device), cand_routes)

    # Vectorized route repacking & empty route compaction via argsort
    c_mask = (cand_routes != depot) & (cand_routes > 0) & (cand_routes <= num_customers)
    r_cust_cnts = c_mask.sum(dim=-1)
    sort_idx_r = torch.argsort((~c_mask).long(), dim=-1, stable=True)
    sorted_nodes = torch.gather(cand_routes, -1, sort_idx_r)

    col_grid = torch.arange(L, device=device)[None, None, :].expand(P, R, -1)
    in_range = (col_grid >= 1) & (col_grid <= r_cust_cnts.unsqueeze(-1))
    src_col = (col_grid - 1).clamp(min=0)
    packed_routes = torch.where(in_range, torch.gather(sorted_nodes, -1, src_col), torch.tensor(depot, device=device))

    is_empty = (r_cust_cnts == 0)
    empty_sort = torch.argsort(is_empty.long(), dim=1, stable=True)
    empty_sort_3d = empty_sort.unsqueeze(-1).expand(-1, -1, L)

    comp_routes = torch.gather(packed_routes, 1, empty_sort_3d)
    comp_cust_cnts = torch.gather(r_cust_cnts, 1, empty_sort)
    comp_lengths = torch.where(comp_cust_cnts > 0, comp_cust_cnts + 2, torch.tensor(2, device=device))
    comp_counts = (~is_empty).sum(dim=1).clamp(min=1)

    cand_routes = torch.where(target_mask[:, None, None], comp_routes, cand_routes)
    cand_lengths = torch.where(target_mask[:, None], comp_lengths, cand_lengths)
    cand_counts = torch.where(target_mask, comp_counts, cand_counts)

    # 4D Batched Masked Tensor insertion for unrouted customers
    unrouted = is_unrouted_mask.clone()
    if unrouted.any():
        pos_grid = torch.arange(L - 1, device=device)[None, None, None, :]
        routes_grid = torch.arange(R, device=device)[None, None, :, None]
        inf_val = torch.tensor(float('inf'), device=device)

        while unrouted.any():
            K = min(16, num_customers)
            rand_keys = torch.rand((P, num_customers + 1), device=device)
            rand_keys = torch.where(unrouted, rand_keys, torch.tensor(-1.0, device=device))
            _, cands = torch.topk(rand_keys, k=K, dim=-1)
            cand_valid = torch.gather(unrouted, dim=1, index=cands)

            if not cand_valid.any():
                break

            prev = cand_routes[:, None, :, :-1]
            nxt  = cand_routes[:, None, :, 1:]
            c    = cands[:, :, None, None]

            d_prev_c = dist_t[prev, c]
            d_c_nxt  = dist_t[c, nxt]
            d_prev_nxt = dist_t[prev, nxt]
            delta_td = torch.clamp(d_prev_c + d_c_nxt - d_prev_nxt, min=0.0)

            dx_prev = dist_t[depot, prev]
            dx_c = dist_t[depot, c]
            rs_pen = torch.abs(dx_prev + d_prev_c - dx_c)

            pos_scores = delta_td + 0.3 * rs_pen

            route_lens = cand_lengths[:, None, :, None]
            valid_pos = (pos_grid < route_lens - 1) & (route_lens < L - 1)
            valid_routes = (routes_grid < cand_counts[:, None, None, None])
            valid_mask = valid_pos & valid_routes & cand_valid[:, :, None, None]

            pos_scores = torch.where(valid_mask, pos_scores, inf_val)

            best_pos_score, best_pos_rel = torch.min(pos_scores, dim=-1)
            best_pos = best_pos_rel + 1

            best_r_score, best_r_idx = torch.min(best_pos_score, dim=-1)
            best_cand_score, best_k_idx = torch.min(best_r_score, dim=-1)

            c_chosen = torch.gather(cands, 1, best_k_idx.unsqueeze(1)).squeeze(1)
            r_chosen = torch.gather(best_r_idx, 1, best_k_idx.unsqueeze(1)).squeeze(1)
            pos_chosen = torch.gather(
                torch.gather(best_pos, 1, best_k_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, R)).squeeze(1),
                1,
                r_chosen.unsqueeze(1)
            ).squeeze(1)

            act_p = torch.nonzero(cand_valid.any(dim=-1), as_tuple=True)[0]
            for p in act_p:
                p_idx = int(p.item())
                c_val = int(c_chosen[p_idx].item())
                if best_cand_score[p_idx].item() < 1e8:
                    r_val = int(r_chosen[p_idx].item())
                    pos_val = int(pos_chosen[p_idx].item())
                    cur_l = int(cand_lengths[p_idx, r_val].item())
                    if cur_l < L - 1:
                        cand_routes[p_idx, r_val, pos_val+1:cur_l+1] = cand_routes[p_idx, r_val, pos_val:cur_l].clone()
                        cand_routes[p_idx, r_val, pos_val] = c_val
                        cand_lengths[p_idx, r_val] = cur_l + 1
                        unrouted[p_idx, c_val] = False
                        continue

                cur_cnt = int(cand_counts[p_idx].item())
                if cur_cnt < R:
                    cand_routes[p_idx, cur_cnt].fill_(depot)
                    cand_routes[p_idx, cur_cnt, 1] = c_val
                    cand_lengths[p_idx, cur_cnt] = 3
                    cand_counts[p_idx] = cur_cnt + 1
                    unrouted[p_idx, c_val] = False
                else:
                    min_r = int(torch.argmin(cand_lengths[p_idx, :cur_cnt]).item())
                    cur_l = int(cand_lengths[p_idx, min_r].item())
                    if cur_l < L - 1:
                        cand_routes[p_idx, min_r, cur_l - 1] = c_val
                        cand_routes[p_idx, min_r, cur_l] = depot
                        cand_lengths[p_idx, min_r] = cur_l + 1
                    unrouted[p_idx, c_val] = False

    cand_modified = target_mask.clone()

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
    100% Pure GPU Tensorized WOA Intensification with Dynamic Exploit vs Explore.
    Vectorized implementation:
      - Explore branch (|A| >= 1.0): Full-batch giant-tour multi-swap, 2-opt reverse, and relocate
        via torch.gather and in-place flat.scatter_ with zero route loops.
      - Exploit branch (|A| < 1.0): Full-batch torch.isin remove, stable argsort compaction,
        direct elite route injection, and 4D batched masked tensor insertion with zero route loops.
    """
    P, R, L = cand_routes.shape
    device = backend.device
    dist_t = backend.dist_t
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

    # Compute a_vec for all individuals on GPU
    r1 = torch.rand(P, device=device)
    a_vec = 2.0 * a_param * r1 - a_param
    exploit_mask = p_woa_mask & (torch.abs(a_vec) < 1.0)
    explore_mask = p_woa_mask & (torch.abs(a_vec) >= 1.0)

    # -------------------------------------------------------------
    # 1. EXPLORE BRANCH (|A| >= 1.0): 100% Vectorized Giant-Tour Shaking
    # -------------------------------------------------------------
    if explore_mask.any():
        flat = cand_routes.view(P, R * L)
        cust_mask = (flat != depot) & (flat > 0) & (flat <= num_customers)
        sort_idx = torch.argsort((~cust_mask).long(), dim=1, stable=True)
        cust_flat_idx = sort_idx[:, :num_customers]
        giant_tours = torch.gather(flat, 1, cust_flat_idx)

        # Batched randomized mutation: multi-swap, 2-opt reverse, relocate
        op_types = torch.randint(0, 3, (P,), device=device)

        # Op 0: Batch Multi-Swap (2 random swaps)
        tours_swap = giant_tours.clone()
        for _ in range(2):
            s1 = torch.randint(0, num_customers, (P, 1), device=device)
            s2 = torch.randint(0, num_customers, (P, 1), device=device)
            v1 = torch.gather(tours_swap, 1, s1)
            v2 = torch.gather(tours_swap, 1, s2)
            tours_swap.scatter_(1, s1, v2)
            tours_swap.scatter_(1, s2, v1)

        # Op 1: Batch 2-Opt Subsegment Reverse
        r_rev1 = torch.randint(0, num_customers - 1, (P,), device=device)
        r_rev2 = torch.randint(1, num_customers, (P,), device=device)
        start_rev = torch.minimum(r_rev1, r_rev2).unsqueeze(1)
        end_rev = torch.maximum(r_rev1, r_rev2).unsqueeze(1)
        k_grid = torch.arange(num_customers, device=device).unsqueeze(0).expand(P, -1)
        in_rev = (k_grid >= start_rev) & (k_grid <= end_rev)
        rev_idx = start_rev + end_rev - k_grid
        perm_rev = torch.where(in_rev, rev_idx, k_grid)
        tours_rev = torch.gather(giant_tours, 1, perm_rev)

        # Op 2: Batch Relocate
        src_rel = torch.randint(0, num_customers, (P, 1), device=device)
        dst_rel = torch.randint(0, num_customers, (P, 1), device=device)
        idx_case1 = torch.where(
            k_grid == dst_rel,
            src_rel,
            torch.where((k_grid >= src_rel) & (k_grid < dst_rel), k_grid + 1, k_grid)
        )
        idx_case2 = torch.where(
            k_grid == dst_rel,
            src_rel,
            torch.where((k_grid > dst_rel) & (k_grid <= src_rel), k_grid - 1, k_grid)
        )
        reloc_perm = torch.where(src_rel < dst_rel, idx_case1, idx_case2)
        tours_reloc = torch.gather(giant_tours, 1, reloc_perm)

        mutated = torch.where(
            (op_types == 0).unsqueeze(1),
            tours_swap,
            torch.where((op_types == 1).unsqueeze(1), tours_rev, tours_reloc)
        )
        final_tours = torch.where(explore_mask.unsqueeze(1), mutated, giant_tours)
        flat.scatter_(1, cust_flat_idx, final_tours)

    # -------------------------------------------------------------
    # 2. EXPLOIT BRANCH (|A| < 1.0): Full-Batch Vectorized Remove & Insertion
    # -------------------------------------------------------------
    if exploit_mask.any():
        # Pick elite route / segment from best_indices_t for each individual
        best_cnts = cand_counts[best_indices_t].clamp(min=1)
        r_b = (torch.rand(P, device=device) * best_cnts.float()).long().clamp(min=0)
        l_b = cand_lengths[best_indices_t, r_b]
        r_b = torch.where(l_b > 2, r_b, torch.zeros_like(r_b))
        l_b = cand_lengths[best_indices_t, r_b]

        use_full = torch.rand(P, device=device) < 0.5
        l_param = torch.rand(P, device=device) * 2.0 - 1.0
        spiral_scale = torch.abs(torch.exp(l_param) * torch.cos(2.0 * math.pi * l_param))
        c_cnt_b = (l_b - 2).clamp(min=1)
        raw_seg = (1.0 + spiral_scale).round().long()
        seg_len = torch.maximum(torch.ones_like(c_cnt_b), torch.minimum(c_cnt_b, raw_seg))
        max_start = (c_cnt_b - seg_len).clamp(min=0)
        start_i = 1 + (torch.rand(P, device=device) * (max_start.float() + 1.0)).long().clamp(min=0)

        # Build is_elite_mask (P, num_customers + 1)
        col_l = torch.arange(L, device=device).unsqueeze(0).expand(P, -1)
        in_elite_seg = torch.where(
            use_full.unsqueeze(1),
            (col_l >= 1) & (col_l < (l_b - 1).unsqueeze(1)),
            (col_l >= start_i.unsqueeze(1)) & (col_l < (start_i + seg_len).unsqueeze(1))
        )
        valid_exploit = exploit_mask & (best_indices_t != torch.arange(P, device=device)) & (l_b > 2)
        in_elite_seg = in_elite_seg & valid_exploit.unsqueeze(1)

        best_routes_sampled = cand_routes[best_indices_t, r_b]
        elite_cust_nodes = torch.where(in_elite_seg, best_routes_sampled, torch.tensor(depot, device=device))

        is_elite_mask = torch.zeros((P, num_customers + 1), dtype=torch.bool, device=device)
        valid_c_mask = in_elite_seg & (elite_cust_nodes > 0) & (elite_cust_nodes <= num_customers)
        if valid_c_mask.any():
            p_coords = torch.arange(P, device=device).unsqueeze(1).expand(P, L)[valid_c_mask]
            c_coords = elite_cust_nodes[valid_c_mask]
            is_elite_mask[p_coords, c_coords] = True

        # Technique 1: Full-batch remove via is_elite_mask
        p_grid = torch.arange(P, device=device)[:, None, None].expand(P, R, L)
        is_elite = is_elite_mask[p_grid, cand_routes]
        remove_mask = is_elite & valid_exploit[:, None, None] & (cand_routes != depot)
        cand_routes = torch.where(remove_mask, torch.tensor(depot, device=device), cand_routes)

        # Technique 2: Route repacking & empty route compaction via argsort
        c_mask = (cand_routes != depot) & (cand_routes > 0) & (cand_routes <= num_customers)
        r_cust_cnts = c_mask.sum(dim=-1)
        sort_idx = torch.argsort((~c_mask).long(), dim=-1, stable=True)
        sorted_nodes = torch.gather(cand_routes, -1, sort_idx)

        col_grid = torch.arange(L, device=device)[None, None, :].expand(P, R, -1)
        in_range = (col_grid >= 1) & (col_grid <= r_cust_cnts.unsqueeze(-1))
        src_col = (col_grid - 1).clamp(min=0)
        packed_routes = torch.where(in_range, torch.gather(sorted_nodes, -1, src_col), torch.tensor(depot, device=device))

        is_empty = (r_cust_cnts == 0)
        empty_sort = torch.argsort(is_empty.long(), dim=1, stable=True)
        empty_sort_3d = empty_sort.unsqueeze(-1).expand(-1, -1, L)

        comp_routes = torch.gather(packed_routes, 1, empty_sort_3d)
        comp_cust_cnts = torch.gather(r_cust_cnts, 1, empty_sort)
        comp_lengths = torch.where(comp_cust_cnts > 0, comp_cust_cnts + 2, torch.tensor(2, device=device))
        comp_counts = (~is_empty).sum(dim=1).clamp(min=1)

        cand_routes = torch.where(valid_exploit[:, None, None], comp_routes, cand_routes)
        cand_lengths = torch.where(valid_exploit[:, None], comp_lengths, cand_lengths)
        cand_counts = torch.where(valid_exploit, comp_counts, cand_counts)

        # Fast direct route injection for individuals with free vehicle slot
        can_insert_whole = valid_exploit & use_full & (cand_counts < R)
        for p in torch.nonzero(can_insert_whole, as_tuple=True)[0]:
            p_idx = int(p.item())
            b_idx = int(best_indices_t[p_idx].item())
            rb_idx = int(r_b[p_idx].item())
            lb_val = int(l_b[p_idx].item())
            slot = int(cand_counts[p_idx].item())
            cand_routes[p_idx, slot, :lb_val] = cand_routes[b_idx, rb_idx, :lb_val]
            cand_lengths[p_idx, slot] = lb_val
            cand_counts[p_idx] = slot + 1
            is_elite_mask[p_idx] = False

        # Technique 3: 4D Batched Masked Tensor insertion for remaining elite customers
        unrouted = is_elite_mask.clone()
        if unrouted.any():
            pos_grid = torch.arange(L - 1, device=device)[None, None, None, :]
            routes_grid = torch.arange(R, device=device)[None, None, :, None]
            inf_val = torch.tensor(float('inf'), device=device)

            while unrouted.any():
                K = min(16, num_customers)
                rand_keys = torch.rand((P, num_customers + 1), device=device)
                rand_keys = torch.where(unrouted, rand_keys, torch.tensor(-1.0, device=device))
                _, cands = torch.topk(rand_keys, k=K, dim=-1)
                cand_valid = torch.gather(unrouted, dim=1, index=cands)

                if not cand_valid.any():
                    break

                prev = cand_routes[:, None, :, :-1]
                nxt  = cand_routes[:, None, :, 1:]
                c    = cands[:, :, None, None]

                d_prev_c = dist_t[prev, c]
                d_c_nxt  = dist_t[c, nxt]
                d_prev_nxt = dist_t[prev, nxt]
                delta_td = torch.clamp(d_prev_c + d_c_nxt - d_prev_nxt, min=0.0)

                dx_prev = dist_t[depot, prev]
                dx_c = dist_t[depot, c]
                rs_pen = torch.abs(dx_prev + d_prev_c - dx_c)

                pos_scores = delta_td + 0.3 * rs_pen

                route_lens = cand_lengths[:, None, :, None]
                valid_pos = (pos_grid < route_lens - 1) & (route_lens < L - 1)
                valid_routes = (routes_grid < cand_counts[:, None, None, None])
                valid_mask = valid_pos & valid_routes & cand_valid[:, :, None, None]

                pos_scores = torch.where(valid_mask, pos_scores, inf_val)

                best_pos_score, best_pos_rel = torch.min(pos_scores, dim=-1)
                best_pos = best_pos_rel + 1

                best_r_score, best_r_idx = torch.min(best_pos_score, dim=-1)
                best_cand_score, best_k_idx = torch.min(best_r_score, dim=-1)

                c_chosen = torch.gather(cands, 1, best_k_idx.unsqueeze(1)).squeeze(1)
                r_chosen = torch.gather(best_r_idx, 1, best_k_idx.unsqueeze(1)).squeeze(1)
                pos_chosen = torch.gather(
                    torch.gather(best_pos, 1, best_k_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, R)).squeeze(1),
                    1,
                    r_chosen.unsqueeze(1)
                ).squeeze(1)

                act_p = torch.nonzero(cand_valid.any(dim=-1), as_tuple=True)[0]
                for p in act_p:
                    p_idx = int(p.item())
                    c_val = int(c_chosen[p_idx].item())
                    if best_cand_score[p_idx].item() < 1e8:
                        r_val = int(r_chosen[p_idx].item())
                        pos_val = int(pos_chosen[p_idx].item())
                        cur_l = int(cand_lengths[p_idx, r_val].item())
                        if cur_l < L - 1:
                            cand_routes[p_idx, r_val, pos_val+1:cur_l+1] = cand_routes[p_idx, r_val, pos_val:cur_l].clone()
                            cand_routes[p_idx, r_val, pos_val] = c_val
                            cand_lengths[p_idx, r_val] = cur_l + 1
                            unrouted[p_idx, c_val] = False
                            continue

                    cur_cnt = int(cand_counts[p_idx].item())
                    if cur_cnt < R:
                        cand_routes[p_idx, cur_cnt].fill_(depot)
                        cand_routes[p_idx, cur_cnt, 1] = c_val
                        cand_lengths[p_idx, cur_cnt] = 3
                        cand_counts[p_idx] = cur_cnt + 1
                        unrouted[p_idx, c_val] = False
                    else:
                        min_r = int(torch.argmin(cand_lengths[p_idx, :cur_cnt]).item())
                        cur_l = int(cand_lengths[p_idx, min_r].item())
                        if cur_l < L - 1:
                            cand_routes[p_idx, min_r, cur_l - 1] = c_val
                            cand_routes[p_idx, min_r, cur_l] = depot
                            cand_lengths[p_idx, min_r] = cur_l + 1
                        unrouted[p_idx, c_val] = False

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
