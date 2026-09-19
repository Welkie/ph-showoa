from __future__ import annotations

import math
import random
from typing import List, Tuple, Set, Optional, Dict, Any

import numpy as np
import torch

from .config import (
    INFEASIBLE,
    MAX_NODE_IN_ROUTE,
    MAX_POINT,
    PRECISION,
    RCRS,
    TD,
)
from .eval import chk_nl_node_pos_O_n, eval_move, evaluate_route_batch, _chk_route_list
from .move import Move, Seq
from .solution import Route, Solution, make_tmp_nl
from .util import argsort, rand, randint

TMP_MOVE = Move()


def _is_customer(node: int, data) -> bool:
    return node != data.DC and 1 <= node <= data.customer_num


def _route_distance(nl: List[int], data) -> float:
    dist_mat = data.dist
    return sum(dist_mat[nl[i]][nl[i+1]] for i in range(len(nl) - 1))


def _cuda_batches_enabled(data) -> bool:
    backend = getattr(data, "backend", None)
    return backend is not None and getattr(backend, "is_cuda", False)


def _build_insertion_sequences(route_nodes: List[int], nodes: List[int]):
    sequences = []
    meta = []
    route_len = len(route_nodes)
    for node_index, node in enumerate(nodes):
        for pos in range(1, route_len):
            sequences.append(route_nodes[:pos] + [node] + route_nodes[pos:])
            meta.append((node_index, pos))
    return sequences, meta


def _evaluate_candidate_routes_gpu(
    cand_routes: List[List[int]],
    backend
) -> Tuple[List[bool], List[float]]:
    if not cand_routes:
        return [], []
    device = backend.device
    depot = backend.depot
    N = len(cand_routes)
    lens = [len(r) for r in cand_routes]
    max_len = max(lens)

    packed = np.full((N, max_len), depot, dtype=np.int64)
    for i, r in enumerate(cand_routes):
        packed[i, :lens[i]] = r

    routes_t = torch.as_tensor(packed, dtype=torch.long, device=device)
    lens_t = torch.as_tensor(lens, dtype=torch.long, device=device)

    feas_t, dists_t = backend.evaluate_routes_gpu(routes_t, lens_t)
    return feas_t.tolist(), dists_t.tolist()


def optimize_route_nodes_2opt(node_list: List[int], data, backend=None) -> List[int]:
    if len(node_list) <= 3:
        return list(node_list)
    b = backend if backend is not None else getattr(data, 'backend', None)
    best_nl = list(node_list)
    improved = True
    dist_mat = data.dist

    while improved:
        improved = False
        length = len(best_nl)
        for i in range(1, length - 2):
            for j in range(i + 1, length - 1):
                a, b_node = best_nl[i-1], best_nl[i]
                c, d = best_nl[j], best_nl[j+1]
                if dist_mat[a][c] + dist_mat[b_node][d] < dist_mat[a][b_node] + dist_mat[c][d] - 1e-6:
                    candidate = best_nl[:i] + list(reversed(best_nl[i:j+1])) + best_nl[j+1:]
                    flag, _ = _chk_route_list(candidate, data)
                    if flag:
                        best_nl = candidate
                        improved = True
                        break
            if improved:
                break
    return best_nl


# =============================================================================
# Feasibility & Algorithm 10 Feasible Repair (Paper Algorithm 10)
# =============================================================================

def quick_check_feasibility(s: Solution, data) -> bool:
    record = set()
    for r in s.route_list:
        flag, _ = _chk_route_list(r.node_list, data)
        if not flag:
            return False
        for node in r.node_list:
            if _is_customer(node, data):
                if node in record:
                    return False
                record.add(node)
    return len(record) == data.customer_num


def check_route_capacity(nl: List[int], data) -> bool:
    length = len(nl)
    if length <= 2:
        return True
    capacity = data.vehicle.capacity
    load = 0.0
    for node in nl:
        load += data.node[node].delivery
    if load > capacity + 1e-5:
        return False
    for i in range(1, length):
        node = nl[i]
        load = load - data.node[node].delivery + data.node[node].pickup
        if load < -1e-5 or load > capacity + 1e-5:
            return False
    return True


def get_route_arrival_times_and_violations(nl: List[int], data) -> Tuple[List[float], List[int]]:
    length = len(nl)
    arrival_times = [0.0] * length
    violations = []
    if length <= 2:
        return arrival_times, violations
    time_val = data.start_time
    arrival_times[0] = time_val
    pre_node = nl[0]
    for i in range(1, length):
        node = nl[i]
        time_val += data.time[pre_node][node]
        arrival_times[i] = time_val
        if node != data.DC:
            if time_val > data.node[node].end + 1e-5:
                violations.append(node)
        time_val = max(time_val, data.node[node].start) + data.node[node].s_time
        pre_node = node
    return arrival_times, violations


def _intra_route_2_opt(nl: List[int], data) -> List[int]:
    improved = True
    best_nl = list(nl)
    flag, best_cost = _chk_route_list(best_nl, data)
    if not flag:
        best_cost = float('inf')

    while improved:
        improved = False
        length = len(best_nl)
        for i in range(1, length - 2):
            for j in range(i + 1, length - 1):
                new_nl = best_nl[:i] + best_nl[i:j+1][::-1] + best_nl[j+1:]
                flag, cost = _chk_route_list(new_nl, data)
                if flag and cost < best_cost - 1e-4:
                    best_nl = new_nl
                    best_cost = cost
                    improved = True
                    break
            if improved:
                break
    return best_nl


def _insert_customer_best_position(s: Solution, customer: int, data, avoid_r_idx: int = -1, allow_new_route: bool = True) -> bool:
    best_r_idx = -1
    best_p_idx = -1
    best_delta = float('inf')

    routes_to_check = []
    route_meta = []
    original_costs = {}

    for r_idx in range(len(s.route_list)):
        if r_idx == avoid_r_idx:
            continue
        route = s.route_list[r_idx]
        original_costs[r_idx] = route.cal_cost(data)
        for p_idx in range(1, len(route.node_list)):
            candidate_nl = route.node_list[:p_idx] + [customer] + route.node_list[p_idx:]
            if not check_route_capacity(candidate_nl, data):
                continue
            routes_to_check.append(candidate_nl)
            route_meta.append((r_idx, p_idx))

    if routes_to_check:
        results = evaluate_route_batch(routes_to_check, data)
        for (r_idx, pos), (flag, cost) in zip(route_meta, results):
            if not flag:
                continue
            delta = cost - original_costs[r_idx]
            if delta < best_delta:
                best_delta = delta
                best_r_idx = r_idx
                best_p_idx = pos

    if best_r_idx != -1:
        s.route_list[best_r_idx].node_list.insert(best_p_idx, customer)
        s.route_list[best_r_idx].update(data)
        return True
    elif allow_new_route:
        r = Route(data)
        r.node_list = [data.DC, customer, data.DC]
        r.update(data)
        s.append(r)
        return True
    return False


def feasible_or_repair_algorithm_10(s: Solution, data, rng: Optional[random.Random] = None) -> Solution:
    if quick_check_feasibility(s, data):
        s.update(data)
        s.cal_cost(data)
        return s

    s_prime = s.clone()
    s_prime.update(data)

    # --- Step 1: Repair duplicate and missing customers ---
    customer_occurrences: Dict[int, List[Tuple[int, int]]] = {}
    for r_idx, route in enumerate(s_prime.route_list):
        for p_idx, node in enumerate(route.node_list):
            if _is_customer(node, data):
                if node not in customer_occurrences:
                    customer_occurrences[node] = []
                customer_occurrences[node].append((r_idx, p_idx))

    for c, occs in customer_occurrences.items():
        if len(occs) > 1:
            best_occ = occs[0]
            best_occ_cost = float('inf')
            for keep_occ in occs:
                cand_s = s_prime.clone()
                for other_occ in occs:
                    if other_occ != keep_occ:
                        r_i, p_i = other_occ
                        if r_i < cand_s.len() and p_i < len(cand_s.route_list[r_i].node_list):
                            cand_s.route_list[r_i].node_list[p_i] = -1
                for r in cand_s.route_list:
                    r.node_list = [n for n in r.node_list if n != -1]
                cand_s.update(data)
                cand_s.cal_cost(data)
                if cand_s.cost < best_occ_cost:
                    best_occ_cost = cand_s.cost
                    best_occ = keep_occ

            for other_occ in occs:
                if other_occ != best_occ:
                    r_i, p_i = other_occ
                    if r_i < s_prime.len() and p_i < len(s_prime.route_list[r_i].node_list):
                        s_prime.route_list[r_i].node_list[p_i] = -1

            for r in s_prime.route_list:
                r.node_list = [n for n in r.node_list if n != -1]

    # Clean empty routes
    s_prime.route_list = [r for r in s_prime.route_list if len(r.node_list) > 2]
    s_prime.update(data)

    visited_customers = set(n for r in s_prime.route_list for n in r.node_list if _is_customer(n, data))
    missing_customers = [c for c in range(1, data.customer_num + 1) if c not in visited_customers]

    for c in missing_customers:
        # Try inserting into existing routes first without creating new routes
        if not _insert_customer_best_position(s_prime, c, data, allow_new_route=False):
            _insert_customer_best_position(s_prime, c, data, allow_new_route=True)

    s_prime.update(data)

    # --- Step 2: Repair capacity violations ---
    for r_idx in range(len(s_prime.route_list)):
        while r_idx < len(s_prime.route_list) and not check_route_capacity(s_prime.route_list[r_idx].node_list, data):
            route = s_prime.route_list[r_idx]
            route_customers = [n for n in route.node_list if _is_customer(n, data)]
            if not route_customers:
                break
            c = max(route_customers, key=lambda node: abs(data.node[node].delivery - data.node[node].pickup))
            route.node_list.remove(c)
            route.update(data)
            if not _insert_customer_best_position(s_prime, c, data, avoid_r_idx=r_idx, allow_new_route=False):
                _insert_customer_best_position(s_prime, c, data, avoid_r_idx=r_idx, allow_new_route=True)
            s_prime.route_list = [r for r in s_prime.route_list if len(r.node_list) > 2]
            s_prime.update(data)

    # --- Step 3: Repair time-window violations ---
    for r_idx in range(len(s_prime.route_list)):
        while r_idx < len(s_prime.route_list):
            arrival_times, violations = get_route_arrival_times_and_violations(s_prime.route_list[r_idx].node_list, data)
            if not violations:
                break
            c = violations[0]
            route = s_prime.route_list[r_idx]
            if c in route.node_list:
                route.node_list.remove(c)
                route.update(data)
            if not _insert_customer_best_position(s_prime, c, data, avoid_r_idx=r_idx, allow_new_route=False):
                _insert_customer_best_position(s_prime, c, data, avoid_r_idx=r_idx, allow_new_route=True)
            s_prime.route_list = [r for r in s_prime.route_list if len(r.node_list) > 2]
            s_prime.update(data)

    # --- Step 4: Route Elimination Pass (Direct NV Reduction) ---
    if len(s_prime.route_list) > 1:
        route_lens = [(i, len(r.node_list)) for i, r in enumerate(s_prime.route_list)]
        route_lens.sort(key=lambda x: x[1])
        for r_cand_idx, _ in route_lens:
            if r_cand_idx >= len(s_prime.route_list):
                continue
            cand_route_nodes = [n for n in s_prime.route_list[r_cand_idx].node_list if _is_customer(n, data)]
            if len(cand_route_nodes) <= 3:
                backup = s_prime.clone()
                del s_prime.route_list[r_cand_idx]
                absorbed = True
                for node in cand_route_nodes:
                    if not _insert_customer_best_position(s_prime, node, data, allow_new_route=False):
                        absorbed = False
                        break
                if not absorbed:
                    s_prime = backup
                else:
                    s_prime.update(data)
                    s_prime.cal_cost(data)
                    break

    # --- Step 5: Intra-route 2-opt clean ---
    for route in s_prime.route_list:
        route.node_list = _intra_route_2_opt(route.node_list, data)
        route.update(data)

    s_prime.update(data)
    s_prime.cal_cost(data)
    return s_prime


# =============================================================================
# RCRS & Population Initialization
# =============================================================================

def _rcrs_score(route, data, c: int, pos: int, w_td: float = 1.0, w_rc: float = 0.5, w_rs: float = 0.3, rc_thr: float = 0.70) -> float:
    nl = route.node_list
    prev = nl[pos - 1]
    next_node = nl[pos]

    delta_td = max(0.0, data.dist[prev][c] + data.dist[c][next_node] - data.dist[prev][next_node])

    load = 0.0
    for node in nl[1:-1]:
        load += data.node[node].delivery
    max_load = load
    for node in nl[1:]:
        load = load - data.node[node].delivery + data.node[node].pickup
        if load > max_load:
            max_load = load
    c_h_new = max_load + max(data.node[c].delivery, data.node[c].pickup)
    capacity = data.vehicle.capacity
    rc_penalty = max(0.0, c_h_new - capacity * rc_thr)

    dx_prev = data.dist[data.DC][prev]
    dx_c = data.dist[data.DC][c]
    rs_penalty = abs(dx_prev + data.dist[prev][c] - dx_c)

    return w_td * delta_td + w_rc * rc_penalty + w_rs * rs_penalty


def rcrs_grasp_initialization(data, rng: random.Random, alpha: float = 0.20) -> Solution:
    s = Solution(data)
    unrouted = [i for i in range(1, data.customer_num + 1) if i != data.DC]
    rng.shuffle(unrouted)

    while unrouted:
        best_per_customer = []
        global_best_score = float('inf')
        max_score = float('-inf')

        for c in unrouted:
            best_c = {"customer": c, "r_idx": -1, "pos": -1, "score": float('inf')}
            for r_idx, route in enumerate(s.route_list):
                nl = route.node_list
                for pos in range(1, len(nl)):
                    candidate_nl = nl[:pos] + [c] + nl[pos:]
                    if not check_route_capacity(candidate_nl, data):
                        continue
                    flag, _ = _chk_route_list(candidate_nl, data)
                    if not flag:
                        continue
                    score = _rcrs_score(route, data, c, pos)
                    if score < best_c["score"]:
                        best_c["score"] = score
                        best_c["r_idx"] = r_idx
                        best_c["pos"] = pos

            if best_c["r_idx"] != -1:
                if best_c["score"] < global_best_score:
                    global_best_score = best_c["score"]
                if best_c["score"] > max_score:
                    max_score = best_c["score"]

            best_per_customer.append(best_c)

        threshold = (
            float('inf')
            if (global_best_score == float('inf') or max_score == float('-inf'))
            else global_best_score + alpha * (max_score - global_best_score)
        )

        rcl_indices = []
        forced_new_indices = []

        for idx, cand in enumerate(best_per_customer):
            if cand["r_idx"] == -1:
                forced_new_indices.append(idx)
            elif cand["score"] <= threshold + 1e-9:
                rcl_indices.append(idx)

        if not rcl_indices:
            if not forced_new_indices:
                break
            pick_forced = rng.randint(0, len(forced_new_indices) - 1)
            ci = forced_new_indices[pick_forced]
            c = best_per_customer[ci]["customer"]
            r = Route(data)
            r.node_list = [data.DC, c, data.DC]
            r.update(data)
            s.append(r)
            unrouted.remove(c)
            continue

        pick = rng.randint(0, len(rcl_indices) - 1)
        chosen = best_per_customer[rcl_indices[pick]]

        s.route_list[chosen["r_idx"]].node_list.insert(chosen["pos"], chosen["customer"])
        s.route_list[chosen["r_idx"]].update(data)
        unrouted.remove(chosen["customer"])

    s.update(data)
    s.cal_cost(data)
    return s


def _sa_initialization(s_0: Solution, data, rng: random.Random, itermax: int = 25) -> Solution:
    s = s_0.clone()
    s.update(data)
    s.cal_cost(data)

    s_best = s.clone()
    best_cost = s_best.cost

    t0 = 100.0
    alpha = 0.95
    tmin = 0.1

    t = t0
    while t > tmin:
        for _ in range(itermax):
            move_type = rng.randint(1, 5)

            if move_type in {1, 2, 3}:
                if s.len() == 0:
                    continue
                r_idx = rng.randint(0, s.len() - 1)
                route = s.route_list[r_idx]
                nl = list(route.node_list)
                if len(nl) < 4:
                    continue

                if move_type == 1:
                    idx1 = rng.randint(1, len(nl) - 2)
                    idx2 = rng.randint(1, len(nl) - 2)
                    while idx1 == idx2:
                        idx2 = rng.randint(1, len(nl) - 2)
                    nl[idx1], nl[idx2] = nl[idx2], nl[idx1]
                elif move_type == 2:
                    idx1 = rng.randint(1, len(nl) - 2)
                    node = nl.pop(idx1)
                    idx2 = rng.randint(1, len(nl) - 1)
                    nl.insert(idx2, node)
                elif move_type == 3:
                    idx1 = rng.randint(1, len(nl) - 2)
                    idx2 = rng.randint(1, len(nl) - 2)
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    nl[idx1:idx2+1] = reversed(nl[idx1:idx2+1])

                flag, r_cost = _chk_route_list(nl, data)
                if flag:
                    old_r_cost = route.cal_cost(data)
                    new_cost = s.cost - old_r_cost + r_cost
                    delta = new_cost - s.cost

                    if delta < 0 or rng.random() < math.exp(-delta / (1e-6 + t * abs(s.cost))):
                        route.node_list = nl
                        route.update(data)
                        s.cost = new_cost
                        if s.cost < best_cost:
                            best_cost = s.cost
                            s_best = s.clone()

            else:
                if s.len() < 2:
                    continue
                r_idx1 = rng.randint(0, s.len() - 1)
                r_idx2 = rng.randint(0, s.len() - 1)
                while r_idx1 == r_idx2:
                    r_idx2 = rng.randint(0, s.len() - 1)

                route1 = s.route_list[r_idx1]
                route2 = s.route_list[r_idx2]
                nl1 = list(route1.node_list)
                nl2 = list(route2.node_list)

                if move_type == 4:
                    if len(nl1) < 3:
                        continue
                    idx1 = rng.randint(1, len(nl1) - 2)
                    node = nl1.pop(idx1)
                    idx2 = rng.randint(1, len(nl2) - 1)
                    nl2.insert(idx2, node)
                elif move_type == 5:
                    if len(nl1) < 3 or len(nl2) < 3:
                        continue
                    idx1 = rng.randint(1, len(nl1) - 2)
                    idx2 = rng.randint(1, len(nl2) - 2)
                    nl1[idx1], nl2[idx2] = nl2[idx2], nl1[idx1]

                flag1, r_cost1 = _chk_route_list(nl1, data)
                flag2, r_cost2 = _chk_route_list(nl2, data)
                if flag1 and flag2:
                    old_r_cost1 = route1.cal_cost(data)
                    old_r_cost2 = route2.cal_cost(data)
                    new_cost = s.cost - (old_r_cost1 + old_r_cost2) + (r_cost1 + r_cost2)
                    delta = new_cost - s.cost

                    if delta < 0 or rng.random() < math.exp(-delta / (1e-6 + t * abs(s.cost))):
                        route1.node_list = nl1
                        route2.node_list = nl2
                        route1.update(data)
                        route2.update(data)
                        if len(nl1) <= 2:
                            s.route_list = [r for r in s.route_list if len(r.node_list) > 2]
                            s.update(data)
                            s.cal_cost(data)
                        else:
                            s.cost = new_cost
                        if s.cost < best_cost:
                            best_cost = s.cost
                            s_best = s.clone()

        t = alpha * t

    s_best.route_list = [r for r in s_best.route_list if len(r.node_list) > 2]
    s_best.update(data)
    s_best.cal_cost(data)
    return s_best


def active_route_elimination(sol: Solution, data, max_elim_len: int = 4) -> Solution:
    if sol.len() <= 1:
        return sol
    improved = True
    while improved and sol.len() > 1:
        improved = False
        route_lens = [(i, len(r.node_list) - 2) for i, r in enumerate(sol.route_list)]
        route_lens.sort(key=lambda x: x[1])
        for r_idx, c_cnt in route_lens:
            if c_cnt > max_elim_len:
                continue
            backup = sol.clone()
            target_route = backup.route_list.pop(r_idx)
            custs = [node for node in target_route.node_list if _is_customer(node, data)]
            all_inserted = True
            for c in custs:
                if not _insert_customer_best_position(backup, c, data, allow_new_route=False):
                    all_inserted = False
                    break
            if all_inserted:
                backup.update(data)
                backup.cal_cost(data)
                sol = backup
                improved = True
                break
    return sol


def routes_to_solution(routes_list: List[List[int]], data) -> Solution:
    sol = Solution(data)
    for r in routes_list:
        if len(r) > 2:
            route = Route(data)
            route.node_list = list(r)
            route.update(data)
            sol.append(route)
    sol.update(data)
    sol.cal_cost(data)
    return sol


def solution_to_routes(sol: Solution) -> List[List[int]]:
    return [list(r.node_list) for r in sol.route_list if len(r.node_list) > 2]


def tensor_rcrs_grasp_init(
    P: int,
    data,
    backend,
    alpha_lo: float = 0.05,
    alpha_hi: float = 0.25,
    sa_iters: int = 25,
    run: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Route-by-route RCRS-GRASP initialization on GPU:
    Packs routes to saturation with spatial-temporal closeness and candidate evaluation.
    Starts population with low vehicle counts (matching BKS target) and low initial TD.
    """
    device = backend.device
    depot = backend.depot
    customer_count = data.customer_num
    max_routes = customer_count
    max_nodes = customer_count + 2
    dist = backend.dist_t

    pop_routes = torch.full((P, max_routes, max_nodes), depot, dtype=torch.long, device=device)
    pop_lengths = torch.full((P, max_routes), 2, dtype=torch.long, device=device)
    pop_counts = torch.zeros(P, dtype=torch.long, device=device)

    for p in range(P):
        rng = random.Random(int(data.seed + run * 100000 + p * 997))
        alpha = alpha_lo + rng.random() * (alpha_hi - alpha_lo)
        unassigned = torch.ones(customer_count + 1, dtype=torch.bool, device=device)
        unassigned[depot] = False
        r_idx = 0

        while unassigned.any():
            dists = torch.where(unassigned, dist[depot, :], torch.tensor(-1e9, device=device))
            top_k = min(4, int(unassigned.sum().item()))
            top_seeds = torch.topk(dists, k=top_k).indices.tolist()
            seed_c = rng.choice(top_seeds)
            route = [depot, seed_c, depot]
            unassigned[seed_c] = False

            while True:
                unassigned_list = torch.where(unassigned)[0].tolist()
                if not unassigned_list:
                    break
                sample_k = min(32, len(unassigned_list)) if customer_count >= 50 else len(unassigned_list)
                test_cands = rng.sample(unassigned_list, sample_k) if len(unassigned_list) > sample_k else unassigned_list

                cands = []
                cand_meta = []
                for c in test_cands:
                    for pos in range(1, len(route)):
                        cands.append(route[:pos] + [c] + route[pos:])
                        cand_meta.append((c, pos))

                cand_t = torch.full((len(cands), max(len(x) for x in cands)), depot, dtype=torch.long, device=device)
                lens_t = torch.tensor([len(x) for x in cands], dtype=torch.long, device=device)
                for i, x in enumerate(cands):
                    cand_t[i, :len(x)] = torch.tensor(x, device=device)

                feas, total_d = backend.evaluate_routes_gpu(cand_t, lens_t)
                if not feas.any():
                    break

                feas_idx = torch.where(feas)[0]
                scores = total_d[feas_idx]
                min_s = scores.min().item()
                max_s = scores.max().item()
                thresh = min_s + alpha * (max_s - min_s)

                rcl = feas_idx[scores <= thresh + 1e-4].tolist()
                chosen = rng.choice(rcl)
                best_c, best_pos = cand_meta[chosen]
                route.insert(best_pos, best_c)
                unassigned[best_c] = False

            l = len(route)
            pop_routes[p, r_idx, :l] = torch.tensor(route, device=device)
            pop_lengths[p, r_idx] = l
            r_idx += 1

        pop_counts[p] = r_idx

    return pop_routes, pop_lengths, pop_counts


def tensor_route_elimination_population(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    backend,
    generator: Optional[torch.Generator] = None,
    max_elim_len: int = 15,
    passes: int = 35,
    k_tries: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure GPU Vectorized Route Ejection Engine:
    Targets routes with the fewest customers and relocates their customers into remaining routes.
    Tests k_tries candidate insertions per pass across the population in parallel.
    Guarantees 100% feasibility and strict preservation of all customers while driving NV down.
    """
    P, R, L = pop_routes.shape
    device = pop_routes.device
    indices = torch.arange(P, device=device)
    offsets = torch.arange(L, device=device).view(1, -1)
    if generator is None:
        generator = torch.Generator(device=device)

    current_routes = pop_routes.clone()
    current_lengths = pop_lengths.clone()
    current_counts = pop_route_counts.clone()

    feasible, _, vehicle_counts, total_distances = backend.evaluate_population_tensor(
        current_routes, current_lengths, current_counts
    )

    for _ in range(passes):
        lens_active = torch.where(
            (current_lengths > 2) & (current_lengths <= max_elim_len + 2),
            current_lengths,
            torch.tensor(9999, device=device, dtype=torch.long),
        )
        source_ids = torch.argmin(lens_active, dim=1)
        source_lens = current_lengths[indices, source_ids]
        has_target = (source_lens > 2) & (source_lens <= max_elim_len + 2) & (vehicle_counts > 1)
        if not has_target.any():
            break

        source_pos = 1 + torch.randint(L, (P,), device=device, generator=generator) % (source_lens - 2).clamp_min(1)
        source_route = current_routes[indices, source_ids]
        moved_node = source_route.gather(1, source_pos.view(-1, 1)).squeeze(1)

        source_indices = torch.where(
            offsets < source_pos.view(-1, 1),
            offsets,
            offsets + 1,
        ).clamp_max(L - 1)
        source_cand = torch.gather(source_route, 1, source_indices)

        found_move = torch.zeros(P, dtype=torch.bool, device=device)

        for _ in range(k_tries):
            target_ids = torch.randint(R, (P,), device=device, generator=generator) % current_counts.clamp_min(1)
            same_route = target_ids == source_ids
            target_ids = torch.where(same_route, (target_ids + 1) % current_counts.clamp_min(1), target_ids)
            target_lens = current_lengths[indices, target_ids]
            target_route = current_routes[indices, target_ids]

            valid_move = has_target & (target_ids != source_ids) & (target_lens > 2) & (target_lens + 1 < L) & (~found_move)
            if not valid_move.any():
                continue

            target_pos = 1 + torch.randint(L, (P,), device=device, generator=generator) % (target_lens - 1).clamp_min(1)
            target_indices = torch.where(
                offsets < target_pos.view(-1, 1),
                offsets,
                offsets - 1,
            ).clamp_min(0)
            target_cand = torch.gather(target_route, 1, target_indices)
            target_cand.scatter_(1, target_pos.view(-1, 1), moved_node.view(-1, 1))

            cand_pop = current_routes.clone()
            cand_pop[indices, source_ids] = torch.where(valid_move.view(-1, 1), source_cand, source_route)
            cand_pop[indices, target_ids] = torch.where(valid_move.view(-1, 1), target_cand, target_route)
            cand_lens = current_lengths.clone()
            cand_lens[indices, source_ids] = torch.where(valid_move, source_lens - 1, source_lens)
            cand_lens[indices, target_ids] = torch.where(valid_move, target_lens + 1, target_lens)

            cand_feas, _, cand_nv, cand_dist = backend.evaluate_population_tensor(
                cand_pop, cand_lens, current_counts
            )

            accept = valid_move & cand_feas & (
                (cand_nv < vehicle_counts)
                | ((cand_nv == vehicle_counts) & (cand_dist < total_distances + 25.0))
            )
            if accept.any():
                found_move |= accept
                current_routes = torch.where(accept.view(P, 1, 1), cand_pop, current_routes)
                current_lengths = torch.where(accept.view(P, 1), cand_lens, current_lengths)
                vehicle_counts = torch.where(accept, cand_nv, vehicle_counts)
                total_distances = torch.where(accept, cand_dist, total_distances)

    return current_routes, current_lengths, current_counts


def tensor_sa_warmup(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    backend,
    data,
    sa_iters: int = 25,
    temp_init: float = 50.0,
    temp_min: float = 0.5,
    cooling: float = 0.50,
    cuda_rng: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tightens initial population on GPU: applies Route Ejection, 2-opt, relocate, and swap in pure GPU tensors.
    """
    device = backend.device
    rng = cuda_rng or torch.Generator(device=device)
    pop_routes, pop_lengths, pop_route_counts = tensor_route_elimination_population(
        pop_routes, pop_lengths, pop_route_counts, backend, generator=rng, max_elim_len=15, passes=35, k_tries=8
    )
    pop_routes, pop_lengths, pop_route_counts = tensor_local_search_batch(
        pop_routes, pop_lengths, pop_route_counts, backend, generator=rng, passes=25
    )
    pop_routes, pop_lengths, pop_route_counts = tensor_relocate_batch(
        pop_routes, pop_lengths, pop_route_counts, backend, generator=rng, passes=20
    )
    pop_routes, pop_lengths, pop_route_counts = tensor_swap_batch(
        pop_routes, pop_lengths, pop_route_counts, backend, generator=rng, passes=15
    )
    return pop_routes, pop_lengths, pop_route_counts


def tensor_generate_offspring_batch(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    backend,
    generator: torch.Generator,
    peer_indices: Optional[torch.Tensor] = None,
    elite_indices: Optional[torch.Tensor] = None,
    hybrid_probability: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Vectorized SHO Guided Crossover + WOA Intensification.
    Guarantees 100% customer coverage preservation (no duplicate/missing customers).
    Applies parallel 2-opt intra-route reversal, inter-route customer swap,
    and targeted relocation from shortest route on GPU tensors.
    """
    P, R, L = pop_routes.shape
    device = pop_routes.device
    indices = torch.arange(P, device=device)
    if peer_indices is None:
        peer_indices = torch.arange(P, device=device)
    if elite_indices is None:
        elite_indices = torch.arange(P, device=device)

    use_elite = torch.rand(P, device=device, generator=generator) < (1.0 - hybrid_probability)
    parent_indices = torch.where(use_elite, elite_indices, peer_indices)
    offspring = pop_routes[parent_indices].clone()
    offspring_lengths = pop_lengths[parent_indices].clone()
    offspring_counts = pop_route_counts[parent_indices].clone()

    route_one = torch.randint(R, (P,), device=device, generator=generator) % offspring_counts.clamp_min(1)
    route_two = (route_one + 1 + torch.randint(max(1, R - 1), (P,), device=device, generator=generator)) % offspring_counts.clamp_min(1)

    len_one = offspring_lengths[indices, route_one]
    len_two = offspring_lengths[indices, route_two]

    # Move 1: Intra-route 2-opt reversal (~50% of the population)
    do_reversal = torch.rand(P, device=device, generator=generator) < 0.5
    pos_a = 1 + torch.randint(L, (P,), device=device, generator=generator) % (len_one - 2).clamp_min(1)
    pos_b = 1 + torch.randint(L, (P,), device=device, generator=generator) % (len_one - 2).clamp_min(1)
    lo = torch.minimum(pos_a, pos_b)
    hi = torch.maximum(pos_a, pos_b)
    valid_rev = do_reversal & (len_one > 3) & (lo < hi) & (hi < len_one - 1)

    route1_nodes = offspring[indices, route_one]
    offsets = torch.arange(L, device=device).view(1, -1)
    rev_mask = (offsets >= lo.view(-1, 1)) & (offsets <= hi.view(-1, 1))
    rev_indices = torch.where(rev_mask, lo.view(-1, 1) + hi.view(-1, 1) - offsets, offsets)
    rev_route = torch.gather(route1_nodes, 1, rev_indices)
    offspring[indices, route_one] = torch.where(valid_rev.view(-1, 1), rev_route, route1_nodes)

    # Move 2: Inter-route customer swap (~50% of the population)
    do_swap = ~do_reversal & (route_one != route_two) & (len_one > 2) & (len_two > 2)
    pos_one = 1 + torch.randint(L, (P,), device=device, generator=generator) % (len_one - 2).clamp_min(1)
    pos_two = 1 + torch.randint(L, (P,), device=device, generator=generator) % (len_two - 2).clamp_min(1)
    first_cust = offspring[indices, route_one, pos_one]
    second_cust = offspring[indices, route_two, pos_two]
    first_new = torch.where(do_swap, second_cust, first_cust)
    second_new = torch.where(do_swap, first_cust, second_cust)
    offspring[indices, route_one, pos_one] = first_new
    offspring[indices, route_two, pos_two] = second_new

    # Move 3: Targeted Shortest-Route Ejection (~40% of the population)
    do_eject = torch.rand(P, device=device, generator=generator) < 0.40
    active_lens = torch.where(offspring_lengths > 2, offspring_lengths, torch.tensor(9999, device=device, dtype=torch.long))
    shortest_r = torch.argmin(active_lens, dim=1)
    shortest_len = offspring_lengths[indices, shortest_r]
    dst_r = torch.randint(R, (P,), device=device, generator=generator) % offspring_counts.clamp_min(1)
    dst_r = torch.where(dst_r == shortest_r, (dst_r + 1) % offspring_counts.clamp_min(1), dst_r)
    dst_len = offspring_lengths[indices, dst_r]

    valid_eject = do_eject & (shortest_len > 2) & (shortest_len <= 15) & (dst_len > 2) & (dst_len + 1 < L)
    eject_pos = 1 + torch.randint(L, (P,), device=device, generator=generator) % (shortest_len - 2).clamp_min(1)
    ejected_cust = offspring[indices, shortest_r, eject_pos]

    src_nodes = offspring[indices, shortest_r]
    src_indices = torch.where(offsets < eject_pos.view(-1, 1), offsets, offsets + 1).clamp_max(L - 1)
    new_src = torch.gather(src_nodes, 1, src_indices)

    insert_pos = 1 + torch.randint(L, (P,), device=device, generator=generator) % (dst_len - 1).clamp_min(1)
    dst_nodes = offspring[indices, dst_r]
    dst_indices = torch.where(offsets < insert_pos.view(-1, 1), offsets, offsets - 1).clamp_min(0)
    new_dst = torch.gather(dst_nodes, 1, dst_indices)
    new_dst.scatter_(1, insert_pos.view(-1, 1), ejected_cust.view(-1, 1))

    offspring[indices, shortest_r] = torch.where(valid_eject.view(-1, 1), new_src, offspring[indices, shortest_r])
    offspring_lengths[indices, shortest_r] = torch.where(valid_eject, shortest_len - 1, shortest_len)
    offspring[indices, dst_r] = torch.where(valid_eject.view(-1, 1), new_dst, offspring[indices, dst_r])
    offspring_lengths[indices, dst_r] = torch.where(valid_eject, dst_len + 1, dst_len)

    return offspring, offspring_lengths, offspring_counts


def tensor_local_search_batch(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    backend,
    generator: Optional[torch.Generator] = None,
    passes: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply pure GPU-only vectorized 2-opt passes to a population batch."""
    population, route_slots, max_nodes = pop_routes.shape
    device = pop_routes.device
    indices = torch.arange(population, device=device)
    if generator is None:
        generator = torch.Generator(device=device)

    current_routes = pop_routes
    current_lengths = pop_lengths
    current_counts = pop_route_counts
    feasible, _, vehicle_counts, total_distances = backend.evaluate_population_tensor(
        current_routes, current_lengths, current_counts
    )

    for _ in range(max(1, passes)):
        route_ids = torch.randint(
            route_slots, (population,), device=device, generator=generator
        ) % current_counts.clamp_min(1)
        selected_lengths = current_lengths[indices, route_ids]
        left = 1 + torch.randint(
            max_nodes, (population,), device=device, generator=generator
        ) % (selected_lengths - 2).clamp_min(1)
        right = 1 + torch.randint(
            max_nodes, (population,), device=device, generator=generator
        ) % (selected_lengths - 2).clamp_min(1)
        lo = torch.minimum(left, right)
        hi = torch.maximum(left, right)
        valid_move = (
            (route_ids < current_counts)
            & (selected_lengths > 3)
            & (lo < hi)
            & (hi < (selected_lengths - 1))
        )

        selected_routes = current_routes[indices, route_ids]
        offsets = torch.arange(max_nodes, device=device).view(1, -1)
        reverse_mask = (offsets >= lo.view(-1, 1)) & (offsets <= hi.view(-1, 1))
        reverse_indices = torch.where(
            reverse_mask,
            lo.view(-1, 1) + hi.view(-1, 1) - offsets,
            offsets,
        )
        candidate_routes = torch.gather(selected_routes, 1, reverse_indices)
        candidate_population = current_routes.clone()
        candidate_population[indices, route_ids] = torch.where(
            valid_move.view(-1, 1), candidate_routes, selected_routes
        )

        candidate_feasible, _, candidate_counts, candidate_distances = (
            backend.evaluate_population_tensor(
                candidate_population, current_lengths, current_counts
            )
        )
        improves = candidate_feasible & (
            (candidate_counts < vehicle_counts)
            | (
                (candidate_counts == vehicle_counts)
                & (candidate_distances < total_distances - 1e-4)
            )
        )
        improves &= valid_move
        current_routes = torch.where(
            improves.view(population, 1, 1), candidate_population, current_routes
        )
        feasible = torch.where(improves, candidate_feasible, feasible)
        vehicle_counts = torch.where(improves, candidate_counts, vehicle_counts)
        total_distances = torch.where(improves, candidate_distances, total_distances)

    return current_routes, current_lengths, current_counts


def tensor_relocate_batch(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    backend,
    generator: Optional[torch.Generator] = None,
    passes: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Try improving inter-route customer relocations on pure CUDA tensors."""
    population, route_slots, max_nodes = pop_routes.shape
    device = pop_routes.device
    indices = torch.arange(population, device=device)
    if generator is None:
        generator = torch.Generator(device=device)

    current_routes = pop_routes
    current_lengths = pop_lengths
    current_counts = pop_route_counts
    feasible, _, vehicle_counts, total_distances = backend.evaluate_population_tensor(
        current_routes, current_lengths, current_counts
    )

    for _ in range(max(1, passes)):
        target_shortest = torch.rand(population, device=device, generator=generator) < 0.5
        active_lens = torch.where(current_lengths > 2, current_lengths, torch.tensor(9999, device=device, dtype=torch.long))
        shortest_ids = torch.argmin(active_lens, dim=1)
        rand_ids = torch.randint(route_slots, (population,), device=device, generator=generator) % current_counts.clamp_min(1)
        source_ids = torch.where(target_shortest, shortest_ids, rand_ids)
        target_ids = torch.randint(route_slots, (population,), device=device, generator=generator) % current_counts.clamp_min(1)
        same_r = target_ids == source_ids
        target_ids = torch.where(same_r, (target_ids + 1) % current_counts.clamp_min(1), target_ids)

        source_lengths = current_lengths[indices, source_ids]
        target_lengths = current_lengths[indices, target_ids]
        source_pos = 1 + torch.randint(max_nodes, (population,), device=device, generator=generator) % (source_lengths - 2).clamp_min(1)
        target_pos = 1 + torch.randint(max_nodes, (population,), device=device, generator=generator) % (target_lengths - 1).clamp_min(1)
        valid_move = (
            (source_ids != target_ids)
            & (source_ids < current_counts)
            & (target_ids < current_counts)
            & (source_lengths > 2)
            & (target_lengths > 2)
            & (source_pos < (source_lengths - 1))
            & (target_pos < target_lengths)
            & (target_lengths + 1 < max_nodes)
        )

        source_route = current_routes[indices, source_ids]
        target_route = current_routes[indices, target_ids]
        source_offsets = torch.arange(max_nodes, device=device).view(1, -1)
        source_indices = torch.where(
            source_offsets < source_pos.view(-1, 1),
            source_offsets,
            source_offsets + 1,
        ).clamp_max(max_nodes - 1)
        source_candidate = torch.gather(source_route, 1, source_indices)
        moved_node = source_route.gather(1, source_pos.view(-1, 1)).squeeze(1)

        target_output = torch.arange(max_nodes, device=device).view(1, -1)
        target_indices = torch.where(
            target_output < target_pos.view(-1, 1),
            target_output,
            target_output - 1,
        ).clamp_min(0)
        target_candidate = torch.gather(target_route, 1, target_indices)
        target_candidate.scatter_(1, target_pos.view(-1, 1), moved_node.view(-1, 1))

        candidate_population = current_routes.clone()
        candidate_population[indices, source_ids] = torch.where(
            valid_move.view(-1, 1), source_candidate, source_route
        )
        candidate_population[indices, target_ids] = torch.where(
            valid_move.view(-1, 1), target_candidate, target_route
        )
        candidate_lengths = current_lengths.clone()
        candidate_lengths[indices, source_ids] = torch.where(
            valid_move, source_lengths - 1, source_lengths
        )
        candidate_lengths[indices, target_ids] = torch.where(
            valid_move, target_lengths + 1, target_lengths
        )

        candidate_feasible, _, candidate_counts, candidate_distances = (
            backend.evaluate_population_tensor(
                candidate_population, candidate_lengths, current_counts
            )
        )
        improves = candidate_feasible & (
            (candidate_counts < vehicle_counts)
            | (
                (candidate_counts == vehicle_counts)
                & (candidate_distances < total_distances - 1e-4)
            )
        )
        improves &= valid_move
        current_routes = torch.where(
            improves.view(population, 1, 1), candidate_population, current_routes
        )
        current_lengths = torch.where(
            improves.view(population, 1), candidate_lengths, current_lengths
        )
        feasible = torch.where(improves, candidate_feasible, feasible)
        vehicle_counts = torch.where(improves, candidate_counts, vehicle_counts)
        total_distances = torch.where(improves, candidate_distances, total_distances)

    return current_routes, current_lengths, current_counts


def tensor_swap_batch(
    pop_routes: torch.Tensor,
    pop_lengths: torch.Tensor,
    pop_route_counts: torch.Tensor,
    backend,
    generator: Optional[torch.Generator] = None,
    passes: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% Pure GPU Vectorized Inter-Route Swap (2-exchange):
    Exchanges one customer from route A with one customer from route B.
    Dramatically reduces total travel distance (TD) by untangling crossing routes.
    """
    population, route_slots, max_nodes = pop_routes.shape
    device = pop_routes.device
    indices = torch.arange(population, device=device)
    if generator is None:
        generator = torch.Generator(device=device)

    current_routes = pop_routes
    current_lengths = pop_lengths
    current_counts = pop_route_counts

    feasible, _, vehicle_counts, total_distances = backend.evaluate_population_tensor(
        current_routes, current_lengths, current_counts
    )

    for _ in range(max(1, passes)):
        r1 = torch.randint(route_slots, (population,), device=device, generator=generator) % current_counts.clamp_min(1)
        r2 = torch.randint(route_slots, (population,), device=device, generator=generator) % current_counts.clamp_min(1)
        same = r1 == r2
        r2 = torch.where(same, (r2 + 1) % current_counts.clamp_min(1), r2)

        l1 = current_lengths[indices, r1]
        l2 = current_lengths[indices, r2]

        pos1 = 1 + torch.randint(max_nodes, (population,), device=device, generator=generator) % (l1 - 2).clamp_min(1)
        pos2 = 1 + torch.randint(max_nodes, (population,), device=device, generator=generator) % (l2 - 2).clamp_min(1)

        valid_move = (
            (r1 != r2)
            & (r1 < current_counts)
            & (r2 < current_counts)
            & (l1 > 2)
            & (l2 > 2)
            & (pos1 < (l1 - 1))
            & (pos2 < (l2 - 1))
        )

        c1 = current_routes[indices, r1, pos1]
        c2 = current_routes[indices, r2, pos2]

        cand_pop = current_routes.clone()
        cand_pop[indices, r1, pos1] = torch.where(valid_move, c2, c1)
        cand_pop[indices, r2, pos2] = torch.where(valid_move, c1, c2)

        cand_feas, _, cand_nv, cand_dist = backend.evaluate_population_tensor(
            cand_pop, current_lengths, current_counts
        )

        improves = valid_move & cand_feas & (
            (cand_nv < vehicle_counts)
            | (
                (cand_nv == vehicle_counts)
                & (cand_dist < total_distances - 1e-4)
            )
        )
        current_routes = torch.where(improves.view(population, 1, 1), cand_pop, current_routes)
        vehicle_counts = torch.where(improves, cand_nv, vehicle_counts)
        total_distances = torch.where(improves, cand_dist, total_distances)

    return current_routes, current_lengths, current_counts


def batched_insert_customer_gpu(
    routes: List[List[int]],
    node: int,
    backend,
    data,
    max_nv: Optional[int] = None
) -> bool:
    """
    Evaluates all possible insertion positions for node across ALL existing routes
    in 1 single batch on GPU. Subtracts base route distance to calculate exact delta TD.
    Inserts at the position with minimal delta TD.
    Opens a new route ONLY if no feasible position exists AND max_nv is not exceeded.
    """
    device = backend.device
    depot = data.DC

    if not routes:
        routes.append([depot, node, depot])
        return True

    # Pre-calculate base distances of all routes
    max_orig_len = max(len(r) for r in routes)
    orig_batch = torch.full((len(routes), max_orig_len), depot, dtype=torch.long, device=device)
    orig_lens = torch.tensor([len(r) for r in routes], dtype=torch.long, device=device)
    for i, r in enumerate(routes):
        orig_batch[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)
    _, orig_dists = backend.evaluate_routes_gpu(orig_batch, orig_lens)

    cands = []
    cands_meta = []
    for r_i, r in enumerate(routes):
        for pos in range(1, len(r)):
            cand_r = r[:pos] + [node] + r[pos:]
            cands.append(cand_r)
            cands_meta.append((r_i, pos))

    if not cands:
        if max_nv is None or len(routes) < max_nv:
            routes.append([depot, node, depot])
            return True
        return False

    M = len(cands)
    max_len = max(len(c) for c in cands)
    batch_t = torch.full((M, max_len), depot, dtype=torch.long, device=device)
    lengths_t = torch.tensor([len(c) for c in cands], dtype=torch.long, device=device)
    for i, c in enumerate(cands):
        batch_t[i, :len(c)] = torch.tensor(c, dtype=torch.long, device=device)

    f, dists = backend.evaluate_routes_gpu(batch_t, lengths_t)
    r_indices = torch.tensor([m[0] for m in cands_meta], dtype=torch.long, device=device)
    deltas = dists - orig_dists[r_indices]
    deltas = torch.where(f, deltas, torch.tensor(float('inf'), device=device))
    best_delta, best_idx_t = torch.min(deltas, dim=0)

    if torch.isfinite(best_delta).item():
        best_idx = best_idx_t.item()
        r_i, pos = cands_meta[best_idx]
        routes[r_i].insert(pos, node)
        return True
    elif max_nv is None or len(routes) < max_nv:
        routes.append([depot, node, depot])
        return True
    return False


def batched_insert_customers_sequence_gpu(
    routes: List[List[int]],
    customers: List[int],
    backend,
    data,
    max_nv: Optional[int] = None,
    rng: Optional[random.Random] = None
) -> None:
    pending = list(customers)
    if rng is not None:
        rng.shuffle(pending)
    for node in pending:
        batched_insert_customer_gpu(routes, node, backend, data, max_nv=max_nv)


def tensor_route_elimination_single(
    child_routes: List[List[int]],
    backend,
    data,
    max_elim_len: int = 4
) -> List[List[int]]:
    """
    CRITICAL: Actively absorbs sparse routes into other routes using GPU batch evaluations
    to strictly drive down the Number of Vehicles (NV).
    Guarantees that 100% of customers are strictly preserved.
    """
    clean_routes = [list(r) for r in child_routes if len(r) > 2]
    if len(clean_routes) <= 1:
        return clean_routes

    orig_customers = set(n for r in clean_routes for n in r if _is_customer(n, data))

    improved = True
    while improved and len(clean_routes) > 1:
        improved = False
        sorted_indices = sorted(range(len(clean_routes)), key=lambda idx: len(clean_routes[idx]))
        for target_idx in sorted_indices:
            target_route = clean_routes[target_idx]
            custs = [n for n in target_route if _is_customer(n, data)]
            if len(custs) > max_elim_len:
                continue

            other_routes = [list(clean_routes[i]) for i in range(len(clean_routes)) if i != target_idx]
            can_reinsert_all = True
            temp_others = [list(r) for r in other_routes]

            for node in custs:
                if not batched_insert_customer_gpu(temp_others, node, backend, data, max_nv=len(other_routes)):
                    can_reinsert_all = False
                    break

            if can_reinsert_all:
                new_custs = set(n for r in temp_others for n in r if _is_customer(n, data))
                if new_custs == orig_customers:
                    clean_routes = [r for r in temp_others if len(r) > 2]
                    improved = True
                    break

    return clean_routes


def tensor_2opt_route_gpu(route: List[int], backend, data) -> List[int]:
    """
    Fast 2-opt edge reversal with geometric pruning on GPU.
    """
    L = len(route)
    if L <= 3:
        return route
    device = backend.device
    depot = data.DC
    improved = True
    dist_mat = data.dist
    while improved:
        improved = False
        cands = []
        cands_ij = []
        for i in range(1, L - 2):
            for j in range(i + 1, L - 1):
                delta_geom = (
                    dist_mat[route[i - 1]][route[j]] + dist_mat[route[i]][route[j + 1]]
                    - (dist_mat[route[i - 1]][route[i]] + dist_mat[route[j]][route[j + 1]])
                )
                if delta_geom < -1e-4:
                    cand = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                    cands.append(cand)
                    cands_ij.append((i, j, cand, delta_geom))

        if not cands:
            break

        M = len(cands)
        batch_t = torch.full((M, L), depot, dtype=torch.long, device=device)
        lengths_t = torch.full((M,), L, dtype=torch.long, device=device)
        for idx, c in enumerate(cands):
            batch_t[idx] = torch.tensor(c, dtype=torch.long, device=device)

        f, d = backend.evaluate_routes_gpu(batch_t, lengths_t)
        d_masked = torch.where(f, d, torch.tensor(float('inf'), device=device))
        best_d, best_k_t = torch.min(d_masked, dim=0)

        if torch.isfinite(best_d).item():
            best_k = best_k_t.item()
            route = cands_ij[best_k][2]
            improved = True
    return route


def tensor_inter_route_relocate_gpu(
    child_routes: List[List[int]],
    backend,
    data,
    rng: random.Random
) -> List[List[int]]:
    if len(child_routes) < 2:
        return child_routes

    r_cnt = len(child_routes)
    r1, r2 = rng.sample(range(r_cnt), 2)
    nl1 = child_routes[r1]
    nl2 = child_routes[r2]
    if len(nl1) < 4:
        return child_routes

    idx = rng.randint(1, len(nl1) - 2)
    node = nl1[idx]
    nl1_cand = nl1[:idx] + nl1[idx + 1 :]

    r1_t = torch.tensor([nl1_cand], dtype=torch.long, device=backend.device)
    l1_t = torch.tensor([len(nl1_cand)], dtype=torch.long, device=backend.device)
    f1, d1 = backend.evaluate_routes_gpu(r1_t, l1_t)
    if not f1[0].item():
        return child_routes

    cands = []
    cands_meta = []
    for pos in range(1, len(nl2)):
        cands.append(nl2[:pos] + [node] + nl2[pos:])
        cands_meta.append(pos)

    M = len(cands)
    max_l = max(len(c) for c in cands)
    batch_t = torch.full((M, max_l), backend.depot, dtype=torch.long, device=backend.device)
    lengths_t = torch.tensor([len(c) for c in cands], dtype=torch.long, device=backend.device)
    for k, c in enumerate(cands):
        batch_t[k, :len(c)] = torch.tensor(c, dtype=torch.long, device=backend.device)

    f2, d2 = backend.evaluate_routes_gpu(batch_t, lengths_t)

    orig_t = torch.zeros((2, max(len(nl1), len(nl2))), dtype=torch.long, device=backend.device)
    orig_t[0, :len(nl1)] = torch.tensor(nl1, dtype=torch.long, device=backend.device)
    orig_t[1, :len(nl2)] = torch.tensor(nl2, dtype=torch.long, device=backend.device)
    orig_lens = torch.tensor([len(nl1), len(nl2)], dtype=torch.long, device=backend.device)
    _, orig_d = backend.evaluate_routes_gpu(orig_t, orig_lens)
    orig_total = orig_d[0] + orig_d[1]

    new_totals = d1[0] + d2
    new_totals = torch.where(f2, new_totals, torch.tensor(float('inf'), device=backend.device))
    best_d, best_k_t = torch.min(new_totals, dim=0)

    if (best_d < orig_total - 1e-4).item():
        best_pos = cands_meta[best_k_t.item()]
        child_routes[r1] = nl1_cand
        child_routes[r2].insert(best_pos, node)

    return [r for r in child_routes if len(r) > 2]


def tensor_inter_route_swap_gpu(
    child_routes: List[List[int]],
    backend,
    data,
    rng: random.Random
) -> List[List[int]]:
    if len(child_routes) < 2:
        return child_routes

    r_cnt = len(child_routes)
    r1, r2 = rng.sample(range(r_cnt), 2)
    nl1 = child_routes[r1]
    nl2 = child_routes[r2]
    if len(nl1) < 4 or len(nl2) < 4:
        return child_routes

    idx1 = rng.randint(1, len(nl1) - 2)
    idx2 = rng.randint(1, len(nl2) - 2)
    cand1 = list(nl1)
    cand2 = list(nl2)
    cand1[idx1], cand2[idx2] = cand2[idx2], cand1[idx1]

    pair_t = torch.zeros((2, max(len(cand1), len(cand2))), dtype=torch.long, device=backend.device)
    pair_t[0, :len(cand1)] = torch.tensor(cand1, dtype=torch.long, device=backend.device)
    pair_t[1, :len(cand2)] = torch.tensor(cand2, dtype=torch.long, device=backend.device)
    pair_lens = torch.tensor([len(cand1), len(cand2)], dtype=torch.long, device=backend.device)
    f, d = backend.evaluate_routes_gpu(pair_t, pair_lens)

    if f[0].item() and f[1].item():
        orig_t = torch.zeros((2, max(len(nl1), len(nl2))), dtype=torch.long, device=backend.device)
        orig_t[0, :len(nl1)] = torch.tensor(nl1, dtype=torch.long, device=backend.device)
        orig_t[1, :len(nl2)] = torch.tensor(nl2, dtype=torch.long, device=backend.device)
        orig_lens = torch.tensor([len(nl1), len(nl2)], dtype=torch.long, device=backend.device)
        _, d_orig = backend.evaluate_routes_gpu(orig_t, orig_lens)
        if (d[0].item() + d[1].item()) < (d_orig[0].item() + d_orig[1].item()) - 1e-4:
            child_routes[r1] = cand1
            child_routes[r2] = cand2

    return child_routes


def tensor_inter_route_relocate_all(routes: List[List[int]], backend, data) -> List[List[int]]:
    """
    Exhaustive inter-route relocation search on GPU.
    Tests moving every customer in every route to the best position in all other routes.
    """
    if not routes or len(routes) < 2:
        return routes

    max_l = max(len(r) for r in routes)
    batch_t = torch.full((len(routes), max_l), backend.depot, dtype=torch.long, device=backend.device)
    lens_t = torch.tensor([len(r) for r in routes], dtype=torch.long, device=backend.device)
    for i, r in enumerate(routes):
        batch_t[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=backend.device)
    _, dists = backend.evaluate_routes_gpu(batch_t, lens_t)
    route_dists = dists.tolist()

    improved = True
    while improved:
        improved = False
        for r1_idx in range(len(routes)):
            if len(routes[r1_idx]) <= 3:
                continue
            for pos1 in range(1, len(routes[r1_idx]) - 1):
                node = routes[r1_idx][pos1]
                cand_r1 = routes[r1_idx][:pos1] + routes[r1_idx][pos1 + 1 :]

                r1_t = torch.tensor([cand_r1], dtype=torch.long, device=backend.device)
                l1_t = torch.tensor([len(cand_r1)], dtype=torch.long, device=backend.device)
                f1, d1 = backend.evaluate_routes_gpu(r1_t, l1_t)
                if not f1[0].item():
                    continue

                delta_r1 = d1[0].item() - route_dists[r1_idx]

                cands = []
                cands_meta = []
                for r2_idx in range(len(routes)):
                    if r2_idx == r1_idx:
                        continue
                    r2 = routes[r2_idx]
                    for pos2 in range(1, len(r2)):
                        cands.append(r2[:pos2] + [node] + r2[pos2:])
                        cands_meta.append((r2_idx, pos2))

                if not cands:
                    continue

                M = len(cands)
                max_cand_l = max(len(c) for c in cands)
                batch_cand = torch.full((M, max_cand_l), backend.depot, dtype=torch.long, device=backend.device)
                lens_cand = torch.tensor([len(c) for c in cands], dtype=torch.long, device=backend.device)
                for idx, c in enumerate(cands):
                    batch_cand[idx, :len(c)] = torch.tensor(c, dtype=torch.long, device=backend.device)

                f2, d2 = backend.evaluate_routes_gpu(batch_cand, lens_cand)
                f2_list = f2.tolist()
                d2_list = d2.tolist()

                best_move = None
                best_net_delta = -1e-4
                for k in range(M):
                    if f2_list[k]:
                        r2_idx, pos2 = cands_meta[k]
                        delta_r2 = d2_list[k] - route_dists[r2_idx]
                        net_delta = delta_r1 + delta_r2
                        if net_delta < best_net_delta:
                            best_net_delta = net_delta
                            best_move = (r2_idx, pos2, cands[k], d1[0].item(), d2_list[k])

                if best_move is not None:
                    r2_idx, pos2, new_r2, new_d1, new_d2 = best_move
                    routes[r1_idx] = cand_r1
                    routes[r2_idx] = new_r2
                    route_dists[r1_idx] = new_d1
                    route_dists[r2_idx] = new_d2
                    improved = True
                    break
            if improved:
                break
    return routes


def tensor_sho_crossover_single(
    best_routes_list: List[List[int]],
    peer_routes_list: List[List[int]],
    cur_routes_list: List[List[int]],
    backend,
    data,
    rng: random.Random,
    max_nv: Optional[int] = None,
    mutation_prob: float = 0.35,
) -> List[List[int]]:
    """
    SHO Guided Route Crossover (Algorithm 4 & 5) + Algorithm 10 repair + Active Route Elimination.
    """
    sol_best = routes_to_solution(best_routes_list, data)
    sol_peer = routes_to_solution(peer_routes_list, data)
    sol_cur = routes_to_solution(cur_routes_list, data)

    # Paper SHO Guided Route Crossover with segment injection & repair
    child_sol = _guided_route_crossover(sol_best, sol_peer, sol_cur, data, rng)
    child_sol = active_route_elimination(child_sol, data)

    child_routes = solution_to_routes(child_sol)
    child_routes = tensor_route_elimination_single(child_routes, backend, data)

    # SHO Mutation
    if rng.random() < mutation_prob:
        if rng.random() < 0.5:
            child_routes = tensor_inter_route_relocate_gpu(child_routes, backend, data, rng)
        else:
            child_routes = tensor_inter_route_swap_gpu(child_routes, backend, data, rng)

    child_routes = [tensor_2opt_route_gpu(r, backend, data) for r in child_routes]
    return child_routes


def tensor_woa_intensification_single(
    cur_routes_list: List[List[int]],
    best_routes_list: List[List[int]],
    a: float,
    backend,
    data,
    rng: random.Random,
    max_nv: Optional[int] = None,
) -> List[List[int]]:
    """
    WOA Intensification (Algorithm 6 & 7) + Algorithm 10 repair + Active Route Elimination.
    """
    sol_cur = routes_to_solution(cur_routes_list, data)
    sol_best = routes_to_solution(best_routes_list, data)

    child_sol = _woa_intensification(sol_cur, sol_best, a, data, rng)
    child_sol = active_route_elimination(child_sol, data)

    child_routes = solution_to_routes(child_sol)
    child_routes = tensor_route_elimination_single(child_routes, backend, data)
    child_routes = [tensor_2opt_route_gpu(r, backend, data) for r in child_routes]
    return child_routes


def tensor_deep_local_search_gpu(
    best_routes: List[List[int]],
    backend,
    data,
    rng: Optional[random.Random] = None
) -> List[List[int]]:
    """
    Deep local search on global best solution:
    1. Active route elimination to crush NV.
    2. Paper local search engine (find_local_optima with 2opt, 2opt*, oropt, 2exchange).
    3. GPU tensor inter-route relocation & swaps.
    4. GPU 2-opt.
    """
    sol = routes_to_solution(best_routes, data)
    sol = active_route_elimination(sol, data, max_elim_len=5)

    if getattr(data, "small_opts", None) and getattr(data, "mem", None):
        try:
            data.clear_mem()
            find_local_optima(sol, data)
            sol.cal_cost(data)
        except Exception:
            pass

    routes = solution_to_routes(sol)
    routes = tensor_route_elimination_single(routes, backend, data)
    routes = tensor_inter_route_relocate_all(routes, backend, data)
    if rng is not None:
        for _ in range(2):
            routes = tensor_inter_route_swap_gpu(routes, backend, data, rng)
    routes = [tensor_2opt_route_gpu(r, backend, data) for r in routes]
    return routes


# =============================================================================
# Crossover, WOA & Repair Recombination (Paper Algs 4, 5, 6, 7, 10)
# =============================================================================

def _build_solution_from_sequence(sequence: List[int], data) -> Solution:
    s = Solution(data)
    route_nodes: List[int] = []
    for node in sequence:
        trial = [data.DC] + route_nodes + [node, data.DC]
        if check_route_capacity(trial, data):
            flag, _ = _chk_route_list(trial, data)
            if flag:
                route_nodes.append(node)
                continue
        if route_nodes:
            r = Route(data)
            r.node_list = [data.DC] + route_nodes + [data.DC]
            r.update(data)
            s.append(r)
        route_nodes = []
        single = [data.DC, node, data.DC]
        flag, _ = _chk_route_list(single, data)
        if flag:
            route_nodes.append(node)
        else:
            r = Route(data)
            r.node_list = [data.DC, node, data.DC]
            r.update(data)
            s.append(r)
    if route_nodes:
        r = Route(data)
        r.node_list = [data.DC] + route_nodes + [data.DC]
        r.update(data)
        s.append(r)
    s.update(data)
    s.cal_cost(data)
    return feasible_or_repair_algorithm_10(s, data)


def _li_lim_random_search(current: Solution, data, rng: random.Random) -> Solution:
    sequence = [node for r in current.route_list for node in r.node_list if _is_customer(node, data)]
    if len(sequence) < 2:
        return current.clone()
    move_type = rng.randint(0, 2)
    if move_type == 0:  # Swap
        i, j = rng.sample(range(len(sequence)), 2)
        sequence[i], sequence[j] = sequence[j], sequence[i]
    elif move_type == 1:  # Insert
        i = rng.randrange(len(sequence))
        node = sequence.pop(i)
        j = rng.randint(0, len(sequence))
        sequence.insert(j, node)
    else:  # Reverse
        i, j = sorted(rng.sample(range(len(sequence)), 2))
        sequence[i:j+1] = reversed(sequence[i:j+1])
    return _build_solution_from_sequence(sequence, data)


def _inject_elite_routes(child: Solution, best: Solution, a: float, data, rng: random.Random) -> None:
    if best.len() == 0:
        return
    count = max(1, min(best.len(), int(round(1.0 + max(0.0, 2.0 - a)))))
    best_indices = list(range(best.len()))
    rng.shuffle(best_indices)

    selected_customers = set()
    for r_idx in best_indices[:count]:
        for node in best.route_list[r_idx].node_list:
            if _is_customer(node, data):
                selected_customers.add(node)

    for r in child.route_list:
        r.node_list = [n for n in r.node_list if n not in selected_customers or not _is_customer(n, data)]
        r.update(data)
    child.route_list = [r for r in child.route_list if len(r.node_list) > 2]

    inserted = set(node for r in child.route_list for node in r.node_list if _is_customer(node, data))

    for r_idx in best_indices[:count]:
        seed_route = best.route_list[r_idx]
        seed_custs = [n for n in seed_route.node_list if _is_customer(n, data)]
        clean = True
        for n in seed_custs:
            if n in inserted:
                clean = False
                break
        if clean and seed_custs:
            r = Route(data)
            r.node_list = list(seed_route.node_list)
            r.update(data)
            child.append(r)
            inserted.update(seed_custs)
        else:
            for node in seed_custs:
                if node not in inserted:
                    _insert_customer_best_position(child, node, data)
                    inserted.add(node)


def _inject_elite_segments(child: Solution, best: Solution, data, rng: random.Random) -> None:
    if best.len() == 0:
        return
    seed_route = best.route_list[rng.randint(0, best.len() - 1)]
    custs = [n for n in seed_route.node_list if _is_customer(n, data)]
    if len(custs) < 2:
        return
    seg_len = rng.randint(2, min(4, len(custs)))
    start_pos = rng.randint(0, len(custs) - seg_len)
    segment = custs[start_pos:start_pos + seg_len]
    seg_set = set(segment)

    for r in child.route_list:
        r.node_list = [n for n in r.node_list if n not in seg_set or not _is_customer(n, data)]
        r.update(data)
    child.route_list = [r for r in child.route_list if len(r.node_list) > 2]

    # Only attempt new route if child has fewer routes than best and the segment is a feasible route
    r_cand = [data.DC] + segment + [data.DC]
    if child.len() < best.len() and check_route_capacity(r_cand, data) and _chk_route_list(r_cand, data)[0]:
        r = Route(data)
        r.node_list = r_cand
        r.update(data)
        child.append(r)
    else:
        # Otherwise insert each customer into existing routes to preserve NV
        for node in segment:
            if not _insert_customer_best_position(child, node, data, allow_new_route=False):
                _insert_customer_best_position(child, node, data, allow_new_route=True)


def _guided_route_crossover(best: Solution, peer: Solution, current: Solution, data, rng: random.Random) -> Solution:
    if best.len() == 0:
        return current.clone()
    child = Solution(data)
    kept_customers = set()

    take = 1 if (best.len() <= 1 or rng.random() < 0.6) else 2
    best_indices = list(range(best.len()))
    rng.shuffle(best_indices)
    for r_idx in best_indices[:take]:
        r = best.route_list[r_idx]
        custs = [node for node in r.node_list if _is_customer(node, data)]
        if custs:
            child_r = Route(data)
            child_r.node_list = [data.DC] + custs + [data.DC]
            child_r.update(data)
            child.append(child_r)
            kept_customers.update(custs)

    remaining = []
    for p in (peer, current):
        for r in p.route_list:
            for node in r.node_list:
                if _is_customer(node, data) and node not in kept_customers and node not in remaining:
                    remaining.append(node)

    current_route = []
    for node in remaining:
        candidate = [data.DC] + current_route + [node, data.DC]
        is_cap_ok = check_route_capacity(candidate, data)
        is_tw_ok = _chk_route_list(candidate, data)[0] if is_cap_ok else False
        if current_route and (not is_cap_ok or not is_tw_ok):
            child_r = Route(data)
            child_r.node_list = [data.DC] + current_route + [data.DC]
            child_r.update(data)
            child.append(child_r)
            current_route = []
        current_route.append(node)
        kept_customers.add(node)
    if current_route:
        child_r = Route(data)
        child_r.node_list = [data.DC] + current_route + [data.DC]
        child_r.update(data)
        child.append(child_r)

    sho_mutation_prob = getattr(data, "sho_mutation_prob", 0.35)
    if rng.random() < sho_mutation_prob and child.len() > 0:
        _inject_elite_segments(child, best, data, rng)

    return feasible_or_repair_algorithm_10(child, data, rng)


def _woa_intensification(current: Solution, best: Solution, a: float, data, rng: random.Random) -> Solution:
    r1 = rng.random()
    a_vector = 2.0 * a * r1 - a

    if abs(a_vector) < 1.0:
        child = current.clone()
        _inject_elite_routes(child, best, a, data, rng)
        return feasible_or_repair_algorithm_10(child, data, rng)

    child = _li_lim_random_search(current, data, rng)
    return feasible_or_repair_algorithm_10(child, data, rng)


def _ruin_and_recreate(s: Solution, data, rng: random.Random) -> Solution:
    child = s.clone()
    removal_from_s_custs = []
    all_custs = [node for r in child.route_list for node in r.node_list if _is_customer(node, data)]
    if len(all_custs) < 4:
        return child
    num_to_remove = max(2, int(round(len(all_custs) * 0.30)))
    rem_set = set(rng.sample(all_custs, num_to_remove))

    for r in child.route_list:
        r.node_list = [n for n in r.node_list if n not in rem_set or not _is_customer(n, data)]
        r.update(data)
    child.route_list = [r for r in child.route_list if len(r.node_list) > 2]

    rem_list = list(rem_set)
    rng.shuffle(rem_list)
    for c in rem_list:
        _insert_customer_best_position(child, c, data)

    return feasible_or_repair_algorithm_10(child, data, rng)


# =============================================================================
# Local Search Multi-Operator Engine (2-Opt, 2-Opt*, Or-Opt, 2-Exchange, Perturb)
# =============================================================================

def two_opt(r1: int, r2: int, s: Solution, data, m: Move) -> None:
    m.delta_cost = float("inf")
    r = s.get(r1)
    n_l = r.node_list
    length = len(n_l)
    if length < 4:
        return
    for start in range(1, length - 2):
        if data.pruning and (
            not data.pm[n_l[start - 1]][n_l[start + 1]]
            or not data.pm[n_l[start + 1]][n_l[start]]
            or not data.pm[n_l[start]][n_l[start + 2]]
        ):
            continue
        TMP_MOVE.r_indice[0] = r1
        TMP_MOVE.r_indice[1] = -2
        TMP_MOVE.len_1 = 3
        TMP_MOVE.seqList_1[0] = Seq(r1, 0, start - 1)
        TMP_MOVE.seqList_1[1] = Seq(r1, start + 1, start)
        TMP_MOVE.seqList_1[2] = Seq(r1, start + 2, length - 1)
        TMP_MOVE.len_2 = 0
        if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
            m.copy_from(TMP_MOVE)


def two_opt_star(r1: int, r2: int, s: Solution, data, m: Move) -> None:
    m.delta_cost = float("inf")
    r_1 = s.get(r1)
    n_l_1 = r_1.node_list
    len_1 = len(n_l_1)

    r_2 = s.get(r2)
    n_l_2 = r_2.node_list
    len_2 = len(n_l_2)
    for pos_1 in range(1, len_1):
        for pos_2 in range(1, len_2):
            if (pos_1 == 1 and pos_2 == 1) or (pos_1 == len_1 - 1 and pos_2 == len_2 - 1):
                continue
            if data.pruning and (
                not data.pm[n_l_1[pos_1 - 1]][n_l_2[pos_2]]
                or not data.pm[n_l_2[pos_2 - 1]][n_l_1[pos_1]]
            ):
                continue
            TMP_MOVE.r_indice[0] = r1
            TMP_MOVE.r_indice[1] = r2
            TMP_MOVE.len_1 = 2
            TMP_MOVE.seqList_1[0] = Seq(r1, 0, pos_1 - 1)
            TMP_MOVE.seqList_1[1] = Seq(r2, pos_2, len_2 - 1)
            TMP_MOVE.len_2 = 2
            TMP_MOVE.seqList_2[0] = Seq(r2, 0, pos_2 - 1)
            TMP_MOVE.seqList_2[1] = Seq(r1, pos_1, len_1 - 1)
            if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
                m.copy_from(TMP_MOVE)


def or_opt_single(r1: int, r2: int, s: Solution, data, m: Move) -> None:
    m.delta_cost = float("inf")
    r = s.get(r1)
    n_l = r.node_list
    length = len(n_l)
    for start in range(1, length - 1):
        for seq_len in range(1, data.or_opt_len + 1):
            end = start + seq_len - 1
            if end >= length - 1:
                continue
            if data.pruning and (not data.pm[n_l[start - 1]][n_l[end + 1]]):
                continue
            for pos in range(1, start):
                if data.pruning and (
                    not data.pm[n_l[pos - 1]][n_l[start]]
                    or not data.pm[n_l[end]][n_l[pos]]
                ):
                    continue
                TMP_MOVE.r_indice[0] = r1
                TMP_MOVE.r_indice[1] = -2
                TMP_MOVE.len_1 = 4
                TMP_MOVE.seqList_1[0] = Seq(r1, 0, pos - 1)
                TMP_MOVE.seqList_1[1] = Seq(r1, start, end)
                TMP_MOVE.seqList_1[2] = Seq(r1, pos, start - 1)
                TMP_MOVE.seqList_1[3] = Seq(r1, end + 1, length - 1)
                TMP_MOVE.len_2 = 0
                if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
                    m.copy_from(TMP_MOVE)
            for pos in range(end + 2, length):
                if data.pruning and (
                    not data.pm[n_l[pos - 1]][n_l[start]] or not data.pm[n_l[end]][n_l[pos]]
                ):
                    continue
                TMP_MOVE.r_indice[0] = r1
                TMP_MOVE.r_indice[1] = -2
                TMP_MOVE.len_1 = 4
                TMP_MOVE.seqList_1[0] = Seq(r1, 0, start - 1)
                TMP_MOVE.seqList_1[1] = Seq(r1, end + 1, pos - 1)
                TMP_MOVE.seqList_1[2] = Seq(r1, start, end)
                TMP_MOVE.seqList_1[3] = Seq(r1, pos, length - 1)
                TMP_MOVE.len_2 = 0
                if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
                    m.copy_from(TMP_MOVE)


def or_opt_double(r_index_1: int, r_index_2: int, s: Solution, data, m: Move) -> None:
    m.delta_cost = float("inf")
    for i in range(2):
        r1 = r_index_1 if i == 0 else r_index_2
        r2 = r_index_2 if i == 0 else r_index_1
        if r1 == r2:
            continue
        r = s.get(r1)
        n_l = r.node_list
        length = len(n_l)
        for start in range(1, length - 1):
            for seq_len in range(1, data.or_opt_len + 1):
                end = start + seq_len - 1
                if end >= length - 1:
                    continue
                if data.pruning and (not data.pm[n_l[start - 1]][n_l[end + 1]]):
                    continue
                r_2 = s.get(r2)
                n_l_2 = r_2.node_list
                len_2 = len(n_l_2)
                for pos in range(1, len_2):
                    if data.pruning and (
                        not data.pm[n_l_2[pos - 1]][n_l[start]]
                        or not data.pm[n_l[end]][n_l_2[pos]]
                    ):
                        continue
                    TMP_MOVE.r_indice[0] = r1
                    TMP_MOVE.r_indice[1] = r2
                    TMP_MOVE.len_1 = 2
                    TMP_MOVE.seqList_1[0] = Seq(r1, 0, start - 1)
                    TMP_MOVE.seqList_1[1] = Seq(r1, end + 1, length - 1)
                    TMP_MOVE.len_2 = 3
                    TMP_MOVE.seqList_2[0] = Seq(r2, 0, pos - 1)
                    TMP_MOVE.seqList_2[1] = Seq(r1, start, end)
                    TMP_MOVE.seqList_2[2] = Seq(r2, pos, len_2 - 1)
                    if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
                        m.copy_from(TMP_MOVE)


def two_exchange(r1: int, r2: int, s: Solution, data, m: Move) -> None:
    m.delta_cost = float("inf")
    r_1 = s.get(r1)
    n_l_1 = r_1.node_list
    len_1 = len(n_l_1)

    r_2 = s.get(r2)
    n_l_2 = r_2.node_list
    len_2 = len(n_l_2)
    for start_1 in range(1, len_1 - 1):
        for seq_len_1 in range(1, data.exchange_len + 1):
            end_1 = start_1 + seq_len_1 - 1
            if end_1 >= len_1 - 1:
                continue
            for start_2 in range(1, len_2 - 1):
                for seq_len_2 in range(1, data.exchange_len + 1):
                    end_2 = start_2 + seq_len_2 - 1
                    if end_2 >= len_2 - 1:
                        continue
                    if data.pruning and (
                        not data.pm[n_l_1[start_1 - 1]][n_l_2[start_2]]
                        or not data.pm[n_l_2[end_2]][n_l_1[end_1 + 1]]
                        or not data.pm[n_l_2[start_2 - 1]][n_l_1[start_1]]
                        or not data.pm[n_l_1[end_1]][n_l_2[end_2 + 1]]
                    ):
                        continue
                    TMP_MOVE.r_indice[0] = r1
                    TMP_MOVE.r_indice[1] = r2
                    TMP_MOVE.len_1 = 3
                    TMP_MOVE.seqList_1[0] = Seq(r1, 0, start_1 - 1)
                    TMP_MOVE.seqList_1[1] = Seq(r2, start_2, end_2)
                    TMP_MOVE.seqList_1[2] = Seq(r1, end_1 + 1, len_1 - 1)
                    TMP_MOVE.len_2 = 3
                    TMP_MOVE.seqList_2[0] = Seq(r2, 0, start_2 - 1)
                    TMP_MOVE.seqList_2[1] = Seq(r1, start_1, end_1)
                    TMP_MOVE.seqList_2[2] = Seq(r2, end_2 + 1, len_2 - 1)
                    if eval_move(s, TMP_MOVE, data) and TMP_MOVE.delta_cost < m.delta_cost:
                        m.copy_from(TMP_MOVE)


def apply_move(s: Solution, m: Move, data) -> List[int]:
    r_indice = [m.r_indice[0]]
    if m.r_indice[1] != -2:
        r_indice.append(m.r_indice[1])

    r = s.get(r_indice[0])
    target_n_l = []

    for i in range(m.len_1):
        seq = m.seqList_1[i]
        source_n_l = s.get(seq.r_index).node_list
        if seq.start_point <= seq.end_point:
            target_n_l.extend(source_n_l[seq.start_point:seq.end_point + 1])
        else:
            target_n_l.extend(source_n_l[seq.start_point:seq.end_point - 1:-1])

    if len(r_indice) == 2:
        target_n_l_2 = []
        for i in range(m.len_2):
            seq = m.seqList_2[i]
            if seq.r_index == -1:
                target_n_l_2.append(data.DC)
                continue
            source_n_l = s.get(seq.r_index).node_list
            if seq.start_point <= seq.end_point:
                target_n_l_2.extend(source_n_l[seq.start_point:seq.end_point + 1])
            else:
                target_n_l_2.extend(source_n_l[seq.start_point:seq.end_point - 1:-1])
        if r_indice[1] == -1:
            r_new = Route(data)
            r_new.node_list = target_n_l_2
            r_new.update(data)
            s.append(r_new)
            r_indice[1] = s.len() - 1
        else:
            r_2 = s.get(r_indice[1])
            r_2.node_list = target_n_l_2
            r_2.update(data)

    r.node_list = target_n_l
    r.update(data)
    s.local_update(r_indice)
    return r_indice


def snippet(r1: int, r2: int, opt: str, s: Solution, data, target: Move) -> None:
    m = data.get_mem(opt, r1, r2)
    small_opt_map[opt](r1, r2, s, data, m)
    if m.delta_cost - target.delta_cost < -PRECISION:
        target.copy_from(m)


def find_local_optima(s: Solution, data) -> None:
    if getattr(data, "skip_finding_lo", False):
        return
    move_list = [Move() for _ in range(len(data.small_opts))]
    length = s.len()
    for i in range(len(move_list)):
        move_list[i].delta_cost = float("inf")
        opt = data.small_opts[i]
        if opt in ("2opt", "oropt_single"):
            for r in range(length):
                snippet(r, -1, opt, s, data, move_list[i])
        elif opt in ("2opt*", "2exchange", "oropt_double"):
            for r1 in range(length):
                for r2 in range(r1 + 1, length):
                    snippet(r1, r2, opt, s, data, move_list[i])

    while True:
        best_index = -1
        min_delta_cost = float("inf")
        for i in range(len(move_list)):
            if move_list[i].delta_cost - min_delta_cost < -PRECISION:
                best_index = i
                min_delta_cost = move_list[i].delta_cost
        if min_delta_cost < -PRECISION:
            tour_id_array = apply_move(s, move_list[best_index], data)
            length = s.len()
            for i in range(len(move_list)):
                move_list[i].delta_cost = float("inf")
                opt = data.small_opts[i]
                if opt in ("2opt", "oropt_single"):
                    for r in tour_id_array:
                        if r < length:
                            snippet(r, -1, opt, s, data, move_list[i])
                    for r in range(length):
                        if data.get_mem(opt, r, -1).delta_cost - move_list[i].delta_cost < -PRECISION:
                            move_list[i].copy_from(data.get_mem(opt, r, -1))
                elif opt in ("2opt*", "2exchange", "oropt_double"):
                    for r in tour_id_array:
                        if r < length:
                            for r1 in range(r):
                                snippet(r1, r, opt, s, data, move_list[i])
                            for r1 in range(r + 1, length):
                                snippet(r, r1, opt, s, data, move_list[i])
                    for r1 in range(length):
                        for r2 in range(r1 + 1, length):
                            if data.get_mem(opt, r1, r2).delta_cost - move_list[i].delta_cost < -PRECISION:
                                move_list[i].copy_from(data.get_mem(opt, r1, r2))
        else:
            break


def removal_from_s(s: Solution, flag: List[int]) -> None:
    for r in s.route_list:
        r.node_list = [node for node in r.node_list if flag[node] == 0 or not (node != 0)]
    s.route_list = [r for r in s.route_list if len(r.node_list) > 2]


def random_removal(s: Solution, data, rng: Optional[random.Random] = None) -> None:
    if rng is None:
        rng = getattr(data, "rng", random)
    customers = [i for i in range(1, data.customer_num + 1) if i != data.DC]
    rng.shuffle(customers)
    flag = [0] * (data.customer_num + 1)
    boundary = int(round(float(data.customer_num) * rand(data.destroy_ratio_l, data.destroy_ratio_u, rng)))
    for i in range(min(boundary + 1, len(customers))):
        flag[customers[i]] = 1
    removal_from_s(s, flag)
    s.update(data)


def related_removal(s: Solution, data, rng: Optional[random.Random] = None) -> None:
    if rng is None:
        rng = getattr(data, "rng", random)
    customers = [i for i in range(1, data.customer_num + 1) if i != data.DC]
    if not customers:
        return
    selected = customers[randint(0, len(customers) - 1, rng)]
    flag = [0] * (data.customer_num + 1)
    selected_cus = [selected]
    flag[selected] = 1
    total_remove = max(2, int(round(data.customer_num * rand(data.destroy_ratio_l, data.destroy_ratio_u, rng))))
    already_remove = 1

    while already_remove < total_remove:
        ref_cus = selected_cus[randint(0, len(selected_cus) - 1, rng)]
        argrank = data.rm_argrank[ref_cus]
        best_two = []
        for cand in argrank:
            if cand != data.DC and flag[cand] == 0:
                best_two.append(cand)
                if len(best_two) == 2:
                    break
        if len(best_two) == 0:
            break
        elif len(best_two) == 1:
            selected = best_two[0]
        else:
            d0 = data.rm[ref_cus][best_two[0]]
            d1 = data.rm[ref_cus][best_two[1]]
            prob = d1 / max(1e-9, d0 + d1)
            selected = best_two[0] if rand(0, 1, rng) < prob else best_two[1]
        flag[selected] = 1
        selected_cus.append(selected)
        already_remove += 1

    removal_from_s(s, flag)
    s.update(data)


def regret_insertion(s: Solution, data, backend=None, rng: Optional[random.Random] = None) -> None:
    visited = set(node for r in s.route_list for node in r.node_list if _is_customer(node, data))
    unrouted = [i for i in range(1, data.customer_num + 1) if i != data.DC and i not in visited]

    while unrouted:
        max_regret = -float("inf")
        best_cust_idx = -1
        best_cand_r = -1
        best_cand_p = -1

        for idx, c in enumerate(unrouted):
            scores = []
            for r_idx, r in enumerate(s.route_list):
                for pos in range(1, len(r.node_list)):
                    cand = r.node_list[:pos] + [c] + r.node_list[pos:]
                    if check_route_capacity(cand, data):
                        flag, cost = _chk_route_list(cand, data)
                        if flag:
                            delta = cost - r.cal_cost(data)
                            scores.append((delta, r_idx, pos))
            scores.sort(key=lambda x: x[0])
            if not scores:
                regret = 0.0
                r_choice = -1
                p_choice = -1
            elif len(scores) == 1:
                regret = 10000.0
                r_choice, p_choice = scores[0][1], scores[0][2]
            else:
                regret = scores[1][0] - scores[0][0]
                r_choice, p_choice = scores[0][1], scores[0][2]

            if regret > max_regret:
                max_regret = regret
                best_cust_idx = idx
                best_cand_r = r_choice
                best_cand_p = p_choice

        c = unrouted.pop(best_cust_idx)
        if best_cand_r != -1:
            s.route_list[best_cand_r].node_list.insert(best_cand_p, c)
            s.route_list[best_cand_r].update(data)
        else:
            r = Route(data)
            r.node_list = [data.DC, c, data.DC]
            r.update(data)
            s.append(r)
    s.update(data)
    s.cal_cost(data)


def greedy_insertion(s: Solution, data, backend=None, rng: Optional[random.Random] = None) -> None:
    visited = set(node for r in s.route_list for node in r.node_list if _is_customer(node, data))
    unrouted = [i for i in range(1, data.customer_num + 1) if i != data.DC and i not in visited]

    while unrouted:
        best_delta = float("inf")
        best_cust_idx = -1
        best_cand_r = -1
        best_cand_p = -1

        for idx, c in enumerate(unrouted):
            for r_idx, r in enumerate(s.route_list):
                for pos in range(1, len(r.node_list)):
                    cand = r.node_list[:pos] + [c] + r.node_list[pos:]
                    if check_route_capacity(cand, data):
                        flag, cost = _chk_route_list(cand, data)
                        if flag:
                            delta = cost - r.cal_cost(data)
                            if delta < best_delta:
                                best_delta = delta
                                best_cust_idx = idx
                                best_cand_r = r_idx
                                best_cand_p = pos

        if best_cust_idx != -1:
            c = unrouted.pop(best_cust_idx)
            s.route_list[best_cand_r].node_list.insert(best_cand_p, c)
            s.route_list[best_cand_r].update(data)
        else:
            c = unrouted.pop(0)
            r = Route(data)
            r.node_list = [data.DC, c, data.DC]
            r.update(data)
            s.append(r)
    s.update(data)
    s.cal_cost(data)


def perturb(s_vector: List[Solution], data) -> None:
    i = randint(0, len(data.destroy_opts) - 1, data.rng)
    j = randint(0, len(data.repair_opts) - 1, data.rng)
    destroy_opt_map[data.destroy_opts[i]](s_vector[0], data)
    repair_opt_map[data.repair_opts[j]](s_vector[0], data)


def do_local_search(s: Solution, data, executor=None) -> None:
    if len(data.small_opts) == 0:
        return
    if getattr(data, "escape_local_optima", 0) == -1:
        return

    find_local_optima(s, data)
    s.cal_cost(data)
    elo = getattr(data, "escape_local_optima", 3)
    if elo <= 0:
        return

    s_vector = [s.clone()]
    no_improve = 0
    while no_improve < elo:
        s_vector[0] = s.clone()
        perturb(s_vector, data)
        find_local_optima(s_vector[0], data)
        s_vector[0].cal_cost(data)

        if s_vector[0].cost - s.cost < -PRECISION:
            s.copy_from(s_vector[0])
            no_improve = 0
        else:
            no_improve += 1


def new_route_insertion(s: Solution, data, initial_node: Optional[int] = None) -> None:
    if initial_node is None:
        num_cus = data.customer_num
        record = [0] * (num_cus + 1)
        for r in s.route_list:
            for node in r.node_list:
                record[node] = 1
        unrouted = [i for i in range(1, num_cus + 1) if i != data.DC and record[i] == 0]
        if not unrouted:
            return
        initial_node = unrouted[randint(0, len(unrouted) - 1, data.rng)]

    sol = rcrs_grasp_initialization(data, data.rng, alpha=0.20)
    s.copy_from(sol)


# Maps for small, destroy, and repair operators
small_opt_map = {
    "2opt": two_opt,
    "2opt*": two_opt_star,
    "oropt_single": or_opt_single,
    "oropt_double": or_opt_double,
    "2exchange": two_exchange,
}

destroy_opt_map = {
    "random_removal": random_removal,
    "related_removal": related_removal,
}

repair_opt_map = {
    "regret_insertion": regret_insertion,
    "greedy_insertion": greedy_insertion,
}
