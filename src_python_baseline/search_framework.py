import concurrent.futures
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from . import state
from .config import (
    HYBRID_MODE_PH_SHOWOA,
    HYBRID_MODE_SHO,
    HYBRID_MODE_WOA,
    OUTPUT_PER_GENS,
    PRECISION,
    RCRS,
    RCRS_RANDOM,
    TD,
)
from .eval import _chk_route_list, evaluate_route_batch
from .move import Move
from .operator import do_local_search, new_route_insertion
from .solution import Route, Solution
from .util import argsort, mean, rand, randint
from .compute_backend import init_pool_worker


@dataclass
class AgentUpdateTask:
    index: int
    current: Solution
    peer: Solution
    best: Solution
    current_fit: float
    p_hybrid: float
    a: float
    iteration: int
    max_iter: int
    seed: int
    data: object


def update_best_solution(s, best_s, used, run, gen, data):
    if s.cost - best_s.cost < -PRECISION:
        best_s.copy_from(s)
        print("Best solution update: %.4f" % best_s.cost)
        state.find_best_time = used
        state.find_best_run = run
        state.find_best_gen = gen
        state.best_s_cost = best_s.cost
        if (not state.find_better) and (
            abs(best_s.cost - data.bks) < PRECISION or (best_s.cost - data.bks < -PRECISION)
        ):
            state.find_better = True
            state.find_bks_time = used
            state.find_bks_run = run
            state.find_bks_gen = gen


def quick_check_feasibility(s: Solution, data) -> bool:
    if s.len() > data.vehicle.max_num:
        return False
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
    if len(record) != data.customer_num:
        return False
    return True


def check_route_capacity(nl: List[int], data) -> bool:
    length = len(nl)
    if length <= 2:
        return True
    capacity = data.vehicle.capacity
    load = 0.0
    for node in nl[1:-1]:
        load += data.node[node].delivery
    if load > capacity:
        return False
    for i in range(1, length):
        node = nl[i]
        load = load - data.node[node].delivery + data.node[node].pickup
        if load < 0 or load > capacity:
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
            if time_val > data.node[node].end:
                violations.append(node)
        elif time_val > data.node[node].end and length > 2:
            # A depot deadline has no movable depot node.  Reinsert the final
            # customer instead; it is the direct source of the late return.
            violations.append(nl[-2])
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
                if flag and cost < best_cost:
                    best_nl = new_nl
                    best_cost = cost
                    improved = True
                    break
            if improved:
                break
    return best_nl


def _insert_customer_best_position(s: Solution, customer: int, data) -> None:
    best_r_idx = -1
    best_p_idx = -1
    best_delta = float('inf')
    
    for r_idx in range(len(s.route_list)):
        route = s.route_list[r_idx]
        for p_idx in range(1, len(route.node_list)):
            candidate_nl = route.node_list[:p_idx] + [customer] + route.node_list[p_idx:]
            flag, cost = _chk_route_list(candidate_nl, data)
            if flag:
                delta = cost - route.cal_cost(data)
                if delta < best_delta:
                    best_delta = delta
                    best_r_idx = r_idx
                    best_p_idx = p_idx
                    
    if best_r_idx != -1:
        s.route_list[best_r_idx].node_list.insert(best_p_idx, customer)
        s.route_list[best_r_idx].update(data)
    elif s.len() < data.vehicle.max_num:
        flag, _ = _chk_route_list([data.DC, customer, data.DC], data)
        if flag:
            s.append(_make_route([customer], data))


def _sa_initialization(s_0: Solution, data, rng: random.Random) -> Solution:
    import math
    s = s_0.clone()
    s.update(data)
    s.cal_cost(data)
    
    def score(solution: Solution) -> float:
        if getattr(data, "paper_flags", False):
            return sum(route.self.dist for route in solution.route_list)
        return solution.cost

    s_best = s.clone()
    best_cost = score(s_best)
    
    t0 = data.sa_t0
    alpha = data.sa_alpha
    tmin = data.sa_tmin
    itermax = data.sa_itermax
    
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
                    before = score(s)
                    candidate = s.clone()
                    candidate.route_list[r_idx].node_list = nl
                    candidate.update(data)
                    candidate.cal_cost(data)
                    new_cost = score(candidate)
                    delta = new_cost - before
                    
                    if delta < 0 or rng.random() < math.exp(-delta / (1e-6 + t * abs(before))):
                        route.node_list = nl
                        route.update(data)
                        s.cal_cost(data)
                        if score(s) < best_cost:
                            best_cost = score(s)
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
                    before = score(s)
                    candidate = s.clone()
                    candidate.route_list[r_idx1].node_list = nl1
                    candidate.route_list[r_idx2].node_list = nl2
                    candidate.update(data)
                    candidate.cal_cost(data)
                    new_cost = score(candidate)
                    delta = new_cost - before
                    
                    if delta < 0 or rng.random() < math.exp(-delta / (1e-6 + t * abs(before))):
                        route1.node_list = nl1
                        route2.node_list = nl2
                        s.update(data)
                        s.cal_cost(data)
                        if score(s) < best_cost:
                            best_cost = score(s)
                            s_best = s.clone()
                            
        t = alpha * t
        
    return s_best


def feasible_or_repair_algorithm_10(s: Solution, data, rng: Optional[random.Random] = None) -> Solution:
    if quick_check_feasibility(s, data):
        return s

    s_prime = s.clone()

    # --- Step 1: Repair duplicate and missing customers ---
    # Rebuild each route in a single pass.  The former occurrence-index list
    # became stale after the first deletion and could remove a depot or crash.
    seen = set()
    for route in s_prime.route_list:
        kept = []
        for node in route.node_list:
            if not _is_customer(node, data) or node not in seen:
                kept.append(node)
                if _is_customer(node, data):
                    seen.add(node)
        route.node_list = kept
        if not route.node_list or route.node_list[0] != data.DC:
            route.node_list.insert(0, data.DC)
        if route.node_list[-1] != data.DC:
            route.node_list.append(data.DC)

    visited_customers = set()
    for route in s_prime.route_list:
        for node in route.node_list:
            if _is_customer(node, data):
                visited_customers.add(node)
                
    all_customers = set(range(1, data.customer_num + 1))
    missing_customers = sorted(list(all_customers - visited_customers))
    
    for c in missing_customers:
        _insert_customer_best_position(s_prime, c, data)
        
    s_prime.update(data)
    s_prime.cal_cost(data)

    # --- Step 2: Repair capacity violations ---
    for r_idx in range(len(s_prime.route_list)):
        route = s_prime.route_list[r_idx]
        while not check_route_capacity(route.node_list, data):
            route_customers = [n for n in route.node_list if _is_customer(n, data)]
            if not route_customers:
                break
            c = max(route_customers, key=lambda node: abs(data.node[node].delivery - data.node[node].pickup))
            route.node_list.remove(c)
            route.update(data)
            
            inserted = False
            best_other_r_idx = -1
            best_other_p_idx = -1
            best_other_cost = float('inf')
            
            for other_r_idx in range(len(s_prime.route_list)):
                if other_r_idx == r_idx:
                    continue
                other_route = s_prime.route_list[other_r_idx]
                for p_idx in range(1, len(other_route.node_list)):
                    candidate_nl = other_route.node_list[:p_idx] + [c] + other_route.node_list[p_idx:]
                    if check_route_capacity(candidate_nl, data):
                        flag, cost = _chk_route_list(candidate_nl, data)
                        if flag and cost < best_other_cost:
                            best_other_cost = cost
                            best_other_r_idx = other_r_idx
                            best_other_p_idx = p_idx
                            inserted = True
                            
            if inserted:
                s_prime.route_list[best_other_r_idx].node_list.insert(best_other_p_idx, c)
                s_prime.route_list[best_other_r_idx].update(data)
            else:
                if s_prime.len() >= data.vehicle.max_num:
                    return s
                s_prime.append(_make_route([c], data))

    s_prime.update(data)
    s_prime.cal_cost(data)

    # --- Step 3: Repair time-window violations ---
    for r_idx in range(len(s_prime.route_list)):
        route = s_prime.route_list[r_idx]
        while True:
            arrival_times, violations = get_route_arrival_times_and_violations(route.node_list, data)
            if not violations:
                break
            c = violations[0]
            route.node_list.remove(c)
            route.update(data)
            
            inserted = False
            best_other_r_idx = -1
            best_other_p_idx = -1
            best_arrival_time = float('inf')
            
            for other_r_idx in range(len(s_prime.route_list)):
                other_route = s_prime.route_list[other_r_idx]
                for p_idx in range(1, len(other_route.node_list)):
                    candidate_nl = other_route.node_list[:p_idx] + [c] + other_route.node_list[p_idx:]
                    flag, _ = _chk_route_list(candidate_nl, data)
                    if flag:
                        cand_arrival_times, _ = get_route_arrival_times_and_violations(candidate_nl, data)
                        c_arrival = cand_arrival_times[p_idx]
                        if c_arrival < best_arrival_time:
                            best_arrival_time = c_arrival
                            best_other_r_idx = other_r_idx
                            best_other_p_idx = p_idx
                            inserted = True
                            
            if inserted:
                s_prime.route_list[best_other_r_idx].node_list.insert(best_other_p_idx, c)
                s_prime.route_list[best_other_r_idx].update(data)
            else:
                if s_prime.len() >= data.vehicle.max_num:
                    return s
                s_prime.append(_make_route([c], data))

    s_prime.update(data)
    s_prime.cal_cost(data)

    # --- Step 4: Local route repair ---
    for route in s_prime.route_list:
        route.node_list = _intra_route_2_opt(route.node_list, data)
        route.update(data)

    s_prime.update(data)
    s_prime.cal_cost(data)

    # --- Step 5: Final validation ---
    if quick_check_feasibility(s_prime, data):
        s.copy_from(s_prime)
        return s
    else:
        return s


def _init_single_solution_worker(task):
    import sys
    index, init_mode, data, seed, lambda_gamma = task
    print(f"[Worker {index}] Starting initialization", file=sys.stderr)
    sys.stderr.flush()
    rng = random.Random(seed)
    previous_rng = data.rng
    data.rng = rng
    data.init = init_mode
    data.ksize = data.k_init
    if init_mode in {RCRS, RCRS_RANDOM, "sa"}:
        data.n_insert = RCRS
    elif init_mode == TD:
        data.n_insert = TD

    if lambda_gamma is not None:
        data.lambda_gamma = lambda_gamma
    elif init_mode in {RCRS_RANDOM, "sa"}:
        data.lambda_gamma = (rand(0, 1, data.rng), rand(0, 1, data.rng))

    data.in_initialization = True
    s = Solution(data)
    try:
        new_route_insertion(s, data)
        s.cal_cost(data)
        if not quick_check_feasibility(s, data):
            feasible_or_repair(s, data, rng)
        if not quick_check_feasibility(s, data):
            raise ValueError("initialization did not produce a feasible solution")
        if init_mode == "sa":
            s = _sa_initialization(s, data, rng)
        s.cal_cost(data)
    finally:
        data.in_initialization = False
        data.rng = previous_rng
    print(f"[Worker {index}] Finished initialization", file=sys.stderr)
    sys.stderr.flush()
    return index, s


def initialization(pop, pop_fit, pop_argrank, data, executor=None, run_seed=None):
    length = len(pop)
    for i in range(length):
        pop[i].clear(data)
    print("Initialization, using %s method" % data.init)
    
    tasks = []
    for i in range(length):
        lambda_gamma = data.latin[i] if data.init in {RCRS, "sa"} else None
        seed = (data.seed if run_seed is None else run_seed) + 100000 + i
        tasks.append((i, data.init, data, seed, lambda_gamma))
        
    if executor is None:
        results = [_init_single_solution_worker(t) for t in tasks]
    else:
        results = list(executor.map(_init_single_solution_worker, tasks))
        
    for index, s in results:
        pop[index].copy_from(s)
        pop_fit[index] = s.cost
        print("Solution %d, cost %.4f" % (index, pop_fit[index]))

    argsort(pop_fit, pop_argrank, length)
    print("Initialization done.")



def output(pop, pop_fit, pop_argrank, data, output_complete=False):
    length = len(pop)
    best = pop[pop_argrank[0]]
    best_cost = pop_fit[pop_argrank[0]]
    worst_cost = pop_fit[pop_argrank[length - 1]]
    avg_cost = mean(pop_fit, 0, length)
    print(
        "Avg %.4f, Best %.4f, Worst %.4f, Best vehicles %d"
        % (avg_cost, best_cost, worst_cost, best.len())
    )
    if output_complete:
        best.output(data)


def _dynamic_parameters(iteration: int, max_iter: int) -> Tuple[float, float]:
    if max_iter <= 0:
        return 0.0, 0.15
    ratio = min(max(float(iteration) / float(max_iter), 0.0), 1.0)
    a = 2.0 - 2.0 * ratio
    p_hybrid = max(0.15, 0.5 * (1.0 - ratio))
    return a, p_hybrid

def _mode_probability(p_hybrid: float, data) -> float:
    if data.hybrid_mode == HYBRID_MODE_SHO:
        return 1.0
    if data.hybrid_mode == HYBRID_MODE_WOA:
        return 0.0
    if data.hybrid_mode == HYBRID_MODE_PH_SHOWOA:
        return p_hybrid
    raise ValueError("Unknown hybrid mode: %s" % data.hybrid_mode)


def _is_customer(node: int, data) -> bool:
    return node != data.DC and 0 <= node <= data.customer_num


def _route_customers(route: Route, data) -> List[int]:
    return [node for node in route.node_list if _is_customer(node, data)]


def _solution_customers(s: Solution, data) -> List[int]:
    customers: List[int] = []
    for route in s.route_list:
        customers.extend(_route_customers(route, data))
    return customers


def _customer_positions(s: Solution, data) -> List[Tuple[int, int]]:
    positions: List[Tuple[int, int]] = []
    for r_index in range(s.len()):
        route = s.get(r_index)
        for pos in range(1, len(route.node_list) - 1):
            if _is_customer(route.node_list[pos], data):
                positions.append((r_index, pos))
    return positions


def _make_route(customers: List[int], data) -> Route:
    route = Route(data)
    route.node_list = [data.DC] + list(customers) + [data.DC]
    route.update(data)
    return route


def _append_route_if_clean(
    target: Solution, route: Route, inserted: Set[int], data
) -> bool:
    customers = _route_customers(route, data)
    if len(customers) == 0 or any(node in inserted for node in customers):
        return False
    flag, _ = _chk_route_list([data.DC] + customers + [data.DC], data)
    if not flag:
        return False
    target.append(route)
    for node in customers:
        inserted.add(node)
    return True


def _append_customer_to_best_position(s: Solution, node: int, data) -> bool:
    best_route = -1
    best_pos = 1
    best_delta = float("inf")

    # Build a batch of all insertion candidate routes
    routes_to_check = [[data.DC, node, data.DC]]
    route_meta = [(-1, 1)]  # Maps routes_to_check index -> (r_index, pos)

    original_costs = {}
    for r_index in range(s.len()):
        route = s.get(r_index)
        original_costs[r_index] = route.cal_cost(data)
        for pos in range(1, len(route.node_list)):
            routes_to_check.append(route.node_list[:pos] + [node] + route.node_list[pos:])
            route_meta.append((r_index, pos))

    # Evaluate all candidates in a single batch
    results = evaluate_route_batch(routes_to_check, data)

    # First result is starting a new route
    new_route_flag, new_route_cost = results[0]
    if new_route_flag and s.len() < data.vehicle.max_num:
        best_delta = new_route_cost

    for (r_index, pos), (flag, cost) in zip(route_meta[1:], results[1:]):
        if not flag:
            continue
        delta = cost - original_costs[r_index]
        if delta - best_delta < -PRECISION:
            best_delta = delta
            best_route = r_index
            best_pos = pos

    if best_delta == float("inf"):
        return False

    if best_route == -1:
        route = Route(data)
        route.node_list.insert(1, node)
        route.update(data)
        s.append(route)
    else:
        route = s.get(best_route)
        route.node_list.insert(best_pos, node)
        route.update(data)
    s.cal_cost(data)
    return True


def _insert_customers(
    s: Solution, customers: List[int], data, rng: Optional[random.Random] = None
) -> None:
    pending = list(customers)
    if rng is not None:
        rng.shuffle(pending)
    for node in pending:
        if not _append_customer_to_best_position(s, node, data):
            raise ValueError("could not repair customer %d into a feasible fleet" % node)
    s.update(data)
    s.cal_cost(data)


def feasible_or_repair(
    s: Solution, data, rng: Optional[random.Random] = None, shuffle_pending: bool = False
) -> Solution:
    if quick_check_feasibility(s, data):
        return s
    if getattr(data, "paper_flags", False):
        return feasible_or_repair_algorithm_10(s, data, rng)
    clean = Solution(data)

    seen: Set[int] = set()
    pending: List[int] = []
    pending_set: Set[int] = set()

    # Fast duplicates-across-routes check
    seen_anywhere = set()
    has_duplicates_across_routes = False
    for route in s.route_list:
        route_seen = set()
        for node in route.node_list:
            if not _is_customer(node, data):
                continue
            if node in route_seen:
                continue
            if node in seen_anywhere:
                has_duplicates_across_routes = True
                break
            route_seen.add(node)
            seen_anywhere.add(node)
        if has_duplicates_across_routes:
            break

    if not has_duplicates_across_routes:
        candidate_routes = []
        routes_customers = []
        for route in s.route_list:
            customers = []
            route_seen = set()
            for node in route.node_list:
                if not _is_customer(node, data):
                    continue
                if node in route_seen:
                    # Duplicate within the same route goes to pending
                    if node not in pending_set:
                        pending.append(node)
                        pending_set.add(node)
                    continue
                customers.append(node)
                route_seen.add(node)
            
            if len(customers) == 0:
                continue
            candidate_routes.append([data.DC] + customers + [data.DC])
            routes_customers.append(customers)

        # Batch evaluate all candidate routes
        results = evaluate_route_batch(candidate_routes, data) if candidate_routes else []

        for customers, (flag, _) in zip(routes_customers, results):
            if flag:
                clean.append(_make_route(customers, data))
                for node in customers:
                    seen.add(node)
            else:
                for node in customers:
                    if node not in pending_set:
                        pending.append(node)
                        pending_set.add(node)
    else:
        # Fallback to sequential checks if duplicates exist across routes
        for route in s.route_list:
            customers = []
            route_seen = set()
            for node in route.node_list:
                if not _is_customer(node, data):
                    continue
                if node in seen or node in route_seen:
                    if node not in seen and node not in pending_set:
                        pending.append(node)
                        pending_set.add(node)
                    continue
                customers.append(node)
                route_seen.add(node)

            if len(customers) == 0:
                continue

            flag, _ = _chk_route_list([data.DC] + customers + [data.DC], data)
            if flag:
                clean.append(_make_route(customers, data))
                for node in customers:
                    seen.add(node)
            else:
                for node in customers:
                    if node not in seen and node not in pending_set:
                        pending.append(node)
                        pending_set.add(node)

    pending = [node for node in pending if node not in seen]
    pending_set = set(pending)
    for node in range(data.customer_num + 1):
        if node == data.DC:
            continue
        if node not in seen and node not in pending_set:
            pending.append(node)
            pending_set.add(node)

    if shuffle_pending and rng is not None:
        rng.shuffle(pending)
    try:
        _insert_customers(clean, pending, data, rng if shuffle_pending else None)
    except ValueError:
        return s
    clean.cal_cost(data)
    if quick_check_feasibility(clean, data):
        s.copy_from(clean)
    return s


def _remove_customers(s: Solution, customers: Set[int], data) -> None:
    for route in s.route_list:
        route.node_list = [node for node in route.node_list if node not in customers]
        if len(route.node_list) == 0 or route.node_list[0] != data.DC:
            route.node_list.insert(0, data.DC)
        if route.node_list[-1] != data.DC:
            route.node_list.append(data.DC)
        route.update(data)
    s.update(data)
    s.cal_cost(data)


def _guided_route_crossover(
    best: Solution, peer: Solution, current: Solution, data, rng: random.Random
) -> Solution:
    if best.len() == 0:
        return current.clone()

    child = Solution(data)
    kept_customers: Set[int] = set()
    route_indices = list(range(best.len()))
    rng.shuffle(route_indices)
    take = 1 if best.len() == 1 or rng.random() < 0.6 else 2

    # Seed the offspring with one or two high-quality routes from the best solution.
    for r_index in route_indices[:take]:
        seed_route = best.get(r_index)
        customers = _route_customers(seed_route, data)
        if not customers:
            continue
        child.append(_make_route(customers, data))
        kept_customers.update(customers)

    # Rebuild the remaining customers in the order they appear in the partner/current parents.
    remaining: List[int] = []
    for parent in (peer, current):
        for route in parent.route_list:
            for node in route.node_list:
                if not _is_customer(node, data):
                    continue
                if node in kept_customers or node in remaining:
                    continue
                remaining.append(node)

    if not remaining:
        feasible_or_repair(child, data, rng)
        return child

    # Pack the remaining customers into depot-closed routes while respecting capacity.
    current_route: List[int] = [data.DC]
    for node in remaining:
        singleton = [data.DC, node, data.DC]
        if not check_route_capacity(singleton, data):
            _append_customer_to_best_position(child, node, data)
            kept_customers.add(node)
            continue

        candidate = current_route + [node, data.DC]
        if len(current_route) > 1 and not check_route_capacity(candidate, data):
            child.append(_make_route(current_route[1:], data))
            current_route = [data.DC]

        current_route.append(node)
        kept_customers.add(node)

    if len(current_route) > 1:
        child.append(_make_route(current_route[1:], data))

    feasible_or_repair(child, data, rng)
    return child

def _swap_two_customers(s: Solution, data, rng: random.Random) -> bool:
    positions = _customer_positions(s, data)
    if len(positions) < 2:
        return False
    first, second = rng.sample(positions, 2)
    r1, p1 = first
    r2, p2 = second
    route_1 = s.get(r1)
    route_2 = s.get(r2)
    route_1.node_list[p1], route_2.node_list[p2] = route_2.node_list[p2], route_1.node_list[p1]
    route_1.update(data)
    if r1 != r2:
        route_2.update(data)
    s.update(data)
    s.cal_cost(data)
    return True


def _relocate_customer(s: Solution, data, rng: random.Random) -> bool:
    positions = _customer_positions(s, data)
    if len(positions) == 0:
        return False
    if getattr(data, "paper_flags", False) and s.len() < 2:
        return False
    r_index, pos = rng.choice(positions)
    source_route = s.get(r_index)
    node = source_route.node_list.pop(pos)
    source_route.update(data)
    s.update(data)

    if s.len() == 0 or (not getattr(data, "paper_flags", False) and rng.random() < 0.15):
        route = Route(data)
        route.node_list.insert(1, node)
        route.update(data)
        s.append(route)
    else:
        candidates = list(range(s.len()))
        if getattr(data, "paper_flags", False) and r_index in candidates:
            candidates.remove(r_index)
        target_index = rng.choice(candidates)
        target_route = s.get(target_index)
        target_pos = rng.randrange(1, len(target_route.node_list))
        target_route.node_list.insert(target_pos, node)
        target_route.update(data)
    s.update(data)
    s.cal_cost(data)
    return True


def _light_mutation(s: Solution, data, rng: random.Random) -> None:
    backup = s.clone()
    if rng.random() < 0.5:
        _swap_two_customers(s, data, rng)
    else:
        _relocate_customer(s, data, rng)
    feasible_or_repair(s, data, rng)
    if not quick_check_feasibility(s, data):
        s.copy_from(backup)


def _inject_elite_routes(
    child: Solution, best: Solution, data, rng: random.Random, c_value: float
) -> None:
    if best.len() == 0:
        return
    count = max(1, min(best.len(), int(round(1.0 + max(0.0, 2.0 - c_value)))))
    route_indices = list(range(best.len()))
    rng.shuffle(route_indices)
    selected_routes = route_indices[:count]
    selected_customers: Set[int] = set()
    for r_index in selected_routes:
        selected_customers.update(_route_customers(best.get(r_index), data))

    _remove_customers(child, selected_customers, data)
    inserted: Set[int] = set(_solution_customers(child, data))
    for r_index in selected_routes:
        if not _append_route_if_clean(child, best.get(r_index), inserted, data):
            for node in _route_customers(best.get(r_index), data):
                if node not in inserted:
                    _append_customer_to_best_position(child, node, data)
                    inserted.add(node)


def _inject_elite_segments(
    child: Solution, best: Solution, data, rng: random.Random
) -> None:
    candidate_routes = [
        best.get(i) for i in range(best.len()) if len(_route_customers(best.get(i), data)) > 0
    ]
    if len(candidate_routes) == 0:
        return

    segment_count = min(len(candidate_routes), rng.randint(1, 3))
    segments: List[List[int]] = []
    selected_customers: Set[int] = set()
    for route in rng.sample(candidate_routes, segment_count):
        customers = _route_customers(route, data)
        if len(customers) == 0:
            continue
        l_value = rng.uniform(-1.0, 1.0)
        spiral_scale = abs(math.exp(l_value) * math.cos(2.0 * math.pi * l_value))
        seg_len = max(1, min(len(customers), int(round(1.0 + spiral_scale))))
        start = rng.randint(0, len(customers) - seg_len)
        segment = [node for node in customers[start : start + seg_len] if node not in selected_customers]
        if len(segment) > 0:
            segments.append(segment)
            selected_customers.update(segment)

    _remove_customers(child, selected_customers, data)
    try_segment_route = [rng.random() < 0.70 for _ in segments]
    
    routes_to_check = []
    routes_indices = []
    for idx, segment in enumerate(segments):
        if try_segment_route[idx]:
            routes_to_check.append([data.DC] + segment + [data.DC])
            routes_indices.append(idx)

    # Batch evaluate segment routes
    results = evaluate_route_batch(routes_to_check, data) if routes_to_check else []
    results_map = {routes_indices[i]: results[i][0] for i in range(len(routes_indices))}

    for idx, segment in enumerate(segments):
        if try_segment_route[idx] and results_map.get(idx, False):
            child.append(_make_route(segment, data))
        else:
            for node in segment:
                _append_customer_to_best_position(child, node, data)


def _build_solution_from_sequence(sequence: List[int], data) -> Solution:
    s = Solution(data)
    route_nodes: List[int] = []
    for node in sequence:
        trial = [data.DC] + route_nodes + [node] + [data.DC]
        flag, _ = _chk_route_list(trial, data)
        if flag:
            route_nodes.append(node)
            continue
        if len(route_nodes) > 0:
            s.append(_make_route(route_nodes, data))
        route_nodes = []
        single_flag, _ = _chk_route_list([data.DC, node, data.DC], data)
        if single_flag:
            route_nodes.append(node)
        else:
            _append_customer_to_best_position(s, node, data)

    if len(route_nodes) > 0:
        s.append(_make_route(route_nodes, data))
    feasible_or_repair(s, data)
    return s


def _li_lim_random_search(current: Solution, data, rng: random.Random) -> Solution:
    sequence = _solution_customers(current, data)
    if len(sequence) < 2:
        return current.clone()

    move_type = rng.randint(0, 2)
    if move_type == 0:
        i, j = rng.sample(range(len(sequence)), 2)
        sequence[i], sequence[j] = sequence[j], sequence[i]
    elif move_type == 1:
        i, j = rng.sample(range(len(sequence)), 2)
        node = sequence.pop(i)
        sequence.insert(j, node)
    else:
        i, j = sorted(rng.sample(range(len(sequence)), 2))
        sequence[i : j + 1] = reversed(sequence[i : j + 1])

    return _build_solution_from_sequence(sequence, data)


def _relink_route(route: Route, guide: Route, amount: float, reverse: bool, data) -> None:
    """Discrete WOA movement: move the common customer order toward the guide."""
    positions = [i for i, node in enumerate(route.node_list) if _is_customer(node, data)]
    guide_order = [node for node in guide.node_list if _is_customer(node, data)]
    common = [route.node_list[i] for i in positions if route.node_list[i] in set(guide_order)]
    if len(common) < 2:
        return
    desired = [node for node in guide_order if node in set(common)]
    if reverse:
        desired.reverse()
    moves = max(1, min(len(common) - 1, int(math.ceil(len(common) * amount))))
    for index in range(moves):
        wanted = desired[index]
        current_index = next(i for i, pos in enumerate(positions) if route.node_list[pos] == wanted)
        route.node_list[positions[index]], route.node_list[positions[current_index]] = (
            route.node_list[positions[current_index]], route.node_list[positions[index]],
        )


def _random_intra_route_operation(s: Solution, data, rng: random.Random) -> bool:
    choices = [route for route in s.route_list if len(_route_customers(route, data)) >= 2]
    if not choices:
        return False
    route = rng.choice(choices)
    customer_positions = list(range(1, len(route.node_list) - 1))
    first, second = sorted(rng.sample(customer_positions, 2))
    operation = rng.randrange(3)
    if operation == 0:
        route.node_list[first], route.node_list[second] = route.node_list[second], route.node_list[first]
    elif operation == 1:
        node = route.node_list.pop(first)
        route.node_list.insert(second, node)
    else:
        route.node_list[first:second + 1] = reversed(route.node_list[first:second + 1])
    route.update(data)
    s.update(data)
    return True


def _woa_intensification(
    current: Solution, best: Solution, a: float, data, rng: random.Random
) -> Solution:
    if not getattr(data, "paper_flags", False):
        r1 = rng.random()
        c_value = 2.0 * rng.random()
        child = current.clone()
        if abs(2.0 * a * r1 - a) < 1.0:
            if rng.random() < 0.5:
                _inject_elite_routes(child, best, data, rng, c_value)
            else:
                _inject_elite_segments(child, best, data, rng)
        else:
            child = _li_lim_random_search(current, data, rng)
        feasible_or_repair(child, data, rng)
        return child

    child = current.clone()
    limit = min(child.len(), best.len())
    for route_index in range(limit):
        a_value = rng.uniform(-a, a)
        c_value = rng.uniform(0.0, 2.0)
        probability = rng.random()
        backup = child.clone()
        if probability < 0.5 and abs(a_value) < 1.0:
            _relink_route(child.get(route_index), best.get(route_index),
                          max(0.0, 1.0 - abs(a_value)) * c_value / 2.0,
                          a_value < 0.0, data)
        elif probability < 0.5:
            if rng.random() < 0.5:
                _random_intra_route_operation(child, data, rng)
            else:
                _light_mutation(child, data, rng)
        else:
            l_value = c_value - 1.0
            spiral = min(1.0, abs(math.exp(l_value) * math.cos(2.0 * math.pi * l_value)))
            _relink_route(child.get(route_index), best.get(route_index), spiral, l_value < 0.0, data)
        child.update(data)
        child.cal_cost(data)
        if not quick_check_feasibility(child, data):
            child = backup
    return child


def _sa_accept(
    new_solution: Solution,
    current: Solution,
    current_fit: float,
    iteration: int,
    max_iter: int,
    rng: random.Random,
) -> bool:
    delta = new_solution.cost - current_fit
    if delta <= PRECISION:
        return True
    temperature = 1.0 - (float(iteration) / float(max_iter)) if max_iter > 0 else 0.0
    denominator = 1e-6 + temperature * abs(current_fit)
    probability = math.exp(-delta / denominator)
    return rng.random() < probability


def _update_agent_worker(task: AgentUpdateTask) -> Tuple[int, Solution, float, bool]:
    data = task.data
    rng = random.Random(task.seed)
    previous_rng = data.rng
    data.rng = rng
    try:
        if rng.random() < task.p_hybrid:
            new_solution = _guided_route_crossover(task.best, task.peer, task.current, data, rng)
            if rng.random() < data.sho_mutation_prob:
                _light_mutation(new_solution, data, rng)
            feasible_or_repair(new_solution, data, rng)
        else:
            new_solution = _woa_intensification(task.current, task.best, task.a, data, rng)

        if not quick_check_feasibility(new_solution, data):
            return task.index, task.current, task.current_fit, False
        new_solution.cal_cost(data)
        accepted = _sa_accept(
            new_solution, task.current, task.current_fit, task.iteration, task.max_iter, rng
        )
        if accepted:
            return task.index, new_solution, new_solution.cost, True
        return task.index, task.current, task.current_fit, False
    finally:
        data.rng = previous_rng


def _tournament_peer_index(pop_fit: List[float], current_index: int, data, k: int = 3) -> int:
    candidates = [index for index in range(len(pop_fit)) if index != current_index]
    if len(candidates) == 0:
        return current_index
    data.rng.shuffle(candidates)
    competitors = candidates[: min(k, len(candidates))]
    return min(competitors, key=lambda index: pop_fit[index])


def _worker_count(data) -> Optional[int]:
    if data.parallel_workers <= 0:
        return None
    return max(1, data.parallel_workers)


def _run_agent_updates(
    tasks: List[AgentUpdateTask],
    executor: Optional[concurrent.futures.ProcessPoolExecutor],
) -> List[Tuple[int, Solution, float, bool]]:
    if executor is None:
        return [_update_agent_worker(task) for task in tasks]
    return list(executor.map(_update_agent_worker, tasks))


def _install_move_memory(data, small_opts: List[str]) -> None:
    data.small_opts = list(small_opts)
    data.mem = {}
    max_num = data.vehicle.max_num
    if "2opt" in small_opts:
        data.mem["2opt"] = [Move() for _ in range(max_num)]
    if "oropt_single" in small_opts:
        data.mem["oropt_single"] = [Move() for _ in range(max_num)]
    if "oropt_double" in small_opts:
        data.mem["oropt_double"] = [Move() for _ in range(max_num * max_num)]
    if "2exchange" in small_opts:
        data.mem["2exchange"] = [Move() for _ in range(max_num * max_num)]


def _combined_local_search(s: Solution, data) -> None:
    """Algorithm 8's ordered, first-improvement neighbourhood search."""
    while True:
        improved = False
        baseline = s.cost

        # 2-opt: reverse every non-trivial intra-route customer segment.
        for r_index, route in enumerate(s.route_list):
            for start in range(1, len(route.node_list) - 2):
                for end in range(start + 1, len(route.node_list) - 1):
                    candidate = s.clone()
                    nodes = candidate.get(r_index).node_list
                    nodes[start:end + 1] = reversed(nodes[start:end + 1])
                    candidate.update(data)
                    candidate.cal_cost(data)
                    if quick_check_feasibility(candidate, data) and candidate.cost < baseline - PRECISION:
                        s.copy_from(candidate)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Relocate one customer, including moves between two existing routes.
        for source_index, source in enumerate(s.route_list):
            for source_pos in range(1, len(source.node_list) - 1):
                for target_index, target in enumerate(s.route_list):
                    for target_pos in range(1, len(target.node_list)):
                        if source_index == target_index and target_pos in (source_pos, source_pos + 1):
                            continue
                        candidate = s.clone()
                        node = candidate.get(source_index).node_list.pop(source_pos)
                        adjusted = target_pos
                        if source_index == target_index and target_pos > source_pos:
                            adjusted -= 1
                        candidate.get(target_index).node_list.insert(adjusted, node)
                        candidate.update(data)
                        candidate.cal_cost(data)
                        if quick_check_feasibility(candidate, data) and candidate.cost < baseline - PRECISION:
                            s.copy_from(candidate)
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

        # Swap exactly one customer from distinct routes.
        for first_index in range(s.len()):
            for second_index in range(first_index + 1, s.len()):
                for first_pos in range(1, len(s.get(first_index).node_list) - 1):
                    for second_pos in range(1, len(s.get(second_index).node_list) - 1):
                        candidate = s.clone()
                        first = candidate.get(first_index).node_list
                        second = candidate.get(second_index).node_list
                        first[first_pos], second[second_pos] = second[second_pos], first[first_pos]
                        candidate.update(data)
                        candidate.cal_cost(data)
                        if quick_check_feasibility(candidate, data) and candidate.cost < baseline - PRECISION:
                            s.copy_from(candidate)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            return


def _deep_local_search_best(s: Solution, data, executor=None) -> None:
    saved_small_opts = list(data.small_opts)
    saved_mem: Dict[str, List[Move]] = data.mem
    saved_escape = data.escape_local_optima
    saved_skip = data.skip_finding_lo

    try:
        data.skip_finding_lo = False
        data.escape_local_optima = 0
        if getattr(data, "paper_flags", False):
            _combined_local_search(s, data)
        else:
            _install_move_memory(data, ["2opt"])
            do_local_search(s, data, executor)
            _install_move_memory(data, ["oropt_single", "oropt_double"])
            do_local_search(s, data, executor)
            _install_move_memory(data, ["2exchange"])
            do_local_search(s, data, executor)
        s.update(data)
        s.cal_cost(data)
    finally:
        data.small_opts = saved_small_opts
        data.mem = saved_mem
        data.escape_local_optima = saved_escape
        data.skip_finding_lo = saved_skip

def _ruin_and_recreate(s: Solution, data, rng: random.Random) -> Solution:
    candidate = s.clone()
    customers = _solution_customers(candidate, data)
    if len(customers) == 0:
        customers = [node for node in range(data.customer_num + 1) if node != data.DC]
    rng.shuffle(customers)
    remove_count = max(1, int(round(len(customers) * rng.uniform(0.20, 0.40))))
    removed = set(customers[:remove_count])
    _remove_customers(candidate, removed, data)
    pending = list(removed)
    rng.shuffle(pending)
    try:
        _insert_customers(candidate, pending, data, rng)
    except ValueError:
        return s.clone()
    feasible_or_repair(candidate, data, rng, shuffle_pending=True)
    return candidate


def _diversify(pop, pop_fit, pop_argrank, best_s, data) -> None:
    if data.diversify_ratio == 0.0:
        return
    argsort(pop_fit, pop_argrank, len(pop))
    elite_index = pop_argrank[0]
    pop[elite_index].copy_from(best_s)
    pop_fit[elite_index] = best_s.cost

    non_elite = [index for index in range(len(pop)) if index != elite_index]
    if len(non_elite) == 0:
        return
    data.rng.shuffle(non_elite)
    diversify_count = int(round(len(non_elite) * data.diversify_ratio))
    if diversify_count == 0:
        return
    for index in non_elite[:diversify_count]:
        candidate = _ruin_and_recreate(pop[index], data, data.rng)
        pop[index].copy_from(candidate)
        pop_fit[index] = candidate.cost
    argsort(pop_fit, pop_argrank, len(pop))


def _inject_global_best(pop, pop_fit, pop_argrank, best_s) -> None:
    argsort(pop_fit, pop_argrank, len(pop))
    worst_index = pop_argrank[-1]
    pop[worst_index].copy_from(best_s)
    pop_fit[worst_index] = best_s.cost
    argsort(pop_fit, pop_argrank, len(pop))


def search_framework(data, best_s):
    pop = [Solution(data) for _ in range(data.p_size)]
    pop_fit = [0.0 for _ in range(data.p_size)]
    pop_argrank = [0 for _ in range(data.p_size)]

    stime = time.perf_counter()
    used = 0
    time_exhausted = False
    run = 1
    completed_runs = 0

    executor = None
    if data.parallel_workers != 1 and data.p_size > 1 and getattr(data.backend, "multi_process_safe", True):
        initargs = ()
        initializer = None
        if hasattr(data.backend, "id_queue"):
            initializer = init_pool_worker
            initargs = (data.backend.id_queue, data.backend._request_queue, data.backend._response_queues)
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=_worker_count(data),
            initializer=initializer,
            initargs=initargs
        )

    try:
        while run <= data.runs:
            print("---------------------------------Run %d---------------------------" % run)
            run_seed = data.seed + run - 1
            data.rng.seed(run_seed)
            run_best = Solution(data)
            run_best.cost = float("inf")
            initialization(pop, pop_fit, pop_argrank, data, executor, run_seed)
            used = int(time.perf_counter() - stime)
            print("already consumed %d sec" % used)
    
            update_best_solution(pop[pop_argrank[0]], run_best, used, run, 0, data)
            output(pop, pop_fit, pop_argrank, data)
    
            no_improve = 0

            for gen in range(1, data.max_iter + 1):
                best_before_generation = run_best.cost
                iteration_index = gen - 1
                a, p_hybrid = _dynamic_parameters(iteration_index, data.max_iter)
                p_mode = _mode_probability(p_hybrid, data)
                best_snapshot = run_best.clone()
                tasks: List[AgentUpdateTask] = []

                for index in range(data.p_size):
                    peer_index = _tournament_peer_index(pop_fit, index, data, k=3)
                    tasks.append(
                        AgentUpdateTask(
                            index=index,
                            current=pop[index].clone(),
                            peer=pop[peer_index].clone(),
                            best=best_snapshot.clone(),
                            current_fit=pop_fit[index],
                            p_hybrid=p_mode,
                            a=a,
                            iteration=iteration_index,
                            max_iter=data.max_iter,
                            seed=data.rng.randint(0, 2**31 - 1),
                            data=data,
                        )
                    )

                results = _run_agent_updates(tasks, executor)
                accepted_count = 0
                for index, solution, cost, accepted in results:
                    pop[index].copy_from(solution)
                    pop_fit[index] = cost
                    if accepted:
                        accepted_count += 1

                argsort(pop_fit, pop_argrank, data.p_size)
                used = int(time.perf_counter() - stime)
                update_best_solution(pop[pop_argrank[0]], run_best, used, run, gen, data)

                local_search_due = (
                    iteration_index % data.local_search_interval == 0
                    if data.paper_flags else gen % data.local_search_interval == 0
                )
                if local_search_due:
                    print("Periodic deep local search on run best.")
                    elite = run_best.clone()
                    _deep_local_search_best(elite, data, executor)
                    update_best_solution(elite, run_best, used, run, gen, data)
                    if not data.paper_flags:
                        _inject_global_best(pop, pop_fit, pop_argrank, run_best)

                if run_best.cost - best_before_generation < -PRECISION:
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= data.stagnation_interval:
                    print("Stagnation detected. Diversifying 40%% of non-elite population.")
                    _diversify(pop, pop_fit, pop_argrank, run_best, data)
                    no_improve = 0

                used = int(time.perf_counter() - stime)
                if gen % OUTPUT_PER_GENS == 0:
                    print("Gen: %d. a %.4f, p_hybrid %.4f, accepted %d. " % (
                        gen,
                        a,
                        p_mode,
                        accepted_count,
                    ), end="")
                    output(pop, pop_fit, pop_argrank, data)
                    print(
                        "Gen %d done, no improvement for %d gens, already consumed %d sec"
                        % (gen, no_improve, used)
                    )

                if data.tmax != -1 and used > int(data.tmax):
                    time_exhausted = True
                    break

            print("Run %d finishes" % run)
            output(pop, pop_fit, pop_argrank, data)
            update_best_solution(run_best, best_s, used, run, gen if data.max_iter else 0, data)
            completed_runs += 1
            if time_exhausted:
                break
            run += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print("------------Summary-----------")
    print("Total %d runs, total consumed %d sec" % (completed_runs, int(used)))
    if not quick_check_feasibility(best_s, data) or not best_s.check(data):
        raise RuntimeError("solver finished without a feasible best solution")
    best_s.output(data)
    print(
        "In run %d, gen %d, find this solution, at time %d."
        % (state.find_best_run, state.find_best_gen, int(state.find_best_time))
    )
    print("Time to surpass BKS: %d." % int(state.find_bks_time))
    if hasattr(data.backend, "shutdown"):
        data.backend.shutdown()
