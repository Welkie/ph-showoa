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
from .eval import _chk_route_list
from .move import Move
from .operator import do_local_search, new_route_insertion
from .solution import Route, Solution
from .util import argsort, mean, rand, randint


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


def initialization(pop, pop_fit, pop_argrank, data):
    length = len(pop)
    for i in range(length):
        pop[i].clear(data)
    print("Initialization, using %s method" % data.init)
    if data.init == RCRS:
        data.n_insert = RCRS
        data.ksize = data.k_init
        for i in range(length):
            data.lambda_gamma = data.latin[i]
            new_route_insertion(pop[i], data)
    elif data.init == RCRS_RANDOM:
        data.n_insert = RCRS
        data.ksize = data.k_init
        for i in range(length):
            data.lambda_gamma = (rand(0, 1, data.rng), rand(0, 1, data.rng))
            print("lambda, gamma: %f, %f" % data.lambda_gamma)
            new_route_insertion(pop[i], data)
    elif data.init == TD:
        data.ksize = data.k_init
        data.n_insert = TD
        for i in range(length):
            new_route_insertion(pop[i], data)

    for i in range(length):
        pop[i].cal_cost(data)
        pop_fit[i] = pop[i].cost
        print("Solution %d, cost %.4f" % (i, pop_fit[i]))
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
    if max_iter == 1:
        t = 0.0
    else:
        t = (float(iteration - 1) / float(max_iter - 1)) * float(max_iter)
    a = 2.0 - 2.0 * (t / float(max_iter))
    p_hybrid = max(0.15, 0.5 * (1.0 - t / float(max_iter)))
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

    flag, best_new_route_cost = _chk_route_list([data.DC, node, data.DC], data)
    if flag:
        best_delta = best_new_route_cost

    for r_index in range(s.len()):
        route = s.get(r_index)
        original_cost = route.cal_cost(data)
        for pos in range(1, len(route.node_list)):
            flag, new_cost = _chk_route_list(
                route.node_list[:pos] + [node] + route.node_list[pos:], data
            )
            if not flag:
                continue
            delta = new_cost - original_cost
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
            print("Error: could not repair customer %d into any feasible route" % node)
            raise SystemExit(-1)
    s.update(data)
    s.cal_cost(data)


def feasible_or_repair(
    s: Solution, data, rng: Optional[random.Random] = None, shuffle_pending: bool = False
) -> Solution:
    clean = Solution(data)
    seen: Set[int] = set()
    pending: List[int] = []
    pending_set: Set[int] = set()

    for route in s.route_list:
        customers: List[int] = []
        route_seen: Set[int] = set()
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
    _insert_customers(clean, pending, data, rng if shuffle_pending else None)
    clean.cal_cost(data)
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
    child = Solution(data)
    inserted: Set[int] = set()
    sources: List[Tuple[Solution, float]] = [(best, 0.85), (peer, 0.60), (current, 0.40)]

    for source, keep_prob in sources:
        route_indices = list(range(source.len()))
        rng.shuffle(route_indices)
        for r_index in route_indices:
            if rng.random() <= keep_prob:
                _append_route_if_clean(child, source.get(r_index), inserted, data)

    if child.len() == 0:
        fallback = best if best.len() > 0 else current
        if fallback.len() > 0:
            _append_route_if_clean(child, fallback.get(rng.randrange(fallback.len())), inserted, data)

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
    r_index, pos = rng.choice(positions)
    source_route = s.get(r_index)
    node = source_route.node_list.pop(pos)
    source_route.update(data)
    s.update(data)

    if s.len() == 0 or rng.random() < 0.15:
        route = Route(data)
        route.node_list.insert(1, node)
        route.update(data)
        s.append(route)
    else:
        target_index = rng.randrange(s.len())
        target_route = s.get(target_index)
        target_pos = rng.randrange(1, len(target_route.node_list))
        target_route.node_list.insert(target_pos, node)
        target_route.update(data)
    s.update(data)
    s.cal_cost(data)
    return True


def _light_mutation(s: Solution, data, rng: random.Random) -> None:
    if rng.random() < 0.5:
        _swap_two_customers(s, data, rng)
    else:
        _relocate_customer(s, data, rng)
    feasible_or_repair(s, data, rng)


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
    for segment in segments:
        if rng.random() < 0.70:
            flag, _ = _chk_route_list([data.DC] + segment + [data.DC], data)
            if flag:
                child.append(_make_route(segment, data))
                continue
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


def _woa_intensification(
    current: Solution, best: Solution, a: float, data, rng: random.Random
) -> Solution:
    r1 = rng.random()
    r2 = rng.random()
    a_vector = 2.0 * a * r1 - a
    c_vector = 2.0 * r2

    if abs(a_vector) < 1.0:
        child = current.clone()
        if rng.random() < 0.5:
            _inject_elite_routes(child, best, data, rng, c_vector)
        else:
            _inject_elite_segments(child, best, data, rng)
        feasible_or_repair(child, data, rng)
        return child

    child = _li_lim_random_search(current, data, rng)
    feasible_or_repair(child, data, rng)
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
    data.rng = rng

    if rng.random() < task.p_hybrid:
        new_solution = _guided_route_crossover(task.best, task.peer, task.current, data, rng)
        if rng.random() < data.sho_mutation_prob:
            _light_mutation(new_solution, data, rng)
        feasible_or_repair(new_solution, data, rng)
    else:
        new_solution = _woa_intensification(task.current, task.best, task.a, data, rng)

    new_solution.cal_cost(data)
    accepted = _sa_accept(
        new_solution, task.current, task.current_fit, task.iteration, task.max_iter, rng
    )
    if accepted:
        return task.index, new_solution, new_solution.cost, True

    return task.index, task.current, task.current_fit, False


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


def _deep_local_search_best(s: Solution, data) -> None:
    saved_small_opts = list(data.small_opts)
    saved_mem: Dict[str, List[Move]] = data.mem
    saved_escape = data.escape_local_optima
    saved_skip = data.skip_finding_lo
    saved_vehicle_max_num = data.vehicle.max_num

    try:
        data.skip_finding_lo = False
        data.escape_local_optima = 0
        data.vehicle.max_num = max(data.vehicle.max_num, s.len() + 2)

        _install_move_memory(data, ["2opt"])
        do_local_search(s, data)
        _install_move_memory(data, ["oropt_single", "oropt_double"])
        do_local_search(s, data)
        _install_move_memory(data, ["2exchange"])
        do_local_search(s, data)
        s.update(data)
        s.cal_cost(data)
    finally:
        data.small_opts = saved_small_opts
        data.mem = saved_mem
        data.escape_local_optima = saved_escape
        data.skip_finding_lo = saved_skip
        data.vehicle.max_num = saved_vehicle_max_num


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
    _insert_customers(candidate, pending, data, rng)
    feasible_or_repair(candidate, data, rng, shuffle_pending=True)
    return candidate


def _diversify(pop, pop_fit, pop_argrank, best_s, data) -> None:
    argsort(pop_fit, pop_argrank, len(pop))
    elite_index = pop_argrank[0]
    pop[elite_index].copy_from(best_s)
    pop_fit[elite_index] = best_s.cost

    non_elite = [index for index in range(len(pop)) if index != elite_index]
    if len(non_elite) == 0:
        return
    data.rng.shuffle(non_elite)
    diversify_count = max(1, int(round(len(non_elite) * data.diversify_ratio)))
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

    while run <= data.runs:
        print("---------------------------------Run %d---------------------------" % run)
        initialization(pop, pop_fit, pop_argrank, data)
        used = int(time.perf_counter() - stime)
        print("already consumed %d sec" % used)

        update_best_solution(pop[pop_argrank[0]], best_s, used, run, 0, data)
        output(pop, pop_fit, pop_argrank, data)

        last_improvement_gen = 0
        executor = None
        if data.parallel_workers != 1 and data.p_size > 1 and not getattr(data.backend, "is_cuda", False):
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=_worker_count(data))

        try:
            for gen in range(1, data.max_iter + 1):
                best_before_generation = best_s.cost
                a, p_hybrid = _dynamic_parameters(gen, data.max_iter)
                p_mode = _mode_probability(p_hybrid, data)
                best_snapshot = best_s.clone()
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
                            iteration=gen,
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
                update_best_solution(pop[pop_argrank[0]], best_s, used, run, gen, data)

                if gen % data.local_search_interval == 0:
                    print("Periodic deep local search on global best.")
                    elite = best_s.clone()
                    _deep_local_search_best(elite, data)
                    update_best_solution(elite, best_s, used, run, gen, data)
                    _inject_global_best(pop, pop_fit, pop_argrank, best_s)

                if best_s.cost - best_before_generation < -PRECISION:
                    last_improvement_gen = gen

                if (
                    gen % data.stagnation_interval == 0
                    and gen - last_improvement_gen >= data.stagnation_interval
                ):
                    print("Stagnation detected. Diversifying 40%% of non-elite population.")
                    _diversify(pop, pop_fit, pop_argrank, best_s, data)
                    last_improvement_gen = gen

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
                        % (gen, gen - last_improvement_gen, used)
                    )

                if data.tmax != -1 and used > int(data.tmax):
                    time_exhausted = True
                    break
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        print("Run %d finishes" % run)
        output(pop, pop_fit, pop_argrank, data)
        completed_runs += 1

        data.rng.seed(data.seed + run)
        if time_exhausted:
            break
        run += 1

    print("------------Summary-----------")
    print("Total %d runs, total consumed %d sec" % (completed_runs, int(used)))
    best_s.output(data)
    print(
        "In run %d, gen %d, find this solution, at time %d."
        % (state.find_best_run, state.find_best_gen, int(state.find_best_time))
    )
    print("Time to surpass BKS: %d." % int(state.find_bks_time))
    best_s.check(data)
