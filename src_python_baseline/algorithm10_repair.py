from __future__ import annotations

from typing import List, Optional
from .problem import Problem
from .solution import Solution
from .algorithm9_feasibility import is_route_feasible, is_solution_feasible


def repair_procedure(solution: Solution, problem: Problem) -> Solution:
    """
    Algorithm 10. RepairProcedure
    Step 1: Repair duplicate and missing customers (exactly-once constraint)
    Step 2: Repair capacity violations
    Step 3: Repair time-window violations
    Step 4: Local route repair (intra-route 2-opt)
    Step 5: Final validation under Section 3 constraints
    """
    routes = [list(r) for r in solution.routes if len(r) > 0]

    # Step 1: Repair duplicate customers
    visited_routes: dict[int, List[int]] = {}
    for r_idx, r in enumerate(routes):
        for c in r:
            visited_routes.setdefault(c, []).append(r_idx)

    for c, r_indices in visited_routes.items():
        if len(r_indices) > 1:
            best_r_idx = r_indices[0]
            best_insertion_cost = float("inf")
            for r_idx in r_indices:
                r = routes[r_idx]
                r_without_c = [x for x in r if x != c]
                # Algorithm 10 line 5: marginal insertion cost = distance(r) - distance(r without c)
                delta_cost = _route_distance(r, problem) - _route_distance(r_without_c, problem)
                if delta_cost < best_insertion_cost:
                    best_insertion_cost = delta_cost
                    best_r_idx = r_idx
            # Remove c from all other routes
            for r_idx in r_indices:
                if r_idx != best_r_idx:
                    routes[r_idx] = [x for x in routes[r_idx] if x != c]

    routes = [r for r in routes if len(r) > 0]

    # Step 1 (cont): Repair missing customers
    visited = set()
    for r in routes:
        visited.update(r)

    unvisited = [c for c in range(1, problem.num_customers + 1) if c not in visited]
    for c in unvisited:
        inserted = False
        best_delta = float("inf")
        best_pos = (-1, -1)
        for r_idx, r in enumerate(routes):
            orig_cost = _route_distance(r, problem)
            for pos in range(len(r) + 1):
                cand = r[:pos] + [c] + r[pos:]
                if is_route_feasible(cand, problem):
                    delta = _route_distance(cand, problem) - orig_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = (r_idx, pos)
        if best_pos[0] != -1:
            routes[best_pos[0]].insert(best_pos[1], c)
            inserted = True
        if not inserted:
            routes.append([c])

    # Step 2: Repair capacity violations
    for r_idx in range(len(routes)):
        r = routes[r_idx]
        attempts = 0
        while attempts < len(r) and not _is_route_capacity_feasible(r, problem):
            attempts += 1
            # Select customer c with largest |delivery - pickup|
            c_sel = max(r, key=lambda x: abs(problem.delivery[x] - problem.pickup[x]))
            r.remove(c_sel)
            # Try to insert c into another route
            placed = False
            best_delta = float("inf")
            best_target = (-1, -1)
            for other_idx, other_r in enumerate(routes):
                if other_idx == r_idx:
                    continue
                orig_cost = _route_distance(other_r, problem)
                for pos in range(len(other_r) + 1):
                    cand = other_r[:pos] + [c_sel] + other_r[pos:]
                    if is_route_feasible(cand, problem):
                        delta = _route_distance(cand, problem) - orig_cost
                        if delta < best_delta:
                            best_delta = delta
                            best_target = (other_idx, pos)
            if best_target[0] != -1:
                routes[best_target[0]].insert(best_target[1], c_sel)
                placed = True
            if not placed:
                routes.append([c_sel])

    routes = [r for r in routes if len(r) > 0]

    # Step 3: Repair time-window violations
    for r_idx in range(len(routes)):
        r = routes[r_idx]
        attempts = 0
        while attempts < len(r):
            late_c = _find_first_late_customer(r, problem)
            if late_c is None:
                break
            attempts += 1
            r.remove(late_c)
            placed = False
            best_arrival_time = float("inf")
            best_target = (-1, -1)
            for target_idx, target_r in enumerate(routes):
                for pos in range(len(target_r) + 1):
                    cand = target_r[:pos] + [late_c] + target_r[pos:]
                    if is_route_feasible(cand, problem):
                        # Algorithm 10 line 31: Reinsert c at best feasible position (earliest arrival)
                        arr_time = _calculate_arrival_time_at_pos(target_r, late_c, pos, problem)
                        if arr_time < best_arrival_time:
                            best_arrival_time = arr_time
                            best_target = (target_idx, pos)
            if best_target[0] != -1:
                routes[best_target[0]].insert(best_target[1], late_c)
                placed = True
            if not placed:
                routes.append([late_c])

    routes = [r for r in routes if len(r) > 0]

    # Step 4: Intra-route 2-opt
    for r in routes:
        _apply_intra_2opt(r, problem)

    repaired = Solution(routes, problem)

    # Step 5: Final validation
    if is_solution_feasible(repaired, problem):
        return repaired
    return solution


def _route_distance(route: List[int], problem: Problem) -> float:
    if not route:
        return 0.0
    dist = problem.distance[0][route[0]]
    for i in range(len(route) - 1):
        dist += problem.distance[route[i]][route[i + 1]]
    dist += problem.distance[route[-1]][0]
    return dist


def _is_route_capacity_feasible(route: List[int], problem: Problem) -> bool:
    load = sum(problem.delivery[c] for c in route)
    if load > problem.capacity + 1e-9:
        return False
    for c in route:
        load = load - problem.delivery[c] + problem.pickup[c]
        if load < -1e-9 or load > problem.capacity + 1e-9:
            return False
    return True


def _find_first_late_customer(route: List[int], problem: Problem) -> Optional[int]:
    time = problem.ready_time[0]
    prev = 0
    for c in route:
        time += problem.travel_time[prev][c]
        if time < problem.ready_time[c]:
            time = problem.ready_time[c]
        if time > problem.due_time[c] + 1e-9:
            return c
        time += problem.service_time[c]
        prev = c
    time += problem.travel_time[prev][0]
    if time > problem.due_time[0] + 1e-9:
        return route[-1]
    return None


def _calculate_arrival_time_at_pos(route: List[int], customer: int, pos: int, problem: Problem) -> float:
    """Computes exact vehicle arrival time at candidate customer when inserted at pos."""
    cand = route[:pos] + [customer] + route[pos:]
    time = problem.ready_time[0]
    prev = 0
    for idx, c in enumerate(cand):
        time += problem.travel_time[prev][c]
        if time < problem.ready_time[c]:
            time = problem.ready_time[c]
        if idx == pos:
            return time
        time += problem.service_time[c]
        prev = c
    return time


def _apply_intra_2opt(route: List[int], problem: Problem) -> None:
    if len(route) < 2:
        return
    best_dist = _route_distance(route, problem)
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                cand = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                if is_route_feasible(cand, problem):
                    d = _route_distance(cand, problem)
                    if d < best_dist - 1e-6:
                        route[:] = cand
                        best_dist = d
                        improved = True
                        break
            if improved:
                break
