from __future__ import annotations

from typing import List, Tuple
from .problem import Problem
from .solution import Solution


def check_feasibility_canonical(solution: Solution, problem: Problem) -> Tuple[bool, str]:
    """
    Standard Canonical Feasibility Check for VRPSPDTW (Priority Level 1: Section 3 Eq. 1-11).
    Resolves the contradiction between Eq. (5) and Algorithm 9:
    - Initial vehicle load at depot = sum of all delivery demands along the route (Std. 1): L_0 = sum(d_i) <= Q.
    - Cumulative dynamic load after customer i (Std. 2): L_k = L_{k-1} - d_i + p_i.
    - Capacity constraint (Std. 3): 0 <= L_k <= Q for all k in [0, m].
    - Hard time windows: Arrival arr_i <= due_i; vehicle waits if arr_i < ready_i; return to depot before due_0.
    - Exactly-once customer service: Every customer in 1..N visited exactly once.
    - Fleet size constraint: NV <= max_vehicles.
    """
    if solution.nv > problem.max_vehicles:
        return False, f"Fleet size exceeded: {solution.nv} > {problem.max_vehicles}"

    visited = [False] * (problem.num_customers + 1)
    for r_idx, r in enumerate(solution.routes):
        if not r:
            continue

        # 1. Initial depot load
        route_delivery = sum(problem.delivery[c] for c in r)
        if route_delivery > problem.capacity + 1e-9:
            return False, f"Route {r_idx} initial delivery exceeds capacity: {route_delivery:.2f} > {problem.capacity}"

        load = route_delivery
        time = problem.ready_time[0]
        prev = 0

        for c in r:
            if c <= 0 or c > problem.num_customers:
                return False, f"Invalid customer id: {c}"

            # Time window check
            time += problem.travel_time[prev][c]
            if time < problem.ready_time[c]:
                time = problem.ready_time[c]
            if time > problem.due_time[c] + 1e-9:
                return False, f"Time window violation at customer {c}: arrival {time:.2f} > due {problem.due_time[c]:.2f}"

            # Dynamic load transition (Section 3, Eq. 5)
            load = load - problem.delivery[c] + problem.pickup[c]
            if load < -1e-9 or load > problem.capacity + 1e-9:
                return False, f"Capacity violation at customer {c}: load={load:.2f} outside [0, {problem.capacity}]"

            if visited[c]:
                return False, f"Duplicate visit: customer {c}"
            visited[c] = True

            time += problem.service_time[c]
            prev = c

        # Depot return check
        time += problem.travel_time[prev][0]
        if time > problem.due_time[0] + 1e-9:
            return False, f"Depot return time window violation on route {r_idx}: return {time:.2f} > due {problem.due_time[0]:.2f}"

    # Missing customers check
    for k in range(1, problem.num_customers + 1):
        if not visited[k]:
            return False, f"Missing customer: {k}"

    return True, ""


def is_route_capacity_feasible(route: List[int], problem: Problem) -> bool:
    """
    Checks vehicle capacity feasibility under Section 3 Eq. (5) dynamic load model.
    Used by Crossover (Algorithm 6) and Repair (Algorithm 10).
    """
    if not route:
        return True

    # Initial departure load at depot L_0 = sum(d_i) <= Q
    load = sum(problem.delivery[c] for c in route)
    if load > problem.capacity + 1e-9:
        return False

    # Dynamic transition L_k = L_{k-1} - d_k + p_k
    for c in route:
        load = load - problem.delivery[c] + problem.pickup[c]
        if load < -1e-9 or load > problem.capacity + 1e-9:
            return False

    return True


def is_route_feasible(route: List[int], problem: Problem) -> bool:
    """
    Checks feasibility of a single route under Section 3 constraints.
    Fast check used during local search, crossover, and repair.
    """
    if not route:
        return True

    # 1. Initial departure load
    load = sum(problem.delivery[c] for c in route)
    if load > problem.capacity + 1e-9:
        return False

    time = problem.ready_time[0]
    prev = 0

    for curr in route:
        if curr <= 0 or curr > problem.num_customers:
            return False

        time += problem.travel_time[prev][curr]
        if time < problem.ready_time[curr]:
            time = problem.ready_time[curr]
        if time > problem.due_time[curr] + 1e-9:
            return False

        time += problem.service_time[curr]
        load = load - problem.delivery[curr] + problem.pickup[curr]
        if load < -1e-9 or load > problem.capacity + 1e-9:
            return False
        prev = curr

    # Depot return
    time += problem.travel_time[prev][0]
    if time > problem.due_time[0] + 1e-9:
        return False

    return True


def is_solution_feasible(solution: Solution, problem: Problem) -> bool:
    """Quick boolean check for candidate solution during optimization."""
    is_feas, _ = check_feasibility_canonical(solution, problem)
    return is_feas
