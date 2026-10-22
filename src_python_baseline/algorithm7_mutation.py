from __future__ import annotations

import random
from typing import List
from .problem import Problem
from .solution import Solution
from .algorithm9_feasibility import is_route_feasible


def mutate_solution(solution: Solution, problem: Problem, rng: random.Random) -> Solution:
    """
    Algorithm 7. MutationOperators
    Randomly chooses SWAP (intra-route move: swap, insert, reverse)
    or RELOCATE (inter-route move: Pd-shift, Pd-exchange).
    Rolls back if constraints are violated.
    """
    if not solution.routes:
        return solution.copy()

    routes = [list(r) for r in solution.routes if len(r) > 0]
    num_routes = len(routes)
    if num_routes == 0:
        return solution.copy()

    use_intra = rng.random() < 0.5 or num_routes < 2

    if use_intra:
        r_idx = rng.randint(0, num_routes - 1)
        r = routes[r_idx]
        if len(r) >= 2:
            backup_r = list(r)
            op = rng.randint(0, 2)  # 0: swap, 1: insert, 2: reverse
            _apply_intra_operation(r, op, rng)
            if not is_route_feasible(r, problem):
                routes[r_idx] = backup_r
    else:
        backup_routes = [list(r) for r in routes]
        op = rng.randint(0, 1)  # 0: Pd-shift, 1: Pd-exchange
        _apply_inter_operation(routes, op, rng)
        all_ok = all(is_route_feasible(r, problem) for r in routes)
        if not all_ok:
            routes = backup_routes

    # Clean empty routes
    routes = [r for r in routes if len(r) > 0]
    return Solution(routes, problem)


def _apply_intra_operation(r: List[int], op: int, rng: random.Random) -> None:
    n = len(r)
    if n < 2:
        return
    if op == 0:
        # Swap two distinct customer positions
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        while j == i and n > 1:
            j = rng.randint(0, n - 1)
        r[i], r[j] = r[j], r[i]
    elif op == 1:
        # Insert: relocate customer to another position
        i = rng.randint(0, n - 1)
        c = r.pop(i)
        new_pos = rng.randint(0, len(r))
        r.insert(new_pos, c)
    elif op == 2:
        # Inversion: reverse segment
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        r[i : j + 1] = r[i : j + 1][::-1]


def _apply_inter_operation(routes: List[List[int]], op: int, rng: random.Random) -> None:
    if len(routes) < 2:
        return
    r1_idx = rng.randint(0, len(routes) - 1)
    r2_idx = rng.randint(0, len(routes) - 1)
    while r2_idx == r1_idx:
        r2_idx = rng.randint(0, len(routes) - 1)

    r1 = routes[r1_idx]
    r2 = routes[r2_idx]
    if not r1:
        return

    if op == 0:
        # Pd-shift: move one customer from r1 to r2
        pos1 = rng.randint(0, len(r1) - 1)
        c = r1.pop(pos1)
        pos2 = rng.randint(0, len(r2))
        r2.insert(pos2, c)
    else:
        # Pd-exchange: swap customer in r1 with customer in r2
        if not r2:
            return
        pos1 = rng.randint(0, len(r1) - 1)
        pos2 = rng.randint(0, len(r2) - 1)
        r1[pos1], r2[pos2] = r2[pos2], r1[pos1]
