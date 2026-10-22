from __future__ import annotations

import math
import random
from typing import List
from .problem import Problem
from .solution import Solution
from .algorithm7_mutation import _apply_intra_operation, _apply_inter_operation
from .algorithm9_feasibility import is_route_feasible, is_solution_feasible
from .algorithm10_repair import repair_procedure


def construct_initial_solution(problem: Problem, rng: random.Random) -> Solution:
    """
    Constructs an initial solution S0:
    Customer order sorted by due date with randomized perturbation across seeds.
    Each customer is inserted into the best feasible position across existing routes,
    or opens a new route if no feasible insertion exists.
    Priority Level 4 selection: the paper takes S0 as input (Algorithm 2) but does not define how it is built.
    """
    customers = list(range(1, problem.num_customers + 1))
    rng.shuffle(customers)
    customers.sort(key=lambda c: problem.due_time[c] + rng.uniform(-10.0, 10.0))

    routes: List[List[int]] = []

    for c in customers:
        best_delta = float("inf")
        best_pos = (-1, -1)
        for r_idx, r in enumerate(routes):
            orig_dist = _route_distance(r, problem)
            for pos in range(len(r) + 1):
                cand = r[:pos] + [c] + r[pos:]
                if is_route_feasible(cand, problem):
                    delta = _route_distance(cand, problem) - orig_dist
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = (r_idx, pos)
        if best_pos[0] != -1:
            routes[best_pos[0]].insert(best_pos[1], c)
        else:
            routes.append([c])

    sol = Solution(routes, problem)
    repaired = repair_procedure(sol, problem)
    return repaired


def initialize_solution_sa(
    s0: Solution,
    problem: Problem,
    rng: random.Random,
    t0: float = 1.0,
    alpha: float = 0.9,
    t_min: float = 0.01,
    iter_max: int = 20,
) -> Solution:
    """
    Algorithm 2. The Initialization Procedure for VRPSPDTW
    Simulated Annealing on Total Cost (Priority Level 2: pseudo-code lines 2, 7, 10 use CalculateObjective).
    T0=1.0, alpha=0.9, t_min=0.01, iter_max=20 are Priority Level 4 selections (Table 2 does not specify SA parameters).
    Perturbs S via swap, insert, reverse, Pd-shift, Pd-exchange.
    """
    s = s0.copy()
    s_best = s.copy()
    e_best = s.cost
    t = t0

    while t > t_min:
        for _ in range(iter_max):
            s_prime = _perturb(s, problem, rng)
            if is_solution_feasible(s_prime, problem):
                cf = s.cost
                nf = s_prime.cost
                delta = nf - cf
                accept = False
                if delta < 0:
                    accept = True
                else:
                    denom = t * abs(cf) + 1e-6
                    prob = math.exp(-delta / denom) if denom > 0 else 0.0
                    if rng.random() < prob:
                        accept = True

                if accept:
                    s = s_prime
                    if s < s_best or s.cost < e_best:
                        s_best = s.copy()
                        e_best = s.cost
        t = alpha * t

    return s_best


def _perturb(solution: Solution, problem: Problem, rng: random.Random) -> Solution:
    routes = [list(r) for r in solution.routes if len(r) > 0]
    num_routes = len(routes)
    if num_routes == 0:
        return solution.copy()

    if rng.random() < 0.5 or num_routes < 2:
        r_idx = rng.randint(0, num_routes - 1)
        r = routes[r_idx]
        if len(r) >= 2:
            op = rng.randint(0, 2)
            _apply_intra_operation(r, op, rng)
    else:
        op = rng.randint(0, 1)
        _apply_inter_operation(routes, op, rng)

    routes = [r for r in routes if len(r) > 0]
    return Solution(routes, problem)


def _route_distance(route: List[int], problem: Problem) -> float:
    if not route:
        return 0.0
    dist = problem.distance[0][route[0]]
    for i in range(len(route) - 1):
        dist += problem.distance[route[i]][route[i + 1]]
    dist += problem.distance[route[-1]][0]
    return dist
