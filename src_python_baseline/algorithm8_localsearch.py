from __future__ import annotations

from typing import List, Optional
from .problem import Problem
from .solution import Solution
from .algorithm9_feasibility import is_route_feasible


def combined_local_search(solution: Solution, problem: Problem) -> Solution:
    """
    Algorithm 8. CombinedLocalSearch
    Sequential first-improvement exploration:
    (a) 2-opt: intra-route edge reversal
    (b) Relocate: single-customer reinsertion (intra and inter route)
    (c) Swap: customer exchange between distinct routes
    Restarts from (a) on the first strict improvement.
    Note on feasibleOrRepair: We evaluate direct route feasibility rather than invoking
    the heavy Algorithm 10 repair on every single candidate move to preserve practical
    polynomial time complexity while maintaining strict solution validity.
    """
    current = solution.copy()
    improved = True

    while improved:
        improved = False

        # (a) 2-opt: intra-route edge reversal
        new_sol = _first_two_opt(current, problem)
        if new_sol is not None:
            current = new_sol
            improved = True
            continue

        # (b) Relocate: move one customer to another position / route
        new_sol = _first_relocate(current, problem)
        if new_sol is not None:
            current = new_sol
            improved = True
            continue

        # (c) Swap: exchange customers between distinct routes
        new_sol = _first_swap(current, problem)
        if new_sol is not None:
            current = new_sol
            improved = True
            continue

    return current


def _first_two_opt(current: Solution, problem: Problem) -> Optional[Solution]:
    routes = [list(r) for r in current.routes]
    for r_idx, r in enumerate(routes):
        n = len(r)
        if n < 2:
            continue
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand_r = r[:i] + r[i : j + 1][::-1] + r[j + 1 :]
                if is_route_feasible(cand_r, problem):
                    cand_routes = [list(x) for x in routes]
                    cand_routes[r_idx] = cand_r
                    cand_sol = Solution(cand_routes, problem)
                    if _is_strictly_better(cand_sol, current):
                        return cand_sol
    return None


def _first_relocate(current: Solution, problem: Problem) -> Optional[Solution]:
    routes = [list(r) for r in current.routes]
    num_routes = len(routes)

    for r_src_idx in range(num_routes):
        r_src = routes[r_src_idx]
        for pos_src in range(len(r_src)):
            c = r_src[pos_src]
            for r_dst_idx in range(num_routes):
                r_dst = routes[r_dst_idx]
                max_pos = len(r_dst) if r_dst_idx != r_src_idx else len(r_dst) - 1
                for pos_dst in range(max_pos + 1):
                    if r_dst_idx == r_src_idx and (pos_dst == pos_src or pos_dst == pos_src + 1):
                        continue
                    cand_routes = [list(x) for x in routes]
                    cand_src = list(cand_routes[r_src_idx])
                    cand_src.pop(pos_src)
                    if r_dst_idx == r_src_idx:
                        cand_src.insert(pos_dst if pos_dst < pos_src else pos_dst - 1, c)
                        if is_route_feasible(cand_src, problem):
                            cand_routes[r_src_idx] = cand_src
                            cand_sol = Solution(cand_routes, problem)
                            if _is_strictly_better(cand_sol, current):
                                return cand_sol
                    else:
                        cand_dst = list(cand_routes[r_dst_idx])
                        cand_dst.insert(pos_dst, c)
                        if is_route_feasible(cand_src, problem) and is_route_feasible(cand_dst, problem):
                            cand_routes[r_src_idx] = cand_src
                            cand_routes[r_dst_idx] = cand_dst
                            cand_sol = Solution(cand_routes, problem)
                            if _is_strictly_better(cand_sol, current):
                                return cand_sol
    return None


def _first_swap(current: Solution, problem: Problem) -> Optional[Solution]:
    routes = [list(r) for r in current.routes]
    num_routes = len(routes)
    if num_routes < 2:
        return None

    for r1_idx in range(num_routes - 1):
        r1 = routes[r1_idx]
        for r2_idx in range(r1_idx + 1, num_routes):
            r2 = routes[r2_idx]
            for p1 in range(len(r1)):
                for p2 in range(len(r2)):
                    cand_r1 = list(r1)
                    cand_r2 = list(r2)
                    cand_r1[p1], cand_r2[p2] = cand_r2[p2], cand_r1[p1]
                    if is_route_feasible(cand_r1, problem) and is_route_feasible(cand_r2, problem):
                        cand_routes = [list(x) for x in routes]
                        cand_routes[r1_idx] = cand_r1
                        cand_routes[r2_idx] = cand_r2
                        cand_sol = Solution(cand_routes, problem)
                        if _is_strictly_better(cand_sol, current):
                            return cand_sol
    return None


def _is_strictly_better(cand: Solution, baseline: Solution) -> bool:
    if cand.nv < baseline.nv:
        return True
    if cand.nv == baseline.nv and cand.td < baseline.td - 1e-4:
        return True
    return False
