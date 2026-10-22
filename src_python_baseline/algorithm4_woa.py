from __future__ import annotations

import math
import random
from typing import List
from .problem import Problem
from .solution import Solution
from .algorithm7_mutation import _apply_intra_operation, _apply_inter_operation
from .algorithm9_feasibility import is_route_feasible, is_solution_feasible
from .algorithm10_repair import repair_procedure


def based_on_woa_update(
    whale: Solution,
    best_whale: Solution,
    a: float,
    problem: Problem,
    rng: random.Random,
) -> Solution:
    """
    Algorithm 4. BasedOnWOAUpdate (Best-Guided Intensification)
    Encircling prey, spiral movement, and controlled local perturbations
    anchored to the current global best solution.
    """
    routes = [list(r) for r in whale.routes if len(r) > 0]
    best_routes = [list(r) for r in best_whale.routes if len(r) > 0]

    dim = len(routes)
    best_dim = len(best_routes)
    limit = min(dim, best_dim)

    coeff_a = [2.0 * a * rng.random() - a for _ in range(dim)]
    coeff_c = [2.0 * rng.random() for _ in range(dim)]

    for i in range(limit):
        p = rng.random()
        route = routes[i]
        best_route = best_routes[i]
        backup_all_routes = [list(r) for r in routes]

        ai = coeff_a[i]
        ci = coeff_c[i]

        if p < 0.5:
            if abs(ai) < 1.0:
                # ENCIRCLING_PREY (line 18): not defined in the paper. Priority Level 4 selection:
                # align the order of shared customers toward best_route.
                fraction = (1.0 - abs(ai)) * (ci / 2.0)
                _relink(route, best_route, fraction, ai >= 0.0)
            else:
                if rng.random() < 0.5:
                    # ApplyRandomOperation (intra: swap, insert, 2-opt)
                    op = rng.randint(0, 2)
                    _apply_intra_operation(route, op, rng)
                else:
                    # ApplyRandomMultiRouteOperation (inter: Pd-shift, Pd-exchange)
                    op = rng.randint(0, 1)
                    _apply_inter_operation(routes, op, rng)
                    all_ok = all(is_route_feasible(r, problem) for r in routes)
                    if not all_ok:
                        routes = backup_all_routes
                        route = routes[i]
        else:
            # SPIRAL_MOVEMENT (line 31): not defined in the paper. Priority Level 4 selection:
            # fraction from the original WOA spiral e^l * cos(2*pi*l).
            l = ci - 1.0
            fraction = min(1.0, abs(math.exp(l) * math.cos(2.0 * math.pi * l)))
            _relink(route, best_route, fraction, l >= 0.0)

        # Lines 33-34: route feasibility check. The paper restores only route i; we restore the
        # whole pre-step copy because a multi-route move (line 24) may also have changed other routes,
        # and restoring route i alone could duplicate or drop customers.
        if not is_route_feasible(route, problem):
            routes = backup_all_routes

    # Clean empty routes
    routes = [r for r in routes if len(r) > 0]
    cand = Solution(routes, problem)

    # Lines 37-38: ValidateRoutes fallback to single-customer routes if invalid
    if not is_solution_feasible(cand, problem):
        fallback_routes = [[c] for c in range(1, problem.num_customers + 1)]
        cand = Solution(fallback_routes, problem)

    # Lines 42-44 & 50: Accept candidate only if better than current whale
    if cand.cost < whale.cost:
        return cand
    else:
        return whale.copy()


def _relink(route: List[int], target: List[int], fraction: float, forward: bool) -> None:
    if len(route) < 2 or not target:
        return

    # Find common customers
    target_pos = {c: idx for idx, c in enumerate(target)}
    common = [c for c in route if c in target_pos]
    if len(common) < 2:
        return

    num_to_reorder = max(2, int(round(len(common) * max(0.1, min(1.0, fraction)))))
    chosen_common = common[:num_to_reorder]

    if forward:
        chosen_common.sort(key=lambda c: target_pos[c])
    else:
        chosen_common.sort(key=lambda c: target_pos[c], reverse=True)

    # Place sorted back into route at original positions
    common_set = set(chosen_common)
    c_idx = 0
    for r_idx in range(len(route)):
        if route[r_idx] in common_set:
            route[r_idx] = chosen_common[c_idx]
            c_idx += 1
