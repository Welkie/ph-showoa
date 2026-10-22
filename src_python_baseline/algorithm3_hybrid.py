from __future__ import annotations

import random
from typing import List, Tuple
from .problem import Problem
from .solution import Solution
from .algorithm4_woa import based_on_woa_update
from .algorithm5_sho import based_on_sho_update


def update_position_hybrid(
    current: Solution,
    best_whale: Solution,
    population: List[Solution],
    p_hybrid: float,
    a: float,
    problem: Problem,
    rng: random.Random,
    mutation_prob: float = 0.35,
) -> Tuple[Solution, str]:
    """
    Algorithm 3. The Hybrid Algorithm: BasedOnWOASHO
    Line 3: p <- random in [0, 1]
    Line 4: if p < pHybrid then
    Line 5:     // SHO-based diversification (exploration)
    Line 6:     candidate <- BasedOnSHOUpdate(current, partner)
    Line 7: else
    Line 8:     // WOA-based intensification (exploitation)
    Line 9:     candidate <- BasedOnWOAUpdate(current)
    """
    p = rng.random()
    if p < p_hybrid:
        sol = based_on_sho_update(
            hyena=current,
            best_hyena=best_whale,
            population=population,
            problem=problem,
            rng=rng,
            mutation_prob=mutation_prob,
        )
        return sol, "SHO"
    else:
        sol = based_on_woa_update(
            whale=current,
            best_whale=best_whale,
            a=a,
            problem=problem,
            rng=rng,
        )
        return sol, "WOA"
