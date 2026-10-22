from __future__ import annotations

import random
from typing import List
from .problem import Problem
from .solution import Solution
from .algorithm6_crossover import crossover_guided
from .algorithm7_mutation import mutate_solution
from .algorithm10_repair import repair_procedure
from .algorithm9_feasibility import is_solution_feasible


def tournament_select(population: List[Solution], k: int, rng: random.Random) -> Solution:
    """Tournament selection without replacement using k individuals (Paper default k=3)."""
    n = len(population)
    k_actual = min(k, n)
    indices = rng.sample(range(n), k_actual)
    contenders = [population[i] for i in indices]
    winner = contenders[0]
    for cand in contenders[1:]:
        if cand < winner:
            winner = cand
    return winner


def based_on_sho_update(
    hyena: Solution,
    best_hyena: Solution,
    population: List[Solution],
    problem: Problem,
    rng: random.Random,
    mutation_prob: float = 0.35,
) -> Solution:
    """
    Algorithm 5. BasedOnSHOUpdate (Population-Guided Diversification)
    Recombines information from best, a tournament partner, and current hyena,
    with probabilistic mutation (pm=0.35) and feasibility repair.
    Table 2 lists h, rd1, rd2 for SHO, but Algorithm 5 does not use them; this follows the pseudo-code.
    """
    base = hyena.copy()

    # Tournament selection for partner (Algorithm 3, line 5)
    partner = tournament_select(population, k=3, rng=rng)

    # Line 4: CrossoverGuided
    combined = crossover_guided(best_hyena, partner, hyena, problem, rng)

    # Line 6-8: Mutation with probability pm = 0.35
    if rng.random() < mutation_prob:
        combined = mutate_solution(combined, problem, rng)

    # Line 10-12: Feasible_Or_Repair
    if not is_solution_feasible(combined, problem):
        combined = repair_procedure(combined, problem)
        if not is_solution_feasible(combined, problem):
            return base

    # Lines 14-21: Return better solution (page 15: compared against reference/current solution)
    if combined.cost < base.cost:
        return combined
    else:
        return base
