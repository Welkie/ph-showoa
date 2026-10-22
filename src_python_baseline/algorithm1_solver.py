from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

from .problem import Problem
from .solution import Solution
from .algorithm2_initialization import construct_initial_solution, initialize_solution_sa
from .algorithm3_hybrid import update_position_hybrid
from .algorithm7_mutation import mutate_solution
from .algorithm8_localsearch import combined_local_search
from .algorithm9_feasibility import check_feasibility_canonical, is_solution_feasible


@dataclass
class SolverConfig:
    pop_size: int = 30
    max_iter: int = 1000
    seed: int = 42
    local_search_interval: int = 25
    stagnation_limit: int = 50
    epsilon: float = 1e-4
    mutation_prob: float = 0.35
    workers: int = 1
    # Initialization SA parameters (Algorithm 2)
    t0_sa: float = 1.0
    alpha_sa: float = 0.9
    t_min_sa: float = 0.01
    iter_max_sa: int = 20


@dataclass
class RunResult:
    instance: str
    algo: str
    run: int
    seed: int
    nv: int
    td: float
    cost: float
    feasible: bool
    violation_note: str
    best_solution: Solution
    elapsed_seconds: float


class PHSHOWOASolver:
    def __init__(self, problem: Problem, config: Optional[SolverConfig] = None) -> None:
        self.problem = problem
        self.config = config or SolverConfig()

    def solve(self, run_index: int = 1) -> RunResult:
        start_time = time.time()
        rng = random.Random(self.config.seed)

        # Algorithm 1: Lines 1-2: Initialize population using Algorithm 2
        pop_seeds = [self.config.seed + i * 1000 for i in range(self.config.pop_size)]
        population: List[Solution] = []

        for p_seed in pop_seeds:
            p_rng = random.Random(p_seed)
            s0 = construct_initial_solution(self.problem, p_rng)
            s_init = initialize_solution_sa(
                s0=s0,
                problem=self.problem,
                rng=p_rng,
                t0=self.config.t0_sa,
                alpha=self.config.alpha_sa,
                t_min=self.config.t_min_sa,
                iter_max=self.config.iter_max_sa,
            )
            population.append(s_init)

        # Lines 3-5: best individual in population
        best = min(population, key=lambda s: (s.nv, s.td)).copy()
        best_so_far = best.cost
        no_improve = 0

        # Lines 6-49: Main iteration loop
        pool = ThreadPoolExecutor(max_workers=self.config.workers) if self.config.workers > 1 else None

        try:
            for iteration in range(self.config.max_iter):
                # Line 7: a linearly decreases from 2 to 0
                a = 2.0 - 2.0 * iteration / self.config.max_iter
                # Line 8: pHybrid decreases adaptively
                p_hybrid = max(0.15, 0.5 * (1.0 - iteration / self.config.max_iter))
                temperature = 1.0 - iteration / self.config.max_iter

                # Lines 10-32: Parallel hybrid update
                def update_individual(idx: int) -> Solution:
                    w = population[idx]
                    w_rng = random.Random(self.config.seed + iteration * 10000 + idx)
                    new_sol, _ = update_position_hybrid(
                        current=w,
                        best_whale=best,
                        population=population,
                        p_hybrid=p_hybrid,
                        a=a,
                        problem=self.problem,
                        rng=w_rng,
                        mutation_prob=self.config.mutation_prob,
                    )

                    # Lines 12-14: Check feasibility under Section 3
                    if not is_solution_feasible(new_sol, self.problem):
                        return w

                    # Lines 15-27: SA Acceptance (strictly nf < cf according to Algorithm 1 line 17)
                    nf = new_sol.cost
                    cf = w.cost
                    if nf < cf:
                        return new_sol
                    else:
                        denom = 1e-6 + temperature * abs(cf)
                        prob = math.exp(-(nf - cf) / denom) if denom > 0 else 0.0
                        if w_rng.random() < prob:
                            return new_sol
                        else:
                            return w

                if pool is not None:
                    new_population = list(pool.map(update_individual, range(len(population))))
                else:
                    new_population = [update_individual(i) for i in range(len(population))]

                population = new_population

                # Update global best from returned population
                for cand in population:
                    if cand < best:
                        best = cand.copy()

                # Lines 34-37: Periodic local search on global best
                if iteration % self.config.local_search_interval == 0:
                    best = combined_local_search(best, self.problem)

                # Lines 38-48: Diversification
                if best.cost < best_so_far - self.config.epsilon:
                    best_so_far = best.cost
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.config.stagnation_limit:
                        population = self._diversify(population, iteration, rng)
                        for cand in population:
                            if cand < best:
                                best = cand.copy()
                        best_so_far = best.cost
                        no_improve = 0
        finally:
            if pool is not None:
                pool.shutdown(wait=False)

        # Lines 49-50: End of loop, return best directly (no post-loop local search)
        elapsed = time.time() - start_time

        # Line 50: Feasibility verification (Section 3 mathematical constraints)
        is_feasible, violation_note = check_feasibility_canonical(best, self.problem)

        return RunResult(
            instance=self.problem.name,
            algo="PH-SHOWOA-Canonical",
            run=run_index,
            seed=self.config.seed,
            nv=best.nv,
            td=best.td,
            cost=2000.0 * best.nv + best.td,
            feasible=is_feasible,
            violation_note=violation_note,
            best_solution=best,
            elapsed_seconds=elapsed,
        )

    def _diversify(self, population: List[Solution], iteration: int, rng: random.Random) -> List[Solution]:
        """DIVERSIFY(population), Algorithm 1 line 45.

        The paper does not define DIVERSIFY (page 10 only says it perturbs the population).
        Priority Level 4 selection: apply 5 mutations (Algorithm 7) to the worst 20% individuals.
        """
        result = [s.copy() for s in population]
        indices = list(range(len(result)))
        indices.sort(key=lambda i: result[i].cost, reverse=True)

        count = max(1, int(math.ceil(len(result) * 0.20)))
        for k in range(count):
            idx = indices[k]
            div_rng = random.Random(self.config.seed + iteration * 5000 + idx)
            for _ in range(5):
                result[idx] = mutate_solution(result[idx], self.problem, div_rng)

        return result
