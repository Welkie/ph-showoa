# Engineering Changelog: PH-SHOWOA Refactor

## Objective Mapping

- Replaced the former memetic generation loop with a PH-SHOWOA loop in `src/search_framework.py`.
- Changed solution fitness to the lexicographic surrogate from the paper:
  `fitness = vehicles * 2000.0 + total_distance * 1.0`.
- Kept the existing route, parser, insertion, feasibility, and local-search data structures intact.

## SHO Exploration

- Each agent receives a per-iteration hybrid probability `p_hybrid`.
- When selected for SHO exploration, the worker chooses a tournament peer (`k=3`) and builds an offspring from global-best, peer, and current-agent routes.
- A 35% light mutation applies swap or relocate neighborhoods before feasibility repair.

## WOA Intensification

- The loop computes `a = 2 - 2 * t / MAX_ITER`, plus WOA vectors `A` and `C` in each worker.
- If `|A| < 1`, the discrete encircling and spiral phases copy elite routes or contiguous route segments from the global best.
- If `|A| >= 1`, the worker uses Li-and-Lim-style customer-sequence neighborhoods: swap, insert, and reverse on a flattened customer permutation, then splits it back into feasible routes.

## Acceptance, Local Search, and Diversification

- Worse offspring are admitted through the simulated annealing probability from the paper.
- Every 25 iterations, only the global best receives deep local search in this order: `2-opt`, `Or-opt`, `Two-Exchange`.
- Every 50 stagnant iterations, 40% of the non-elite population is diversified using random ruin-and-recreate.

## Parallelization

- Agent updates are dispatched through `concurrent.futures.ProcessPoolExecutor` by default.
- `--workers 1` forces serial execution for debugging; omitting `--workers` uses all available CPU cores.
- `--max_iter` controls the PH-SHOWOA iteration budget while the legacy `--g_1` remains accepted and acts as the default iteration count.
