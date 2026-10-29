"""Improvement 3 (docs/improved.md section 3): island model with ring migration.

The population is split into islands of equal size. Tournament selection and the guide solution of SHO / WOA
(the island best instead of the global best) stay inside an island; every migration_interval iterations the best of
island i replaces the worst individual of island i + 1 when it is better.
"""

from __future__ import annotations

try:
    from numba import cuda
except ImportError:  # pragma: no cover - optional dependency
    cuda = None


def island_layout(pop_size: int, requested: int) -> int:
    """Islands used for a population: the requested count when it divides pop_size, otherwise the divisor of
    pop_size closest to it (>= 2 islands of >= 3 individuals; ties go to the smaller count), else 1 island."""
    requested = max(1, int(requested))
    if pop_size % requested == 0:
        return requested
    divisors = [d for d in range(2, pop_size // 3 + 1) if pop_size % d == 0]
    if divisors:
        return min(divisors, key=lambda d: (abs(d - requested), d))
    return 1


def build_kernels(is_cuda, k_fn, k, kernels):
    is_better_lex = k["is_better_lex"]
    copy_solution = k["copy_solution"]

    @k["dev_fn"]
    def island_best(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                    ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl, island_size):
        best_s = isl * island_size
        for i in range(1, island_size):
            cand_s = isl * island_size + i
            if is_better_lex(pop_nr[cand_s], pop_dist[cand_s], pop_nr[best_s], pop_dist[best_s]):
                best_s = cand_s
        if ibest_nr[isl] == 0 or is_better_lex(pop_nr[best_s], pop_dist[best_s], ibest_nr[isl], ibest_dist[isl]):
            copy_solution(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost, best_s,
                          ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl)

    @k["dev_fn"]
    def global_best(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                    gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost, num_islands):
        best_isl = 0
        for isl in range(1, num_islands):
            if is_better_lex(ibest_nr[isl], ibest_dist[isl], ibest_nr[best_isl], ibest_dist[best_isl]):
                best_isl = isl
        if gbest_nr[0] == 0 or is_better_lex(ibest_nr[best_isl], ibest_dist[best_isl], gbest_nr[0], gbest_dist[0]):
            copy_solution(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, best_isl,
                          gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost, 0)

    @k["dev_fn"]
    def ring_migration(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                       ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, num_islands, island_size):
        for isl in range(num_islands):
            dst = (isl + 1) % num_islands
            worst_s = dst * island_size
            for i in range(1, island_size):
                cand_s = dst * island_size + i
                if is_better_lex(pop_nr[worst_s], pop_dist[worst_s], pop_nr[cand_s], pop_dist[cand_s]):
                    worst_s = cand_s
            if is_better_lex(ibest_nr[isl], ibest_dist[isl], pop_nr[worst_s], pop_dist[worst_s]):
                copy_solution(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl,
                              pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost, worst_s)

    if is_cuda:
        @k_fn
        def update_island_bests_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                       ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                       num_islands, island_size):
            isl = cuda.grid(1)
            if isl < num_islands:
                island_best(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                            ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl, island_size)

        @k_fn
        def update_global_best_kernel(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                      gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost, num_islands):
            if cuda.grid(1) == 0:
                global_best(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                            gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost, num_islands)

        @k_fn
        def island_migration_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                    ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                    num_islands, island_size):
            if cuda.grid(1) == 0:
                ring_migration(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                               ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, num_islands, island_size)
    else:
        @k_fn
        def update_island_bests_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                       ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                       num_islands, island_size):
            for isl in range(num_islands):
                island_best(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                            ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, isl, island_size)

        @k_fn
        def update_global_best_kernel(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                      gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost, num_islands):
            global_best(ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                        gbest_nodes, gbest_rlen, gbest_nr, gbest_dist, gbest_cost, num_islands)

        @k_fn
        def island_migration_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                    ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                                    num_islands, island_size):
            ring_migration(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                           ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost, num_islands, island_size)

    kernels.update({
        "update_island_bests": update_island_bests_kernel,
        "update_global_best": update_global_best_kernel,
        "island_migration": island_migration_kernel,
    })
