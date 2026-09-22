"""Device-side control for a fixed-length CUDA search graph.

The search operators are the existing kernels compiled as device functions;
these wrappers only move scheduling, parameters, RNG setup and logging to GPU.
"""

import math

import numpy as np
from numba import cuda


def build_graph_kernels(kernels):
    update = cuda.jit(device=True)(kernels["update_population"].py_func)
    eliminate = cuda.jit(device=True)(kernels["route_elimination"].py_func)
    search = cuda.jit(device=True)(kernels["local_search"].py_func)
    diversify = cuda.jit(device=True)(kernels["stagnation_diversify"].py_func)
    migrate = cuda.jit(device=True)(kernels["island_migration"].py_func)

    @cuda.jit
    def reset(rng, seed, generation, parameters, ibest_nr, gbest_nr):
        s = cuda.grid(1)
        if s < rng.shape[0]:
            # Identical four uint32 words to the original host _init_rng.
            value = seed[0]
            rng[s, 0] = np.uint32((value + s * 1337) & 0xFFFFFFFF) | np.uint32(1)
            rng[s, 1] = np.uint32((value + s * 2749 + 362436069) & 0xFFFFFFFF) | np.uint32(1)
            rng[s, 2] = np.uint32((value + s * 5171 + 521288629) & 0xFFFFFFFF) | np.uint32(1)
            rng[s, 3] = np.uint32((value + s * 7919 + 88675123) & 0xFFFFFFFF) | np.uint32(1)
        if s < ibest_nr.size:
            ibest_nr[s] = 0
        if s == 0:
            gbest_nr[0] = 0
            generation[0] = 1
            parameters[0] = 0.0
            parameters[1] = 0.0

    @cuda.jit
    def parameters_kernel(generation, parameters, max_iter, mode):
        if cuda.grid(1) == 0:
            iteration = generation[0] - 1
            a = 2.0 - 2.0 * (float(iteration) / float(max_iter))
            p = 0.5 * (1.0 + math.cos(math.pi * float(iteration) / float(max_iter)))
            if mode == 1:
                p = 1.0
            elif mode == 2:
                p = 0.0
            parameters[0] = a
            parameters[1] = p

    @cuda.jit
    def update_kernel(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                      next_nodes, next_rlen, next_nr, next_dist, next_cost,
                      cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                      ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                      scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                      prob_data, parameters, generation, max_iter, rng_states, island_size):
        update(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
               next_nodes, next_rlen, next_nr, next_dist, next_cost,
               cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
               ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
               scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
               prob_data, parameters[0], parameters[1], generation[0] - 1,
               max_iter, rng_states, island_size)

    @cuda.jit
    def elimination_kernel(nodes, rlen, nr, dist, cost, scratch_route,
                           scratch_unrouted, scratch_flags, prob_data,
                           generation, interval):
        if generation[0] % interval == 0:
            eliminate(nodes, rlen, nr, dist, cost, scratch_route,
                      scratch_unrouted, scratch_flags, prob_data, 5)

    @cuda.jit
    def search_kernel(nodes, rlen, nr, dist, cost, scratch_route,
                      scratch_route2, prob_data, generation, interval):
        if generation[0] % interval == 0:
            search(nodes, rlen, nr, dist, cost, scratch_route, scratch_route2,
                   prob_data, 2)

    @cuda.jit
    def diversify_kernel(nodes, rlen, nr, dist, cost, scratch_route,
                         scratch_unrouted, scratch_flags, prob_data, rng_states,
                         num_islands, island_size, generation, interval):
        if generation[0] % interval == 0:
            diversify(nodes, rlen, nr, dist, cost, scratch_route,
                      scratch_unrouted, scratch_flags, prob_data, rng_states,
                      num_islands, island_size)

    @cuda.jit
    def migration_kernel(nodes, rlen, nr, dist, cost,
                         ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                         num_islands, island_size, generation, interval):
        if generation[0] % interval == 0:
            migrate(nodes, rlen, nr, dist, cost,
                    ibest_nodes, ibest_rlen, ibest_nr, ibest_dist, ibest_cost,
                    num_islands, island_size)

    @cuda.jit
    def finish_generation(generation, parameters, gbest_nr, gbest_dist, history):
        if cuda.grid(1) == 0:
            row = generation[0] - 1
            history[row, 0] = parameters[0]
            history[row, 1] = parameters[1]
            history[row, 2] = gbest_nr[0]
            history[row, 3] = gbest_dist[0]
            generation[0] += 1

    return {
        "reset": reset,
        "parameters": parameters_kernel,
        "update": update_kernel,
        "eliminate": elimination_kernel,
        "search": search_kernel,
        "diversify": diversify_kernel,
        "migrate": migration_kernel,
        "finish": finish_generation,
    }
