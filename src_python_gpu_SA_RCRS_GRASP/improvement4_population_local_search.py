"""Improvement 4 (docs/improved.md section 4): local search for the whole population + Inter 2-opt*.

Algorithm 8 (algorithm8_localsearch.py) is applied to every individual every 25 iterations instead of only the
global best, and Inter 2-opt* (tail exchange between two routes) is added after 2-opt, Relocate and Swap.
"""

from __future__ import annotations

try:
    from numba import cuda
except ImportError:  # pragma: no cover - optional dependency
    cuda = None


def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""
    eval_route = k["eval_route"]

    @dev_fn
    def inter_2opt_star_first_improvement(nodes, rlen, nr, s, scratch_route, scratch_route2, prob_data, tol):
        # Inter 2-opt*: swap the tails of two routes; applies the first improving move and returns True
        num_r = nr[s]
        L = scratch_route.shape[1]
        dist_matrix = prob_data[10]
        found = False
        for r1 in range(num_r - 1):
            len1 = rlen[s, r1]
            if len1 < 4:
                continue
            for r2 in range(r1 + 1, num_r):
                len2 = rlen[s, r2]
                if len2 < 4:
                    continue
                for p1 in range(2, len1 - 1):
                    u1 = nodes[s, r1, p1 - 1]
                    v1 = nodes[s, r1, p1]
                    for p2 in range(2, len2 - 1):
                        u2 = nodes[s, r2, p2 - 1]
                        v2 = nodes[s, r2, p2]
                        new_len1 = p1 + len2 - p2
                        new_len2 = p2 + len1 - p1
                        if new_len1 >= L or new_len2 >= L:
                            continue
                        delta_d = (dist_matrix[u1, v2] + dist_matrix[u2, v1] -
                                   dist_matrix[u1, v1] - dist_matrix[u2, v2])
                        if delta_d >= -tol:
                            continue
                        for k in range(p1):
                            scratch_route[s, k] = nodes[s, r1, k]
                        for k in range(p2, len2):
                            scratch_route[s, p1 + (k - p2)] = nodes[s, r2, k]
                        ok1, d1 = eval_route(scratch_route[s, :new_len1], new_len1, prob_data)
                        if not ok1:
                            continue
                        for k in range(p2):
                            scratch_route2[s, k] = nodes[s, r2, k]
                        for k in range(p1, len1):
                            scratch_route2[s, p2 + (k - p1)] = nodes[s, r1, k]
                        ok2, d2 = eval_route(scratch_route2[s, :new_len2], new_len2, prob_data)
                        if not ok2:
                            continue
                        _, old_d1 = eval_route(nodes[s, r1, :len1], len1, prob_data)
                        _, old_d2 = eval_route(nodes[s, r2, :len2], len2, prob_data)
                        if (d1 + d2) < (old_d1 + old_d2) - tol:
                            for k in range(new_len1):
                                nodes[s, r1, k] = scratch_route[s, k]
                            for k in range(new_len1, len1):
                                nodes[s, r1, k] = 0
                            for k in range(new_len2):
                                nodes[s, r2, k] = scratch_route2[s, k]
                            for k in range(new_len2, len2):
                                nodes[s, r2, k] = 0
                            rlen[s, r1] = new_len1
                            rlen[s, r2] = new_len2
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        return found

    k.update({"inter_2opt_star_first_improvement": inter_2opt_star_first_improvement})


def build_kernels(is_cuda, k_fn, k, kernels):
    """Algorithm 8 (+ Inter 2-opt*) applied to every individual of the population."""
    deep_local_search_single = k["deep_local_search_single"]
    if is_cuda:
        @k_fn
        def local_search_kernel(nodes, rlen, nr, dist, cost, scratch_route, scratch_route2, prob_data, max_moves):
            s = cuda.grid(1)
            if s < nodes.shape[0]:
                deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_route2, prob_data, max_moves)
    else:
        @k_fn
        def local_search_kernel(nodes, rlen, nr, dist, cost, scratch_route, scratch_route2, prob_data, max_moves):
            for s in range(nodes.shape[0]):
                deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_route2, prob_data, max_moves)
    kernels["local_search"] = local_search_kernel
