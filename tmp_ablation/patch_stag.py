p='src_stag_baseline/gpu_kernels.py'
s=open(p,encoding='utf-8',newline='').read().replace('\r\n','\n')
# 1. extract per-individual diversify
a=s.index("    @dev_fn\n    def diversify_island_single(")
b=s.index("    # -------------------------------------------------------------------------\n    # 9. Top-Level Kernels")
fn=s[a:b]
marker="        for idx in range(diversify_count):\n            s = island_start + island_size - 1 - idx\n"
assert marker in fn
body=fn.split(marker,1)[1]
body="\n".join(l[4:] if l.startswith("    ") else l for l in body.split("\n"))
indiv=('''    @dev_fn
    def diversify_individual(nodes, rlen, nr, dist, cost, s,
                             scratch_route, scratch_unrouted, scratch_flags,
                             prob_data, rng_states):
'''+body)
s=s[:b]+indiv+s[b:]
# 2. replace CPU stagnation kernels
c0=s.index("        @k_fn\n        def island_stagnation_update_kernel(ibest_nr, ibest_dist, stag_state, num_islands):\n            for isl in range(num_islands):")
c1=s.index("        @k_fn\n        def init_device_rng_kernel(rng_states, seed):\n            for s in range(rng_states.shape[0]):")
new='''        @k_fn
        def island_stagnation_update_kernel(ibest_nr, ibest_dist, stag_state, num_islands):
            # BASELINE: one global stagnation counter on the best-of-population
            best_i = -1
            for isl in range(num_islands):
                if ibest_nr[isl] > 0:
                    if best_i < 0 or is_better_lex(ibest_nr[isl], ibest_dist[isl],
                                                   ibest_nr[best_i], ibest_dist[best_i]):
                        best_i = isl
            if best_i >= 0:
                if stag_state[0, 0] <= 0.0 or is_better_lex(ibest_nr[best_i], ibest_dist[best_i],
                                                            int(stag_state[0, 0]), stag_state[0, 1]):
                    stag_state[0, 0] = float(ibest_nr[best_i])
                    stag_state[0, 1] = ibest_dist[best_i]
                    stag_state[0, 2] = 0.0
                else:
                    stag_state[0, 2] += 1.0

        @k_fn
        def stagnation_diversify_kernel(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost,
                                        scratch_route, scratch_unrouted, scratch_flags,
                                        prob_data, rng_states, num_islands, island_size,
                                        stag_state, stag_interval):
            # BASELINE: if the whole population stagnates, ruin & recreate 40% random individuals
            if stag_state[0, 2] >= stag_interval:
                P = pop_nodes.shape[0]
                cnt = max(1, int(P * 0.40))
                idx = np.arange(P)
                for i in range(P - 1, 0, -1):
                    j = randint(rng_states, 0, 0, i)
                    tmp = idx[i]
                    idx[i] = idx[j]
                    idx[j] = tmp
                for k in range(cnt):
                    diversify_individual(pop_nodes, pop_rlen, pop_nr, pop_dist, pop_cost, idx[k],
                                         scratch_route, scratch_unrouted, scratch_flags,
                                         prob_data, rng_states)
                stag_state[0, 2] = 0.0

'''
s=s[:c0]+new+s[c1:]
open(p,'w',encoding='utf-8',newline='').write(s)
print("ok")
