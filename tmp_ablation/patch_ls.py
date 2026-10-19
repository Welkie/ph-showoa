# (A) periodic LS only on the best individual
p='src_ls_best/gpu_kernels.py'
s=open(p,encoding='utf-8',newline='').read().replace('\r\n','\n')
old_cuda='''        def local_search_kernel(nodes, rlen, nr, dist, cost,
                                scratch_route, scratch_route2, prob_data, max_passes):
            s = cuda.grid(1)
            if s < nodes.shape[0]:
                deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_route2, prob_data, max_passes)
'''
new_cuda='''        def local_search_kernel(nodes, rlen, nr, dist, cost,
                                scratch_route, scratch_route2, prob_data, max_passes):
            # BASELINE: periodic LS refines only the best individual of the population
            if cuda.grid(1) == 0:
                b = 0
                for s in range(1, nodes.shape[0]):
                    if is_better_lex(nr[s], dist[s], nr[b], dist[b]):
                        b = s
                deep_local_search_single(nodes, rlen, nr, dist, cost, b,
                                         scratch_route, scratch_route2, prob_data, max_passes)
'''
old_cpu='''        def local_search_kernel(nodes, rlen, nr, dist, cost,
                                 scratch_route, scratch_route2, prob_data, max_passes):
            for s in range(nodes.shape[0]):
                deep_local_search_single(nodes, rlen, nr, dist, cost, s,
                                         scratch_route, scratch_route2, prob_data, max_passes)
'''
new_cpu='''        def local_search_kernel(nodes, rlen, nr, dist, cost,
                                 scratch_route, scratch_route2, prob_data, max_passes):
            # BASELINE: periodic LS refines only the best individual of the population
            b = 0
            for s in range(1, nodes.shape[0]):
                if is_better_lex(nr[s], dist[s], nr[b], dist[b]):
                    b = s
            deep_local_search_single(nodes, rlen, nr, dist, cost, b,
                                     scratch_route, scratch_route2, prob_data, max_passes)
'''
assert s.count(old_cuda)==1 and s.count(old_cpu)==1
s=s.replace(old_cuda,new_cuda).replace(old_cpu,new_cpu)
open(p,'w',encoding='utf-8',newline='').write(s)

# (C) remove Inter-route 2-opt* from Deep LS (all-population LS kept)
p='src_ls_no2opt/gpu_kernels.py'
s=open(p,encoding='utf-8',newline='').read().replace('\r\n','\n')
a=s.index("            # 3. Inter-route 2-opt* (swap tails of two routes)\n")
b=s.index("            # 4. Inter-route Relocate (move single customer between routes)\n")
blk=s[a:b]
head="            # 3. Inter-route 2-opt* (swap tails of two routes)\n            L = scratch_route.shape[1]\n            dist_matrix = prob_data[10]\n"
assert blk.startswith(head)
rest=blk[len(head):]
rest="".join(("    "+l if l.strip() else l) for l in rest.splitlines(True))
s=s[:a]+head+"            if False:  # ABLATION: Inter-route 2-opt* disabled\n"+rest+s[b:]
open(p,'w',encoding='utf-8',newline='').write(s)
print("patched")
