import re
p='src_sa_baseline/gpu_kernels.py'
t=open(p,encoding='utf-8').read()
t=t.replace("prob_data, rng_states, sa_iters=25):\n        temp = 100.0\n        cooling = 0.85\n        num_r = nr[s]","prob_data, rng_states, sa_iters, temp, cooling, tmin):\n        num_r = nr[s]",1)
t=t.replace("while temp > 0.5:\n            for _ in range(sa_iters):","while temp > tmin:\n            for _ in range(sa_iters):",1)
singleton='''    @dev_fn
    def construct_singletons(nodes, rlen, nr, dist, cost, s, scratch_unrouted, prob_data, rng_states):
        customer_num = int(prob_data[14])
        depot = int(prob_data[0])
        dispatch_cost = prob_data[3]
        unit_cost = prob_data[4]
        for i in range(customer_num):
            scratch_unrouted[s, i] = i + 1
        shuffle_ints(scratch_unrouted[s], customer_num, rng_states, s)
        tot = 0.0
        c_cost = 0.0
        for i in range(customer_num):
            c = scratch_unrouted[s, i]
            nodes[s, i, 0] = depot
            nodes[s, i, 1] = c
            nodes[s, i, 2] = depot
            rlen[s, i] = 3
            ok, d = eval_route(nodes[s, i, :3], 3, prob_data)
            tot += d
            c_cost += dispatch_cost + d * unit_cost
        nr[s] = customer_num
        dist[s] = tot
        cost[s] = c_cost

'''
marker="    # -------------------------------------------------------------------------\n    # 5. Simulated Annealing Warmup"
assert marker in t
t=t.replace(marker,singleton+marker,1)
pat=re.compile(r"( +)construct_rcrs_grasp\(nodes, rlen, nr, dist, cost, s,\n +scratch_route, scratch_unrouted, scratch_flags, scratch_scores,\n +prob_data, alpha, rng_states\)\n +sa_warmup_single\(nodes, rlen, nr, dist, cost, s,\n +scratch_route, scratch_route2,\n +prob_data, rng_states, sa_iters=25\)\n +route_elimination_single\(.*?\n.*?\n +deep_local_search_single\(.*?\n.*?\n.*?max_passes=2\)\n",re.S)
def rep(m):
    i=m.group(1)
    return (f"{i}construct_singletons(nodes, rlen, nr, dist, cost, s, scratch_unrouted, prob_data, rng_states)\n"
            f"{i}sa_warmup_single(nodes, rlen, nr, dist, cost, s,\n{i}                 scratch_route, scratch_route2,\n{i}                 prob_data, rng_states, 100, 100.0, 0.95, 0.1)\n")
t,n=pat.subn(rep,t)
print("replaced",n)
open(p,'w',encoding='utf-8').write(t)
