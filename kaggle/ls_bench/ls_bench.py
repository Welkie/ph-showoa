"""Benchmark one local search on CPU vs GPU: current first-improvement LS vs the parallel best-improvement prototype.

usage: python ls_bench.py <dir containing src_python_gpu_SA_RCRS_GRASP> <inputs dir> <dataset dir> inst [inst ...]
For every instance the 30 saved solutions (population right before the LS of generation 25) are improved by:
  FI-CPU  current deep_local_search_single (Numba CPU, 1 solution at a time)
  BI-CPU  prototype best improvement, same scan with 1 thread (Numba CPU)
  FI-GPU  current local_search kernel (1 thread per solution); single solution, then all 30 at once
  BI-GPU  prototype, one block of 128 threads per solution; single solution, then all 30 at once
Times exclude JIT compilation (a warm-up call on a copy comes first) and host<->device copies.
"""

import json
import os
import statistics as st
import sys
import time

import numpy as np

pkg_parent, in_dir, data_dir = sys.argv[1:4]
insts = sys.argv[4:]
sys.path.insert(0, pkg_parent)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src_python_gpu_SA_RCRS_GRASP.improvement1_full_gpu import build_kernel_bundle, prepare_problem_data  # noqa
from src_python_gpu_SA_RCRS_GRASP.problem import Problem  # noqa
from src_python_gpu_SA_RCRS_GRASP.solution import Solution  # noqa
from src_python_gpu_SA_RCRS_GRASP.algorithm9_feasibility import check_feasibility_canonical  # noqa
import bi_ls  # noqa

try:
    from numba import cuda
    has_gpu = cuda.is_available()
except Exception:  # noqa: BLE001
    has_gpu = False
print("GPU:", cuda.get_current_device().name if has_gpu else "none", flush=True)

fi_cpu = build_kernel_bundle(False)["local_search"]
bi_cpu = bi_ls.build(False)
if has_gpu:
    fi_gpu = build_kernel_bundle(True)["local_search"]
    bi_gpu = bi_ls.build(True)


def evaluate(nodes, rlen, nr, problem):
    out = []
    for s in range(nodes.shape[0]):
        routes = [[int(nodes[s, r, i]) for i in range(1, int(rlen[s, r]) - 1)] for r in range(int(nr[s]))
                  if rlen[s, r] > 2]
        sol = Solution(routes, problem)
        ok, _ = check_feasibility_canonical(sol, problem)
        out.append((sol.nv, sol.td, ok))
    return out


def host_arrays(d, sel):
    return (d["nodes"][sel].copy(), d["rlen"][sel].copy(), d["nr"][sel].copy(),
            d["dist"][sel].copy(), d["cost"][sel].copy())


summary = {}
for inst in insts:
    problem = Problem.load(os.path.join(data_dir, inst + ".txt"))
    prob_h = prepare_problem_data(problem)
    d = dict(np.load(os.path.join(in_dir, inst + ".npz")))
    P, R, L = d["nodes"].shape
    SMAX = problem.num_customers + 2 * R
    res = {}

    def run_fi_cpu(sel):
        nodes, rlen, nr, dist, cost = host_arrays(d, sel)
        n = nodes.shape[0]
        t = time.perf_counter()
        fi_cpu(nodes, rlen, nr, dist, cost, np.zeros((n, L), np.int32), np.zeros((n, L), np.int32), prob_h, 1000000)
        return time.perf_counter() - t, nodes, rlen, nr, None

    def run_bi_cpu(sel):
        nodes, rlen, nr, _, _ = host_arrays(d, sel)
        n = nodes.shape[0]
        mv = np.zeros(n, np.int64)
        t = time.perf_counter()
        bi_cpu(nodes, rlen, nr, np.zeros((n, SMAX), np.int32), np.zeros((n, SMAX), np.int32),
               np.zeros((n, L), np.int32), np.zeros((n, L), np.int32), prob_h, 1000000, mv)
        return time.perf_counter() - t, nodes, rlen, nr, mv

    variants = [("FI-CPU", run_fi_cpu), ("BI-CPU", run_bi_cpu)]
    if has_gpu:
        prob_d = tuple(cuda.to_device(x) if isinstance(x, np.ndarray) else x for x in prob_h)

        def run_fi_gpu(sel):
            arrs = [cuda.to_device(a) for a in host_arrays(d, sel)]
            n = arrs[0].shape[0]
            sr, sr2 = cuda.to_device(np.zeros((n, L), np.int32)), cuda.to_device(np.zeros((n, L), np.int32))
            cuda.synchronize()
            t = time.perf_counter()
            fi_gpu[(n + 31) // 32, 32](*arrs, sr, sr2, prob_d, 1000000)
            cuda.synchronize()
            dt = time.perf_counter() - t
            return dt, arrs[0].copy_to_host(), arrs[1].copy_to_host(), arrs[2].copy_to_host(), None

        def run_bi_gpu(sel):
            nodes, rlen, nr, _, _ = host_arrays(d, sel)
            n = nodes.shape[0]
            dn, dr, dnr = cuda.to_device(nodes), cuda.to_device(rlen), cuda.to_device(nr)
            sl1, sl2 = cuda.to_device(np.zeros((n, SMAX), np.int32)), cuda.to_device(np.zeros((n, SMAX), np.int32))
            s1, s2 = cuda.to_device(np.zeros((n, L), np.int32)), cuda.to_device(np.zeros((n, L), np.int32))
            mv = cuda.to_device(np.zeros(n, np.int64))
            cuda.synchronize()
            t = time.perf_counter()
            bi_gpu[n, bi_ls.NT](dn, dr, dnr, sl1, sl2, s1, s2, prob_d, 1000000, mv)
            cuda.synchronize()
            dt = time.perf_counter() - t
            return dt, dn.copy_to_host(), dr.copy_to_host(), dnr.copy_to_host(), mv.copy_to_host()

        variants += [("FI-GPU", run_fi_gpu), ("BI-GPU", run_bi_gpu)]

    before = evaluate(d["nodes"], d["rlen"], d["nr"], problem)
    for name, fn in variants:
        fn(slice(0, 1))                                          # warm-up (JIT compile)
        single, outs, moves = [], [], []
        for s in range(P):
            dt, nodes, rlen, nr, mv = fn(slice(s, s + 1))
            single.append(dt)
            outs += evaluate(nodes, rlen, nr, problem)
            if mv is not None:
                moves.append(int(mv[0]))
        r = {"single_median_s": st.median(single), "single_mean_s": st.mean(single), "single_max_s": max(single),
             "single_sum_s": sum(single),
             "mean_cost_after": st.mean(2000 * nv + td for nv, td, _ in outs),
             "infeasible": sum(1 for *_, ok in outs if not ok),
             "mean_moves": st.mean(moves) if moves else None}
        if name.endswith("GPU"):
            dt, nodes, rlen, nr, _ = fn(slice(0, P))
            r["population_s"] = dt
            pop_out = evaluate(nodes, rlen, nr, problem)
            r["population_same_as_single"] = [o[:2] for o in pop_out] == [o[:2] for o in outs]
        res[name] = r
        print(f"{inst:8s} {name:6s} single median {r['single_median_s']*1000:9.1f} ms  mean {r['single_mean_s']*1000:9.1f} ms"
              f"  max {r['single_max_s']*1000:9.1f} ms  sum30 {r['single_sum_s']:7.2f} s"
              + (f"  pop30 {r['population_s']:7.2f} s" if "population_s" in r else "")
              + f"  cost {r['mean_cost_after']:.1f}  infeasible {r['infeasible']}"
              + (f"  moves {r['mean_moves']:.0f}" if r["mean_moves"] is not None else ""), flush=True)
    res["cost_before"] = st.mean(2000 * nv + td for nv, td, _ in before)
    print(f"{inst:8s} cost before LS {res['cost_before']:.1f}", flush=True)
    summary[inst] = res

with open(os.path.join(os.environ.get("BENCH_OUT", "."), "ls_bench.json"), "w") as f:
    json.dump(summary, f, indent=1)
