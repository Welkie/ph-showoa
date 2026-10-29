"""Initialization tuning on the CPU: quality and diversity of the population when the evolution starts.

usage: python kaggle/init_tuning.py <variant name> '<json settings>' <out.csv>
settings: init (sa_rcrs_grasp | sa | mixed), rc_weight, rc_threshold, rs_weight, seed_rule (0 random, 1 earliest due
time, 2 farthest), re_passes, alpha_lo, alpha_hi. Unset keys keep the code defaults.

For each of the 12 ablation instances and seeds 42, 43, 44 the solver runs one generation (initialization, then
iteration 0 with route elimination and local search) and the population is measured: mean / min NV, mean / best cost
and diversity (mean pairwise Jaccard distance between the undirected edge sets of the 30 individuals).
"""
import csv
import itertools
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
name, settings, out = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]

from src_python_gpu_SA_RCRS_GRASP import improvement2_sa_rcrs_grasp as i2  # noqa: E402
for key, attr in (("rc_weight", "RC_WEIGHT"), ("rc_threshold", "RC_THRESHOLD"), ("rs_weight", "RS_WEIGHT"),
                  ("seed_rule", "SEED_RULE")):
    if key in settings:
        setattr(i2, attr, settings[key])
from src_python_gpu_SA_RCRS_GRASP import algorithm1_solver as a1  # noqa: E402
from src_python_gpu_SA_RCRS_GRASP.problem import Problem  # noqa: E402

_bundle = {}
_orig = a1.build_kernel_bundle
a1.build_kernel_bundle = lambda is_cuda: _bundle.setdefault(is_cuda, None) or _bundle.__setitem__(
    is_cuda, _orig(is_cuda)) or _bundle[is_cuda]

INSTS = ["cdp101", "cdp103", "cdp203", "cdp207", "rdp103", "rdp107", "rdp201", "rdp210",
         "rcdp101", "rcdp104", "rcdp202", "rcdp207"]


def edges(nodes, rlen, nr, s):
    e = set()
    for r in range(int(nr[s])):
        l = int(rlen[s, r])
        for k in range(l - 1):
            a, b = int(nodes[s, r, k]), int(nodes[s, r, k + 1])
            e.add((a, b) if a < b else (b, a))
    return e


rows = []
for inst in INSTS:
    problem = Problem.load(os.path.join(ROOT, "dataset", inst + ".txt"))
    cfg = a1.SolverConfig(compute_backend="cpu", max_iter=1, init=settings.get("init", "sa_rcrs_grasp"))
    for key, attr in (("re_passes", "route_elimination_passes"), ("alpha_lo", "grasp_alpha_lo"),
                      ("alpha_hi", "grasp_alpha_hi")):
        if key in settings:
            setattr(cfg, attr, settings[key])
    solver = a1.PHSHOWOAGpuSolver(problem, cfg)
    for seed in (42, 43, 44):
        cfg.seed = seed
        t0 = time.perf_counter()
        solver.solve(1)
        dt = time.perf_counter() - t0
        nodes, rlen, nr, dist, cost = solver.buf["next"]      # 1 generation: the current population is "next"
        P = nodes.shape[0]
        es = [edges(nodes, rlen, nr, s) for s in range(P)]
        div = sum(1 - len(es[a] & es[b]) / len(es[a] | es[b]) for a, b in itertools.combinations(range(P), 2))
        div /= P * (P - 1) / 2
        rows.append({"variant": name, "instance": inst, "seed": seed,
                     "nv_mean": float(nr[:P].mean()), "nv_min": int(nr[:P].min()),
                     "cost_mean": float(cost[:P].mean()), "cost_best": float(cost[:P].min()),
                     "diversity": round(div, 4), "seconds": round(dt, 2)})
        print(name, inst, seed, rows[-1]["nv_min"], round(rows[-1]["cost_best"], 1), rows[-1]["diversity"], flush=True)

with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
