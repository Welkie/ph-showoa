"""Run ablation configurations on the local CPU (same code as the Kaggle runs, --compute_backend cpu).

usage: python kaggle/cpu_ablation.py <name> '<json SolverConfig fields>' [instances...]
One process compiles the kernels once and runs 30 runs (seeds 42..71) for each instance; results go to
kaggle/cpu_ablation_results/<name>__<inst>.csv (+ __conv.csv), same columns as kaggle/ablation_driver.py.
Module constants of improvement2_sa_rcrs_grasp can be set with keys rc_weight, rc_threshold, rs_weight, seed_rule.
"""
import csv
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
name, cfg_json = sys.argv[1], json.loads(sys.argv[2])
insts = sys.argv[3:] or ["cdp101", "cdp103", "cdp203", "cdp207", "rdp103", "rdp107", "rdp201", "rdp210",
                         "rcdp101", "rcdp104", "rcdp202", "rcdp207"]

from src_python_gpu_SA_RCRS_GRASP import improvement2_sa_rcrs_grasp as i2  # noqa: E402
for key, attr in (("rc_weight", "RC_WEIGHT"), ("rc_threshold", "RC_THRESHOLD"), ("rs_weight", "RS_WEIGHT"),
                  ("seed_rule", "SEED_RULE")):
    if key in cfg_json:
        setattr(i2, attr, cfg_json.pop(key))
from src_python_gpu_SA_RCRS_GRASP import algorithm1_solver as a1  # noqa: E402
from src_python_gpu_SA_RCRS_GRASP.problem import Problem  # noqa: E402

_bundle = {}
_orig = a1.build_kernel_bundle
a1.build_kernel_bundle = lambda is_cuda: _bundle.setdefault(is_cuda, None) or _bundle.__setitem__(
    is_cuda, _orig(is_cuda)) or _bundle[is_cuda]

RES = os.path.join(ROOT, "kaggle", "cpu_ablation_results")
os.makedirs(RES, exist_ok=True)
for inst in insts:
    out = os.path.join(RES, f"{name}__{inst}.csv")
    if os.path.exists(out):
        continue
    problem = Problem.load(os.path.join(ROOT, "dataset", inst + ".txt"))
    cfg = a1.SolverConfig(compute_backend="cpu", **cfg_json)
    solver = a1.PHSHOWOAGpuSolver(problem, cfg)
    rows, conv = [], []
    t0 = time.perf_counter()
    for r in range(1, 31):
        cfg.seed = 42 + r - 1
        res = solver.solve(run_index=r)
        rows.append([res.instance, res.algo, r, cfg.seed, res.nv, f"{res.td:.2f}", f"{res.cost:.2f}",
                     res.feasible, res.violation_note, f"{res.elapsed_seconds:.2f}"])
        conv += [[res.instance, r, cfg.seed, g, nv, f"{td:.4f}", f"{2000.0 * nv + td:.4f}"]
                 for g, nv, td in res.convergence if g % 10 == 0 or g == 1]
    with open(out + ".part", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "algo", "run", "seed", "NV", "TD", "cost", "feasible", "violation_note",
                    "elapsed_seconds"])
        w.writerows(rows)
    with open(os.path.join(RES, f"{name}__{inst}__conv.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "run", "seed", "iteration", "NV", "TD", "cost"])
        w.writerows(conv)
    os.replace(out + ".part", out)
    print(f"done {name} {inst} {time.perf_counter() - t0:.0f}s", flush=True)
print(f"FINISHED {name}", flush=True)
