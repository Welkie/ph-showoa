"""Run one ablation job of docs/improved.md (30 runs, seeds 42..71) in a single process on the CUDA backend.

usage: python ablation_driver.py <dir containing the package copy> <job.json> <results dir> <dataset dir>

The kernels are compiled once per process (first run) and reused by the other 29 runs; elapsed_seconds of run 1
therefore includes the JIT compile. Writes <name>__<inst>.csv (same columns as the package CLI) and
<name>__<inst>__conv.csv (best NV/TD of the population every 10 iterations).
"""

import csv
import json
import os
import sys
import time

pkg_parent, job_file, res_dir, data_dir = sys.argv[1:5]
sys.path.insert(0, pkg_parent)
from src_python_gpu_SA_RCRS_GRASP import algorithm1_solver as a1  # noqa: E402
from src_python_gpu_SA_RCRS_GRASP.problem import Problem  # noqa: E402

job = json.load(open(job_file, encoding="utf-8"))
backend = job.get("backend", "cuda")
os.makedirs(res_dir, exist_ok=True)
out = os.path.join(res_dir, f"{job['name']}__{job['inst']}.csv")
problem = Problem.load(os.path.join(data_dir, job["inst"] + ".txt"))
cfg = a1.SolverConfig(compute_backend=backend, **job["cfg"])
solver = a1.PHSHOWOAGpuSolver(problem, cfg)
print(f"{job['name']} {job['inst']}: backend={'cuda' if solver.is_cuda else 'cpu'} islands={solver.num_islands}",
      flush=True)
rows, conv = [], []
t0 = time.perf_counter()
for r in range(1, job.get("runs", 30) + 1):
    cfg.seed = 42 + r - 1
    res = solver.solve(run_index=r)
    rows.append([res.instance, res.algo, r, cfg.seed, res.nv, f"{res.td:.2f}", f"{res.cost:.2f}",
                 res.feasible, res.violation_note, f"{res.elapsed_seconds:.2f}"])
    conv += [[res.instance, r, cfg.seed, g, nv, f"{td:.4f}", f"{2000.0 * nv + td:.4f}"]
             for g, nv, td in res.convergence if g % 10 == 0 or g == 1]
    print(f"  run {r:02d} seed={cfg.seed} NV={res.nv} TD={res.td:.2f} feasible={res.feasible} "
          f"({res.elapsed_seconds:.1f}s)", flush=True)
with open(out + ".part", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["instance", "algo", "run", "seed", "NV", "TD", "cost", "feasible", "violation_note",
                "elapsed_seconds"])
    w.writerows(rows)
with open(os.path.join(res_dir, f"{job['name']}__{job['inst']}__conv.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["instance", "run", "seed", "iteration", "NV", "TD", "cost"])
    w.writerows(conv)
os.replace(out + ".part", out)
print(f"DONE {job['name']} {job['inst']} {time.perf_counter() - t0:.0f}s", flush=True)
