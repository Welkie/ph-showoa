"""Full runs of the GPU package (30 runs, seeds 42..71) for docs/improved.md section 1.

usage: python full_run.py <root containing the package> <package> <dataset dir> <results dir> <name> <inst...>
Environment: CFG = JSON of SolverConfig fields, I2 / EP = JSON of improvement2_sa_rcrs_grasp /
improvement2_ejection_pool module constants, RUNS = number
of runs (default 30).
One process; the kernels are compiled once (bundle cache) and every instance gets one untimed warm-up solve, so
elapsed_seconds of all 30 runs excludes the JIT compile. Writes <name>__<inst>.csv (columns of the package CLI) and
<name>__<inst>__conv.csv (best NV / TD of the population every 10 iterations), as kaggle/ablation_driver.py.
"""
import csv
import importlib
import json
import os
import sys
import time

root, pkg, data_dir, res_dir, name = sys.argv[1:6]
sys.path.insert(0, root)
a1 = importlib.import_module(f"{pkg}.algorithm1_solver")
Problem = importlib.import_module(f"{pkg}.problem").Problem
cfg_extra = json.loads(os.environ.get("CFG", "{}"))
i2 = importlib.import_module(f"{pkg}.improvement2_sa_rcrs_grasp")
ep = importlib.import_module(f"{pkg}.improvement2_ejection_pool")
for env, mod in (("I2", i2), ("EP", ep)):                        # module constants, read at compile time
    for key, val in json.loads(os.environ.get(env, "{}")).items():
        assert hasattr(mod, key), key
        setattr(mod, key, val)

_bundle = {}
_orig = a1.build_kernel_bundle


def _cached(is_cuda):
    if is_cuda not in _bundle:
        _bundle[is_cuda] = _orig(is_cuda)
    return _bundle[is_cuda]


a1.build_kernel_bundle = _cached
os.makedirs(res_dir, exist_ok=True)
for inst in sys.argv[6:]:
    problem = Problem.load(os.path.join(data_dir, inst + ".txt"))
    cfg = a1.SolverConfig(compute_backend=os.environ.get("BACKEND", "cuda"), **cfg_extra)
    solver = a1.PHSHOWOAGpuSolver(problem, cfg)
    cfg.seed = 1
    solver.solve(0)                                  # warm-up (compilation, first-touch of the buffers)
    rows, conv = [], []
    t0 = time.perf_counter()
    for r in range(1, int(os.environ.get("RUNS", "30")) + 1):
        cfg.seed = 42 + r - 1
        res = solver.solve(run_index=r)
        rows.append([res.instance, res.algo, r, cfg.seed, res.nv, f"{res.td:.2f}", f"{res.cost:.2f}",
                     res.feasible, res.violation_note, f"{res.elapsed_seconds:.2f}"])
        conv += [[res.instance, r, cfg.seed, g, nv, f"{td:.4f}", f"{2000.0 * nv + td:.4f}"]
                 for g, nv, td in res.convergence if g % 10 == 0 or g == 1]
    out = os.path.join(res_dir, f"{name}__{inst}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "algo", "run", "seed", "NV", "TD", "cost", "feasible", "violation_note",
                    "elapsed_seconds"])
        w.writerows(rows)
    with open(os.path.join(res_dir, f"{name}__{inst}__conv.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "run", "seed", "iteration", "NV", "TD", "cost"])
        w.writerows(conv)
    print(f"DONE {name} {inst} {time.perf_counter() - t0:.0f}s", flush=True)
