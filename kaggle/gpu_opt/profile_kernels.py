"""Time every kernel of the GPU solver (improvement 1 profiling).

usage: python profile_kernels.py <root containing the package> <package name> <dataset dir> <gens> <runs> <inst...>
For each instance: one untimed solve (compilation), then <runs> solves with seeds 42.. measured two ways:
  wall    the plain solve (kernels queued asynchronously, as in production)
  kernels a solve where every launch is followed by cuda.synchronize(), giving the time of each kernel name
Prints one JSON line per instance: {"inst", "pkg", "wall", "cost", "kernels": {name: [seconds, launches]}}.
"""
import importlib
import json
import os
import sys
import time

root, pkg, data_dir, gens, runs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, root)
a1 = importlib.import_module(f"{pkg}.algorithm1_solver")
Problem = importlib.import_module(f"{pkg}.problem").Problem
cuda = a1.cuda
backend = os.environ.get("BACKEND", "cuda")

_bundle = {}
_orig = a1.build_kernel_bundle
def _cached(is_cuda):
    if is_cuda not in _bundle:
        _bundle[is_cuda] = _orig(is_cuda)
    return _bundle[is_cuda]


a1.build_kernel_bundle = _cached

for inst in sys.argv[6:]:
    problem = Problem.load(os.path.join(data_dir, inst + ".txt"))
    pop = int(os.environ.get("POP", "30"))
    cfg = a1.SolverConfig(compute_backend=backend, max_iter=gens, init="mixed", pop_size=pop)
    solver = a1.PHSHOWOAGpuSolver(problem, cfg)
    if "OPT" in os.environ and hasattr(solver, "opt"):
        solver.opt = int(os.environ["OPT"])
    t0 = time.perf_counter()
    solver.solve(0)
    compile_s = time.perf_counter() - t0

    wall, costs = [], []
    for r in range(runs):
        cfg.seed = 42 + r
        res = solver.solve(r + 1)
        wall.append(res.elapsed_seconds)
        costs.append(round(res.cost, 3))

    times = {}
    plain = solver._launch

    probe = solver.is_cuda and os.environ.get("PROBE") == "1" and "update_probe" in solver.kernels
    if probe:                                   # stage timings of the update kernel on copies (run unchanged)
        P, R, L = solver.P, solver.R, solver.L
        dev = lambda shape, dt: cuda.device_array(shape, dtype=dt)
        twin = lambda: (dev((P, R, L), "int32"), dev((P, R), "int32"), dev(P, "int32"), dev(P, "float64"),
                        dev(P, "float64"))
        cand2, bak2, rng2 = twin(), twin(), dev((P, 4), "uint32")

    def timed(name, grid, *args, threads=None):
        if probe and name == "update_population":
            for stage in (0, 1, 2):
                cuda.synchronize()
                rng2.copy_to_device(args[28])
                t = time.perf_counter()
                solver.kernels["update_probe"][grid, threads](*args[:5], *bak2, *cand2, *args[15:28], rng2,
                                                             *args[29:], stage)
                cuda.synchronize()
                e = times.setdefault(f"probe_stage{stage}", [0.0, 0])
                e[0] += time.perf_counter() - t
                e[1] += 1
        if solver.is_cuda:
            cuda.synchronize()
        t = time.perf_counter()
        plain(name, grid, *args, threads=threads)
        if solver.is_cuda:
            cuda.synchronize()
        e = times.setdefault(name, [0.0, 0])
        e[0] += time.perf_counter() - t
        e[1] += 1

    solver._launch = timed
    for r in range(runs):
        cfg.seed = 42 + r
        solver.solve(r + 1)
    solver._launch = plain
    kern = {n: [round(v[0] / runs, 4), v[1] // runs] for n, v in sorted(times.items(), key=lambda x: -x[1][0])}
    res_use = {}
    if solver.is_cuda:                          # registers / local memory (spills) / shared memory per kernel
        for n in ("update_population", "init_population", "route_elimination", "local_search_bi"):
            try:
                kk = solver.kernels[n]
                res_use[n] = [str(kk.get_regs_per_thread()), str(kk.get_local_mem_per_thread()), str(kk.get_shared_mem_per_block())]
            except Exception as e:  # noqa: BLE001
                res_use[n] = str(e)[:80]
    print(json.dumps({"inst": inst, "pkg": pkg, "pop": pop, "opt": os.environ.get("OPT", ""), "backend": backend, "gens": gens, "compile": round(compile_s, 1),
                      "wall": [round(w, 3) for w in wall], "cost": costs, "kernels": kern, "res": res_use}, default=str), flush=True)
