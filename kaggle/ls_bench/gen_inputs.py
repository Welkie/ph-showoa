"""Build realistic LS inputs: the population right before the periodic local search of generation 25.

The current solver (CPU backend) runs the normal initialization (which ends with Deep LS) and 24 generations of
WOA/SHO with the SA acceptance; the periodic route elimination / local search launches are skipped, so the saved
population is what the LS kernel receives at iteration 25. usage: python gen_inputs.py <out_dir> inst [inst ...]
"""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src_python_gpu_SA_RCRS_GRASP import algorithm1_solver as a1
from src_python_gpu_SA_RCRS_GRASP.problem import Problem

_bundle = {}
_orig_bundle = a1.build_kernel_bundle
a1.build_kernel_bundle = lambda is_cuda: _bundle.setdefault(is_cuda, None) or _bundle.__setitem__(is_cuda, _orig_bundle(is_cuda)) or _bundle[is_cuda]

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
for inst in sys.argv[2:]:
    problem = Problem.load(os.path.join(ROOT, "dataset", inst + ".txt"))
    solver = a1.PHSHOWOAGpuSolver(problem, a1.SolverConfig(compute_backend="cpu", max_iter=24, seed=42))
    orig = solver._launch
    solver._launch = lambda name, grid, *args: None if name in ("local_search", "route_elimination") else orig(name, grid, *args)
    solver.solve(1)
    nodes, rlen, nr, dist, cost = solver.buf["pop"]          # 24 generations (even) -> current population is "pop"
    np.savez(os.path.join(out_dir, f"{inst}.npz"), nodes=nodes, rlen=rlen, nr=nr, dist=dist, cost=cost)
    print(inst, "saved: NV", nr.min(), "-", nr.max(), "mean cost", round(float(cost.mean()), 1), flush=True)
