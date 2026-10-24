"""Check that the CUDA prototype gives exactly the CPU result (run with NUMBA_ENABLE_CUDASIM=1 on a small instance)."""
import os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..", ".."))); sys.path.insert(0, here)
from numba import cuda
from src_python_gpu_SA_RCRS_GRASP.improvement1_full_gpu import prepare_problem_data
from src_python_gpu_SA_RCRS_GRASP.problem import Problem
import bi_ls
inst, n, i0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0
problem = Problem.load(os.path.join(here, "..", "..", "dataset", inst + ".txt"))
prob = prepare_problem_data(problem)
d = np.load(os.path.join(here, "inputs", inst + ".npz"))
P, R, L = d["nodes"].shape
SMAX = problem.num_customers + 2 * R
def bufs():
    return (d["nodes"][i0:i0 + n].copy(), d["rlen"][i0:i0 + n].copy(), d["nr"][i0:i0 + n].copy(), np.zeros((n, SMAX), np.int32),
            np.zeros((n, SMAX), np.int32), np.zeros((n, L), np.int32), np.zeros((n, L), np.int32))
a = bufs(); mv_c = np.zeros(n, np.int64)
bi_ls.build(False)(*a, prob, 1000000, mv_c)
b = [cuda.to_device(x) for x in bufs()]; mv_g = cuda.to_device(np.zeros(n, np.int64))
prob_d = tuple(cuda.to_device(x) if isinstance(x, np.ndarray) else x for x in prob)
bi_ls.build(True)[n, bi_ls.NT](*b, prob_d, 1000000, mv_g)
same = all(np.array_equal(a[i], b[i].copy_to_host()) for i in range(3))
print("moves cpu", mv_c, "gpu", mv_g.copy_to_host(), "identical:", same)
