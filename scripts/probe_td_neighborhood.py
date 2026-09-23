"""Reproduce a lost 2-opt* improvement; run using the workspace Python."""
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import argparse
import subprocess
from types import SimpleNamespace
import numpy as np
from src_python_gpu_SA_RCRS_GRASP.gpu_engine import GpuEngine

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--batch', action='store_true', help='Evaluate all cases, not only the first witness')
parser.add_argument('--cases', type=int, default=100)
args = parser.parse_args()
if args.cases <= 0:
    parser.error('--cases must be positive')
records = []

old = {'__name__': 'old_kernels'}
exec(subprocess.check_output(['git', 'show', 'd88da6e:src_python_gpu_SA_RCRS_GRASP/gpu_kernels.py'], text=True), old)
old_kernels = old['build_kernel_bundle'](False)
rng = np.random.default_rng(20260923)
for case in range(args.cases):
    n = 16
    xy = rng.integers(0, 100, (n + 1, 2))
    matrix = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(axis=2))
    data = SimpleNamespace(DC=0, customer_num=n, start_time=0.,
        vehicle=SimpleNamespace(capacity=4., max_num=4),
        node=[SimpleNamespace(delivery=float(i > 0), pickup=0., start=0., end=1e6, s_time=0.) for i in range(n + 1)],
        dist=matrix, time=matrix, p_size=1, num_islands=1, sa_iterations=0,
        gpu_2opt_star=False)
    engine = GpuEngine(data, False)
    b = engine._allocate_buffers()
    permutation = rng.permutation(np.arange(1, n + 1))
    b[2][0] = 4
    for r in range(4):
        b[1][0, r] = 6
        b[0][0, r, :6] = [0, *permutation[r*4:(r+1)*4], 0]
    def routes():
        return [b[0][0, r, :b[1][0, r]].tolist() for r in range(b[2][0])]
    engine.kernels['local_search'](*b[:5], b[25], b[26], b[30], 0)
    before, before_routes = float(b[3][0]), routes()
    old_kernels['local_search'](*b[:5], b[25], b[26], b[30], 2)
    valid, nv, td, cost = engine.kernels['device_evaluate'](*b[:3], 0, b[30])
    records.append(dict(case=case, valid=bool(valid), nv=int(nv),
                        before=before, after=float(td), gain=float(before-td)))
    if args.batch:
        continue
    if valid and td < before - .001:
        direct = None
        def length(route):
            return sum(matrix[a, z] for a, z in zip(route, route[1:]))
        for r1 in range(4):
            for r2 in range(r1 + 1, 4):
                a, z = before_routes[r1], before_routes[r2]
                for p in range(2, len(a) - 1):
                    for q in range(2, len(z) - 1):
                        x, y = a[:p] + z[q:], z[:q] + a[p:]
                        gain = length(a) + length(z) - length(x) - length(y)
                        if len(x) <= 6 and len(y) <= 6 and gain > .001:
                            direct = dict(routes=[r1, r2], cuts=[p, q], gain=gain)
        print(json.dumps(dict(case=case, seed=20260923, coordinates=xy.tolist(),
            current_local_optimum=before_routes, after_old_search=routes(), nv=nv,
            current_td=before, after_old_td=td, improvement=before-td,
            checked_by_current_evaluator=bool(valid), direct_tail_swap=direct), indent=2))
        break
if args.batch:
    valid_records = [r for r in records if r['valid']]
    print(json.dumps(dict(cases=args.cases, seed=20260923,
        valid=len(valid_records), improved=sum(r['gain'] > .001 for r in valid_records),
        worsened=sum(r['gain'] < -.001 for r in valid_records),
        mean_before=float(np.mean([r['before'] for r in valid_records])),
        mean_after=float(np.mean([r['after'] for r in valid_records])),
        records=records), indent=2))
elif not any(r['valid'] and r['gain'] > .001 for r in records):
    raise SystemExit(f'No witness found within {args.cases} cases')
