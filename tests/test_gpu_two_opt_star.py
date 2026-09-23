"""The extension escapes a base local optimum without changing the objective."""
from types import SimpleNamespace

import numpy as np
import pytest
from numba import cuda

from src_python_gpu_SA_RCRS_GRASP.gpu_engine import GpuEngine


@pytest.mark.parametrize('is_cuda', [False, True])
def test_tail_exchange_escapes_base_optimum(is_cuda):
    if is_cuda and not cuda.is_available():
        pytest.skip('CUDA or simulator required')
    xy = np.array([[1,78],[12,11],[22,38],[15,68],[50,42],[52,4],
                   [84,49],[42,7],[97,18],[42,93],[0,32],[54,29],
                   [61,81],[34,39],[24,61],[72,46],[82,15]], dtype=float)
    dm = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(axis=2))
    routes = [[0,15,6,8,16,0], [0,7,5,12,9,0],
              [0,10,1,2,3,0], [0,14,4,11,13,0]]
    results = []
    for enabled in (False, True):
        data = SimpleNamespace(DC=0, customer_num=16, start_time=0.,
            vehicle=SimpleNamespace(capacity=4., max_num=4),
            node=[SimpleNamespace(delivery=float(i > 0), pickup=0., start=0., end=1e6, s_time=0.)
                  for i in range(17)], dist=dm, time=dm, p_size=1, num_islands=1,
            sa_iterations=0, gpu_2opt_star=enabled)
        engine = GpuEngine(data, is_cuda)
        b = engine._allocate_buffers()
        host = [np.zeros(x.shape, dtype=x.dtype) for x in b[:5]]
        host[2][0] = 4
        for r, route in enumerate(routes):
            host[0][0, r, :len(route)] = route
            host[1][0, r] = len(route)
        for dst, src in zip(b[:5], host):
            if is_cuda:
                dst.copy_to_device(src)
            else:
                dst[:] = src
        args = (*b[:5], b[25], b[26], b[30], 0)
        if is_cuda:
            engine.kernels['local_search'][1, 1](*args)
            cuda.synchronize()
            group = [x.copy_to_host() for x in b[:5]]
        else:
            engine.kernels['local_search'](*args)
            group = b[:5]
        actual = [group[0][0,r,:group[1][0,r]].tolist() for r in range(group[2][0])]
        assert group[2][0] == 4
        assert sorted(c for route in actual for c in route[1:-1]) == list(range(1,17))
        assert all(len(route) == 6 and route[0] == route[-1] == 0 for route in actual)
        distance = sum(dm[a,z] for route in actual for a,z in zip(route,route[1:]))
        assert group[3][0] == pytest.approx(distance)
        assert group[4][0] == pytest.approx(8000 + distance)
        results.append(distance)
    assert results[0] == pytest.approx(772.7120002170607)
    assert results[1] < results[0] - 25
