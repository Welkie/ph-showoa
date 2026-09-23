"""Regression cases for Algorithm 1's scalar-cost SA acceptance."""

import math
from types import SimpleNamespace

import pytest

from src_python_gpu_SA_RCRS_GRASP.gpu_kernels import scalar_sa_probability


@pytest.mark.parametrize("old_nv,old_td,new_nv,new_td,iteration,max_iter", [
    (2, 100, 3, 100, 0, 100),  # An extra vehicle is not forbidden.
    (2, 100, 2, 200, 50, 100),  # Scale by total cost, not distance.
    (3, 100, 2, 3000, 50, 100),  # Fewer vehicles can still cost more.
    (2, 3000, 3, 100, 50, 100),  # More vehicles can cost less.
    (2, 100, 2, 100, 100, 100),
    (2, 100, 2, 100.0005, 100, 100),  # No unconditional epsilon acceptance.
    (2, 100, 3, 100, 100, 100),
    (2, 100, 3, 100, 0, 0),
])
def test_scalar_sa_matches_paper(old_nv, old_td, new_nv, new_td, iteration, max_iter):
    old_cost = 2000.0 * old_nv + old_td
    new_cost = 2000.0 * new_nv + new_td
    temperature = 1.0 - iteration / max_iter if max_iter else 0.0
    expected = (1.0 if new_cost < old_cost else
                math.exp(-(new_cost - old_cost) / (1e-6 + temperature * abs(old_cost))))
    assert scalar_sa_probability(new_cost, old_cost, iteration, max_iter) == pytest.approx(expected)


def test_python_fallback_accepts_extra_vehicle_and_rejects_tiny_increase_when_cold():
    pytest.importorskip("torch")
    from src_python_gpu_SA_RCRS_GRASP.search_framework import _sa_accept

    rng = SimpleNamespace(random=lambda: 0.25)
    assert _sa_accept(SimpleNamespace(cost=6100.0), None, 4100.0, 0, 100, rng)
    assert not _sa_accept(SimpleNamespace(cost=4100.0005), None, 4100.0, 100, 100, rng)


def test_numba_compiled_sa_probability():
    numba = pytest.importorskip("numba")
    compiled = numba.njit(scalar_sa_probability)
    assert compiled(6100.0, 4100.0, 0, 100) == pytest.approx(math.exp(-2000 / (1e-6 + 4100)))
