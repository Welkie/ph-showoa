"""Semantic regressions against src_python, independent of graph replay parity."""

import copy
import math

import numpy as np
import pytest

from src_python.search_framework import (
    _combined_local_search, _guided_route_crossover, _relink_route,
    _sa_initialization, feasible_or_repair_algorithm_10, quick_check_feasibility,
    _woa_intensification,
)
from src_python.solution import Route, Solution
from src_python_gpu_SA_RCRS_GRASP.gpu_engine import GpuEngine
from test_gpu_search_graph import tiny_data


@pytest.fixture(scope="module")
def engine():
    data = tiny_data(population=4, islands=1)
    data.paper_flags = True
    data.gpu_2opt_star = False  # Test base neighborhoods, excluding the GPU extension.
    return GpuEngine(data, is_cuda=False)


def solution(data, routes):
    result = Solution(data)
    for nodes in routes:
        route = Route(data)
        route.node_list = list(nodes)
        route.update(data)
        result.route_list.append(route)
    result.cal_cost(data)
    return result


def encode(group, routes, s=0):
    nodes, lengths, counts, distances, costs = group
    counts[s] = len(routes)
    for r, route in enumerate(routes):
        lengths[s, r] = len(route)
        nodes[s, r, :len(route)] = route


def decode(group, s=0):
    return [group[0][s, r, :group[1][s, r]].tolist() for r in range(group[2][s])]


class DeviceRandom:
    """Expose the device's RNG stream to the base SA implementation."""
    def __init__(self, words):
        self.words = [int(v) for v in words]

    def next(self):
        x, y, z, w = self.words
        t = (x ^ ((x << 11) & 0xFFFFFFFF)) & 0xFFFFFFFF
        value = ((w ^ (w >> 19)) ^ (t ^ (t >> 8))) & 0xFFFFFFFF
        self.words = [y, z, w, value]
        return value

    def random(self):
        return self.next() / 4294967296.0

    def randint(self, low, high):
        return low if low >= high else low + self.next() % (high - low + 1)

    def uniform(self, low, high):
        return low + (high - low) * self.random()


@pytest.mark.parametrize("routes,valid", [
    ([[0, 1, 2, 0], [0, 3, 4, 0], [0, 5, 6, 0]], True),
    ([[0, 1, 0]], False),
    ([[0, 1, 2, 0], [0, 3, 4, 0], [0, 5, 5, 0]], False),
    ([[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]], False),
])
def test_coverage_and_capacity_match_base(engine, routes, valid):
    b = engine._allocate_buffers()
    encode(b[:5], routes)
    actual = engine.kernels['device_evaluate'](*b[:3], 0, b[30])
    assert actual[0] == valid == quick_check_feasibility(solution(engine.data, routes), engine.data)
    if valid:
        assert actual[3] == pytest.approx(solution(engine.data, routes).cost)


def test_refresh_compacts_empty_middle_route_without_losing_tail(engine):
    b = engine._allocate_buffers()
    routes = [[0, 1, 2, 0], [0, 0], [0, 3, 4, 0], [0, 5, 6, 0]]
    encode(b[:5], routes)
    b[3][0], b[4][0] = -1, -1
    assert engine.kernels['device_refresh'](*b[:5], 0, b[30])
    assert decode(b[:5]) == [routes[0], routes[2], routes[3]]
    expected = solution(engine.data, decode(b[:5]))
    assert b[4][0] == pytest.approx(expected.cost)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("amount", [0.0, 0.4, 1.0])
def test_relink_matches_base_with_partial_customer_overlap(engine, amount, reverse):
    route = [0, 1, 2, 3, 4, 0]
    guide = [0, 4, 6, 2, 3, 0]
    a = solution(engine.data, [route]).get(0)
    g = solution(engine.data, [guide]).get(0)
    _relink_route(a, g, amount, reverse, engine.data)
    actual = np.array(route, dtype=np.int32)
    engine.kernels['device_relink'](actual, len(actual), np.array(guide, dtype=np.int32),
                                    len(guide), amount, reverse, np.zeros(7, dtype=np.int32))
    assert actual.tolist() == a.node_list


@pytest.mark.parametrize("routes", [
    [[0, 1, 4, 0], [0, 2, 6, 0], [0, 3, 5, 0]],
    [[0, i, 0] for i in range(1, 7)],
])
def test_combined_local_search_matches_base_to_convergence(engine, routes):
    b = engine._allocate_buffers()
    encode(b[:5], routes)
    expected = solution(engine.data, routes)
    _combined_local_search(expected, engine.data)
    engine.kernels['device_local_search'](*b[:5], 0, b[25], b[26], b[30], 0)
    assert decode(b[:5]) == [r.node_list for r in expected.route_list]
    assert b[4][0] == pytest.approx(expected.cost)


@pytest.mark.parametrize("seed", [42, 900, 100045])
def test_sa_initialization_matches_base_with_identical_rng_stream(engine, seed):
    b = engine._allocate_buffers()
    routes = [[0, 1, 4, 0], [0, 2, 6, 0], [0, 3, 5, 0]]
    encode(b[:5], routes)
    rng = engine._init_rng(seed)
    expected_rng = DeviceRandom(rng[0])
    expected = _sa_initialization(solution(engine.data, routes), engine.data, expected_rng)
    engine.kernels['device_warmup'](*b[:5], 0, *b[5:15], b[25], b[26], b[30], rng)
    assert decode(b[:5]) == [r.node_list for r in expected.route_list]
    assert b[4][0] == pytest.approx(expected.cost)
    np.testing.assert_array_equal(rng[0], expected_rng.words)


def test_repair_duplicates_and_missing_customers_matches_base(engine):
    b = engine._allocate_buffers()
    routes = [[0, 1, 2, 0], [0, 2, 4, 0], [0, 5, 0]]
    encode(b[:5], routes)
    expected = solution(engine.data, routes)
    feasible_or_repair_algorithm_10(expected, engine.data)
    assert engine.kernels['device_repair'](*b[:5], 0, b[25], b[28], b[30])
    assert decode(b[:5]) == [r.node_list for r in expected.route_list]
    assert b[4][0] == pytest.approx(expected.cost)


@pytest.mark.parametrize("violation", ["capacity", "time_window"])
def test_repair_load_and_time_violations_matches_base(engine, violation):
    b = engine._allocate_buffers()
    data = copy.deepcopy(engine.data)
    problem = list(b[30])
    if violation == "capacity":
        routes = [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]]
    else:
        routes = [[0, 1, 2, 0], [0, 3, 4, 0], [0, 5, 6, 0]]
        data.node[6].end = 10.0
        problem[8] = problem[8].copy()
        problem[8][6] = 10.0
    encode(b[:5], routes)
    expected = solution(data, routes)
    feasible_or_repair_algorithm_10(expected, data)
    assert engine.kernels['device_repair'](*b[:5], 0, b[25], b[28], tuple(problem))
    assert decode(b[:5]) == [r.node_list for r in expected.route_list]
    assert b[4][0] == pytest.approx(expected.cost)


def test_infeasible_population_is_never_recorded_as_best(engine):
    b = engine._allocate_buffers()
    b[2][:] = 6
    b[4][:] = math.inf
    b[19][:] = math.inf
    engine.kernels['update_island_bests'](*b[:5], *b[15:20], 1, engine.P)
    assert b[17][0] == 0
    assert math.isinf(b[19][0])


def test_route_elimination_rejects_a_scalar_cost_increase(engine):
    b = engine._allocate_buffers()
    problem = list(b[30])
    problem[1] = 20.0
    matrix = np.full((7, 7), 5000.0)
    np.fill_diagonal(matrix, 0)
    matrix[0, :] = matrix[:, 0] = 1.0
    problem[10] = matrix
    # Keep travel-time feasibility independent of expensive customer edges.
    encode(b[:5], [[0, i, 0] for i in range(1, 7)])
    assert engine.kernels['device_refresh'](*b[:5], 0, tuple(problem))
    before = decode(b[:5])
    cost = b[4][0]
    # Other rows are empty; the active row's proposed merge costs +2998.
    engine.kernels['route_elimination'](*b[:5], b[25], b[27], b[28], tuple(problem), 5)
    assert decode(b[:5]) == before
    assert b[4][0] == cost


def test_diversification_keeps_population_feasible_and_metadata_fresh(engine):
    b = engine._allocate_buffers()
    routes = [[0, i, 0] for i in range(1, 7)]
    for s in range(engine.P):
        encode(b[:5], routes, s)
        assert engine.kernels['device_refresh'](*b[:5], s, b[30])
    engine.kernels['update_island_bests'](*b[:5], *b[15:20], 1, engine.P)
    before = b[0].copy()
    engine.kernels['stagnation_diversify'](*b[:5], *b[10:20], b[25], b[27], b[28],
                                          b[30], engine._init_rng(42), 1, engine.P)
    changed = False
    for s in range(engine.P):
        actual = engine.kernels['device_evaluate'](*b[:3], s, b[30])
        assert actual[0]
        assert b[2][s] == actual[1]
        assert b[3][s] == pytest.approx(actual[2])
        assert b[4][s] == pytest.approx(actual[3])
        changed |= not np.array_equal(b[0][s], before[s])
    assert changed


def test_best_selection_uses_scalar_cost_even_when_vehicle_count_increases(engine):
    b = engine._allocate_buffers()
    b[2][:] = [2, 3, 4, 5]
    b[3][:] = [3000, 100, 100, 100]
    b[4][:] = 2000 * b[2] + b[3]
    b[1][:] = 3
    engine.kernels['update_island_bests'](*b[:5], *b[15:20], 1, engine.P)
    assert b[17][0] == 3
    assert b[19][0] == 6100


def test_failed_repair_is_rejected(engine):
    b = engine._allocate_buffers()
    problem = list(b[30])
    problem[5] = problem[5].copy()
    problem[5][1] = 1000.0
    encode(b[:5], [[0, i, 0] for i in range(1, 7)])
    assert not engine.kernels['device_repair'](*b[:5], 0, b[25], b[28], tuple(problem))


def test_route_capacity_buffer_has_no_105_route_cap():
    data = copy.deepcopy(tiny_data())
    data.customer_num = 120
    data.node.extend([copy.deepcopy(data.node[1]) for _ in range(114)])
    data.dist = data.time = np.zeros((121, 121))
    data.vehicle.max_num = 120
    assert GpuEngine(data, is_cuda=False).R >= 120


@pytest.mark.parametrize("seed", [42, 100045, 72])
def test_woa_encircling_and_spiral_match_base(engine, seed):
    b = engine._allocate_buffers()
    data = copy.deepcopy(engine.data)
    data.vehicle.capacity = 20
    problem = list(b[30])
    problem[1] = 20.0
    current = [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]]
    guide = [[0, 3, 2, 1, 0], [0, 6, 4, 5, 0]]
    encode(b[:5], current)
    encode(b[15:20], guide)
    for group in (b[:5], b[15:20]):
        assert engine.kernels['device_refresh'](*group, 0, tuple(problem))
    rng = engine._init_rng(seed)
    expected_rng = DeviceRandom(rng[0])
    expected = _woa_intensification(solution(data, current), solution(data, guide),
                                    0.0, data, expected_rng)
    engine.kernels['device_woa'](*b[:5], 0, *b[15:20], 0, *b[10:15], *b[5:10],
                                 b[25], b[26], b[27], b[28], 0.0, tuple(problem), rng)
    assert decode(b[10:15]) == [r.node_list for r in expected.route_list]
    assert b[14][0] == pytest.approx(expected.cost)
    np.testing.assert_array_equal(rng[0], expected_rng.words)


def test_crossover_matches_base_and_depends_on_partner(engine):
    b = engine._allocate_buffers()
    current = [[0, i, 0] for i in range(1, 7)]
    guide = current
    encode(b[:5], current)
    encode(b[15:20], guide)
    outputs = []
    for order in ([6, 5, 4, 3, 2, 1], [2, 4, 6, 1, 3, 5]):
        peer = [[0, c, 0] for c in order]
        encode(b[:5], peer, 1)
        rng = engine._init_rng(42)
        # Replay the GPU's selected seed routes to the base; the PRNG algorithm
        # differs from Python, but the discrete crossover decisions are equal.
        decisions = DeviceRandom(rng[0])
        take = 1 if decisions.random() < 0.6 else 2
        first = decisions.randint(0, 5)
        selected = [first]
        if take == 2:
            second = decisions.randint(0, 4)
            selected.append(second + (second >= first))

        class CrossoverRandom:
            def shuffle(self, sequence):
                sequence[:] = selected + [r for r in sequence if r not in selected]

            def random(self):
                return 0.0 if take == 1 else 0.9

        expected = _guided_route_crossover(solution(engine.data, guide),
                                           solution(engine.data, peer),
                                           solution(engine.data, current), engine.data, CrossoverRandom())
        expected.cal_cost(engine.data)
        engine.kernels['device_crossover'](*b[:3], 0, *b[:3], 1, *b[15:18], 0,
                                          *b[10:15], b[25], b[27], b[28], b[30], rng)
        outputs.append(decode(b[10:15]))
        assert outputs[-1] == [r.node_list for r in expected.route_list]
        assert b[14][0] == pytest.approx(expected.cost)
    assert outputs[0] != outputs[1]
