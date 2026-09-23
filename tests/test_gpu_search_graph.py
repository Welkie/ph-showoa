"""Run with NUMBA_ENABLE_CUDASIM=1, or on a real CUDA GPU to test capture/replay.

The oracle is an independent host schedule using the same CUDA operators.
Simulator tests exercise device logic but do NOT validate CUDA Graph APIs/JIT.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("numba")
from numba import cuda

from src_python_gpu_SA_RCRS_GRASP.gpu_engine import (
    GpuEngine, _dynamic_parameters, _mode_probability,
)
from src_python_gpu_SA_RCRS_GRASP.gpu_graph import CudaSearchGraph
from src_python_gpu_SA_RCRS_GRASP.solution import Solution


def tiny_data(mode="ph_showoa", iterations=6, population=4, islands=2):
    xy = np.array([[0, 0], [2, 1], [3, 4], [7, 2], [8, 5], [1, 8], [5, 7]], dtype=float)
    distance = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(axis=2))
    nodes = [SimpleNamespace(delivery=0.0, pickup=0.0, start=0.0, end=200.0, s_time=0.0)]
    nodes += [SimpleNamespace(delivery=2.0, pickup=1.0, start=float(i), end=200.0, s_time=1.0)
              for i in range(6)]
    return SimpleNamespace(
        customer_num=6, DC=0, vehicle=SimpleNamespace(capacity=5.0, max_num=6),
        start_time=0.0, end_time=200.0, node=nodes, dist=distance, time=distance,
        p_size=population, num_islands=islands, max_iter=iterations, runs=2,
        local_search_interval=2, stagnation_interval=3, migration_interval=2,
        grasp_alpha_lo=0.1, grasp_alpha_hi=0.4, hybrid_mode=mode,
        compute_backend="cuda", seed=42, output_per_gens=2,
        sa_t0=2.0, sa_tmin=0.5, sa_alpha=0.5, sa_itermax=3,
        sho_mutation_prob=0.35, diversify_ratio=0.4,
    )


def reference_cuda_schedule(engine, buffers, seed):
    """Independent host implementation of the base-aligned scheduling rules."""
    pop, nxt, cand = buffers[:5], buffers[5:10], buffers[10:15]
    ibest, gbest = buffers[15:20], buffers[20:25]
    route, route2, unrouted, flags, scores = buffers[25:30]
    problem = buffers[30]
    rng = engine._init_rng(seed)
    ibest[2].copy_to_device(np.zeros(engine.num_islands, dtype=np.int32))
    gbest[2].copy_to_device(np.zeros(1, dtype=np.int32))
    for group in (ibest, gbest):
        for arr in group[3:5]:
            arr.copy_to_device(np.full(arr.shape, np.inf, dtype=np.float64))
    k = engine.kernels

    def launch(name, args, blocks=None, threads=None):
        k[name][engine.blocks if blocks is None else blocks,
                engine.threads_per_block if threads is None else threads](*args)

    def bests():
        launch("update_island_bests", (*pop, *ibest, engine.num_islands, engine.island_size), engine.isl_blocks)
        launch("update_global_best", (*ibest, *gbest, engine.num_islands), 1, 1)

    launch("init_population", (*pop, *nxt, *cand, route, route2, unrouted, flags, scores, problem, 0.1, 0.4, rng))
    bests()
    history = []
    no_improve = 0
    for gen in range(1, engine.data.max_iter + 1):
        previous_best = gbest[4].copy_to_host()[0]
        a, p = _dynamic_parameters(gen - 1, engine.data.max_iter)
        p = _mode_probability(p, engine.data)
        launch("update_population", (*pop, *nxt, *cand, *ibest,
               route, route2, unrouted, flags, problem, a, p, gen - 1,
               engine.data.max_iter, rng, engine.island_size))
        pop, nxt = nxt, pop
        bests()
        if (gen - 1) % engine.data.local_search_interval == 0:
            launch("route_elimination", (*gbest, route, unrouted, flags, problem, 5), 1, 1)
            launch("local_search", (*gbest, route, route2, problem, 0), 1, 1)
        launch("publish_global_best", (*gbest, *ibest, engine.num_islands), 1, 1)
        if gbest[4].copy_to_host()[0] < previous_best - 0.001:
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= engine.data.stagnation_interval:
            launch("stagnation_diversify", (*pop, *cand, *ibest, route, unrouted, flags, problem,
                   rng, engine.num_islands, engine.island_size), engine.isl_blocks)
            no_improve = 0
        if engine.num_islands > 1 and gen % engine.data.migration_interval == 0:
            launch("island_migration", (*pop, *ibest, engine.num_islands, engine.island_size), 1, 1)
        bests()
        history.append((a, p, gbest[2].copy_to_host()[0], gbest[3].copy_to_host()[0]))
    cuda.synchronize()
    return rng.copy_to_host(), np.array(history).reshape((-1, 4)), pop


def assert_solutions_equal(actual, expected):
    a = [x.copy_to_host() for x in actual]
    b = [x.copy_to_host() for x in expected]
    np.testing.assert_array_equal(a[2], b[2])
    np.testing.assert_allclose(a[3], b[3], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(a[4], b[4], rtol=1e-12, atol=1e-10)
    # Unused route slots intentionally remain uninitialized, so compare only
    # the active representation, including every customer and depot endpoint.
    for s, nr in enumerate(a[2]):
        np.testing.assert_array_equal(a[1][s, :nr], b[1][s, :nr])
        for r in range(nr):
            length = a[1][s, r]
            np.testing.assert_array_equal(a[0][s, r, :length], b[0][s, r, :length])


requires_cuda = pytest.mark.skipif(not cuda.is_available(), reason="Requires CUDA or NUMBA_ENABLE_CUDASIM=1")


@requires_cuda
@pytest.mark.parametrize("mode,iterations,islands,population", [
    ("ph_showoa", 6, 2, 4), ("sho", 3, 2, 4), ("woa", 4, 1, 4),
    ("ph_showoa", 0, 2, 4), ("ph_showoa", 2, 2, 34),
])
def test_replay_matches_reference_schedule_and_rng(mode, iterations, islands, population):
    engine = GpuEngine(tiny_data(mode, iterations, population, islands), is_cuda=True)
    actual, expected = engine._allocate_buffers(), engine._allocate_buffers()
    graph = CudaSearchGraph(engine, actual)
    snapshots = []
    try:
        # Replay with different seeds, then repeat the first seed. Detect stale
        # bests, generation state, RNG state and odd/even ping-pong parity.
        for seed in (100045, 200048, 100045):
            graph.prepare_run(seed)
            graph.launch()
            graph.synchronize()
            rng, history, old_pop = reference_cuda_schedule(engine, expected, seed)
            np.testing.assert_array_equal(graph.rng.copy_to_host(), rng)
            np.testing.assert_allclose(graph.read_history(), history, rtol=1e-12, atol=1e-10)
            assert graph.generation.copy_to_host()[0] == iterations + 1
            new_pop = actual[5:10] if iterations % 2 else actual[:5]
            assert_solutions_equal(new_pop, old_pop)
            assert_solutions_equal(actual[15:20], expected[15:20])
            assert_solutions_equal(actual[20:25], expected[20:25])
            snapshots.append(graph.rng.copy_to_host())
        np.testing.assert_array_equal(snapshots[0], snapshots[2])
    finally:
        graph.close()


@requires_cuda
def test_device_rng_matches_host_across_blocks_and_uint32_wrap():
    engine = GpuEngine(tiny_data(iterations=0, population=34), is_cuda=True)
    graph = CudaSearchGraph(engine, engine._allocate_buffers())
    try:
        reset = graph.steps[0]
        for seed in (0, -1, 2**32 + 123, 2**63 + 456):
            graph.prepare_run(seed)
            reset.kernel[reset.blocks, reset.threads, graph.stream](*reset.args)
            graph.synchronize()
            np.testing.assert_array_equal(graph.rng.copy_to_host(), engine._init_rng(seed).copy_to_host())
    finally:
        graph.close()


@requires_cuda
def test_solver_decodes_only_after_run_and_checks_feasibility(capsys):
    data = tiny_data(iterations=2)
    solution = Solution(data)
    engine = GpuEngine(data, is_cuda=True)
    assert engine.run_solve(solution)
    assert solution.check(data, False)
    output = capsys.readouterr().out
    for run in (1, 2):
        start = output.index(f"Run {run} RUN_")
        end = output.index("_END", start)
        decode = output.index(f"Run {run} CPU_DECODE")
        assert "Gen:" not in output[start:end]
        assert end < decode


def test_explicit_cuda_never_silently_falls_back(monkeypatch):
    import src_python_gpu_SA_RCRS_GRASP.gpu_engine as module
    monkeypatch.setattr(module, "cuda", None)
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        GpuEngine(tiny_data(), is_cuda=True)


@requires_cuda
def test_stagnation_counts_non_improving_generations_and_resets_on_improvement():
    engine = GpuEngine(tiny_data(iterations=1), is_cuda=True)
    graph = CudaSearchGraph(engine, engine._allocate_buffers())
    try:
        step = next(step for step in graph.steps if step.name == 'stagnation')
        best, previous, count, due, interval = step.args
        count.copy_to_device(np.zeros(1, dtype=np.int64))
        previous.copy_to_device(np.array([6100.0]))
        for value, expected_count, expected_due in [
            (6100.0, 1, 0), (6100.0, 2, 0), (6090.0, 0, 0),
            (6100.0, 1, 0), (6100.0, 2, 0), (6100.0, 0, 1),
        ]:
            best.copy_to_device(np.array([value]))
            step.kernel[1, 1, graph.stream](*step.args)
            graph.synchronize()
            assert count.copy_to_host()[0] == expected_count
            assert due.copy_to_host()[0] == expected_due
    finally:
        graph.close()
