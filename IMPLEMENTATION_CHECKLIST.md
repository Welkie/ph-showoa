# PH-SHOWOA Implementation Checklist

Reference paper: `journal.pone.0343262.pdf`

## Core Algorithms Implemented

- [x] VRPSPDTW route representation with simultaneous delivery, pickup, capacity, and time-window feasibility.
- [x] Hierarchical objective approximation prioritizing vehicle count, then distance, through a high per-route dispatch weight plus distance cost.
- [x] Population initialization with RCRS, randomized RCRS, and TD insertion heuristics.
- [x] SHO-style population diversification through peer-guided route recombination and light mutation.
- [x] WOA-style best-guided intensification through elite route and segment injection.
- [x] Adaptive hybrid probability that shifts from population-guided diversification toward best-guided intensification over iterations.
- [x] Simulated-annealing acceptance for occasional non-improving solution updates.
- [x] Periodic deep local search on the global best.
- [x] Stagnation-triggered population diversification with ruin-and-recreate repair.
- [x] Optional process-level parallel solution updates.
- [x] Ablation mode control for `ph_showoa`, `sho`, and `woa`.

## Known Implementation Caveats

- [ ] Replace the weighted vehicle/distance objective with strict lexicographic ranking if exact paper-objective fidelity is required in every comparison.
- [ ] Wire or remove legacy CLI options that are parsed but not active in the PH-SHOWOA loop, including `--no_crossover`, `--cross_repair`, `--parent_selection`, and `--replacement`.
- [ ] Benchmark Windows process-pool overhead on large instances before using `--workers 0` as the default for reproduction experiments.

## Local Search And Repair

- [x] 2-opt.
- [x] 2-opt*.
- [x] Or-opt with configurable maximum segment length.
- [x] 2-exchange with configurable maximum exchange length.
- [x] Random removal.
- [x] Related removal.
- [x] Greedy insertion.
- [x] Regret insertion.

## Paper-Specific Runtime Controls

| Parameter | CLI option | Default |
| --- | --- | --- |
| Population size | `--pop_size` | `64` |
| PH-SHOWOA iterations | `--max_iter` or `--g_1` | `500` |
| Worker processes | `--workers` | all CPU cores |
| Periodic local-search interval | `--local_search_interval` | `25` |
| Stagnation interval | `--stagnation_interval` | `50` |
| Diversified non-elite fraction | `--diversify_ratio` | `0.40` |
| SHO mutation probability | `--sho_mutation_prob` | `0.35` |
| PH/SHO/WOA ablation mode | `--hybrid_mode` | `ph_showoa` |

## Verification

- [ ] Run a short smoke test on a small benchmark instance.
- [ ] Compare PH-SHOWOA against standalone SHO-like and WOA-like ablations.
- [x] Generate paper-performance target file from Tables 3-5.
- [x] Generate gated paper-performance regression test.
- [x] Run paper-performance test for RCdp1001.
- [ ] Run paper-performance test for representative 100-customer WC instances within timeout.
- [ ] Reproduce the paper's complete benchmark table settings.
