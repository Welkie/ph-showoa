"""Algorithm 1. Parallel Hybrid SHOWOA with SA Acceptance for VRPSPDTW (GPU version).

Same structure as src_python_baseline/algorithm1_solver.py; the host launches one kernel per step and the whole run
stays on the device until the best solution is copied back once at the end (improvement 1). Differences from the
baseline are the declared improvements: initialization (improvement 2, unless --init sa) and local search for
every individual with Inter 2-opt*, executed block-parallel with best improvement (improvement 3).
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .algorithm9_feasibility import check_feasibility_canonical
from .improvement1_block_parallel import NTU as UPD_THREADS
from .improvement1_full_gpu import allocate_buffers, build_kernel_bundle, cuda, prepare_problem_data
from .improvement3_population_local_search import NT as LS_THREADS
from .problem import Problem
from .solution import Solution


@dataclass
class SolverConfig:
    pop_size: int = 30
    max_iter: int = 1000
    seed: int = 42
    local_search_interval: int = 25
    stagnation_limit: int = 50
    mutation_prob: float = 0.35
    compute_backend: str = "auto"          # auto | cuda | cpu
    # Initialization: "sa_rcrs_grasp" (improvement 2), "sa" (Algorithm 2 of the paper) or "mixed" (half each)
    init: str = "sa_rcrs_grasp"
    grasp_alpha_lo: float = 0.10
    grasp_alpha_hi: float = 0.40
    sa_warmup_iterations: int = 0          # SA warm-up after RCRS-GRASP (off: the ejection pool replaces it)
    route_elimination_passes: int = 5
    ep_budget: int = 5000                  # ejection-pool route minimization after RCRS-GRASP (CUDA only; not for init "sa")
    # Algorithm 2 parameters (same Priority Level 4 choice as src_python_baseline)
    t0_sa: float = 1.0
    alpha_sa: float = 0.9
    t_min_sa: float = 0.01
    iter_max_sa: int = 20
    # Improvement 3 (the defaults are the full improvement; the other values are for the ablation)
    ls_strategy: str = "best"              # best: block-parallel best improvement | first: Algorithm 8, one thread
    ls_scope: str = "population"           # population | best (periodic LS only for the best individual)
    ls_2opt_star: bool = True              # add Inter 2-opt*


@dataclass
class RunResult:
    instance: str
    algo: str
    run: int
    seed: int
    nv: int
    td: float
    cost: float
    feasible: bool
    violation_note: str
    best_solution: Solution
    elapsed_seconds: float
    convergence: List[tuple] = field(default_factory=list)   # (generation, NV, TD) of the global best


class PHSHOWOAGpuSolver:
    def __init__(self, problem: Problem, config: Optional[SolverConfig] = None) -> None:
        self.problem = problem
        self.config = config or SolverConfig()
        backend = self.config.compute_backend
        cuda_ok = cuda is not None and (cuda.is_available() or os.environ.get("NUMBA_ENABLE_CUDASIM") == "1")
        if backend == "cuda" and not cuda_ok:
            raise RuntimeError("compute_backend=cuda requested but no CUDA device is available")
        self.is_cuda = cuda_ok if backend == "auto" else backend == "cuda"

        self.P = int(self.config.pop_size)
        self.N = problem.num_customers
        self.R = min(self.N + 5, 105)
        self.L = self.N + 2
        self.threads = 32
        self.blocks = (self.P + self.threads - 1) // self.threads
        if self.config.ls_strategy not in ("best", "first") or self.config.ls_scope not in ("population", "best"):
            raise ValueError("ls_strategy must be best|first and ls_scope population|best")

        self.kernels = build_kernel_bundle(self.is_cuda)

    def _launch(self, name, grid, *args, threads=None):
        kernel = self.kernels[name]
        if self.is_cuda:
            kernel[grid, self.threads if threads is None else threads](*args)
        else:
            kernel(*args)

    def _local_search(self, pop, only_best):
        """Improvement 3: local search for the population (or only its best individual, ablation)."""
        cfg, b = self.config, self.buf
        if cfg.ls_strategy == "best":
            self._launch("local_search_bi", 1 if only_best else self.P, *pop, b["slot_r"], b["slot_k"],
                         b["scratch_route"], b["scratch_route2"], b["prob"], 1000000, int(cfg.ls_2opt_star),
                         int(only_best), threads=LS_THREADS)
        elif only_best:
            self._launch("local_search_best_fi", 1, *pop, b["scratch_route"], b["scratch_route2"], b["prob"],
                         1000000, int(cfg.ls_2opt_star), threads=1)
        else:
            self._launch("local_search", self.blocks, *pop, b["scratch_route"], b["scratch_route2"], b["prob"],
                         1000000, int(cfg.ls_2opt_star))

    def solve(self, run_index: int = 1) -> RunResult:
        start_time = time.time()
        cfg = self.config
        # Every run is timed on its own: the problem data is prepared and copied to the device and the buffers are
        # allocated inside the timed part (only the kernel compilation of the first run is shared)
        self.prob_data = prepare_problem_data(self.problem)
        # Improvement 1: prefix tables of the block-parallel insertions need integer demands and capacity
        q, dl, pk = self.prob_data[1], self.prob_data[5], self.prob_data[6]
        self.opt = 2 if (float(q).is_integer() and np.all(dl == np.round(dl)) and np.all(pk == np.round(pk))) else 0
        self.buf = b = allocate_buffers(self.is_cuda, self.P, self.R, self.L, self.N, int(cfg.max_iter),
                                        self.prob_data)
        prob = b["prob"]
        max_iter = int(cfg.max_iter)
        P_grid = self.blocks
        sr, sr2, su, sf, ss = (b["scratch_route"], b["scratch_route2"], b["scratch_unrouted"],
                               b["scratch_flags"], b["scratch_scores"])
        rng = b["rng_states"]

        # Fresh run state
        stag_zero = np.zeros((1, 3), dtype=np.float64)
        if self.is_cuda:
            stag_state = cuda.to_device(stag_zero)
            b["gbest"][2].copy_to_device(np.zeros(1, dtype=np.int32))
        else:
            stag_state = stag_zero
            b["gbest"][2][:] = 0
        self._launch("init_device_rng", P_grid, rng, int(cfg.seed))

        # Algorithm 1, lines 1-5: initial population and best
        pop, nxt, cand, bak, gbest = b["pop"], b["next"], b["cand"], b["bak"], b["gbest"]
        init_mode = {"sa_rcrs_grasp": 0, "sa": 1, "mixed": 2}[cfg.init]
        ini = (*pop, *cand, *bak, sr, sr2, su, sf, ss, prob, init_mode,
               float(cfg.grasp_alpha_lo), float(cfg.grasp_alpha_hi), int(cfg.sa_warmup_iterations),
               int(cfg.route_elimination_passes), float(cfg.t0_sa), float(cfg.alpha_sa), float(cfg.t_min_sa),
               int(cfg.iter_max_sa), rng,
               # first-improvement local search in the init kernel: 0 = none, 1 = with Inter 2-opt*, 2 = without
               int(init_mode != 1 and cfg.ls_strategy == "first") * (1 if cfg.ls_2opt_star else 2))
        ep_budget = int(cfg.ep_budget) if init_mode != 1 else 0     # init "sa" stays Algorithm 2 of the paper
        if self.is_cuda:                                               # improvement 1: one block per individual
            self._launch("init_population", self.P, *ini, ep_budget, self.opt, threads=UPD_THREADS)
        else:
            if ep_budget > 0:
                warnings.warn("the ejection pool (ep_budget > 0) needs the CUDA backend; skipped on the CPU")
            self._launch("init_population", P_grid, *ini)
        if init_mode != 1 and cfg.ls_strategy == "best":
            self._local_search(pop, only_best=False)     # last step of the sa_rcrs_grasp chain, every individual
        self._launch("update_global_best", 1, *pop, *gbest)

        # Lines 6-49: main loop
        cur, nxt_pop = pop, nxt
        for iteration in range(max_iter):
            a = 2.0 - 2.0 * iteration / max_iter                       # line 7
            p_hybrid = max(0.15, 0.5 * (1.0 - iteration / max_iter))  # line 8
            # Lines 10-33: every whale updated in parallel, population replaced synchronously (ping-pong buffers)
            upd = (*cur, *nxt_pop, *cand, gbest[0], gbest[1], gbest[2], sr, sr2, su, sf, prob, a, p_hybrid,
                   float(cfg.mutation_prob), iteration, max_iter, rng)
            if self.is_cuda:                                           # improvement 1: one block per whale
                self._launch("update_population", self.P, *upd, self.opt, threads=UPD_THREADS)
            else:
                self._launch("update_population", P_grid, *upd)
            cur, nxt_pop = nxt_pop, cur

            # Lines 34-37: periodic local search (improvement 3: every individual; improvement 2: route elimination)
            if iteration % cfg.local_search_interval == 0:
                if self.is_cuda:
                    self._launch("route_elimination", self.P, *cur, su, sf, prob, int(cfg.route_elimination_passes),
                                 self.opt, threads=UPD_THREADS)
                else:
                    self._launch("route_elimination", P_grid, *cur, sr, su, sf, prob,
                                 int(cfg.route_elimination_passes))
                self._local_search(cur, only_best=cfg.ls_scope == "best")

            self._launch("update_global_best", 1, *cur, *gbest)
            self._launch("record_log", 1, b["run_log"], gbest[2], gbest[3], iteration + 1)

            # Lines 38-48: stagnation counter on the best and DIVERSIFY
            self._launch("stagnation", 1, *cur, gbest[2], gbest[3], stag_state, int(cfg.stagnation_limit),
                         sr, sr2, sf, prob, rng)

        # Line 50: copy the best back once and verify it with the baseline's check (Section 3 constraints)
        if self.is_cuda:
            cuda.synchronize()
            g_nodes, g_rlen, g_nr = (gbest[0].copy_to_host(), gbest[1].copy_to_host(), gbest[2].copy_to_host())
            run_log = b["run_log"].copy_to_host()
        else:
            g_nodes, g_rlen, g_nr = gbest[0], gbest[1], gbest[2]
            run_log = b["run_log"]
        routes = []
        for r in range(int(g_nr[0])):
            l = int(g_rlen[0, r])
            if l > 2:
                routes.append([int(g_nodes[0, r, i]) for i in range(1, l - 1)])
        best = Solution(routes, self.problem)
        elapsed = time.time() - start_time
        is_feasible, violation_note = check_feasibility_canonical(best, self.problem)
        convergence = [(g, int(run_log[g - 1, 0]), float(run_log[g - 1, 1])) for g in range(1, max_iter + 1)]

        return RunResult(
            instance=self.problem.name,
            algo="PH-SHOWOA-GPU",
            run=run_index,
            seed=cfg.seed,
            nv=best.nv,
            td=best.td,
            cost=2000.0 * best.nv + best.td,
            feasible=is_feasible,
            violation_note=violation_note,
            best_solution=best,
            elapsed_seconds=elapsed,
            convergence=convergence,
        )


# ---------------------------------------------------------------------------
# Device part: Algorithm 1 lines 11-31 (one whale) and DIVERSIFY helper
# ---------------------------------------------------------------------------

def build(dev_fn, k):
    """Compile this module's device functions (CUDA device functions or CPU njit) into the bundle k."""
    copy_solution = k["copy_solution"]
    eval_solution = k["eval_solution"]
    rand_u01 = k["rand_u01"]
    update_position_hybrid = k["update_position_hybrid"]

    @dev_fn
    def update_individual(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost,
                          next_nodes, next_rlen, next_nr, next_dist, next_cost,
                          cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                          best_nodes, best_rlen, best_nr,
                          scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                          s,
                          prob_data, a_param, p_hybrid, mutation_prob, iteration, max_iter, rng_states):
        # Algorithm 1, lines 11-31 for one whale; the next buffer doubles as WOA's rollback copy
        took = update_position_hybrid(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                                      best_nodes, best_rlen, best_nr,
                                      cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost,
                                      next_nodes, next_rlen, next_nr, next_dist, next_cost,
                                      scratch_route, scratch_route2, scratch_unrouted, scratch_flags,
                                      a_param, p_hybrid, mutation_prob, prob_data, rng_states)
        if not took:
            copy_solution(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                          next_nodes, next_rlen, next_nr, next_dist, next_cost, s)
            return False

        # Lines 12-14: infeasible -> keep w; lines 15-27: nf <= cf accepted, otherwise SA (Metropolis) acceptance
        ok, act_r, t_d, t_c = eval_solution(cand_nodes, cand_rlen, cand_nr, s, prob_data)
        accepted = False
        if ok:
            nf = t_c
            cf = cur_cost[s]
            if nf <= cf:
                accepted = True
            else:
                temp = 1.0 - float(iteration) / float(max_iter) if max_iter > 0 else 0.0
                prob = math.exp(-(nf - cf) / (1e-6 + temp * abs(cf)))
                if rand_u01(rng_states, s) < prob:
                    accepted = True
        if accepted:
            copy_solution(cand_nodes, cand_rlen, cand_nr, cand_dist, cand_cost, s,
                          next_nodes, next_rlen, next_nr, next_dist, next_cost, s)
        else:
            copy_solution(cur_nodes, cur_rlen, cur_nr, cur_dist, cur_cost, s,
                          next_nodes, next_rlen, next_nr, next_dist, next_cost, s)
        return accepted

    @dev_fn
    def mark_worst_individuals(cost, scratch_flags, P):
        for s in range(P):
            scratch_flags[s, 0] = 0
        cnt = int(math.ceil(P * 0.20))
        if cnt < 1:
            cnt = 1
        for _ in range(cnt):
            worst = -1
            for s in range(P):
                if scratch_flags[s, 0] == 0 and (worst < 0 or cost[s] > cost[worst]):
                    worst = s
            scratch_flags[worst, 0] = 1

    k.update({"update_individual": update_individual, "mark_worst_individuals": mark_worst_individuals})
