from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .problem import Problem
from .solution import Solution
from .algorithm1_solver import PHSHOWOAGpuSolver, SolverConfig, RunResult


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="PH-SHOWOA GPU solver (paper-aligned baseline + GPU improvements)")
    parser.add_argument("instance_path", nargs="?", default=None, type=str, help="Path to VRPSDPTW instance file (.txt or .vrpsdptw)")
    parser.add_argument("--problem", "-p", type=str, default=None, help="Alias for instance_path")
    parser.add_argument("--pop_size", type=int, default=30, help="Population size (default: 30)")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs (default: 1)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker threads (default: 1)")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path for run results")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save detailed solution logs")
    # GPU version options
    parser.add_argument("--compute_backend", choices=["auto", "cuda", "cpu"], default="auto",
                        help="cuda: one GPU thread per individual; cpu: same Numba code on the CPU (default: auto)")
    parser.add_argument("--init", choices=["sa_rcrs_grasp", "sa"], default="sa_rcrs_grasp",
                        help="sa_rcrs_grasp: improvement 2 (default); sa: Algorithm 2 of the paper")
    parser.add_argument("--num_islands", type=int, default=6, help="Islands (improvement 3, default: 6; 1 = one population)")
    parser.add_argument("--migration_interval", type=int, default=20, help="Iterations between ring migrations (default: 20)")
    parser.add_argument("--local_search_interval", type=int, default=25, help="Iterations between local searches (default: 25)")
    parser.add_argument("--grasp_alpha_lo", type=float, default=0.10)
    parser.add_argument("--grasp_alpha_hi", type=float, default=0.40)
    parser.add_argument("--ls_strategy", choices=["best", "first"], default="best",
                        help="Improvement 4 local search: best = block-parallel best improvement (default); "
                             "first = Algorithm 8, first improvement, one thread per individual (ablation)")
    parser.add_argument("--ls_scope", choices=["population", "best"], default="population",
                        help="Periodic local search for every individual (default) or only the best (ablation)")
    parser.add_argument("--no_ls_2opt_star", action="store_true",
                        help="Drop Inter 2-opt* from the local search (ablation; ls_strategy best only)")
    parser.add_argument("--convergence_csv", type=str, default=None,
                        help="Append the best NV/TD every --convergence_every iterations to this CSV")
    parser.add_argument("--convergence_every", type=int, default=10)
    return parser.parse_args(args)


def main():
    args = parse_args()
    path_str = args.instance_path or args.problem
    if not path_str:
        print("Error: instance_path or --problem must be provided.", file=sys.stderr)
        sys.exit(1)
    inst_path = Path(path_str)
    if not inst_path.exists():
        # Convenient fallback to data/Wang_Chen
        candidates = [
            Path("data/Wang_Chen") / f"explicit_{path_str.lower()}.vrpsdptw",
            Path("data/Wang_Chen") / f"{path_str.lower()}.txt",
            Path("data/Wang_Chen") / f"{path_str}.txt",
            Path("data/Wang_Chen") / path_str,
        ]
        found = False
        for cand in candidates:
            if cand.exists():
                inst_path = cand
                found = True
                break
        if not found:
            print(f"Error: Instance file {inst_path} does not exist.", file=sys.stderr)
            sys.exit(1)

    problem = Problem.load(inst_path)
    print(f"=== Loaded instance {problem.name} ({problem.num_customers} customers, Q={problem.capacity}, maxV={problem.max_vehicles}) ===")

    out_csv = Path(args.output_csv) if args.output_csv else None
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not out_csv.exists():
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["instance", "algo", "run", "seed", "NV", "TD", "cost", "feasible", "violation_note", "elapsed_seconds"])

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = SolverConfig(
        pop_size=args.pop_size,
        max_iter=args.max_iter,
        seed=args.seed,
        compute_backend=args.compute_backend,
        init=args.init,
        num_islands=args.num_islands,
        migration_interval=args.migration_interval,
        local_search_interval=args.local_search_interval,
        grasp_alpha_lo=args.grasp_alpha_lo,
        grasp_alpha_hi=args.grasp_alpha_hi,
        ls_strategy=args.ls_strategy,
        ls_scope=args.ls_scope,
        ls_2opt_star=not args.no_ls_2opt_star,
    )
    # One solver per process: kernels are compiled and device buffers allocated once, then reused by every run
    solver = PHSHOWOAGpuSolver(problem, config)
    print(f"Backend: {'cuda' if solver.is_cuda else 'cpu'} | init={config.init} | islands={solver.num_islands} "
          f"x {solver.island_size} | LS={config.ls_strategy}/{config.ls_scope}"
          f"{'' if config.ls_2opt_star else ' (no 2-opt*)'}")

    conv_csv = Path(args.convergence_csv) if args.convergence_csv else None
    if conv_csv:
        conv_csv.parent.mkdir(parents=True, exist_ok=True)
        if not conv_csv.exists():
            with open(conv_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["instance", "run", "seed", "iteration", "NV", "TD", "cost"])

    results = []
    for r in range(1, args.runs + 1):
        run_seed = args.seed + (r - 1)
        config.seed = run_seed
        res = solver.solve(run_index=r)
        results.append(res)

        if conv_csv:
            with open(conv_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for g, nv, td in res.convergence:
                    if g % args.convergence_every == 0 or g == 1:
                        w.writerow([res.instance, r, run_seed, g, nv, f"{td:.4f}", f"{2000.0 * nv + td:.4f}"])

        print(
            f"Run {r:02d}/{args.runs:02d} (seed={run_seed}): "
            f"NV={res.nv}, TD={res.td:.2f}, Cost={res.cost:.2f}, "
            f"Feasible={res.feasible} ({res.elapsed_seconds:.2f}s)"
        )
        if not res.feasible:
            print(f"   --> Violation note: {res.violation_note}")

        if out_csv:
            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    res.instance,
                    res.algo,
                    res.run,
                    res.seed,
                    res.nv,
                    f"{res.td:.2f}",
                    f"{res.cost:.2f}",
                    res.feasible,
                    res.violation_note,
                    f"{res.elapsed_seconds:.2f}",
                ])

        if out_dir:
            sol_path = out_dir / f"{problem.name}_run_{r:02d}_seed_{run_seed}.txt"
            with open(sol_path, "w", encoding="utf-8") as f:
                f.write(f"Instance: {problem.name}\n")
                f.write(f"Run: {r}\n")
                f.write(f"Seed: {run_seed}\n")
                f.write(f"NV: {res.nv}\n")
                f.write(f"TD: {res.td:.2f}\n")
                f.write(f"Cost: {res.cost:.2f}\n")
                f.write(f"Feasible: {res.feasible}\n")
                f.write(f"Violation: {res.violation_note}\n")
                f.write(f"Elapsed: {res.elapsed_seconds:.2f}s\n")
                f.write("Routes:\n")
                for idx, route in enumerate(res.best_solution.routes):
                    f.write(f"Route {idx+1}: 0 {' '.join(map(str, route))} 0\n")

    if results:
        avg_nv = sum(r.nv for r in results) / len(results)
        avg_td = sum(r.td for r in results) / len(results)
        feas_count = sum(1 for r in results if r.feasible)
        print("-" * 60)
        print(f"Summary for {problem.name} ({len(results)} runs):")
        print(f"  Mean NV: {avg_nv:.2f}")
        print(f"  Mean TD: {avg_td:.2f}")
        print(f"  Feasible runs: {feas_count}/{len(results)} ({feas_count/len(results)*100:.1f}%)")
        print("-" * 60)


if __name__ == "__main__":
    main()
