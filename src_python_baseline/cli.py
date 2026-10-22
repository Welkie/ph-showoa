from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .problem import Problem
from .solution import Solution
from .algorithm1_solver import PHSHOWOASolver, SolverConfig, RunResult


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="PH-SHOWOA Canonical Paper-Aligned Solver")
    parser.add_argument("instance_path", nargs="?", default=None, type=str, help="Path to VRPSDPTW instance file (.txt or .vrpsdptw)")
    parser.add_argument("--problem", "-p", type=str, default=None, help="Alias for instance_path")
    parser.add_argument("--pop_size", type=int, default=30, help="Population size (default: 30)")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs (default: 1)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker threads (default: 1)")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path for run results")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save detailed solution logs")
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

    results = []
    for r in range(1, args.runs + 1):
        run_seed = args.seed + (r - 1)
        config = SolverConfig(
            pop_size=args.pop_size,
            max_iter=args.max_iter,
            seed=run_seed,
            workers=args.workers,
        )
        solver = PHSHOWOASolver(problem, config)
        res = solver.solve(run_index=r)
        results.append(res)

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
