import argparse
import csv
import itertools
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PROBLEM = REPO_ROOT / "data" / "Wang_Chen" / "explicit_rdp101.vrpsdptw"
RESULTS_FILE = REPO_ROOT / "experiment_results.csv"
TARGET_VEHICLES = 19
TARGET_DISTANCE = 1400.0
VEHICLE_WEIGHT = 2000.0


@dataclass(frozen=True)
class Experiment:
    pop_size: int
    g_1: int
    time_limit: int
    seed: int
    max_iter: Optional[int]
    workers: int


@dataclass
class Result:
    vehicles: int
    distance: float
    total_cost: float
    time_consumed: int
    stdout_tail: str
    timed_out: bool = False


def parse_int_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_experiments(args: argparse.Namespace) -> Iterable[Experiment]:
    if args.defensive:
        pop_sizes = [args.defensive_pop_size]
        g_values = [args.defensive_g1]
        time_limits = [args.defensive_time]
        seeds = [args.defensive_seed]
    else:
        pop_sizes = parse_int_list(args.pop_sizes)
        g_values = parse_int_list(args.g_values)
        time_limits = parse_int_list(args.time_limits)
        seeds = parse_int_list(args.seeds)

    products = itertools.product(pop_sizes, g_values, time_limits, seeds)
    for pop_size, g_1, time_limit, seed in products:
        max_iter = args.max_iter if args.max_iter is not None else g_1
        yield Experiment(
            pop_size=pop_size,
            g_1=g_1,
            time_limit=time_limit,
            seed=seed,
            max_iter=max_iter,
            workers=args.workers,
        )


def build_command(problem: Path, exp: Experiment) -> List[str]:
    return [
        sys.executable,
        "-m",
        "src",
        "--problem",
        str(problem),
        "--runs",
        "1",
        "--pop_size",
        str(exp.pop_size),
        "--g_1",
        str(exp.g_1),
        "--max_iter",
        str(exp.max_iter if exp.max_iter is not None else exp.g_1),
        "--time",
        str(exp.time_limit),
        "--random_seed",
        str(exp.seed),
        "--workers",
        str(exp.workers),
    ]


def parse_result(output: str) -> Result:
    vehicle_match = re.search(r"vehicle \(route\) number:\s*(\d+)", output)
    cost_match = re.search(r"Total cost:\s*([0-9]+(?:\.[0-9]+)?)", output)
    time_match = re.search(r"Total\s+\d+\s+runs,\s+total consumed\s+(\d+)\s+sec", output)

    if vehicle_match is None or cost_match is None or time_match is None:
        tail = "\n".join(output.splitlines()[-80:])
        raise ValueError("Could not parse solver output. Tail:\n%s" % tail)

    vehicles = int(vehicle_match.group(1))
    total_cost = float(cost_match.group(1))
    distance = total_cost - vehicles * VEHICLE_WEIGHT
    time_consumed = int(time_match.group(1))
    stdout_tail = "\n".join(output.splitlines()[-40:])
    return Result(
        vehicles=vehicles,
        distance=distance,
        total_cost=total_cost,
        time_consumed=time_consumed,
        stdout_tail=stdout_tail,
    )


def ensure_results_header(path: Path, reset: bool) -> None:
    if reset and path.exists():
        path.unlink()
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "pop_size",
                "g_1",
                "seed",
                "vehicles",
                "distance",
                "total_cost",
                "time_consumed",
            ]
        )


def append_result(path: Path, exp: Experiment, result: Result) -> None:
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                exp.pop_size,
                exp.g_1,
                exp.seed,
                result.vehicles,
                "%.6f" % result.distance,
                "%.6f" % result.total_cost,
                result.time_consumed,
            ]
        )


def load_best(path: Path) -> Optional[Tuple[int, float, float, int, int, int]]:
    if not path.exists():
        return None
    best_row = None
    with path.open("r", newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            vehicles = int(float(row["vehicles"]))
            distance = float(row["distance"])
            total_cost = float(row["total_cost"])
            if vehicles < 0 or math_is_inf(distance) or math_is_inf(total_cost):
                continue
            pop_size = int(row["pop_size"])
            g_1 = int(row["g_1"])
            seed = int(row["seed"])
            candidate = (vehicles, distance, total_cost, pop_size, g_1, seed)
            if best_row is None:
                best_row = candidate
                continue
            if (vehicles, distance, total_cost) < (best_row[0], best_row[1], best_row[2]):
                best_row = candidate
    return best_row


def math_is_inf(value: float) -> bool:
    return value == float("inf") or value == -float("inf")


def kill_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform.startswith("win"):
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    proc.kill()


def run_experiment(problem: Path, exp: Experiment, timeout_margin: int) -> Result:
    command = build_command(problem, exp)
    print("RUN", " ".join('"%s"' % item if " " in item else item for item in command), flush=True)
    timeout = max(exp.time_limit + timeout_margin, timeout_margin)
    proc = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        stdout, stderr = proc.communicate()
        tail = "\n".join((stdout + "\n" + stderr).splitlines()[-80:])
        print("TIMEOUT after %d sec. Tail:\n%s" % (timeout, tail), flush=True)
        return Result(
            vehicles=-1,
            distance=float("inf"),
            total_cost=float("inf"),
            time_consumed=timeout,
            stdout_tail=tail,
            timed_out=True,
        )

    combined = stdout + "\n" + stderr
    if proc.returncode != 0:
        tail = "\n".join(combined.splitlines()[-100:])
        raise RuntimeError("Solver exited with code %d. Tail:\n%s" % (proc.returncode, tail))
    return parse_result(combined)


def is_victory(result: Result) -> bool:
    return result.vehicles == TARGET_VEHICLES and result.distance <= TARGET_DISTANCE


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PH-SHOWOA tuning sweep for explicit_rdp101.")
    parser.add_argument("--problem", default=str(DEFAULT_PROBLEM))
    parser.add_argument("--results", default=str(RESULTS_FILE))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--defensive", action="store_true")
    parser.add_argument("--pop-sizes", default="36,64,100")
    parser.add_argument("--g-values", default="100,200,300")
    parser.add_argument("--time-limits", default="180,300,450")
    parser.add_argument("--seeds", default="42,101,2026")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--timeout-margin", type=int, default=90)
    parser.add_argument("--defensive-pop-size", type=int, default=4)
    parser.add_argument("--defensive-g1", type=int, default=1)
    parser.add_argument("--defensive-time", type=int, default=20)
    parser.add_argument("--defensive-seed", type=int, default=42)
    args = parser.parse_args(argv)

    problem = Path(args.problem)
    results_path = Path(args.results)
    ensure_results_header(results_path, args.reset)

    start = time.perf_counter()
    for run_index, exp in enumerate(build_experiments(args), start=1):
        if args.max_runs > 0 and run_index > args.max_runs:
            break
        result = run_experiment(problem, exp, args.timeout_margin)
        append_result(results_path, exp, result)
        print(
            "RESULT pop_size=%d g_1=%d seed=%d vehicles=%d distance=%.4f total_cost=%.4f time=%d"
            % (
                exp.pop_size,
                exp.g_1,
                exp.seed,
                result.vehicles,
                result.distance,
                result.total_cost,
                result.time_consumed,
            ),
            flush=True,
        )
        if is_victory(result):
            print("VICTORY target reached.", flush=True)
            print(result.stdout_tail)
            return 0

    best = load_best(results_path)
    elapsed = int(time.perf_counter() - start)
    if best is not None:
        vehicles, distance, total_cost, pop_size, g_1, seed = best
        print(
            "BEST_SO_FAR vehicles=%d distance=%.4f total_cost=%.4f pop_size=%d g_1=%d seed=%d elapsed=%d"
            % (vehicles, distance, total_cost, pop_size, g_1, seed, elapsed),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
