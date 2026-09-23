from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "tests" / "paper_performance_targets.csv"

DEFAULT_INSTANCES = [
    "RCdp1001",
    "RCdp5001",
    "RCdp5007",
    "RCdp5004",
    "RCdp101",
    "Cdp103",
    "Rcdp205",
    "Rdp210",
    "Rcdp207",
    "Rcdp202",
    "Rdp103",
    "cdp104",
    "cdp102",
    "rcdp205",
    "rdp203",
    "rcdp104",
]

PAPER_EXTRA_FLAGS = [
    "--pruning",
    "--O_1_eval",
    "--two_opt",
    "--two_opt_star",
    "--or_opt",
    "2",
    "--two_exchange",
    "2",
    "--related_removal",
    "--regret_insertion",
]

VEHICLE_WEIGHT = 2000.0


@dataclass
class RunRecord:
    seed: int
    returncode: int
    vehicles: int | None = None
    distance: float | None = None
    total_cost: float | None = None
    log_tail: str = ""
    error: str = ""


@dataclass
class InstanceSummary:
    instance: str
    paper_vehicles: int
    paper_distance: float
    paper_total_cost: float
    requested_runs: int
    successful_runs: int
    failed_runs: int
    avg_vehicles: float | None
    avg_distance: float | None
    avg_total_cost: float | None
    best_seed: int | None
    best_vehicles: int | None
    best_distance: float | None
    best_total_cost: float | None
    avg_gap_distance_pct: float | None
    avg_gap_total_cost_pct: float | None
    best_gap_distance_pct: float | None
    best_gap_total_cost_pct: float | None
    best_meets_paper: bool | None
    failure_messages: list[str]


def load_targets(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def resolve_instances(requested: list[str], targets: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(requested) == 1 and requested[0].strip().lower() == "all":
        return targets

    by_name = {row["instance"].strip().lower(): row for row in targets}
    selected: list[dict[str, str]] = []
    missing: list[str] = []
    for item in requested:
        row = by_name.get(item.strip().lower())
        if row is None:
            missing.append(item)
            continue
        selected.append(row)

    if missing:
        raise SystemExit(
            "Unknown instance name(s): %s\nAvailable examples: %s"
            % (", ".join(missing), ", ".join(row["instance"] for row in targets[:8]))
        )

    return selected


def parse_solution_file(path: Path) -> tuple[int, float, float]:
    text = path.read_text(encoding="utf-8")
    vehicle_marker = "vehicle (route) number:"
    cost_marker = "Total cost:"

    if vehicle_marker not in text or cost_marker not in text:
        raise ValueError(f"Could not parse solution file: {path}")

    vehicle_line = next(line for line in text.splitlines() if vehicle_marker in line)
    cost_line = next(line for line in text.splitlines() if cost_marker in line)

    vehicles = int(vehicle_line.split(vehicle_marker, 1)[1].strip())
    total_cost = float(cost_line.split(cost_marker, 1)[1].strip())
    distance = total_cost - vehicles * VEHICLE_WEIGHT
    return vehicles, distance, total_cost


def tail_text(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[-limit:]


def build_solver_command(args, problem_path: Path, seed: int, output_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "src",
        "--problem",
        str(problem_path),
        "--runs",
        "1",
        "--pop_size",
        str(args.pop_size),
        "--max_iter",
        str(args.max_iter),
        "--workers",
        str(args.workers),
        "--hybrid_mode",
        args.hybrid_mode,
        "--local_search_interval",
        str(args.local_search_interval),
        "--stagnation_interval",
        str(args.stagnation_interval),
        "--diversify_ratio",
        str(args.diversify_ratio),
        "--sho_mutation_prob",
        str(args.sho_mutation_prob),
        "--init",
        args.init,
        "--elo",
        str(args.elo),
        "--removal_lower",
        str(args.removal_lower),
        "--removal_upper",
        str(args.removal_upper),
        "--random_seed",
        str(seed),
        "--output",
        str(output_path),
        "--compute_backend",
        args.compute_backend,
    ]

    if args.paper_flags:
        cmd.extend(PAPER_EXTRA_FLAGS)

    if args.time is not None:
        cmd.extend(["--time", str(args.time)])

    if args.solver_extra:
        cmd.extend(shlex.split(args.solver_extra))

    return cmd


def run_solver(cmd: list[str], timeout: int, log_path: Path) -> tuple[int | None, str]:
    try:
        with log_path.open("w", encoding="utf-8") as log_fp:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            return result.returncode, ""
    except subprocess.TimeoutExpired:
        return None, tail_text(log_path)


def summarize_instance(row: dict[str, str], args) -> InstanceSummary:
    instance_name = row["instance"].strip()
    problem_path = (ROOT / row["path"]).resolve()

    paper_vehicles = int(row["target_vehicles"])
    paper_distance = float(row["target_distance"])
    paper_total_cost = paper_vehicles * VEHICLE_WEIGHT + paper_distance

    run_records: list[RunRecord] = []
    failure_messages: list[str] = []

    for run_index in range(args.repeats):
        seed = args.seed_start + run_index
        with tempfile.TemporaryDirectory(prefix=f"{instance_name}_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            output_path = tmp_dir_path / "solution.txt"
            log_path = tmp_dir_path / "run.log"
            cmd = build_solver_command(args, problem_path, seed, output_path)
            returncode, log_tail = run_solver(cmd, args.timeout, log_path)

            if returncode is None:
                message = (
                    f"{instance_name} seed {seed} timed out after {args.timeout} seconds.\n"
                    f"{log_tail[-2000:]}"
                )
                failure_messages.append(message)
                run_records.append(RunRecord(seed=seed, returncode=-1, error="timeout", log_tail=log_tail))
                continue

            if returncode != 0:
                message = (
                    f"{instance_name} seed {seed} exited {returncode}.\n"
                    f"{log_tail[-2000:]}"
                )
                failure_messages.append(message)
                run_records.append(RunRecord(seed=seed, returncode=returncode, error="nonzero", log_tail=log_tail))
                continue

            if not output_path.exists():
                message = (
                    f"{instance_name} seed {seed} finished successfully but did not write {output_path}.\n"
                    f"{log_tail[-2000:]}"
                )
                failure_messages.append(message)
                run_records.append(RunRecord(seed=seed, returncode=0, error="missing_output", log_tail=log_tail))
                continue

            try:
                vehicles, distance, total_cost = parse_solution_file(output_path)
            except Exception as exc:  # noqa: BLE001
                message = (
                    f"{instance_name} seed {seed} produced an unreadable solution file: {exc}.\n"
                    f"{log_tail[-2000:]}"
                )
                failure_messages.append(message)
                run_records.append(RunRecord(seed=seed, returncode=0, error="parse_error", log_tail=log_tail))
                continue

            run_records.append(
                RunRecord(
                    seed=seed,
                    returncode=0,
                    vehicles=vehicles,
                    distance=distance,
                    total_cost=total_cost,
                )
            )

    successes = [record for record in run_records if record.returncode == 0 and record.total_cost is not None]
    successful_runs = len(successes)
    failed_runs = len(run_records) - successful_runs

    if successful_runs == 0:
        return InstanceSummary(
            instance=instance_name,
            paper_vehicles=paper_vehicles,
            paper_distance=paper_distance,
            paper_total_cost=paper_total_cost,
            requested_runs=args.repeats,
            successful_runs=0,
            failed_runs=failed_runs,
            avg_vehicles=None,
            avg_distance=None,
            avg_total_cost=None,
            best_seed=None,
            best_vehicles=None,
            best_distance=None,
            best_total_cost=None,
            avg_gap_distance_pct=None,
            avg_gap_total_cost_pct=None,
            best_gap_distance_pct=None,
            best_gap_total_cost_pct=None,
            best_meets_paper=None,
            failure_messages=failure_messages,
        )

    avg_vehicles = mean(record.vehicles for record in successes if record.vehicles is not None)
    avg_distance = mean(record.distance for record in successes if record.distance is not None)
    avg_total_cost = mean(record.total_cost for record in successes if record.total_cost is not None)

    best = min(
        successes,
        key=lambda record: (
            record.vehicles if record.vehicles is not None else sys.maxsize,
            record.distance if record.distance is not None else float("inf"),
            record.total_cost if record.total_cost is not None else float("inf"),
        ),
    )

    def pct_gap(value: float, baseline: float) -> float:
        if baseline == 0:
            return 0.0
        return (value - baseline) / baseline * 100.0

    best_meets_paper = bool(
        best.vehicles is not None
        and best.distance is not None
        and best.vehicles <= paper_vehicles
        and best.distance <= paper_distance
    )

    return InstanceSummary(
        instance=instance_name,
        paper_vehicles=paper_vehicles,
        paper_distance=paper_distance,
        paper_total_cost=paper_total_cost,
        requested_runs=args.repeats,
        successful_runs=successful_runs,
        failed_runs=failed_runs,
        avg_vehicles=avg_vehicles,
        avg_distance=avg_distance,
        avg_total_cost=avg_total_cost,
        best_seed=best.seed,
        best_vehicles=best.vehicles,
        best_distance=best.distance,
        best_total_cost=best.total_cost,
        avg_gap_distance_pct=pct_gap(avg_distance, paper_distance),
        avg_gap_total_cost_pct=pct_gap(avg_total_cost, paper_total_cost),
        best_gap_distance_pct=pct_gap(best.distance, paper_distance) if best.distance is not None else None,
        best_gap_total_cost_pct=pct_gap(best.total_cost, paper_total_cost) if best.total_cost is not None else None,
        best_meets_paper=best_meets_paper,
        failure_messages=failure_messages,
    )


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}%"


def render_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    lines = [format_row(headers), format_row(["-" * width for width in widths])]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def write_csv(path: Path, summaries: list[InstanceSummary]) -> None:
    fieldnames = [
        "instance",
        "paper_vehicles",
        "paper_distance",
        "paper_total_cost",
        "requested_runs",
        "successful_runs",
        "failed_runs",
        "avg_vehicles",
        "avg_distance",
        "avg_total_cost",
        "best_seed",
        "best_vehicles",
        "best_distance",
        "best_total_cost",
        "avg_gap_distance_pct",
        "avg_gap_total_cost_pct",
        "best_gap_distance_pct",
        "best_gap_total_cost_pct",
        "best_meets_paper",
    ]

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    "instance": item.instance,
                    "paper_vehicles": item.paper_vehicles,
                    "paper_distance": f"{item.paper_distance:.2f}",
                    "paper_total_cost": f"{item.paper_total_cost:.2f}",
                    "requested_runs": item.requested_runs,
                    "successful_runs": item.successful_runs,
                    "failed_runs": item.failed_runs,
                    "avg_vehicles": fmt_float(item.avg_vehicles),
                    "avg_distance": fmt_float(item.avg_distance),
                    "avg_total_cost": fmt_float(item.avg_total_cost),
                    "best_seed": item.best_seed if item.best_seed is not None else "",
                    "best_vehicles": item.best_vehicles if item.best_vehicles is not None else "",
                    "best_distance": fmt_float(item.best_distance),
                    "best_total_cost": fmt_float(item.best_total_cost),
                    "avg_gap_distance_pct": fmt_pct(item.avg_gap_distance_pct),
                    "avg_gap_total_cost_pct": fmt_pct(item.avg_gap_total_cost_pct),
                    "best_gap_distance_pct": fmt_pct(item.best_gap_distance_pct),
                    "best_gap_total_cost_pct": fmt_pct(item.best_gap_total_cost_pct),
                    "best_meets_paper": item.best_meets_paper if item.best_meets_paper is not None else "",
                }
            )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a batch of solver tests for selected paper instances and compare "
            "avg/best results against the paper targets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--instances",
        nargs="*",
        default=DEFAULT_INSTANCES,
        help='Paper instance names to run, or "all" for the full target file.',
    )
    parser.add_argument("--paper-targets", type=Path, default=TARGETS, help="CSV file with paper targets.")
    parser.add_argument("--repeats", type=int, default=36, help="Number of solver runs per instance.")
    parser.add_argument("--seed_start", type=int, default=42, help="First random seed to use.")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout per solver run, in seconds.")
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional CSV report path.")
    parser.add_argument(
        "--solver_extra",
        default="",
        help="Extra raw CLI flags appended to every solver invocation.",
    )
    parser.add_argument("--pop_size", type=int, default=36)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hybrid_mode", default="ph_showoa")
    parser.add_argument("--compute_backend", default="auto")
    parser.add_argument("--local_search_interval", type=int, default=25)
    parser.add_argument("--stagnation_interval", type=int, default=50)
    parser.add_argument("--diversify_ratio", type=float, default=0.40)
    parser.add_argument("--sho_mutation_prob", type=float, default=0.35)
    parser.add_argument("--init", default="rcrs")
    parser.add_argument("--elo", type=int, default=1)
    parser.add_argument("--removal_lower", type=float, default=0.25)
    parser.add_argument("--removal_upper", type=float, default=0.40)
    parser.add_argument("--time", type=int, default=None)
    parser.add_argument(
        "--paper_flags",
        dest="paper_flags",
        action="store_true",
        default=True,
        help="Inject the paper-style extra flags used by the repository benchmark.",
    )
    parser.add_argument(
        "--no_paper_flags",
        dest="paper_flags",
        action="store_false",
        help="Skip the built-in paper-style extra flags.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    targets = load_targets(args.paper_targets)
    selected = resolve_instances(args.instances, targets)

    summaries: list[InstanceSummary] = []
    for row in selected:
        summary = summarize_instance(row, args)
        summaries.append(summary)

    table_rows: list[list[str]] = []
    for item in summaries:
        table_rows.append(
            [
                item.instance,
                f"{item.paper_vehicles}/{item.paper_distance:.2f}",
                f"{fmt_float(item.avg_vehicles)}/{fmt_float(item.avg_distance)}/{fmt_float(item.avg_total_cost)}",
                f"{fmt_float(item.best_vehicles, 0 if item.best_vehicles is not None else 2)}/{fmt_float(item.best_distance)}/{fmt_float(item.best_total_cost)}",
                f"{fmt_pct(item.avg_gap_distance_pct)}/{fmt_pct(item.avg_gap_total_cost_pct)}",
                f"{fmt_pct(item.best_gap_distance_pct)}/{fmt_pct(item.best_gap_total_cost_pct)}",
                str(item.best_seed) if item.best_seed is not None else "-",
                f"{item.successful_runs}/{item.requested_runs}",
                "yes" if item.best_meets_paper else ("-" if item.best_meets_paper is None else "no"),
            ]
        )

    headers = [
        "instance",
        "paper veh/dist",
        "avg veh/dist/tot",
        "best veh/dist/tot",
        "avg gap dist/tot",
        "best gap dist/tot",
        "best seed",
        "ok/runs",
        "meets paper",
    ]

    print(render_table(table_rows, headers))

    total_runs = sum(item.requested_runs for item in summaries)
    total_success = sum(item.successful_runs for item in summaries)
    avg_gap_distance = [item.avg_gap_distance_pct for item in summaries if item.avg_gap_distance_pct is not None]
    avg_gap_total = [item.avg_gap_total_cost_pct for item in summaries if item.avg_gap_total_cost_pct is not None]
    best_gap_distance = [item.best_gap_distance_pct for item in summaries if item.best_gap_distance_pct is not None]
    best_gap_total = [item.best_gap_total_cost_pct for item in summaries if item.best_gap_total_cost_pct is not None]

    print()
    print("Summary")
    print("  instances       :", len(summaries))
    print("  runs succeeded  :", f"{total_success}/{total_runs}")
    print("  mean avg gap TD :", fmt_pct(mean(avg_gap_distance) if avg_gap_distance else None))
    print("  mean avg gap TC :", fmt_pct(mean(avg_gap_total) if avg_gap_total else None))
    print("  mean best gap TD:", fmt_pct(mean(best_gap_distance) if best_gap_distance else None))
    print("  mean best gap TC:", fmt_pct(mean(best_gap_total) if best_gap_total else None))

    failures = [item for item in summaries if item.failed_runs > 0]
    if failures:
        print()
        print("Failures")
        for item in failures:
            for message in item.failure_messages:
                print(f"- {message}")

    if args.output_csv is not None:
        write_csv(args.output_csv, summaries)
        print()
        print(f"CSV written to {args.output_csv}")

    return 0 if total_success == total_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
