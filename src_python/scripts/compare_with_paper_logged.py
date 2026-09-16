from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from shutil import copy2
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "tests" / "paper_performance_targets.csv"
DEFAULT_RESULTS_ROOT = ROOT / "batch_results"
VEHICLE_WEIGHT = 2000.0

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


@dataclass
class RunRecord:
    seed: int
    returncode: int
    duration_sec: float
    status: str
    vehicles: int | None = None
    distance: float | None = None
    total_cost: float | None = None
    command: str = ""
    run_dir: str = ""
    solution_path: str = ""
    log_path: str = ""
    log_tail: str = ""
    error: str = ""


@dataclass
class InstanceSummary:
    instance: str
    result_dir: str
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


@dataclass
class BatchProgress:
    started_at: float
    total_instances: int
    total_runs: int
    completed_instances: int = 0
    completed_runs: int = 0


def slugify(text: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip()]
    slug = "".join(cleaned).strip("_")
    return slug or "instance"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimate_remaining(completed: int, total: int, elapsed: float) -> float | None:
    if completed <= 0:
        return None
    if completed >= total:
        return 0.0
    return (elapsed / completed) * (total - completed)


def log(message: str = "") -> None:
    print(message, flush=True)


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
        else:
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


def run_solver(cmd: list[str], timeout: int, log_path: Path) -> tuple[int | None, float, str]:
    started = time.perf_counter()
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
        return result.returncode, time.perf_counter() - started, tail_text(log_path)
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - started, tail_text(log_path)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_run_manifest(instance_dir: Path, run_records: list[RunRecord]) -> None:
    rows = []
    for record in run_records:
        rows.append(
            {
                "seed": record.seed,
                "status": record.status,
                "returncode": record.returncode,
                "duration_sec": f"{record.duration_sec:.4f}",
                "vehicles": "" if record.vehicles is None else record.vehicles,
                "distance": "" if record.distance is None else f"{record.distance:.2f}",
                "total_cost": "" if record.total_cost is None else f"{record.total_cost:.2f}",
                "error": record.error,
                "run_dir": record.run_dir,
                "solution_path": record.solution_path,
                "log_path": record.log_path,
                "command": record.command,
            }
        )

    fieldnames = [
        "seed",
        "status",
        "returncode",
        "duration_sec",
        "vehicles",
        "distance",
        "total_cost",
        "error",
        "run_dir",
        "solution_path",
        "log_path",
        "command",
    ]
    write_csv(instance_dir / "runs.csv", rows, fieldnames)
    write_json(instance_dir / "runs.json", [asdict(item) for item in run_records])


def save_instance_summary(
    instance_dir: Path,
    summary: InstanceSummary,
    run_records: list[RunRecord],
    best_record: RunRecord | None,
) -> None:
    write_json(
        instance_dir / "summary.json",
        {
            "summary": asdict(summary),
            "runs": [asdict(item) for item in run_records],
        },
    )

    summary_row = {
        "instance": summary.instance,
        "paper_vehicles": summary.paper_vehicles,
        "paper_distance": f"{summary.paper_distance:.2f}",
        "paper_total_cost": f"{summary.paper_total_cost:.2f}",
        "requested_runs": summary.requested_runs,
        "successful_runs": summary.successful_runs,
        "failed_runs": summary.failed_runs,
        "avg_vehicles": "" if summary.avg_vehicles is None else f"{summary.avg_vehicles:.2f}",
        "avg_distance": "" if summary.avg_distance is None else f"{summary.avg_distance:.2f}",
        "avg_total_cost": "" if summary.avg_total_cost is None else f"{summary.avg_total_cost:.2f}",
        "best_seed": "" if summary.best_seed is None else summary.best_seed,
        "best_vehicles": "" if summary.best_vehicles is None else summary.best_vehicles,
        "best_distance": "" if summary.best_distance is None else f"{summary.best_distance:.2f}",
        "best_total_cost": "" if summary.best_total_cost is None else f"{summary.best_total_cost:.2f}",
        "avg_gap_distance_pct": "" if summary.avg_gap_distance_pct is None else f"{summary.avg_gap_distance_pct:+.2f}%",
        "avg_gap_total_cost_pct": "" if summary.avg_gap_total_cost_pct is None else f"{summary.avg_gap_total_cost_pct:+.2f}%",
        "best_gap_distance_pct": "" if summary.best_gap_distance_pct is None else f"{summary.best_gap_distance_pct:+.2f}%",
        "best_gap_total_cost_pct": "" if summary.best_gap_total_cost_pct is None else f"{summary.best_gap_total_cost_pct:+.2f}%",
        "best_meets_paper": "" if summary.best_meets_paper is None else summary.best_meets_paper,
        "result_dir": summary.result_dir,
    }
    write_csv(instance_dir / "summary.csv", [summary_row], list(summary_row.keys()))

    summary_text = [
        f"instance: {summary.instance}",
        f"result_dir: {summary.result_dir}",
        f"paper: vehicles={summary.paper_vehicles} distance={summary.paper_distance:.2f} total={summary.paper_total_cost:.2f}",
        f"avg: vehicles={summary.avg_vehicles if summary.avg_vehicles is not None else '-'} distance={summary.avg_distance if summary.avg_distance is not None else '-'} total={summary.avg_total_cost if summary.avg_total_cost is not None else '-'}",
        f"best: seed={summary.best_seed if summary.best_seed is not None else '-'} vehicles={summary.best_vehicles if summary.best_vehicles is not None else '-'} distance={summary.best_distance if summary.best_distance is not None else '-'} total={summary.best_total_cost if summary.best_total_cost is not None else '-'}",
        f"gaps: avg_dist={summary.avg_gap_distance_pct if summary.avg_gap_distance_pct is not None else '-'} avg_total={summary.avg_gap_total_cost_pct if summary.avg_gap_total_cost_pct is not None else '-'} best_dist={summary.best_gap_distance_pct if summary.best_gap_distance_pct is not None else '-'} best_total={summary.best_gap_total_cost_pct if summary.best_gap_total_cost_pct is not None else '-'}",
        f"meets_paper: {summary.best_meets_paper}",
    ]
    (instance_dir / "summary.txt").write_text("\n".join(summary_text) + "\n", encoding="utf-8")

    if best_record is not None:
        best_solution = Path(best_record.solution_path)
        best_log = Path(best_record.log_path)
        if best_solution.exists():
            copy2(best_solution, instance_dir / "best_solution.txt")
        if best_log.exists():
            copy2(best_log, instance_dir / "best_log.txt")


def summarize_instance(
    row: dict[str, str],
    args,
    index: int,
    total_instances: int,
    results_root: Path,
    progress: BatchProgress,
) -> InstanceSummary:
    instance_name = row["instance"].strip()
    problem_path = (ROOT / row["path"]).resolve()
    instance_dir = results_root / f"{index:02d}_{slugify(instance_name)}"
    runs_dir = instance_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    paper_vehicles = int(row["target_vehicles"])
    paper_distance = float(row["target_distance"])
    paper_total_cost = paper_vehicles * VEHICLE_WEIGHT + paper_distance

    log(
        f"[{index}/{total_instances}] {instance_name}: start | paper NV={paper_vehicles} TD={paper_distance:.2f} | input={problem_path}"
    )

    instance_started = time.perf_counter()
    run_records: list[RunRecord] = []

    for run_index in range(1, args.repeats + 1):
        seed = args.seed_start + run_index - 1
        run_dir = runs_dir / f"run_{run_index:02d}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        solution_path = run_dir / "solution.txt"
        log_path = run_dir / "solver.log"

        cmd = build_solver_command(args, problem_path, seed, solution_path)
        command_text = subprocess.list2cmdline([str(part) for part in cmd])
        (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

        log(f"  [{index}/{total_instances}] run {run_index}/{args.repeats} seed={seed} -> {run_dir}")
        returncode, duration, log_tail = run_solver(cmd, args.timeout, log_path)

        status = "ok"
        vehicles = None
        distance = None
        total_cost = None
        error = ""

        if returncode is None:
            status = "timeout"
            error = f"timed out after {args.timeout} seconds"
        elif returncode != 0:
            status = "failed"
            error = f"exit code {returncode}"
        elif not solution_path.exists():
            status = "missing_output"
            error = "solver did not write solution file"
        else:
            try:
                vehicles, distance, total_cost = parse_solution_file(solution_path)
            except Exception as exc:  # noqa: BLE001
                status = "parse_error"
                error = str(exc)

        record = RunRecord(
            seed=seed,
            returncode=-1 if returncode is None else returncode,
            duration_sec=duration,
            status=status,
            vehicles=vehicles,
            distance=distance,
            total_cost=total_cost,
            command=command_text,
            run_dir=str(run_dir),
            solution_path=str(solution_path),
            log_path=str(log_path),
            log_tail=log_tail[-2000:],
            error=error,
        )
        run_records.append(record)
        save_run_manifest(instance_dir, run_records)
        write_json(run_dir / "run.json", asdict(record))

        progress.completed_runs += 1
        instance_elapsed = time.perf_counter() - instance_started
        batch_elapsed = time.perf_counter() - progress.started_at
        instance_eta = estimate_remaining(len(run_records), args.repeats, instance_elapsed)
        batch_eta = estimate_remaining(progress.completed_runs, progress.total_runs, batch_elapsed)

        if status == "ok":
            log(
                f"    done in {format_duration(duration)} | NV={vehicles} TD={distance:.2f} TC={total_cost:.2f} | instance ETA {format_duration(instance_eta)} | batch ETA {format_duration(batch_eta)}"
            )
        else:
            log(
                f"    {status} in {format_duration(duration)} | {error} | instance ETA {format_duration(instance_eta)} | batch ETA {format_duration(batch_eta)}"
            )
            if log_tail.strip():
                tail_lines = log_tail.strip().splitlines()[-10:]
                log("    log tail:")
                for line in tail_lines:
                    log(f"      {line}")

    successes = [record for record in run_records if record.status == "ok" and record.vehicles is not None]
    successful_runs = len(successes)
    failed_runs = len(run_records) - successful_runs

    if successful_runs == 0:
        summary = InstanceSummary(
            instance=instance_name,
            result_dir=str(instance_dir),
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
        )
        save_instance_summary(instance_dir, summary, run_records, None)
        return summary

    avg_vehicles = mean(record.vehicles for record in successes if record.vehicles is not None)
    avg_distance = mean(record.distance for record in successes if record.distance is not None)
    avg_total_cost = mean(record.total_cost for record in successes if record.total_cost is not None)

    best_record = min(
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

    summary = InstanceSummary(
        instance=instance_name,
        result_dir=str(instance_dir),
        paper_vehicles=paper_vehicles,
        paper_distance=paper_distance,
        paper_total_cost=paper_total_cost,
        requested_runs=args.repeats,
        successful_runs=successful_runs,
        failed_runs=failed_runs,
        avg_vehicles=avg_vehicles,
        avg_distance=avg_distance,
        avg_total_cost=avg_total_cost,
        best_seed=best_record.seed,
        best_vehicles=best_record.vehicles,
        best_distance=best_record.distance,
        best_total_cost=best_record.total_cost,
        avg_gap_distance_pct=pct_gap(avg_distance, paper_distance),
        avg_gap_total_cost_pct=pct_gap(avg_total_cost, paper_total_cost),
        best_gap_distance_pct=pct_gap(best_record.distance, paper_distance) if best_record.distance is not None else None,
        best_gap_total_cost_pct=pct_gap(best_record.total_cost, paper_total_cost) if best_record.total_cost is not None else None,
        best_meets_paper=bool(
            best_record.vehicles is not None
            and best_record.distance is not None
            and best_record.vehicles <= paper_vehicles
            and best_record.distance <= paper_distance
        ),
    )
    save_instance_summary(instance_dir, summary, run_records, best_record)
    log(
        f"[{index}/{total_instances}] {instance_name}: avg NV={avg_vehicles:.2f} TD={avg_distance:.2f} TC={avg_total_cost:.2f} | best seed={best_record.seed} NV={best_record.vehicles} TD={best_record.distance:.2f} TC={best_record.total_cost:.2f} | saved -> {instance_dir}"
    )
    return summary


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


def write_batch_reports(results_root: Path, summaries: list[InstanceSummary]) -> None:
    rows = []
    for item in summaries:
        rows.append(
            {
                "instance": item.instance,
                "result_dir": item.result_dir,
                "paper_vehicles": item.paper_vehicles,
                "paper_distance": f"{item.paper_distance:.2f}",
                "paper_total_cost": f"{item.paper_total_cost:.2f}",
                "requested_runs": item.requested_runs,
                "successful_runs": item.successful_runs,
                "failed_runs": item.failed_runs,
                "avg_vehicles": "" if item.avg_vehicles is None else f"{item.avg_vehicles:.2f}",
                "avg_distance": "" if item.avg_distance is None else f"{item.avg_distance:.2f}",
                "avg_total_cost": "" if item.avg_total_cost is None else f"{item.avg_total_cost:.2f}",
                "best_seed": "" if item.best_seed is None else item.best_seed,
                "best_vehicles": "" if item.best_vehicles is None else item.best_vehicles,
                "best_distance": "" if item.best_distance is None else f"{item.best_distance:.2f}",
                "best_total_cost": "" if item.best_total_cost is None else f"{item.best_total_cost:.2f}",
                "avg_gap_distance_pct": "" if item.avg_gap_distance_pct is None else f"{item.avg_gap_distance_pct:+.2f}%",
                "avg_gap_total_cost_pct": "" if item.avg_gap_total_cost_pct is None else f"{item.avg_gap_total_cost_pct:+.2f}%",
                "best_gap_distance_pct": "" if item.best_gap_distance_pct is None else f"{item.best_gap_distance_pct:+.2f}%",
                "best_gap_total_cost_pct": "" if item.best_gap_total_cost_pct is None else f"{item.best_gap_total_cost_pct:+.2f}%",
                "best_meets_paper": "" if item.best_meets_paper is None else item.best_meets_paper,
            }
        )

    fieldnames = list(rows[0].keys()) if rows else ["instance"]
    write_csv(results_root / "batch_summary.csv", rows, fieldnames)
    write_json(results_root / "batch_summary.json", [asdict(item) for item in summaries])


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a batch of solver tests for selected paper instances, save per-instance artifacts, "
            "and compare avg/best results against the paper targets."
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
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Directory where batch and per-instance artifacts are stored.",
    )
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional extra CSV report path.")
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

    if args.results_dir is None:
        results_root = DEFAULT_RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        results_root = args.results_dir
    results_root.mkdir(parents=True, exist_ok=True)

    total_instances = len(selected)
    total_runs = total_instances * args.repeats
    progress = BatchProgress(started_at=time.perf_counter(), total_instances=total_instances, total_runs=total_runs)

    log(f"[batch] results dir: {results_root}")
    log(f"[batch] instances: {total_instances} | repeats: {args.repeats} | total runs: {total_runs}")
    log(
        f"[batch] solver: pop_size={args.pop_size}, max_iter={args.max_iter}, workers={args.workers}, backend={args.compute_backend}, hybrid_mode={args.hybrid_mode}"
    )
    log(f"[batch] paper flags: {'on' if args.paper_flags else 'off'}")
    if args.output_csv is not None:
        log(f"[batch] extra CSV: {args.output_csv}")

    summaries: list[InstanceSummary] = []
    for index, row in enumerate(selected, start=1):
        summary = summarize_instance(row, args, index, total_instances, results_root, progress)
        summaries.append(summary)
        progress.completed_instances += 1
        batch_elapsed = time.perf_counter() - progress.started_at
        batch_eta = estimate_remaining(progress.completed_instances, total_instances, batch_elapsed)
        log(
            f"[batch] completed {progress.completed_instances}/{total_instances} instances | elapsed {format_duration(batch_elapsed)} | ETA {format_duration(batch_eta)}"
        )

    write_batch_reports(results_root, summaries)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        copy2(results_root / "batch_summary.csv", args.output_csv)

    table_rows: list[list[str]] = []
    for item in summaries:
        table_rows.append(
            [
                item.instance,
                f"{item.paper_vehicles}/{item.paper_distance:.2f}",
                f"{item.avg_vehicles:.2f}/{item.avg_distance:.2f}/{item.avg_total_cost:.2f}" if item.avg_vehicles is not None else "-",
                f"{item.best_vehicles}/{item.best_distance:.2f}/{item.best_total_cost:.2f}" if item.best_vehicles is not None else "-",
                f"{item.avg_gap_distance_pct:+.2f}%/{item.avg_gap_total_cost_pct:+.2f}%" if item.avg_gap_distance_pct is not None else "-",
                f"{item.best_gap_distance_pct:+.2f}%/{item.best_gap_total_cost_pct:+.2f}%" if item.best_gap_distance_pct is not None else "-",
                str(item.best_seed) if item.best_seed is not None else "-",
                f"{item.successful_runs}/{item.requested_runs}",
                "yes" if item.best_meets_paper else ("-" if item.best_meets_paper is None else "no"),
            ]
        )

    log()
    log(render_table(
        table_rows,
        [
            "instance",
            "paper veh/dist",
            "avg veh/dist/tot",
            "best veh/dist/tot",
            "avg gap dist/tot",
            "best gap dist/tot",
            "best seed",
            "ok/runs",
            "meets paper",
        ],
    ))

    total_success = sum(item.successful_runs for item in summaries)
    avg_gap_distance = [item.avg_gap_distance_pct for item in summaries if item.avg_gap_distance_pct is not None]
    avg_gap_total = [item.avg_gap_total_cost_pct for item in summaries if item.avg_gap_total_cost_pct is not None]
    best_gap_distance = [item.best_gap_distance_pct for item in summaries if item.best_gap_distance_pct is not None]
    best_gap_total = [item.best_gap_total_cost_pct for item in summaries if item.best_gap_total_cost_pct is not None]

    log()
    log("Summary")
    log(f"  results dir     : {results_root}")
    log(f"  instances       : {len(summaries)}")
    log(f"  runs succeeded   : {total_success}/{total_runs}")
    log(f"  mean avg gap TD  : {mean(avg_gap_distance):+.2f}%" if avg_gap_distance else "  mean avg gap TD  : -")
    log(f"  mean avg gap TC  : {mean(avg_gap_total):+.2f}%" if avg_gap_total else "  mean avg gap TC  : -")
    log(f"  mean best gap TD : {mean(best_gap_distance):+.2f}%" if best_gap_distance else "  mean best gap TD : -")
    log(f"  mean best gap TC : {mean(best_gap_total):+.2f}%" if best_gap_total else "  mean best gap TC : -")

    return 0 if total_success == total_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
