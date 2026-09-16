from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

DEFAULT_PROJECT_ROOT = Path("/kaggle/input/datasets/keith1101/ph-code")
DEFAULT_DATA_ROOT = Path("/kaggle/input/datasets/keith1101/ph-showoa")
DEFAULT_WORK_ROOT = Path("/kaggle/working")
DEFAULT_OUTPUT_ROOT = DEFAULT_WORK_ROOT / "outputs"
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
DEFAULT_BACKENDS = ["cpu", "cuda"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper benchmark on Kaggle for both CPU and CUDA backends, "
            "then compare runtime and solution quality."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work_root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--instances",
        nargs="*",
        default=DEFAULT_INSTANCES,
        help='Instance names to run, or "all" for the built-in default list.',
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["cpu", "cuda"],
        default=DEFAULT_BACKENDS,
        help="Backends to benchmark. Use both to compare CPU and GPU.",
    )
    parser.add_argument("--repeats", type=int, default=36, help="Runs per instance for each backend.")
    parser.add_argument("--seed_start", type=int, default=42, help="First random seed to use.")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per solver run, in seconds.")
    parser.add_argument("--pop_size", type=int, default=36)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--local_search_interval", type=int, default=25)
    parser.add_argument("--stagnation_interval", type=int, default=50)
    parser.add_argument("--diversify_ratio", type=float, default=0.40)
    parser.add_argument("--sho_mutation_prob", type=float, default=0.35)
    parser.add_argument("--init", default="rcrs")
    parser.add_argument("--elo", type=int, default=1)
    parser.add_argument("--removal_lower", type=float, default=0.25)
    parser.add_argument("--removal_upper", type=float, default=0.40)
    parser.add_argument("--hybrid_mode", default="ph_showoa", choices=["ph_showoa", "sho", "woa"])
    parser.add_argument("--time", type=int, default=None, help="Optional solver time limit in seconds.")
    parser.add_argument(
        "--paper_flags",
        dest="paper_flags",
        action="store_true",
        default=True,
        help="Inject the paper-style local-search flags used by the repository benchmark.",
    )
    parser.add_argument(
        "--no_paper_flags",
        dest="paper_flags",
        action="store_false",
        help="Skip the built-in paper-style flags.",
    )
    parser.add_argument(
        "--solver_extra",
        default="",
        help="Extra raw CLI flags appended to every solver invocation.",
    )
    parser.add_argument(
        "--copy_project",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the source tree to /kaggle/working before running.",
    )
    parser.add_argument(
        "--stop_on_error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop immediately if one backend or instance fails.",
    )
    return parser.parse_args()


def normalize_instances(values: list[str]) -> list[str]:
    if not values:
        return list(DEFAULT_INSTANCES)
    if len(values) == 1 and values[0].strip().lower() == "all":
        return list(DEFAULT_INSTANCES)
    return values


def detect_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except FileNotFoundError:
        return False
    except Exception:
        return False


def ensure_project_copy(project_root: Path, work_root: Path, copy_project: bool) -> Path:
    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")

    if not copy_project:
        return project_root

    copied_root = work_root / "ph-code"
    copied_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project_root, copied_root, dirs_exist_ok=True)
    return copied_root


def ensure_data_link(project_root: Path, data_root: Path) -> Path:
    """Make benchmark data visible under project_root/data for paper scripts."""
    target_root = data_root
    nested_data = data_root / "data"
    if nested_data.exists() and nested_data.is_dir():
        target_root = nested_data

    link_path = project_root / "data"
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            if link_path.resolve() == target_root.resolve():
                return link_path
            link_path.unlink()
        elif link_path.is_dir():
            if (link_path / "Wang_Chen").exists():
                return link_path
            shutil.rmtree(link_path)
        else:
            link_path.unlink()

    try:
        link_path.symlink_to(target_root, target_is_directory=True)
        return link_path
    except Exception:
        shutil.copytree(target_root, link_path, dirs_exist_ok=True)
        return link_path


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


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tail_text(path: Path, limit: int = 3000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[-limit:]


def stream_command(cmd: list[str], cwd: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return proc.wait()


def find_problem_file(data_root: Path, instance_name: str) -> Path:
    target = instance_name.strip().lower()
    matches: list[Path] = []
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".vrpsdptw"}:
            continue
        if path.stem.lower() == target:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"Cannot find instance '{instance_name}' under {data_root}")
    matches.sort(key=lambda p: (len(str(p)), str(p).lower()))
    return matches[0]


def build_compare_command(
    project_root: Path,
    results_dir: Path,
    instances: list[str],
    backend: str,
    workers: int,
    args: argparse.Namespace,
) -> list[str]:
    script_path = project_root / "scripts" / "compare_with_paper_logged.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--instances",
        *instances,
        "--repeats",
        str(args.repeats),
        "--seed_start",
        str(args.seed_start),
        "--timeout",
        str(args.timeout),
        "--results_dir",
        str(results_dir),
        "--pop_size",
        str(args.pop_size),
        "--max_iter",
        str(args.max_iter),
        "--workers",
        str(workers),
        "--hybrid_mode",
        args.hybrid_mode,
        "--compute_backend",
        backend,
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
    ]
    if args.time is not None:
        cmd.extend(["--time", str(args.time)])
    if not args.paper_flags:
        cmd.append("--no_paper_flags")
    if args.solver_extra:
        cmd.extend(["--solver_extra", args.solver_extra])
    return cmd


def load_backend_report(backend: str, backend_root: Path) -> list[dict[str, object]]:
    batch_summary = backend_root / "batch_summary.json"
    if not batch_summary.exists():
        raise FileNotFoundError(f"Missing backend summary: {batch_summary}")

    summaries = json.loads(batch_summary.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    for item in summaries:
        result_dir = Path(item["result_dir"])
        runs_json = result_dir / "runs.json"
        durations: list[float] = []
        if runs_json.exists():
            run_records = json.loads(runs_json.read_text(encoding="utf-8"))
            for record in run_records:
                if record.get("status") == "ok" and record.get("duration_sec") is not None:
                    durations.append(float(record["duration_sec"]))
        avg_run_sec = mean(durations) if durations else None
        fastest_run_sec = min(durations) if durations else None
        total_success_runtime_sec = sum(durations) if durations else None

        row = {
            "backend": backend,
            "instance": item["instance"],
            "result_dir": item["result_dir"],
            "paper_vehicles": item["paper_vehicles"],
            "paper_distance": item["paper_distance"],
            "paper_total_cost": item["paper_total_cost"],
            "requested_runs": item["requested_runs"],
            "successful_runs": item["successful_runs"],
            "failed_runs": item["failed_runs"],
            "avg_vehicles": item["avg_vehicles"],
            "avg_distance": item["avg_distance"],
            "avg_total_cost": item["avg_total_cost"],
            "best_seed": item["best_seed"],
            "best_vehicles": item["best_vehicles"],
            "best_distance": item["best_distance"],
            "best_total_cost": item["best_total_cost"],
            "avg_gap_distance_pct": item["avg_gap_distance_pct"],
            "avg_gap_total_cost_pct": item["avg_gap_total_cost_pct"],
            "best_gap_distance_pct": item["best_gap_distance_pct"],
            "best_gap_total_cost_pct": item["best_gap_total_cost_pct"],
            "best_meets_paper": item["best_meets_paper"],
            "avg_run_sec": avg_run_sec,
            "fastest_run_sec": fastest_run_sec,
            "total_success_runtime_sec": total_success_runtime_sec,
        }
        rows.append(row)

    return rows


def format_summary_value(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    lines = [fmt(headers), fmt(["-" * width for width in widths])]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def backend_stats(rows: list[dict[str, object]]) -> dict[str, float | None]:
    avg_run = [float(r["avg_run_sec"]) for r in rows if r.get("avg_run_sec") is not None]
    fastest_run = [float(r["fastest_run_sec"]) for r in rows if r.get("fastest_run_sec") is not None]
    avg_gap_td = [float(r["avg_gap_distance_pct"]) for r in rows if r.get("avg_gap_distance_pct") is not None]
    avg_gap_tc = [float(r["avg_gap_total_cost_pct"]) for r in rows if r.get("avg_gap_total_cost_pct") is not None]
    best_gap_td = [float(r["best_gap_distance_pct"]) for r in rows if r.get("best_gap_distance_pct") is not None]
    best_gap_tc = [float(r["best_gap_total_cost_pct"]) for r in rows if r.get("best_gap_total_cost_pct") is not None]
    return {
        "mean_avg_run_sec": mean(avg_run) if avg_run else None,
        "mean_fastest_run_sec": mean(fastest_run) if fastest_run else None,
        "mean_avg_gap_td": mean(avg_gap_td) if avg_gap_td else None,
        "mean_avg_gap_tc": mean(avg_gap_tc) if avg_gap_tc else None,
        "mean_best_gap_td": mean(best_gap_td) if best_gap_td else None,
        "mean_best_gap_tc": mean(best_gap_tc) if best_gap_tc else None,
    }


def write_backend_reports(backend_root: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    fieldnames = [
        "backend",
        "instance",
        "result_dir",
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
        "avg_run_sec",
        "fastest_run_sec",
        "total_success_runtime_sec",
    ]
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                key: format_summary_value(row.get(key), 4 if key.endswith("_sec") else 2)
                for key in fieldnames
            }
        )
    write_csv(backend_root / "backend_summary.csv", csv_rows, fieldnames)
    write_json(backend_root / "backend_summary.json", rows)


def build_comparison_rows(backends: dict[str, list[dict[str, object]]], instance_order: list[str]) -> list[dict[str, object]]:
    by_backend: dict[str, dict[str, dict[str, object]]] = {
        backend: {str(row["instance"]): row for row in rows}
        for backend, rows in backends.items()
    }

    comparison_rows: list[dict[str, object]] = []
    for instance in instance_order:
        cpu = by_backend.get("cpu", {}).get(instance)
        cuda = by_backend.get("cuda", {}).get(instance)
        sample = cpu or cuda
        if sample is None:
            continue

        def get(row, key):
            if row is None:
                return None
            return row.get(key)

        cpu_avg_run = get(cpu, "avg_run_sec")
        cuda_avg_run = get(cuda, "avg_run_sec")
        speedup = None
        if cpu_avg_run is not None and cuda_avg_run not in (None, 0):
            speedup = float(cpu_avg_run) / float(cuda_avg_run)

        cpu_best_total = get(cpu, "best_total_cost")
        cuda_best_total = get(cuda, "best_total_cost")
        winner_time = None
        if cpu_avg_run is not None and cuda_avg_run is not None:
            if abs(float(cpu_avg_run) - float(cuda_avg_run)) < 1e-9:
                winner_time = "tie"
            elif float(cpu_avg_run) < float(cuda_avg_run):
                winner_time = "cpu"
            else:
                winner_time = "cuda"

        winner_cost = None
        if cpu_best_total is not None and cuda_best_total is not None:
            if abs(float(cpu_best_total) - float(cuda_best_total)) < 1e-9:
                winner_cost = "tie"
            elif float(cpu_best_total) < float(cuda_best_total):
                winner_cost = "cpu"
            else:
                winner_cost = "cuda"

        comparison_rows.append(
            {
                "instance": instance,
                "paper_vehicles": sample["paper_vehicles"],
                "paper_distance": sample["paper_distance"],
                "paper_total_cost": sample["paper_total_cost"],
                "cpu_avg_run_sec": cpu_avg_run,
                "cpu_fastest_run_sec": get(cpu, "fastest_run_sec"),
                "cpu_avg_total_cost": get(cpu, "avg_total_cost"),
                "cpu_best_total_cost": cpu_best_total,
                "cpu_avg_gap_total_cost_pct": get(cpu, "avg_gap_total_cost_pct"),
                "cpu_best_gap_total_cost_pct": get(cpu, "best_gap_total_cost_pct"),
                "cuda_avg_run_sec": cuda_avg_run,
                "cuda_fastest_run_sec": get(cuda, "fastest_run_sec"),
                "cuda_avg_total_cost": get(cuda, "avg_total_cost"),
                "cuda_best_total_cost": cuda_best_total,
                "cuda_avg_gap_total_cost_pct": get(cuda, "avg_gap_total_cost_pct"),
                "cuda_best_gap_total_cost_pct": get(cuda, "best_gap_total_cost_pct"),
                "speedup_cpu_over_cuda": speedup,
                "time_diff_sec": (
                    float(cpu_avg_run) - float(cuda_avg_run)
                    if cpu_avg_run is not None and cuda_avg_run is not None
                    else None
                ),
                "best_cost_diff_cuda_minus_cpu": (
                    float(cuda_best_total) - float(cpu_best_total)
                    if cpu_best_total is not None and cuda_best_total is not None
                    else None
                ),
                "faster_backend": winner_time,
                "better_backend_best_cost": winner_cost,
            }
        )

    return comparison_rows


def write_comparison_reports(compare_root: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    fieldnames = [
        "instance",
        "paper_vehicles",
        "paper_distance",
        "paper_total_cost",
        "cpu_avg_run_sec",
        "cpu_fastest_run_sec",
        "cpu_avg_total_cost",
        "cpu_best_total_cost",
        "cpu_avg_gap_total_cost_pct",
        "cpu_best_gap_total_cost_pct",
        "cuda_avg_run_sec",
        "cuda_fastest_run_sec",
        "cuda_avg_total_cost",
        "cuda_best_total_cost",
        "cuda_avg_gap_total_cost_pct",
        "cuda_best_gap_total_cost_pct",
        "speedup_cpu_over_cuda",
        "time_diff_sec",
        "best_cost_diff_cuda_minus_cpu",
        "faster_backend",
        "better_backend_best_cost",
    ]
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                key: format_summary_value(row.get(key), 4 if key.endswith("_sec") or key == "speedup_cpu_over_cuda" else 2)
                for key in fieldnames
            }
        )
    write_csv(compare_root / "cpu_vs_cuda_summary.csv", csv_rows, fieldnames)
    write_json(compare_root / "cpu_vs_cuda_summary.json", rows)


def main() -> int:
    args = parse_args()
    instances = normalize_instances(list(args.instances))

    print("Project root:", args.project_root)
    print("Data root:", args.data_root)
    print("Work root:", args.work_root)
    print("Output root:", args.output_root)
    print("GPU available:", detect_gpu())
    print("Backends:", args.backends)
    print("Instances:", instances)

    if "cuda" in args.backends and not detect_gpu():
        raise SystemExit("CUDA backend requested, but no GPU is visible in this Kaggle session.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    project_root = ensure_project_copy(args.project_root, args.work_root, args.copy_project)
    data_link = ensure_data_link(project_root, args.data_root)

    benchmark_root = args.output_root / f"cpu_cuda_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "project_root": str(project_root),
        "data_root": str(args.data_root),
        "data_link": str(data_link),
        "output_root": str(benchmark_root),
        "instances": instances,
        "backends": args.backends,
        "repeats": args.repeats,
        "seed_start": args.seed_start,
        "timeout": args.timeout,
        "pop_size": args.pop_size,
        "max_iter": args.max_iter,
        "local_search_interval": args.local_search_interval,
        "stagnation_interval": args.stagnation_interval,
        "diversify_ratio": args.diversify_ratio,
        "sho_mutation_prob": args.sho_mutation_prob,
        "init": args.init,
        "elo": args.elo,
        "removal_lower": args.removal_lower,
        "removal_upper": args.removal_upper,
        "hybrid_mode": args.hybrid_mode,
        "time": args.time,
        "paper_flags": args.paper_flags,
        "solver_extra": args.solver_extra,
    }
    write_json(benchmark_root / "run_config.json", run_config)

    backend_rows: dict[str, list[dict[str, object]]] = {}
    backend_logs: dict[str, Path] = {}
    start_all = time.perf_counter()

    for index, backend in enumerate(args.backends, start=1):
        workers = 0 if backend == "cpu" else 1
        backend_root = benchmark_root / backend
        backend_root.mkdir(parents=True, exist_ok=True)
        log_path = backend_root / f"{backend}.log"
        backend_logs[backend] = log_path

        print()
        print(f"[{index}/{len(args.backends)}] backend={backend} workers={workers}")
        print(f"  results -> {backend_root}")

        cmd = build_compare_command(project_root, backend_root, instances, backend, workers, args)
        print("  command:", subprocess.list2cmdline([str(part) for part in cmd]))

        backend_started = time.perf_counter()
        rc = stream_command(cmd, project_root, log_path)
        backend_elapsed = time.perf_counter() - backend_started
        print(f"  backend {backend} finished with rc={rc} in {format_duration(backend_elapsed)}")

        if not (backend_root / "batch_summary.json").exists():
            tail = tail_text(log_path)
            if tail.strip():
                print("  log tail:")
                for line in tail.strip().splitlines()[-10:]:
                    print("   ", line)
            if args.stop_on_error:
                raise SystemExit(f"Backend {backend} did not produce batch_summary.json")
            continue

        if rc != 0:
            print(f"  warning: backend {backend} returned non-zero exit code {rc}")
            if args.stop_on_error:
                raise SystemExit(rc)

        rows = load_backend_report(backend, backend_root)
        backend_rows[backend] = rows
        write_backend_reports(backend_root, rows)

        stats = backend_stats(rows)
        print(
            "  summary: avg_run={avg} fastest_run={fastest} avg_gap_TC={gap_avg} best_gap_TC={gap_best}".format(
                avg=format_duration(stats["mean_avg_run_sec"]),
                fastest=format_duration(stats["mean_fastest_run_sec"]),
                gap_avg=("{:+.2f}%".format(stats["mean_avg_gap_tc"]) if stats["mean_avg_gap_tc"] is not None else "-"),
                gap_best=("{:+.2f}%".format(stats["mean_best_gap_tc"]) if stats["mean_best_gap_tc"] is not None else "-"),
            )
        )

    if not backend_rows:
        raise SystemExit("No backend reports were produced.")

    compare_rows = build_comparison_rows(backend_rows, instances)
    write_comparison_reports(benchmark_root, compare_rows)

    compare_csv_rows = []
    for row in compare_rows:
        compare_csv_rows.append(
            {
                "instance": row["instance"],
                "cpu_avg_run_sec": format_summary_value(row.get("cpu_avg_run_sec"), 4),
                "cuda_avg_run_sec": format_summary_value(row.get("cuda_avg_run_sec"), 4),
                "speedup_cpu_over_cuda": format_summary_value(row.get("speedup_cpu_over_cuda"), 4),
                "cpu_best_total_cost": format_summary_value(row.get("cpu_best_total_cost"), 2),
                "cuda_best_total_cost": format_summary_value(row.get("cuda_best_total_cost"), 2),
                "faster_backend": format_summary_value(row.get("faster_backend"), 0),
                "better_backend_best_cost": format_summary_value(row.get("better_backend_best_cost"), 0),
            }
        )

    print()
    print(
        render_table(
            [
                [
                    row["instance"],
                    row["cpu_avg_run_sec"],
                    row["cuda_avg_run_sec"],
                    row["speedup_cpu_over_cuda"],
                    row["cpu_best_total_cost"],
                    row["cuda_best_total_cost"],
                    row["faster_backend"],
                    row["better_backend_best_cost"],
                ]
                for row in compare_csv_rows
            ],
            [
                "instance",
                "cpu avg sec",
                "cuda avg sec",
                "speedup",
                "cpu best TC",
                "cuda best TC",
                "faster",
                "better cost",
            ],
        )
    )

    overall_rows = []
    for backend in args.backends:
        rows = backend_rows.get(backend, [])
        stats = backend_stats(rows)
        overall_rows.append(
            {
                "backend": backend,
                "instances": len(rows),
                "mean_avg_run_sec": stats["mean_avg_run_sec"],
                "mean_fastest_run_sec": stats["mean_fastest_run_sec"],
                "mean_avg_gap_td_pct": stats["mean_avg_gap_td"],
                "mean_avg_gap_tc_pct": stats["mean_avg_gap_tc"],
                "mean_best_gap_td_pct": stats["mean_best_gap_td"],
                "mean_best_gap_tc_pct": stats["mean_best_gap_tc"],
            }
        )

    write_json(benchmark_root / "backend_overall_summary.json", overall_rows)
    overall_csv_rows = []
    for row in overall_rows:
        overall_csv_rows.append(
            {
                "backend": row["backend"],
                "instances": row["instances"],
                "mean_avg_run_sec": format_summary_value(row.get("mean_avg_run_sec"), 4),
                "mean_fastest_run_sec": format_summary_value(row.get("mean_fastest_run_sec"), 4),
                "mean_avg_gap_td_pct": format_summary_value(row.get("mean_avg_gap_td_pct"), 2),
                "mean_avg_gap_tc_pct": format_summary_value(row.get("mean_avg_gap_tc_pct"), 2),
                "mean_best_gap_td_pct": format_summary_value(row.get("mean_best_gap_td_pct"), 2),
                "mean_best_gap_tc_pct": format_summary_value(row.get("mean_best_gap_tc_pct"), 2),
            }
        )
    write_csv(
        benchmark_root / "backend_overall_summary.csv",
        overall_csv_rows,
        [
            "backend",
            "instances",
            "mean_avg_run_sec",
            "mean_fastest_run_sec",
            "mean_avg_gap_td_pct",
            "mean_avg_gap_tc_pct",
            "mean_best_gap_td_pct",
            "mean_best_gap_tc_pct",
        ],
    )

    if len(args.backends) == 2 and "cpu" in backend_rows and "cuda" in backend_rows:
        cpu_rows = {row["instance"]: row for row in backend_rows["cpu"]}
        cuda_rows = {row["instance"]: row for row in backend_rows["cuda"]}
        speedups = []
        for instance in instances:
            cpu_row = cpu_rows.get(instance)
            cuda_row = cuda_rows.get(instance)
            if cpu_row is None or cuda_row is None:
                continue
            cpu_avg = cpu_row.get("avg_run_sec")
            cuda_avg = cuda_row.get("avg_run_sec")
            if cpu_avg is None or cuda_avg in (None, 0):
                continue
            speedups.append(float(cpu_avg) / float(cuda_avg))
        if speedups:
            print()
            print(f"Mean CPU/CUDA speedup over available instances: {mean(speedups):.2f}x")

    total_elapsed = time.perf_counter() - start_all
    print()
    print("Benchmark complete.")
    print("Output root:", benchmark_root)
    print("Total elapsed:", format_duration(total_elapsed))
    print("Compare CSV:", benchmark_root / "cpu_vs_cuda_summary.csv")
    print("Backend CSV:", benchmark_root / "backend_overall_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

