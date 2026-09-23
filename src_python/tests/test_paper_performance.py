import csv
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "tests" / "paper_performance_targets.csv"
VEHICLE_WEIGHT = 2000.0


def _selected_targets():
    wanted = {
        item.strip().lower()
        for item in os.environ.get("PAPER_PERF_INSTANCES", "RCdp1001").split(",")
        if item.strip()
    }
    with TARGETS.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if "all" in wanted:
        return rows
    return [row for row in rows if row["instance"].lower() in wanted]


def _parse_solution(path):
    text = path.read_text(encoding="utf-8")
    vehicles = int(re.search(r"vehicle \(route\) number:\s+(\d+)", text).group(1))
    total_cost = float(re.search(r"Total cost:\s+([0-9.]+)", text).group(1))
    distance = total_cost - vehicles * VEHICLE_WEIGHT
    return vehicles, distance, total_cost


@unittest.skipUnless(
    os.environ.get("RUN_PAPER_PERF") == "1",
    "Set RUN_PAPER_PERF=1 to run stochastic paper-performance checks.",
)
class PaperPerformanceTest(unittest.TestCase):
    def test_ph_showoa_matches_paper_targets_with_tolerance(self):
        targets = _selected_targets()
        self.assertGreater(len(targets), 0, "No paper targets selected")

        pop_size = os.environ.get("PAPER_PERF_POP_SIZE", "36")
        max_iter = os.environ.get("PAPER_PERF_MAX_ITER", "1000")
        seeds = [
            item.strip()
            for item in os.environ.get("PAPER_PERF_SEEDS", "42").split(",")
            if item.strip()
        ]
        distance_tolerance = float(os.environ.get("PAPER_PERF_DISTANCE_TOLERANCE", "0.10"))
        vehicle_slack = int(os.environ.get("PAPER_PERF_VEHICLE_SLACK", "0"))
        timeout = int(os.environ.get("PAPER_PERF_TIMEOUT", "900"))
        extra_args = shlex.split(os.environ.get("PAPER_PERF_EXTRA_ARGS", ""))

        failures = []
        for target in targets:
            best = None
            for seed in seeds:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    out_file = Path(tmp_dir) / "solution.txt"
                    cmd = [
                        sys.executable,
                        "-m",
                        "src",
                        "--problem",
                        target["path"],
                        "--runs",
                        "1",
                        "--pop_size",
                        pop_size,
                        "--max_iter",
                        max_iter,
                        "--workers",
                        "1",
                        "--hybrid_mode",
                        "ph_showoa",
                        "--local_search_interval",
                        "25",
                        "--stagnation_interval",
                        "50",
                        "--random_seed",
                        seed,
                        "--output",
                        str(out_file),
                    ]
                    cmd.extend(extra_args)
                    try:
                        result = subprocess.run(
                            cmd,
                            cwd=ROOT,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            timeout=timeout,
                            check=False,
                        )
                    except subprocess.TimeoutExpired as exc:
                        failures.append(
                            "%s seed %s timed out after %d seconds. "
                            "Increase PAPER_PERF_TIMEOUT, reduce PAPER_PERF_MAX_ITER, "
                            "or test smaller instances first.\n%s"
                            % (
                                target["instance"],
                                seed,
                                timeout,
                                (exc.stdout or "")[-2000:],
                            )
                        )
                        continue
                    if result.returncode != 0:
                        failures.append(
                            "%s seed %s exited %s\n%s"
                            % (target["instance"], seed, result.returncode, result.stdout[-2000:])
                        )
                        continue
                    vehicles, distance, total_cost = _parse_solution(out_file)
                    candidate = (vehicles, distance, total_cost, seed)
                    if best is None or (vehicles, distance) < (best[0], best[1]):
                        best = candidate

            if best is None:
                continue

            target_vehicles = int(target["target_vehicles"])
            target_distance = float(target["target_distance"])
            max_vehicles = target_vehicles + vehicle_slack
            max_distance = target_distance * (1.0 + distance_tolerance)

            if best[0] > max_vehicles or best[1] > max_distance:
                failures.append(
                    "%s best seed %s got NV=%d TD=%.2f; target NV<=%d TD<=%.2f"
                    % (target["instance"], best[3], best[0], best[1], max_vehicles, max_distance)
                )

        if failures:
            self.fail("\n\n".join(failures))


if __name__ == "__main__":
    unittest.main()
