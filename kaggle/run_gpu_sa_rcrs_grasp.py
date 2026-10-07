#!/usr/bin/env python3
"""
Orchestrator to run benchmark instances across 3 Kaggle accounts (2 workers each = 6 slots).
Each slot runs instances one-by-one with:
  pop_size=30, max_iter=1000, runs=30
Results are continuously downloaded and written directly to:
  docs/instances_benchmark_pop_size30_runs=30_iter1000.csv
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "docs" / "instances_benchmark_pop_size30_runs=30_iter1000.csv"
STATE_FILE = REPO_ROOT / "kaggle" / "runs" / "benchmark_state.json"
RUNS_DIR = REPO_ROOT / "kaggle" / "runs"

# Worker configurations (3 accounts, 2 workers each)
WORKERS = [
    {
        "id": "a1",
        "account": "a",
        "slug": "saostken/ph-showoa",
        "title": "PH-SHOWOA",
        "dir": RUNS_DIR / "a1",
        "code_file": "ph-showoa.ipynb",
    },
    {
        "id": "a2",
        "account": "a",
        "slug": "saostken/ph-showoa-gpu-sa-rcrs-grasp-a2",
        "title": "PH-SHOWOA GPU SA-RCRS-GRASP (a2)",
        "dir": RUNS_DIR / "a2",
        "code_file": "notebook.ipynb",
    },
    {
        "id": "b1",
        "account": "b",
        "slug": "welkie/wel-ph-showoa-rcrs-grasp",
        "title": "Wel_PH-SHOWOA_rcrs_grasp",
        "dir": RUNS_DIR / "b1",
        "code_file": "notebook.ipynb",
    },
    {
        "id": "b2",
        "account": "b",
        "slug": "welkie/wel-ph-showoa-gpu-sa-rcrs-grasp-b2",
        "title": "Wel_PH-SHOWOA GPU SA-RCRS-GRASP (b2)",
        "dir": RUNS_DIR / "b2",
        "code_file": "notebook.ipynb",
    },
    {
        "id": "c1",
        "account": "c",
        "slug": "minhanhphm1676/kaggle-gpu-sa-rcrs-grasp",
        "title": "kaggle_gpu_sa_rcrs_grasp",
        "dir": RUNS_DIR / "c1",
        "code_file": "notebook.ipynb",
    },
    {
        "id": "c2",
        "account": "c",
        "slug": "minhanhphm1676/ma-ph-showoa-gpu-sa-rcrs-grasp-c2",
        "title": "MA_PH-SHOWOA GPU SA-RCRS-GRASP (c2)",
        "dir": RUNS_DIR / "c2",
        "code_file": "notebook.ipynb",
    },
]


def find_kaggle_exe():
    """Locate the kaggle CLI executable."""
    venv_kaggle = REPO_ROOT / ".venv" / "Scripts" / "kaggle.exe"
    if venv_kaggle.exists():
        return str(venv_kaggle)
    venv_kaggle_unix = REPO_ROOT / ".venv" / "bin" / "kaggle"
    if venv_kaggle_unix.exists():
        return str(venv_kaggle_unix)
    kaggle_in_path = shutil.which("kaggle")
    if kaggle_in_path:
        return kaggle_in_path
    raise RuntimeError("Could not find kaggle CLI executable in .venv or PATH")


def run_kaggle_cmd(account: str, args: list, timeout: int = 120):
    """Run a kaggle CLI command under the specified account configuration."""
    kaggle_exe = find_kaggle_exe()
    acc_dir = REPO_ROOT / "kaggle" / f"account_{account}"
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(acc_dir)
    cmd = [kaggle_exe] + args
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=timeout,
        )
        return result
    except subprocess.TimeoutExpired:
        print(f">> [TIMEOUT] Command '{' '.join(args[:3])}' timed out after {timeout}s.")
        return subprocess.CompletedProcess(args=cmd, returncode=124, stdout="", stderr="TimeoutExpired")
    except Exception as e:
        print(f">> [ERROR] Failed running kaggle command: {e}")
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr=str(e))



def load_csv(csv_path: Path):
    """Load benchmark CSV rows."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)
    return rows, fieldnames


def save_csv(csv_path: Path, rows: list, fieldnames: list):
    """Save benchmark CSV atomically."""
    tmp_path = csv_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    shutil.move(tmp_path, csv_path)


def is_instance_completed(row: dict) -> bool:
    """Check if an instance already has valid results in the CSV."""
    nv = str(row.get("NV", "")).strip()
    td = str(row.get("TD", "")).strip()
    return bool(nv and td and nv.lower() != "nan" and td.lower() != "nan")


def get_pending_instances(rows: list, active_instances: set) -> list:
    """Get list of instance names that are not completed and not currently running."""
    pending = []
    for r in rows:
        name = r.get("Instance", "").strip()
        if not name:
            continue
        if not is_instance_completed(r) and name.lower() not in active_instances:
            pending.append(name)
    return pending


def load_state() -> dict:
    """Load orchestrator state file."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f">> [WARN] Failed to load state file: {e}")
    return {"workers": {}, "disabled_accounts": [], "retries": {}}


def save_state(state: dict):
    """Save orchestrator state file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_FILE.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    shutil.move(tmp_path, STATE_FILE)


def generate_single_instance_notebook(instance_name: str, pop_size: int, max_iter: int, runs: int) -> dict:
    """Generate the notebook JSON for a single instance execution."""
    name_clean = instance_name.strip().lower()
    
    cell_clone = [
        "import os, shutil\n",
        "REPO_DIR = '/tmp/ph-showoa'\n",
        "if os.path.exists(REPO_DIR):\n",
        "    !cd {REPO_DIR} && git fetch origin && git reset --hard origin/main\n",
        "else:\n",
        "    !git clone --depth 1 https://github.com/Welkie/ph-showoa.git {REPO_DIR}\n",
        "if os.path.exists('/kaggle/working/ph-showoa'):\n",
        "    shutil.rmtree('/kaggle/working/ph-showoa', ignore_errors=True)\n"
    ]
    
    cell_gpu = [
        "!nvidia-smi\n",
        "import torch\n",
        "print(\"PyTorch version:\", torch.__version__)\n",
        "print(\"CUDA available:\", torch.cuda.is_available())\n",
        "if not torch.cuda.is_available():\n",
        "    raise RuntimeError(\"Kaggle GPU runtime is required: enable a CUDA accelerator before running the benchmark.\")\n",
        "torch.cuda.set_device(0)\n",
        "print(\"Device name:\", torch.cuda.get_device_name(0))\n",
        "print(\"CUDA capability:\", torch.cuda.get_device_capability(0))\n"
    ]
    
    cell_check_solver = [
        "import torch\n",
        "if not torch.cuda.is_available():\n",
        "    raise RuntimeError(\"CUDA GPU is required for the Python/PyTorch CUDA solver.\")\n",
        "torch.cuda.set_device(0)\n",
        "print(\"Python/PyTorch CUDA backend ready\")\n",
        "print(\"PyTorch version:\", torch.__version__)\n",
        "print(\"Device:\", torch.cuda.get_device_name(0))\n",
        "print(\"Capability:\", torch.cuda.get_device_capability(0))\n",
        "print(\"Implementation: Python + PyTorch CUDA tensors\")\n"
    ]
    
    cell_solver_code = [
        "import glob\n",
        "import os\n",
        "import re\n",
        "import subprocess\n",
        "import sys\n",
        "import time as T\n",
        "from collections import deque\n",
        "import pandas as pd\n",
        "\n",
        f"DATASETS = [\"{name_clean}\"]\n",
        "CWD = \"/tmp/ph-showoa\" if os.path.exists(\"/tmp/ph-showoa\") else (\"ph-showoa\" if os.path.exists(\"ph-showoa\") else \".\")\n",
        "DATASET_DIR = \"/kaggle/input/datasets/keith1101/ph-showoa/Wang_Chen\"\n",
        "OUTPUT_CSV = \"/kaggle/working/summary_python_pytorch_cuda.csv\"\n",
        "LOG_DIR = \"/kaggle/working/logs_python_pytorch_cuda\"\n",
        f"RUNS = {runs}\n",
        f"MAX_ITER = {max_iter}\n",
        f"POP_SIZE = {pop_size}\n",
        "INIT_MODE = \"sa_rcrs_grasp\"\n",
        "COMPUTE_BACKEND = \"cuda\"\n",
        "ARCHITECTURE = \"python_cuda\"\n",
        "WORKERS = 1\n",
        "RESUME = True\n",
        "MAX_TIMEOUT_S = None\n",
        "TAIL_LINES = 20\n",
        "os.makedirs(LOG_DIR, exist_ok=True)\n",
        "\n",
        "def find_file(name):\n",
        "    nc = name.strip().lower()\n",
        "    candidates = [\n",
        "        os.path.join(CWD, \"dataset\", f\"explicit_{nc}.vrpsdptw\"),\n",
        "        os.path.join(DATASET_DIR, f\"explicit_{nc}.vrpsdptw\"),\n",
        "    ]\n",
        "    for pattern in (f\"/kaggle/input/**/explicit_{nc}.vrpsdptw\", f\"**/explicit_{nc}.vrpsdptw\"):\n",
        "        candidates.extend(glob.glob(pattern, recursive=True))\n",
        "    found = next((path for path in candidates if os.path.isfile(path)), None)\n",
        "    if found:\n",
        "        return found\n",
        "    for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
        "        for f in files:\n",
        "            if f.lower() == f\"explicit_{nc}.vrpsdptw\" or f.lower() == f\"{nc}.vrpsdptw\":\n",
        "                return os.path.join(root, f)\n",
        "    for root, dirs, files in os.walk(CWD):\n",
        "        for f in files:\n",
        "            if f.lower() == f\"explicit_{nc}.vrpsdptw\" or f.lower() == f\"{nc}.vrpsdptw\":\n",
        "                return os.path.join(root, f)\n",
        "    return None\n",
        "\n",
        "def parse_output(text):\n",
        "    if \"Traceback (most recent call last)\" in text or \"RuntimeError:\" in text:\n",
        "        return None\n",
        "    runs = re.search(r\"Total (\\d+) runs, total consumed (\\d+) sec\", text)\n",
        "    nv = re.search(r\"(?:Vehicle count|vehicle \\(route\\) number):\\s*(\\d+)\", text)\n",
        "    cost = re.search(r\"Total cost:\\s*([\\d.]+)\", text)\n",
        "    distance = re.search(r\"Total distance:\\s*([\\d.]+)\", text)\n",
        "    if not (runs and nv and cost) or int(runs.group(1)) != RUNS:\n",
        "        return None\n",
        "    total_runs = int(runs.group(1))\n",
        "    total_cost = float(cost.group(1))\n",
        "    total_distance = float(distance.group(1)) if distance else total_cost - 2000.0 * int(nv.group(1))\n",
        "    return {\n",
        "        \"best_NV\": int(nv.group(1)),\n",
        "        \"best_TD\": f\"{total_distance:.4f}\",\n",
        "        \"total_cost\": f\"{total_cost:.4f}\",\n",
        "        \"avg_time_s\": f\"{float(runs.group(2)) / total_runs:.2f}\",\n",
        "        \"total_runs\": total_runs,\n",
        "        \"Status\": \"Success\",\n",
        "    }\n",
        "\n",
        "def save_summary(rows):\n",
        "    columns = [\"Dataset\", \"best_NV\", \"best_TD\", \"total_cost\", \"avg_time_s\", \"wall_time\", \"total_runs\", \"Status\", \"Error\"]\n",
        "    frame = pd.DataFrame(rows)\n",
        "    for column in columns:\n",
        "        if column not in frame:\n",
        "            frame[column] = \"N/A\"\n",
        "    frame = frame[columns]\n",
        "    frame.columns = [\"Dataset\", \"Best NV\", \"Best TD\", \"Total Cost\", \"Avg/Run\", \"Wall Time\", \"Runs\", \"Status\", \"Error\"]\n",
        "    frame.to_csv(OUTPUT_CSV, index=False)\n",
        "\n",
        "results = []\n",
        "for name in DATASETS:\n",
        "    problem = find_file(name)\n",
        "    if problem is None:\n",
        "        print(f\"[ERROR] File for dataset {name} not found!\")\n",
        "        results.append({\"Dataset\": name, \"Status\": \"File Not Found\", \"Error\": \"dataset file not found\"})\n",
        "        save_summary(results)\n",
        "        continue\n",
        "\n",
        "    command = [\n",
        "        sys.executable, \"-u\", \"-m\", \"src_python_gpu_SA_RCRS_GRASP.main\",\n",
        "        \"--problem\", problem, \"--compute_backend\", COMPUTE_BACKEND,\n",
        "        \"--init\", INIT_MODE, \"--paper_flags\", \"--architecture\", ARCHITECTURE,\n",
        "        \"--objective\", \"lexicographic\", \"--grasp_alpha_lo\", \"0.10\",\n",
        "        \"--grasp_alpha_hi\", \"0.40\", \"--sa_iterations\", \"25\",\n",
        "        \"--runs\", str(RUNS), \"--max_iter\", str(MAX_ITER),\n",
        "        \"--pop_size\", str(POP_SIZE), \"--workers\", str(WORKERS),\n",
        "    ]\n",
        "    print(f\"\\n[RUN] {name}: Python/PyTorch CUDA | target runs={RUNS} pop_size={POP_SIZE} max_iter={MAX_ITER}\")\n",
        "    run_started = T.time()\n",
        "    last_report = run_started\n",
        "    lines = []\n",
        "    tail = deque(maxlen=TAIL_LINES)\n",
        "    timed_out = False\n",
        "    process = subprocess.Popen(command, cwd=CWD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
        "    try:\n",
        "        for line in process.stdout:\n",
        "            lines.append(line)\n",
        "            clean = line.rstrip()\n",
        "            tail.append(clean)\n",
        "            now = T.time()\n",
        "            run_marker = re.search(r\"Run (\\d+).*\", clean)\n",
        "            total_marker = re.search(r\"Total (\\d+) runs, total consumed\", clean)\n",
        "            if run_marker and \"Run \" in clean:\n",
        "                print(f\"  [{now - run_started:.1f}s] {clean}\", flush=True)\n",
        "            elif total_marker:\n",
        "                print(f\"  [{now - run_started:.1f}s] {clean}\", flush=True)\n",
        "            elif now - last_report >= 20:\n",
        "                print(f\"  [{now - run_started:.0f}s] running... last: {clean[:100]}\", flush=True)\n",
        "                last_report = now\n",
        "            if MAX_TIMEOUT_S and now - run_started > MAX_TIMEOUT_S:\n",
        "                timed_out = True\n",
        "                process.kill()\n",
        "                break\n",
        "        process.wait()\n",
        "    except Exception:\n",
        "        process.kill()\n",
        "        process.wait()\n",
        "        raise\n",
        "\n",
        "    wall = T.time() - run_started\n",
        "    output = \"\".join(lines)\n",
        "    with open(os.path.join(LOG_DIR, f\"{name}.log\"), \"w\", encoding=\"utf-8\") as log:\n",
        "        log.write(output)\n",
        "    return_code = process.returncode\n",
        "    parsed = parse_output(output) if return_code == 0 and not timed_out else None\n",
        "    if parsed is None:\n",
        "        error_lines = [line.strip() for line in lines if \"error\" in line.lower() or \"traceback\" in line.lower()]\n",
        "        row = {\"Dataset\": name, \"Status\": \"Timeout\" if timed_out else \"Failed\", \"Error\": \" | \".join(error_lines[-3:]) or f\"Python process exit code {return_code}\", \"wall_time\": f\"{wall:.1f}s\"}\n",
        "        print(f\"[FAILED] {name}: {row['Error']}\")\n",
        "    else:\n",
        "        row = parsed | {\"Dataset\": name, \"wall_time\": f\"{wall:.1f}s\", \"Error\": \"\"}\n",
        "        print(f\"[OK] {name}: completed {parsed['total_runs']}/{RUNS} runs | NV={row['best_NV']} TD={row['best_TD']} Avg/Run={row['avg_time_s']}s\")\n",
        "    results.append(row)\n",
        "    save_summary(results)\n",
        "\n",
        "print(f\"Finished. Results saved to {OUTPUT_CSV}\")\n"
    ]
    
    cell_summary = [
        "import os\n",
        "import pandas as pd\n",
        "OUTPUT_CSV = \"/kaggle/working/summary_python_pytorch_cuda.csv\"\n",
        "if os.path.exists(OUTPUT_CSV):\n",
        "    df = pd.read_csv(OUTPUT_CSV)\n",
        "    print(\"SUMMARY RESULT:\")\n",
        "    print(df.to_string(index=False))\n"
    ]

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# Benchmark Instance: {instance_name} (pop_size={pop_size}, max_iter={max_iter}, runs={runs})\n"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_clone
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_gpu
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_check_solver
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_solver_code
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_summary
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return notebook


def write_worker_files(worker: dict, instance_name: str, pop_size: int, max_iter: int, runs: int):
    """Write notebook and kernel-metadata.json into worker directory."""
    worker_dir = worker["dir"]
    worker_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate notebook
    nb = generate_single_instance_notebook(instance_name, pop_size, max_iter, runs)
    code_path = worker_dir / worker["code_file"]
    with open(code_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
        
    # Write metadata
    meta = {
        "id": worker["slug"],
        "title": worker["title"],
        "code_file": worker["code_file"],
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            "keith1101/ph-showoa"
        ],
        "competition_sources": [],
        "kernel_sources": []
    }
    meta_path = worker_dir / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def push_worker(worker: dict, instance_name: str, pop_size: int, max_iter: int, runs: int) -> bool:
    """Prepare notebook and push worker kernel to Kaggle."""
    worker_dir = worker["dir"]
    write_worker_files(worker, instance_name, pop_size, max_iter, runs)
    
    # Remove existing output folder
    out_dir = worker_dir / "output"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
        
    res = run_kaggle_cmd(worker["account"], ["kernels", "push", "-p", str(worker_dir)])
    combined = res.stdout + "\n" + res.stderr
    if "successfully pushed" in combined.lower():
        print(f">> [PUSH OK] Worker {worker['id']} ({worker['slug']}) dispatched for instance '{instance_name}'")
        return True
    else:
        print(f">> [PUSH FAILED] Worker {worker['id']} ({worker['slug']}):\n{combined.strip()}")
        return False


def get_worker_status(worker: dict) -> str:
    """Get current status string of a worker kernel."""
    res = run_kaggle_cmd(worker["account"], ["kernels", "status", worker["slug"]])
    text = res.stdout.strip()
    match = re.search(r'status "([^"]+)"', text)
    if match:
        return match.group(1)
    if "KernelWorkerStatus." in text:
        return text.split("KernelWorkerStatus.")[-1].split()[0].replace('"', '')
    return text


def download_and_parse_results(worker: dict, expected_instance: str):
    """Download kernel outputs and extract benchmark results for expected_instance."""
    worker_dir = worker["dir"]
    out_dir = worker_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_kaggle_cmd(worker["account"], ["kernels", "output", worker["slug"], "-p", str(out_dir)], timeout=90)
    
    # 1. Try reading summary_python_pytorch_cuda.csv
    csv_candidates = [
        out_dir / "summary_python_pytorch_cuda.csv",
        out_dir / "summary.csv",
    ]
    for c in csv_candidates:
        if c.exists():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ds = str(row.get("Dataset", "")).strip().lower()
                        if ds == expected_instance.strip().lower() or len(list(reader)) == 1:
                            status = str(row.get("Status", "")).strip().lower()
                            if status == "success":
                                return {
                                    "NV": row.get("Best NV", ""),
                                    "TD": row.get("Best TD", ""),
                                    "Avg/Run (s)": row.get("Avg/Run", ""),
                                }
            except Exception as e:
                print(f">> [WARN] Failed parsing {c}: {e}")
                
    # 2. Try parsing log file
    log_candidates = list(out_dir.glob("*.log")) + list(out_dir.glob("logs_python_pytorch_cuda/*.log"))
    for log_file in log_candidates:
        try:
            text = log_file.read_text(encoding="utf-8", errors="ignore")
            runs_m = re.search(r"Total (\d+) runs, total consumed (\d+) sec", text)
            nv_m = re.search(r"(?:Vehicle count|vehicle \(route\) number):\s*(\d+)", text)
            cost_m = re.search(r"Total cost:\s*([\d.]+)", text)
            dist_m = re.search(r"Total distance:\s*([\d.]+)", text)
            if runs_m and nv_m and cost_m:
                total_runs = int(runs_m.group(1))
                best_nv = int(nv_m.group(1))
                total_cost = float(cost_m.group(1))
                total_dist = float(dist_m.group(1)) if dist_m else total_cost - 2000.0 * best_nv
                avg_time = float(runs_m.group(2)) / total_runs
                return {
                    "NV": str(best_nv),
                    "TD": f"{total_dist:.4f}",
                    "Avg/Run (s)": f"{avg_time:.2f}",
                }
        except Exception:
            pass

    return None


def print_status_table(rows: list, state: dict):
    """Print an informative table of progress and worker status."""
    total = len(rows)
    done = sum(1 for r in rows if is_instance_completed(r))
    active = {w_id: info.get("instance") for w_id, info in state.get("workers", {}).items() if info.get("status") == "RUNNING"}
    
    print("\n" + "=" * 70)
    print(f" BENCHMARK STATUS: {done}/{total} completed ({done/total*100:.1f}%) | Active workers: {len(active)}/6")
    print("=" * 70)
    for w in WORKERS:
        w_id = w["id"]
        w_info = state.get("workers", {}).get(w_id, {})
        w_stat = w_info.get("status", "IDLE")
        w_inst = str(w_info.get("instance") or "-")
        w_elapsed = ""
        if "start_time" in w_info and w_stat == "RUNNING":
            elapsed_sec = int(time.time() - w_info["start_time"])
            w_elapsed = f"({elapsed_sec // 60}m {elapsed_sec % 60}s)"
        print(f"  Worker {w_id} [Acc {w['account'].upper()}]: {w_stat:<10} | Instance: {w_inst:<12} {w_elapsed}")
    print("=" * 70 + "\n")


def orchestrate(csv_path: Path, pop_size: int, max_iter: int, runs: int, poll_interval: int, single_step: bool = False):
    """Main orchestration loop."""
    print(">> Initializing Kaggle Multi-Account Benchmark Orchestrator...")
    print(f">> Benchmark CSV: {csv_path}")
    print(f">> Parameters: pop_size={pop_size}, max_iter={max_iter}, runs={runs}")
    print(f">> Accounts: a, b, c (6 slots total)\n")

    state = load_state()
    disabled_accounts = set(state.get("disabled_accounts", []))
    worker_states = state.get("workers", {})
    retries = state.get("retries", {})

    while True:
        rows, fieldnames = load_csv(csv_path)
        active_instances = {info["instance"].lower() for info in worker_states.values() if info.get("status") in ("JUST_PUSHED", "RUNNING") and info.get("instance")}
        pending_instances = get_pending_instances(rows, active_instances)

        # 1. Dispatch pending instances to any IDLE workers immediately
        for worker in WORKERS:
            w_id = worker["id"]
            acc = worker["account"]
            if acc in disabled_accounts:
                continue

            current_w = worker_states.get(w_id, {"status": "IDLE"})
            if current_w.get("status") in ("IDLE", None):
                if pending_instances:
                    next_instance = pending_instances.pop(0)
                    print(f">> Dispatching '{next_instance}' to Worker {w_id} (Account {acc.upper()})...")
                    ok = push_worker(worker, next_instance, pop_size, max_iter, runs)
                    if ok:
                        current_w = {
                            "status": "JUST_PUSHED",
                            "instance": next_instance,
                            "pushed_at": time.time(),
                            "start_time": time.time(),
                            "has_started": False,
                        }
                        worker_states[w_id] = current_w
                    else:
                        pending_instances.insert(0, next_instance)

        # 2. Poll active workers
        for worker in WORKERS:
            w_id = worker["id"]
            if worker["account"] in disabled_accounts:
                continue

            current_w = worker_states.get(w_id, {})
            current_status = current_w.get("status", "IDLE")

            if current_status in ("JUST_PUSHED", "RUNNING"):
                instance = current_w.get("instance")
                raw_stat = get_worker_status(worker)
                
                # Check for running or complete
                if "RUNNING" in raw_stat or "QUEUED" in raw_stat:
                    current_w["status"] = "RUNNING"
                    current_w["has_started"] = True
                elif "COMPLETE" in raw_stat:
                    pushed_at = current_w.get("pushed_at", 0)
                    has_started = current_w.get("has_started", False)
                    # Ignore stale COMPLETE from previous version until new version was observed running or 120s passed
                    if not has_started and (time.time() - pushed_at < 120):
                        continue
                    
                    print(f">> [COMPLETE] Worker {w_id} finished instance '{instance}'. Downloading output...")
                    result = download_and_parse_results(worker, instance)
                    if result:
                        print(f">> [SUCCESS] '{instance}': NV={result['NV']}, TD={result['TD']}, Avg/Run={result['Avg/Run (s)']}s")
                        # Update CSV
                        for r in rows:
                            if r.get("Instance", "").strip().lower() == instance.strip().lower():
                                r["NV"] = result["NV"]
                                r["TD"] = result["TD"]
                                r["Avg/Run (s)"] = result["Avg/Run (s)"]
                                break
                        save_csv(csv_path, rows, fieldnames)
                        current_w["status"] = "IDLE"
                        current_w["instance"] = None
                        current_w["has_started"] = False
                    else:
                        print(f">> [WARN] Worker {w_id} finished but failed to parse results for '{instance}'.")
                        retry_count = retries.get(instance, 0) + 1
                        retries[instance] = retry_count
                        current_w["status"] = "IDLE"
                        current_w["instance"] = None
                        current_w["has_started"] = False

                elif "CANCEL" in raw_stat or "ERROR" in raw_stat or "FAILED" in raw_stat:
                    print(f">> [WARN] Worker {w_id} status '{raw_stat}' on instance '{instance}'.")
                    worker_dir = worker["dir"]
                    out_dir = worker_dir / "output"
                    run_kaggle_cmd(worker["account"], ["kernels", "output", worker["slug"], "-p", str(out_dir)], timeout=30)
                    logs_text = ""
                    for p in out_dir.glob("*.log"):
                        logs_text += p.read_text(encoding="utf-8", errors="ignore") + "\n"
                    if "quota" in logs_text.lower() or "limit" in logs_text.lower():
                        print(f">> [QUOTA EXCEEDED] Account {worker['account'].upper()} reached Kaggle quota. Disabling account.")
                        disabled_accounts.add(worker["account"])
                    retry_count = retries.get(instance, 0) + 1
                    retries[instance] = retry_count
                    current_w["status"] = "IDLE"
                    current_w["instance"] = None
                    current_w["has_started"] = False

            worker_states[w_id] = current_w


        # Update saved state
        state["workers"] = worker_states
        state["disabled_accounts"] = list(disabled_accounts)
        state["retries"] = retries
        save_state(state)

        # Print status summary
        print_status_table(rows, state)

        # Check termination
        all_done = (len(pending_instances) == 0 and all(w.get("status") in ("IDLE", None) for w in worker_states.values()))
        all_disabled = len(disabled_accounts) == 3
        if all_done:
            print(">> [ALL COMPLETED] All instances in benchmark CSV have completed successfully!")
            break
        if all_disabled:
            print(">> [QUOTA LIMIT] All 3 Kaggle accounts have reached quota limits. Stopped.")
            break
        if single_step:
            print(">> [SINGLE STEP] Completed one check/dispatch cycle. Exiting.")
            break

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="PH-SHOWOA Kaggle Multi-Account Benchmark Orchestrator")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Path to benchmark instances CSV")
    parser.add_argument("--pop_size", type=int, default=30, help="Population size (default: 30)")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations (default: 1000)")
    parser.add_argument("--runs", type=int, default=30, help="Number of independent runs (default: 30)")
    parser.add_argument("--poll_interval", type=int, default=30, help="Polling interval in seconds (default: 30)")
    parser.add_argument("--status", action="store_true", help="Print current status and exit")
    parser.add_argument("--once", action="store_true", help="Run a single dispatch/check step and exit")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if args.status:
        rows, _ = load_csv(csv_path)
        state = load_state()
        print_status_table(rows, state)
        return

    orchestrate(
        csv_path=csv_path,
        pop_size=args.pop_size,
        max_iter=args.max_iter,
        runs=args.runs,
        poll_interval=args.poll_interval,
        single_step=args.once,
    )


if __name__ == "__main__":
    main()
