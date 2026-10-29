#!/usr/bin/env python3
"""
Kaggle multi-account orchestrator for the full benchmark of src_python_gpu_SA_RCRS_GRASP (fullcaitienGPU, the
default SolverConfig) on every instance of docs/instances_benchmark_pop_size30_runs=30_iter1000.csv.

- 10 accounts (1..10 = kaggle/1_account_*/ .. 10_account_*/) x 2 concurrent kernels = 20 workers.
- Each push is one batch of up to 2 instances; a T4 x2 session runs them in parallel, one instance per GPU
  (CUDA_VISIBLE_DEVICES), via kaggle/ablation_driver.py, exactly as the ablation of docs/improved.md.
- One instance = the 30 runs (seed 41 + run, i.e. 42..71) in ONE process: the kernels are compiled once (JIT inside
  run 1) and reused by runs 2..30.
- Timing is the one of src_python_baseline: elapsed_seconds of solve(), from the start of solve() until the best
  solution is available (problem loading before solve() and the final feasibility check are not timed). The GPU
  version additionally times its own overheads inside solve(): problem upload, buffer allocation and, in run 1, the
  JIT compile. Process start-up and imports are not timed, as for the baseline.
- The code is not fetched from GitHub: the current src_python_gpu_SA_RCRS_GRASP and ablation_driver.py are zipped
  into kaggle/runs_full/payload.zip and embedded in every notebook. The hash of the sources is kept in the state;
  the orchestrator refuses to continue with changed code once results exist (start a new WORK_DIR instead).
- Instances: Wang_Chen/<name>.txt of the Kaggle dataset DATASET_SOURCE (same files as dataset/ of the repository).
- On completion of an instance (all 30 runs feasible) the orchestrator writes:
    * docs/ph_showoa_caitiengpu_all.csv                          -> NV, TD, cost, time of the 30 runs
    * docs/instances_benchmark_pop_size30_runs=30_iter1000.csv  -> best_NV, best_TD, Avg/Run (s)
  and keeps the raw CSVs (runs and convergence) in kaggle/runs_full/results/.
- Before each push the remaining weekly GPU quota of the account is checked.

Usage:
    python kaggle/orchestrate_gpu_all.py            # run until everything is done or quota is exhausted
    python kaggle/orchestrate_gpu_all.py --status   # print progress and exit
    python kaggle/orchestrate_gpu_all.py --smoke    # tiny end-to-end test (2 runs, 20 iterations), no CSV writes
    python kaggle/orchestrate_gpu_all.py --accounts 1,2,3   # only these accounts
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG = "src_python_gpu_SA_RCRS_GRASP"
DRIVER = REPO_ROOT / "kaggle" / "ablation_driver.py"
BENCH_CSV = REPO_ROOT / "docs" / "instances_benchmark_pop_size30_runs=30_iter1000.csv"
RUNS_CSV = REPO_ROOT / "docs" / "ph_showoa_caitiengpu_all.csv"
WORK_DIR = REPO_ROOT / "kaggle" / "runs_full"
STATE_FILE = WORK_DIR / "state.json"
LOG_FILE = WORK_DIR / "orchestrator.log"
RESULTS_DIR = WORK_DIR / "results"
RAW_DIR = WORK_DIR / "raw"
PAYLOAD = WORK_DIR / "payload.zip"

DATASET_SOURCE = "keith1101/ph-showoa"
ACCOUNTS = [str(n) for n in range(1, 11)]       # n = kaggle/<n>_account_<username>/
SLOTS_PER_ACCOUNT = 2
RUNS = 30
MAX_ITER = 1000
POP_SIZE = 30
MAX_RETRIES = 2
JIT_S = 90                   # kernel compile inside run 1
SESSION_OVERHEAD_H = 0.2     # start-up, payload, nvidia-smi
QUOTA_SAFETY = 1.3           # dispatch only if remaining GPU hours >= estimate * safety
STALE_GUARD_S = 180          # ignore a COMPLETE status this soon after a push unless it was seen running

# Estimated seconds per run on a T4 (fullcaitienGPU of the ablation, largest instance of the class x 1.5)
EST_RUN_S = {"C1": 20, "C2": 110, "R1": 35, "R2": 195, "RC1": 30, "RC2": 170, "small": 15}


# --------------------------------------------------------------------------------------
# Logging / state
# --------------------------------------------------------------------------------------
def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"workers": {}, "retries": {}, "failed": {}, "done": {}, "hours": [], "quota_cache": {},
            "account_ok": [], "account_broken": {}, "code_hash": None}


def save_state(state: dict):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


# --------------------------------------------------------------------------------------
# Kaggle CLI
# --------------------------------------------------------------------------------------
def kaggle_exe() -> str:
    for p in (REPO_ROOT / ".venv" / "Scripts" / "kaggle.exe", REPO_ROOT / ".venv" / "bin" / "kaggle"):
        if p.exists():
            return str(p)
    found = shutil.which("kaggle")
    if found:
        return found
    raise RuntimeError("kaggle CLI not found")


def account_dir(acc: str) -> Path:
    """Credential folder of account n: kaggle/<n>_account_<username>/."""
    n = int(acc)
    found = sorted((REPO_ROOT / "kaggle").glob(f"{n}_account_*"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one kaggle/{n}_account_*/ folder for account {acc!r}, found {len(found)}")
    return found[0]


def account_env(acc: str) -> dict:
    cfg = account_dir(acc)
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(cfg)
    env.pop("KAGGLE_API_TOKEN", None)
    with open(cfg / "kaggle.json", "r", encoding="utf-8") as f:
        key = json.load(f).get("key", "")
    if key.startswith("KGAT_"):          # new-style access token
        env["KAGGLE_API_TOKEN"] = key
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def account_username(acc: str) -> str:
    with open(account_dir(acc) / "kaggle.json", "r", encoding="utf-8") as f:
        return json.load(f)["username"]


def kcmd(acc: str, args: list, timeout: int = 180) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([kaggle_exe()] + args, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=account_env(acc), timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", "timeout")
    except Exception as e:  # noqa: BLE001
        return subprocess.CompletedProcess(args, 1, "", str(e))


def gpu_quota_remaining(acc: str, state: dict, max_age_s: int = 600):
    cache = state.setdefault("quota_cache", {}).get(acc)
    if cache and time.time() - cache["t"] < max_age_s:
        return cache["remaining"]
    r = kcmd(acc, ["quota"], timeout=90)
    m = re.search(r"^GPU\s+([\d.]+)h\s+([\d.]+)h", r.stdout, re.M)
    if not m:
        log(f"[WARN] could not read GPU quota for account {acc}: {(r.stdout + r.stderr).strip()[:200]}")
        return None
    remaining = float(m.group(2))
    state["quota_cache"][acc] = {"t": time.time(), "remaining": remaining}
    return remaining


def kernel_status(worker: dict) -> str:
    r = kcmd(worker["account"], ["kernels", "status", worker["slug"]], timeout=120)
    text = (r.stdout + " " + r.stderr).strip()
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?([A-Za-z_]+)"?', text)
    return (m.group(1) if m else text).upper()


def download_output(worker: dict, dest: Path) -> str:
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    r = kcmd(worker["account"], ["kernels", "output", worker["slug"], "-p", str(dest)], timeout=600)
    return (r.stdout + r.stderr)


# --------------------------------------------------------------------------------------
# Workers, jobs and batches
# --------------------------------------------------------------------------------------
def build_workers(accounts):
    workers = []
    for acc in accounts:
        user = account_username(acc)
        for slot in range(1, SLOTS_PER_ACCOUNT + 1):
            wid = f"{acc}-{slot}"
            name = f"phshowoa-gpu-bench-{wid}"
            workers.append({"id": wid, "account": acc, "slug": f"{user}/{name}", "title": name,
                            "dir": WORK_DIR / "workers" / wid})
    return workers


def instance_class(name: str, size: str) -> str:
    if size.strip() and int(size) < 100:
        return "small"
    m = re.match(r"(rc|c|r)dp([12])\d\d$", name.strip().lower())
    return f"{m.group(1).upper()}{m.group(2)}" if m else "RC2"


def run_est_s(name: str, size: str) -> int:
    return EST_RUN_S[instance_class(name, size)]


def make_job(inst: dict, runs: int, max_iter: int, pop_size: int) -> dict:
    cfg = {}
    if max_iter != MAX_ITER:
        cfg["max_iter"] = max_iter
    if pop_size != POP_SIZE:
        cfg["pop_size"] = pop_size
    return {"name": "full", "inst": inst["instance"].lower(), "instance": inst["instance"], "cfg": cfg,
            "stem": inst["instance"], "run_lo": 1, "run_hi": runs,
            "est_s": JIT_S + runs * run_est_s(inst["instance"], inst["size"])}


def build_batches(names, insts, runs, max_iter, pop_size):
    """Instances of names paired into batches of 2 of similar length, longest first."""
    by_name = {x["instance"]: x for x in insts}
    jobs = sorted((make_job(by_name[n], runs, max_iter, pop_size) for n in names), key=lambda j: -j["est_s"])
    return [{"batch": "+".join(j["stem"] for j in jobs[i:i + 2]), "jobs": jobs[i:i + 2]}
            for i in range(0, len(jobs), 2)]


def est_hours(batch) -> float:
    return max(j["est_s"] for j in batch["jobs"]) / 3600 + SESSION_OVERHEAD_H


# --------------------------------------------------------------------------------------
# Payload: package + driver
# --------------------------------------------------------------------------------------
def payload_sources():
    files = [(f"pkg/main/{PKG}/{p.name}", p) for p in sorted((REPO_ROOT / PKG).glob("*.py"))]
    files.append(("ablation_driver.py", DRIVER))
    return files


def code_hash() -> str:
    h = hashlib.sha256()
    for arc, p in payload_sources():
        h.update(arc.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def build_payload():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, p in payload_sources():
            z.write(p, arc)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    PAYLOAD.write_bytes(buf.getvalue())


# --------------------------------------------------------------------------------------
# Notebook
# --------------------------------------------------------------------------------------
NOTEBOOK_RUN = r'''
import base64, glob, io, json, os, subprocess, sys, zipfile
ROOT = "/tmp/full"
OUT = "/kaggle/working"
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(ROOT)
subprocess.run(["nvidia-smi", "-L"])
from numba import cuda
if not cuda.is_available():
    raise RuntimeError("NO_GPU: CUDA GPU is not available in this session")
n_gpu = len(cuda.gpus)
print("GPUs:", n_gpu, flush=True)
hits = glob.glob("/kaggle/input/**/Wang_Chen/" + JOBS[0]["inst"] + ".txt", recursive=True)
if not hits:
    raise FileNotFoundError("Wang_Chen/" + JOBS[0]["inst"] + ".txt not found under /kaggle/input")
DATA = os.path.dirname(hits[0])
print("Instances:", DATA, flush=True)

procs = []
for k, job in enumerate(JOBS):
    jf = os.path.join(ROOT, f"job{k}.json")
    json.dump(dict(job, backend="cuda"), open(jf, "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k % n_gpu))
    log = open(os.path.join(OUT, f"{job['stem']}.log"), "w")
    cmd = [sys.executable, os.path.join(ROOT, "ablation_driver.py"), os.path.join(ROOT, "pkg", "main"),
           jf, RES, DATA]
    procs.append((job, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    if len(procs) % n_gpu == 0:                     # more jobs than GPUs: wait for this wave
        for _, p, _ in procs[-n_gpu:]:
            p.wait()
status = {}
for job, p, log in procs:
    p.wait(); log.close()
    ok = p.returncode == 0 and os.path.exists(os.path.join(RES, f"{job['stem']}.csv"))
    status[job["stem"]] = "ok" if ok else f"failed rc={p.returncode}"
summary = {"batch_id": BATCH_ID, "code_hash": CODE_HASH, "status": status, "n_gpu": n_gpu}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print("SUMMARY:", json.dumps(summary))
for name, s in status.items():
    if s != "ok":
        print(open(os.path.join(OUT, name + ".log")).read()[-3000:])
'''


def make_notebook(batch, batch_id, payload_b64, chash):
    params = (f"BATCH_ID = {batch_id!r}\nCODE_HASH = {chash!r}\nJOBS = {batch['jobs']!r}\n"
              f"PAYLOAD = {payload_b64!r}\n")

    def cell(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": src.strip("\n").splitlines(keepends=True)}

    title = ", ".join(j["stem"] for j in batch["jobs"])
    return {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [f"# PH-SHOWOA GPU benchmark: {title}\n"]},
                      cell(params), cell(NOTEBOOK_RUN)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


def push(worker, batch, batch_id, payload_b64, chash):
    d = worker["dir"]
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    (d / "notebook.ipynb").write_text(json.dumps(make_notebook(batch, batch_id, payload_b64, chash), indent=1),
                                      encoding="utf-8")
    meta = {"id": worker["slug"], "title": worker["title"], "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET_SOURCE],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = kcmd(worker["account"], ["kernels", "push", "-p", str(d)], timeout=300)
    text = (r.stdout + "\n" + r.stderr).strip()
    ok = r.returncode == 0 and "error" not in text.lower() and ("pushed" in text.lower() or "version" in text.lower())
    return ok, text


def read_job_result(dest: Path, job: dict):
    """Rows of the 30 runs of one job, or (None, reason)."""
    p = dest / "results" / f"{job['stem']}.csv"
    if not p.exists():
        return None, "results CSV missing"
    with open(p, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    want = job["run_hi"] - job["run_lo"] + 1
    if len(rows) != want:
        return None, f"incomplete output ({len(rows)}/{want} runs)"
    bad = [r for r in rows if r["feasible"] != "True"]
    if bad:
        return None, f"infeasible runs {[r['run'] for r in bad]}: {bad[0]['violation_note'][:200]}"
    return rows, ""


def collect(worker, ws, batch, smoke):
    """Download a finished kernel and record its instances.
    Returns (ok | None for stale output, message, instances written to the docs CSVs)."""
    dest = RAW_DIR / f"{ws['batch_id']}"
    download_output(worker, dest)
    sp = dest / "summary.json"
    if not sp.exists():
        return False, "summary.json missing", []
    summary = json.loads(sp.read_text(encoding="utf-8"))
    if summary.get("batch_id") != ws["batch_id"]:
        return None, f"stale output ({summary.get('batch_id')})", []
    msgs, all_ok, done = [], True, []
    for job in batch["jobs"]:
        rows, why = read_job_result(dest, job)
        if rows is None:
            all_ok = False
            msgs.append(f"{job['stem']}: {why}")
            continue
        best = min(rows, key=lambda r: (int(r["NV"]), float(r["TD"])))
        avg_t = sum(float(r["elapsed_seconds"]) for r in rows) / len(rows)
        msgs.append(f"{job['stem']}: best NV={best['NV']} TD={float(best['TD']):.2f} avg/run={avg_t:.1f}s "
                    f"(run 1 incl. JIT: {float(rows[0]['elapsed_seconds']):.1f}s)")
        if not smoke:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            for suffix in ("__conv.csv", ".csv"):
                shutil.copy2(dest / "results" / f"{job['stem']}{suffix}", RESULTS_DIR / f"{job['stem']}{suffix}")
            update_runs_csv(job["instance"], rows)
            update_bench_csv(job["instance"], int(best["NV"]), float(best["TD"]), avg_t)
            done.append(job["instance"])
    return all_ok, "; ".join(msgs), done


# --------------------------------------------------------------------------------------
# CSV handling (preserve the existing file layout)
# --------------------------------------------------------------------------------------
def read_bench_instances():
    with open(BENCH_CSV, "r", encoding="utf-8-sig", newline="") as f:
        lines = f.read().splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    i_inst, i_nv, i_td = header.index("Instance"), header.index("best_NV"), header.index("best_TD")
    out = []
    for line in lines[1:]:
        cells = line.split(",")
        if len(cells) <= i_td:
            continue
        name = cells[i_inst].strip()
        done = bool(cells[i_nv].strip()) and bool(cells[i_td].strip())
        size = cells[2].strip()
        out.append({"instance": name, "done": done, "size": size})
    return out


def write_with_retry(path: Path, text: str, encoding: str):
    for attempt in range(30):
        try:
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding=encoding, newline="") as f:
                f.write(text)
            os.replace(tmp, path)
            return
        except PermissionError:
            log(f"[WARN] {path.name} is locked (open in Excel?). Retrying in 20s...")
            time.sleep(20)
    raise PermissionError(f"could not write {path}")


def update_bench_csv(instance: str, best_nv: int, best_td: float, avg_time: float):
    with open(BENCH_CSV, "rb") as f:
        raw = f.read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    header = [h.strip() for h in lines[0].split(",")]
    i_inst, i_nv, i_td, i_t = (header.index("Instance"), header.index("best_NV"),
                               header.index("best_TD"), header.index("Avg/Run (s)"))
    hit = False
    for k in range(1, len(lines)):
        cells = lines[k].split(",")
        if len(cells) > i_t and cells[i_inst].strip().lower() == instance.lower():
            cells[i_nv] = f" {best_nv}"
            cells[i_td] = f" {best_td:.2f}"
            cells[i_t] = f" {avg_time:.2f}"
            lines[k] = ",".join(cells)
            hit = True
    if not hit:
        raise KeyError(f"{instance} not found in {BENCH_CSV.name}")
    write_with_retry(BENCH_CSV, nl.join(lines), "utf-8-sig" if bom else "utf-8")


def update_runs_csv(instance: str, rows: list):
    """rows: the driver CSV of one instance (run, seed, NV, TD, elapsed_seconds = solve() time)."""
    with open(RUNS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        data = list(csv.reader(f))
    header = data[0]
    idx = {h: header.index(h) for h in ("instance", "run", "seed", "NV", "TD", "cost", "time")}
    by_run = {int(r["run"]): r for r in rows}
    filled = 0
    for line in data[1:]:
        while len(line) < len(header):
            line.append("")
        if line[idx["instance"]].strip().lower() != instance.lower():
            continue
        r = by_run.get(int(line[idx["run"]]))
        if r is None:
            continue
        if int(r["seed"]) != int(line[idx["seed"]]):
            raise ValueError(f"seed mismatch for {instance} run {r['run']}: {r['seed']} vs {line[idx['seed']]}")
        nv, td = int(r["NV"]), float(r["TD"])
        line[idx["NV"]] = str(nv)
        line[idx["TD"]] = f"{td:.2f}"
        line[idx["cost"]] = f"{2000 * nv + td:.2f}"
        line[idx["time"]] = f"{float(r['elapsed_seconds']):.2f}"
        filled += 1
    if filled != len(rows):
        raise ValueError(f"{instance}: filled {filled}/{len(rows)} rows in {RUNS_CSV.name}")
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\r\n").writerows(data)
    write_with_retry(RUNS_CSV, buf.getvalue(), "utf-8-sig")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def print_status(state: dict, workers: list):
    insts = read_bench_instances()
    done = sum(1 for x in insts if x["done"])
    print("=" * 78)
    print(f" {done}/{len(insts)} instances done | code {state.get('code_hash')} | "
          f"failed: {list(state.get('failed', {}).keys()) or '-'}")
    for w in workers:
        ws = state["workers"].get(w["id"], {})
        el = ""
        if ws.get("status") == "RUNNING":
            s = int(time.time() - ws["pushed_at"])
            el = f"{s // 3600}h{(s % 3600) // 60:02d}m"
        print(f"  {w['id']:<5} {w['slug']:<45} {ws.get('status', 'IDLE'):<9} {ws.get('batch') or '-':<20} {el}")
    q = state.get("quota_cache", {})
    print(" GPU quota remaining: " + ", ".join(f"{a}={v['remaining']:.1f}h" for a, v in sorted(q.items())))
    print("=" * 78, flush=True)


def orchestrate(accounts, runs, max_iter, pop_size, poll, smoke):
    state = load_state()
    chash = code_hash()
    if state.get("code_hash") and state["code_hash"] != chash and (state.get("done") or any(
            ws.get("status") == "RUNNING" for ws in state["workers"].values())):
        raise SystemExit(f"{PKG} changed since this benchmark started (state {state['code_hash']}, now {chash}); "
                         f"results of two code versions must not be mixed. Restore the code or use a new WORK_DIR.")
    build_payload()                       # always from the current sources, which match chash
    log(f"Built payload {PAYLOAD} ({PAYLOAD.stat().st_size // 1024} KB), code {chash}")
    state["code_hash"] = chash
    payload_b64 = base64.b64encode(PAYLOAD.read_bytes()).decode("ascii")
    workers = build_workers(accounts)
    if smoke:
        workers = workers[:1]
    insts = read_bench_instances()
    if smoke:
        names = ["RCdp1001", "RCdp1004"]
    else:
        names = [x["instance"] for x in insts if not x["done"]]
    batches = build_batches(names, insts, runs, max_iter, pop_size)
    by_id = {b["batch"]: b for b in batches}
    # batches already running (from an earlier start of the orchestrator) keep their definition
    for ws in state["workers"].values():
        if ws.get("status") == "RUNNING" and ws.get("jobs"):
            by_id[ws["batch"]] = {"batch": ws["batch"], "jobs": ws["jobs"]}
    log(f"Start: {len(workers)} workers, {len(batches)} batches, runs={runs}, max_iter={max_iter}, "
        f"pop_size={pop_size}, code={chash}, smoke={smoke}")
    finished = set()
    last_status = 0.0

    while True:
        active = {ws.get("batch") for ws in state["workers"].values() if ws.get("status") == "RUNNING"}
        pending = [b for b in batches if b["batch"] not in active and b["batch"] not in finished
                   and b["batch"] not in state["failed"]]

        # 1) dispatch
        for w in workers:
            ws = state["workers"].get(w["id"], {"status": "IDLE"})
            if ws.get("status") == "RUNNING" or not pending:
                continue
            if ws.get("status") == "NO_QUOTA" and time.time() - ws.get("since", 0) < 3600:
                continue
            if w["account"] in state.setdefault("account_broken", {}):
                continue
            remaining = gpu_quota_remaining(w["account"], state)
            if remaining is not None:
                # quota still to be used by the other running session of this account (2 sessions per account)
                for w2 in workers:
                    ws2 = state["workers"].get(w2["id"], {})
                    if w2["account"] == w["account"] and ws2.get("status") == "RUNNING" and ws2["batch"] in by_id:
                        left = est_hours(by_id[ws2["batch"]]) * QUOTA_SAFETY - (time.time() - ws2["pushed_at"]) / 3600
                        remaining -= max(0.25, left)
            fits = [b for b in pending if remaining is None or est_hours(b) * QUOTA_SAFETY <= remaining]
            if not fits:
                if ws.get("status") != "NO_QUOTA":
                    log(f"[QUOTA] {w['id']}: {remaining:.2f}h left after the running session, no pending batch fits")
                state["workers"][w["id"]] = {"status": "NO_QUOTA", "since": time.time()}
                save_state(state)
                continue
            b = fits[0]
            batch_id = uuid.uuid4().hex[:12]
            ok, text = push(w, b, batch_id, payload_b64, chash)
            if ok:
                pending.remove(b)
                state["workers"][w["id"]] = {"status": "RUNNING", "batch": b["batch"], "batch_id": batch_id,
                                             "jobs": b["jobs"], "pushed_at": time.time(), "seen_running": False}
                log(f"[PUSH] {w['id']} <- {b['batch']} (batch {batch_id}, est {est_hours(b):.2f}h)")
            else:
                log(f"[PUSH FAILED] {w['id']} {b['batch']}: {text[:400]}")
                if "quota" in text.lower():
                    state["workers"][w["id"]] = {"status": "NO_QUOTA", "since": time.time()}
            save_state(state)

        # 2) poll
        for w in workers:
            ws = state["workers"].get(w["id"], {})
            if ws.get("status") != "RUNNING":
                continue
            st = kernel_status(w)
            if st in ("RUNNING", "QUEUED"):
                ws["seen_running"] = True
                continue
            if st == "COMPLETE" and not ws.get("seen_running") and time.time() - ws["pushed_at"] < STALE_GUARD_S:
                continue
            if st not in ("COMPLETE", "ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED", "FAILED"):
                log(f"[WARN] {w['id']} unexpected status {st}")
                continue
            batch = by_id[ws["batch"]]
            ok, msg, done_now = collect(w, ws, batch, smoke)
            if ok is None and time.time() - ws["pushed_at"] < 900:
                continue                           # an old version still reported as latest; wait
            hours = (time.time() - ws["pushed_at"]) / 3600
            state.get("quota_cache", {}).pop(w["account"], None)
            for inst in done_now:
                state.setdefault("done", {})[inst] = ws["batch_id"]
            if ok:
                finished.add(ws["batch"])
                state["hours"].append(round(hours, 3))
                if w["account"] not in state.setdefault("account_ok", []):
                    state["account_ok"].append(w["account"])
                log(f"[DONE] {ws['batch']} on {w['id']} in {hours:.2f}h: {msg}")
                state["workers"][w["id"]] = {"status": "IDLE"}
            else:
                txt = msg + "\n" + kcmd(w["account"], ["kernels", "logs", w["slug"]], timeout=180).stdout[-6000:]
                no_gpu = "NO_GPU" in txt or "quota" in txt.lower()
                if w["account"] not in state.setdefault("account_ok", []) and not no_gpu and not done_now:
                    state["account_broken"][w["account"]] = f"{st}: {msg[-200:]}"
                    log(f"[ACCOUNT DISABLED] account {w['account']} failed before any success (status={st}: "
                        f"{msg[:200]}); '{ws['batch']}' goes back to the queue. Check this account on Kaggle.")
                else:
                    n = state["retries"].get(ws["batch"], 0) + 1
                    state["retries"][ws["batch"]] = n
                    log(f"[FAILED] {ws['batch']} on {w['id']} status={st} retry={n} no_gpu={no_gpu}: {msg[:400]}")
                    if n > MAX_RETRIES and not no_gpu:
                        state["failed"][ws["batch"]] = msg[:300]
                # instances of the batch that did finish are not run again
                rest = [j for j in batch["jobs"] if j["instance"] not in done_now]
                if rest and len(rest) < len(batch["jobs"]):
                    finished.add(ws["batch"])
                    nb = {"batch": "+".join(j["stem"] for j in rest), "jobs": rest}
                    batches.append(nb)
                    by_id[nb["batch"]] = nb
                state["workers"][w["id"]] = ({"status": "NO_QUOTA", "since": time.time()} if no_gpu
                                             else {"status": "IDLE"})
                if smoke:
                    finished.add(ws["batch"])
            save_state(state)

        if time.time() - last_status > 1800:
            print_status(state, workers)
            last_status = time.time()
        running = [ws for wid, ws in state["workers"].items()
                   if ws.get("status") == "RUNNING" and any(w["id"] == wid for w in workers)]
        if not pending and not running:
            log("Smoke test finished." if smoke else "All instances completed (or failed permanently).")
            print_status(state, workers)
            return
        if not running and pending and all(state["workers"].get(w["id"], {}).get("status") == "NO_QUOTA"
                                           or w["account"] in state.get("account_broken", {}) for w in workers):
            log("All accounts are out of GPU quota. Stopping; rerun after the quota refreshes.")
            return
        time.sleep(poll)


def main():
    global STATE_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--max_iter", type=int, default=MAX_ITER)
    ap.add_argument("--pop_size", type=int, default=POP_SIZE)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--accounts", help="comma-separated subset of ACCOUNTS (e.g. accounts without GPU access left out)")
    a = ap.parse_args()
    accounts = [x for x in a.accounts.split(",") if x in ACCOUNTS] if a.accounts else ACCOUNTS
    if a.status:
        print_status(load_state(), build_workers(accounts))
        return
    if a.smoke:
        STATE_FILE = WORK_DIR / "state_smoke.json"
        orchestrate(accounts, runs=2, max_iter=20, pop_size=a.pop_size, poll=60, smoke=True)
        return
    if (a.runs, a.max_iter, a.pop_size) != (RUNS, MAX_ITER, POP_SIZE):
        raise SystemExit("the docs CSVs hold runs=30, max_iter=1000, pop_size=30; use --smoke for a test")
    orchestrate(accounts, a.runs, a.max_iter, a.pop_size, a.poll, smoke=False)


if __name__ == "__main__":
    main()
