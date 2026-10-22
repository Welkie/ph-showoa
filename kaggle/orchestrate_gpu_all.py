#!/usr/bin/env python3
"""
Kaggle multi-account benchmark orchestrator for src_python_gpu_SA_RCRS_GRASP.

- 5 accounts (a..e) x 2 concurrent kernels = 10 workers, each worker = one private Kaggle notebook.
- Each worker runs ONE instance at a time: pop_size=30, max_iter=1000, 30 runs on a T4 GPU.
- Run r uses `--runs 1 --random_seed 41+r` (seeds 42..71, same as docs/ph_showoa_caitiengpu_all.csv).
  Timing matches the src_python_baseline runner: every run is a fresh `python -m ...main` process,
  timed launch-to-exit (start-up, imports, data loading, Numba JIT compile and the solve).
- The notebook checks out a pinned commit of https://github.com/Welkie/ph-showoa.
- On completion the orchestrator downloads the kernel output and writes:
    * docs/instances_benchmark_pop_size30_runs=30_iter1000.csv  -> best_NV, best_TD, Avg/Run (s)
    * docs/ph_showoa_caitiengpu_all.csv                          -> NV, TD, cost, time of the 30 runs
- Before dispatching, the remaining weekly GPU quota of the account (`kaggle quota`) is checked.

Usage:
    python kaggle/orchestrate_gpu_all.py            # run until everything is done or quota is exhausted
    python kaggle/orchestrate_gpu_all.py --status   # print progress and exit
    python kaggle/orchestrate_gpu_all.py --smoke    # tiny end-to-end test (1 worker per account), no CSV writes
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
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_CSV = REPO_ROOT / "docs" / "instances_benchmark_pop_size30_runs=30_iter1000.csv"
RUNS_CSV = REPO_ROOT / "docs" / "ph_showoa_caitiengpu_all.csv"
WORK_DIR = REPO_ROOT / "kaggle" / "runs_gpu_all"
STATE_FILE = WORK_DIR / "state.json"
LOG_FILE = WORK_DIR / "orchestrator.log"
RESULTS_DIR = WORK_DIR / "results"

GIT_URL = "https://github.com/Welkie/ph-showoa.git"
GIT_COMMIT = "f33b63fa7da189ec91109813e896985261226889"
DATASET_SOURCE = "keith1101/ph-showoa"
ACCOUNTS = ["a", "b", "c", "d", "e"]
SLOTS_PER_ACCOUNT = 2
SEED0 = 42
MAX_RETRIES = 2
DEFAULT_EST_HOURS = 2.0      # conservative runtime estimate before any instance has finished
QUOTA_SAFETY = 1.3           # dispatch only if remaining GPU hours >= estimate * safety
STALE_GUARD_S = 180          # ignore a COMPLETE status this soon after a push unless the job id matches


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
    return {"workers": {}, "retries": {}, "failed": {}, "runtime_hours": {}, "quota_cache": {},
            "account_ok": [], "account_broken": {}}


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


def account_env(acc: str) -> dict:
    cfg = REPO_ROOT / "kaggle" / f"account_{acc}"
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
    with open(REPO_ROOT / "kaggle" / f"account_{acc}" / "kaggle.json", "r", encoding="utf-8") as f:
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


# --------------------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------------------
def build_workers(accounts):
    workers = []
    for acc in accounts:
        user = account_username(acc)
        for slot in range(1, SLOTS_PER_ACCOUNT + 1):
            wid = f"{acc}{slot}"
            name = f"phshowoa-gpu-bench-{wid}"
            workers.append({"id": wid, "account": acc, "slug": f"{user}/{name}", "title": name,
                            "dir": WORK_DIR / "workers" / wid})
    return workers


NOTEBOOK_SETUP = r'''
import os, subprocess, sys
REPO = "/tmp/ph-showoa"
if not os.path.isdir(os.path.join(REPO, "src_python_gpu_SA_RCRS_GRASP")):
    subprocess.run(["git", "init", "-q", REPO], check=True)
    subprocess.run(["git", "-C", REPO, "remote", "add", "origin", GIT_URL], check=True)
    if subprocess.run(["git", "-C", REPO, "fetch", "-q", "--depth", "1", "origin", GIT_COMMIT]).returncode != 0:
        subprocess.run(["git", "-C", REPO, "fetch", "-q", "origin"], check=True)
        subprocess.run(["git", "-C", REPO, "checkout", "-q", GIT_COMMIT], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "checkout", "-q", "FETCH_HEAD"], check=True)
head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("Code commit:", head)
assert head == GIT_COMMIT, "unexpected commit checked out"
subprocess.run(["nvidia-smi"])
from numba import cuda
import torch
if not (cuda.is_available() and torch.cuda.is_available()):
    raise RuntimeError("NO_GPU: CUDA GPU is not available in this session (GPU quota exhausted or accelerator disabled)")
print("GPU:", torch.cuda.get_device_name(0))
'''

NOTEBOOK_SOLVE = r'''
import csv, glob, json, os, re, subprocess, sys, time
OUT = "/kaggle/working"
LOGS = os.path.join(OUT, "logs")
os.makedirs(LOGS, exist_ok=True)

def find_problem(name):
    fn = f"explicit_{name.lower()}.vrpsdptw"
    hits = glob.glob(f"/kaggle/input/**/Wang_Chen/{fn}", recursive=True) or glob.glob(f"/kaggle/input/**/{fn}", recursive=True)
    if not hits:
        local = os.path.join(REPO, "dataset", fn)
        hits = [local] if os.path.isfile(local) else []
    if not hits:
        raise FileNotFoundError(fn)
    return hits[0]

PROBLEM = find_problem(INSTANCE)
print("Problem file:", PROBLEM)

def run_solver(seed, tag):
    # Same timing as the src_python_baseline runner: one fresh process per run, timed from
    # launch to exit (Python start-up, imports, data loading, Numba JIT compile and the solve).
    cmd = [sys.executable, "-m", "src_python_gpu_SA_RCRS_GRASP.main", "--problem", PROBLEM,
           "--compute_backend", "cuda", "--init", "sa_rcrs_grasp", "--paper_flags",
           "--architecture", "python_cuda", "--runs", "1", "--max_iter", str(MAX_ITER),
           "--pop_size", str(POP_SIZE), "--workers", "1", "--random_seed", str(seed)]
    log_path = os.path.join(LOGS, f"{tag}.log")
    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_fp:
        result = subprocess.run(cmd, cwd=REPO, text=True, stdout=log_fp, stderr=subprocess.STDOUT, check=False)
    dt = time.perf_counter() - started
    out = open(log_path, encoding="utf-8", errors="replace").read()
    m = re.search(r"Run 1 finishes \| Run Best: NV=(\d+), TD=([\d.]+)", out)
    if result.returncode != 0 or not m:
        raise RuntimeError(f"{tag} failed (exit code {result.returncode}):\n" + out[-3000:])
    return int(m.group(1)), float(m.group(2)), dt

rows = []
runs_path = os.path.join(OUT, "runs.csv")
for r in range(1, RUNS + 1):
    seed = SEED0 + r - 1
    nv, td, dt = run_solver(seed, f"run{r:02d}_seed{seed}")
    rows.append({"instance": INSTANCE, "run": r, "seed": seed, "NV": nv, "TD": round(td, 4),
                 "cost": round(2000 * nv + td, 4), "time": round(dt, 3)})
    with open(runs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[run {r:2d}/{RUNS}] seed={seed} NV={nv} TD={td:.4f} time={dt:.1f}s", flush=True)

best = min(rows, key=lambda x: (x["NV"], x["TD"]))
summary = {"job_id": JOB_ID, "instance": INSTANCE, "status": "Success", "runs": len(rows),
           "best_NV": best["NV"], "best_TD": best["TD"],
           "avg_time_s": round(sum(x["time"] for x in rows) / len(rows), 3),
           "timing": "per-run fresh process, launch-to-exit wall clock", "commit": GIT_COMMIT}
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("SUMMARY:", json.dumps(summary))
'''


def make_notebook(instance: str, job_id: str, runs: int, max_iter: int, pop_size: int) -> dict:
    params = (f"INSTANCE = {instance!r}\nJOB_ID = {job_id!r}\nRUNS = {runs}\nMAX_ITER = {max_iter}\n"
              f"POP_SIZE = {pop_size}\nSEED0 = {SEED0}\nGIT_URL = {GIT_URL!r}\nGIT_COMMIT = {GIT_COMMIT!r}\n")

    def cell(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": src.strip("\n").splitlines(keepends=True)}

    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {},
             "source": [f"# PH-SHOWOA GPU benchmark: {instance} (pop_size={pop_size}, max_iter={max_iter}, runs={runs})\n"]},
            cell(params), cell(NOTEBOOK_SETUP), cell(NOTEBOOK_SOLVE),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4,
    }


def push(worker: dict, instance: str, job_id: str, runs: int, max_iter: int, pop_size: int):
    d = worker["dir"]
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "notebook.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(instance, job_id, runs, max_iter, pop_size), f, indent=1)
    meta = {"id": worker["slug"], "title": worker["title"], "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [DATASET_SOURCE], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}
    with open(d / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    r = kcmd(worker["account"], ["kernels", "push", "-p", str(d)], timeout=300)
    text = (r.stdout + "\n" + r.stderr).strip()
    ok = r.returncode == 0 and "error" not in text.lower() and ("pushed" in text.lower() or "version" in text.lower())
    return ok, text


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


def read_result(dest: Path, job_id: str, instance: str, runs: int):
    sp, rp = dest / "summary.json", dest / "runs.csv"
    if not (sp.exists() and rp.exists()):
        return None, "summary.json/runs.csv missing"
    with open(sp, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if summary.get("job_id") != job_id or summary.get("instance", "").lower() != instance.lower():
        return None, f"stale output (job {summary.get('job_id')} / {summary.get('instance')})"
    with open(rp, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if summary.get("status") != "Success" or len(rows) != runs:
        return None, f"incomplete output ({len(rows)}/{runs} runs)"
    return {"summary": summary, "rows": rows}, ""


def failure_text(dest: Path) -> str:
    texts = []
    for p in list(dest.glob("*.log")) + list(dest.glob("*.txt")) + list(dest.glob("*.json")):
        try:
            texts.append(p.read_text(encoding="utf-8", errors="ignore")[-4000:])
        except OSError:
            pass
    return "\n".join(texts)


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
        line[idx["time"]] = f"{float(r['time']):.2f}"
        filled += 1
    if filled != len(rows):
        raise ValueError(f"{instance}: filled {filled}/{len(rows)} rows in {RUNS_CSV.name}")
    import io
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\r\n").writerows(data)
    write_with_retry(RUNS_CSV, buf.getvalue(), "utf-8-sig")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def est_hours(state: dict, size: str) -> float:
    vals = state.get("runtime_hours", {}).get(size)
    return max(vals) if vals else DEFAULT_EST_HOURS


def print_status(state: dict, workers: list):
    insts = read_bench_instances()
    done = sum(1 for x in insts if x["done"])
    print("=" * 78)
    print(f" {done}/{len(insts)} instances done | failed: {list(state.get('failed', {}).keys()) or '-'}")
    for w in workers:
        ws = state["workers"].get(w["id"], {})
        el = ""
        if ws.get("status") == "RUNNING":
            s = int(time.time() - ws["pushed_at"])
            el = f"{s // 3600}h{(s % 3600) // 60:02d}m"
        print(f"  {w['id']:<3} {w['slug']:<45} {ws.get('status', 'IDLE'):<9} {ws.get('instance') or '-':<10} {el}")
    q = state.get("quota_cache", {})
    print(" GPU quota remaining: " + ", ".join(f"{a}={v['remaining']:.1f}h" for a, v in sorted(q.items())))
    print("=" * 78, flush=True)


def orchestrate(runs: int, max_iter: int, pop_size: int, poll: int, smoke: bool):
    state = load_state()
    workers = build_workers(ACCOUNTS)
    if smoke:
        workers = [w for w in workers if w["id"].endswith("1")]
    log(f"Start: {len(workers)} workers, runs={runs}, max_iter={max_iter}, pop_size={pop_size}, "
        f"commit={GIT_COMMIT[:7]}, smoke={smoke}")
    smoke_queue = ["RCdp1001"] * len(workers) if smoke else None
    smoke_done = 0

    while True:
        insts = read_bench_instances()
        size_of = {x["instance"].lower(): x["size"] for x in insts}
        active = {ws.get("instance", "").lower() for ws in state["workers"].values()
                  if ws.get("status") == "RUNNING"}
        if smoke:
            pending = smoke_queue
        else:
            pending = [x["instance"] for x in insts if not x["done"] and x["instance"].lower() not in active
                       and x["instance"] not in state["failed"]]

        # 1) dispatch
        for w in workers:
            ws = state["workers"].get(w["id"], {"status": "IDLE"})
            if ws.get("status") == "RUNNING" or not pending:
                continue
            if ws.get("status") == "NO_QUOTA" and time.time() - ws.get("since", 0) < 3600:
                continue
            if w["account"] in state.setdefault("account_broken", {}):
                continue
            inst = pending[0]
            remaining = gpu_quota_remaining(w["account"], state)
            need = 0.2 if smoke else est_hours(state, size_of.get(inst.lower(), "")) * QUOTA_SAFETY
            if remaining is not None and remaining < need:
                if ws.get("status") != "NO_QUOTA":
                    log(f"[QUOTA] {w['id']}: {remaining:.2f}h left < {need:.2f}h needed. Pausing this account.")
                state["workers"][w["id"]] = {"status": "NO_QUOTA", "since": time.time()}
                continue
            job_id = uuid.uuid4().hex[:12]
            ok, text = push(w, inst, job_id, runs, max_iter, pop_size)
            if ok:
                pending.pop(0)
                state["workers"][w["id"]] = {"status": "RUNNING", "instance": inst, "job_id": job_id,
                                             "pushed_at": time.time(), "seen_running": False}
                log(f"[PUSH] {w['id']} <- {inst} (job {job_id})")
            else:
                log(f"[PUSH FAILED] {w['id']} ({inst}): {text[:400]}")
                if "quota" in text.lower():
                    state["workers"][w["id"]] = {"status": "NO_QUOTA", "since": time.time()}
            save_state(state)

        # 2) poll
        for w in workers:
            ws = state["workers"].get(w["id"], {})
            if ws.get("status") != "RUNNING":
                continue
            st = kernel_status(w)
            inst, job_id = ws["instance"], ws["job_id"]
            if st in ("RUNNING", "QUEUED"):
                ws["seen_running"] = True
                continue
            if st == "COMPLETE":
                if not ws.get("seen_running") and time.time() - ws["pushed_at"] < STALE_GUARD_S:
                    continue
                dest = RESULTS_DIR / f"{inst}_{job_id}"
                download_output(w, dest)
                res, why = read_result(dest, job_id, inst, runs)
                if res is None and "stale" in why and time.time() - ws["pushed_at"] < 900:
                    continue                       # an old version still reported as latest; wait
                if res is None:
                    log(f"[RESULT ERROR] {w['id']} {inst}: {why}")
                    st = "ERROR"
                else:
                    s = res["summary"]
                    hours = (time.time() - ws["pushed_at"]) / 3600
                    if smoke:
                        smoke_done += 1
                        log(f"[SMOKE OK] {w['id']}: NV={s['best_NV']} TD={s['best_TD']} avg={s['avg_time_s']}s "
                            f"({hours * 60:.1f} min incl. queue)")
                    else:
                        update_runs_csv(inst, res["rows"])
                        update_bench_csv(inst, int(s["best_NV"]), float(s["best_TD"]), float(s["avg_time_s"]))
                        state.setdefault("runtime_hours", {}).setdefault(size_of.get(inst.lower(), ""), []).append(round(hours, 3))
                        log(f"[DONE] {inst} on {w['id']}: best NV={s['best_NV']} TD={s['best_TD']:.2f} "
                            f"avg/run={s['avg_time_s']:.1f}s, wall={hours:.2f}h")
                    state["workers"][w["id"]] = {"status": "IDLE"}
                    state.get("quota_cache", {}).pop(w["account"], None)
                    if w["account"] not in state.setdefault("account_ok", []):
                        state["account_ok"].append(w["account"])
            if st in ("ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED", "FAILED"):
                dest = RESULTS_DIR / f"{inst}_{job_id}_failed"
                download_output(w, dest)
                txt = failure_text(dest)
                r = kcmd(w["account"], ["kernels", "logs", w["slug"]], timeout=180)
                txt += "\n" + r.stdout[-6000:]
                no_gpu = "NO_GPU" in txt or "quota" in txt.lower()
                unverified = w["account"] not in state.setdefault("account_ok", [])
                if unverified and not no_gpu and not smoke:
                    state.setdefault("account_broken", {})[w["account"]] = f"{st}: {txt.strip()[-200:]}"
                    log(f"[ACCOUNT DISABLED] account {w['account']} failed before any success "
                        f"(status={st}); '{inst}' goes back to the queue. Check this account on Kaggle.")
                    state["workers"][w["id"]] = {"status": "IDLE"}
                    save_state(state)
                    continue
                n = state["retries"].get(inst, 0) + 1
                state["retries"][inst] = n
                log(f"[FAILED] {w['id']} {inst} status={st} retry={n} no_gpu={no_gpu}: "
                    f"{' '.join(txt.strip().splitlines()[-3:])[:300]}")
                if n > MAX_RETRIES and not no_gpu:
                    state["failed"][inst] = st
                state["workers"][w["id"]] = {"status": "NO_QUOTA", "since": time.time()} if no_gpu else {"status": "IDLE"}
                state.get("quota_cache", {}).pop(w["account"], None)
                if smoke:
                    smoke_done += 1
            save_state(state)

        print_status(state, workers)
        running = [ws for wid, ws in state["workers"].items()
                   if ws.get("status") == "RUNNING" and any(w["id"] == wid for w in workers)]
        if smoke and smoke_done >= len(workers) and not running:
            log("Smoke test finished.")
            return
        if not smoke and not pending and not running:
            log("All instances completed (or failed permanently).")
            return
        if not running and pending and all(state["workers"].get(w["id"], {}).get("status") == "NO_QUOTA"
                                           or w["account"] in state.get("account_broken", {}) for w in workers):
            log("All accounts are out of GPU quota. Stopping; rerun after the quota refreshes.")
            return
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--pop_size", type=int, default=30)
    ap.add_argument("--poll", type=int, default=120)
    a = ap.parse_args()
    if a.status:
        print_status(load_state(), build_workers(ACCOUNTS))
        return
    if a.smoke:
        global STATE_FILE
        STATE_FILE = WORK_DIR / "state_smoke.json"
        orchestrate(runs=2, max_iter=20, pop_size=a.pop_size, poll=60, smoke=True)
        return
    orchestrate(a.runs, a.max_iter, a.pop_size, a.poll, smoke=False)


if __name__ == "__main__":
    main()
