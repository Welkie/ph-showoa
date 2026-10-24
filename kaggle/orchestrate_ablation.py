#!/usr/bin/env python3
"""
Kaggle orchestrator for the ablation study of docs/improved.md (sections 2-4) on T4 GPUs.

- 5 accounts (a..e) x 2 concurrent kernels = 10 workers (Kaggle allows at most 2 GPU sessions per account).
- Each push is one batch of up to 2 jobs; a T4 x2 session runs them in parallel, one job per GPU
  (CUDA_VISIBLE_DEVICES), via kaggle/ablation_driver.py. A job = one configuration x one instance, 30 runs,
  seeds 42..71, kernels compiled once per job (run 1 includes the JIT compile).
- The code is not fetched from GitHub: the current src_python_gpu_SA_RCRS_GRASP and the .txt instances are zipped
  once into kaggle/runs_ablation/payload.zip and embedded in every notebook, so every batch runs exactly the same
  code. The variants are SolverConfig fields (--num_islands, --init, --ls_scope, --ls_2opt_star, --ls_strategy).
- Results are downloaded to kaggle/runs_ablation/results/ (<name>__<inst>.csv and <name>__<inst>__conv.csv).
- Before each push the remaining weekly GPU quota of the account is checked.

Usage:
    python kaggle/orchestrate_ablation.py            # run until all jobs are done or quota is exhausted
    python kaggle/orchestrate_ablation.py --status   # print progress and exit
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orchestrate_gpu_all import account_username, download_output, kcmd, kernel_status  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG = "src_python_gpu_SA_RCRS_GRASP"
WORK_DIR = REPO_ROOT / "kaggle" / "runs_ablation"
STATE_FILE = WORK_DIR / "state.json"
LOG_FILE = WORK_DIR / "orchestrator.log"
RESULTS_DIR = WORK_DIR / "results"
RAW_DIR = WORK_DIR / "raw"
PAYLOAD = WORK_DIR / "payload.zip"
ACCOUNTS = ["a", "b", "c", "d", "e"]
SLOTS_PER_ACCOUNT = 2
DEFAULT_EST_HOURS = 3.0
QUOTA_SAFETY = 1.2
MAX_RETRIES = 2
STALE_GUARD_S = 180

# 2 instances per class of the 100-customer Wang & Chen set: Cdp1, Cdp2, Rdp1, Rdp2, RCdp1, RCdp2
ALL12 = ["cdp101", "cdp103", "cdp203", "cdp207", "rdp103", "rdp107", "rdp201", "rdp210",
         "rcdp101", "rcdp104", "rcdp202", "rcdp207"]
SEC2 = ALL12          # every section uses the same 12 instances
SEC3B = ALL12


def build_jobs():
    """Batches of 2 jobs in priority order: section 4 and 3a, then 2, then 3b (docs/improved.md, Ablation)."""
    batches = []
    for inst in ALL12:
        batches.append([("B", inst, {}), ("isl1", inst, {"num_islands": 1})])
        batches.append([("A1", inst, {"ls_scope": "best"}), ("A2", inst, {"ls_2opt_star": False})])
    for i in range(0, len(ALL12), 2):
        batches.append([("A3", inst, {"ls_strategy": "first"}) for inst in ALL12[i:i + 2]])
    for inst in SEC2:
        batches.append([("sa_1000", inst, {"init": "sa"}), ("sa_100", inst, {"init": "sa", "max_iter": 100})])
    shorts = [(f"{k}_{it}", inst, dict({"max_iter": it}, **({"init": "sa"} if k == "sa" else {})))
              for inst in SEC2 for it in (100, 200) for k in ("sa", "grasp") if (k, it) != ("sa", 100)]
    for i in range(0, len(shorts), 2):
        batches.append(shorts[i:i + 2])
    for inst in SEC3B:
        batches.append([("P30_isl3", inst, {"num_islands": 3}), ("P30_isl10", inst, {"num_islands": 10})])
        batches.append([("P60_isl12", inst, {"pop_size": 60, "num_islands": 12}),
                        ("P60_isl1", inst, {"pop_size": 60, "num_islands": 1})])
    return [{"batch": f"b{k:02d}", "jobs": [{"name": n, "variant": "main", "inst": i, "cfg": c} for n, i, c in b]}
            for k, b in enumerate(batches)]


# --------------------------------------------------------------------------------------
# Payload: package + variants + instances + driver
# --------------------------------------------------------------------------------------
def build_payload():
    insts = sorted(set(ALL12))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((REPO_ROOT / PKG).glob("*.py")):
            z.write(p, f"pkg/main/{PKG}/{p.name}")
        for i in insts:
            z.write(REPO_ROOT / "dataset" / f"{i}.txt", f"dataset/{i}.txt")
        z.write(REPO_ROOT / "kaggle" / "ablation_driver.py", "ablation_driver.py")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    PAYLOAD.write_bytes(buf.getvalue())


# --------------------------------------------------------------------------------------
# Notebook
# --------------------------------------------------------------------------------------
NOTEBOOK_RUN = r'''
import base64, io, json, os, subprocess, sys, time, zipfile
ROOT = "/tmp/ab"
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

procs = []
for k, job in enumerate(JOBS):
    jf = os.path.join(ROOT, f"job{k}.json")
    json.dump(dict(job, backend="cuda"), open(jf, "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k % n_gpu))
    log = open(os.path.join(OUT, f"{job['name']}__{job['inst']}.log"), "w")
    cmd = [sys.executable, os.path.join(ROOT, "ablation_driver.py"), os.path.join(ROOT, "pkg", job["variant"]),
           jf, RES, os.path.join(ROOT, "dataset")]
    procs.append((job, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    if len(procs) % n_gpu == 0:                     # more jobs than GPUs: wait for this wave
        for _, p, _ in procs[-n_gpu:]:
            p.wait()
status = {}
for job, p, log in procs:
    p.wait(); log.close()
    ok = p.returncode == 0 and os.path.exists(os.path.join(RES, f"{job['name']}__{job['inst']}.csv"))
    status[f"{job['name']}__{job['inst']}"] = "ok" if ok else f"failed rc={p.returncode}"
summary = {"batch_id": BATCH_ID, "status": status, "n_gpu": n_gpu}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print("SUMMARY:", json.dumps(summary))
for name, s in status.items():
    if s != "ok":
        print(open(os.path.join(OUT, name + ".log")).read()[-3000:])
'''


def make_notebook(batch, batch_id, payload_b64):
    params = f"BATCH_ID = {batch_id!r}\nJOBS = {json.dumps(batch['jobs'])}\nPAYLOAD = {payload_b64!r}\n"

    def cell(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": src.strip("\n").splitlines(keepends=True)}

    title = ", ".join(f"{j['name']} {j['inst']}" for j in batch["jobs"])
    return {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [f"# PH-SHOWOA ablation: {title}\n"]},
                      cell(params), cell(NOTEBOOK_RUN)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"workers": {}, "retries": {}, "failed": {}, "hours": [], "quota": {}}


def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def quota_remaining(acc, state, max_age_s=600):
    c = state["quota"].get(acc)
    if c and time.time() - c["t"] < max_age_s:
        return c["remaining"]
    r = kcmd(acc, ["quota"], timeout=90)
    m = re.search(r"^GPU\s+([\d.]+)h\s+([\d.]+)h", r.stdout, re.M)
    if not m:
        log(f"[WARN] could not read GPU quota for account {acc}")
        return None
    state["quota"][acc] = {"t": time.time(), "remaining": float(m.group(2))}
    return float(m.group(2))


def build_workers():
    workers = []
    for acc in ACCOUNTS:
        user = account_username(acc)
        for slot in range(1, SLOTS_PER_ACCOUNT + 1):
            wid = f"{acc}{slot}"
            name = f"phshowoa-ablation-{wid}"
            workers.append({"id": wid, "account": acc, "slug": f"{user}/{name}", "title": name,
                            "dir": WORK_DIR / "workers" / wid})
    return workers


def batch_done(batch):
    return all((RESULTS_DIR / f"{j['name']}__{j['inst']}.csv").exists() for j in batch["jobs"])


def push(worker, batch, batch_id, payload_b64):
    d = worker["dir"]
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    (d / "notebook.ipynb").write_text(json.dumps(make_notebook(batch, batch_id, payload_b64), indent=1),
                                      encoding="utf-8")
    meta = {"id": worker["slug"], "title": worker["title"], "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = kcmd(worker["account"], ["kernels", "push", "-p", str(d)], timeout=300)
    text = (r.stdout + "\n" + r.stderr).strip()
    ok = r.returncode == 0 and "error" not in text.lower() and ("pushed" in text.lower() or "version" in text.lower())
    return ok, text


def collect(worker, ws, batches_by_id):
    """Download a finished kernel; copy the job CSVs of this batch into RESULTS_DIR. Returns (ok, message)."""
    dest = RAW_DIR / f"{ws['batch']}_{ws['batch_id']}"
    download_output(worker, dest)
    sp = dest / "summary.json"
    if not sp.exists():
        return False, "summary.json missing"
    summary = json.loads(sp.read_text(encoding="utf-8"))
    if summary.get("batch_id") != ws["batch_id"]:
        return None, f"stale output ({summary.get('batch_id')})"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for job in batches_by_id[ws["batch"]]["jobs"]:
        stem = f"{job['name']}__{job['inst']}"
        for suffix in (".csv", "__conv.csv"):
            src = dest / "results" / f"{stem}{suffix}"
            if src.exists():
                shutil.copy2(src, RESULTS_DIR / src.name)
    bad = {k: v for k, v in summary["status"].items() if v != "ok"}
    return (not bad), (f"failed jobs {bad}" if bad else f"n_gpu={summary.get('n_gpu')}")


def print_status(state, workers, batches):
    done = sum(batch_done(b) for b in batches)
    print("=" * 78)
    print(f" {done}/{len(batches)} batches done | failed: {list(state['failed']) or '-'}")
    for w in workers:
        ws = state["workers"].get(w["id"], {})
        el = ""
        if ws.get("status") == "RUNNING":
            s = int(time.time() - ws["pushed_at"])
            el = f"{s // 3600}h{(s % 3600) // 60:02d}m"
        jobs = ""
        if ws.get("batch"):
            jobs = ", ".join(f"{j['name']} {j['inst']}" for j in
                             next(b for b in batches if b["batch"] == ws["batch"])["jobs"])
        print(f"  {w['id']:<3} {ws.get('status', 'IDLE'):<9} {ws.get('batch') or '-':<4} {jobs:<34} {el}")
    q = state.get("quota", {})
    print(" GPU quota remaining: " + ", ".join(f"{a}={v['remaining']:.1f}h" for a, v in sorted(q.items())))
    if state["hours"]:
        print(f" batch wall time (h): {sorted(state['hours'])}")
    print("=" * 78, flush=True)


def orchestrate(poll):
    state = load_state()
    workers = build_workers()
    batches = build_jobs()
    by_id = {b["batch"]: b for b in batches}
    if not PAYLOAD.exists():
        build_payload()
        log(f"Built payload {PAYLOAD} ({PAYLOAD.stat().st_size // 1024} KB)")
    payload_b64 = base64.b64encode(PAYLOAD.read_bytes()).decode("ascii")
    log(f"Start: {len(workers)} workers, {len(batches)} batches")
    last_status = 0.0
    while True:
        active = {ws.get("batch") for ws in state["workers"].values() if ws.get("status") == "RUNNING"}
        pending = [b for b in batches if not batch_done(b) and b["batch"] not in active
                   and b["batch"] not in state["failed"]]
        est = max(state["hours"]) if state["hours"] else DEFAULT_EST_HOURS

        for w in workers:
            ws = state["workers"].get(w["id"], {"status": "IDLE"})
            if ws.get("status") == "RUNNING" or not pending:
                continue
            if ws.get("status") == "NO_QUOTA" and time.time() - ws.get("since", 0) < 3600:
                continue
            remaining = quota_remaining(w["account"], state)
            if remaining is not None and remaining < est * QUOTA_SAFETY:
                if ws.get("status") != "NO_QUOTA":
                    log(f"[QUOTA] {w['id']}: {remaining:.2f}h left < {est * QUOTA_SAFETY:.2f}h needed")
                state["workers"][w["id"]] = {"status": "NO_QUOTA", "since": time.time()}
                save_state(state)
                continue
            b = pending[0]
            batch_id = uuid.uuid4().hex[:12]
            ok, text = push(w, b, batch_id, payload_b64)
            if ok:
                pending.pop(0)
                state["workers"][w["id"]] = {"status": "RUNNING", "batch": b["batch"], "batch_id": batch_id,
                                             "pushed_at": time.time(), "seen_running": False}
                log(f"[PUSH] {w['id']} <- {b['batch']}: " + ", ".join(f"{j['name']} {j['inst']}" for j in b["jobs"]))
            else:
                log(f"[PUSH FAILED] {w['id']} {b['batch']}: {text[:300]}")
            save_state(state)

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
            if st in ("COMPLETE", "ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED", "FAILED"):
                ok, msg = collect(w, ws, by_id)
                if ok is None and time.time() - ws["pushed_at"] < 900:
                    continue
                hours = (time.time() - ws["pushed_at"]) / 3600
                state["quota"].pop(w["account"], None)
                if ok:
                    state["hours"].append(round(hours, 3))
                    log(f"[DONE] {ws['batch']} on {w['id']} in {hours:.2f}h ({msg})")
                else:
                    n = state["retries"].get(ws["batch"], 0) + 1
                    state["retries"][ws["batch"]] = n
                    log(f"[FAILED] {ws['batch']} on {w['id']} status={st}: {msg} (attempt {n})")
                    if n >= MAX_RETRIES:
                        state["failed"][ws["batch"]] = msg
                state["workers"][w["id"]] = {"status": "IDLE"}
                save_state(state)
            else:
                log(f"[WARN] {w['id']} unexpected status {st}")
        save_state(state)

        if time.time() - last_status > 1800:
            print_status(state, workers, batches)
            last_status = time.time()
        if all(batch_done(b) or b["batch"] in state["failed"] for b in batches) and not any(
                ws.get("status") == "RUNNING" for ws in state["workers"].values()):
            log("All batches finished")
            print_status(state, workers, batches)
            return
        time.sleep(poll)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()
    if args.status:
        print_status(load_state(), build_workers(), build_jobs())
    else:
        orchestrate(args.poll)
