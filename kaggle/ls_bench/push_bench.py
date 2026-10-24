"""Push the LS benchmark to one Kaggle T4 session, wait for it and download ls_bench.json + the log.

usage: python kaggle/ls_bench/push_bench.py [account]   (default account a)
"""
import base64
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
from orchestrate_gpu_all import account_username, download_output, kcmd, kernel_status  # noqa: E402

INSTS = ["cdp101", "rcdp101", "rcdp104", "rcdp202", "rdp210"]
acc = sys.argv[1] if len(sys.argv) > 1 else "a"
work = HERE / "kaggle_run"

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted((REPO / "src_python_gpu_SA_RCRS_GRASP").glob("*.py")):
        z.write(p, f"src_python_gpu_SA_RCRS_GRASP/{p.name}")
    for name in ("bi_ls.py", "ls_bench.py"):
        z.write(HERE / name, name)
    for i in INSTS:
        z.write(HERE / "inputs" / f"{i}.npz", f"inputs/{i}.npz")
        z.write(REPO / "dataset" / f"{i}.txt", f"dataset/{i}.txt")
payload = base64.b64encode(buf.getvalue()).decode("ascii")

code = f'''
import base64, io, os, subprocess, sys, zipfile
PAYLOAD = {payload!r}
ROOT = "/tmp/lsb"
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(ROOT)
subprocess.run(["nvidia-smi", "-L"])
env = dict(os.environ, BENCH_OUT="/kaggle/working", CUDA_VISIBLE_DEVICES="0")
r = subprocess.run([sys.executable, os.path.join(ROOT, "ls_bench.py"), ROOT, os.path.join(ROOT, "inputs"),
                    os.path.join(ROOT, "dataset")] + {INSTS!r}, env=env, capture_output=True, text=True)
open("/kaggle/working/ls_bench.log", "w").write(r.stdout + r.stderr)
print(r.stdout[-20000:]); print(r.stderr[-5000:])
'''
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": code.strip("\n").splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
(work / "notebook.ipynb").write_text(json.dumps(nb), encoding="utf-8")
slug = f"{account_username(acc)}/phshowoa-lsbench"
meta = {"id": slug, "title": "phshowoa-lsbench", "code_file": "notebook.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}
(work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
r = kcmd(acc, ["kernels", "push", "-p", str(work)], timeout=300)
print((r.stdout + r.stderr).strip()[-500:], flush=True)
worker = {"account": acc, "slug": slug}
t0 = time.time()
seen = False
while True:
    time.sleep(60)
    st = kernel_status(worker)
    print(f"[{(time.time() - t0) / 60:5.1f} min] {st}", flush=True)
    if st in ("RUNNING", "QUEUED"):
        seen = True
        continue
    if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED", "FAILED") and (seen or time.time() - t0 > 300):
        out = HERE / "kaggle_out"
        download_output(worker, out)
        log = out / "ls_bench.log"
        print(log.read_text(encoding="utf-8", errors="replace")[-15000:] if log.exists() else "no log", flush=True)
        break
