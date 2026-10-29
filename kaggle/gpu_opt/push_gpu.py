"""Run profile_kernels.py for one or more package copies in one Kaggle T4 session and download the log.

usage: python kaggle/gpu_opt/push_gpu.py <account> <tag> <gens> <runs> <pkg>[:K=V,...] [...] -- <inst...>
(K=V: environment of that run, e.g. OPT=0 or POP=120)
Each <pkg dir> is a directory under kaggle/gpu_opt/ or the repo root that holds a copy of the solver package;
all copies run in the same session so their timings are comparable. Output: kaggle/gpu_opt/out_<tag>/profile.log
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

acc, tag, gens, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sep = sys.argv.index("--")
pkgs, insts = sys.argv[5:sep], sys.argv[sep + 1:]

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for d in sorted({spec.split(":")[0] for spec in pkgs}):
        src = (HERE / d) if (HERE / d).exists() else (REPO / d)
        for p in sorted(src.glob("*.py")):
            z.write(p, f"{src.name}/{p.name}")
    z.write(HERE / "profile_kernels.py", "profile_kernels.py")
    for i in insts:
        z.write(REPO / "dataset" / f"{i}.txt", f"dataset/{i}.txt")
payload = base64.b64encode(buf.getvalue()).decode("ascii")
names = [Path(spec.split(":")[0]).name + (":" + spec.split(":")[1] if ":" in spec else "") for spec in pkgs]

code = f'''
import base64, io, os, subprocess, sys, zipfile
PAYLOAD = {payload!r}
ROOT = "/tmp/gpo"
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(ROOT)
subprocess.run(["nvidia-smi", "-L"])
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", BACKEND="cuda", PROBE="1")
log = open("/kaggle/working/profile.log", "w")
for spec in {names!r}:
    pkg, _, extra = spec.partition(":")
    e = dict(env, **dict(kv.split("=") for kv in extra.split(",") if kv))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "profile_kernels.py"), ROOT, pkg, os.path.join(ROOT, "dataset"),
                        "{gens}", "{runs}"] + {insts!r}, env=e, capture_output=True, text=True)
    log.write(r.stdout + r.stderr[-5000:]); log.flush()
    print(r.stdout[-20000:]); print(r.stderr[-3000:])
'''
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": code.strip("\n").splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
work = HERE / f"run_{tag}"
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
(work / "notebook.ipynb").write_text(json.dumps(nb), encoding="utf-8")
slug = f"{account_username(acc)}/phshowoa-gpuopt-{tag}"
meta = {"id": slug, "title": f"phshowoa-gpuopt-{tag}", "code_file": "notebook.ipynb", "language": "python",
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
        out = HERE / f"out_{tag}"
        download_output(worker, out)
        log = out / "profile.log"
        print(log.read_text(encoding="utf-8", errors="replace") if log.exists() else "no log", flush=True)
        break
