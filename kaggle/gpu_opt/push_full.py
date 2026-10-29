"""Push full_run.py for one package copy to a Kaggle T4 x2 session (instances split over the two GPUs) and download
the result CSVs to kaggle/gpu_opt/full_<tag>/.

usage: python kaggle/gpu_opt/push_full.py <account> <tag> <pkg dir> <specs.json | name> -- <inst...>
specs.json: list of {"name", "cfg": SolverConfig fields, "i2": improvement2 constants}, run one after another.
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

acc, tag, pkg_dir, spec_arg = sys.argv[1:5]
specs = (json.loads(Path(spec_arg).read_text(encoding="utf-8")) if spec_arg.endswith(".json")
         else [{"name": spec_arg}])
insts = sys.argv[sys.argv.index("--") + 1:]
src = (HERE / pkg_dir) if (HERE / pkg_dir).exists() else (REPO / pkg_dir)
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.glob("*.py")):
        z.write(p, f"{src.name}/{p.name}")
    z.write(HERE / "full_run.py", "full_run.py")
    for i in insts:
        z.write(REPO / "dataset" / f"{i}.txt", f"dataset/{i}.txt")
payload = base64.b64encode(buf.getvalue()).decode("ascii")
halves = [insts[0::2], insts[1::2]]

code = f'''
import base64, io, os, subprocess, sys, zipfile
PAYLOAD = {payload!r}
ROOT = "/tmp/gpf"
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(ROOT)
subprocess.run(["nvidia-smi", "-L"])
import json
logs = [open(f"/kaggle/working/gpu{{g}}.log", "w") for g in (0, 1)]
for spec in {specs!r}:
    procs = []
    for gpu, part in enumerate({halves!r}):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CFG=json.dumps(spec.get("cfg", {{}})),
                   I2=json.dumps(spec.get("i2", {{}})), EP=json.dumps(spec.get("ep", {{}})), RUNS=str(spec.get("runs", 30)))
        procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "full_run.py"), ROOT, "{src.name}",
                                       os.path.join(ROOT, "dataset"), "/kaggle/working/results", spec["name"]] + part,
                                      env=env, stdout=logs[gpu], stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
for f in logs:
    f.close()
print(open("/kaggle/working/gpu0.log").read()[-3000:]); print(open("/kaggle/working/gpu1.log").read()[-3000:])
'''
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": code.strip("\n").splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
work = HERE / f"run_full_{tag}"
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
(work / "notebook.ipynb").write_text(json.dumps(nb), encoding="utf-8")
slug = f"{account_username(acc)}/phshowoa-full-{tag}"
meta = {"id": slug, "title": f"phshowoa-full-{tag}", "code_file": "notebook.ipynb", "language": "python",
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
    time.sleep(120)
    st = kernel_status(worker)
    print(f"[{(time.time() - t0) / 60:5.1f} min] {st}", flush=True)
    if st in ("RUNNING", "QUEUED"):
        seen = True
        continue
    if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED", "FAILED") and (seen or time.time() - t0 > 300):
        out = HERE / f"full_{tag}"
        download_output(worker, out)
        for g in ("gpu0.log", "gpu1.log"):
            p = out / g
            print(p.read_text(encoding="utf-8", errors="replace")[-2000:] if p.exists() else f"no {g}", flush=True)
        break
