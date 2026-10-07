#!/usr/bin/env python3
"""Root forwarder for kaggle/run_gpu_sa_rcrs_grasp.py"""
import sys
from pathlib import Path

# Add kaggle dir to sys.path and invoke main
kaggle_script = Path(__file__).resolve().parent / "kaggle" / "run_gpu_sa_rcrs_grasp.py"
with open(kaggle_script, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(kaggle_script), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(kaggle_script)})
