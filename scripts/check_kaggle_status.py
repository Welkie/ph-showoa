import os
import sys
import json
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def check_accounts():
    kaggle_dir = Path(r"d:\AI_Research\PH-SHOWOA\kaggle")
    python_exe = sys.executable
    kaggle_exe = Path(sys.executable).parent / "kaggle.exe"
    if not kaggle_exe.exists():
        kaggle_cmd = [python_exe, "-m", "kaggle"]
    else:
        kaggle_cmd = [str(kaggle_exe)]

    accounts = sorted([d for d in kaggle_dir.iterdir() if d.is_dir() and (d / "kaggle.json").exists()])
    
    print("=" * 70)
    print(f"KAGGLE ACCOUNTS STATUS REPORT ({len(accounts)} accounts found)")
    print("=" * 70)

    for acc in accounts:
        acc_name = acc.name
        # Read username from kaggle.json
        try:
            with open(acc / "kaggle.json") as f:
                creds = json.load(f)
                username = creds.get("username", "unknown")
        except Exception:
            username = "unknown"

        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(acc)

        print(f"\n[{acc_name.upper()}] Username: @{username} (Path: {acc})")
        print("-" * 50)

        # 1. Quota
        print("[GPU / TPU Quota]")
        p_quota = subprocess.run(kaggle_cmd + ["quota"], env=env, capture_output=True, text=True)
        if p_quota.returncode == 0:
            for line in p_quota.stdout.strip().splitlines():
                print(f"  {line}")
        else:
            print(f"  Error fetching quota: {p_quota.stderr.strip()}")

        # 2. Kernels
        print("\n[Recent Kernels]")
        p_kernels = subprocess.run(kaggle_cmd + ["kernels", "list", "--mine", "--page-size", "5"], env=env, capture_output=True, text=True)
        if p_kernels.returncode == 0:
            lines = p_kernels.stdout.strip().splitlines()
            if lines:
                for line in lines[:8]:
                    s = line.strip()
                    if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
                        try:
                            import ast
                            decoded = ast.literal_eval(s)
                            if isinstance(decoded, bytes):
                                line = decoded.decode('utf-8', errors='replace')
                        except Exception:
                            pass
                    print(f"  {line}")
            else:
                print("  No kernels found.")
        else:
            print(f"  Error fetching kernels: {p_kernels.stderr.strip()}")

        # Check status for kernels with SHOWOA / VRPSPDTW in title if any
        # or top 2 kernels
        p_kernels_all = subprocess.run(kaggle_cmd + ["kernels", "list", "--mine", "--page-size", "15"], env=env, capture_output=True, text=True)
        if p_kernels_all.returncode == 0:
            print("\n[Kernel Run Status (Target / Active)]")
            k_lines = p_kernels_all.stdout.strip().splitlines()
            checked_any = False
            for line in k_lines[2:]:  # skip headers
                parts = line.strip().split()
                if not parts:
                    continue
                k_ref = parts[0].strip("b'").strip("'").strip('"')
                if any(tag in k_ref.lower() for tag in ["showoa", "grasp", "vrp"]):
                    p_stat = subprocess.run(kaggle_cmd + ["kernels", "status", k_ref], env=env, capture_output=True, text=True)
                    if p_stat.returncode == 0:
                        stat_text = p_stat.stdout.strip()
                    else:
                        err_line = p_stat.stderr.strip().splitlines()[-1] if p_stat.stderr.strip() else "Error checking status"
                        stat_text = f"Status unavailable ({err_line})"
                    print(f"  * {k_ref}: {stat_text}")
                    checked_any = True
            if not checked_any:
                print("  No SHOWOA-related kernels found.")

        print("=" * 70)

if __name__ == "__main__":
    check_accounts()
