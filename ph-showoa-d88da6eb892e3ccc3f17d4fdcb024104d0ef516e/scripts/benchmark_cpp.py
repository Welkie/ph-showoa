import subprocess
import time
import csv
import re
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TARGETS_CSV = ROOT / "tests" / "paper_performance_targets.csv"
SOLVER_EXE = ROOT / "src_cpp" / "phshowoa_cpp.exe"

DEFAULT_INSTANCES = [
    "RCdp1001",
    "RCdp5001",
    "RCdp5007",
    "RCdp5004",
    "RCdp101",
    "Cdp103",
    "Rcdp205",
    "Rdp210",
    "Rcdp207",
    "Rcdp202",
    "Rdp103",
    "cdp104",
    "cdp102",
    "rcdp205",
    "rdp203",
    "rcdp104",
]

def load_targets():
    targets = {}
    with open(TARGETS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst_name = row["instance"].strip().lower()
            targets[inst_name] = {
                "instance": row["instance"].strip(),
                "path": str(ROOT / row["path"].strip()),
                "target_vehicles": int(row["target_vehicles"]),
                "target_distance": float(row["target_distance"]),
            }
    return targets

def run_solver(instance_name, instance_path, seed, repeat):
    # Create directory structure
    run_dir = ROOT / "batch_results" / "cpp_paper_run_30r_all" / instance_name / "runs" / f"run_{repeat+1:02d}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    solution_path = run_dir / "solution.txt"
    log_path = run_dir / "log.txt"
    summary_path = run_dir / "summary.json"
    
    cmd = [
        str(SOLVER_EXE),
        instance_path,
        "--max_iter", "1000",
        "--pop_size", "30",
        "--random_seed", str(seed),
        "--runs", "1",
        "--output", str(solution_path),
        "--paper_flags"
    ]
    
    start_t = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration = time.perf_counter() - start_t
    
    stdout = result.stdout
    
    # Save stdout as log.txt
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(stdout)
        
    nv_match = None
    tc_match = None
    if solution_path.exists():
        with open(solution_path, "r", encoding="utf-8") as f:
            sol_content = f.read()
        nv_match = re.search(r"vehicle \(route\) number:\s*(\d+)", sol_content)
        tc_match = re.search(r"Total cost:\s*([\d.]+)", sol_content)
    
    if not nv_match or not tc_match:
        print("STDOUT:", stdout)
        if solution_path.exists():
            with open(solution_path, "r", encoding="utf-8") as f:
                print("SOLUTION FILE:", f.read())
        raise RuntimeError(f"Failed to parse output for seed {seed}")
        
    nv = int(nv_match.group(1))
    tc = float(tc_match.group(1))
    td = tc - nv * 2000.0
    
    # Save summary.json
    import json
    summary_data = {
        "seed": seed,
        "vehicles": nv,
        "distance": td,
        "total_cost": tc,
        "duration_sec": duration,
        "status": "success",
        "command": " ".join(cmd),
        "run_dir": str(run_dir),
        "solution_path": str(solution_path),
        "log_path": str(log_path)
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
        
    return nv, td, tc, duration

def main():
    print("Starting C++ Benchmark Suite...", flush=True)
    targets = load_targets()
    
    results = []
    
    for idx, inst in enumerate(DEFAULT_INSTANCES, 1):
        target = targets.get(inst.lower())
        if not target:
            print(f"Warning: target not found for {inst}", flush=True)
            continue
            
        print(f"[{idx}/{len(DEFAULT_INSTANCES)}] Running {target['instance']}...", flush=True)
        
        runs_nv = []
        runs_td = []
        runs_tc = []
        runs_time = []
        
        for repeat in range(30):
            seed = 42 + repeat
            nv, td, tc, duration = run_solver(target["instance"], target["path"], seed, repeat)
            runs_nv.append(nv)
            runs_td.append(td)
            runs_tc.append(tc)
            runs_time.append(duration)
            print(f"      run {repeat+1:02d}/30 seed={seed} -> NV={nv} TD={td:.2f} cost={tc:.2f} ({duration:.3f}s)", flush=True)
            
        avg_nv = mean(runs_nv)
        avg_td = mean(runs_td)
        avg_tc = mean(runs_tc)
        avg_time = mean(runs_time)
        
        best_nv = min(runs_nv)
        best_td = min(runs_td)
        best_tc = min(runs_tc)
        
        # Calculate gaps
        paper_td = target["target_distance"]
        gap_avg_pct = ((avg_td - paper_td) / paper_td) * 100.0
        gap_best_pct = ((best_td - paper_td) / paper_td) * 100.0
        
        results.append({
            "instance": target["instance"],
            "paper_nv": target["target_vehicles"],
            "paper_td": paper_td,
            "avg_nv": avg_nv,
            "avg_td": avg_td,
            "best_nv": best_nv,
            "best_td": best_td,
            "gap_avg_pct": gap_avg_pct,
            "gap_best_pct": gap_best_pct,
            "avg_time": avg_time,
        })
        
        print(f"   Done in {sum(runs_time):.2f}s | Avg TD={avg_td:.2f} (Gap {gap_avg_pct:+.2f}%) | Best TD={best_td:.2f} (Gap {gap_best_pct:+.2f}%)\n", flush=True)
        
    # Write Markdown Report
    report_path = ROOT / "C++_paper_benchmark_results.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# C++ PH-SHOWOA Benchmark Results with Paper Flags (30 repeats, pop_size=30)\n\n")
        f.write("Below are the performance and quality targets compared against the paper benchmarks. Runs were executed in parallel using C++ CPU thread updates.\n\n")
        f.write("| Instance | Paper NV | Paper TD | C++ Avg NV | C++ Avg TD | C++ Best NV | C++ Best TD | Gap Avg | Gap Best | Avg Time/Run |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            f.write(f"| **{r['instance']}** | {r['paper_nv']} | {r['paper_td']:.2f} | {r['avg_nv']:.2f} | {r['avg_td']:.2f} | {r['best_nv']} | {r['best_td']:.2f} | {r['gap_avg_pct']:+.2f}% | {r['gap_best_pct']:+.2f}% | {r['avg_time']:.3f}s |\n")
            
    print(f"\nAll instances completed! Report written to: {report_path}", flush=True)

if __name__ == "__main__":
    main()
