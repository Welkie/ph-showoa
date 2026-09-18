"""
Native CUDA C++ Extension Bridge for PH-SHOWOA VRPSDPTW Solver.
Provides 100% native CUDA hardware execution on NVIDIA GPU environments (e.g. Kaggle / Colab / Local).
"""
import os
import sys
import shutil
import subprocess
from typing import Optional, Tuple, List

try:
    import torch
except ImportError:
    torch = None

from .solution import Route


def is_nvcc_available() -> bool:
    """Check if NVIDIA CUDA compiler (nvcc) is available in PATH or CUDA_HOME."""
    if shutil.which("nvcc") is not None:
        return True
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
        return True
    if cuda_home and os.path.exists(os.path.join(cuda_home, "bin", "nvcc.exe")):
        return True
    return False


def check_binary_supports_cuda(binary_path: str) -> bool:
    """Checks if the given binary is executable and was compiled with CUDA full_gpu support."""
    if not os.path.isfile(binary_path):
        return False
    if sys.platform != "win32" and not os.access(binary_path, os.X_OK):
        try:
            os.chmod(binary_path, 0o755)
        except Exception:
            pass
    try:
        res = subprocess.run(
            [binary_path, "--check_cuda"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        combined = (res.stdout or "") + (res.stderr or "")
        if "CUDA_ENABLED" in combined:
            return True
        if "CUDA_DISABLED" in combined or "requires a CUDA-enabled build" in combined:
            return False
        return False
    except Exception:
        return False


def build_or_get_native_binary() -> Optional[str]:
    """
    Locates or builds the standalone native CUDA solver binary (phshowoa_cpp)
    strictly within src_python_gpu_SA_RCRS_GRASP/cpp_native/.
    Returns path to executable if available and compiled with CUDA support.
    """
    local_cpp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "cpp_native"))
    candidate_paths = [
        os.path.join(local_cpp_dir, "build", "phshowoa_cpp"),
        os.path.join(local_cpp_dir, "build", "phshowoa_cpp.exe"),
        os.path.join(local_cpp_dir, "phshowoa_cpp"),
        os.path.join(local_cpp_dir, "phshowoa_cpp.exe"),
    ]

    # 1. Check if any existing binary was already compiled with CUDA
    for p in candidate_paths:
        if os.path.isfile(p) and check_binary_supports_cuda(p):
            return p

    # 2. On Kaggle / Linux with NVCC, auto-compile with CMake strictly in cpp_native
    if os.path.isdir(local_cpp_dir) and is_nvcc_available():
        build_dir = os.path.join(local_cpp_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        bin_target = os.path.join(build_dir, "phshowoa_cpp" + (".exe" if sys.platform == "win32" else ""))

        try:
            print("[CUDA Bridge] Compiling 100% Native CUDA Hardware Solver in cpp_native/...", flush=True)
            subprocess.check_call(
                ["cmake", "-B", build_dir, "-S", local_cpp_dir, "-DENABLE_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"],
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            subprocess.check_call(
                ["cmake", "--build", build_dir, "--config", "Release", "-j"],
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            if os.path.isfile(bin_target):
                if sys.platform != "win32":
                    os.chmod(bin_target, 0o755)
                if check_binary_supports_cuda(bin_target):
                    print("[CUDA Bridge] 100% Native CUDA Solver compiled successfully!", flush=True)
                    return bin_target
                else:
                    print("[CUDA Bridge] Compiled binary does not report CUDA support.", flush=True)
        except Exception as e:
            print(f"[CUDA Bridge] Auto-compile skipped or failed: {e}", flush=True)

    return None


def is_native_cuda_available() -> bool:
    """Check if native CUDA full GPU solver binary is present and verified with CUDA support."""
    bin_path = build_or_get_native_binary()
    if bin_path is not None and check_binary_supports_cuda(bin_path):
        return True
    return False


def run_native_cuda_solver(data, best_s, args_list: Optional[List[str]] = None) -> bool:
    """
    Executes the 100% Native CUDA C++ Solver if binary is available.
    Returns True on success.
    """
    binary = build_or_get_native_binary()
    if not binary:
        return False

    problem_file = getattr(data, "filepath", None)
    if not problem_file or not os.path.isfile(problem_file):
        # Fallback to checking problem_name or dataset folder
        p_name = getattr(data, "problem_name", "")
        for cand in [problem_file, f"dataset/explicit_{p_name}.vrpsdptw", f"../dataset/explicit_{p_name}.vrpsdptw"]:
            if cand and os.path.isfile(cand):
                problem_file = os.path.abspath(cand)
                break
        if not problem_file or not os.path.isfile(problem_file):
            return False

    init_mode = getattr(data, "init", "sa_rcrs_grasp")
    if init_mode not in {"sa_rcrs_grasp", "rcrs_grasp", "rcg", "td", "rcrs", "sa"}:
        init_mode = "sa_rcrs_grasp"

    cmd = [
        binary,
        problem_file,
        "--architecture", "full_gpu",
        "--compute_backend", "cuda",
        "--execution_policy", "cuda_force",
        "--objective", getattr(data, "objective", "lexicographic"),
        "--runs", str(getattr(data, "runs", 1)),
        "--max_iter", str(getattr(data, "max_iter", 1000)),
        "--pop_size", str(getattr(data, "p_size", 36)),
        "--init", init_mode,
        "--grasp_alpha_lo", str(getattr(data, "grasp_alpha_lo", 0.10)),
        "--grasp_alpha_hi", str(getattr(data, "grasp_alpha_hi", 0.40)),
        "--sa_iterations", str(getattr(data, "sa_iterations", 25)),
    ]

    if hasattr(data, "num_islands") and data.num_islands:
        cmd.extend(["--num_islands", str(data.num_islands)])
    if hasattr(data, "migration_interval") and data.migration_interval:
        cmd.extend(["--migration_interval", str(data.migration_interval)])
    if hasattr(data, "seed") and data.seed is not None:
        cmd.extend(["--random_seed", str(data.seed)])
    if getattr(data, "paper_flags", False):
        cmd.append("--paper_flags")

    def _execute_cmd(cmd_to_run):
        print(f"[CUDA Bridge] Executing 100% Native CUDA Solver: {' '.join(cmd_to_run)}", flush=True)
        try:
            proc = subprocess.Popen(
                cmd_to_run,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            total_cost = float("inf")
            nv = 0
            parsed_routes = []
            output_lines = []

            if proc.stdout:
                for line in proc.stdout:
                    output_lines.append(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    if "------------Summary-----------" in line or "Summary-----------" in line:
                        parsed_routes.clear()
                    if "nodes:" in line and "route " in line:
                        parts = line.split("nodes:")
                        if len(parts) > 1:
                            nodes = [int(x) for x in parts[1].strip().split() if x.isdigit()]
                            if len(nodes) > 2:
                                r = Route(data)
                                r.node_list = nodes
                                r.update(data)
                                r.cal_cost(data)
                                parsed_routes.append(r)
                    if "Vehicle count:" in line:
                        parts = line.split("Vehicle count:")
                        if len(parts) > 1:
                            nv = int(parts[1].strip().split()[0])
                    elif "vehicle (route) number:" in line:
                        parts = line.split("vehicle (route) number:")
                        if len(parts) > 1:
                            nv = int(parts[1].strip().split()[0])
                    if "Total cost:" in line:
                        parts = line.split("Total cost:")
                        if len(parts) > 1:
                            val_str = parts[1].strip().split()[0].replace(",", "")
                            try:
                                total_cost = float(val_str)
                            except ValueError:
                                pass
                    elif "This cost " in line:
                        parts = line.split("This cost ")
                        if len(parts) > 1:
                            val_str = parts[1].strip().split(",")[0].split()[0]
                            try:
                                total_cost = float(val_str)
                            except ValueError:
                                pass
                    elif "cost=" in line and total_cost == float("inf"):
                        parts = line.split("cost=")
                        if len(parts) > 1:
                            val_str = parts[1].strip().split()[0].replace(",", "")
                            try:
                                total_cost = float(val_str)
                            except ValueError:
                                pass

            proc.wait()
            return proc.returncode, total_cost, parsed_routes, output_lines
        except Exception as e:
            print(f"[CUDA Bridge] Native execution failed: {e}", flush=True)
            return -1, float("inf"), [], []

    code, total_cost, parsed_routes, lines = _execute_cmd(cmd)

    if code == 0 and total_cost < float("inf"):
        if parsed_routes:
            candidate_s = best_s.clone()
            candidate_s.route_list = parsed_routes
            candidate_s.update(data)
            candidate_s.cal_cost(data)
            # The native full-GPU solver validates the device solution before
            # copying it back. Do not re-enter the Python scalar evaluator in
            # full_gpu mode; that would violate the CPU-outside-only contract.
            best_s.copy_from(candidate_s)
            from . import state
            state.best_s_cost = best_s.cost
            state.best_s = best_s
            return True
        else:
            best_s.cost = total_cost
            from . import state
            state.best_s_cost = total_cost
            state.best_s = best_s
            return True

    return False
