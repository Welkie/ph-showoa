param(
    [Parameter(Position=0)]
    [string]$Account = "a",

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$templateDir = Join-Path $repoRoot "syk4y-apt-main\syk4y-apt-main\templates"
$genCli = Join-Path $repoRoot "syk4y-apt-main\syk4y-apt-main\syk4y-lib\gen_snapshot_cli.py"
$targetFolder = Join-Path $repoRoot "kaggle_rtx6000"
$mainPy = Join-Path $targetFolder "main.py"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " [SYK4Y] PACKAGING LOCAL CODEBASE INTO OFFLINE SNAPSHOT..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Generate snapshot script using syk4y engine
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

& $pythonExe $genCli $repoRoot $mainPy $templateDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to generate snapshot script with syk4y."
    exit 1
}

# 2. Append solver execution snippet to main.py
$runnerSnippet = @"

# ==============================================================================
# AUTO-GENERATED EXECUTION RUNNER FOR KAGGLE OFFLINE RUN
# ==============================================================================
import os, sys, subprocess
from pathlib import Path

restored_dir = (Path.cwd() / 'PH-SHOWOA').resolve()
print(f">> Project root restored at: {restored_dir}", flush=True)
sys.path.insert(0, str(restored_dir))

# Find test instance or run benchmark
test_instance = restored_dir / "dataset" / "explicit_rdp103.vrpsdptw"
if test_instance.exists():
    print(f">> Found instance: {test_instance}, launching GPU solver...", flush=True)
    cmd = [
        sys.executable, "-u", "-m", "src_python_gpu_SA_RCRS_GRASP.main",
        "--problem", str(test_instance),
        "--runs", "1",
        "--max_iter", "50",
        "--pop_size", "32",
        "--paper_flags",
        "--compute_backend", "auto"
    ]
    subprocess.run(cmd, cwd=str(restored_dir))
else:
    print(">> Warning: dataset directory not found in restored files.", flush=True)
"@

Add-Content -Path $mainPy -Value $runnerSnippet -Encoding UTF8
Write-Host ">> Appended solver execution hook into $mainPy" -ForegroundColor Green

# 3. Push to Kaggle
Write-Host "`n>> Pushing to Kaggle via account '$Account'..." -ForegroundColor Cyan
& (Join-Path $scriptDir "run.ps1") $Account kernels push -p ./kaggle_rtx6000 @ExtraArgs
