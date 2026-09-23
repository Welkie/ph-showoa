if ($args.Count -eq 0) {
    Write-Error "Usage: run.ps1 <a|b|c> <kaggle arguments...>"
    exit 1
}

$Account = $args[0]
$KaggleArgs = if ($args.Count -gt 1) { $args[1..($args.Count - 1)] } else { @() }

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Normalize account name
$accMap = @{
    "a" = "account_a"
    "b" = "account_b"
    "c" = "account_c"
    "account_a" = "account_a"
    "account_b" = "account_b"
    "account_c" = "account_c"
}

if (-not $accMap.ContainsKey($Account.ToLower())) {
    Write-Error "Unknown account '$Account'. Choose from: a, b, c (or account_a, account_b, account_c)"
    exit 1
}

$accFolder = Join-Path $scriptDir $accMap[$Account.ToLower()]
$env:KAGGLE_CONFIG_DIR = $accFolder

$kaggleExe = Join-Path (Split-Path -Parent $scriptDir) ".venv\Scripts\kaggle.exe"
if (-not (Test-Path $kaggleExe)) {
    $kaggleExe = "kaggle"
}

Write-Host ">> Using Kaggle Account: $($accMap[$Account.ToLower()]) ($accFolder)" -ForegroundColor Cyan
Write-Host ">> Executing: $kaggleExe $($KaggleArgs -join ' ')" -ForegroundColor DarkGray
& $kaggleExe $KaggleArgs
