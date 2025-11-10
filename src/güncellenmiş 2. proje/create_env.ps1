param(
    [switch]$SkipInstall
)

Write-Host "==> Creating Python virtual environment (.venv)..." -ForegroundColor Cyan
python -m venv .venv

Write-Host "==> Activating virtual environment..." -ForegroundColor Cyan
. ".\.venv\Scripts\Activate.ps1"

if (-not $SkipInstall) {
    Write-Host "==> Upgrading pip and installing requirements..." -ForegroundColor Cyan
    python -m pip install --upgrade pip
    pip install -r requirements.txt
}

Write-Host "==> Ensuring baboon.png exists in project root..." -ForegroundColor Cyan
python ensure_baboon.py

if (Test-Path ".\baboon.png") {
    Write-Host "baboon.png is present." -ForegroundColor Green
} else {
    Write-Host "Could not ensure baboon.png" -ForegroundColor Yellow
}

Write-Host "Done." -ForegroundColor Green


