# PowerShell脚本：使用虚拟环境运行train.py
$ErrorActionPreference = "Stop"

# 激活虚拟环境并运行train.py
& ".\venv\Scripts\python.exe" train.py $args

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError: Training failed. Exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

