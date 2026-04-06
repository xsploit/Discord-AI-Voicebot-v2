param(
    [string]$Python = "python",
    [switch]$SkipVenv,
    [switch]$InstallCudaTorch
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Resolve-Python {
    param([string]$BasePython)

    if (-not $SkipVenv) {
        $VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
        if (-not (Test-Path $VenvPython)) {
            Write-Host "Creating virtual environment..."
            & $BasePython -m venv .venv
        }
        return $VenvPython
    }

    return $BasePython
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Action
    )

    Write-Host $Label
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

$PyExe = Resolve-Python -BasePython $Python

Write-Host "Using Python: $PyExe"

Invoke-Step "Upgrading pip..." { & $PyExe -m pip install --upgrade pip }
Invoke-Step "Installing requirements..." { & $PyExe -m pip install -r requirements.txt }

if ($InstallCudaTorch) {
    Invoke-Step "Installing CUDA PyTorch..." { & $PyExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 }
}

Write-Host ""
Write-Host "Running environment doctor..."
Invoke-Step "Running doctor..." { & $PyExe doctor_bot.py }

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next steps:"
Write-Host "1. Copy .env.example to .env"
Write-Host "2. Fill in your Discord token and Inworld key"
Write-Host "3. Start LM Studio and load a model"
Write-Host "4. Run: $PyExe Main.py"
