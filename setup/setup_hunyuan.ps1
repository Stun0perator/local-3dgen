# Installs Tencent Hunyuan3D-2.1 (shape model only) for Local 3D Gen.
# Needs: git, uv (https://docs.astral.sh/uv/), an NVIDIA GPU with ~10 GB+ memory.
# Usage:  powershell -ExecutionPolicy Bypass -File setup\setup_hunyuan.ps1 [-Dir C:\path\Hunyuan3D]
param([string]$Dir = "$env:USERPROFILE\Hunyuan3D")
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path "$Dir\hy3dshape")) {
    git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 $Dir
}
Push-Location $Dir
try {
    if (-not (Test-Path ".venv")) { uv venv --python 3.11 .venv }
    # cu128 wheels include RTX 50-series (sm_120) kernels and still support older cards
    uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv\Scripts\python.exe -r "$here\requirements-hunyuan.txt"
    .venv\Scripts\python.exe -c "import torch; print('CUDA OK:', torch.cuda.get_device_name(0))"
} finally { Pop-Location }
Write-Host "Done. The model weights (tencent/Hunyuan3D-2.1, several GB) download on first use."
