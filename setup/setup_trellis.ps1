# Installs Microsoft TRELLIS (image-large) for Local 3D Gen.
# Needs: git, uv (https://docs.astral.sh/uv/), an NVIDIA GPU with ~8 GB+ memory.
# No xformers / flash-attn / kaolin needed: engine/shims covers the few calls TRELLIS makes.
# Usage:  powershell -ExecutionPolicy Bypass -File setup\setup_trellis.ps1 [-Dir C:\path\TRELLIS]
param([string]$Dir = "$env:USERPROFILE\TRELLIS")
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path "$Dir\trellis")) {
    git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/microsoft/TRELLIS $Dir
}
Push-Location $Dir
try {
    git submodule update --init --depth 1
    if (-not (Test-Path ".venv")) { uv venv --python 3.11 .venv }
    uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv\Scripts\python.exe -r "$here\requirements-trellis.txt"
    .venv\Scripts\python.exe -c "import torch, spconv.pytorch; print('CUDA OK:', torch.cuda.get_device_name(0))"
} finally { Pop-Location }
Write-Host "Done. The model weights (microsoft/TRELLIS-image-large + DINOv2) download on first use."
