# Adds Hunyuan3D-Paint (clean, de-lit colours for AMS printing) to the Hunyuan3D-2 install.
# Run setup_hunyuan_mv.ps1 first. Needs:
#   - NVIDIA CUDA Toolkit 12.8:  winget install --id Nvidia.CUDA --version 12.8
#   - Visual Studio 2022 Build Tools with "Desktop development with C++"
# Usage:  powershell -ExecutionPolicy Bypass -File setup\setup_paint.ps1 [-Dir C:\path\Hunyuan3D-2]
param([string]$Dir = "$env:USERPROFILE\Hunyuan3D-2")
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe")) {
    throw "CUDA Toolkit 12.8 not found. Install it with: winget install --id Nvidia.CUDA --version 12.8"
}
uv pip install --python "$Dir\.venv\Scripts\python.exe" pygltflib xatlas pybind11 ninja pip setuptools wheel
cmd /c "`"$here\build_paint_ext.cmd`" `"$Dir`""
if ($LASTEXITCODE -ne 0) { throw "Building custom_rasterizer failed" }
& "$Dir\.venv\Scripts\python.exe" -c "import torch, custom_rasterizer; print('Hunyuan3D-Paint ready on', torch.cuda.get_device_name(0))"
Write-Host "Done. Weights (tencent/Hunyuan3D-2: delight + paint turbo, several GB) download on first use."
