@echo off
rem Builds Hunyuan3D-Paint's CUDA rasterizer (hy3dgen/texgen/custom_rasterizer) into the Hunyuan3D-2 venv.
rem Needs: NVIDIA CUDA Toolkit 12.8 and Visual Studio 2022 Build Tools (C++).
rem Usage: setup\build_paint_ext.cmd [C:\path\Hunyuan3D-2]
setlocal
set H3D=%~1
if "%H3D%"=="" set H3D=%USERPROFILE%\Hunyuan3D-2
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_HOME%\bin;%PATH%
rem compile for the card in this PC (RTX 50 = 12.0); override by setting TORCH_CUDA_ARCH_LIST first
if "%TORCH_CUDA_ARCH_LIST%"=="" set TORCH_CUDA_ARCH_LIST=12.0
set DISTUTILS_USE_SDK=1
cd /d "%H3D%\hy3dgen\texgen\custom_rasterizer" || exit /b 1
rem Newer torch headers (torch/extension.h -> dynamo/compiled_autograd.h) don't compile under nvcc+MSVC
rem ("'std': ambiguous symbol"). The .cu file only needs tensor types, so give it torch/types.h instead.
"%H3D%\.venv\Scripts\python.exe" -c "p='lib/custom_rasterizer_kernel/rasterizer.h'; s=open(p).read(); a='#include <torch/extension.h>\n'; b='#ifdef __CUDACC__\n#include <torch/types.h>\n#else\n#include <torch/extension.h>\n#endif\n'; open(p,'w').write(s.replace(a,b,1)) if '__CUDACC__' not in s else None"
"%H3D%\.venv\Scripts\python.exe" -m pip --version >nul 2>&1 || uv pip install --python "%H3D%\.venv\Scripts\python.exe" pip setuptools wheel
"%H3D%\.venv\Scripts\python.exe" -m pip install --no-build-isolation .
