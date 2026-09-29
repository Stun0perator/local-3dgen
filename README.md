# Local 3D Gen

Drag images into Blender and get 3D models back, generated **locally on your own GPU** with open
image-to-3D models. No API keys, no cloud, no per-model fees.

- **Two models, one switch:** [Hunyuan3D 2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) (Tencent) and
  [TRELLIS](https://huggingface.co/microsoft/TRELLIS-image-large) (Microsoft).
- **Drag and drop:** drop one or more image files onto the *Image to 3D* sidebar tab; each becomes a model.
- **Stays loaded:** the model loads once per session (the first load takes a few minutes), then each model
  takes about a minute. Blender stays responsive while it works.
- **Cleans up after itself:** the engine exits when Blender closes, including if Blender crashes, so it never
  keeps holding GPU memory.
- **Works on RTX 50-series (Blackwell) under Windows:** TRELLIS runs without xformers, flash-attn or kaolin,
  none of which currently build for sm_120. See [`engine/shims`](engine/shims).

## The models

Weights download automatically the first time you use each model. To fetch them yourself, or read the
licences before you do:

| Model | Weights | Code | Licence | GPU memory |
|---|---|---|---|---|
| Hunyuan3D 2.1 (shape) | [tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) | [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | [Tencent Hunyuan 3D 2.1 Community License](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE): not licensed in the EU, UK or South Korea | ~10 GB |
| TRELLIS image-large | [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) | [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) | [MIT](https://github.com/microsoft/TRELLIS/blob/main/LICENSE) | ~8 GB |

Downloaded automatically as dependencies:
- [DINOv2](https://github.com/facebookresearch/dinov2), used by TRELLIS.
- [U²-Net](https://github.com/xuebinqin/U-2-Net), for background removal through [rembg](https://github.com/danielgatis/rembg).

TRELLIS.2 is not supported: it currently needs Linux and 24 GB or more of GPU memory.

## Requirements

- Windows 10/11 (Linux should work with small path changes, but it's untested)
- NVIDIA GPU, 12 GB+ memory recommended (tested on an RTX 5070 Ti 16 GB)
- [Blender](https://www.blender.org/) 4.2 or newer (tested on 5.1)
- [git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/)
- About 30 GB of disk: two Python environments plus model weights

## Setup

1. Install one or both models. Each gets its own folder and Python environment:
   ```powershell
   powershell -ExecutionPolicy Bypass -File setup\setup_hunyuan.ps1   # -> %USERPROFILE%\Hunyuan3D
   powershell -ExecutionPolicy Bypass -File setup\setup_trellis.ps1   # -> %USERPROFILE%\TRELLIS
   ```
   Pass `-Dir D:\somewhere` to install elsewhere.
2. Install the Blender add-on. Pick one option:
   - **Link it**, so it updates with `git pull`:
     ```powershell
     New-Item -ItemType Junction -Path "$env:APPDATA\Blender Foundation\Blender\<version>\scripts\addons\local_3dgen" -Target "$PWD\addon\local_3dgen"
     ```
   - **Or zip `addon/local_3dgen`** and install the zip with *Edit > Preferences > Add-ons > Install from Disk*.
3. Enable **Local 3D Gen** in Preferences. In its settings, check the repo folder and the model folders.

## Using it

1. In the 3D Viewport press **N** and open the **Image to 3D** tab.
2. Pick **Hunyuan3D 2.1** or **TRELLIS** at the top.
3. **Drop image files onto the panel** (or use **+**). With *Generate on drop* on, they start straight away.
   Otherwise select one and press **Generate**, or press **All**.
4. Models appear at the 3D cursor in an **AI Meshes** collection, scaled so the longest side matches
   *Longest side (mm)*. Variants are laid out side by side.

Tips:
- Use a clean, single-object image with a plain background. The background is removed automatically.
- Try a few seeds (*Variants* 3–4) and keep the best one.
- Hunyuan: raise *Detail* to 512 for finer surfaces. It takes more time and GPU memory.
- Switching models, or pressing ✕, unloads the current model to free GPU memory.
- Each result is also saved as a GLB in `outputs/<image name>/`, next to the cut-out input image.

## Command line

The same engine works from a terminal:
```
python scripts/img2mesh.py photo.png --model trellis --variants 3
```

## How it works

```
Blender add-on (addon/local_3dgen)
   └─ starts engine/server.py with the chosen model's own .venv Python
        ├─ backends/hunyuan.py  → hy3dshape pipeline
        └─ backends/trellis.py  → TRELLIS pipeline (+ shims for xformers / kaolin)
   JSON lines over stdin/stdout: generate requests in; progress, done and error events out
```
The engine watches Blender's process ID and exits when Blender is gone.

## Troubleshooting

- **The engine stops right away:** read `engine_hunyuan.log` or `engine_trellis.log` in the repo folder.
- **"Out of GPU memory":** close other GPU apps, lower *Detail*, or switch models.
- **The first run seems stuck at "Loading model":** it's downloading several GB of weights; later loads are faster.

## Credits

Built on [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) by Tencent and
[TRELLIS](https://github.com/microsoft/TRELLIS) by Microsoft. Generated models are subject to each model's
licence.

## Licence

The code in this repo is [MIT](LICENSE). The models are not covered by it: they come under their own
licences (see [The models](#the-models)), and so do the meshes you generate with them.
