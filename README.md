# Local 3D Gen

Drag images into Blender and get 3D models back, generated **locally on your own GPU** with open
image-to-3D models. No API keys, no cloud, no per-model fees.

- **Three models, one switch:** [Hunyuan3D 2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) and
  [Hunyuan3D 2mv](https://huggingface.co/tencent/Hunyuan3D-2mv) (Tencent), and
  [TRELLIS](https://huggingface.co/microsoft/TRELLIS-image-large) (Microsoft).
- **Drag and drop:** drop one or more image files onto the *Image to 3D* sidebar tab; each becomes a model.
- **Multi-view:** give several photos of one object (front, left, back, right) for a more accurate model.
- **AMS colours:** split the model into 1–16 filament colours and export a painted 3MF for multi-colour printing.
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
| Hunyuan3D 2mv (multi-view shape) | [tencent/Hunyuan3D-2mv](https://huggingface.co/tencent/Hunyuan3D-2mv) | [Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | [Tencent Hunyuan 3D 2.0 Community License](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE): not licensed in the EU, UK or South Korea | ~10 GB |
| TRELLIS image-large | [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) | [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) | [MIT](https://github.com/microsoft/TRELLIS/blob/main/LICENSE) | ~8 GB |

Which one to use:
- **Hunyuan 2.1:** the sharpest detail from a single image, about 80 s per model.
- **Hunyuan 2mv:** takes labelled views (front, left, back, right), about 60 s. It's the most accurate when
  you have photos from several sides.
- **TRELLIS:** very fast (about 10 s), and takes one image or several views from any angles.

Downloaded automatically as dependencies:
- [DINOv2](https://github.com/facebookresearch/dinov2), used by TRELLIS.
- [U²-Net](https://github.com/xuebinqin/U-2-Net), for background removal through [rembg](https://github.com/danielgatis/rembg).

Optional, for clean AMS colours: **Hunyuan3D-Paint** (the weights are
[tencent/Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2): `hunyuan3d-delight-v2-0` and
`hunyuan3d-paint-v2-0-turbo`, about 12 GB of GPU memory). It removes lighting and shadows from your photo, then
paints the model from several angles.

TRELLIS.2 is not supported: it currently needs Linux and 24 GB or more of GPU memory.

## Requirements

- Windows 10/11 (Linux should work with small path changes, but it's untested)
- NVIDIA GPU, 12 GB+ memory recommended (tested on an RTX 5070 Ti 16 GB)
- [Blender](https://www.blender.org/) 4.2 or newer (tested on 5.1)
- [git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/)
- About 45 GB of disk for all three models: one Python environment per model, plus the weights

## Setup

1. Install any of the models. Each gets its own folder and Python environment:
   ```powershell
   powershell -ExecutionPolicy Bypass -File setup\setup_hunyuan.ps1      # -> %USERPROFILE%\Hunyuan3D
   powershell -ExecutionPolicy Bypass -File setup\setup_hunyuan_mv.ps1   # -> %USERPROFILE%\Hunyuan3D-2
   powershell -ExecutionPolicy Bypass -File setup\setup_trellis.ps1      # -> %USERPROFILE%\TRELLIS
   ```
   Pass `-Dir D:\somewhere` to install elsewhere.
   For clean AMS colours, also install [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive)
   (`winget install --id Nvidia.CUDA --version 12.8`) and Visual Studio 2022 Build Tools (C++), then run
   `setup\setup_paint.ps1`, which compiles Hunyuan3D-Paint's GPU rasterizer.
2. Install the Blender add-on. Pick one option:
   - **Link it**, so it updates with `git pull`:
     ```powershell
     New-Item -ItemType Junction -Path "$env:APPDATA\Blender Foundation\Blender\<version>\scripts\addons\local_3dgen" -Target "$PWD\addon\local_3dgen"
     ```
   - **Or zip `addon/local_3dgen`** and install the zip with *Edit > Preferences > Add-ons > Install from Disk*.
3. Enable **Local 3D Gen** in Preferences. In its settings, check the repo folder and the model folders.

## Using it

1. In the 3D Viewport press **N** and open the **Image to 3D** tab.
2. Pick a model at the top: **Hunyuan 2.1**, **Hunyuan 2mv** or **TRELLIS**.
3. **Drop image files onto the panel** (or use **+**). With *Generate on drop* on, they start straight away.
   Otherwise select one and press **Generate**, or press **All**.
   For **multi-view**, switch the mode to *Multi-view (one object)*. Drop photos of the same object taken from
   different sides, and check each one's view (front, left, back or right). They're filled in automatically in
   the order you drop them. Then press **Generate from N views**. This needs Hunyuan 2mv or TRELLIS.
4. Models appear at the 3D cursor in an **AI Meshes** collection, scaled so the longest side matches
   *Longest side (mm)*. Variants are laid out side by side.

Tips:
- Use a clean, single-object image with a plain background. The background is removed automatically.
- Try a few seeds (*Variants* 3–4) and keep the best one.
- Hunyuan: *Detail* 512 gives finer surfaces, but the final "building mesh" step runs on the CPU and can take
  several minutes with a lot of RAM. 384 with 50 steps is the sweet spot.
- Multi-view photos: same object, lighting and distance, roughly 90° apart, plain background.
- Switching models, or pressing ✕, unloads the current model to free GPU memory.
- Each result is also saved as a GLB in `outputs/<image name>/`, next to the cut-out input image.

## Multi-colour printing (AMS)

Similar to Meshy's multi-colour print, but local. Select a generated model and open **Image to 3D > AMS colours**:

1. Optional but recommended: **Clean colours (Hunyuan3D-Paint)**. It repaints the model with the lighting and
   shadows removed, in about 90 s, and replaces it with a painted copy. Without this, shadows in the photo or
   model turn into dark patches.
2. Set **Colours** (the number of filaments), then click the eyedropper. The palette is taken from the source
   photo(s), so the colours are clean, not the shaded ones on the mesh. You can edit any swatch.
3. **Regions from**: *Auto* uses the model's own colours when it generated them (TRELLIS; they line up exactly
   with the geometry), and otherwise flattens the photos into regions and projects them. When projecting, the
   camera angle of the photo is matched automatically.
4. **Mirror**: for symmetric pieces like pots and vases, copy the front's colours onto the back, since the back
   is guessed.
5. **Assign colours**. Specks smaller than *Smallest patch* are merged, and borders are smoothed over
   *Border smoothing* mm and subdivided so they can run between triangle edges. The defaults (1 mm² and
   1.2 mm) suit a 0.4 mm nozzle: finer detail than that doesn't print as distinct colour. Touch up
   by hand if needed: Edit Mode, select faces, then *Material > Assign*.
6. **Fix-ups**:
   - *Fix cutout rims*: the walls of holes take the colour of the part they're cut into (Rim), not the colour
     showing through them (Through).
   - *Level border*: snaps the border of the bottom colour region (e.g. a base band) to a horizontal line.
     Height 0 detects it automatically.
7. **Export 3MF**: a Bambu Studio project with per-triangle colour painting, using the same layout as Meshy's
   multi-colour export. The chosen printer and one PLA filament per palette colour are preset. Load it with
   *File > Import > Import 3D Models*.

## Command line

The same engine works from a terminal:
```
python scripts/img2mesh.py photo.png --model trellis --variants 3
python scripts/img2mesh.py --model hunyuan_mv --views front=f.png left=l.png back=b.png
```

## How it works

```
Blender add-on (addon/local_3dgen)
   └─ starts engine/server.py with the chosen model's own .venv Python
        ├─ backends/hunyuan.py     → Hunyuan3D-2.1 (hy3dshape)
        ├─ backends/hunyuan_mv.py  → Hunyuan3D-2mv (hy3dgen)
        ├─ backends/hunyuan_paint.py → Hunyuan3D-Paint: repaints an existing mesh (the "paint" request)
        └─ backends/trellis.py     → TRELLIS, single or multi-image (+ shims for xformers / kaolin)
   JSON lines over stdin/stdout: generate requests in; progress, done and error events out
```
The engine watches Blender's process ID and exits when Blender is gone.

## Troubleshooting

- **The engine stops right away:** read `engine_<model>.log` in the repo folder.
- **"Out of GPU memory":** close other GPU apps, lower *Detail*, or switch models.
- **The first run seems stuck at "Loading model":** it's downloading several GB of weights; later loads are faster.

## Credits

Built on [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) and
[Hunyuan3D 2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) by Tencent, and
[TRELLIS](https://github.com/microsoft/TRELLIS) by Microsoft. Generated models are subject to each model's
licence.

## Licence

The code in this repo is [MIT](LICENSE). The models are not covered by it: they come under their own
licences (see [The models](#the-models)), and so do the meshes you generate with them.
