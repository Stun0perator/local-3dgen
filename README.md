# Local 3D Gen

Drag images into Blender and get 3D models back, generated **locally on your own GPU** with open
image-to-3D models. No API keys, no cloud, no per-model fees.

- **Three models, one switch:** [Hunyuan3D 2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) and
  [Hunyuan3D 2mv](https://huggingface.co/tencent/Hunyuan3D-2mv) (Tencent), and
  [TRELLIS](https://huggingface.co/microsoft/TRELLIS-image-large) (Microsoft).
- **Drag and drop:** drop one or more image files onto the *Image to 3D* sidebar tab; each becomes a model.
- **Multi-view:** give several photos of one object (front, left, back, right) for a more accurate model.
- **AMS colours:** click regions of the model to colour them, then export a painted 3MF for multi-colour
  printing.
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

The AI makes the shape; you decide the colours, by clicking. Select a generated model and open
**Image to 3D > Colour for AMS**:

1. Set **Filaments** and pick the swatch colours. The eyedropper takes them from the source photo.
2. Pick your base colour's swatch and press **Fill all**.
3. **Split into regions**: the model is cut into regions along its shape (ribs, rims, lips, feet and knobs
   become their own regions; the smoother surface is cut where parts meet). **Seams** sets how readily it
   cuts.
4. **Click to colour**, then in the viewport:
   - **Click** a region to fill it with the active swatch.
   - **Shift-click** fills every region of the same size and shape (all the feet, for example).
   - **Alt-drag** paints with a brush, for colour changes the shape doesn't mark. `[` and `]` resize it.
   - **1–9** switch swatch, **Ctrl+Z** undoes, and **right-click, Enter or Esc** finishes.
   - Orbit, pan and zoom work as usual.
5. **Band**: everything between two heights gets the active swatch, with a straight, subdivided edge
   (bases, stripes).
6. **Export**: a Bambu Studio 3MF with one PLA filament per swatch and the printer preset set. Load it with
   *File > Import > Import 3D Models*.

Region borders sit on the model's triangles, which are about 0.2–0.4 mm across at 1M triangles. That's as
fine as a 0.4 mm nozzle can place a colour change.

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
