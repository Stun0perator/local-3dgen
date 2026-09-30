"""Tencent Hunyuan3D-Paint v2.0 (turbo or full): paints a clean, de-lit texture onto an existing mesh from its photo(s).
Weights: https://huggingface.co/tencent/Hunyuan3D-2 (hunyuan3d-delight-v2-0 + hunyuan3d-paint-v2-0-turbo).
Code: https://github.com/Tencent-Hunyuan/Hunyuan3D-2 (hy3dgen.texgen; needs the custom_rasterizer CUDA
extension, see setup/build_paint_ext.cmd).

The first step removes lighting and shadows from the photo, which is what makes colours usable for AMS
printing. Painting works on a reduced copy of the mesh (UV unwrapping a dense mesh is slow); the texture's
colours are then transferred back onto every vertex of the full mesh as vertex colours.
Job options: paint_faces (60000), paint_quality ('turbo' default | 'full': the non-turbo model, slower,
more detailed), texture_size (2048; 4096 for more colour detail, more GPU memory).
"""
import sys

import numpy as np

MODEL_ID = 'tencent/Hunyuan3D-2'
SUBFOLDERS = {'turbo': 'hunyuan3d-paint-v2-0-turbo', 'full': 'hunyuan3d-paint-v2-0'}


class Backend:
    label = 'Hunyuan3D-Paint'
    multiview = True

    def __init__(self, model_dir):
        sys.path.insert(0, model_dir)
        from hy3dgen.shapegen.postprocessors import FaceReducer
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        self.reduce = FaceReducer()
        self._cls = Hunyuan3DPaintPipeline
        self.quality = None
        self.pipe = None
        self._load('turbo')

    def _load(self, quality):
        """One paint model in GPU memory at a time; switching quality reloads."""
        if quality == self.quality:
            return
        import gc
        import torch
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        self.pipe = self._cls.from_pretrained(MODEL_ID, subfolder=SUBFOLDERS[quality])
        self.quality = quality

    def generate(self, views, job, progress):
        raise ValueError('Hunyuan3D-Paint colours an existing mesh; send a "paint" request')

    def paint(self, mesh, views, job, progress):
        """-> (textured low-poly mesh, sRGB colours for each vertex of `mesh`)."""
        import trimesh
        import colorize
        quality = job.get('paint_quality', 'turbo')
        if quality != self.quality:
            progress('loading paint model', 0.0)
            self._load(quality)
        size = int(job.get('texture_size', 2048))
        self.pipe.config.texture_size = size
        self.pipe.render.set_default_texture_resolution(size)
        progress('preparing mesh', 0.0)
        low = self.reduce(mesh.copy(), max_facenum=int(job.get('paint_faces', 60000)))
        order = [v for v in ('front', 'left', 'back', 'right') if v in views]  # front first: the reference
        progress('painting', 0.1)
        textured = self.pipe(low, image=[views[v] for v in order])
        progress('transferring colours', 0.9)
        mat = textured.visual.material
        img = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
        tex = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
        h, w = tex.shape[:2]
        pts, fid = trimesh.sample.sample_surface(textured, 900000, seed=0)
        bary = trimesh.triangles.points_to_barycentric(textured.triangles[fid], pts)
        uv = (textured.visual.uv[textured.faces[fid]] * bary[..., None]).sum(1)
        x = np.clip(np.round(uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
        y = np.clip(np.round((1 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
        colors = colorize.transfer(np.asarray(mesh.vertices), pts, tex[y, x], k=6)
        return textured, colors
