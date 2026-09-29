"""Tencent Hunyuan3D-2mv multi-view shape model (https://huggingface.co/tencent/Hunyuan3D-2mv).

Takes up to four labelled views (front, left, back, right); any subset works, a lone front view too.
Code: https://github.com/Tencent-Hunyuan/Hunyuan3D-2 (hy3dgen package).
Job options: steps (50), guidance (5.0), res = octree resolution (256/384/512).
"""
import sys

MODEL_ID = 'tencent/Hunyuan3D-2mv'
SUBFOLDER = 'hunyuan3d-dit-v2-mv'
VIEWS = ('front', 'left', 'back', 'right')


class Backend:
    label = 'Hunyuan3D 2mv'
    multiview = True

    def __init__(self, model_dir):
        sys.path.insert(0, model_dir)
        from hy3dgen.shapegen import (DegenerateFaceRemover, FloaterRemover,
                                      Hunyuan3DDiTFlowMatchingPipeline)
        self.pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_ID, subfolder=SUBFOLDER,
                                                                     variant='fp16')
        self.cleanup = lambda m: DegenerateFaceRemover()(FloaterRemover()(m))

    def generate(self, views, job, progress):
        import torch
        bad = set(views) - set(VIEWS)
        if bad:
            raise ValueError(f'unknown view(s) {sorted(bad)}; use {", ".join(VIEWS)}')
        steps = int(job.get('steps', 50))

        def cb(step, _t, _o):
            done = step + 1 >= steps
            progress('building mesh' if done else 'shaping', 0.0 if done else (step + 1) / steps)

        mesh = self.pipe(image=dict(views), num_inference_steps=steps,
                         octree_resolution=int(job.get('res', 384)),
                         guidance_scale=float(job.get('guidance', 5.0)),
                         num_chunks=20000, generator=torch.manual_seed(int(job['seed'])),
                         callback=cb, callback_steps=1, enable_pbar=False, output_type='trimesh')[0]
        progress('cleaning up', 1.0)
        return self.cleanup(mesh)
