"""Tencent Hunyuan3D-2.1 shape model (https://huggingface.co/tencent/Hunyuan3D-2.1).

Job options: steps (50), guidance (5.0), res = octree resolution (256/384/512).
"""
import os
import sys

MODEL_ID = 'tencent/Hunyuan3D-2.1'


class Backend:
    label = 'Hunyuan3D 2.1'

    def __init__(self, model_dir):
        sys.path.insert(0, os.path.join(model_dir, 'hy3dshape'))
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dshape.postprocessors import DegenerateFaceRemover, FloaterRemover
        self.pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_ID)
        self.cleanup = lambda m: DegenerateFaceRemover()(FloaterRemover()(m))

    def generate(self, img, job, progress):
        import torch
        steps = int(job.get('steps', 50))

        def cb(step, _t, _o):
            progress('shaping', (step + 1) / steps)

        mesh = self.pipe(image=img, num_inference_steps=steps,
                         octree_resolution=int(job.get('res', 384)),
                         guidance_scale=float(job.get('guidance', 5.0)),
                         generator=torch.manual_seed(int(job['seed'])),
                         callback=cb, callback_steps=1, enable_pbar=False)[0]
        progress('cleaning up', 1.0)
        return self.cleanup(mesh)  # already Y-up, like glTF expects
