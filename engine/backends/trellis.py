"""Microsoft TRELLIS image-to-3D (https://huggingface.co/microsoft/TRELLIS-image-large).

Job options: steps (12, used for both stages), guidance (7.5 structure; detail stage uses 3.0).
Several views: TRELLIS's tuning-free multi-image mode (the view names are ignored, order doesn't matter).
Runs without xformers/flash-attn: attention goes through torch SDPA (see ../shims),
which is what makes it work on RTX 50-series cards under Windows.
"""
import os
import sys

MODEL_ID = 'microsoft/TRELLIS-image-large'
HERE = os.path.dirname(os.path.abspath(__file__))


class Backend:
    label = 'TRELLIS (image-large)'
    multiview = True

    def __init__(self, model_dir):
        os.environ.setdefault('ATTN_BACKEND', 'sdpa')
        os.environ.setdefault('SPARSE_ATTN_BACKEND', 'xformers')  # served by the SDPA shim
        os.environ.setdefault('SPCONV_ALGO', 'native')             # skip spconv's benchmark pass
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'shims'))
        sys.path.insert(0, model_dir)
        import torch
        # DINOv2 comes via torch.hub; keep it in the TRELLIS folder so a stale shared cache can't break it
        torch.hub.set_dir(os.path.join(model_dir, '.torch_hub'))
        from trellis.pipelines import TrellisImageTo3DPipeline
        import trellis.pipelines.samplers.flow_euler as flow_euler
        self.flow_euler = flow_euler
        self.pipe = TrellisImageTo3DPipeline.from_pretrained(MODEL_ID)
        self.pipe.cuda()

    def _track(self, progress, stages):
        """Route the sampler's tqdm loop into progress events (one stage per sampler call)."""
        calls = iter(stages)

        def tqdm(it, desc=None, disable=False, **kw):
            items = list(it)
            stage = next(calls, 'sampling')
            for i, x in enumerate(items):
                progress(stage, i / len(items))
                yield x
            progress(stage, 1.0)

        self.flow_euler.tqdm = tqdm

    def generate(self, views, job, progress):
        import numpy as np
        import trimesh
        steps = int(job.get('steps', 12))
        self._track(progress, ['shaping (structure)', 'shaping (detail)'])
        # the model's own colours, only when asked for (default is projecting the photos, done by the server)
        want_colors = job.get('colors', True) and job.get('color_source', 'photo') == 'model'
        params = dict(seed=int(job['seed']), formats=['mesh', 'gaussian'] if want_colors else ['mesh'],
                      sparse_structure_sampler_params={'steps': steps, 'cfg_strength': float(job.get('guidance', 7.5))},
                      slat_sampler_params={'steps': steps, 'cfg_strength': 3.0})
        images = list(views.values())
        if len(images) == 1:
            out = self.pipe.run(images[0], **params)
        else:
            out = self.pipe.run_multi_image(images, mode='stochastic', **params)
        m = out['mesh'][0]
        v = m.vertices.detach().float().cpu().numpy()
        f = m.faces.detach().cpu().numpy()
        to_gltf = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)  # Z-up -> glTF Y-up
        v = v @ to_gltf
        progress('cleaning up', 1.0)
        mesh = trimesh.Trimesh(v, f, process=True)
        mesh.remove_unreferenced_vertices()
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:  # drop floating specks (<1% of faces), keep real separate pieces
            keep = [p for p in parts if len(p.faces) >= 0.01 * len(mesh.faces)]
            mesh = trimesh.util.concatenate(keep)
        trimesh.repair.fix_normals(mesh)
        if want_colors:
            # TRELLIS colours live in its gaussians (same space as the mesh): copy them to the vertices
            import colorize
            progress('colouring', 1.0)
            g = out['gaussian'][0]
            pts = g.get_xyz.detach().float().cpu().numpy() @ to_gltf
            rgb = np.clip(g._features_dc.detach().float().cpu().numpy()[:, 0, :3] * 0.28209479177387814 + 0.5, 0, 1)
            opacity = g.get_opacity.detach().float().cpu().numpy().reshape(-1)
            mesh.metadata['l3d_colors'] = colorize.transfer(np.asarray(mesh.vertices), pts, rgb, weights=opacity)
        return mesh
