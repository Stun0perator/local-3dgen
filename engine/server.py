"""Long-running image-to-3D engine used by the Blender panel (and scripts/img2mesh.py).

Run with the Python of the chosen model's own environment:
    <model_dir>/.venv/Scripts/python.exe engine/server.py --backend hunyuan --model-dir <model_dir>

Protocol: one JSON request per line on stdin, one "@@" + JSON event per line on stdout.
  in : {"cmd": "generate", "id": ..., "image": path, "out": dir, "name": str, "seed": int,
        "remove_bg": bool, ...backend options (see backends/*.py)}
       {"cmd": "quit"}
  out: {"ev": "status", "msg": str, "ready": bool?}
       {"ev": "progress", "id": ..., "stage": str, "frac": 0..1}
       {"ev": "done", "id": ..., "glb": path, "verts": n, "faces": n, "watertight": bool, "seconds": s}
       {"ev": "error", "id": ..., "msg": str}
Library output goes to stderr. The engine exits when stdin closes or when --parent-pid dies,
so the model never lingers in GPU memory after Blender closes or crashes.
"""
import argparse, json, os, sys, threading, time, traceback

PROTO = sys.stdout
sys.stdout = sys.stderr  # keep library chatter off the protocol channel
_lock = threading.Lock()


def emit(**kw):
    with _lock:
        PROTO.write('@@' + json.dumps(kw) + '\n'); PROTO.flush()


def watch_parent(pid):
    """Exit as soon as the parent process (Blender) is gone — even if it crashed."""
    def run():
        if os.name == 'nt':
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            if h:
                k32.WaitForSingleObject(h, 0xFFFFFFFF)
                os._exit(0)
        while True:
            try:
                os.kill(pid, 0)
            except OSError:
                os._exit(0)
            time.sleep(2)
    threading.Thread(target=run, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True, choices=['hunyuan', 'trellis'])
    ap.add_argument('--model-dir', required=True, help='clone of the model repo (with its .venv)')
    ap.add_argument('--parent-pid', type=int, default=0)
    a = ap.parse_args()
    if a.parent_pid:
        watch_parent(a.parent_pid)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    emit(ev='status', msg='Loading libraries…')
    if a.backend == 'hunyuan':
        from backends.hunyuan import Backend
    else:
        from backends.trellis import Backend
    import torch
    from PIL import Image

    emit(ev='status', msg='Loading model onto GPU…')
    t0 = time.time()
    backend = Backend(a.model_dir)
    emit(ev='status', msg=f'Ready: {backend.label} (loaded in {time.time() - t0:.0f}s)', ready=True)

    rembg_session = None

    def prepare(job):
        nonlocal rembg_session
        img = Image.open(job['image'])
        img.load()
        has_alpha = img.mode in ('RGBA', 'LA') and img.getchannel('A').getextrema()[0] < 255
        if has_alpha or not job.get('remove_bg', True):
            return img.convert('RGBA')
        emit(ev='progress', id=job.get('id'), stage='removing background', frac=0.0)
        from rembg import new_session, remove
        rembg_session = rembg_session or new_session('u2net')
        return remove(img.convert('RGB'), session=rembg_session)

    def generate(job):
        jid = job.get('id')
        t = time.time()
        img = prepare(job)
        os.makedirs(job['out'], exist_ok=True)
        base = os.path.join(job['out'], f"{job.get('name', 'model')}_{a.backend}_s{job['seed']}")
        img.save(base + '_input.png')
        mesh = backend.generate(img, job, lambda stage, frac: emit(ev='progress', id=jid, stage=stage, frac=frac))
        emit(ev='progress', id=jid, stage='saving', frac=1.0)
        mesh.export(base + '.glb')
        emit(ev='done', id=jid, glb=base + '.glb', verts=len(mesh.vertices), faces=len(mesh.faces),
             watertight=bool(mesh.is_watertight), seconds=round(time.time() - t, 1))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception:
            emit(ev='error', id=None, msg='bad request: ' + line[:200]); continue
        if job.get('cmd') == 'quit':
            break
        if job.get('cmd') == 'generate':
            try:
                generate(job)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                emit(ev='error', id=job.get('id'),
                     msg='Out of GPU memory: lower the detail setting or close other GPU apps')
            except Exception as e:
                traceback.print_exc()
                emit(ev='error', id=job.get('id'), msg=f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
