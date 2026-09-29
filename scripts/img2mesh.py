"""Command-line image -> 3D mesh through the same engine the Blender panel uses.

Run with any Python 3.9+ (it starts the engine with the model's own environment):
    python scripts/img2mesh.py photo.png [more.png ...] --model trellis --model-dir C:/Users/me/TRELLIS
Options: --seed N, --variants N, --steps N, --guidance X, --res 256|384|512 (Hunyuan only), --out DIR
"""
import argparse, json, os, random, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIRS = {'hunyuan': os.path.expanduser('~/Hunyuan3D'), 'trellis': os.path.expanduser('~/TRELLIS')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('images', nargs='+')
    ap.add_argument('--model', choices=['hunyuan', 'trellis'], default='hunyuan')
    ap.add_argument('--model-dir')
    ap.add_argument('--out', default=os.path.join(REPO, 'outputs'))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--variants', type=int, default=1)
    ap.add_argument('--steps', type=int)
    ap.add_argument('--guidance', type=float)
    ap.add_argument('--res', type=int, choices=[256, 384, 512])
    ap.add_argument('--keep-background', action='store_true')
    a = ap.parse_args()

    model_dir = os.path.abspath(a.model_dir or DEFAULT_DIRS[a.model])
    py = os.path.join(model_dir, '.venv', 'Scripts', 'python.exe')
    if not os.path.exists(py):
        py = os.path.join(model_dir, '.venv', 'bin', 'python')
    proc = subprocess.Popen([py, os.path.join(REPO, 'engine', 'server.py'), '--backend', a.model,
                             '--model-dir', model_dir, '--parent-pid', str(os.getpid())],
                            cwd=model_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                            encoding='utf-8', env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    jobs = 0
    for img in a.images:
        name = os.path.splitext(os.path.basename(img))[0]
        base = a.seed if a.seed is not None else random.randint(0, 2**31 - 1)
        for i in range(a.variants):
            req = dict(cmd='generate', id=f'{name}-{base + i}', image=os.path.abspath(img),
                       out=os.path.join(a.out, name), name=name, seed=base + i, remove_bg=not a.keep_background)
            for k in ('steps', 'guidance', 'res'):
                if getattr(a, k) is not None:
                    req[k] = getattr(a, k)
            proc.stdin.write(json.dumps(req) + '\n'); jobs += 1
    proc.stdin.write('{"cmd": "quit"}\n'); proc.stdin.flush()

    last = None
    for line in proc.stdout:
        if not line.startswith('@@'):
            continue
        ev = json.loads(line[2:])
        if ev['ev'] == 'status':
            print(ev['msg'])
        elif ev['ev'] == 'progress':
            msg = f"  {ev['id']}: {ev['stage']} {int(ev['frac'] * 100)}%"
            if msg != last:
                print(msg, end='\r', flush=True); last = msg
        elif ev['ev'] == 'done':
            print(f"\n  wrote {ev['glb']} ({ev['verts']:,} verts, {ev['seconds']:.0f}s)")
        elif ev['ev'] == 'error':
            print(f"\n  FAILED {ev['id']}: {ev['msg']}", file=sys.stderr)
    sys.exit(proc.wait())


if __name__ == '__main__':
    main()
