bl_info = {
    "name": "Local 3D Gen (Hunyuan3D / TRELLIS)",
    "author": "Stun0perator",
    "version": (1, 3, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > Image to 3D",
    "description": "Drop images in, get 3D models out (Hunyuan3D / TRELLIS, locally), then split into AMS colours",
    "doc_url": "https://huggingface.co/tencent/Hunyuan3D-2.1",
    "category": "3D View",
}

import atexit, json, os, queue, random, subprocess, threading, time

import bpy
import bpy.utils.previews
from . import colour
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

MODELS = {
    'hunyuan': dict(
        label="Hunyuan3D 2.1",
        weights="https://huggingface.co/tencent/Hunyuan3D-2.1",
        code="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1",
        license="Tencent Hunyuan 3D 2.1 Community License (not valid in the EU, UK, South Korea)",
        about="Sharper detail, closer to the image. One image only. ~10 GB GPU memory, ~80 s per model",
        multiview=False, pref="hunyuan_dir"),
    'hunyuan_mv': dict(
        label="Hunyuan3D 2mv",
        weights="https://huggingface.co/tencent/Hunyuan3D-2mv",
        code="https://github.com/Tencent-Hunyuan/Hunyuan3D-2",
        license="Tencent Hunyuan 3D 2.0 Community License (not valid in the EU, UK, South Korea)",
        about="Built for several labelled views (front/left/back/right). ~10 GB GPU memory, ~60 s per model",
        multiview=True, pref="hunyuan_mv_dir"),
    'trellis': dict(
        label="TRELLIS",
        weights="https://huggingface.co/microsoft/TRELLIS-image-large",
        code="https://github.com/microsoft/TRELLIS",
        license="MIT",
        about="Fast (~10 s). Accepts several views of any angle. ~8 GB GPU memory",
        multiview=True, pref="trellis_dir"),
}
VIEWS = [('front', "Front", ""), ('left', "Left", ""), ('back', "Back", ""), ('right', "Right", "")]

_previews = None


def _addon_dir():
    return os.path.dirname(os.path.realpath(__file__))


def _default_repo():
    d = os.path.dirname(os.path.dirname(_addon_dir()))  # <repo>/addon/local_3dgen -> <repo>
    return d if os.path.isfile(os.path.join(d, "engine", "server.py")) else ""


def _prefs():
    return bpy.context.preferences.addons[__name__].preferences


def _p(path):
    return os.path.normpath(bpy.path.abspath(path)) if path else ""


# ---------------------------------------------------------------- engine (background process)

class Engine:
    proc = None
    backend = None
    events = queue.Queue()
    status = "Stopped"
    ready = False
    pending = []          # jobs waiting to be sent
    current = None        # job being generated
    stage = ""
    frac = 0.0
    batch_total = 0
    batch_done = 0
    last_error = ""
    results = []          # (object name, info text)
    started_at = 0.0
    stage_at = 0.0        # when the current stage started
    last_event = 0.0      # last engine event, to spot a job that stopped reporting
    log = None

    @classmethod
    def alive(cls):
        return cls.proc is not None and cls.proc.poll() is None

    @classmethod
    def start(cls, backend):
        if cls.alive() and cls.backend == backend:
            return
        if cls.alive():
            cls.stop(keep_queue=True)
        pr = _prefs()
        repo = _p(pr.repo_dir)
        model_dir = _p(getattr(pr, MODELS[backend]["pref"]))
        py = os.path.join(model_dir, ".venv", "Scripts", "python.exe")
        if not os.path.exists(py):
            py = os.path.join(model_dir, ".venv", "bin", "python")
        server = os.path.join(repo, "engine", "server.py")
        for path, what in ((server, "engine (set the repo folder in add-on preferences)"),
                           (py, f"{MODELS[backend]['label']} environment (see README setup)")):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {what}: {path}")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        cls.log = open(os.path.join(repo, f"engine_{backend}.log"), "w", encoding="utf-8")
        cls.proc = subprocess.Popen(
            [py, server, "--backend", backend, "--model-dir", model_dir, "--parent-pid", str(os.getpid())],
            cwd=model_dir, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=cls.log,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        cls.backend = backend
        cls.ready = False
        cls.status = f"Starting {MODELS[backend]['label']}…"
        cls.started_at = time.time()
        threading.Thread(target=cls._read, args=(cls.proc,), daemon=True).start()
        _ensure_timer()

    @classmethod
    def _read(cls, proc):
        for line in proc.stdout:
            if line.startswith("@@"):
                try:
                    ev = json.loads(line[2:])
                    ev["_proc"] = proc
                    cls.events.put(ev)
                except Exception:
                    pass
        cls.events.put({"ev": "exit", "_proc": proc})

    @classmethod
    def send(cls, obj):
        cls.proc.stdin.write(json.dumps(obj) + "\n")
        cls.proc.stdin.flush()

    @classmethod
    def stop(cls, hard=False, keep_queue=False):
        proc, cls.proc = cls.proc, None
        if proc is not None and proc.poll() is None:
            try:
                if hard:
                    raise RuntimeError
                proc.stdin.write('{"cmd": "quit"}\n'); proc.stdin.flush()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        cls.ready = False
        cls.backend = None
        if cls.current is not None and keep_queue:
            cls.pending.insert(0, cls.current)
        cls.current = None
        if not keep_queue:
            cls.pending.clear()
        cls.status = "Stopped (GPU memory freed)"
        cls.stage, cls.frac = "", 0.0
        if cls.log:
            cls.log.close(); cls.log = None


# Normal quit: stop the engine. Crash / force-quit: the engine watches Blender's PID and exits itself.
atexit.register(lambda: Engine.stop(hard=True))


def _redraw():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _ensure_timer():
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.2)


def _set_item_status(job, text):
    scene = bpy.data.scenes.get(job["ctx"]["scene"])
    paths = set(job["req"].get("views", {}).values()) or {job["req"].get("image")}
    if scene:
        for it in scene.l3d.images:
            if _p(it.path) in paths:
                it.status = text


def _tick():
    """Main-thread timer: drain engine events, dispatch jobs, import results."""
    changed = False
    while True:
        try:
            ev = Engine.events.get_nowait()
        except queue.Empty:
            break
        if ev.get("_proc") is not Engine.proc:
            continue  # stale event from an engine we already stopped
        changed = True
        Engine.last_event = time.time()
        kind = ev.get("ev")
        if kind == "status":
            Engine.status = ev["msg"]
            Engine.ready = bool(ev.get("ready")) or Engine.ready
        elif kind == "progress":
            if ev["stage"] != Engine.stage:
                Engine.stage_at = time.time()
            Engine.stage, Engine.frac = ev["stage"], ev["frac"]
        elif kind == "done":
            job, Engine.current = Engine.current, None
            Engine.batch_done += 1
            try:
                name = _import_result(ev["glb"], job, ev.get("inputs"), ev.get("color_source"))
                info = f'{ev["verts"]:,} verts · {ev["seconds"]:.0f}s' + ("" if ev.get("watertight") else " · open mesh")
                Engine.results.insert(0, (name, info))
                del Engine.results[8:]
                _set_item_status(job, "done")
            except Exception as e:
                Engine.last_error = f"Import failed: {e}"
        elif kind == "error":
            job, Engine.current = Engine.current, None
            Engine.batch_done += 1
            Engine.last_error = ev.get("msg", "unknown error")
            if job:
                _set_item_status(job, "failed")
        elif kind == "exit":
            Engine.proc = None
            Engine.ready = False
            Engine.backend = None
            Engine.status = "Engine stopped unexpectedly (see engine log in the repo folder)"
            Engine.last_error = Engine.status
            Engine.current = None
            Engine.pending.clear()

    if Engine.pending and Engine.current is None:
        nxt = Engine.pending[0]
        if Engine.backend != nxt["backend"] or not Engine.alive():
            try:
                Engine.start(nxt["backend"])  # switch models between jobs
            except Exception as e:
                Engine.last_error = str(e)
                Engine.pending.clear()
            changed = True
        elif Engine.ready:
            Engine.current = Engine.pending.pop(0)
            Engine.stage, Engine.frac = "starting", 0.0
            Engine.stage_at = Engine.last_event = time.time()
            _set_item_status(Engine.current, "generating")
            Engine.send(Engine.current["req"])
            changed = True
    if Engine.alive() and (not Engine.ready or Engine.current is not None):
        changed = True  # keep the timers moving

    if changed:
        _redraw()
    return 0.25 if (Engine.alive() or Engine.pending or not Engine.events.empty()) else None


def _import_result(glb, job, inputs=None, color_source=None):
    from mathutils import Matrix, Vector
    ctx = job["ctx"]
    scene = bpy.data.scenes.get(ctx["scene"]) or bpy.context.scene
    win = bpy.context.window_manager.windows[0]
    before = set(bpy.data.objects)
    with bpy.context.temp_override(window=win, scene=scene):
        bpy.ops.import_scene.gltf(filepath=glb)
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == 'MESH']
    for o in new:
        if o.type != 'MESH':
            bpy.data.objects.remove(o, do_unlink=True)
    if not meshes:
        raise RuntimeError("no mesh in result")
    ob = meshes[0]
    mw = ob.matrix_world.copy()
    ob.parent = None
    ob.data.transform(mw)
    ob.matrix_world.identity()
    co = [v.co for v in ob.data.vertices]
    lo = [min(c[i] for c in co) for i in range(3)]
    hi = [max(c[i] for c in co) for i in range(3)]
    k = ctx["size_bu"] / (max(h - l for h, l in zip(hi, lo)) or 1.0)
    centre = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]))
    ob.data.transform(Matrix.Scale(k, 4) @ Matrix.Translation(-centre))
    ob.location = Vector(ctx["cursor"]) + Vector((job["slot"] * ctx["size_bu"] * 1.25, 0, 0))
    for poly in ob.data.polygons:
        poly.use_smooth = ctx["smooth"]
    ob.name = ob.data.name = f'{job["name"]}_{job["backend"]}_s{job["seed"]}'
    ob["l3d_model"] = MODELS[job["backend"]]["label"]
    ob["l3d_seed"] = job["seed"]
    ob["l3d_images"] = json.dumps(job["req"].get("views") or {"front": job["req"]["image"]})
    ob["l3d_glb"] = glb
    if inputs:
        ob["l3d_inputs"] = json.dumps(inputs)  # cut-out photos, used for AMS colour regions
    if color_source:
        ob["l3d_color_source"] = color_source  # 'model': vertex colours line up with the geometry
    coll = bpy.data.collections.get("AI Meshes") or bpy.data.collections.new("AI Meshes")
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    coll.objects.link(ob)
    return ob.name


def _model_opts(p):
    if p.model in ('hunyuan', 'hunyuan_mv'):
        return dict(steps=p.hy_steps, guidance=p.hy_guidance, res=int(p.hy_res))
    return dict(steps=p.tr_steps, guidance=p.tr_guidance, color_source=p.tr_colors)


def queue_multiview(context, views):
    """Queue one job (x variants) that uses several views of the same object: {view: path}."""
    p = context.scene.l3d
    pr = _prefs()
    unit = context.scene.unit_settings.scale_length or 1.0
    out_root = _p(pr.output_dir) or os.path.join(_p(pr.repo_dir), "outputs")
    first = views.get('front') or next(iter(views.values()))
    name = bpy.path.clean_name(os.path.splitext(os.path.basename(first))[0]) + "_mv"
    if Engine.current is None and not Engine.pending:
        Engine.batch_total = Engine.batch_done = 0
    Engine.last_error = ""
    base_seed = random.randint(0, 2**31 - 1) if p.random_seed else p.seed
    for i in range(p.variants):
        seed = base_seed + i
        ctx = dict(scene=context.scene.name, cursor=tuple(context.scene.cursor.location),
                   size_bu=p.size_mm / 1000.0 / unit, smooth=p.smooth)
        Engine.pending.append(dict(
            backend=p.model, name=name, seed=seed, slot=i, ctx=ctx,
            req=dict(cmd="generate", id=f"{name}-{seed}", views=views, out=os.path.join(out_root, name),
                     name=name, seed=seed, remove_bg=p.remove_bg, colors=False, **_model_opts(p))))
        Engine.batch_total += 1
    if p.random_seed:
        p.seed = base_seed
    _set_item_status(Engine.pending[-1], "queued")
    _ensure_timer()


def queue_images(context, paths):
    """Queue generation jobs for image paths with the current panel settings."""
    p = context.scene.l3d
    pr = _prefs()
    backend = p.model
    unit = context.scene.unit_settings.scale_length or 1.0
    out_root = _p(pr.output_dir) or os.path.join(_p(pr.repo_dir), "outputs")
    if Engine.current is None and not Engine.pending:
        Engine.batch_total = Engine.batch_done = 0
    Engine.last_error = ""
    slot = 0
    for img in paths:
        name = bpy.path.clean_name(os.path.splitext(os.path.basename(img))[0])
        base_seed = random.randint(0, 2**31 - 1) if p.random_seed else p.seed
        for i in range(p.variants):
            seed = base_seed + i
            opts = _model_opts(p)
            ctx = dict(scene=context.scene.name, cursor=tuple(context.scene.cursor.location),
                       size_bu=p.size_mm / 1000.0 / unit, smooth=p.smooth)
            Engine.pending.append(dict(
                backend=backend, name=name, seed=seed, slot=slot, ctx=ctx,
                req=dict(cmd="generate", id=f"{name}-{seed}", image=img, out=os.path.join(out_root, name),
                         name=name, seed=seed, remove_bg=p.remove_bg, colors=False, **opts)))
            slot += 1
            Engine.batch_total += 1
        if p.random_seed:
            p.seed = base_seed
        _set_item_status(Engine.pending[-1], "queued")
    _ensure_timer()


# ---------------------------------------------------------------- settings

class L3D_Prefs(bpy.types.AddonPreferences):
    bl_idname = __name__
    repo_dir: StringProperty(name="Local 3D Gen repo", subtype='DIR_PATH', default=_default_repo(),
                             description="Folder containing engine/server.py")
    hunyuan_dir: StringProperty(name="Hunyuan3D-2.1 folder", subtype='DIR_PATH',
                                default=os.path.expanduser(r"~\Hunyuan3D"),
                                description="Clone of Tencent-Hunyuan/Hunyuan3D-2.1 with its .venv")
    hunyuan_mv_dir: StringProperty(name="Hunyuan3D-2 (2mv) folder", subtype='DIR_PATH',
                                   default=os.path.expanduser(r"~\Hunyuan3D-2"),
                                   description="Clone of Tencent-Hunyuan/Hunyuan3D-2 with its .venv")
    trellis_dir: StringProperty(name="TRELLIS folder", subtype='DIR_PATH',
                                default=os.path.expanduser(r"~\TRELLIS"),
                                description="Clone of microsoft/TRELLIS with its .venv")
    output_dir: StringProperty(name="Output folder", subtype='DIR_PATH', default="",
                               description="Where generated GLBs go (blank = <repo>/outputs)")

    def draw(self, context):
        L = self.layout
        L.prop(self, "repo_dir")
        L.prop(self, "hunyuan_dir")
        L.prop(self, "hunyuan_mv_dir")
        L.prop(self, "trellis_dir")
        L.prop(self, "output_dir")
        box = L.box()
        box.label(text="Get the models (weights download automatically on first run):")
        for key, m in MODELS.items():
            row = box.row()
            row.label(text=m["label"])
            row.operator("wm.url_open", text="Weights", icon='URL').url = m["weights"]
            row.operator("wm.url_open", text="Code", icon='URL').url = m["code"]


class L3D_Image(bpy.types.PropertyGroup):
    path: StringProperty(name="Path", subtype='FILE_PATH')
    status: StringProperty(default="")
    view: EnumProperty(name="View", items=VIEWS, default='front',
                       description="Which side of the object this photo shows (multi-view mode)")


class L3D_Props(bpy.types.PropertyGroup):
    model: EnumProperty(name="Model", default='hunyuan', items=[
        ('hunyuan', "Hunyuan 2.1", MODELS['hunyuan']['about']),
        ('hunyuan_mv', "Hunyuan 2mv", MODELS['hunyuan_mv']['about']),
        ('trellis', "TRELLIS", MODELS['trellis']['about'])])
    mode: EnumProperty(name="Mode", default='single', items=[
        ('single', "One model per image", "Each image becomes its own model"),
        ('multi', "Multi-view (one object)", "All images are different angles of the same object")])
    images: CollectionProperty(type=L3D_Image)
    active: IntProperty(default=0)
    auto_generate: BoolProperty(name="Generate on drop", default=True,
                                description="Start generating as soon as images are dropped in")
    remove_bg: BoolProperty(name="Remove background", default=True)
    # Hunyuan3D
    hy_steps: IntProperty(name="Steps", default=50, min=5, max=100, description="More = cleaner, slower")
    hy_guidance: FloatProperty(name="Guidance", default=5.0, min=1.0, max=15.0,
                               description="How strictly to follow the image")
    hy_res: EnumProperty(name="Detail", default='384', items=[
        ('256', "Low (256)", "Fast, blobby"), ('384', "Normal (384)", "Default"),
        ('512', "High (512)", "Finer detail; the final mesh build runs on the CPU and can take several minutes")])
    # TRELLIS
    tr_steps: IntProperty(name="Steps", default=12, min=4, max=50, description="Per stage (structure, detail)")
    tr_guidance: FloatProperty(name="Guidance", default=7.5, min=1.0, max=15.0,
                               description="How strictly the overall shape follows the image")
    tr_colors: EnumProperty(name="Colours", default='model', items=[
        ('model', "TRELLIS's own", "Generated with the shape, so they line up with the geometry. Unseen "
                                   "sides are guessed (use AMS Mirror for symmetric objects)"),
        ('photo', "From photo", "Project the photo(s) onto the model")])
    # common
    seed: IntProperty(name="Seed", default=1234, min=0)
    random_seed: BoolProperty(name="Random", default=True, description="New random seed each run")
    variants: IntProperty(name="Variants", default=1, min=1, max=8,
                          description="Attempts per image with different seeds, laid out side by side")
    size_mm: FloatProperty(name="Longest side (mm)", default=200.0, min=1.0, max=5000.0)
    smooth: BoolProperty(name="Smooth shading", default=True)


# ---------------------------------------------------------------- operators

class L3D_OT_add_images(bpy.types.Operator):
    """Add images to the list (also what runs when you drop image files onto the panel)"""
    bl_idname = "l3d.add_images"
    bl_label = "Add images"
    bl_options = {'REGISTER', 'UNDO'}

    directory: StringProperty(subtype='DIR_PATH', options={'SKIP_SAVE', 'HIDDEN'})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={'SKIP_SAVE', 'HIDDEN'})
    filepath: StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE', 'HIDDEN'})
    filter_image: BoolProperty(default=True, options={'HIDDEN', 'SKIP_SAVE'})
    filter_folder: BoolProperty(default=True, options={'HIDDEN', 'SKIP_SAVE'})
    dropped: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        if self.files or self.filepath:  # came from a drag and drop
            self.dropped = True
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        paths = []
        if self.files and self.directory:
            paths = [os.path.join(self.directory, f.name) for f in self.files if f.name]
        elif self.filepath:
            paths = [self.filepath]
        paths = [_p(x) for x in paths if x.lower().endswith(IMAGE_EXTS) and os.path.isfile(_p(x))]
        if not paths:
            self.report({'WARNING'}, "No image files")
            return {'CANCELLED'}
        p = context.scene.l3d
        known = {_p(it.path): i for i, it in enumerate(p.images)}
        for path in paths:
            if path in known:
                p.active = known[path]
                continue
            used = {it.view for it in p.images}
            it = p.images.add()
            it.path = path
            it.name = os.path.basename(path)
            it.view = next((v for v, _, _ in VIEWS if v not in used), 'front')
            p.active = len(p.images) - 1
        if self.dropped and p.auto_generate and p.mode == 'single':
            queue_images(context, paths)
        return {'FINISHED'}


class L3D_FH_images(bpy.types.FileHandler):
    bl_idname = "L3D_FH_images"
    bl_label = "Make 3D model"
    bl_import_operator = "l3d.add_images"
    bl_file_extensions = ";".join(IMAGE_EXTS)

    @classmethod
    def poll_drop(cls, context):
        # Only the sidebar, so dropping images into the viewport itself keeps Blender's usual behaviour.
        region = getattr(context, "region", None)
        return (context.area is not None and context.area.type == 'VIEW_3D'
                and region is not None and region.type == 'UI')


class L3D_OT_remove_image(bpy.types.Operator):
    bl_idname = "l3d.remove_image"
    bl_label = "Remove image"
    bl_options = {'REGISTER', 'UNDO'}
    clear_all: BoolProperty(default=False)

    def execute(self, context):
        p = context.scene.l3d
        if self.clear_all:
            p.images.clear()
        elif 0 <= p.active < len(p.images):
            p.images.remove(p.active)
        p.active = min(p.active, len(p.images) - 1)
        return {'FINISHED'}


class L3D_OT_generate(bpy.types.Operator):
    bl_idname = "l3d.generate"
    bl_label = "Generate"
    bl_description = "Generate 3D models and import them at the 3D cursor"
    all_images: BoolProperty(default=False)

    def execute(self, context):
        p = context.scene.l3d
        if p.mode == 'multi':
            if not MODELS[p.model]["multiview"]:
                self.report({'ERROR'}, f"{MODELS[p.model]['label']} takes one image: pick Hunyuan 2mv or TRELLIS")
                return {'CANCELLED'}
            views = {}
            for it in p.images:
                if it.view in views:
                    self.report({'ERROR'}, f"Two images are set to '{it.view}'; give each a different view")
                    return {'CANCELLED'}
                if os.path.isfile(_p(it.path)):
                    views[it.view] = _p(it.path)
            if not views:
                self.report({'ERROR'}, "Drop or add images first")
                return {'CANCELLED'}
            if p.model == 'hunyuan_mv' and 'front' not in views:
                self.report({'ERROR'}, "Hunyuan 2mv needs a front view")
                return {'CANCELLED'}
            queue_multiview(context, views)
            return {'FINISHED'}
        items = list(p.images) if self.all_images else ([p.images[p.active]] if 0 <= p.active < len(p.images) else [])
        paths = [_p(it.path) for it in items if os.path.isfile(_p(it.path))]
        if not paths:
            self.report({'ERROR'}, "Drop or add an image first")
            return {'CANCELLED'}
        queue_images(context, paths)
        return {'FINISHED'}


class L3D_OT_start(bpy.types.Operator):
    bl_idname = "l3d.start_engine"
    bl_label = "Start engine"
    bl_description = "Load the selected model onto the GPU now, so the first generation starts faster"

    def execute(self, context):
        try:
            Engine.start(context.scene.l3d.model)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class L3D_OT_stop(bpy.types.Operator):
    bl_idname = "l3d.stop_engine"
    bl_label = "Stop engine"
    bl_description = "Shut the engine down and free GPU memory"

    def execute(self, context):
        Engine.stop()
        _redraw()
        return {'FINISHED'}


class L3D_OT_cancel(bpy.types.Operator):
    bl_idname = "l3d.cancel"
    bl_label = "Cancel"
    bl_description = "Drop queued jobs and stop the one in progress (the model reloads next time)"

    def execute(self, context):
        running = Engine.current is not None
        Engine.pending.clear()
        for it in context.scene.l3d.images:
            if it.status in ("queued", "generating"):
                it.status = ""
        if running:
            Engine.stop(hard=True)
        Engine.batch_total = Engine.batch_done = 0
        _redraw()
        return {'FINISHED'}


class L3D_OT_open_outputs(bpy.types.Operator):
    bl_idname = "l3d.open_outputs"
    bl_label = "Open outputs folder"

    def execute(self, context):
        pr = _prefs()
        d = _p(pr.output_dir) or os.path.join(_p(pr.repo_dir), "outputs")
        os.makedirs(d, exist_ok=True)
        os.startfile(d)
        return {'FINISHED'}


class L3D_OT_select(bpy.types.Operator):
    bl_idname = "l3d.select"
    bl_label = "Select"
    name: StringProperty()

    def execute(self, context):
        ob = bpy.data.objects.get(self.name)
        if ob is None or ob.name not in context.view_layer.objects:
            self.report({'WARNING'}, "Object not in this scene")
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        return {'FINISHED'}


# ---------------------------------------------------------------- panel

def _icon(path):
    if not path or not os.path.isfile(path):
        return 0
    if path not in _previews:
        try:
            _previews.load(path, path, 'IMAGE')
        except Exception:
            return 0
    return _previews[path].icon_id


class L3D_UL_images(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index):
        row = layout.row(align=True)
        if context.scene.l3d.mode == 'multi':
            sub = row.row(align=True)
            sub.ui_units_x = 3.2
            sub.prop(item, "view", text="")
        row.label(text=item.name, icon_value=_icon(_p(item.path)) or 'IMAGE_DATA')
        if item.status:
            row.label(text=item.status)


class L3D_PT_panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Image to 3D"
    bl_label = "Generate"
    bl_order = 0

    def draw(self, context):
        p = context.scene.l3d
        L = self.layout
        busy = Engine.current is not None or bool(Engine.pending)

        row = L.row(align=True)
        row.prop(p, "model", expand=True)
        m = MODELS[p.model]
        row.operator("wm.url_open", text="", icon='URL').url = m["weights"]

        row = L.row(align=True)
        if Engine.alive() and Engine.ready:
            row.label(text=f"{MODELS[Engine.backend]['label']} loaded", icon='CHECKMARK')
            row.operator("l3d.stop_engine", text="", icon='X')
        elif Engine.alive():
            row.label(text=f"{Engine.status} {time.time() - Engine.started_at:.0f}s", icon='TIME')
            row.operator("l3d.stop_engine", text="", icon='X')
        else:
            row.label(text="Model not loaded", icon='SHADING_WIRE')
            row.operator("l3d.start_engine", text="Load", icon='PLAY')

        L.row(align=True).prop(p, "mode", expand=True)
        if p.mode == 'multi' and not MODELS[p.model]["multiview"]:
            warn = L.row()
            warn.alert = True
            warn.label(text="Pick Hunyuan 2mv or TRELLIS for several views", icon='ERROR')
        row = L.row()
        row.template_list("L3D_UL_images", "", p, "images", p, "active", rows=2)
        col = row.column(align=True)
        col.operator("l3d.add_images", text="", icon='ADD')
        col.operator("l3d.remove_image", text="", icon='REMOVE').clear_all = False
        col.operator("l3d.remove_image", text="", icon='TRASH').clear_all = True
        if 0 <= p.active < len(p.images):
            ic = _icon(_p(p.images[p.active].path))
            if ic:
                L.template_icon(icon_value=ic, scale=4)
        else:
            L.label(text="Drop images onto this panel", icon='IMPORT')

        row = L.row(align=True)
        row.scale_y = 1.4
        if p.mode == 'multi':
            text = f"Generate from {len(p.images)} view{'s' if len(p.images) != 1 else ''}"
            row.operator("l3d.generate", text="Queue more" if busy else text, icon='MESH_MONKEY').all_images = True
        else:
            row.operator("l3d.generate", text="Queue more" if busy else "Generate", icon='MESH_MONKEY').all_images = False
            if len(p.images) > 1:
                row.operator("l3d.generate", text="All").all_images = True
        if busy:
            row.operator("l3d.cancel", text="", icon='CANCEL')
            n = f"  {Engine.batch_done + 1}/{Engine.batch_total}" if Engine.batch_total > 1 else ""
            if Engine.current is None:
                L.progress(factor=0.0, type='BAR', text=f"Loading model...{n}")
            else:
                secs = int(time.time() - Engine.stage_at)
                if Engine.stage in ("building mesh", "cleaning up", "saving", "removing background", "starting",
                                    "colouring"):
                    text = f"{Engine.stage.capitalize()}... {secs}s{n}"   # no step count for these stages
                else:
                    text = f"{Engine.stage.capitalize()} {int(Engine.frac * 100)}%{n}"
                L.progress(factor=Engine.frac, type='BAR', text=text)
        if Engine.last_error:
            col = L.column()
            col.alert = True
            col.label(text=Engine.last_error, icon='ERROR')


class L3D_PT_settings(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Image to 3D"
    bl_label = "Settings"
    bl_parent_id = "L3D_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        p = context.scene.l3d
        L = self.layout
        col = L.column(align=True)
        if p.model in ('hunyuan', 'hunyuan_mv'):
            col.prop(p, "hy_res")
            col.prop(p, "hy_steps")
            col.prop(p, "hy_guidance")
        else:
            col.prop(p, "tr_steps")
            col.prop(p, "tr_guidance")
        row = L.row(align=True)
        sub = row.row(align=True)
        sub.enabled = not p.random_seed
        sub.prop(p, "seed")
        row.prop(p, "random_seed", toggle=True, icon='FILE_REFRESH', text="")
        L.prop(p, "variants")
        L.prop(p, "size_mm")
        row = L.row()
        row.prop(p, "auto_generate", text="Generate on drop")
        row.prop(p, "remove_bg", text="Cut out")
        L.operator("l3d.open_outputs", icon='FILE_FOLDER')


classes = (L3D_Prefs, L3D_Image, L3D_Props, L3D_OT_add_images, L3D_FH_images, L3D_OT_remove_image,
           L3D_OT_generate, L3D_OT_start, L3D_OT_stop, L3D_OT_cancel, L3D_OT_open_outputs, L3D_OT_select,
           L3D_UL_images, L3D_PT_panel, L3D_PT_settings)


def register():
    global _previews
    _previews = bpy.utils.previews.new()
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.l3d = PointerProperty(type=L3D_Props)
    colour.register()


def unregister():
    global _previews
    Engine.stop(hard=True)
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    colour.unregister()
    del bpy.types.Scene.l3d
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    bpy.utils.previews.remove(_previews)
    _previews = None
