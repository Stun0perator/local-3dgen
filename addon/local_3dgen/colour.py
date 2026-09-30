"""Colour a model for multi-filament (AMS) printing by clicking regions.

1. Split into regions: the surface is cut into patches along its creases and seams (where a rib rises off
   a shell, where a rim turns into a hole, where a foot meets the base), so region borders sit on the shape.
2. Click to colour: pick a filament swatch, click a region to fill it. Shift-click fills every region with
   the same size and shape (all the feet, all the rib arms). Alt-drag paints with a brush for colour changes
   that don't follow the shape. Ctrl+Z undoes, 1-9 pick a swatch, [ ] change the brush size.
   Fill all sets a base colour first.
3. Band: colour everything between two heights, with a straight edge (bases, stripes on pots).
4. Export 3MF for Bambu Studio with one filament per swatch.

Colours are stored as material slots on the mesh (one material per swatch), so Blender's own Edit Mode
material assignment works too. Pure numpy + Blender, no extra dependencies.
"""
import glob
import json
import os
import zipfile

import bpy
import gpu
import numpy as np
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader

from . import bambu3mf

MAT_PREFIX = "AMS "
DEFAULT_COLOURS = [(0.93, 0.91, 0.87), (0.10, 0.10, 0.10), (0.64, 0.24, 0.17), (0.82, 0.65, 0.36),
                   (0.20, 0.35, 0.65), (0.30, 0.55, 0.30), (0.85, 0.45, 0.15), (0.55, 0.30, 0.60)]


# ---------------------------------------------------------------- small helpers

def srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def features(srgb):
    """Lab with lightness down-weighted: shading in photos changes lightness more than hue."""
    lin = srgb_to_linear(srgb)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([0.45 * (116 * f[:, 1] - 16), 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def kmeans(x, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min(((x[:, None] - np.array(c)[None]) ** 2).sum(-1), 1)
        c.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
    c = np.array(c)
    for _ in range(iters):
        lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any():
                c[j] = x[lab == j].mean(0)
    return np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), 1)


def mm_per_unit(context, ob):
    """Millimetres per local mesh unit, including the object's scale."""
    unit = context.scene.unit_settings.scale_length or 1.0
    s = np.cbrt(abs(np.linalg.det(np.array(ob.matrix_world.to_3x3()))))
    return unit * 1000.0 * s


def face_adjacency(me):
    """(face a, face b, edge) for every edge shared by exactly two faces."""
    nf = len(me.polygons)
    totals = np.empty(nf, dtype=np.int64); me.polygons.foreach_get('loop_total', totals)
    loop_face = np.repeat(np.arange(nf), totals)
    loop_edge = np.empty(len(me.loops), dtype=np.int64); me.loops.foreach_get('edge_index', loop_edge)
    order = np.argsort(loop_edge, kind='stable')
    e, f = loop_edge[order], loop_face[order]
    same = e[1:] == e[:-1]
    cnt = np.bincount(e, minlength=len(me.edges))
    ok = cnt[e[:-1][same]] == 2
    return f[:-1][same][ok], f[1:][same][ok], e[:-1][same][ok]


def components(n, a, b):
    """Connected components over edges (a, b): a root id per node."""
    parent = np.arange(n)
    if len(a) == 0:
        return parent
    while True:
        pa, pb = parent[a], parent[b]
        if (pa == pb).all():
            return parent
        m = np.minimum(pa, pb)
        np.minimum.at(parent, pa, m)
        np.minimum.at(parent, pb, m)
        while True:
            pp = parent[parent]
            if (pp == parent).all():
                break
            parent = pp


def face_centres_mm(context, ob):
    me = ob.data
    cen = np.empty(len(me.polygons) * 3); me.polygons.foreach_get('center', cen)
    M = np.array(ob.matrix_world)
    unit = (context.scene.unit_settings.scale_length or 1.0) * 1000.0
    return (cen.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]) * unit


def get_labels(me):
    lab = np.empty(len(me.polygons), dtype=np.int32); me.polygons.foreach_get('material_index', lab)
    return lab


def set_labels(me, labels):
    me.polygons.foreach_set('material_index', labels.astype(np.int32))
    me.update()


def active_mesh(context):
    ob = context.active_object
    return ob if ob is not None and ob.type == 'MESH' else None


# ---------------------------------------------------------------- palette / materials

def _colour_changed(self, context):
    for mat in bpy.data.materials:
        if mat.get("l3d_ams_slot") and context.scene.l3d_colour.palette:
            i = mat["l3d_ams_slot"] - 1
            pal = context.scene.l3d_colour.palette
            if i < len(pal):
                _set_mat_colour(mat, pal[i].color)


def _count_changed(self, context):
    pal = self.palette
    while len(pal) < self.count:
        it = pal.add(); it.color = DEFAULT_COLOURS[(len(pal) - 1) % len(DEFAULT_COLOURS)]
    while len(pal) > self.count:
        pal.remove(len(pal) - 1)
    self.active_slot = min(self.active_slot, self.count)
    ob = active_mesh(context)
    if ob is not None and has_colouring(ob):
        ensure_materials(context, ob)


def _set_mat_colour(mat, srgb):
    lin = tuple(float(x) for x in srgb_to_linear(np.array(srgb[:3]))) + (1.0,)
    mat.diffuse_color = lin
    if mat.use_nodes:
        bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf:
            bsdf.inputs['Base Color'].default_value = lin


def has_colouring(ob):
    return any(m is not None and m.get("l3d_ams_slot") for m in ob.data.materials)


def ensure_materials(context, ob):
    """One material per swatch on the object, slots in palette order; existing face colours are kept (faces on
    removed swatches fall back to swatch 1)."""
    p = context.scene.l3d_colour
    me = ob.data
    labels = get_labels(me) if has_colouring(ob) else np.zeros(len(me.polygons), dtype=np.int32)
    me.materials.clear()
    for i, it in enumerate(p.palette):
        name = f"{MAT_PREFIX}{i + 1} ({ob.name})"
        mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        mat["l3d_ams_slot"] = i + 1
        _set_mat_colour(mat, it.color)
        me.materials.append(mat)
    labels[labels >= len(p.palette)] = 0
    set_labels(me, labels)


def source_image(ob):
    """Path of the cut-out front photo the engine saved next to the model, if any."""
    raw = ob.get("l3d_inputs")
    if raw:
        views = json.loads(raw)
        for key in ('front', 'left', 'back', 'right'):
            if key in views and os.path.isfile(views[key]):
                return views[key]
    glb = ob.get("l3d_glb", "")
    if glb:
        hits = sorted(glob.glob(os.path.splitext(glb)[0] + '_input*.png'))
        if hits:
            return hits[0]
    return None


def load_rgba(path, max_side=512):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        px = np.empty(w * h * 4, dtype=np.float32); img.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(img)
    px = px.reshape(h, w, 4)[::-1]
    step = int(np.ceil(max(w, h) / max_side))
    return px[::step, ::step]


# ---------------------------------------------------------------- regions

def seam_strength(context, ob, a, b, radius_mm=1.5):
    """Per face: how far the surrounding surface (within ~radius_mm) sits above the face along its normal, in mm.
    Positive where the surface folds inward: the seam where a rib meets a shell, a foot meets a base, a liner
    meets a rim. Convex shapes (a rib's rounded top, a rim's lip) come out negative, so they stay whole."""
    me = ob.data
    nf = len(me.polygons)
    nor = np.empty(nf * 3); me.polygons.foreach_get('normal', nor); nor = nor.reshape(-1, 3)
    cen = np.empty(nf * 3); me.polygons.foreach_get('center', cen)
    cen = cen.reshape(-1, 3) * mm_per_unit(context, ob)
    deg = (np.bincount(a, minlength=nf) + np.bincount(b, minlength=nf) + 1)[:, None]
    edge = np.linalg.norm(cen[a] - cen[b], axis=1).mean()
    iters = int(np.clip(round((radius_mm / max(edge, 1e-6)) ** 2 / 2), 1, 200))
    x = cen.copy()
    for _ in range(iters):  # average the neighbourhood's position (a diffusion over ~radius_mm)
        acc = x.copy(); np.add.at(acc, a, x[b]); np.add.at(acc, b, x[a]); x = acc / deg
    return ((x - cen) * nor).sum(1)


def split_regions(context, ob, min_mm2, sensitivity=1.0):
    """Cut the surface into regions by its shape (see seam_strength):
    - ridges (clearly convex: ribs, rims, lips, feet, knobs) form regions of their own,
    - the smoother surface between them is cut along its concave seams (where parts meet),
    - seam faces and regions smaller than min_mm2 are absorbed by their neighbours.
    Stored as the face attribute 'l3d_region'. Returns the number of regions."""
    me = ob.data
    nf = len(me.polygons)
    a, b, _ = face_adjacency(me)
    H = seam_strength(context, ob, a, b)
    s = max(sensitivity, 1e-3)
    cls = np.where(H < -0.02 / s, 1, np.where(H > 0.006 / s, 2, 0))  # 1 ridge, 0 surface, 2 seam
    keep = (cls[a] == cls[b]) & (cls[a] != 2)
    comp = components(nf, a[keep], b[keep])
    area = np.empty(nf); me.polygons.foreach_get('area', area)
    area *= mm_per_unit(context, ob) ** 2
    comp_area = np.bincount(comp, weights=area, minlength=nf)
    label = np.where((cls == 2) | (comp_area[comp] < min_mm2), -1, comp)
    src, dst = np.concatenate([a, b]), np.concatenate([b, a])
    for _ in range(3000):  # grow the regions into the seams and the dropped slivers
        free = label < 0
        if not free.any():
            break
        m = free[dst] & ~free[src]
        if not m.any():
            label[free] = comp[free]  # isolated leftovers keep their own component
            break
        label[dst[m]] = label[src[m]]
    _, region = np.unique(label, return_inverse=True)
    at = me.attributes.get("l3d_region") or me.attributes.new("l3d_region", 'INT', 'FACE')
    at.data.foreach_set('value', region.astype(np.int32))
    ob["l3d_region_version"] = ob.get("l3d_region_version", 0) + 1
    return int(region.max()) + 1


def get_regions(ob):
    at = ob.data.attributes.get("l3d_region")
    if at is None or at.domain != 'FACE' or len(at.data) != len(ob.data.polygons):
        return None
    r = np.empty(len(at.data), dtype=np.int32); at.data.foreach_get('value', r)
    return r


class RegionData:
    """Per-object cache for the click tool: region -> faces, shape descriptors, border lines."""

    def __init__(self, context, ob):
        me = ob.data
        self.region = get_regions(ob)
        order = np.argsort(self.region, kind='stable')
        counts = np.bincount(self.region)
        self.order, self.starts, self.counts = order, np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
        self.cen = face_centres_mm(context, ob)
        area = np.empty(len(me.polygons)); me.polygons.foreach_get('area', area)
        area *= mm_per_unit(context, ob) ** 2
        n = len(counts)
        self.area = np.bincount(self.region, weights=area, minlength=n)
        # shape descriptor: spread along the region's three principal axes (area weighted)
        mean = np.stack([np.bincount(self.region, weights=area * self.cen[:, i], minlength=n) for i in range(3)], 1)
        mean /= np.maximum(self.area, 1e-9)[:, None]
        dev = self.cen - mean[self.region]
        cov = np.zeros((n, 3, 3))
        for i in range(3):
            for j in range(i, 3):
                cov[:, i, j] = cov[:, j, i] = np.bincount(self.region, weights=area * dev[:, i] * dev[:, j],
                                                          minlength=n) / np.maximum(self.area, 1e-9)
        self.spread = np.sqrt(np.clip(np.sort(np.linalg.eigvalsh(cov), 1)[:, ::-1], 0, None))
        # border lines between regions, nudged off the surface so they draw on top
        a, b, e = face_adjacency(me)
        diff = self.region[a] != self.region[b]
        ev = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get('vertices', ev)
        ev = ev.reshape(-1, 2)[e[diff]]
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
        vn = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('normal', vn); vn = vn.reshape(-1, 3)
        M = np.array(ob.matrix_world)
        lift = 0.15 / mm_per_unit(context, ob)
        pts = (co[ev.ravel()] + vn[ev.ravel()] * lift) @ M[:3, :3].T + M[:3, 3]
        self.lines = pts.astype(np.float32)

    def faces(self, r):
        return self.order[self.starts[r]:self.starts[r] + self.counts[r]]

    def similar(self, r, tol=0.2):
        """Regions of about the same size and shape as r (e.g. the other feet)."""
        ok = np.abs(self.area / max(self.area[r], 1e-9) - 1) < tol * 1.5
        s = self.spread / np.maximum(self.spread[r], 1e-6)
        ok &= np.all(np.abs(s - 1) < tol + 0.1, axis=1)
        return np.nonzero(ok)[0]


# ---------------------------------------------------------------- the click tool

class L3D_OT_paint_regions(bpy.types.Operator):
    """Click regions to colour them with the active swatch.
    Click: fill region. Shift-click: fill every matching region. Alt-drag: brush. 1-9: swatch.
    [ ]: brush size. Ctrl+Z: undo. Right-click, Enter or Esc: done"""
    bl_idname = "l3d.paint_regions"
    bl_label = "Click to colour"
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        ob = active_mesh(context)
        if ob is None:
            self.report({'ERROR'}, "Select a model first"); return {'CANCELLED'}
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Use it from the 3D Viewport"); return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        p = context.scene.l3d_colour
        if get_regions(ob) is None:
            split_regions(context, ob, p.region_size, p.sensitivity)
        if not has_colouring(ob):
            ensure_materials(context, ob)
        self.ob = ob
        self.data = RegionData(context, ob)
        self.labels = get_labels(ob.data)
        self.undo = []
        self.painting = False
        self.stroke = None
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        self.batch = batch_for_shader(shader, 'LINES', {"pos": self.data.lines})
        self.shader = shader
        args = (context,)
        self._draw = bpy.types.SpaceView3D.draw_handler_add(self.draw_lines, args, 'WINDOW', 'POST_VIEW')
        self._hud = bpy.types.SpaceView3D.draw_handler_add(self.draw_hud, args, 'WINDOW', 'POST_PIXEL')
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.color_type = 'MATERIAL'
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    # -- drawing
    def draw_lines(self, context):
        if not context.scene.l3d_colour.show_regions:
            return
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.blend_set('ALPHA')
        self.shader.bind()
        self.shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
        self.shader.uniform_float("lineWidth", 1.0)
        self.shader.uniform_float("color", (0.05, 0.05, 0.05, 0.55))
        self.batch.draw(self.shader)
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')

    def draw_hud(self, context):
        import blf
        p = context.scene.l3d_colour
        txt = (f"Click to colour  |  swatch {p.active_slot}  |  brush {p.brush_mm:.1f} mm  |  "
               "click fill · shift matching · alt-drag brush · 1-9 swatch · [ ] brush · ctrl+Z undo · "
               "right-click done")
        blf.size(0, 13)
        blf.color(0, 1, 1, 1, 0.9)
        blf.position(0, 20, 30, 0)
        blf.draw(0, txt)

    # -- actions
    def hit(self, context, event):
        region, rv3d = context.region, context.region_data
        co = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, co)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, co)
        inv = self.ob.matrix_world.inverted()
        o = inv @ origin
        d = (inv.to_3x3() @ direction).normalized()
        ok, loc, _, face = self.ob.ray_cast(o, d)
        if not ok:
            return None, None
        unit = (context.scene.unit_settings.scale_length or 1.0) * 1000.0
        return face, np.array(self.ob.matrix_world @ loc) * unit

    def apply(self, faces, slot):
        faces = faces[self.labels[faces] != slot]
        if len(faces) == 0:
            return
        if self.stroke is not None:
            self.stroke.append((faces, self.labels[faces].copy()))
        else:
            self.undo.append([(faces, self.labels[faces].copy())])
        self.labels[faces] = slot
        set_labels(self.ob.data, self.labels)

    def brush(self, context, point):
        r = context.scene.l3d_colour.brush_mm
        near = np.nonzero(((self.data.cen - point) ** 2).sum(1) <= r * r)[0]
        self.apply(near, context.scene.l3d_colour.active_slot - 1)

    def finish(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self._draw, 'WINDOW')
        bpy.types.SpaceView3D.draw_handler_remove(self._hud, 'WINDOW')
        _update_shares(context, self.ob)
        context.area.tag_redraw()

    def modal(self, context, event):
        p = context.scene.l3d_colour
        context.area.tag_redraw()
        if event.type in {'RIGHTMOUSE', 'ESC', 'RET'} and event.value == 'PRESS':
            self.finish(context)
            return {'FINISHED'}
        if event.type == 'Z' and event.ctrl and event.value == 'PRESS':
            if self.undo:
                for faces, old in reversed(self.undo.pop()):
                    self.labels[faces] = old
                set_labels(self.ob.data, self.labels)
            return {'RUNNING_MODAL'}
        keys = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']
        if event.type in keys and event.value == 'PRESS':
            n = keys.index(event.type) + 1
            if n <= p.count:
                p.active_slot = n
            return {'RUNNING_MODAL'}
        if event.type in {'LEFT_BRACKET', 'RIGHT_BRACKET'} and event.value == 'PRESS':
            p.brush_mm = max(0.2, p.brush_mm * (0.8 if event.type == 'LEFT_BRACKET' else 1.25))
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                face, point = self.hit(context, event)
                if face is None:
                    return {"RUNNING_MODAL"}  # a click beside the model does nothing (no deselecting)
                slot = p.active_slot - 1
                if event.alt:
                    self.painting = True
                    self.stroke = []
                    self.brush(context, point)
                elif event.shift:
                    r = self.data.region[face]
                    faces = np.concatenate([self.data.faces(x) for x in self.data.similar(r)])
                    self.apply(faces, slot)
                else:
                    self.apply(self.data.faces(self.data.region[face]), slot)
                return {'RUNNING_MODAL'}
            if event.value == 'RELEASE' and self.painting:
                self.painting = False
                if self.stroke:
                    self.undo.append(self.stroke)
                self.stroke = None
                return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE' and self.painting:
            face, point = self.hit(context, event)
            if face is not None:
                self.brush(context, point)
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}  # orbit, pan, zoom work as usual


def _update_shares(context, ob):
    p = context.scene.l3d_colour
    me = ob.data
    area = np.empty(len(me.polygons)); me.polygons.foreach_get('area', area)
    shares = np.bincount(get_labels(me), weights=area, minlength=len(p.palette)) / max(area.sum(), 1e-12)
    for it, s in zip(p.palette, shares):
        it.share = float(s)


# ---------------------------------------------------------------- other operators

class L3D_OT_split_regions(bpy.types.Operator):
    """Cut the model into regions along its creases and seams, ready for Click to colour"""
    bl_idname = "l3d.split_regions"
    bl_label = "Split into regions"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = active_mesh(context)
        if ob is None:
            self.report({'ERROR'}, "Select a model first"); return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        p = context.scene.l3d_colour
        me = ob.data
        if p.max_faces and len(me.polygons) > p.max_faces * 1.05:
            md = ob.modifiers.new("l3d_simplify", 'DECIMATE')
            md.ratio = p.max_faces / len(me.polygons)
            with context.temp_override(object=ob, active_object=ob):
                bpy.ops.object.modifier_apply(modifier=md.name)
        n = split_regions(context, ob, p.region_size, p.sensitivity)
        if not has_colouring(ob):
            ensure_materials(context, ob)
        self.report({'INFO'}, f"{n:,} regions on {len(ob.data.polygons):,} triangles")
        return {'FINISHED'}


class L3D_OT_set_slot(bpy.types.Operator):
    """Make this the swatch that clicks paint with"""
    bl_idname = "l3d.set_slot"
    bl_label = "Use swatch"
    slot: IntProperty()

    def execute(self, context):
        context.scene.l3d_colour.active_slot = self.slot
        return {'FINISHED'}


class L3D_OT_palette_from_photo(bpy.types.Operator):
    """Fill the swatches with the main colours of the model's source photo"""
    bl_idname = "l3d.palette_from_photo"
    bl_label = "Colours from photo"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = active_mesh(context)
        path = source_image(ob) if ob else None
        if not path:
            self.report({'ERROR'}, "No source photo for this model"); return {'CANCELLED'}
        px = load_rgba(path)
        px = px[px[..., 3] > 0.5][:, :3]
        rng = np.random.default_rng(0)
        px = px[rng.choice(len(px), size=min(len(px), 40000), replace=False)]
        p = context.scene.l3d_colour
        lab = kmeans(features(px), p.count)
        cols = []
        for j in np.argsort(-np.bincount(lab, minlength=p.count)):
            m = px[lab == j]
            if len(m) == 0:
                continue
            L = features(m)[:, 0]
            lo, hi = np.percentile(L, [50, 90])  # the lit part: shadows pull a plain mean down
            sel = (L >= lo) & (L <= hi)
            cols.append(m[sel].mean(0) if sel.any() else m.mean(0))
        for it, c in zip(p.palette, cols):
            it.color = tuple(float(x) for x in c)
        return {'FINISHED'}


class L3D_OT_fill_all(bpy.types.Operator):
    """Give the whole model the active swatch (a base colour to start clicking from)"""
    bl_idname = "l3d.fill_all"
    bl_label = "Fill all"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = active_mesh(context)
        if ob is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if not has_colouring(ob):
            ensure_materials(context, ob)
        set_labels(ob.data, np.full(len(ob.data.polygons), context.scene.l3d_colour.active_slot - 1))
        _update_shares(context, ob)
        return {'FINISHED'}


class L3D_OT_level_band(bpy.types.Operator):
    """Colour a horizontal band: everything between two heights gets the active swatch, with a straight,
    subdivided edge (bases, stripes on pots)"""
    bl_idname = "l3d.level_band"
    bl_label = "Paint band"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import bmesh
        ob = active_mesh(context)
        if ob is None:
            return {'CANCELLED'}
        p = context.scene.l3d_colour
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if not has_colouring(ob):
            ensure_materials(context, ob)
        lo, hi = sorted((p.band_from, p.band_to))
        slot = p.active_slot - 1
        me = ob.data
        for level in range(3):  # colour, split the triangles the band edges cross, colour again
            z = face_centres_mm(context, ob)[:, 2]
            z -= z.min()
            labels = get_labels(me)
            inside = (z >= lo) & (z < hi)
            labels[inside] = slot
            set_labels(me, labels)
            if level == 2:
                break
            a, b, _ = face_adjacency(me)
            edge_faces = np.zeros(len(labels), bool)
            cross = inside[a] != inside[b]
            edge_faces[a[cross]] = True
            edge_faces[b[cross]] = True
            if not edge_faces.any():
                break
            bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
            edges = {e for i in np.nonzero(edge_faces)[0] for e in bm.faces[i].edges}
            res = bmesh.ops.subdivide_edges(bm, edges=list(edges), cuts=1, use_grid_fill=True)
            touched = {f for e in res['geom_split'] if isinstance(e, bmesh.types.BMEdge) for f in e.link_faces}
            bmesh.ops.triangulate(bm, faces=[f for f in touched if len(f.verts) > 3])
            bm.to_mesh(me); bm.free(); me.update()
        if get_regions(ob) is None:  # subdividing reset the region attribute on new faces: recompute
            split_regions(context, ob, p.region_size, p.sensitivity)
        _update_shares(context, ob)
        return {'FINISHED'}


class L3D_OT_export_3mf(bpy.types.Operator):
    """Export the selected model as a Bambu Studio 3MF, one filament per swatch"""
    bl_idname = "l3d.export_3mf"
    bl_label = "Export 3MF"

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.3mf", options={'HIDDEN'})

    def invoke(self, context, event):
        ob = active_mesh(context)
        if ob is None:
            self.report({'ERROR'}, "Select a model first"); return {'CANCELLED'}
        if not self.filepath:
            self.filepath = bpy.path.clean_name(ob.name) + ".3mf"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        ob = active_mesh(context)
        if ob is None or not has_colouring(ob):
            self.report({'ERROR'}, "Colour the model first"); return {'CANCELLED'}
        p = context.scene.l3d_colour
        path = bpy.path.ensure_ext(bpy.path.abspath(self.filepath), ".3mf")
        unit = (context.scene.unit_settings.scale_length or 1.0) * 1000.0
        co, tri, mat = bambu3mf.triangles(ob, context.evaluated_depsgraph_get(), unit)
        slots = np.array([int(m.get("l3d_ams_slot", i + 1)) if m else i + 1
                          for i, m in enumerate(ob.data.materials)] or [1])
        ext = slots[np.clip(mat, 0, len(slots) - 1)]
        lo, hi = co.min(0), co.max(0)
        co = co - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])  # centred, on the bed
        cols = [it.color[:] for it in p.palette]
        cols += [(0.8, 0.8, 0.8)] * max(0, int(ext.max()) - len(cols))
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            for fn, text in bambu3mf.package(bpy.path.clean_name(ob.name), co, tri, ext, cols, p.printer):
                z.writestr(fn, text)
        s = hi - lo
        self.report({'INFO'}, f"Saved {os.path.basename(path)}: {len(tri):,} triangles, "
                              f"{s[0]:.0f} x {s[1]:.0f} x {s[2]:.0f} mm")
        return {'FINISHED'}


# ---------------------------------------------------------------- settings + panels

class L3D_Swatch(bpy.types.PropertyGroup):
    color: FloatVectorProperty(name="Colour", subtype='COLOR_GAMMA', size=3, min=0, max=1,
                               default=(0.8, 0.8, 0.8), update=_colour_changed)
    share: FloatProperty(default=0.0)


class L3D_ColourProps(bpy.types.PropertyGroup):
    count: IntProperty(name="Filaments", default=4, min=1, max=16, update=_count_changed,
                       description="How many filaments (AMS slots)")
    palette: CollectionProperty(type=L3D_Swatch)
    active_slot: IntProperty(name="Swatch", default=1, min=1, max=16)
    region_size: FloatProperty(name="Smallest region (mm²)", default=20.0, min=0.5, soft_max=500.0,
                               description="Regions smaller than this are absorbed by their neighbours")
    sensitivity: FloatProperty(name="Seam sensitivity", default=1.0, min=0.1, soft_max=5.0,
                               description="Higher = shallower seams also split regions")
    max_faces: IntProperty(name="Max triangles", default=1000000, min=0, soft_max=3000000,
                           description="Simplify dense meshes to this before splitting (0 = keep all). "
                                       "Bambu Studio gets slow above ~1M")
    show_regions: BoolProperty(name="Show region outlines", default=True)
    brush_mm: FloatProperty(name="Brush (mm)", default=2.0, min=0.2, soft_max=30.0)
    band_from: FloatProperty(name="From (mm)", default=0.0, min=0.0, description="Band bottom, from the base")
    band_to: FloatProperty(name="To (mm)", default=20.0, min=0.0, description="Band top, from the base")
    printer: EnumProperty(name="Printer", default='X1C', items=[
        ('X1C', "X1 Carbon", ""), ('X1E', "X1E", ""), ('P1S', "P1S", ""), ('P1P', "P1P", ""),
        ('A1', "A1", ""), ('A1M', "A1 mini", "")])


class L3D_PT_colour(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Image to 3D"
    bl_label = "Colour for AMS"
    bl_order = 1

    def draw(self, context):
        p = context.scene.l3d_colour
        L = self.layout
        ob = active_mesh(context)
        if ob is None:
            L.label(text="Select a model", icon='INFO')
            return
        row = L.row(align=True)
        row.prop(p, "count")
        row.operator("l3d.palette_from_photo", text="", icon='EYEDROPPER')
        grid = L.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
        for i, it in enumerate(p.palette):
            r = grid.row(align=True)
            op = r.operator("l3d.set_slot", text=str(i + 1), depress=(p.active_slot == i + 1))
            op.slot = i + 1
            r.prop(it, "color", text="")
        row = L.row(align=True)
        row.operator("l3d.fill_all", icon='SNAP_FACE')
        row = L.row(align=True)
        row.operator("l3d.split_regions", icon='MOD_EDGESPLIT')
        row.prop(p, "sensitivity", text="Seams")
        row = L.row()
        row.scale_y = 1.5
        row.operator("l3d.paint_regions", icon='BRUSH_DATA')
        row = L.row(align=True)
        row.prop(p, "brush_mm")
        row.prop(p, "show_regions", text="", icon='MOD_WIREFRAME')


class L3D_PT_colour_regions(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Image to 3D"
    bl_label = "Region options"
    bl_parent_id = "L3D_PT_colour"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        p = context.scene.l3d_colour
        self.layout.prop(p, "region_size")
        self.layout.prop(p, "max_faces")


class L3D_PT_colour_band(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Image to 3D"
    bl_label = "Band"
    bl_parent_id = "L3D_PT_colour"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        p = context.scene.l3d_colour
        row = self.layout.row(align=True)
        row.prop(p, "band_from", text="From")
        row.prop(p, "band_to", text="To")
        self.layout.operator("l3d.level_band", icon='ALIGN_JUSTIFY')


class L3D_PT_colour_export(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Image to 3D"
    bl_label = "Export"
    bl_parent_id = "L3D_PT_colour"

    def draw(self, context):
        p = context.scene.l3d_colour
        row = self.layout.row(align=True)
        row.prop(p, "printer", text="")
        row.operator("l3d.export_3mf", icon='EXPORT')


classes = (L3D_Swatch, L3D_ColourProps, L3D_OT_paint_regions, L3D_OT_split_regions, L3D_OT_set_slot,
           L3D_OT_palette_from_photo, L3D_OT_fill_all, L3D_OT_level_band, L3D_OT_export_3mf,
           L3D_PT_colour, L3D_PT_colour_regions, L3D_PT_colour_band, L3D_PT_colour_export)


def _fill_palettes(*_):
    """Make sure every scene has its swatches (new scenes start with an empty collection)."""
    for sc in bpy.data.scenes:
        p = sc.l3d_colour
        while len(p.palette) < p.count:
            it = p.palette.add(); it.color = DEFAULT_COLOURS[(len(p.palette) - 1) % len(DEFAULT_COLOURS)]
    return None


@bpy.app.handlers.persistent
def _on_load(*_):
    _fill_palettes()


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.l3d_colour = PointerProperty(type=L3D_ColourProps)
    bpy.app.timers.register(_fill_palettes, first_interval=0.1)
    bpy.app.handlers.load_post.append(_on_load)


def unregister():
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    del bpy.types.Scene.l3d_colour
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
