"""AMS / multi-filament colour assignment and 3MF export.

Takes a mesh with vertex colours (every model the engine generates has them), reduces the colours to N
filament colours, cleans up specks so every region is printable, shows the result as materials (repaint by
hand in Edit Mode with the material slots if you like), and exports a 3MF for Bambu Studio / OrcaSlicer /
PrusaSlicer: one solid object with per-triangle colour painting, the slicers' own multi-colour format.
(Splitting into one part per colour isn't offered: that yields open surface patches, not printable solids.)
Pure numpy: no extra dependencies inside Blender.
"""
import os
import zipfile

import bpy
import numpy as np
from bpy.props import (CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)

MAT_PREFIX = "AMS "


# ---------------------------------------------------------------- colour helpers

def srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb_to_lab(c):
    lin = srgb_to_linear(c)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


L_WEIGHT = 0.45  # generated colours carry baked-in shading: compare hue/chroma more than lightness


def features(srgb):
    lab = srgb_to_lab(srgb)
    lab[:, 0] *= L_WEIGHT
    return lab


def lit_color(srgb):
    """A region's filament colour: the brighter half of its faces (shadowed faces pull the mean down),
    excluding the brightest 10% (specular highlights)."""
    L = srgb_to_lab(srgb)[:, 0]
    lo, hi = np.percentile(L, [50, 90])
    m = (L >= lo) & (L <= hi)
    return srgb[m].mean(0) if m.any() else srgb.mean(0)


def kmeans(x, k, weights=None, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    w = np.ones(len(x)) if weights is None else weights
    centers = [x[rng.choice(len(x), p=w / w.sum())]]
    for _ in range(1, k):  # k-means++ seeding
        d = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1) * w
        centers.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
    c = np.array(centers)
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = (x[m] * w[m, None]).sum(0) / w[m].sum()
    lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
    return c, lab


# ---------------------------------------------------------------- mesh data

def face_colors(me):
    """Per-face sRGB colour (average of the face's corners/points) from the mesh's colour attribute."""
    attr = me.color_attributes.active_color or (me.color_attributes[0] if len(me.color_attributes) else None)
    if attr is None:
        return None
    n = len(attr.data)
    buf = np.empty(n * 4, dtype=np.float32)
    prop = 'color_srgb' if hasattr(attr.data[0], 'color_srgb') else 'color'
    attr.data.foreach_get(prop, buf)
    col = buf.reshape(-1, 4)[:, :3]
    if prop == 'color':  # linear -> sRGB
        col = np.where(col <= 0.0031308, col * 12.92, 1.055 * np.power(np.clip(col, 0, None), 1 / 2.4) - 0.055)
    nf = len(me.polygons)
    starts = np.empty(nf, dtype=np.int64); me.polygons.foreach_get('loop_start', starts)
    totals = np.empty(nf, dtype=np.int64); me.polygons.foreach_get('loop_total', totals)
    if attr.domain == 'POINT':
        lv = np.empty(len(me.loops), dtype=np.int64); me.loops.foreach_get('vertex_index', lv)
        col = col[lv]
    return np.add.reduceat(col, starts) / totals[:, None]


def face_adjacency(me):
    """Pairs of faces sharing an edge (manifold edges only)."""
    nf = len(me.polygons)
    totals = np.empty(nf, dtype=np.int64); me.polygons.foreach_get('loop_total', totals)
    loop_face = np.repeat(np.arange(nf), totals)
    loop_edge = np.empty(len(me.loops), dtype=np.int64); me.loops.foreach_get('edge_index', loop_edge)
    order = np.argsort(loop_edge, kind='stable')
    e, f = loop_edge[order], loop_face[order]
    same = e[1:] == e[:-1]
    a, b = f[:-1][same], f[1:][same]
    # drop edges shared by more than 2 faces
    cnt = np.bincount(e, minlength=len(me.edges))
    ok = cnt[e[:-1][same]] == 2
    return a[ok], b[ok]


def components(n, a, b):
    """Connected components over edges (a, b): returns a root id per node (pointer-jumping union-find)."""
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


def majority_smooth(labels, a, b, k, passes):
    for _ in range(passes):
        votes = np.zeros((len(labels), k), dtype=np.float32)
        np.add.at(votes, (a, labels[b]), 1.0)
        np.add.at(votes, (b, labels[a]), 1.0)
        votes[np.arange(len(labels)), labels] += 1.5  # a face keeps its colour unless clearly outvoted
        labels = votes.argmax(1)
    return labels


def merge_small_regions(labels, a, b, area, k, min_area, rounds=4):
    """Regions smaller than min_area (mm²) take the colour they share the longest border with."""
    for _ in range(rounds):
        same = labels[a] == labels[b]
        comp = components(len(labels), a[same], b[same])
        comp_area = np.bincount(comp, weights=area, minlength=len(labels))
        small = comp_area[comp] < min_area
        if not small.any():
            break
        diff = ~same
        ca, cb = comp[a[diff]], comp[b[diff]]
        la, lb = labels[a[diff]], labels[b[diff]]
        # votes: small component -> neighbour label
        src = np.concatenate([ca, cb]); lab = np.concatenate([lb, la])
        keep = small[np.concatenate([a[diff], b[diff]])]
        src, lab = src[keep], lab[keep]
        if len(src) == 0:
            break
        key = src * k + lab
        cnt = np.bincount(key)
        nz = np.nonzero(cnt)[0]
        best = {}
        for kk in nz[np.argsort(cnt[nz])]:  # ascending, so the biggest count wins
            best[kk // k] = kk % k
        roots = np.fromiter(best.keys(), dtype=np.int64)
        new = np.full(len(labels), -1)
        new[roots] = np.fromiter(best.values(), dtype=np.int64)
        target = new[comp]
        m = small & (target >= 0)
        labels = np.where(m, target, labels)
    return labels


def _vertex_neighbours(me):
    """CSR-style neighbour lists: (order, starts, counts) so that neighbours of v are order[starts[v]:+counts[v]]."""
    nv = len(me.vertices)
    ev = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get('vertices', ev)
    ev = ev.reshape(-1, 2)
    src = np.concatenate([ev[:, 0], ev[:, 1]]); dst = np.concatenate([ev[:, 1], ev[:, 0]])
    o = np.argsort(src, kind='stable')
    counts = np.bincount(src, minlength=nv)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return dst[o], starts, counts


def _face_corners(me):
    nf = len(me.polygons)
    starts = np.empty(nf, dtype=np.int64); me.polygons.foreach_get('loop_start', starts)
    totals = np.empty(nf, dtype=np.int64); me.polygons.foreach_get('loop_total', totals)
    lv = np.empty(len(me.loops), dtype=np.int64); me.loops.foreach_get('vertex_index', lv)
    return starts, totals, lv


def smooth_borders(context, ob, labels, k, radius_mm=1.5, levels=2):
    """Turn per-face labels into smooth, curved colour borders:
    1. each colour becomes a soft field on the vertices, blurred over ~radius_mm of surface (a diffusion),
       which removes the triangle staircase and thin slivers;
    2. triangles the border crosses are subdivided (the fields interpolate onto the new vertices), `levels`
       times, so the border can run through the middle of big triangles;
    3. each face takes the colour with the strongest field."""
    import bmesh
    me = ob.data
    mm, mw = world_scale_mm(context, ob)
    scale = np.cbrt(abs(np.linalg.det(np.array(mw.to_3x3())))) * mm
    nv = len(me.vertices)
    starts, totals, lv = _face_corners(me)
    area = np.empty(len(me.polygons)); me.polygons.foreach_get('area', area)
    # face one-hot -> vertex fields (area weighted)
    F = np.zeros((nv, k)); W = np.zeros(nv)
    fl = np.repeat(labels, totals); fa = np.repeat(area, totals)
    np.add.at(F, (lv, fl), fa); np.add.at(W, lv, fa)
    F /= np.maximum(W, 1e-12)[:, None]
    # diffusion over edges, iterations from the wanted radius in mm
    co = np.empty(nv * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    order, vstart, cnt = _vertex_neighbours(me)
    ev = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get('vertices', ev); ev = ev.reshape(-1, 2)
    edge_mm = np.linalg.norm(co[ev[:, 0]] - co[ev[:, 1]], axis=1).mean() * scale
    iters = int(np.clip((radius_mm / max(edge_mm, 1e-6)) ** 2, 1, 300))
    has = cnt > 0
    for _ in range(iters):
        nb = np.add.reduceat(F[order], vstart[has], axis=0)
        avg = F.copy()
        avg[has] = nb / cnt[has, None]
        F = 0.5 * F + 0.5 * avg
    # store fields as vertex attributes so subdivision interpolates them
    names = [f"_l3d_f{j}" for j in range(k)]
    for j, nm in enumerate(names):
        at = me.attributes.get(nm) or me.attributes.new(nm, 'FLOAT', 'POINT')
        at.data.foreach_set('value', F[:, j].astype(np.float32))
    for _ in range(levels):
        vlab = F.argmax(1)
        starts, totals, lv = _face_corners(me)
        fl_min = np.minimum.reduceat(vlab[lv], starts); fl_max = np.maximum.reduceat(vlab[lv], starts)
        border = np.nonzero(fl_min != fl_max)[0]
        if len(border) == 0 or len(border) > 600000:
            break
        bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
        edges = {e for i in border for e in bm.faces[i].edges}
        res = bmesh.ops.subdivide_edges(bm, edges=list(edges), cuts=1, use_grid_fill=True)
        touched = {f for e in res['geom_split'] if isinstance(e, bmesh.types.BMEdge) for f in e.link_faces}
        bmesh.ops.triangulate(bm, faces=[f for f in touched if len(f.verts) > 3])
        bm.to_mesh(me); bm.free(); me.update()
        F = np.stack([_read_float(me, nm) for nm in names], 1)
    starts, totals, lv = _face_corners(me)
    face_f = np.add.reduceat(F[lv], starts, axis=0)
    for nm in names:
        me.attributes.remove(me.attributes[nm])
    return face_f.argmax(1)


def _read_float(me, name):
    at = me.attributes[name]
    buf = np.empty(len(at.data), dtype=np.float32); at.data.foreach_get('value', buf)
    return buf


def _labels(me):
    lab = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get('material_index', lab)
    return lab


def _write_labels(context, ob, labels):
    me = ob.data
    me.polygons.foreach_set('material_index', labels.astype(np.int32))
    me.update()
    p = context.scene.l3d_ams
    area = np.empty(len(me.polygons)); me.polygons.foreach_get('area', area)
    shares = np.bincount(labels, weights=area, minlength=len(p.palette)) / area.sum()
    for it, sh in zip(p.palette, shares):
        it.share = float(sh)


def _face_geometry(context, ob):
    """Face centres (world, mm) and unit normals (world)."""
    me = ob.data
    nf = len(me.polygons)
    cen = np.empty(nf * 3); me.polygons.foreach_get('center', cen); cen = cen.reshape(-1, 3)
    nor = np.empty(nf * 3); me.polygons.foreach_get('normal', nor); nor = nor.reshape(-1, 3)
    M = np.array(ob.matrix_world)
    mm, _ = world_scale_mm(context, ob)
    cen_w = (cen @ M[:3, :3].T + M[:3, 3]) * mm
    nor_w = nor @ np.linalg.inv(M[:3, :3])  # normals transform by the inverse-transpose
    nor_w /= np.maximum(np.linalg.norm(nor_w, axis=1, keepdims=True), 1e-12)
    return cen_w, nor_w


def fix_rims(context, ob, labels, rim, through, depth_mm):
    """Faces of colour `through` inside a cutout (within depth_mm of the `rim` colour along the surface, and
    not facing the same way as the rim surface they are reached from) get the rim colour: the walls of a
    hole belong to the part the hole is cut into, not to what shows through it."""
    me = ob.data
    a, b = face_adjacency(me)
    cen, nor = _face_geometry(context, ob)
    step = np.linalg.norm(cen[a] - cen[b], axis=1)
    dist = np.where(labels == rim, 0.0, np.inf)
    srcn = nor.copy()
    ok = (labels == rim) | (labels == through)
    for _ in range(2000):
        changed = False
        for u, v in ((a, b), (b, a)):
            cand = dist[u] + step
            m = (cand < dist[v]) & ok[v] & (cand <= depth_mm)
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            idx = idx[np.argsort(-cand[idx])]  # smallest distance is written last, so it wins
            dist[v[idx]] = cand[idx]
            srcn[v[idx]] = srcn[u[idx]]
            changed = True
        if not changed:
            break
    facing_same = (nor * srcn).sum(1) > 0.5  # e.g. the liner's outside, seen through the hole
    sel = (labels == through) & np.isfinite(dist) & ~facing_same
    out = labels.copy()
    out[sel] = rim
    return out, int(sel.sum())


def level_border(context, ob, below, above, height_mm, band_mm, levels=2):
    """Snap the border between the bottom region of colour `below` (e.g. a base band: the `below` area that
    touches the bottom of the model) and colour `above` to a horizontal line, Z in mm from the model's bottom.
    Other areas of the `below` colour (e.g. black stars higher up) are left alone. Border triangles are
    subdivided `levels` times so the line comes out crisp."""
    import bmesh
    me = ob.data
    cen, _ = _face_geometry(context, ob)
    z0 = cen[:, 2].min()

    def base_region(labels, z):
        a, b = face_adjacency(me)
        same = (labels[a] == below) & (labels[b] == below)
        comp = components(len(labels), a[same], b[same])
        bottom = np.unique(comp[(labels == below) & (z < 1.5)])
        return np.isin(comp, bottom) & (labels == below), a, b

    labels = _labels(me)
    z = cen[:, 2] - z0
    base, a, b = base_region(labels, z)
    if not base.any():
        raise ValueError("No area of the 'Below' colour touches the bottom of the model")
    if height_mm <= 0:  # detect it from the base region's border with the `above` colour
        m = (base[a] & (labels[b] == above)) | (base[b] & (labels[a] == above))
        if not m.any():
            raise ValueError("The bottom region doesn't touch the 'Above' colour")
        height_mm = float(np.median(np.concatenate([z[a[m]], z[b[m]]])))
    for level in range(levels + 1):
        cen, _ = _face_geometry(context, ob)
        labels = _labels(me)
        z = cen[:, 2] - z0
        base, a, b = base_region(labels, z)
        band = np.abs(z - height_mm) <= band_mm
        to_below = band & (labels == above) & (z < height_mm)
        to_above = band & base & (z >= height_mm)
        labels[to_below] = below
        labels[to_above] = above
        sel = band & ((labels == below) | (labels == above))
        me.polygons.foreach_set('material_index', labels.astype(np.int32))
        me.update()
        if level == levels:
            break
        a, b = face_adjacency(me)
        diff = labels[a] != labels[b]
        on_line = np.zeros(len(labels), dtype=bool)
        on_line[a[diff]] = True
        on_line[b[diff]] = True
        on_line &= sel
        if not on_line.any():
            break
        bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
        edges = {e for i in np.nonzero(on_line)[0] for e in bm.faces[i].edges}
        res = bmesh.ops.subdivide_edges(bm, edges=list(edges), cuts=1, use_grid_fill=True)
        touched = {f for e in res['geom_split'] if isinstance(e, bmesh.types.BMEdge) for f in e.link_faces}
        bmesh.ops.triangulate(bm, faces=[f for f in touched if len(f.verts) > 3])
        bm.to_mesh(me); bm.free(); me.update()
    return height_mm


def mirror_labels(me, labels, axis):
    """Copy the labels of one half onto the other (for symmetric objects: the photographed half is the
    reliable one). axis 'front_back': front half (-Y, where the photo camera is) -> back half.
    'left_right': the -X half -> +X half."""
    from mathutils.kdtree import KDTree
    nf = len(me.polygons)
    cen = np.empty(nf * 3); me.polygons.foreach_get('center', cen); cen = cen.reshape(-1, 3)
    ax = 1 if axis == 'front_back' else 0
    mid = (cen[:, ax].min() + cen[:, ax].max()) / 2
    src = np.nonzero(cen[:, ax] <= mid)[0]
    dst = np.nonzero(cen[:, ax] > mid)[0]
    tree = KDTree(len(src))
    for i, j in enumerate(src):
        tree.insert(cen[j], i)
    tree.balance()
    out = labels.copy()
    m = cen[dst].copy()
    m[:, ax] = 2 * mid - m[:, ax]
    for j, q in zip(dst, m):
        out[j] = labels[src[tree.find(q)[1]]]
    return out


def world_scale_mm(context, ob):
    unit = context.scene.unit_settings.scale_length or 1.0
    return unit * 1000.0, ob.matrix_world


# ---------------------------------------------------------------- properties

class L3D_AMSColor(bpy.types.PropertyGroup):
    color: FloatVectorProperty(name="Colour", subtype='COLOR_GAMMA', size=3, min=0, max=1,
                               default=(0.8, 0.8, 0.8))
    share: FloatProperty(default=0.0)


class L3D_AMSProps(bpy.types.PropertyGroup):
    count: IntProperty(name="Colours", default=4, min=1, max=16,
                       description="How many filaments (AMS slots) to use")
    palette: CollectionProperty(type=L3D_AMSColor)
    target_faces: IntProperty(name="Simplify to faces", default=200000, min=0, soft_max=1000000,
                              description="Reduce the mesh before assigning colours (0 = keep). Slicers get "
                                          "slow with millions of painted triangles")
    refine: IntProperty(name="Border detail", default=1, min=0, max=2,
                        description="Split triangles along colour borders this many times so borders follow "
                                    "the photo (each level up to 4x the border triangles)")
    min_area: FloatProperty(name="Smallest patch (mm²)", default=1.0, min=0.0, soft_max=50.0, precision=2,
                            step=1, description="Colour patches smaller than this merge into their neighbour. "
                                                "About 0.16 mm² (one 0.4 mm line) is the smallest dot that prints")
    printer: EnumProperty(name="Printer", default='X1C', items=[
        ('X1C', "X1 Carbon", ""), ('X1E', "X1E", ""), ('P1S', "P1S", ""), ('P1P', "P1P", ""),
        ('A1', "A1", ""), ('A1M', "A1 mini", "")],
        description="Written into the 3MF so Bambu Studio opens it with this printer and PLA per colour")
    regions: EnumProperty(name="Regions from", default='auto', items=[
        ('auto', "Auto", "The model's own colours when it made them (TRELLIS), else the photos"),
        ('model', "Model colours", "The mesh's vertex colours: line up with the geometry, carry some shading"),
        ('photo', "Photos", "Flatten the photos into regions and project them (Hunyuan models)")])
    mirror: EnumProperty(name="Mirror", default='none', items=[
        ('none', "No mirroring", "Colour every side from the model / photos"),
        ('front_back', "Front to back", "Symmetric object: copy the front's colours onto the back (the "
                                        "back is guessed and often darker)"),
        ('left_right', "Left to right", "Symmetric object: copy one side's colours onto the other")])
    smooth: IntProperty(name="Smooth borders", default=3, min=0, max=15,
                        description="Pre-clean passes (neighbour vote) before the border smoothing")
    border_mm: FloatProperty(name="Border smoothing (mm)", default=1.2, min=0.0, soft_max=6.0,
                             description="How far colour borders are smoothed along the surface: removes the "
                                         "triangle staircase and hair-thin slivers (0 = off)")
    rim_slot: IntProperty(name="Rim colour", default=2, min=1, max=16,
                          description="Slot of the part the holes are cut into (e.g. the white shell)")
    through_slot: IntProperty(name="Seen-through colour", default=1, min=1, max=16,
                              description="Slot of what shows through the holes (e.g. the red liner)")
    rim_depth: FloatProperty(name="Rim depth (mm)", default=6.0, min=0.5, soft_max=30.0,
                             description="How far into a hole the rim colour reaches (about the wall thickness)")
    level_below: IntProperty(name="Below", default=3, min=1, max=16, description="Slot under the border")
    level_above: IntProperty(name="Above", default=2, min=1, max=16, description="Slot above the border")
    level_height: FloatProperty(name="Height (mm)", default=0.0, min=0.0, soft_max=500.0,
                                description="Border height from the bottom of the model (0 = detect it from the "
                                            "current border)")
    level_band: FloatProperty(name="Band (mm)", default=6.0, min=0.5, soft_max=40.0,
                              description="Only faces this close to the border height are changed")
    border_levels: IntProperty(name="Border detail", default=2, min=0, max=3,
                               description="Subdivide triangles along colour borders this many times so the "
                                           "smoothed border can run through them")


# ---------------------------------------------------------------- operators

def _active_mesh(context):
    ob = context.active_object
    return ob if ob is not None and ob.type == 'MESH' else None


class L3D_OT_ams_pick(bpy.types.Operator):
    """Find the main colours of the selected model and fill the palette with them"""
    bl_idname = "l3d.ams_pick"
    bl_label = "Pick colours from model"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = _active_mesh(context)
        if ob is None:
            self.report({'ERROR'}, "Select a generated model first"); return {'CANCELLED'}
        p = context.scene.l3d_ams
        from . import paint2d
        srcs = paint2d.source_images(ob)
        if srcs:  # palette from the photo(s): far cleaner than the mesh's shaded colours
            cols, shares = paint2d.palette_from_images([paint2d.load_rgba(x) for x in srcs.values()], p.count)
            p.palette.clear()
            for c, s in zip(cols, shares):
                it = p.palette.add(); it.color = tuple(float(x) for x in c); it.share = float(s)
            return {'FINISHED'}
        fc = face_colors(ob.data)
        if fc is None:
            self.report({'ERROR'}, "This model has no colours or source photos")
            return {'CANCELLED'}
        area = np.empty(len(ob.data.polygons)); ob.data.polygons.foreach_get('area', area)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(fc), size=min(len(fc), 40000), replace=False, p=area / area.sum())
        centers, lab = kmeans(features(fc[idx]), p.count, iters=30)
        # report the palette as real sRGB colours (lit part of each cluster), most-used first
        cols, shares = [], []
        for j in range(p.count):
            m = lab == j
            cols.append(lit_color(fc[idx][m]) if m.any() else np.array([0.5, 0.5, 0.5]))
            shares.append(m.mean())
        order = np.argsort(shares)[::-1]
        p.palette.clear()
        for j in order:
            it = p.palette.add(); it.color = tuple(float(x) for x in cols[j]); it.share = float(shares[j])
        return {'FINISHED'}


class L3D_OT_ams_rims(bpy.types.Operator):
    """Give the walls of cutouts the colour of the part they are cut into, instead of the colour that shows
    through the hole"""
    bl_idname = "l3d.ams_rims"
    bl_label = "Fix cutout rims"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = _active_mesh(context)
        p = context.scene.l3d_ams
        if ob is None or not any(MAT_PREFIX in (m.name if m else "") for m in ob.data.materials):
            self.report({'ERROR'}, "Assign colours first"); return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        labels, n = fix_rims(context, ob, _labels(ob.data), p.rim_slot - 1, p.through_slot - 1, p.rim_depth)
        _write_labels(context, ob, labels)
        self.report({'INFO'}, f"{n:,} rim triangles set to slot {p.rim_slot}")
        return {'FINISHED'}


class L3D_OT_ams_level(bpy.types.Operator):
    """Snap the border between two colours to a straight horizontal line (bands on pots, bases)"""
    bl_idname = "l3d.ams_level"
    bl_label = "Level border"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = _active_mesh(context)
        p = context.scene.l3d_ams
        if ob is None or not any(MAT_PREFIX in (m.name if m else "") for m in ob.data.materials):
            self.report({'ERROR'}, "Assign colours first"); return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            h = level_border(context, ob, p.level_below - 1, p.level_above - 1, p.level_height, p.level_band)
        except ValueError as e:
            self.report({'ERROR'}, str(e)); return {'CANCELLED'}
        _write_labels(context, ob, _labels(ob.data))
        p.level_height = round(h, 2)
        self.report({'INFO'}, f"Border levelled at {h:.1f} mm")
        return {'FINISHED'}


class L3D_OT_ams_paint(bpy.types.Operator):
    """Repaint this model's colours with Hunyuan3D-Paint: removes lighting and shadows from the photo and paints
    the model from several angles, so colour regions come out clean (runs in the background, ~1-2 min)"""
    bl_idname = "l3d.ams_paint"
    bl_label = "Clean colours (Hunyuan3D-Paint)"

    def execute(self, context):
        import json
        from . import Engine, _ensure_timer, _prefs, paint2d
        ob = _active_mesh(context)
        glb = ob.get("l3d_glb") if ob else None
        views = paint2d.source_images(ob) if ob else {}
        if not glb or not os.path.isfile(glb) or not views:
            self.report({'ERROR'}, "Select a model made with the Image to 3D panel (needs its GLB and photos)")
            return {'CANCELLED'}
        unit = context.scene.unit_settings.scale_length or 1.0
        size_bu = max(ob.dimensions)
        name = ob.name
        ctx = dict(scene=context.scene.name, cursor=tuple(ob.location), size_bu=size_bu, smooth=True)
        if Engine.current is None and not Engine.pending:
            Engine.batch_total = Engine.batch_done = 0
        Engine.last_error = ""
        Engine.pending.append(dict(
            backend='hunyuan_paint', name=name, seed=int(ob.get("l3d_seed", 0)), slot=0, ctx=ctx,
            obname=name + "_painted", replaces=name,
            req=dict(cmd="paint", id=name + "-paint", mesh=glb, views=views, remove_bg=False)))
        Engine.batch_total += 1
        _ensure_timer()
        return {'FINISHED'}


class L3D_OT_ams_assign(bpy.types.Operator):
    """Snap every triangle to the nearest palette colour, clean up specks and show the result as materials.
    Re-running it replaces any hand repainting"""
    bl_idname = "l3d.ams_assign"
    bl_label = "Assign colours"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = _active_mesh(context)
        p = context.scene.l3d_ams
        if ob is None:
            self.report({'ERROR'}, "Select a generated model first"); return {'CANCELLED'}
        if not len(p.palette):
            bpy.ops.l3d.ams_pick()
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        me = ob.data
        if p.target_faces and len(me.polygons) > p.target_faces * 1.05:
            md = ob.modifiers.new("l3d_simplify", 'DECIMATE')
            md.ratio = p.target_faces / len(me.polygons)
            with context.temp_override(object=ob, active_object=ob):
                bpy.ops.object.modifier_apply(modifier=md.name)
            me = ob.data
        k = len(p.palette)
        pal = np.array([it.color[:] for it in p.palette])
        from . import paint2d
        use_photos = bool(paint2d.source_images(ob)) and (
            p.regions == 'photo' or (p.regions == 'auto' and ob.get("l3d_color_source") not in ('model', 'paint')))
        if use_photos:
            # Meshy-style: flatten the photo(s) into regions, project them, refine the borders
            labels, _ = paint2d.assign_from_photos(context, ob, pal, refine=0 if p.border_mm > 0 else p.refine,
                                                    smooth=p.smooth, min_area=p.min_area)
            me = ob.data
            if p.mirror != 'none':
                labels = mirror_labels(me, labels, p.mirror)
        else:  # no photos: use the mesh's own vertex colours
            fc = face_colors(me)
            if fc is None:
                self.report({'ERROR'}, "This model has no colours or source photos"); return {'CANCELLED'}
            labels = np.argmin(((features(fc)[:, None, :] - features(pal)[None]) ** 2).sum(-1), axis=1)
            a, b = face_adjacency(me)
            labels = majority_smooth(labels, a, b, k, p.smooth)
            if p.mirror != 'none':
                labels = mirror_labels(me, labels, p.mirror)
            if p.min_area > 0:
                mm, mw = world_scale_mm(context, ob)
                scale = np.cbrt(abs(np.linalg.det(np.array(mw.to_3x3())))) * mm
                ar = np.empty(len(me.polygons)); me.polygons.foreach_get('area', ar)
                labels = merge_small_regions(labels, a, b, ar * scale ** 2, k, p.min_area)
                labels = majority_smooth(labels, a, b, k, 1)
        if p.border_mm > 0:
            labels = smooth_borders(context, ob, labels, k, p.border_mm, p.border_levels)
            me = ob.data
        mm, mw = world_scale_mm(context, ob)
        scale = np.cbrt(abs(np.linalg.det(np.array(mw.to_3x3())))) * mm
        area = np.empty(len(me.polygons)); me.polygons.foreach_get('area', area)
        area *= scale ** 2

        me.materials.clear()
        for i, it in enumerate(p.palette):
            name = f"{MAT_PREFIX}{i + 1}"
            mat = bpy.data.materials.get(f"{name} ({ob.name})") or bpy.data.materials.new(f"{name} ({ob.name})")
            lin = tuple(float(x) for x in srgb_to_linear(np.array(it.color[:]))) + (1.0,)
            mat.diffuse_color = lin
            if mat.use_nodes:
                bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if bsdf:
                    bsdf.inputs['Base Color'].default_value = lin
            mat["l3d_ams_slot"] = i + 1
            me.materials.append(mat)
        me.polygons.foreach_set('material_index', labels.astype(np.int32))
        me.update()
        shares = np.bincount(labels, weights=area, minlength=k) / area.sum()
        for it, s in zip(p.palette, shares):
            it.share = float(s)
        for area3d in context.screen.areas:
            if area3d.type == 'VIEW_3D':
                sh = area3d.spaces.active.shading
                if sh.type == 'SOLID':
                    sh.color_type = 'MATERIAL'
        self.report({'INFO'}, f"{len(me.polygons):,} triangles in {k} colours")
        return {'FINISHED'}


def _paint_code(state):
    """PrusaSlicer/Bambu TriangleSelector code for a whole (unsplit) triangle painted with extruder `state`."""
    if state < 3:
        return format(state << 2, 'X')
    n, nibbles = state - 3, [0b1100]
    while n >= 15:
        nibbles.append(0b1111); n -= 15
    nibbles.append(n)
    return ''.join(format(x, 'X') for x in reversed(nibbles))


def _triangles(ob, depsgraph, mm):
    ev = ob.evaluated_get(depsgraph)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    nv = len(me.vertices)
    co = np.empty(nv * 3); me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    M = np.array(ob.matrix_world)
    co = (co @ M[:3, :3].T + M[:3, 3]) * mm
    nt = len(me.loop_triangles)
    tri = np.empty(nt * 3, dtype=np.int64); me.loop_triangles.foreach_get('vertices', tri)
    mat = np.empty(nt, dtype=np.int64); me.loop_triangles.foreach_get('material_index', mat)
    ev.to_mesh_clear()
    return co, tri.reshape(-1, 3), mat


# printer: (Bambu printer name, preset suffix, bed centre mm)
PRINTERS = {
    'X1C': ("Bambu Lab X1 Carbon", "X1C", 128), 'X1E': ("Bambu Lab X1E", "X1E", 128),
    'P1S': ("Bambu Lab P1S", "P1S", 128), 'P1P': ("Bambu Lab P1P", "P1P", 128),
    'A1': ("Bambu Lab A1", "A1", 128), 'A1M': ("Bambu Lab A1 mini", "A1M", 90),
}


def _bambu_package(name, co, tri, paint, colors, printer):
    """Files of a Bambu Studio project 3MF, laid out like Bambu's own (and Meshy's) multi-colour exports:
    the mesh in 3D/Objects/object_1.model with paint_color per triangle, and a project config naming the
    printer and one filament per colour, so the slicer shows the right colours instead of its defaults."""
    import json
    import uuid
    model_name, suffix, centre = PRINTERS.get(printer, PRINTERS['X1C'])
    n = len(colors)
    u = lambda: str(uuid.uuid4())
    head = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
            'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
            'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
            'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">\n')
    obj = [head, ' <metadata name="BambuStudio:3mfVersion">2</metadata>\n <resources>\n',
           f'  <object id="1" p:UUID="{u()}" type="model">\n   <mesh>\n    <vertices>\n']
    obj += [f'     <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>\n' for x, y, z in co]
    obj.append('    </vertices>\n    <triangles>\n')
    obj += [f'     <triangle v1="{a}" v2="{b}" v3="{c}" paint_color="{pc}"/>\n' for (a, b, c), pc in zip(tri, paint)]
    obj.append('    </triangles>\n   </mesh>\n  </object>\n </resources>\n <build/>\n</model>\n')
    main = (head + ' <metadata name="Application">BambuStudio-02.03.01.00</metadata>\n'
            ' <metadata name="BambuStudio:3mfVersion">2</metadata>\n'
            ' <metadata name="BambuStudio:MmPaintingVersion">1</metadata>\n'
            f' <metadata name="Title">{name}</metadata>\n <resources>\n'
            f'  <object id="2" p:UUID="{u()}" type="model">\n   <components>\n'
            f'    <component p:path="/3D/Objects/object_1.model" objectid="1" p:UUID="{u()}" '
            'transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n   </components>\n  </object>\n </resources>\n'
            f' <build p:UUID="{u()}">\n  <item objectid="2" p:UUID="{u()}" '
            f'transform="1 0 0 0 1 0 0 0 1 {centre} {centre} 0" printable="1"/>\n </build>\n</model>\n')
    settings = (f'<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <object id="2">\n'
                f'    <metadata key="name" value="{name}"/>\n    <metadata key="extruder" value="1"/>\n'
                f'    <metadata face_count="{len(tri)}"/>\n    <part id="1" subtype="normal_part">\n'
                f'      <metadata key="name" value="{name}"/>\n'
                '      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
                f'      <mesh_stat face_count="{len(tri)}" edges_fixed="0" degenerate_facets="0" facets_removed="0" '
                'facets_reversed="0" backwards_edges="0"/>\n    </part>\n  </object>\n  <plate>\n'
                '    <metadata key="plater_id" value="1"/>\n    <metadata key="plater_name" value=""/>\n'
                '    <metadata key="locked" value="false"/>\n    <model_instance>\n'
                '      <metadata key="object_id" value="2"/>\n      <metadata key="instance_id" value="0"/>\n'
                '      <metadata key="identify_id" value="1"/>\n    </model_instance>\n  </plate>\n'
                '  <assemble>\n  </assemble>\n</config>\n')
    hexes = ["#" + "".join(format(round(max(0, min(1, c)) * 255), '02X') for c in col) + "FF" for col in colors]
    per = lambda v: [v] * n
    project = {
        "filament_colour": hexes,
        "filament_settings_id": per(f"Bambu PLA Basic @BBL {suffix}"),
        "filament_type": per("PLA"),
        "filament_minimal_purge_on_wipe_tower": per("15"),
        "filament_flow_ratio": per("0.98"),
        "filament_diameter": per("1.75"),
        "wipe": per("1"), "wipe_distance": per("2"),
        "retract_length_toolchange": per("2"), "retract_restart_extra_toolchange": per("0"),
        "standby_temperature_delta": "-5",
        "enable_prime_tower": "1" if n > 1 else "0",
        "prime_tower_width": "20", "prime_tower_brim_width": "3", "prime_volume": "20",
        "flush_into_infill": "0", "flush_into_objects": "0", "flush_into_support": "1",
        "flush_multiplier": "1",
        "flush_volumes_matrix": ["0" if i == j else "280" for i in range(n) for j in range(n)],
        "flush_volumes_vector": ["140"] * (2 * n),
        "wipe_tower_x": ["15"], "wipe_tower_y": ["145" if centre > 100 else "100"],
        "wipe_tower_rotation_angle": "0", "wipe_tower_no_sparse_layers": "0", "wipe_speed": "80%",
        "single_extruder_multi_material": "1", "print_sequence": "by layer",
        "printer_model": model_name,
        "printer_settings_id": f"{model_name} 0.4 nozzle",
        "nozzle_diameter": ["0.4"],
        "print_settings_id": f"0.20mm Standard @BBL {suffix}",
        "layer_change_gcode": "; layer change\nG92 E0\n",
        "from": "project", "name": "project_settings", "version": "02.03.01.00",
    }
    ct = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
          'content-types">\n <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
          'relationships+xml"/>\n <Default Extension="model" ContentType="application/vnd.ms-package.'
          '3dmanufacturing-3dmodel+xml"/>\n</Types>')
    rel = lambda target: ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxml'
                          f'formats.org/package/2006/relationships">\n <Relationship Target="{target}" Id="rel-1" '
                          'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>')
    return [('[Content_Types].xml', ct), ('_rels/.rels', rel('/3D/3dmodel.model')),
            ('3D/_rels/3dmodel.model.rels', rel('/3D/Objects/object_1.model')),
            ('3D/3dmodel.model', main), ('3D/Objects/object_1.model', ''.join(obj)),
            ('Metadata/model_settings.config', settings),
            ('Metadata/project_settings.config', json.dumps(project, indent=4))]


class L3D_OT_ams_export(bpy.types.Operator):
    """Export the selected model as a 3MF with its AMS colours"""
    bl_idname = "l3d.ams_export"
    bl_label = "Export 3MF"

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.3mf", options={'HIDDEN'})

    def invoke(self, context, event):
        ob = _active_mesh(context)
        if ob is None:
            self.report({'ERROR'}, "Select a model first"); return {'CANCELLED'}
        if not self.filepath:
            self.filepath = bpy.path.clean_name(ob.name) + ".3mf"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        ob = _active_mesh(context)
        if ob is None:
            self.report({'ERROR'}, "Select a model first"); return {'CANCELLED'}
        if not any(MAT_PREFIX in (m.name if m else "") for m in ob.data.materials):
            self.report({'ERROR'}, "Assign colours first"); return {'CANCELLED'}
        path = bpy.path.ensure_ext(bpy.path.abspath(self.filepath), ".3mf")
        mm, _ = world_scale_mm(context, ob)
        co, tri, mat = _triangles(ob, context.evaluated_depsgraph_get(), mm)
        slot = np.array([int(m.get("l3d_ams_slot", i + 1)) if m else i + 1
                         for i, m in enumerate(ob.data.materials)] or [1])
        ext = slot[np.clip(mat, 0, len(slot) - 1)]  # extruder / AMS slot per triangle, 1-based
        # sit on the bed, centred on a 256 mm plate (the slicer can re-arrange)
        lo, hi = co.min(0), co.max(0)
        co = co - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])

        codes = np.array([_paint_code(s) for s in range(1, 17)])
        name = bpy.path.clean_name(ob.name)
        p = context.scene.l3d_ams
        cols = [it.color[:] for it in p.palette] or [(0.8, 0.8, 0.8)]
        n = max(len(cols), int(ext.max()))
        cols = (cols + [(0.8, 0.8, 0.8)] * n)[:n]
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            for fn, text in _bambu_package(name, co, tri, codes[ext - 1], cols, p.printer):
                z.writestr(fn, text)
        size = hi - lo
        self.report({'INFO'}, f"Saved {os.path.basename(path)}: {len(tri):,} triangles, "
                              f"{size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm")
        return {'FINISHED'}


# ---------------------------------------------------------------- panel

class L3D_PT_ams(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Image to 3D"
    bl_label = "AMS colours"
    bl_parent_id = "L3D_PT_panel"


    def draw(self, context):
        p = context.scene.l3d_ams
        L = self.layout
        ob = _active_mesh(context)
        if ob is None:
            L.label(text="Select a generated model", icon='INFO')
            return
        from . import paint2d
        has_cols = len(ob.data.color_attributes) > 0 or bool(paint2d.source_images(ob))
        if not has_cols:
            L.label(text="No colours on this model", icon='ERROR')
        L.label(text=ob.name, icon='OBJECT_DATA')
        row = L.row(align=True)
        row.prop(p, "count")
        row.operator("l3d.ams_pick", text="", icon='EYEDROPPER')
        col = L.column(align=True)
        for i, it in enumerate(p.palette):
            r = col.row(align=True)
            r.label(text=f"Slot {i + 1}")
            r.prop(it, "color", text="")
            r.label(text=f"{it.share:.0%}" if it.share else "")
        from . import paint2d
        srcs = paint2d.source_images(ob)
        L.label(text=(f"Colours from {len(srcs)} photo{'s' if len(srcs) != 1 else ''} ({', '.join(srcs)})"
                      if srcs else "Colours from the mesh (no source photos found)"), icon='IMAGE_DATA')
        if ob.get("l3d_color_source") != 'paint':
            row = L.row()
            row.operator("l3d.ams_paint", icon='BRUSHES_ALL')
        L.prop(p, "regions")
        L.prop(p, "target_faces")
        L.prop(p, "border_mm")
        L.prop(p, "border_levels")
        L.prop(p, "mirror", text="")
        L.prop(p, "min_area")
        L.prop(p, "smooth")
        row = L.row()
        row.scale_y = 1.3
        row.enabled = has_cols
        row.operator("l3d.ams_assign", icon='BRUSH_DATA')
        box = L.box()
        box.label(text="Fix-ups", icon='MODIFIER')
        r = box.row(align=True)
        r.prop(p, "rim_slot", text="Rim")
        r.prop(p, "through_slot", text="Through")
        r.prop(p, "rim_depth", text="Depth")
        box.operator("l3d.ams_rims", icon='MOD_SOLIDIFY')
        r = box.row(align=True)
        r.prop(p, "level_below")
        r.prop(p, "level_above")
        r = box.row(align=True)
        r.prop(p, "level_height", text="Height (0 = auto)")
        r.prop(p, "level_band", text="Band")
        box.operator("l3d.ams_level", icon='ALIGN_JUSTIFY')
        L.label(text="Touch up: Edit Mode, select faces, Material > Assign", icon='INFO')
        row = L.row()
        row.scale_y = 1.3
        L.prop(p, "printer")
        row.operator("l3d.ams_export", icon='EXPORT')


classes = (L3D_AMSColor, L3D_AMSProps, L3D_OT_ams_rims, L3D_OT_ams_level, L3D_OT_ams_paint, L3D_OT_ams_pick, L3D_OT_ams_assign, L3D_OT_ams_export, L3D_PT_ams)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.l3d_ams = PointerProperty(type=L3D_AMSProps)


def unregister():
    del bpy.types.Scene.l3d_ams
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
