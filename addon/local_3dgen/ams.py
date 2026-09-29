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
from bpy.props import (CollectionProperty, FloatProperty, FloatVectorProperty,
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
    min_area: FloatProperty(name="Smallest patch (mm²)", default=12.0, min=0.0, soft_max=200.0,
                            description="Colour patches smaller than this merge into their neighbour")
    smooth: IntProperty(name="Smooth borders", default=3, min=0, max=15,
                        description="Passes that straighten jagged colour borders")


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
        fc = face_colors(ob.data)
        if fc is None:
            self.report({'ERROR'}, "This model has no colours (generate it again with this version)")
            return {'CANCELLED'}
        area = np.empty(len(ob.data.polygons)); ob.data.polygons.foreach_get('area', area)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(fc), size=min(len(fc), 40000), replace=False, p=area / area.sum())
        p = context.scene.l3d_ams
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
        fc = face_colors(me)
        if fc is None:
            self.report({'ERROR'}, "This model has no colours"); return {'CANCELLED'}
        k = len(p.palette)
        pal = np.array([it.color[:] for it in p.palette])
        labels = np.argmin(((features(fc)[:, None, :] - features(pal)[None]) ** 2).sum(-1), axis=1)

        a, b = face_adjacency(me)
        mm, mw = world_scale_mm(context, ob)
        scale = np.cbrt(abs(np.linalg.det(np.array(mw.to_3x3())))) * mm
        area = np.empty(len(me.polygons)); me.polygons.foreach_get('area', area)
        area *= scale ** 2
        labels = majority_smooth(labels, a, b, k, p.smooth)
        if p.min_area > 0:
            labels = merge_small_regions(labels, a, b, area, k, p.min_area)
            labels = majority_smooth(labels, a, b, k, 1)

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


def _mesh_xml(obj_id, co, tri, paint=None):
    out = [f'  <object id="{obj_id}" type="model">\n   <mesh>\n    <vertices>\n']
    out += [f'     <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>\n' for x, y, z in co]
    out.append('    </vertices>\n    <triangles>\n')
    if paint is None:
        out += [f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>\n' for a, b, c in tri]
    else:
        out += [f'     <triangle v1="{a}" v2="{b}" v3="{c}" paint_color="{p}" slic3rpe:mmu_segmentation="{p}"/>\n'
                for (a, b, c), p in zip(tri, paint)]
    out.append('    </triangles>\n   </mesh>\n  </object>\n')
    return ''.join(out)


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
        objects = [_mesh_xml(1, co, tri, codes[ext - 1])]
        items = ['  <item objectid="1" transform="1 0 0 0 1 0 0 0 1 128 128 0" printable="1"/>\n']

        model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                 '<model unit="millimeter" xml:lang="en-US" '
                 'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
                 'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">\n'
                 f' <metadata name="Title">{bpy.path.clean_name(ob.name)}</metadata>\n'
                 ' <metadata name="Application">Local 3D Gen (Blender)</metadata>\n'
                 ' <resources>\n' + ''.join(objects) + ' </resources>\n'
                 ' <build>\n' + ''.join(items) + ' </build>\n</model>\n')
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('[Content_Types].xml',
                       '<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/'
                       'package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.'
                       'openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType='
                       '"application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
            z.writestr('_rels/.rels',
                       '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxml'
                       'formats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" '
                       'Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
                       '</Relationships>')
            z.writestr('3D/3dmodel.model', model)
            # colour key next to the model, handy when loading filaments into the AMS
            p = context.scene.l3d_ams
            key = [f"Slot {i + 1}: #{''.join(format(round(c * 255), '02X') for c in it.color)}  "
                   f"({it.share:.0%} of surface)" for i, it in enumerate(p.palette)]
            z.writestr('Metadata/ams_colours.txt', "\n".join(key) + "\n")
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
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        p = context.scene.l3d_ams
        L = self.layout
        ob = _active_mesh(context)
        if ob is None:
            L.label(text="Select a generated model", icon='INFO')
            return
        has_cols = len(ob.data.color_attributes) > 0
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
        L.prop(p, "target_faces")
        L.prop(p, "min_area")
        L.prop(p, "smooth")
        row = L.row()
        row.scale_y = 1.3
        row.enabled = has_cols
        row.operator("l3d.ams_assign", icon='BRUSH_DATA')
        L.label(text="Touch up: Edit Mode, select faces, Material > Assign", icon='INFO')
        row = L.row()
        row.scale_y = 1.3
        row.operator("l3d.ams_export", icon='EXPORT')


classes = (L3D_AMSColor, L3D_AMSProps, L3D_OT_ams_pick, L3D_OT_ams_assign, L3D_OT_ams_export, L3D_PT_ams)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.l3d_ams = PointerProperty(type=L3D_AMSProps)


def unregister():
    del bpy.types.Scene.l3d_ams
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
