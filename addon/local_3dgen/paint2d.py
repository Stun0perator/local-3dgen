"""Photo-based colour regions for AMS printing, the way Meshy's "cartoon" multi-colour mode works:

1. flatten the cut-out photo(s) into N clean colour regions in 2D (k-means, then a mode filter and
   small-blob removal, so gradients and shading don't turn into speckle),
2. project those region labels onto the mesh along each photo's view direction, with a depth test,
   using see-through projection for sides no photo shows,
3. subdivide only the triangles on colour borders and re-sample them, so borders follow the artwork
   instead of zig-zagging along big triangles.

Coordinates: the mesh as imported into Blender (Z up). All generators put the front photo's camera on -Y
(glTF +Z); left = +X, back = +Y, right = -X (measured on Hunyuan3D-2mv and TRELLIS outputs).
"""
import glob
import json
import os

import bmesh
import bpy
import numpy as np

from . import ams

UP = np.array([0.0, 0.0, 1.0])
VIEW_DIRS = {'front': np.array([0, -1.0, 0]), 'left': np.array([1.0, 0, 0]),
             'back': np.array([0, 1.0, 0]), 'right': np.array([-1.0, 0, 0])}


# ---------------------------------------------------------------- inputs

def source_images(ob):
    """{view: path} of the cut-out photos the engine saved next to the model, if any."""
    raw = ob.get("l3d_inputs")
    if raw:
        views = json.loads(raw)
        return {k: v for k, v in views.items() if os.path.isfile(v)}
    glb = ob.get("l3d_glb", "")
    if glb:
        base = os.path.splitext(glb)[0]
        found = {os.path.splitext(p)[0].rsplit('_input_', 1)[1]: p for p in glob.glob(base + '_input_*.png')}
        if not found and os.path.isfile(base + '_input.png'):
            found = {'front': base + '_input.png'}
        return {k: v for k, v in found.items() if k in VIEW_DIRS}
    return {}


def load_rgba(path, max_side=640):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        px = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(img)
    px = px.reshape(h, w, 4)[::-1]  # Blender stores rows bottom-up
    step = int(np.ceil(max(w, h) / max_side))
    return np.ascontiguousarray(px[::step, ::step])


# ---------------------------------------------------------------- 2D segmentation

def _box_sum(a, r):
    """Sum over a (2r+1)^2 window, per channel, via an integral image. a: [H, W, C]."""
    p = np.pad(a, ((r + 1, r), (r + 1, r), (0, 0)))
    c = p.cumsum(0).cumsum(1)
    return c[2 * r + 1:, 2 * r + 1:] - c[:-2 * r - 1, 2 * r + 1:] - c[2 * r + 1:, :-2 * r - 1] + c[:-2 * r - 1, :-2 * r - 1]


def mode_filter(labels, k, r=2, passes=2):
    """Replace each opaque pixel's label with the most common label around it (labels < 0 = empty)."""
    for _ in range(passes):
        onehot = np.zeros(labels.shape + (k,), dtype=np.float32)
        m = labels >= 0
        onehot[m, labels[m]] = 1.0
        votes = _box_sum(onehot, r)
        votes[m, labels[m]] += 0.5  # tie -> keep
        labels = np.where(m, votes.argmax(-1), -1)
    return labels


def remove_small_blobs(labels, k, min_px):
    """Blobs smaller than min_px take the label they border most."""
    h, w = labels.shape
    idx = np.arange(h * w).reshape(h, w)
    pairs = [(idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])]
    a = np.concatenate([p[0].ravel() for p in pairs])
    b = np.concatenate([p[1].ravel() for p in pairs])
    flat = labels.ravel()
    ok = (flat[a] >= 0) & (flat[b] >= 0)
    a, b = a[ok], b[ok]
    ones = np.ones(h * w)
    out = ams.merge_small_regions(np.where(flat >= 0, flat, 0), a, b, ones, k, min_px)
    return np.where(flat >= 0, out, -1).reshape(h, w)


def palette_from_images(images, k):
    px = np.concatenate([im[im[..., 3] > 0.5][:, :3] for im in images])
    rng = np.random.default_rng(0)
    sample = px[rng.choice(len(px), size=min(len(px), 60000), replace=False)]
    _, lab = ams.kmeans(ams.features(sample), k, iters=30)
    cols, shares = [], []
    for j in range(k):
        m = lab == j
        cols.append(ams.lit_color(sample[m]) if m.any() else np.array([0.5, 0.5, 0.5]))
        shares.append(m.mean())
    order = np.argsort(shares)[::-1]
    return np.array(cols)[order], np.array(shares)[order]


def segment(img, palette, blob_frac=0.0015):
    """Label map for one cut-out photo: nearest palette colour, then cleaned up."""
    k = len(palette)
    alpha = img[..., 3] > 0.5
    labels = np.full(alpha.shape, -1)
    feats = ams.features(img[alpha][:, :3])
    pal = ams.features(np.asarray(palette))
    labels[alpha] = np.argmin(((feats[:, None, :] - pal[None]) ** 2).sum(-1), axis=1)
    labels = mode_filter(labels, k)
    labels = remove_small_blobs(labels, k, max(4, blob_frac * alpha.sum()))
    return labels


# ---------------------------------------------------------------- camera

class Pose:
    """Camera looking at the mesh from direction d (unit, object -> camera), tilted up by `el`.
    dist = camera distance in mesh sizes (0 = orthographic)."""

    def __init__(self, az, el, dist, center, size):
        az, el = np.radians(az), np.radians(el)
        self.d = np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
        up = UP - (UP @ self.d) * self.d
        self.up = up / np.linalg.norm(up)
        self.right = np.cross(self.up, self.d)
        self.dist, self.center, self.size = dist, center, size

    def project(self, p):
        q = p - self.center
        u, v = q @ self.right, -(q @ self.up)
        if self.dist:
            z = self.dist * self.size - q @ self.d  # distance from the camera along the view axis
            k = (self.dist * self.size) / np.maximum(z, 1e-6)
            u, v = u * k, v * k
        return u, v


def _fit(points, verts, pose, alpha):
    """Pixel coords of `points`: the mesh's projected bounding box (from `verts`) is matched to the
    photo's opaque bounding box."""
    ys, xs = np.nonzero(alpha)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    vu, vv = pose.project(verts)
    s = ((x1 - x0) / max(np.ptp(vu), 1e-9) + (y1 - y0) / max(np.ptp(vv), 1e-9)) / 2
    cu, cv = (vu.min() + vu.max()) / 2, (vv.min() + vv.max()) / 2
    pu, pv = pose.project(points)
    return (pu - cu) * s + (x0 + x1) / 2, (pv - cv) * s + (y0 + y1) / 2


def _iou(verts, pose, alpha_small):
    h, w = alpha_small.shape
    px, py = _fit(verts, verts, pose, alpha_small)
    xi = np.clip(np.round(px).astype(int), 0, w - 1)
    yi = np.clip(np.round(py).astype(int), 0, h - 1)
    m = np.zeros((h, w), dtype=bool)
    m[yi, xi] = True
    # close the splatted points into a solid silhouette (dilate 1px)
    m = np.max([np.pad(m, 1)[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], axis=0)
    return (m & alpha_small).sum() / max((m | alpha_small).sum(), 1)


def match_camera(verts, alpha, view='front'):
    """Search the camera angle, tilt and perspective whose silhouette of the mesh best matches the photo."""
    center = (verts.min(0) + verts.max(0)) / 2
    size = np.ptp(verts, axis=0).max()
    step = max(1, int(np.ceil(max(alpha.shape) / 128)))
    small = alpha[::step, ::step]
    rng = np.random.default_rng(0)
    sub = verts[rng.choice(len(verts), size=min(len(verts), 60000), replace=False)]
    base = {'front': 0, 'left': 90, 'back': 180, 'right': -90}[view]

    def score(az, el, dist):
        return _iou(sub, Pose(base + az, el, dist, center, size), small)

    best = max(((score(az, el, 0), az, el, 0) for az in range(-60, 61, 10) for el in range(-10, 61, 10)))
    _, az0, el0, _ = best
    for dist in (0, 4.0, 2.5, 1.8):  # orthographic, then a few lens angles
        for az in np.arange(az0 - 8, az0 + 8.1, 2):
            for el in np.arange(el0 - 8, el0 + 8.1, 2):
                sc = score(az, el, dist)
                if sc > best[0]:
                    best = (sc, az, el, dist)
    sc, az, el, dist = best
    return Pose(base + az, el, dist, center, size), dict(iou=round(float(sc), 3), azimuth=float(az),
                                                          tilt=float(el), dist=float(dist))


def mesh_arrays(me):
    nv, nf = len(me.vertices), len(me.polygons)
    verts = np.empty(nv * 3); me.vertices.foreach_get('co', verts); verts = verts.reshape(-1, 3)
    cen = np.empty(nf * 3); me.polygons.foreach_get('center', cen); cen = cen.reshape(-1, 3)
    nor = np.empty(nf * 3); me.polygons.foreach_get('normal', nor); nor = nor.reshape(-1, 3)
    starts = np.empty(nf, dtype=np.int64); me.polygons.foreach_get('loop_start', starts)
    lv = np.empty(len(me.loops), dtype=np.int64); me.loops.foreach_get('vertex_index', lv)
    corners = np.stack([lv[starts], lv[starts + 1], lv[starts + 2]], 1)  # first 3 corners per face
    return verts, cen, nor, corners


def face_labels(me, views, k, poses):
    """views: {name: (rgba image, label map)}, poses: {name: Pose} -> label per face."""
    verts, cen, nor, corners = mesh_arrays(me)
    size = np.ptp(verts, axis=0).max()
    votes = np.zeros((len(cen), k))
    through = np.zeros((len(cen), k))
    for name, (img, lab) in views.items():
        pose = poses[name]
        d = pose.d
        alpha = img[..., 3] > 0.5
        h, w = alpha.shape
        # depth buffer from vertices, gaps closed with a 3x3 max
        vx, vy = _fit(verts, verts, pose, alpha)
        xi = np.clip(np.round(vx).astype(int), 0, w - 1)
        yi = np.clip(np.round(vy).astype(int), 0, h - 1)
        zbuf = np.full((h + 2, w + 2), -np.inf)
        np.maximum.at(zbuf, (yi + 1, xi + 1), verts @ d)
        zbuf = np.max([zbuf[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], axis=0)
        cx, cy = _fit(cen, verts, pose, alpha)
        cxi = np.clip(np.round(cx).astype(int), 0, w - 1)
        cyi = np.clip(np.round(cy).astype(int), 0, h - 1)
        visible = (cen @ d) >= zbuf[cyi, cxi] - 0.012 * size
        facing = np.clip(nor @ d, 0, 1) ** 2
        # sample the label map at the centre and the three corners of each face
        pts = [(cxi, cyi)]
        for c in range(3):
            px, py = _fit(verts[corners[:, c]], verts, pose, alpha)
            pts.append((np.clip(np.round(px).astype(int), 0, w - 1), np.clip(np.round(py).astype(int), 0, h - 1)))
        for j, (sx, sy) in enumerate(pts):
            l = lab[sy, sx]
            ok = l >= 0
            wt = (2.0 if j == 0 else 1.0)
            rows = np.nonzero(ok)[0]
            np.add.at(votes, (rows, l[ok]), (facing * visible)[ok] * wt)
            np.add.at(through, (rows, l[ok]), (np.abs(nor @ d) ** 2 + 1e-3)[ok] * wt)
    seen = votes.sum(1) > 1e-6
    labels = np.where(seen, votes.argmax(1), through.argmax(1))
    unknown = ~seen & (through.sum(1) <= 0)
    return labels, unknown


def fill_unknown(labels, unknown, a, b, rounds=50):
    """Faces no photo reaches take their neighbours' labels."""
    labels = labels.copy()
    for _ in range(rounds):
        if not unknown.any():
            break
        src = np.concatenate([a, b]); dst = np.concatenate([b, a])
        m = unknown[dst] & ~unknown[src]
        labels[dst[m]] = labels[src[m]]
        unknown = unknown.copy()
        unknown[dst[m]] = False
    return labels


def refine_borders(ob, labels, a, b, max_new=400000):
    """Split the triangles along colour borders once (each into four), so borders can follow the photo."""
    me = ob.data
    border = np.zeros(len(me.polygons), dtype=bool)
    diff = labels[a] != labels[b]
    border[a[diff]] = True
    border[b[diff]] = True
    if border.sum() * 3 > max_new:
        return False
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    edges = {e for i in np.nonzero(border)[0] for e in bm.faces[i].edges}
    res = bmesh.ops.subdivide_edges(bm, edges=list(edges), cuts=1, use_grid_fill=True)
    touched = {f for e in res['geom_split'] if isinstance(e, bmesh.types.BMEdge) for f in e.link_faces}
    bmesh.ops.triangulate(bm, faces=[f for f in touched if len(f.verts) > 3])
    bm.to_mesh(me)
    bm.free()
    me.update()
    return True


def assign_from_photos(context, ob, palette, refine=1, smooth=1, min_area=0.0):
    """Full photo pipeline; returns per-face labels (and leaves the mesh refined)."""
    srcs = source_images(ob)
    imgs = {v: load_rgba(p) for v, p in srcs.items()}
    k = len(palette)
    views = {v: (im, segment(im, palette)) for v, im in imgs.items()}
    me = ob.data
    verts = mesh_arrays(me)[0]
    poses, report = {}, {}
    for v, (im, _) in views.items():
        poses[v], report[v] = match_camera(verts, im[..., 3] > 0.5, v)
    ob["l3d_cameras"] = str(report)
    for level in range(refine + 1):
        labels, unknown = face_labels(me, views, k, poses)
        a, b = ams.face_adjacency(me)
        labels = fill_unknown(labels, unknown, a, b)
        if level < refine and not refine_borders(ob, labels, a, b):
            break
    if smooth:
        labels = ams.majority_smooth(labels, a, b, k, smooth)
    if min_area > 0:
        mm, mw = ams.world_scale_mm(context, ob)
        scale = np.cbrt(abs(np.linalg.det(np.array(mw.to_3x3())))) * mm
        area = np.empty(len(me.polygons)); me.polygons.foreach_get('area', area)
        labels = ams.merge_small_regions(labels, a, b, area * scale ** 2, k, min_area)
    return labels, {v: lab for v, (_, lab) in views.items()}
