"""Put colours on a generated mesh so it can be split into AMS filament colours later.

Two sources:
  * photo projection: the cut-out input photo(s) are projected onto the mesh along their view direction,
    with a depth test so hidden surfaces don't pick up colour; unseen vertices take the nearest seen colour.
  * per-point colours from the model itself (TRELLIS gaussians), transferred by nearest neighbour.

Meshes here are glTF-oriented (Y up). A view "direction" is the unit vector from the object towards the
camera; the camera looks back along -direction with +Y up in the image.
Colours are handled as sRGB floats 0..1 and written to the GLB as linear, as glTF expects.
"""
import numpy as np

UP = np.array([0.0, 1.0, 0.0])
# candidate horizontal camera directions, in the order they're tried
DIRS = {'+z': np.array([0, 0, 1.0]), '-z': np.array([0, 0, -1.0]),
        '+x': np.array([1.0, 0, 0]), '-x': np.array([-1.0, 0, 0])}
# which direction each view sits at, relative to the front direction (rotation about +Y, degrees)
VIEW_TURN = {'front': 0, 'left': 90, 'back': 180, 'right': -90}  # measured on Hunyuan3D-2mv / TRELLIS


def _rot_y(v, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])


def _frame(d):
    right = np.cross(UP, d)
    return right / np.linalg.norm(right)


def _fit(verts, d, alpha):
    """Map mesh vertices to image pixels for camera direction d, matching the mesh's projected bbox to the
    image's opaque bbox. Returns (px, py, depth) float arrays."""
    r = _frame(d)
    u = verts @ r
    v = -(verts @ UP)
    ys, xs = np.nonzero(alpha > 0.5)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    su = (x1 - x0) / max(np.ptp(u), 1e-9)
    sv = (y1 - y0) / max(np.ptp(v), 1e-9)
    s = (su + sv) / 2
    px = (u - (u.min() + u.max()) / 2) * s + (x0 + x1) / 2
    py = (v - (v.min() + v.max()) / 2) * s + (y0 + y1) / 2
    return px, py, verts @ d


def _silhouette_iou(verts, d, alpha, res=160):
    h, w = alpha.shape
    k = res / max(h, w)
    px, py, _ = _fit(verts, d, alpha)
    a = alpha[:: max(1, int(1 / k)), :: max(1, int(1 / k))] > 0.5
    hh, ww = a.shape
    xi = np.clip((px / w * ww).astype(int), 0, ww - 1)
    yi = np.clip((py / h * hh).astype(int), 0, hh - 1)
    m = np.zeros_like(a)
    m[yi, xi] = True
    from scipy.ndimage import binary_closing, binary_fill_holes
    m = binary_fill_holes(binary_closing(m, iterations=2))
    return (m & a).sum() / max((m | a).sum(), 1)


def find_front(verts, alpha, prefer=None):
    """Pick the camera direction whose silhouette best matches the front photo.
    Front and back silhouettes mirror each other, so a known model convention (prefer) wins near-ties."""
    scores = {k: _silhouette_iou(verts, d, alpha) for k, d in DIRS.items()}
    best = max(scores, key=scores.get)
    if prefer and scores[prefer] >= scores[best] - 0.05:
        best = prefer
    return best, scores


def project(mesh, views, front_dir, res=768):
    """views: {name: RGBA PIL image}. Returns (colors Nx3 sRGB 0..1, seen mask)."""
    from scipy.ndimage import maximum_filter
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    size = np.ptp(verts, axis=0).max()
    acc = np.zeros((len(verts), 3))
    wsum = np.zeros(len(verts))
    samples = []
    for name, img in views.items():
        d = _rot_y(DIRS[front_dir], VIEW_TURN.get(name, 0))
        im = img.convert('RGBA')
        k = res / max(im.size)
        if k < 1:
            im = im.resize((round(im.width * k), round(im.height * k)))
        rgba = np.asarray(im, dtype=np.float64) / 255.0
        alpha = rgba[..., 3]
        h, w = alpha.shape
        px, py, depth = _fit(verts, d, alpha)
        xi = np.clip(np.round(px).astype(int), 0, w - 1)
        yi = np.clip(np.round(py).astype(int), 0, h - 1)
        zbuf = np.full((h, w), -np.inf)
        np.maximum.at(zbuf, (yi, xi), depth)
        zbuf = maximum_filter(zbuf, size=3)  # close gaps between splatted vertices
        visible = depth >= zbuf[yi, xi] - 0.01 * size
        facing = np.clip(normals @ d, 0, 1) ** 2
        opaque = alpha[yi, xi] > 0.5
        wgt = facing * visible * opaque
        acc += rgba[yi, xi, :3] * wgt[:, None]
        wsum += wgt
        samples.append((rgba, alpha, xi, yi, d))
    seen = wsum > 1e-4
    colors = np.zeros((len(verts), 3))
    colors[seen] = acc[seen] / wsum[seen, None]
    # Surfaces no photo sees (e.g. the back, with a single photo): take the colour at the same spot in the
    # photo as if looking through the object. Right for symmetric things like pots, better than smearing.
    if not seen.all():
        acc2 = np.zeros((len(verts), 3))
        wsum2 = np.zeros(len(verts))
        for rgba, alpha, xi, yi, d in samples:
            wgt = (np.abs(normals @ d) ** 2 + 1e-3) * (alpha[yi, xi] > 0.5)
            acc2 += rgba[yi, xi, :3] * wgt[:, None]
            wsum2 += wgt
        through = ~seen & (wsum2 > 0)
        colors[through] = acc2[through] / wsum2[through, None]
        seen = seen | through
    return fill_unseen(verts, colors, seen), seen


def fill_unseen(verts, colors, seen):
    if seen.all() or not seen.any():
        return colors
    from scipy.spatial import cKDTree
    _, idx = cKDTree(verts[seen]).query(verts[~seen], k=4)
    colors = colors.copy()
    colors[~seen] = colors[seen][idx].mean(axis=1)
    return colors


def transfer(verts, points, point_colors, weights=None, k=8):
    """Nearest-neighbour colour transfer from coloured points (e.g. gaussians) to mesh vertices."""
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(points).query(verts, k=k)
    w = 1.0 / (dist + 1e-6)
    if weights is not None:
        w = w * weights[idx]
    return (point_colors[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)


def srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def attach(mesh, colors_srgb):
    """Store colours on a trimesh as linear RGBA bytes (glTF COLOR_0 is linear)."""
    lin = srgb_to_linear(colors_srgb)
    rgba = np.concatenate([np.round(lin * 255), np.full((len(lin), 1), 255)], axis=1).astype(np.uint8)
    mesh.visual.vertex_colors = rgba
    return mesh
