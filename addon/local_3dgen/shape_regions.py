"""Shape-aware colour regions: colour changes are made to happen where the shape changes.

The mesh is split into patches with a graph segmentation (Felzenszwalb-Huttenlocher) over the faces. Edge
weights are high across creases and concave seams (where a rib rises off a shell, where a rim turns into a
hole, where a foot meets the base) and across different current colours; small patches merge into their
neighbours. Each patch then takes the colour covering most of its area. Result: borders sit on the geometry,
smudges inside a patch disappear, and there are far fewer colour changes to print.
"""
import numpy as np

from . import ams


def _smoothed_normals(nor, a, b, iters):
    n = nor.copy()
    for _ in range(iters):
        acc = n.copy()
        np.add.at(acc, a, n[b])
        np.add.at(acc, b, n[a])
        n = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    return n


def segment(context, ob, labels, crease=1.0, colour_weight=1.0, k_mm2=8.0, min_mm2=1.0):
    """Returns (patch id per face, number of patches)."""
    me = ob.data
    a, b = ams.face_adjacency(me)
    cen, nor = ams._face_geometry(context, ob)
    mm, mw = ams.world_scale_mm(context, ob)
    scale = np.cbrt(abs(np.linalg.det(np.array(mw.to_3x3())))) * mm
    area = np.empty(len(me.polygons)); me.polygons.foreach_get('area', area)
    area = np.maximum(area * scale ** 2, 1e-6)  # simplification can leave zero-area slivers
    ns = _smoothed_normals(nor, a, b, 2)  # take the scan-like noise out of the normals
    bend = 1.0 - np.clip((ns[a] * ns[b]).sum(1), -1, 1)       # 0 = flat, 2 = folded back
    d = cen[b] - cen[a]
    concave = ((ns[a] * d).sum(1) > 0) & ((ns[b] * -d).sum(1) > 0)
    w = crease * bend * np.where(concave, 3.0, 1.0) * 20.0 + colour_weight * (labels[a] != labels[b])

    # Felzenszwalb-Huttenlocher merge over edges in increasing weight
    order = np.argsort(w, kind='stable')
    parent = np.arange(len(labels))
    size = area.copy()
    internal = np.zeros(len(labels))
    aa, bb, ww = a[order].tolist(), b[order].tolist(), w[order].tolist()
    par = parent.tolist(); sz = size.tolist(); it = internal.tolist()

    def find(x):
        root = x
        while par[root] != root:
            root = par[root]
        while par[x] != root:
            par[x], x = root, par[x]
        return root

    for u, v, wt in zip(aa, bb, ww):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if wt <= min(it[ru] + k_mm2 / sz[ru], it[rv] + k_mm2 / sz[rv]):
            if sz[ru] < sz[rv]:
                ru, rv = rv, ru
            par[rv] = ru
            sz[ru] += sz[rv]
            it[ru] = wt
    # tiny leftovers join the neighbour they share the weakest edge with
    for u, v, wt in zip(aa, bb, ww):
        ru, rv = find(u), find(v)
        if ru != rv and (sz[ru] < min_mm2 or sz[rv] < min_mm2):
            if sz[ru] < sz[rv]:
                ru, rv = rv, ru
            par[rv] = ru
            sz[ru] += sz[rv]
    comp = np.array([find(i) for i in range(len(par))])
    _, comp = np.unique(comp, return_inverse=True)
    return comp, int(comp.max()) + 1


def vote(labels, comp, area, k):
    """Each patch takes the colour covering most of its area."""
    tally = np.zeros((comp.max() + 1, k))
    np.add.at(tally, (comp, labels), area)
    return tally.argmax(1)[comp]


def follow_shape(context, ob, labels, k, crease=1.0, patch_mm2=40.0, min_mm2=1.0):
    me = ob.data
    comp, n = segment(context, ob, labels, crease=crease, k_mm2=patch_mm2, min_mm2=min_mm2)
    mm, mw = ams.world_scale_mm(context, ob)
    scale = np.cbrt(abs(np.linalg.det(np.array(mw.to_3x3())))) * mm
    area = np.empty(len(me.polygons)); me.polygons.foreach_get('area', area)
    return vote(labels, comp, area * scale ** 2, k), n
