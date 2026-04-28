"""
Surface mesh (triangulation) of a waverider geometry for pressure-force integration.

Public API
----------
panelize_geometry(geom)            -> lower_mesh, upper_mesh
panelization_wetted_area(lo, up)   -> float  (m^2)
panelization_volume(lo, up)        -> float  (m^3, approximate)
plot_panelization(lo, up, ...)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample_curve(pts, n):
    """Resample a (M, 3) curve to n evenly-spaced points by arc length."""
    pts = np.asarray(pts, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumlen[-1]
    if total < 1e-30:
        return np.tile(pts[0], (n, 1))
    t = np.linspace(0.0, total, n)
    out = np.empty((n, 3))
    for j in range(3):
        out[:, j] = np.interp(t, cumlen, pts[:, j])
    return out


def _strip_triangles(s1, s2):
    """Triangulate the quad strip between two n-point curves.

    Each quad (s1[i], s1[i+1], s2[i+1], s2[i]) is split into two triangles:
        T1 = (s1[i], s1[i+1], s2[i+1])
        T2 = (s1[i], s2[i+1], s2[i])

    Returns
    -------
    tris   : (2*(n-1), 3, 3)  vertex coordinates
    norms  : (2*(n-1), 3)     raw (non-oriented) unit normals
    areas  : (2*(n-1),)       triangle areas
    """
    n = s1.shape[0]
    tris  = np.empty((2 * (n - 1), 3, 3))
    norms = np.empty((2 * (n - 1), 3))
    areas = np.empty(2 * (n - 1))
    k = 0
    for i in range(n - 1):
        A, B, C, D = s1[i], s1[i + 1], s2[i + 1], s2[i]
        for v0, v1, v2 in ((A, B, C), (A, C, D)):
            e1, e2 = v1 - v0, v2 - v0
            cross = np.cross(e1, e2)
            area  = 0.5 * np.linalg.norm(cross)
            tris[k]  = (v0, v1, v2)
            norms[k] = cross / (2.0 * area) if area > 1e-30 else cross
            areas[k] = area
            k += 1
    return tris, norms, areas


def _orient_outward(tris, norms, areas, interior_pt):
    """Flip any normals that point toward interior_pt rather than away from it."""
    centroids = tris.mean(axis=1)          # (N, 3)
    out_vec   = centroids - interior_pt    # points from interior toward face
    dot = (out_vec * norms).sum(axis=1)
    norms = norms.copy()
    norms[dot < 0] *= -1
    return norms


def _mesh_half_surface(curves, n_pts):
    """Build triangulated strips for an ordered list of resampled curves."""
    all_t, all_n, all_a = [], [], []
    for i in range(len(curves) - 1):
        t, n, a = _strip_triangles(curves[i], curves[i + 1])
        all_t.append(t)
        all_n.append(n)
        all_a.append(a)
    return np.vstack(all_t), np.vstack(all_n), np.concatenate(all_a)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def panelize_geometry(geom):
    """Build lower and upper surface triangle meshes from a waverider geometry dict.

    The mesh for each surface is a structured triangulation derived directly from
    the N_l / N_up streamlines stored in *geom*.  No additional resolution
    parameters are needed.

    Parameters
    ----------
    geom : dict
        Returned by ``design_waverider``.  Must contain keys
        ``"lower_surface"``, ``"upper_surface"``, and ``"parameters"``.

    Returns
    -------
    lower_mesh, upper_mesh : dict, each with keys

        ``'triangles'`` : ndarray (N_tri, 3, 3)
            Vertex coordinates ``[triangle_index, vertex_index, xyz]``.
        ``'normals'``   : ndarray (N_tri, 3)
            Outward unit normals (pointing away from vehicle interior).
        ``'areas'``     : ndarray (N_tri,)
            Triangle areas in the same units as geometry coordinates.
        ``'centroids'`` : ndarray (N_tri, 3)
            Triangle centroids, convenient for per-panel pressure evaluation.

    Notes
    -----
    For pressure-force integration use::

        F = sum_i  p_i * mesh['areas'][i] * mesh['normals'][i]

    where ``p_i`` is the gauge pressure evaluated at ``mesh['centroids'][i]``.
    """
    params = geom["parameters"]
    N_l    = params["N_l"]
    N_up   = params["N_up"]
    L      = params["L"]

    # ── Lower surface ────────────────────────────────────────────────────────
    # Streamlines are ordered: index 0 ≈ symmetry plane, index N_l-1 = outer edge.
    pos_lo = [_resample_curve(ls["curve"],          N_l) for ls in geom["lower_surface"]]
    neg_lo = [_resample_curve(ls["mirrored_curve"], N_l) for ls in geom["lower_surface"]]

    # Positive-y and negative-y halves meshed independently
    lt, ln, la = _mesh_half_surface(pos_lo, N_l)
    mt, mn, ma = _mesh_half_surface(neg_lo, N_l)

    lower_tris  = np.vstack([lt, mt])
    lower_norms = np.vstack([ln, mn])
    lower_areas = np.concatenate([la, ma])

    # ── Upper surface ────────────────────────────────────────────────────────
    # Each entry has a 1-D x-array and constant (y, z); mirror in y for both halves.
    def _upper_pts(us, sign_y):
        x = np.asarray(us["x"])
        y = np.full_like(x, sign_y * float(us["y"]))
        z = np.full_like(x, float(us["z"]))
        return _resample_curve(np.column_stack([x, y, z]), N_up)

    pos_up = [_upper_pts(us,  1.0) for us in geom["upper_surface"]]
    neg_up = [_upper_pts(us, -1.0) for us in geom["upper_surface"]]

    ut, un, ua = _mesh_half_surface(pos_up, N_up)
    vt, vn, va = _mesh_half_surface(neg_up, N_up)

    upper_tris  = np.vstack([ut, vt])
    upper_norms = np.vstack([un, vn])
    upper_areas = np.concatenate([ua, va])

    # ── Orient normals outward ────────────────────────────────────────────────
    # Interior reference point: centroid of all surface vertices, slightly offset
    # toward the interior (less negative z, since the vehicle sits at z < 0).
    lo_cent = lower_tris.mean(axis=1)
    up_cent = upper_tris.mean(axis=1)
    interior_pt = 0.5 * (lo_cent.mean(axis=0) + up_cent.mean(axis=0))

    lower_norms = _orient_outward(lower_tris, lower_norms, lower_areas, interior_pt)
    upper_norms = _orient_outward(upper_tris, upper_norms, upper_areas, interior_pt)

    def _build(tris, norms, areas):
        return {
            "triangles": tris,
            "normals":   norms,
            "areas":     areas,
            "centroids": tris.mean(axis=1),
        }

    return _build(lower_tris, lower_norms, lower_areas), \
           _build(upper_tris, upper_norms, upper_areas)


def panelization_wetted_area(lower_mesh, upper_mesh):
    """Return total wetted area (lower + upper surfaces)."""
    return float(lower_mesh["areas"].sum() + upper_mesh["areas"].sum())


def panelization_volume(lower_mesh, upper_mesh):
    """Volume enclosed between lower and upper surfaces.

    For each triangle, project a prism up to the z=0 plane.  The prism volume
    is A_xy × (|z1|+|z2|+|z3|)/3, which is exact for a linearly-varying wedge.
    Subtracting upper from lower gives the vehicle thickness integral.
    """
    def _proj_vol(mesh):
        tris = mesh["triangles"]             # (N, 3, 3)
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        cross_z = (v1[:, 0]-v0[:, 0])*(v2[:, 1]-v0[:, 1]) \
                - (v1[:, 1]-v0[:, 1])*(v2[:, 0]-v0[:, 0])
        A_xy  = np.abs(cross_z) / 2.0
        z_avg = np.abs(tris[:, :, 2]).sum(axis=1) / 3.0
        return float((A_xy * z_avg).sum())

    return _proj_vol(lower_mesh) - _proj_vol(upper_mesh)


def plot_panelization(lower_mesh, upper_mesh,
                      alpha=0.6, save_path=None, show=True):
    """3-D matplotlib plot of the lower and upper surface meshes.

    Parameters
    ----------
    lower_mesh, upper_mesh : dicts from ``panelize_geometry``
    alpha      : face transparency (0–1)
    save_path  : file path to save figure; ``None`` skips saving
    show       : call ``plt.show()`` if True
    """
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    def _add_mesh(mesh, facecolor, edgecolor, label):
        verts = [tri for tri in mesh["triangles"]]   # list of (3,3)
        poly  = Poly3DCollection(verts, alpha=alpha,
                                 facecolor=facecolor,
                                 edgecolor=edgecolor,
                                 linewidth=0.2)
        ax.add_collection3d(poly)
        # Invisible scatter for legend proxy
        c = mesh["centroids"]
        ax.scatter([], [], [], color=facecolor, label=label, s=20)

    _add_mesh(lower_mesh, facecolor="steelblue",  edgecolor="navy",   label="Lower surface")
    _add_mesh(upper_mesh, facecolor="coral",      edgecolor="firebrick", label="Upper surface")

    # Axis limits from all vertices
    all_verts = np.vstack([lower_mesh["triangles"].reshape(-1, 3),
                           upper_mesh["triangles"].reshape(-1, 3)])
    for i, lbl in enumerate(("x", "y", "z")):
        lo, hi = all_verts[:, i].min(), all_verts[:, i].max()
        pad = 0.05 * (hi - lo) if (hi - lo) > 0 else 1.0
        getattr(ax, f"set_{lbl}lim")(lo - pad, hi + pad)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Waverider surface mesh")
    ax.legend(loc="best")
    ax.set_box_aspect([1, 1, 1])
    ax.set_aspect('equal')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
