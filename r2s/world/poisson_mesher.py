"""
Poisson collision mesher for ``world.collider``, run in the repo env
(``python -m r2s.world.poisson_mesher IN.ply OUT.glb <preset flags>``; needs Open3D,
installed by scripts/setup_env.sh). The code below this docstring is the user's
``ply2glb.py`` (2026-05-17), unchanged; tests/test_world_collider.py pins it.
Note: ``--meshfix`` needs ``pymeshfix``, which the env does not have, so the
``planar`` preset prints "pymeshfix not installed - skip" and exports the
trimesh-repaired mesh -- that is what every production planar collider got, and
the behaviour is kept.

3DGS PLY -> Density Volume -> Signed-Distance Field -> Collision Mesh GLB

Pipeline:
  1. Parse 3DGS PLY (centers, opacity, scale, rotation)
  2. Filter floaters (low opacity, oversized scales)
  3. Splat anisotropic 3D Gaussian densities (opacity * exp(-1/2 dᵀΣ⁻¹d))
     into a uniform voxel grid. Each Gaussian only touches voxels inside
     its 3σ axis-aligned bbox.
  4. Threshold to a binary occupancy mask, convert to a signed-distance
     field via the EDT, then run marching cubes at iso=0.
  5. Clean (largest components) + simplify, fill holes, export GLB.

Notes:
  - This avoids the synthetic-depth-rendering approach entirely; no view-
    dependent holes.
  - The signed-distance step gives smoother, more collision-friendly
    surfaces than running marching cubes directly on raw density.
"""

import argparse
import numpy as np
import open3d as o3d
import trimesh
from plyfile import PlyData
from skimage import measure
from scipy.ndimage import (
    distance_transform_edt, gaussian_filter, binary_closing,
    binary_fill_holes, generate_binary_structure, iterate_structure,
)


# ---------------------------------------------------------------------------
# PLY parsing
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def quat_to_rotmat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def gaussian_axis_normals(scale, rot):
    """
    Derive one surface-normal candidate per Gaussian from its thinnest axis.

    For the 3DGS covariance convention used in this file,
    Σ = R diag(S²) Rᵀ, so the columns of R are the Gaussian's local axes in
    world coordinates. The column paired with the smallest scale is the
    axis most likely to be normal to the represented surface.
    """
    R = quat_to_rotmat(rot)
    smallest_axis = np.argmin(scale, axis=1)
    normals = np.take_along_axis(
        R, smallest_axis[:, None, None], axis=2
    ).squeeze(axis=2)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    return normals.astype(np.float32)


def parse_3dgs_ply(path, opacity_thresh=0.1, max_scale_quantile=0.995):
    ply = PlyData.read(path)
    v = ply.elements[0].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    opacity = sigmoid(np.asarray(v["opacity"], dtype=np.float32))
    scale = np.exp(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    )
    rot = np.stack(
        [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1
    ).astype(np.float32)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9

    keep = opacity > opacity_thresh
    if max_scale_quantile < 1.0:
        smax = scale.max(axis=1)
        cutoff = np.quantile(smax, max_scale_quantile)
        keep &= smax < cutoff

    print(f"  parsed {len(xyz)} gaussians, kept {keep.sum()} "
          f"({100*keep.mean():.1f}%) after filtering")
    return xyz[keep], opacity[keep], scale[keep], rot[keep]


# ---------------------------------------------------------------------------
# Density volume splatting
# ---------------------------------------------------------------------------

def splat_density_grid(xyz, opacity, scale, rot, resolution, padding=0.1, n_sigma=2.5,
                       chunk_size=20_000, verbose=True):
    bb_min = xyz.min(0) - padding
    bb_max = xyz.max(0) + padding
    extent = bb_max - bb_min
    voxel_size = float(extent.max()) / resolution
    res = np.ceil(extent / voxel_size).astype(np.int32)
    if verbose:
        print(f"  grid: {tuple(res)} voxels, voxel_size={voxel_size:.4f}m, "
              f"bbox extent={tuple(np.round(extent,3))}")

    R = quat_to_rotmat(rot)                              # (N,3,3)
    S2 = (scale * scale).astype(np.float32)              # (N,3)
    inv_S2 = (1.0 / S2).astype(np.float32)
    # Σ⁻¹ = R diag(1/S²) Rᵀ
    inv_cov = np.einsum('nij,nj,nkj->nik', R, inv_S2, R).astype(np.float32)  # (N,3,3)

    # Axis-aligned 3σ bbox of rotated ellipsoid:
    # r_dim = n_sigma * sqrt( sum_k (R[dim,k]*S_k)² )
    R_S = R * scale[:, None, :]                           # (N,3,3)
    bbox_r = (n_sigma * np.linalg.norm(R_S, axis=2)).astype(np.float32)  # (N,3)

    centers_grid = (xyz - bb_min) / voxel_size            # voxel coords
    radii_grid = bbox_r / voxel_size

    lo = np.floor(centers_grid - radii_grid).astype(np.int32)
    hi = np.ceil(centers_grid + radii_grid).astype(np.int32) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, res[None, :])

    sizes = hi - lo
    valid = (sizes > 0).all(axis=1)
    if not valid.all() and verbose:
        print(f"  skipping {(~valid).sum()} gaussians outside grid")
    keep_idx = np.where(valid)[0]

    density = np.zeros(tuple(res), dtype=np.float32)
    N = len(keep_idx)
    if verbose:
        print(f"  splatting {N} gaussians in chunks of {chunk_size}...")

    for c0 in range(0, N, chunk_size):
        c1 = min(N, c0 + chunk_size)
        idx = keep_idx[c0:c1]
        for k in idx:
            x0, y0, z0 = lo[k]
            x1, y1, z1 = hi[k]
            # voxel-center world coords inside this block
            xs = (np.arange(x0, x1, dtype=np.float32) + 0.5) * voxel_size + bb_min[0]
            ys = (np.arange(y0, y1, dtype=np.float32) + 0.5) * voxel_size + bb_min[1]
            zs = (np.arange(z0, z1, dtype=np.float32) + 0.5) * voxel_size + bb_min[2]
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
            d0 = gx - xyz[k, 0]
            d1 = gy - xyz[k, 1]
            d2 = gz - xyz[k, 2]
            ic = inv_cov[k]
            # Mahalanobis squared distance
            m = (ic[0, 0] * d0 * d0 + ic[1, 1] * d1 * d1 + ic[2, 2] * d2 * d2
                 + 2 * ic[0, 1] * d0 * d1 + 2 * ic[0, 2] * d0 * d2
                 + 2 * ic[1, 2] * d1 * d2)
            mask = m < (n_sigma * n_sigma)
            if not mask.any():
                continue
            contrib = opacity[k] * np.exp(-0.5 * m)
            contrib *= mask  # zero outside ellipsoid
            density[x0:x1, y0:y1, z0:z1] += contrib
        if verbose and ((c1 % (chunk_size * 5) == 0) or c1 == N):
            print(f"    {c1}/{N}")

    return density, voxel_size, bb_min


# ---------------------------------------------------------------------------
# Density -> SDF -> mesh
# ---------------------------------------------------------------------------

def density_to_sdf(density, iso=None, iso_quantile=0.1,
                   smooth_sigma=0.0, close_radius=0, fill_2d_axes=""):
    nz = density[density > 0]
    print(f"  non-zero voxels: {len(nz)} ({100*len(nz)/density.size:.2f}% of grid), "
          f"max={density.max():.3f}, q10={np.quantile(nz,0.1):.3f}, "
          f"q50={np.quantile(nz,0.5):.3f}, q90={np.quantile(nz,0.9):.3f}")

    if smooth_sigma > 0:
        density = gaussian_filter(density, sigma=smooth_sigma)
        nz2 = density[density > 0]
        print(f"  after smooth(σ={smooth_sigma}): non-zero={len(nz2)} "
              f"({100*len(nz2)/density.size:.2f}%)")
        nz = nz2

    if iso is None:
        iso = float(np.quantile(nz, iso_quantile)) if len(nz) else 0.0
    print(f"  density iso threshold = {iso:.4f}")

    occ = density > iso
    if not occ.any():
        raise RuntimeError("Empty occupancy after thresholding; lower iso or check splatting.")
    print(f"  occupied voxels: {occ.sum()} ({100*occ.mean():.2f}%)")

    if close_radius > 0:
        struct = iterate_structure(generate_binary_structure(3, 1), close_radius)
        occ = binary_closing(occ, structure=struct)
        print(f"  after morphological close(r={close_radius}): "
              f"{occ.sum()} occupied ({100*occ.mean():.2f}%)")

    # Per-slice 2D hole fill — fixes large planar gaps (ceiling/floor) that
    # 3D closing can't reach. Each slice perpendicular to the named axis is
    # treated as a 2D mask; any region enclosed in the slice is filled.
    axis_map = {"x": 0, "y": 1, "z": 2}
    for ax in fill_2d_axes.lower():
        if ax not in axis_map:
            continue
        a = axis_map[ax]
        before = occ.sum()
        occ_swapped = np.moveaxis(occ, a, 0)
        for i in range(occ_swapped.shape[0]):
            occ_swapped[i] = binary_fill_holes(occ_swapped[i])
        occ = np.moveaxis(occ_swapped, 0, a)
        print(f"  after 2D fill along {ax}-axis: {occ.sum()} occupied "
              f"({100*occ.mean():.2f}%, +{occ.sum()-before} voxels)")

    dist_out = distance_transform_edt(~occ)
    dist_in = distance_transform_edt(occ)
    sdf = dist_out - dist_in
    return sdf.astype(np.float32), iso


def marching_cubes_sdf(sdf, voxel_size, bb_min, level=0.0):
    verts, faces, normals, _ = measure.marching_cubes(sdf, level=level)
    # verts are in (i,j,k) voxel index space. Convert to world.
    world = verts * voxel_size + bb_min
    return world.astype(np.float32), faces.astype(np.int32), normals.astype(np.float32)


# ---------------------------------------------------------------------------
# Poisson reconstruction (alternative method)
# ---------------------------------------------------------------------------

def poisson_reconstruct(xyz, opacity, scale, rot,
                        depth=10, normal_k=30, density_quantile=0.05,
                        normal_source="pca"):
    """
    Poisson surface reconstruction directly on Gaussian centers.

    - normals come either from k-NN PCA or each Gaussian's thinnest axis,
      then are oriented consistently
    - opacity has already been applied as an upstream keep/drop filter; it
      is not passed to Open3D Poisson as a continuous reconstruction weight
    - Poisson 'density' field returned by Open3D is used to trim the noisy
      far field (vertices in low-density regions are removed).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    if normal_source == "pca":
        print(f"  estimating normals from point-center PCA (k={normal_k})...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_k)
        )
    elif normal_source == "gaussian-axis":
        print("  deriving normals from each Gaussian's thinnest axis...")
        normals = gaussian_axis_normals(scale, rot)
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    else:
        raise ValueError(f"unknown Poisson normal source: {normal_source}")

    print(f"  orienting normals consistently (k={normal_k})...")
    pcd.orient_normals_consistent_tangent_plane(k=normal_k)

    print(f"  running Poisson reconstruction (depth={depth})...")
    mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=1.1, linear_fit=True
        )
    )
    densities = np.asarray(densities)
    print(f"  raw poisson mesh: {len(mesh.vertices)} verts, "
          f"{len(mesh.triangles)} tris, density "
          f"q05={np.quantile(densities,0.05):.2f} "
          f"q50={np.quantile(densities,0.5):.2f}")

    if density_quantile > 0:
        thresh = np.quantile(densities, density_quantile)
        keep = densities >= thresh
        mesh.remove_vertices_by_mask(~keep)
        print(f"  after density trim (q{density_quantile}): "
              f"{len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    return V, F


# ---------------------------------------------------------------------------
# Mesh cleanup + export
# ---------------------------------------------------------------------------

def cleanup_and_export(verts, faces, out_path, target_faces=80_000,
                       min_cluster_tris=500, z_up_to_y_up=False,
                       taubin_iters=0):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    if taubin_iters > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=taubin_iters)
        mesh.compute_vertex_normals()
        print(f"  after Taubin smooth ({taubin_iters} iters): "
              f"{len(mesh.vertices)} verts")

    print(f"  raw mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # drop tiny components
    tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    cluster_n_tris = np.asarray(cluster_n_tris)
    keep_mask = cluster_n_tris[tri_clusters] >= min_cluster_tris
    mesh.remove_triangles_by_mask(~keep_mask)
    mesh.remove_unreferenced_vertices()
    print(f"  after cluster filter: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    if target_faces and len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)
        print(f"  after simplify:       {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    tm = trimesh.Trimesh(vertices=V, faces=F, process=True)
    trimesh.repair.fill_holes(tm)
    tm.fix_normals()

    if globals().get("USE_MESHFIX", False):
        try:
            import pymeshfix
            print("  running pymeshfix (guaranteed watertight)...")
            mf = pymeshfix.MeshFix(tm.vertices, tm.faces)
            mf.repair(verbose=False)
            tm = trimesh.Trimesh(vertices=mf.v, faces=mf.f, process=True)
            print(f"  after pymeshfix: {len(tm.vertices)} verts, {len(tm.faces)} faces")
        except ImportError:
            print("  pymeshfix not installed — skip. `pip install pymeshfix`")

    if z_up_to_y_up:
        Rm = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        tm.apply_transform(Rm)

    print(f"  final: {len(tm.vertices)} verts, {len(tm.faces)} faces, "
          f"watertight={tm.is_watertight}")
    tm.export(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_ply")
    ap.add_argument("output_glb")
    ap.add_argument("--method", choices=["density", "poisson"], default="density",
                    help="reconstruction method: 'density' (TSDF-like) or 'poisson'")
    ap.add_argument("--opacity-thresh", type=float, default=0.1)
    ap.add_argument("--resolution", type=int, default=256,
                    help="voxel grid resolution along longest axis")
    ap.add_argument("--padding", type=float, default=0.1,
                    help="padding around bbox in world units")
    ap.add_argument("--n-sigma", type=float, default=2.5,
                    help="Gaussian truncation in std-devs")
    ap.add_argument("--iso", type=float, default=None,
                    help="density threshold; default: --iso-quantile of non-zero density")
    ap.add_argument("--iso-quantile", type=float, default=0.1,
                    help="if --iso not given, use this quantile of non-zero density (default 0.1)")
    ap.add_argument("--smooth-sigma", type=float, default=0.0,
                    help="Gaussian blur sigma (in voxels) on density before threshold")
    ap.add_argument("--close-radius", type=int, default=0,
                    help="morphological closing radius (in voxels) on occupancy mask")
    ap.add_argument("--fill-2d-axes", type=str, default="",
                    help="axes for per-slice 2D hole fill (e.g. 'z' for ceiling/floor, "
                         "'xyz' for all). Fills regions enclosed within each slice.")
    ap.add_argument("--mc-level", type=float, default=0.0,
                    help="marching-cubes level on the SDF (in voxel units). "
                         ">0 erodes (thinner walls), <0 dilates (thicker)")
    ap.add_argument("--target-faces", type=int, default=80_000)
    ap.add_argument("--min-cluster", type=int, default=500)
    ap.add_argument("--chunk-size", type=int, default=20_000)
    ap.add_argument("--z-up-to-y-up", action="store_true")
    ap.add_argument("--meshfix", action="store_true",
                    help="run pymeshfix as final pass (guaranteed watertight). "
                         "Requires `pip install pymeshfix`.")
    ap.add_argument("--taubin-iters", type=int, default=0,
                    help="Taubin smoothing iterations (post-MC). Tries 5-15.")
    # poisson-specific
    ap.add_argument("--poisson-depth", type=int, default=10,
                    help="[poisson] octree depth, higher=more detail (8-12)")
    ap.add_argument("--poisson-normal-k", type=int, default=30,
                    help="[poisson] kNN for PCA normal estimation and "
                         "normal-orientation consistency")
    ap.add_argument("--poisson-normal-source",
                    choices=["pca", "gaussian-axis"], default="pca",
                    help="[poisson] normal source: 'pca' from Gaussian centers "
                         "or 'gaussian-axis' from each splat's thinnest axis")
    ap.add_argument("--poisson-density-quantile", type=float, default=0.05,
                    help="[poisson] trim vertices below this density quantile")
    args = ap.parse_args()

    globals()["USE_MESHFIX"] = args.meshfix
    print(f"[1/4] parsing {args.input_ply}")
    xyz, opacity, scale, rot = parse_3dgs_ply(
        args.input_ply, opacity_thresh=args.opacity_thresh
    )

    if args.method == "density":
        print(f"[2/4] splatting density volume @ resolution {args.resolution}")
        density, voxel_size, bb_min = splat_density_grid(
            xyz, opacity, scale, rot,
            resolution=args.resolution,
            padding=args.padding,
            n_sigma=args.n_sigma,
            chunk_size=args.chunk_size,
        )

        print("[3/4] density -> SDF -> marching cubes")
        sdf, iso = density_to_sdf(
            density, iso=args.iso, iso_quantile=args.iso_quantile,
            smooth_sigma=args.smooth_sigma, close_radius=args.close_radius,
            fill_2d_axes=args.fill_2d_axes,
        )
        verts, faces, _ = marching_cubes_sdf(
            sdf, voxel_size, bb_min, level=args.mc_level
        )
    else:
        print("[2/4] poisson surface reconstruction")
        verts, faces = poisson_reconstruct(
            xyz, opacity, scale, rot,
            depth=args.poisson_depth,
            normal_k=args.poisson_normal_k,
            density_quantile=args.poisson_density_quantile,
            normal_source=args.poisson_normal_source,
        )

    print(f"[4/4] cleanup + export -> {args.output_glb}")
    cleanup_and_export(
        verts, faces, args.output_glb,
        target_faces=args.target_faces,
        min_cluster_tris=args.min_cluster,
        z_up_to_y_up=args.z_up_to_y_up,
        taubin_iters=args.taubin_iters,
    )
    print("done")


if __name__ == "__main__":
    main()
