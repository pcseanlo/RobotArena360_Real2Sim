"""Worker for ``inputs.mesh_sheet`` (env ``match``): sheet-wrap the inpainted depth into a mesh.

Port of scene_generation ``src/pipeline/sheet_wraping_updated.py`` with
``--depth_source aligned`` (the production setting). Body unchanged except:
  - the colour image is passed explicitly (``--rgb``, the background mask image
    written by ``inputs.table_depth``) instead of "last sorted masks/*_mask.png";
  - depth / intrinsics / output paths are explicit;
  - the unused ``mirror`` depth source (and its ``--conf_threshold``) is dropped;
  - ``beta`` / ``alpha_threshold`` / ``depth_eps`` are flags (the old config had no
    ``sheet_wrap:`` block, so the old defaults 10 / 0.15 / 1e-4 applied).
"""
import argparse
import sys
from pathlib import Path

# sg_glue lives in r2s/gen; append (not prepend) so it cannot shadow env packages.
sys.path.append(str(Path(__file__).resolve().parents[1]))


def _get_ray_directions(H, W, fx, fy, cx, cy):
    import torch

    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    x = (u.float() - cx) / fx
    y = (v.float() - cy) / fy
    z = torch.ones_like(x)
    return torch.stack([x, y, z], dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rgb", required=True, help="masks/image_<bg>_mask.png (masked background RGB)")
    parser.add_argument("--depth", required=True, help="inpainted_depth.npy")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--out", required=True, help="mesh_sheet.glb")
    parser.add_argument("--dataset", type=str, default="DROID",
                        help="the disparity edge filter only runs for DROID (old behaviour)")
    parser.add_argument("--target_tris", type=int, default=200_000,
                        help="Quadric decimation target. Set <=0 to disable.")
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--alpha_threshold", type=float, default=0.15)
    parser.add_argument("--depth_eps", type=float, default=1e-4)
    args = parser.parse_args()

    import imageio.v2 as imageio
    import numpy as np
    import open3d as o3d
    import torch
    import trimesh

    from sg_glue.hunyuanworld import get_filtered_mask, sheet_warping

    sw_cfg = {"beta": args.beta, "alpha_threshold": args.alpha_threshold, "depth_eps": args.depth_eps}
    print(f"[sheet_wrap_updated] cfg: {sw_cfg}")

    rgb = imageio.imread(args.rgb)
    depth = np.load(args.depth)
    intrinsics = np.load(args.intrinsics)
    mirror_conf = None
    print(f"[sheet_wrap_updated] depth_source=aligned  depth={args.depth}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_t = torch.from_numpy(depth).float().to(device)
    disparity_t = torch.reciprocal(depth_t + 1e-6)
    disparity_bhwc = disparity_t.unsqueeze(0).unsqueeze(-1)
    use_disparity_filter = (args.dataset == "DROID")
    if use_disparity_filter:
        vis_mask_t = get_filtered_mask(
            disparity_bhwc, beta=sw_cfg["beta"], alpha_threshold=sw_cfg["alpha_threshold"]
        )
        filtered_mask = vis_mask_t.squeeze().cpu().numpy().astype(bool)
    else:
        filtered_mask = np.ones_like(depth, dtype=bool)

    initial_valid_mask = depth > sw_cfg["depth_eps"]
    valid = initial_valid_mask
    if use_disparity_filter:
        valid = initial_valid_mask & filtered_mask
    if mirror_conf is not None:
        valid = valid & (mirror_conf > 0.3)

    rgb = rgb * valid[..., None]
    depth = depth * valid

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    H, W = depth.shape

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    valid = valid & (z > 0)

    rays = _get_ray_directions(H, W, fx, fy, cx, cy)  # CPU; small tensor, avoids host<->device traffic
    rays = rays * torch.from_numpy(valid[..., None]).to(rays.device)

    distance_t = torch.from_numpy(depth).float()
    distance_t[~torch.from_numpy(valid)] = float("nan")

    predictions = {
        "rgb": torch.from_numpy(rgb).float(),
        "distance": distance_t,
        "rays": rays,
        "intrinsics": torch.from_numpy(intrinsics).float(),
    }

    mesh = sheet_warping(predictions)
    mesh.compute_vertex_normals()
    mesh.orient_triangles()
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])

    verts = np.asarray(mesh.vertices)
    to_remove = (np.linalg.norm(verts, axis=1) < 1e-6)
    mesh.remove_vertices_by_mask(to_remove)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if args.target_tris and args.target_tris > 0:
        cur_tris = len(mesh.triangles)
        if cur_tris > args.target_tris:
            print(f"[sheet_wrap_updated] decimating {cur_tris} -> {args.target_tris}")
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.target_tris)
            mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vcolors = np.asarray(mesh.vertex_colors)
    if vcolors.size and vcolors.max() > 1.0:
        vcolors = vcolors / 255.0
    if vcolors.shape[1] == 3:
        vcolors = np.hstack([vcolors, np.ones((vcolors.shape[0], 1))])
    vcolors_u8 = (np.clip(vcolors[:, :3], 0, 1) * 255).astype(np.uint8)

    # Apply the two 180° rotations directly on numpy verts (was done on an
    # o3d copy and exported — but the trimesh winding fix was silently lost).
    R_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)  # rot pi about Y
    R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)  # rot pi about X
    R = R_x @ R_y
    verts_rot = verts @ R.T

    mesh_tm = trimesh.Trimesh(vertices=verts_rot, faces=faces, vertex_colors=vcolors_u8, process=False)
    if not mesh_tm.is_winding_consistent:
        mesh_tm.invert()

    mesh_tm.export(args.out)
    print(f"Saved mesh: {args.out}  (verts={len(verts_rot)}, faces={len(faces)})")


if __name__ == "__main__":
    main()
