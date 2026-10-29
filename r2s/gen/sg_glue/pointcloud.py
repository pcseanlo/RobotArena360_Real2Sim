"""Per-object point clouds for step 0c-1, from SG ``src/utils/dataloader_utils.py``.

Moved from ``SamGemini``: the intrinsics handling of ``__init__``, ``get_mask``
(non-bypass branch), ``get_rgb``, ``get_depth`` (non-background branch),
``filter_pointclouds`` and ``mask_to_point_cloud``; the class state became
arguments. SG's ``__init__`` also wrote ``intrinsics[0]`` back to
``intrinsics.npy`` when the file was not 3x3; here that is only done in memory
(``inputs.episode`` writes the normalised file instead).
"""

import json
import os

import cv2
import numpy as np


def load_camera(meta_dir):
    intrinsic = np.load(os.path.join(meta_dir, "intrinsics.npy"), allow_pickle=True)
    extrinsic = np.load(os.path.join(meta_dir, "extrinsics.npy"), allow_pickle=True)
    if intrinsic.shape != (3, 3):
        intrinsic = intrinsic[0]
    return intrinsic, extrinsic


def get_mask(save_dict, frame_idx):
    """Masks of ``frame_{idx:05d}_bounding_boxes_and_masks.json`` keyed by ``count`` (``"0000"``...)."""
    file_path = os.path.join(save_dict, f"frame_{frame_idx:05d}_bounding_boxes_and_masks.json")
    if not os.path.exists(file_path):
        print(f"[ERROR] Mask file for frame {frame_idx} not found.")
        return None
    with open(file_path, "r") as f:
        frame_data = json.load(f)
    masks_dict = {}
    for obj in frame_data:
        if "count" in obj:
            count = str(obj["count"]).zfill(4)
            masks_dict[count] = np.array(obj["mask"])
        else:
            masks_dict[obj["object_name"]] = np.array(obj["mask"])
    return masks_dict


def get_rgb(video_dir, frame_idx):
    frame_path = os.path.join(video_dir, f"{frame_idx}.jpg")
    if os.path.exists(frame_path):
        return cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
    print(f"RGB frame {frame_idx} not found.")
    return None


def get_depth(depth_dir, frame_idx):
    depth_path = os.path.join(depth_dir, f"{frame_idx}.npy")
    if os.path.exists(depth_path):
        return np.load(depth_path)
    print(f"Depth frame {frame_idx} not found at {depth_path}.")
    return None


def filter_pointclouds(pcd, num_neighbors, std_ratio, radius):
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=num_neighbors, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)
    if radius > 0:
        _, ind = pcd.remove_radius_outlier(nb_points=num_neighbors, radius=radius)
        pcd = pcd.select_by_index(ind)
    return pcd


def mask_to_point_cloud(meta_dir, frame_idx=1):
    """Write ``pcds/{count:04d}_frame{idx}.ply`` for every object of the frame's mask JSON."""
    import open3d as o3d

    save_dict = os.path.join(meta_dir, "bounding_boxes_and_masks")
    pcd_dir = os.path.join(meta_dir, "pcds")
    os.makedirs(pcd_dir, exist_ok=True)
    intrinsic, extrinsic = load_camera(meta_dir)

    masks = get_mask(save_dict, frame_idx)
    rgb = get_rgb(os.path.join(meta_dir, "images0"), frame_idx)
    depth = get_depth(os.path.join(meta_dir, "depth"), frame_idx)

    if rgb is None or depth is None:
        return None

    width, height = rgb.shape[1], rgb.shape[0]
    rgb = cv2.resize(rgb, (int(width), int(height))).astype(np.uint8)
    depth = cv2.resize(depth, (int(width), int(height)))

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_o3d.set_intrinsics(int(width), int(height),
                                 intrinsic[0, 0], intrinsic[1, 1],
                                 intrinsic[0, 2], intrinsic[1, 2])
    written = {}
    for obj_name, mask in masks.items():
        masked_rgb = np.zeros_like(rgb)
        masked_rgb[mask] = rgb[mask]
        masked_depth = np.zeros_like(depth)
        masked_depth[mask] = depth[mask]
        masked_depth = masked_depth.astype(np.float32)
        rgbd_image = o3d.geometry.RGBDImage()
        rgbd_image = rgbd_image.create_from_color_and_depth(
            o3d.geometry.Image(masked_rgb),
            o3d.geometry.Image(masked_depth),
            depth_scale=1.0,
            convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_o3d, extrinsic=extrinsic)
        pcd = filter_pointclouds(pcd, 10, 2.0, 0.01)
        path = os.path.join(pcd_dir, f"{obj_name}_frame{frame_idx}.ply")
        o3d.io.write_point_cloud(path, pcd)
        written[obj_name] = int(len(pcd.points))
        print(f"Saved point cloud for {obj_name} at frame {frame_idx}")
    return written
