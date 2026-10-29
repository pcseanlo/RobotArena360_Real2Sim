"""Keypoint unprojection and rigid / similarity solvers used by ``inputs.place``.

Vendored from scene_generation ``src/utils/matching_utils.py`` (bodies unchanged).
Removed: the plotting helper and the unused module-level noise filters, and in
``Unprojector.unproject_to_point_cloud`` the full-depth-image unprojection loop
for ``pair == 0`` whose result was never read (pure dead work, ~270k Python
iterations per object).
"""
import os

import cv2
import numpy as np
import open3d as o3d


class Unprojector:
    def __init__(self, intrinsics, extrinsics, depth, keypoints,pair,output_path):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.depth_npy = depth
        self.keypoints = keypoints
        self.pair = pair
        self.output_path = output_path
    
    def project_pixel_to_point(self, u, v, depth):
        fx, fy, cx, cy = self.intrinsics[0, 0], self.intrinsics[1, 1], self.intrinsics[0, 2], self.intrinsics[1, 2]
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        points = np.array([x, y, z, 1])
        extrinsics = np.linalg.inv(self.extrinsics)
        points = extrinsics @ points  # Apply extrinsics transformation
        return points[:3]

    def save_point_cloud(self, point_cloud, filename):
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.io.write_point_cloud(filename, pcl)

    # def unproject_to_point_cloud(self,X,Z,h,w,ww,hh,image_path):
    def unproject_to_point_cloud(self,X,Z,image_path):
        # (dead full-image unprojection loop of the original removed: its result was never used)
        image = cv2.imread(image_path)
        points_2d = []
        for u, v in self.keypoints:
            x, y = int(u), int(v)
            if self.pair == 1:
                y += X
                x += Z
            points_2d.append([x,y])
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        
        if self.pair == 1:
            self.keypoints = points_2d
        
        keypoints_3d = []
        keypoints_3d_npy = []
        
        if self.pair == 0:
            for u, v in self.keypoints:
                depth = self.depth_npy[int(v), int(u)] 
                if depth <= 0:
                    keypoints_3d_npy.append([-1, -1, -1])
                if depth > 0: 
                    point_3d = self.project_pixel_to_point(u, v, depth)
                    keypoints_3d.append(point_3d)
                    keypoints_3d_npy.append(point_3d)

            keypoints_3d = np.array(keypoints_3d)
            keypoints_3d_npy = np.array(keypoints_3d_npy)
        
            np.save(os.path.join(self.output_path, f"keypoints3d{self.pair}.npy"), keypoints_3d_npy)
            self.save_point_cloud(keypoints_3d_npy, os.path.join(self.output_path, f"keypoints3d{self.pair}.ply"))


def apply_transformation(pcd, R, t):
    """
    Apply transformation (R, t) to an Open3D point cloud.
    """
    pcd_np = np.asarray(pcd.points)
    transformed_points = (R @ pcd_np.T).T + t
    pcd.points = o3d.utility.Vector3dVector(transformed_points)
    return pcd


def compute_rigid_transform(source_corr, target_corr):
    """
    Compute the optimal rigid transformation (R, t) from source to target using SVD.
    """
    assert source_corr.shape == target_corr.shape, "Point sets must have the same shape"

    # Compute centroids
    centroid_source = np.mean(source_corr, axis=0)
    centroid_target = np.mean(target_corr, axis=0)

    # Center the points
    source_centered = source_corr - centroid_source
    target_centered = target_corr - centroid_target

    # Compute cross-covariance matrix
    H = source_centered.T @ target_centered

    # Compute SVD
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T  # Rotation

    # Ensure a proper rotation matrix (det(R) = 1, not -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_target - R @ centroid_source

    return R, t


def crop_and_resize(image, padding=40, output_path=None):
    # Convert image to HSV color space
    coords = np.argwhere(np.any(image != [0, 0, 0], axis=-1)) 
    # Compute the bounding box around all non-black pixels
    x_min = coords[:, 1].min()
    y_min = coords[:, 0].min()
    x_max = coords[:, 1].max()
    y_max = coords[:, 0].max()
    
    # Apply padding while ensuring the crop does not exceed image boundaries
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, image.shape[1])
    y_max = min(y_max + padding, image.shape[0])

    # Crop the image based on the calculated bounding box
    cropped = image[y_min:y_max, x_min:x_max]

    # Save the cropped image if an output path is provided
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

    return y_min, x_min, cropped


def get_uncropped_keypoints(keypoints,X,Z):
    points_2d = []
    for (u, v) in keypoints:
        x, y = int(u), int(v)
        # y =int(y * w/ww)
        # x =int( x * h/hh)
        y += X
        x += Z
        points_2d.append([x,y])
    return np.array(points_2d)


def project_pixel_to_point(u, v, depth, intrinsics, extrinsics=None): 
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    points = np.array([x, y, z, 1])
    extrinsics = np.linalg.inv(extrinsics)
    points = extrinsics @ points  # Apply extrinsics transformation
    return points[:3]


def umeyama_alignment(src, dst, with_scale=True, save_path="umeyama_alignment.png"):
    """
    Computes the similarity transform (s, R, t) and optionally saves a
    visualization of the alignment.
    
    Args:
        src (np.ndarray): Source point cloud.
        dst (np.ndarray): Destination point cloud.
        with_scale (bool): Whether to compute scaling.
        save_path (str, optional): If provided, saves a plot to this path.
    
    Returns:
        tuple: The estimated scale (s), rotation (R), and translation (t).
    """
    # 1. --- Calculation (Original Logic) ---
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = src_c.T @ dst_c / len(src)
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    if with_scale:
        var = (src_c**2).sum() / len(src)
        s = S.sum() / var
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src

    # 2. --- Apply Transformation ---
    src_transformed = s * (src @ R.T) + t

    # # 3. --- Call Visualization/Save Logic ---
    # if save_path:
    #     _visualize_and_save(src, dst, src_transformed, save_path)
    return s, R, t


def two_stage_alignment(src, dst):
    # --- Stage 1: Rigid alignment ---
    R1, t1 = compute_rigid_transform(src, dst)
    src_aligned = (src @ R1.T) + t1

    # --- Stage 2: Similarity alignment (on rigid-aligned src) ---
    s2, R2, t2 = umeyama_alignment(src_aligned, dst, with_scale=True)

    # --- Compose transforms ---
    R_final = R2 @ R1
    s_final = s2
    t_final = R2 @ (t1 * s2) + t2
    # Apply final transform
    src_transformed = s_final * (src @ R_final.T) + t_final
    return s_final, R_final, t_final

