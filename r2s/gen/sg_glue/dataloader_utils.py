"""Scene loader for ``inputs.place`` (the parts of ``SamGemini`` MatchFindRot uses).

Vendored from scene_generation ``src/utils/dataloader_utils.py``. Kept with
their bodies unchanged: ``get_mask`` (the bypass branch MatchFindRot always
takes), ``get_rgb``, ``get_depth``, ``filter_pointclouds``, ``get_original``.

Changed:
  - ``__init__`` takes the scene folder explicitly instead of
    ``config.paths.data_dir / task``, and no longer creates the unused
    ``./output/{tracking_results,gemini_output,molmo,SoM,gemini_non}`` folders;
  - the old constructor rewrote ``intrinsics.npy`` in place when it was not
    3x3; here the first matrix is taken in memory only (inputs are read only);
  - ``get_depth(background=True)`` no longer silently falls back to
    ``depth/<i>.npy`` when ``inpainted_depth.npy`` is missing; the worker picks
    the depth file explicitly (``--background-depth``), so the old fallback is
    an explicit choice instead of a file-existence race.
"""
import os

import cv2
import numpy as np
import open3d as o3d

from .matching_utils import get_uncropped_keypoints, project_pixel_to_point


class SamGemini:
    def __init__(self, scene_dir, config, background_depth_path=None):
        self.config = config
        self.dataset = self.config['dataset']
        self.data_dir_task = str(scene_dir)
        self.video_dir = os.path.join(self.data_dir_task, "images0")
        self.depth_dir = os.path.join(self.data_dir_task, "depth")
        self.mask_dir = os.path.join(self.data_dir_task, "masks")
        self.pcd_dir = os.path.join(self.data_dir_task, "pcds")
        self.background_depth_path = background_depth_path

        assert(os.path.exists(self.data_dir_task))
        assert(os.path.exists(self.video_dir))
        assert(os.path.exists(self.depth_dir))
        self.use_bypass = False
        self.intrinsic = np.load(os.path.join(self.data_dir_task, "intrinsics.npy"), allow_pickle=True)
        self.extrinsic = np.load(os.path.join(self.data_dir_task, "extrinsics.npy"), allow_pickle=True)
        if self.intrinsic.shape != (3, 3):
            self.intrinsic = self.intrinsic[0]

    def get_mask(self, frame_idx, use_bypass=False):
        # Use the bypass method to get the mask
        assert(frame_idx == 1)
        masks = {}
        for img in os.listdir(self.mask_dir):
            if img.endswith("_mask.png"):
                img_path = os.path.join(self.mask_dir, img)
                img_cnt = img.split("_")[1]
                # turn 0001 to int
                mask_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # trun the mask to true or false
                masks[img_cnt] = mask_image > 0
        return masks

    def get_rgb(self, frame_idx):
        frame_path = os.path.join(self.video_dir, f"{frame_idx}.jpg")
        if os.path.exists(frame_path):
            return cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        print(f"RGB frame {frame_idx} not found.")
        return None

    def get_depth(self, frame_idx, background=False):
        depth_path = os.path.join(self.depth_dir, f"{frame_idx}.npy")
        if background == True and self.background_depth_path is not None:
            depth_path = str(self.background_depth_path)
        if os.path.exists(depth_path):
            return np.load(depth_path)
        raise FileNotFoundError(f"Depth frame {frame_idx} not found at {depth_path}.")

    def filter_pointclouds(self, pcd, num_neighbors:int, std_ratio:float, radius):
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=num_neighbors, std_ratio=std_ratio)
        pcd = pcd.select_by_index(ind)
        if radius > 0:
            _, ind = pcd.remove_radius_outlier(nb_points=num_neighbors, radius=radius)
            pcd = pcd.select_by_index(ind)

        return pcd

    def get_original(self, frame_idx, mesh_cnt, X,Z, keypoints = None, background=False):
        masks = self.get_mask(frame_idx, use_bypass=True)
        rgb = self.get_rgb(frame_idx)
        depth = self.get_depth(frame_idx, background=background)
        demo_image = None

        # convert depth to float32
        depth = depth.astype(np.float32)
        if rgb is None or depth is None:
            return None

        width, height = rgb.shape[1], rgb.shape[0]
        rgb = cv2.resize(rgb, (int(width ), int(height ))).astype(np.int8)
        depth = cv2.resize(depth, (int(width ), int(height)))

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.set_intrinsics(int(width ), int(height ),
                                     self.intrinsic[0, 0] , self.intrinsic[1, 1] ,
                                     self.intrinsic[0, 2] , self.intrinsic[1, 2] )
        extrinsics = self.extrinsic

        for obj_name, mask in masks.items():
            if obj_name == mesh_cnt:
                masked_rgb = np.zeros_like(rgb, dtype=np.uint8)
                masked_depth = np.zeros_like(depth)
                # Apply mask to rgb and depth arrays
                masked_rgb[mask] = rgb[mask]
                masked_depth[mask] = depth[mask]
                # Set the regions outside the mask to white (255)
                masked_rgb[~mask] = 0
                demo_image = masked_rgb

                image1 = np.zeros(masked_rgb.shape, dtype=np.uint8)
                rgbd_image = o3d.geometry.RGBDImage()
                rgbd_image = rgbd_image.create_from_color_and_depth(
                    o3d.geometry.Image(masked_rgb), o3d.geometry.Image(masked_depth),
                    depth_scale=1.0, convert_rgb_to_intensity=False)

                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_o3d, extrinsic=self.extrinsic)
                pcd = self.filter_pointclouds(pcd, 10, 2.0, 0.01)
                points_3d = []

                if keypoints is not None:
                    keypoints = get_uncropped_keypoints(keypoints, X,Z)
                    points_2d = keypoints
                    depth_key = np.zeros(masked_depth.shape)
                    for point in points_2d:
                        depth_point = masked_depth[point[1], point[0]]
                        u, v = point
                        if depth_point <= 0:
                            # find the nearest point
                            for distance in range(1, 10):
                                for m, n in [(v-distance, u), (v+distance, u), (v, u-distance), (v, u+distance)]:
                                    if masked_depth[int(m), int(n)] > 0 and int(m) < height and int(n) < width:
                                        depth_point = masked_depth[int(m), int(n)]
                                        u, v = n, m
                                        break

                        depth_key[v,u] = depth_point
                        cv2.circle(masked_rgb, (point[0], point[1]), radius=1, color=(0, 255, 0))

                        if depth_point > 0:
                            point3d = project_pixel_to_point(u,v,depth_point, self.intrinsic, self.extrinsic)
                        else:
                            point3d = [-1, -1, -1]
                        points_3d.append(point3d)
                        image1[v, u] = (0, 255, 0)

                    points_3d = np.array(points_3d)
                    depth = np.array(depth_key)

        if self.config['gemini_mask'] and not self.config['use_bypass']:
            for img in os.listdir(self.mask_dir):
                if img.endswith("_mask.png"):
                    img_path = os.path.join(self.mask_dir, img)
                    img_cnt = img.split("_")[1]
                    if img_cnt == mesh_cnt:
                        demo_image = cv2.imread(img_path)
                        demo_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)

        if keypoints is not None:
            return points_3d,pcd
        else:
            return demo_image, depth, self.intrinsic, extrinsics
