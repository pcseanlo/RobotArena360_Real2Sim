"""Place one SAM 3D mesh in the DROID scene: scale/ICP init, Genesis multi-view
render, SuperPoint+LightGlue match against the masked camera image, then a
rigid (object) or similarity (``background``) fit on the 3D correspondences.

Vendored from scene_generation ``src/pipeline/MatchFindRot.py`` -- the ORIGINAL
file, which production used (``transformations.json`` entries carry its
``original_rotation`` / ``quat_rotation`` keys). Bodies are unchanged except:

  - paths are explicit: ``assets_dir`` replaces ``config.assets.asset_dir/<scene>``
    and ``scene_dir`` replaces ``config.paths.data_dir/<scene>``; the global
    ``args.background`` became the ``background`` parameter;
  - ``PointCloudProcessor`` reads the same ``noise_removal`` / ``icp`` values,
    which are identical in ``config_rh20t.yaml`` (the file its default argument
    pointed at, and what production silently used) and ``config_DROID.yaml``;
  - ``process_mesh`` returns the ``transformations.json`` entry instead of
    read-modify-writing the file (the orchestrator merges the entries);
  - the 240 rendered views are matched in view-index order instead of
    ``os.listdir`` order (filesystem dependent), so ties in match count are broken
    reproducibly (lowest view index wins);
  - removed dead code: the ``*_scaled.glb`` visual-colour branch and
    ``save_transformation`` (both no-ops for production inputs).
"""
import copy
import os
import re

import cv2
import genesis as gs
import numpy as np
import open3d as o3d
import torch
import trimesh

from .dataloader_utils import SamGemini
from .matching_utils import Unprojector, apply_transformation, compute_rigid_transform, crop_and_resize, two_stage_alignment
from .minima_sp_lg import run_pose_benchmark
from .simulator_genesis import SimulatorGenesis


class PointCloudProcessor:
    def __init__(self, assets_dir, mesh_glb, pcd_path, mesh_cnt, config):
        self.config = config
        self.reset_params()
        self.result_dir = str(assets_dir)
        self.mesh_result_dir = os.path.join(str(assets_dir), mesh_cnt)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.mesh_result_dir, exist_ok=True)
        self.mesh_glb_path = mesh_glb
        self.pcd_path = pcd_path
    def reset_params(self):
        self.scale_factor = None
        self.translation = None
        self.transformation_matrix = np.identity(4)
        self.ori_ply = None
        self.mesh_ply = None
        self.pcd = None
        self.R = None
        self.translation2 = None
    def load_ply_pointcloud(self, ply_path):
        return o3d.io.read_point_cloud(ply_path)

    def remove_noise_neigh(self, pointcloud):
        params = self.config["noise_removal"]
        clean, _ = pointcloud.remove_statistical_outlier(
            nb_neighbors=params["nb_neighbors"], std_ratio=params["std_ratio"]
        )
        return clean

    def remove_noise(self, pointcloud):
        params = self.config["noise_removal"]
        pointcloud = self.remove_noise_neigh(pointcloud)
        plane_model, inliers = pointcloud.segment_plane(
            params["distance_threshold"], params["ransac_n"], params["num_iterations"]
        )
        return pointcloud.select_by_index(inliers)

    def compute_scale_ply(self, pointcloud):
        bbox = pointcloud.get_axis_aligned_bounding_box()
        return np.linalg.norm(bbox.get_extent())

    def extract_face_colors(self, mesh, count=50000):
        if not isinstance(mesh.visual, trimesh.visual.TextureVisuals):
            print("The mesh does not have texture visuals.")
            return None
        face_colors = mesh.visual.to_color().vertex_colors[:, :3] / 255.0
        face_colors = face_colors[mesh.faces].mean(axis=1)
        sampled_points, face_index = trimesh.sample.sample_surface(mesh, count=count)
        return sampled_points, face_colors[face_index]

    def transform_glb_to_match_ply(self, glb=True):
        # Load the mesh
        ply_pointcloud = self.load_ply_pointcloud(self.pcd_path)
        ply_pointcloud = self.remove_noise(ply_pointcloud)
        self.pcd = ply_pointcloud

        if glb == True:
            if self.mesh_glb_path.endswith("_scaled.glb"):
                raise NotImplementedError("the *_scaled.glb + *_painted.glb branch is not vendored")
            mesh_path = os.path.join(self.result_dir, self.mesh_glb_path)
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.geometry.values())
            glb_mesh, glb_color = self.extract_face_colors(mesh)
            mesh_ply = o3d.geometry.PointCloud()
            mesh_ply.points = o3d.utility.Vector3dVector(glb_mesh)
            mesh_ply.colors = o3d.utility.Vector3dVector(glb_color)
        else:
            mesh_ply = self.load_ply_pointcloud(self.glb_path)
            self.ori_ply = copy.deepcopy(mesh_ply)
            self.mesh_ply = mesh_ply
            self.scale_factor = 1
            self.translation = np.array([0, 0, 0])

        # Load the point cloud
        self.ori_ply = copy.deepcopy(mesh_ply)  # Store original mesh as point cloud

        # Compute transformation parameters
        # 1. Do the scaling
        self.scale_factor = self.compute_scale_ply(ply_pointcloud) / self.compute_scale_ply(mesh_ply)
        mesh_ply.scale(self.scale_factor, center=mesh_ply.get_center())
        # 2. Do the translation
        self.translation = ply_pointcloud.get_center() - mesh_ply.get_center()
        mesh_ply.translate(self.translation)
        # 3. save the transformation matrix
        self.mesh_ply = mesh_ply

    def register_point_clouds(self, source, target):
        params = self.config["icp"]
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))

        trans_init = np.identity(4)
        threshold = params["initial_threshold"]

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        trans_init = reg_p2p.transformation

        threshold = params["fine_threshold"]
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=params["max_iterations"])
        )
        trans_init = reg_p2p.transformation
        threshold = params["final_threshold"]
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        self.transformation_matrix = reg_p2l.transformation
        return self.transformation_matrix
    def apply_transformation(self):
        """
        Apply the transformation to the original point cloud.
        """
        self.ori_ply.scale(self.scale_factor, center=self.ori_ply.get_center())
        self.ori_ply.translate(self.translation)
        if self.R is not None and self.translation2 is not None:
            self.ori_ply = apply_transformation(self.ori_ply, self.R, self.translation2)
        self.ori_ply.transform(self.transformation_matrix)
        return self.ori_ply

    def process(self):
        self.transform_glb_to_match_ply()
        ori = copy.deepcopy(self.ori_ply)
        target = self.pcd
        source = self.mesh_ply
        transformation = self.register_point_clouds(source, target)
        transformed_final = self.apply_transformation()
        return transformed_final, ori

    def render_genesis(self, similator, view_direction, pos, cam, distance):
        directory = os.path.join(self.mesh_result_dir, "segmented_images")
        os.makedirs(directory, exist_ok=True)
        camera = cam

        n = 60
        theta_values = np.linspace(0, 2 * np.pi, n)
        n *= 4
        intrinsics = []
        extrinsics = []

        z = [0.1, 0.5, 0.7, 0.85]
        for i in range(n):
            theta = theta_values[i % len(theta_values)]
            new_x = np.cos(theta)
            new_y = np.sin(theta)
            new_z = z[int(i//(n/4))]
            new_view_direction = np.array([new_x, new_y, new_z])


            pos_camera = pos + distance * new_view_direction
            camera.set_pose(
                pos=pos_camera
            )
            rgb, depth, seg, _ = camera.render(rgb=True, depth=True, segmentation=True)
            seg_binary = np.where(seg > 0, 255, 0).astype(np.uint8)
            segmented_part = cv2.bitwise_and(rgb, rgb, mask=seg_binary)
            segmented_background = np.zeros_like(rgb)
            segmented_image = cv2.add(segmented_part, segmented_background)
            seg_img_path = f"{directory}/segmented_image{i}.png" # {i}

            cv2.imwrite(seg_img_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
            segmented_depth = cv2.bitwise_and(depth, depth, mask=seg_binary)
            # Save the depth npy
            np.save(f"{directory}/depth_{i}.npy", segmented_depth)
            self.depth_mesh = segmented_depth
            # save intrinsics and extrinsics
            intrinsics.append(camera.intrinsics)
            extrinsics.append(camera.extrinsics)
        intrinsics = np.array(intrinsics)
        extrinsics = np.array(extrinsics)
        output = f"{directory}/intrinsics_extrinsics.npy"
        np.save(output, {"intrinsics": intrinsics, "extrinsics": extrinsics})
        return intrinsics, extrinsics

    def load_keypoints(self, keypoints0, keypoints1, threshold):
        mesh_center = self.mesh_ply.get_center()
        filtered_keypoint_matches = []
        for points1, points2 in zip(keypoints0, keypoints1):
            if np.array_equal(points1, [-1, -1, -1]) or np.array_equal(points2, [-1, -1, -1]):
                continue
            points1_transformed = (points1 - mesh_center) * self.scale_factor + mesh_center
            points1_transformed = points1_transformed + self.translation
            bbox_mesh = self.mesh_ply.get_axis_aligned_bounding_box()
            bbox_ply = self.pcd.get_axis_aligned_bounding_box()

            if (np.any(points1_transformed < bbox_mesh.min_bound - threshold)
                or np.any(points1_transformed > bbox_mesh.max_bound + threshold)):
                continue
            if (np.any(points2 < bbox_ply.min_bound - threshold)
                or np.any(points2 > bbox_ply.max_bound + threshold)):
                continue
            filtered_keypoint_matches.append((points1_transformed, points2))

        keypoints0_filtered = np.array([pair[0] for pair in filtered_keypoint_matches])
        keypoints1_filtered = np.array([pair[1] for pair in filtered_keypoint_matches])

        if keypoints0_filtered.shape[0] == 0:
            print("No keypoints is good")
            return None, None

        return keypoints0_filtered, keypoints1_filtered

    def apply_rigid_transform(self, keys0, keys1):
        R, t = compute_rigid_transform(keys0, keys1)
        self.transformation_matrix = np.eye(4)
        self.R = R
        self.translation2 = t
        transformed_final = self.apply_transformation()
        return transformed_final

    def apply_similarity_transform(self, keys0, keys1):
        s, R, t = two_stage_alignment(keys0, keys1)
        self.R = s * R  # Store scaled rotation for other uses
        self.translation2 = t
        transformed_final = self.apply_transformation()
        return transformed_final

def start_simulation(asset_path,output_path, pos, bounding_box, scale, background=False, dataset="bridge"):
    simulator = SimulatorGenesis("ur5", 1, show_viewer=False, add_robot=False)
    simulator.start_sim()
    quat = simulator.asset_addtion(asset_path=asset_path, pos=pos, scale=scale)

    camera = simulator.scene.add_camera(
        res=(480, 480),
        pos=(-0.095, -0.154387184491962615, 1.6611819729252184),
        lookat=(0.44, 0.20, 0.20),
        fov=60,
        GUI=False,
    )

    bbox =np.linalg.norm(bounding_box.get_extent())

    if background == False:
        scale_factor = 2.8
    else:
        if dataset == "bridge":
            scale_factor = 1.2
        else:
            scale_factor = 0.4

    camera_distance = scale_factor * bbox
    lookat = pos
    view_direction = np.array([1, 0.6 ,1])
    pos_camera = pos + camera_distance * view_direction

    camera1 = simulator.scene.add_camera(
        res=(520, 520),
        pos= pos_camera,
        lookat=lookat,
        fov = 32,
        GUI=False,
    )

    simulator.scene.build()
    asset = simulator.asset_ID[scale][0]
    pos = asset.get_links_pos()
    quat = asset.get_links_quat()

    res = []
    colors = []

    for link in asset.links:
        link_points = []
        link_colors = []

        for mesh in [g.get_trimesh() for g in link.vgeoms]:
            # Sample points from the mesh surface
            points, face_index, colors_ = trimesh.sample.sample_surface(mesh,count=100000,sample_color=True)
            sampled_colors = np.array(colors_)[:, :3] / 255.0
            points = np.array(points)

            link_points.append(points)
            link_colors.append(sampled_colors)

        res.append(np.concatenate(link_points, axis=0))
        colors.append(np.concatenate(link_colors, axis=0))

    # Transform point clouds using the simulator
    for i in range(len(res)):
        t = simulator.transform_point_cloud(torch.tensor(res[i], dtype=torch.float32).to(gs.device), pos[i], quat[i])
        res[i] = t.cpu().numpy()
    # Concatenate all sampled points and colors
    all_points = np.concatenate(res, axis=0)  # Shape: (N, 3)
    all_colors = np.concatenate(colors, axis=0)  # Shape: (N, 3)
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)  # Open3D requires color values in [0, 1]
    # Save the point cloud
    o3d.io.write_point_cloud(output_path, pcd)
    return simulator, view_direction, camera1, camera_distance, quat


def _view_index(file):
    match = re.search(r'\d+', file)
    return int(match.group()) if match else -1


def process_mesh(scene_dir, assets_dir, mesh_glb, mesh_cnt, checkpoint_dir, config, background=False,
                 background_depth_path=None):
    """Return ``(key, entry)`` for ``transformations.json``, or ``(key, None)``
    when no filtered keypoint survives (the old script then wrote nothing)."""
    frame_index = 1
    sam = SamGemini(scene_dir, config, background_depth_path=background_depth_path)
    if background:
        demo_image, demo_depth, demo_intrinsics, demo_extrinsics = sam.get_original(frame_index, mesh_cnt, 0, 0, keypoints=None, background=True)
    else:
        demo_image, demo_depth, demo_intrinsics, demo_extrinsics = sam.get_original(frame_index, mesh_cnt, 0, 0, keypoints=None, background=False)

    pcd_ply = os.path.join(sam.pcd_dir, f"{mesh_cnt}_frame1.ply")

    processor = PointCloudProcessor(assets_dir, mesh_glb, pcd_ply, mesh_cnt, config)
    _, ori_ply = processor.process()


    genesis_mesh_ply = os.path.join(processor.mesh_result_dir, "mesh.ply")
    bounding_box = processor.mesh_ply.get_axis_aligned_bounding_box()

    simulator, view_direction, camera, distance, quat = start_simulation(
        os.path.join(processor.result_dir, processor.mesh_glb_path),
        genesis_mesh_ply, processor.translation, bounding_box,
        processor.scale_factor, background=background, dataset=sam.dataset
    )

    genesis_intrinsics, genesis_extrinsics = processor.render_genesis(simulator, view_direction, processor.translation, camera, distance)
    demo_path = os.path.join(processor.mesh_result_dir, "demo.png")
    cv2.imwrite(demo_path, demo_image)

    X, Z, demo_cropped = crop_and_resize(demo_image, padding=10, output_path=os.path.join(processor.mesh_result_dir, "cropped_image.png"))

    image_pair1 = demo_cropped
    image_pair1_path = os.path.join(processor.mesh_result_dir, "cropped_image.png")
    image_pair2_path = None
    image_pair_dir = os.path.join(processor.mesh_result_dir, "segmented_images")

    max_points = 0
    image_count = -1
    keypoints0 = None
    keypoints1 = None

    for file in sorted(os.listdir(image_pair_dir), key=lambda f: (_view_index(f), f)):
        if file.endswith(".png") and "rgb" not in file:
            image = cv2.imread(os.path.join(image_pair_dir, file))
            count = int(re.search(r'\d+', file).group()) if re.search(r'\d+', file) else None
            genesis_image_path = os.path.join(image_pair_dir, file)
            keypoints0_, keypoints1_, points = run_pose_benchmark("sp_lg", genesis_image_path, image_pair1_path,
                                                                  processor.mesh_result_dir, checkpoint_dir)
            if points > max_points:
                max_points = points
                image_count = count
                keypoints0 = keypoints0_
                keypoints1 = keypoints1_
                image_pair2_path = genesis_image_path

    print(f"Best matching image is {image_count}")

    mesh_key = f"image_{mesh_cnt}_segmented"

    if image_count == -1:
        rota = np.eye(3)
        translation1 = processor.translation
        scale1 = processor.scale_factor
        # extract rotation and translation from transformation matrix
        return mesh_key, {
            "scale": scale1.tolist(),
            "translation": translation1.tolist(),
            "rotation": rota.tolist()
        }

    intrinsics_extrinsics = np.load(os.path.join(processor.mesh_result_dir, "segmented_images/intrinsics_extrinsics.npy"), allow_pickle=True).item()
    genesis_intrinsics = intrinsics_extrinsics["intrinsics"][image_count]
    genesis_extrinsics = intrinsics_extrinsics["extrinsics"][image_count]
    genesis_depth = np.load(os.path.join(processor.mesh_result_dir, f"segmented_images/depth_{image_count}.npy"))

    unprojector0 = Unprojector(genesis_intrinsics, genesis_extrinsics, genesis_depth, keypoints0, 0, processor.mesh_result_dir)
    unprojector1 = Unprojector(demo_intrinsics, demo_extrinsics, demo_depth, keypoints1, 1, processor.mesh_result_dir)

    keypoints0 = unprojector0.unproject_to_point_cloud(X, Z, image_pair2_path)
    unprojector1.unproject_to_point_cloud(X, Z, demo_path)
    if background:
        keypoints1, pcd = sam.get_original(frame_index, mesh_cnt, X, Z, keypoints1, background=True)
    else:
        keypoints1, pcd = sam.get_original(frame_index, mesh_cnt, X, Z, keypoints1, background=False)

    keypoint_pc = o3d.geometry.PointCloud()
    keypoint_pc.points = o3d.utility.Vector3dVector(keypoints1)

    processor2 = PointCloudProcessor(assets_dir, mesh_glb, pcd_ply, mesh_cnt, config)
    processor2.glb_path = genesis_mesh_ply

    processor2.transform_glb_to_match_ply(glb=False)
    keypoints0 = np.load(os.path.join(processor.mesh_result_dir, "keypoints3d0.npy"))
    keys0, keys1 = processor2.load_keypoints(keypoints0, keypoints1, 0.1)

    if keys0 is None:
        return mesh_key, None

    if background:
        processor2.apply_similarity_transform(keys0, keys1)
    else:
        processor2.apply_rigid_transform(keys0, keys1)

    scale1 = processor.scale_factor * processor2.scale_factor
    translation2 = processor2.translation2
    rotation = processor2.R

    qw, qx, qy, qz = quat.unbind(-1)
    rot_matrix = torch.stack([
        1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
        2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
        2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
    ], dim=-1).reshape(*quat.shape[:-1], 3, 3)

    R1 = rot_matrix.cpu().numpy()[0] # quat (.717,.717,0,0) [1]
    R2 = rotation                    # rotation for the matching [2]
    R_final = R2 @ R1
    t1 = processor.translation       # initial translation [1]
    t2 = processor2.translation2     # translation for the matching [2]
    t0 = processor2.translation      # here is another translation I do before the matching(last time I missed) [1]
    t_final = R2 @ (t1+t0)+ t2

    entry = {
        "scale": scale1.tolist(),
        "translation": t_final.tolist(),
        "rotation": R_final.tolist(),
        "original_rotation": R2.tolist(),
        "quat_rotation": R1.tolist()
    }

    if background:
        base_dir = os.path.dirname(processor2.mesh_result_dir)
        mesh_path = os.path.join(base_dir, processor2.mesh_glb_path)
        mesh = trimesh.load(mesh_path)

        # Build transform (scale, rotation, translation)
        S = np.eye(4); S[:3, :3] *= scale1
        R = np.eye(4); R[:3, :3] = R_final
        T = np.eye(4); T[:3, 3] = t_final
        M = T @ R @ S
        mesh.apply_transform(M)
        mesh.export(os.path.join(base_dir, "table.glb"))

    return mesh_key, entry
