"""Minimal Genesis scene used by ``inputs.place`` to render a mesh from many views.

Vendored from scene_generation ``src/sim/simulator_genesis.py`` (Genesis 0.2.1,
conda env ``match``). Kept with bodies unchanged: ``asset_addtion`` (mesh branch
without physics, the only one MatchFindRot uses), ``set_scene``,
``gs_transform_by_quat``, ``transform_point_cloud``, ``start_sim``.
Removed: robot loading (``configs/robot_config.yaml``), surfaces, IK and point
cloud helpers for robots -- MatchFindRot builds the scene with ``add_robot=False``.
The worker seeds ``random``/``numpy``/``torch`` once at start-up instead of
passing a seed to ``gs.init`` (which would re-seed them mid-run).
"""
import genesis as gs
import torch


class SimulatorGenesis:
    def __init__(self, robot_name, envs, add_robot = None, device = None, show_viewer =False, seed=None):
        self.robot_name = robot_name
        self.robot = None
        self.envs = envs
        self.seed = seed
        self.asset_path = None
        self.device = device
        self.show_viewer = show_viewer
        self.robot_addition = add_robot
        self.asset_ID = {}

    def asset_addtion(self, asset_path, pos = None,
                      quat = None, scale = None,
                      physics = None, fixed = False):
        if pos is None:
            pos = (1, 0.5, 1.0)
        if quat is None:
            quat = (0.717, 0.717, 0, 0)
        if scale is None:
            scale = 0.12
        if physics is not None or asset_path.endswith('.urdf') or asset_path.endswith('.xml'):
            raise NotImplementedError("only plain mesh assets without physics are supported")
        asset = (self.scene.add_entity(
        gs.morphs.Mesh(file=asset_path,
                    fixed=fixed,
                    pos= pos,
                    quat= quat,
                    scale= scale,
                    ),
        ),
        None,
        None,
        )
        self.asset_ID[scale] = asset
        return quat

    def set_scene(self):
        viewer_options = gs.options.ViewerOptions(
                camera_pos=(3, -1, 2.5),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=30,
                max_FPS=60,
            )

        self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(
                    dt= 0.005,
                ),
                viewer_options=viewer_options,
                vis_options=gs.options.VisOptions(
                    show_world_frame=False,
                    shadow=False,
                    ambient_light=(0.4, 0.4, 0.4),
                ),
                rigid_options=gs.options.RigidOptions(
                    dt=0.005,
                ),
                show_viewer=self.show_viewer,
                show_FPS=False,
            )

        plane = self.scene.add_entity(
            gs.morphs.Plane(
                fixed=True,
                visualization=False,
            ),
            )
        return self.scene

    def gs_transform_by_quat(self, pos, quat):
        qw, qx, qy, qz = quat.unbind(-1)

        rot_matrix = torch.stack([
            1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
            2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
            2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
        ], dim=-1).reshape(*quat.shape[:-1], 3, 3)

        return torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

    def transform_point_cloud(self, point_cloud, translation, quaternion):
        rotated_points  = self.gs_transform_by_quat(point_cloud, quaternion)
        translated_points = rotated_points  + translation
        return translated_points

    def start_sim(self):
        gs.init(backend=gs.cpu if self.device is not None else gs.gpu)
        self.set_scene()
        if self.robot_addition:
            raise NotImplementedError("robot loading is not vendored")
