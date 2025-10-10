import fpsample
import numpy as np
import robosuite as suite
from typing import Optional, Union, Tuple, Literal, List, Dict
from robosuite.controllers import load_controller_config
from robosuite.utils.camera_utils import (
    get_real_depth_map,
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix
)
from sim_demo_collector.util.pcd_util import depth2pcd

# Adapted from https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/envs/env_robosuite.py
class RobosuiteEnv:

    def __init__(
        self,
        env_name: str,
        robots: Union[str, List[str]],
        camera_names: Union[str, List[str]],
        use_object_obs: Optional[bool] = False,
        use_image_obs: Optional[bool] = False,
        use_depth_obs: Optional[bool] = False,
        use_mask_obs: Optional[bool] = False,
        image_size: Optional[Tuple[int, int]] = (84, 84),
        controller: Optional[str] = "OSC_POSE",
        delta_action: Optional[bool] = False,
        control_freq: Optional[int] = 20,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        render_camera: Optional[str] = "agentview",
        render_image_size: Optional[Tuple[int, int]] = (256, 256)
    ) -> None:
        """
        Wrapper class for Robosuite (https://robosuite.ai/docs/overview.html).

        Args:
            env_name (str): Robosuite environment (https://robosuite.ai/docs/modules/environments.html).
            robots (str or list): Robot name(s) (https://robosuite.ai/docs/modules/robots.html).
            use_object_obs (bool): If True, returns object states.
            use_image_obs (bool): If True, returns camera observations.
            use_depth_obs (bool): If True, returns depth observations.
            use_mask_obs (bool): If True, returns binary robot masks.
            image_size (tuple): height and width of returned images.
            controller (str): Controller type (https://robosuite.ai/docs/modules/controllers.html).
            delta_action (bool): If True, uses delta action control.
            control_freq (int): control frequency.
            render_mode (str): "human" or "rgb_array". If "human", opens a
                real-time onscreen viewer; otherwise, returns a NumPy array.
        """
        env_kwargs = dict(
            env_name=env_name,
            robots=robots,
            camera_names=camera_names,
            use_object_obs=use_object_obs,
            use_camera_obs=use_image_obs,
            camera_depths=use_depth_obs,
            camera_heights=image_size[0],
            camera_widths=image_size[1],
            control_freq=control_freq,
            ignore_done=True
        )
        # ==================================================================== #
        # ======================= Observation Settings ======================= #
        # ==================================================================== #
        self.image_size = image_size
        if use_depth_obs:
            assert use_image_obs, "Must set @use_image_obs = True"
        if use_mask_obs:
            env_kwargs['camera_segmentations'] = "element"
        # =================================================================== #
        # ======================= Controller Settings ======================= #
        # =================================================================== #
        controller_config = load_controller_config(default_controller=controller)
        controller_config['control_delta'] = delta_action
        env_kwargs['controller_configs'] = controller_config
        # ================================================================= #
        # ======================= Renderer Settings ======================= #
        # ================================================================= #
        self.render_mode = render_mode
        self.render_camera = render_camera
        self.render_image_size = render_image_size
        env_kwargs['has_renderer'] = render_mode == "human"
        env_kwargs['has_offscreen_renderer'] = True if use_image_obs else False
        
        self.env = suite.make(**env_kwargs)
        self.task_completion_hold_count = -1

    def _extract_seg_mask(self, seg_image: np.ndarray) -> np.ndarray:
        """
        Extract robot segmentation masks.

        Returns:
            seg_mask (np.ndarray): binary segmentation masks where
                1 for pixels occupied by the robot and 0 otherwise.
        """
        arm_geom_names = self.env.robots[0].robot_model.visual_geoms
        gripper_geom_names = self.env.robots[0].gripper.visual_geoms
        robot_geom_names = arm_geom_names + gripper_geom_names
        robot_geom_ids = [self.env.sim.model.geom_name2id(n) for n in robot_geom_names]
        seg_mask = np.isin(seg_image, robot_geom_ids).astype(np.uint8)
        return seg_mask

    def _extract_obs(self, raw_obs: Dict) -> Dict:
        obs = {}
        for key in raw_obs.keys():
            if key.endswith("image"):
                # By default MuJoCo returns vertically flipped images
                obs[key] = raw_obs[key][::-1].copy()
            elif key.endswith("depth"):
                # By default MuJoCo returns vertically flipped images
                # By default MuJoCo returns normalized depth values
                obs[key] = get_real_depth_map(self.env.sim, raw_obs[key][::-1].copy())
            elif key.endswith("segmentation_element"):
                # By default MuJoCo returns vertically flipped images
                seg_mask = self._extract_seg_mask(raw_obs[key][::-1].copy())
                obs[key.replace("segmentation_element", "image_mask")] = seg_mask
            else:
                obs[key] = raw_obs[key].copy()
        return obs

    def reset(self) -> Dict:
        self.task_completion_hold_count = -1
        obs = self.env.reset()
        return self._extract_obs(obs)
    
    def reset_to(self, state: Dict) -> Dict:
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment.
                - model (str): mujoco scene xml.
        
        Returns:
            obs (dict): observation dictionary after setting the simulator
                state (only if "states" is in @state).
        """
        self.task_completion_hold_count = -1
        should_ret = False
        if "model" in state:
            self.reset(unset_ep_meta=False)
            xml = self.env.edit_model_xml(state["model"])
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
        if "states" in state:
            self.env.sim.set_state_from_flattened(state["states"])
            self.env.sim.forward()
            should_ret = True
        if "goal" in state:
            self.env.set_goal(**state["goal"])
        if should_ret:
            return self._extract_obs(self.env._get_observations(force_update=True))
        return None

    def step(self, action: np.ndarray) -> Tuple:
        obs, reward, _, info = self.env.step(action)
        # Task is done if having a success for 10 consecutive timesteps. Code is adapted from
        # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/collect_human_demonstrations.py
        if self.env._check_success():
            if self.task_completion_hold_count > 0:
                self.task_completion_hold_count -= 1
            else:
                self.task_completion_hold_count = 10
        else:
            self.task_completion_hold_count = -1
        done = not self.task_completion_hold_count
        
        return self._extract_obs(obs), reward, done, info

    def render(self) -> None:
        """Render from simulation to either an on-screen window or off-screen to RGB array."""
        if self.render_mode == "human":
            cam_id = self.env.sim.model.camera_name2id(self.render_camera)
            self.env.viewer.set_camera(cam_id)
            return self.env.render()
        elif self.render_mode == "rgb_array":
            image = self.env.sim.render(
                height=self.render_image_size[0],
                width=self.render_image_size[1],
                camera_name=self.render_camera
            )
            return image[::-1]

    def get_camera_intrinsic_matrix(self, camera_name: str) -> np.ndarray:
        """Return the intrinsic matrix of @camera_name."""
        return get_camera_intrinsic_matrix(
            self.env.sim, camera_name, *self.image_size
        )

    def get_camera_extrinsic_matrix(self, camera_name: str) -> np.ndarray:
        """Return the extrinsic matrix of @camera_name in the world frame."""
        return get_camera_extrinsic_matrix(self.env.sim, camera_name)


class RobosuiteEnv3D(RobosuiteEnv):

    def __init__(
        self,
        use_pcd_obs: Optional[bool] = False,
        num_points: Optional[int] = 512,
        bounding_boxes: Optional[Dict] = dict(),
        **kwargs
    ) -> None:
        """
        Args:
            use_pcd_obs (bool): If True, returns point cloud observations.
            bounding_boxes (dict): Per-camera bounding boxes of the point clouds.
        """
        if use_pcd_obs:
            assert kwargs.get('use_image_obs') and kwargs.get('use_depth_obs')
        super().__init__(**kwargs)
        self.use_pcd_obs = use_pcd_obs
        self.num_points = num_points
        self.bounding_boxes = bounding_boxes
        
    def _extract_obs(self, raw_obs: Dict) -> Dict:
        obs = super()._extract_obs(raw_obs)
        if self.use_pcd_obs:
            for camera_name in self.CAMERA_NAMES:
                # By default the camera intrinsic matrix computed from
                # MuJoCo’s camera parameters already assumes image flip.
                cam_intrin_mat = super().get_camera_intrinsic_matrix(camera_name)
                depth = obs[f'{camera_name}_depth'][::-1].copy()
                seg_mask = obs.get(f'{camera_name}_image_mask')
                if seg_mask is not None:
                    seg_mask = seg_mask[::-1].copy()
                point_cloud, seg_mask = depth2pcd(
                    depth=depth,
                    camera_intrinsic_matrix=cam_intrin_mat,
                    bounding_box=self.bounding_boxes.get(camera_name),
                    seg_mask=seg_mask
                )
                fps_idx = fpsample.bucket_fps_kdline_sampling(point_cloud, self.num_points, h=3)
                obs[f'{camera_name}_pcd'] = point_cloud[fps_idx]
                if seg_mask is not None:
                    obs[f'{camera_name}_pcd_mask'] = seg_mask[fps_idx]
        return obs
    

def test():
    import matplotlib.pyplot as plt
    import open3d as o3d

    env = RobosuiteEnv3D(
        env_name="Lift",
        robots="Panda",
        use_object_obs=True,
        use_image_obs=True,
        use_depth_obs=True,
        use_pcd_obs=True,
        use_mask_obs=True,
    )
    obs = env.reset()

    camera_name = "frontview"
    image = obs[f'{camera_name}_image']
    depth = obs[f'{camera_name}_depth']
    image_mask = obs[f'{camera_name}_image_mask']
    pcd = obs[f'{camera_name}_pcd']
    pcd_mask = obs[f'{camera_name}_pcd_mask']

    # Visualize images
    _, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(image)
    axs[0].set_title("RGB Image")
    axs[0].axis("off")

    axs[1].imshow(depth, cmap="plasma")
    axs[1].set_title("Depth Map")
    axs[1].axis("off")

    axs[2].imshow(image_mask, cmap="tab20")
    axs[2].set_title("Segmentation Mask")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

    # Visualize point clouds
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    colors = np.zeros_like(pcd)
    colors[pcd_mask == 1] = [1, 0, 0]   # red = robot
    colors[pcd_mask == 0] = [0, 1, 0]   # green = environment
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_o3d])