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


class RobosuiteEnv:

    CAMERA_NAMES = ["agentview", "frontview", "robot0_eye_in_hand"]

    def __init__(
        self,
        env_name: str,
        robots: Union[str, List[str]],
        use_object_obs: Optional[bool] = False,
        use_image_obs: Optional[bool] = False,
        use_depth_obs: Optional[bool] = False,
        use_mask_obs: Optional[bool] = False,
        image_size: Optional[Tuple[int, int]] = (84, 84),
        controller: Optional[str] = "OSC_POSE",
        delta_action: Optional[bool] = False,
        control_freq: Optional[int] = 20,
        render_mode: Literal["human", "rgb_array"] = "human",
        render_camera: Optional[str] = "agentview",
        render_image_size: Optional[Tuple[int, int]] = (256, 256)
    ) -> None:
        """
        A gym wrapper for Robosuite.

        Args:
            env_name (str): Robosuite environment (see official documentation).
            robots (str or list): Robot name(s) (see official documentation).
            use_object_obs (bool): If True, returns object states.
            use_image_obs (bool): If True, returns camera observations.
            use_depth_obs (bool): If True, returns depth observations.
            use_mask_obs (bool): If True, returns binary robot masks.
            image_size (tuple): height and width of returned images.
            controller (str): Controller type (see official documentation).
            delta_action (bool): If True, uses delta action control.
            control_freq (int): control frequency.
            render_mode (str): "human" or "rgb_array". If "human", opens a
                real-time onscreen viewer; otherwise, returns a NumPy array.
        """
        env_kwargs = dict(
            env_name=env_name,
            robots=robots,
            camera_names=self.CAMERA_NAMES,
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
                obs[key.replace("segmentation_element", "mask")] = seg_mask
            else:
                obs[key] = raw_obs[key].copy()
        return obs

    def reset(self) -> Dict:
        obs = self.env.reset()
        return self._extract_obs(obs)
    
    def step(self, action: np.ndarray) -> Tuple:
        obs, reward, _, info = self.env.step(action)
        return self._extract_obs(obs), reward, False, info

    def render(self) -> None:
        """Render from simulation to either an on-screen window or off-screen to RGB array."""
        if self.render_mode == "human":
            cam_id = self.env.sim.model.camera_name2id(self.render_camera)
            self.env.viewer.set_camera(cam_id)
            return self.env.render()
        elif self.render_mode == "rgb_array":
            image = self.env.sim.render(*self.render_image_size, self.render_camera)
            return image[::-1]
    
    def get_camera_intrinsic_matrix(self, camera_name: str) -> np.ndarray:
        """Return the intrinsic matrix of @camera_name."""
        return get_camera_intrinsic_matrix(
            self.env.sim, camera_name, *self.image_size
        )
    
    def get_camera_extrinsic_matrix(self, camera_name: str) -> np.ndarray:
        """Return the extrinsic matrix of @camera_name in the world frame."""
        return get_camera_extrinsic_matrix(self.env.sim, camera_name)
    
    def is_success(self) -> bool:
        """Check if the task is successfully done"""
        return self.env._check_success()