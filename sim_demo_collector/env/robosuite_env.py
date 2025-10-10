"""
Reference code:
https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/wrappers/gym_wrapper.py
https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/envs/env_robosuite.py
https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_image_wrapper.py
"""
import fpsample
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Most APIs between gym and gymnasium are compatible
    print("WARNING! gymnasium is not installed. We will try to use openai gym instead.")
    import gym
    from gym import spaces
    if not gym.__version__ >= "0.26.0":
        # Due to API Changes in gym>=0.26.0, we need to ensure that the version is correct
        # Please check: https://github.com/openai/gym/releases/tag/0.26.0
        raise ImportError("Please ensure version of gym>=0.26.0 to use the GymWrapper.")
    
import robosuite as suite
from typing import Optional, Union, Tuple, Literal, List, Dict
from robosuite.controllers import load_controller_config
from robosuite.utils.camera_utils import (
    get_real_depth_map,
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix
)
from sim_demo_collector.util.pc_util import pc_normalize, depth2pc


def get_box_space(sample: np.ndarray) -> spaces.Box:
    if np.issubdtype(sample.dtype, np.integer):
        low = np.iinfo(sample.dtype).min
        high = np.iinfo(sample.dtype).max
    elif np.issubdtype(sample.dtype, np.inexact):
        low = float("-inf")
        high = float("inf")
    else:
        raise ValueError()
    return spaces.Box(low=low, high=high, shape=sample.shape, dtype=sample.dtype)


class RobosuiteEnv(gym.Env):

    def __init__(
        self,
        env_name: str,
        robots: Union[str, List[str]],
        camera_names: Union[str, List[str]],
        use_object_obs: Optional[bool] = False,
        use_image_obs: Optional[bool] = False,
        use_depth_obs: Optional[bool] = False,
        use_mask_obs: Optional[bool] = False,
        flatten_obs: Optional[bool] = False,
        image_size: Optional[Tuple[int, int]] = (84, 84),
        controller: Optional[str] = "OSC_POSE",
        delta_action: Optional[bool] = False,
        control_freq: Optional[int] = 20,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        render_camera: Optional[str] = "agentview",
        render_image_size: Optional[Tuple[int, int]] = (256, 256)
    ) -> None:
        """
        Gym wrapper class for Robosuite (https://robosuite.ai/docs/overview.html).

        Args:
            env_name (str): Robosuite environment (https://robosuite.ai/docs/modules/environments.html).
            robots (str or list): Robot name(s) (https://robosuite.ai/docs/modules/robots.html).
            use_object_obs (bool): If True, returns object states.
            use_image_obs (bool): If True, returns camera observations.
            use_depth_obs (bool): If True, returns depth observations.
            use_mask_obs (bool): If True, returns binary robot masks.
            flatten_obs (bool): If True, returns flattened observations (1D arrays).
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
        self.camera_names = [camera_names] if isinstance(camera_names, str) else camera_names
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
        # Create environments
        self.env = suite.make(**env_kwargs)
        # Set up observation and action spaces
        obs = self._extract_obs(self.env.reset())
        self.flatten_obs = flatten_obs
        if self.flatten_obs:
            flat_obs = self._flatten_obs(obs)
            obs_shape = flat_obs.shape
            high = np.inf * np.ones(obs_shape)
            low = -high
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                shape=obs_shape
            )
        else:
            self.observation_space = spaces.Dict({
                key: get_box_space(obs[key]) for key in obs.keys()
            })
        low, high = self.env.action_spec
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )

        self.seed_state_map = {}
        self.task_completion_hold_count = -1

    def _flatten_obs(self, obs_dict: Dict) -> np.ndarray:
        """
        Flatten observations to 1D arrays and concatenate.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations.

        Returns:
            obs (np.ndarray): observations flattened into a 1D array.
        """
        obs_list = []
        for obs in obs_dict.values():
            obs_list.append(np.array(obs).flatten())
        obs = np.concatenate(obs_list, dtype=np.float32)
        return obs

    def _extract_obs(self, raw_obs: Dict) -> Dict:
        obs = {}
        for key in raw_obs.keys():
            if key.endswith("image"):
                # By default MuJoCo returns vertically flipped images
                obs[key] = raw_obs[key][::-1].copy()
            elif key.endswith("depth"):
                # By default MuJoCo returns vertically flipped images
                # By default MuJoCo returns normalized depth values
                obs[key] = get_real_depth_map(self.env.sim, raw_obs[key][::-1].copy()).astype(np.float64)
            elif key.endswith("segmentation_element"):
                # By default MuJoCo returns vertically flipped images
                seg_mask = self._extract_seg_mask(raw_obs[key][::-1].copy())
                obs[key.replace("segmentation_element", "image_mask")] = seg_mask
            else:
                obs[key] = raw_obs[key].copy()
        return obs
    
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

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Dict:
        self.task_completion_hold_count = -1
        init_state = None
        if options is not None:
            init_state = options.get('init_state')
        if init_state is not None:
            raw_obs = self.reset_to({'states': init_state})
        elif seed is not None:
            if seed in self.seed_state_map:
                raw_obs = self.reset_to({'states': self.seed_state_map[seed]})
            else:
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.sim.get_state()
                self.seed_state_map[seed] = state
        else:
            raw_obs = self.env.reset()
        obs = self._extract_obs(raw_obs)
        if self.flatten_obs:
            obs = self._flatten_obs(obs)
        return obs
    
    def reset_to(self, state: Dict) -> Dict:
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment.
                - model (str): mujoco scene xml.
        
        Returns:
            raw_obs (dict): observation dictionary after setting the simulator
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
            self.env.sim.set_state(state["states"])
            self.env.sim.forward()
            should_ret = True
        if "goal" in state:
            if hasattr(self.env, "set_goal"):
                self.env.set_goal(**state["goal"])
            else:
                print("Warning: Environment does not support goal setting.")
        if should_ret:
            raw_obs = self.env._get_observations(force_update=True)
            return raw_obs
        return None

    def step(self, action: np.ndarray) -> Tuple:
        raw_obs, reward, _, info = self.env.step(action)
        obs = self._extract_obs(raw_obs)
        if self.flatten_obs:
            obs = self._flatten_obs(obs)
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
        
        return obs, reward, done, info

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
        use_pc_obs: Optional[bool] = False,
        num_points: Optional[int] = 512,
        bounding_boxes: Optional[Dict] = dict(),
        normalize_pc: Optional[bool] = True,
        **kwargs
    ) -> None:
        """
        Args:
            use_pc_obs (bool): If True, returns point cloud observations.
            num_points (int): Number of points in the point clouds.
            bounding_boxes (dict): Per-camera bounding boxes of the point clouds.
            normalize_pc (bool): If True, returns normalized point cloud observations.
        """
        if use_pc_obs:
            assert kwargs.get('use_image_obs') and kwargs.get('use_depth_obs')
        self.use_pc_obs = use_pc_obs
        self.num_points = num_points
        self.bounding_boxes = bounding_boxes
        self.normalize_pc = normalize_pc
        super().__init__(**kwargs)
        
    def _extract_obs(self, raw_obs: Dict) -> Dict:
        obs = super()._extract_obs(raw_obs)
        if self.use_pc_obs:
            for camera_name in self.camera_names:
                # By default the camera intrinsic matrix computed from
                # MuJoCo’s camera parameters already assumes image flip.
                cam_intrin_mat = super().get_camera_intrinsic_matrix(camera_name)
                depth = obs[f'{camera_name}_depth'][::-1].copy()
                seg_mask = obs.get(f'{camera_name}_image_mask')
                if seg_mask is not None:
                    seg_mask = seg_mask[::-1].copy()
                pc, seg_mask = depth2pc(
                    depth=depth,
                    camera_intrinsic_matrix=cam_intrin_mat,
                    bounding_box=self.bounding_boxes.get(camera_name),
                    seg_mask=seg_mask
                )
                fps_idx = fpsample.bucket_fps_kdline_sampling(pc, self.num_points, h=3)
                pc = pc[fps_idx]
                if self.normalize_pc:
                    pc = pc_normalize(pc)
                obs[f'{camera_name}_pc'] = pc
                if seg_mask is not None:
                    obs[f'{camera_name}_pc_mask'] = seg_mask[fps_idx]
        return obs
    

def test():
    import matplotlib.pyplot as plt
    import open3d as o3d

    camera_name = "frontview"

    env = RobosuiteEnv3D(
        env_name="Lift",
        robots="Panda",
        camera_names=camera_name,
        use_object_obs=True,
        use_image_obs=True,
        use_depth_obs=True,
        use_pc_obs=True,
        use_mask_obs=True,
    )
    obs = env.reset(seed=42)
    env.render()
    for _ in range(10):
        env.step(env.action_space.sample())
        env.render()
    obs = env.reset(seed=42)
    env.render()

    image = obs[f'{camera_name}_image']
    depth = obs[f'{camera_name}_depth']
    image_mask = obs[f'{camera_name}_image_mask']
    pc = obs[f'{camera_name}_pc']
    pc_mask = obs[f'{camera_name}_pc_mask']

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
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc)
    colors = np.zeros_like(pc)
    colors[pc_mask == 1] = [1, 0, 0]   # red = robot
    colors[pc_mask == 0] = [0, 1, 0]   # green = environment
    pc_o3d.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pc_o3d])