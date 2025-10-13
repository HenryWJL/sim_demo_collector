import zarr
import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, List


class DataRecordingWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        obs_keys: List,
        zarr_path: str
    ) -> None:
        super().__init__(env)
        self.obs_keys = obs_keys
        # Create Zarr datasets
        self.root = zarr.open(zarr_path, mode="a")
        for key in self.obs_keys:
            shape = self.observation_space[key].shape
            dtype = np.float32
            if key.endswith("image"):
                dtype = np.uint8
            elif key.endswith("mask"):
                dtype = bool
            self.root.create_dataset(
                name=key,
                shape=(0, *shape),
                chunks=(10, *shape),
                dtype=dtype
            )
        self.root.create_dataset(
            name='action',
            shape=(0, *self.action_space.shape),
            chunks=(10, *self.action_space.shape),
            dtype=np.float32
        )
        self.root.create_dataset(
            name='meta/episode_ends',
            shape=(0,),
            chunks=(10,),
            dtype=np.int64
        )

        self._is_done = False
        self._data_buffer = {key: [] for key in self.obs_keys + ['action']}

    def reset(self, *args, **kwargs) -> Dict:
        if self._is_done:
            # Remove the last observations to align with action sequences
            for key in self.obs_keys:
                self._data_buffer[key].pop()
            self._save()
        self._clear_buffer()
        self._is_done = False
        obs = super().reset(*args, **kwargs)
        for key in self.obs_keys:
            self._data_buffer[key].append(obs[key])
        return obs
    
    def step(self, action: np.ndarray) -> Tuple:
        self._data_buffer['action'].append(action)
        outputs = super().step(action)
        obs = outputs[0]
        for key in self.obs_keys:
            self._data_buffer[key].append(obs[key])
        self._is_done = outputs[2]
        return outputs

    def render(self) -> None:
        return super().render()

    def _save(self) -> None:
        for key in self.obs_keys + ['action']:
            data = self.root[key]
            new_data = np.stack(self._data_buffer[key])
            start = data.shape[0]
            end = start + new_data.shape[0]
            data.resize(end, *data.shape[1:])
            data[start: end] = new_data

        episode_ends = self.root['meta/episode_ends']
        episode_len = np.array(len(self._data_buffer['action']))
        last_end = episode_ends[-1] if episode_ends.shape[0] > 0 else 0
        new_end = last_end + episode_len
        episode_ends.resize(episode_ends.shape[0] + 1)
        episode_ends[-1] = new_end
        
    def _clear_buffer(self) -> None:
        for key in self._data_buffer.keys():
            self._data_buffer[key].clear()


def test():
    from sim_demo_collector.env.robosuite_env import RobosuiteEnv3D

    camera_names = ["frontview", "robot0_eye_in_hand"]

    env = RobosuiteEnv3D(
        env_name="Lift",
        robots="Panda",
        camera_names=camera_names,
        use_object_obs=True,
        use_image_obs=True,
        use_depth_obs=True,
        use_pc_obs=True,
        use_mask_obs=True,
    )
    env = DataRecordingWrapper(
        env,
        obs_keys=[
            'frontview_image',
            'robot0_eye_in_hand_image_mask',
            'frontview_pc',
            'robot0_eef_pos'
        ],
        zarr_path="test.zarr"
    )
    env.reset()
    for i in range(10):
        obs, rew, done, info = env.step(env.action_space.sample())
    env._is_done = True
    env.reset()
    for i in range(15):
        obs, rew, done, info = env.step(env.action_space.sample())
    env._is_done = True
    env.reset()