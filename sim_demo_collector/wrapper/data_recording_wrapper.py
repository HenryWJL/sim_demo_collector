import zarr
import numpy as np
import gymnasium as gym
from typing import Tuple, Dict


class DataRecordingWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        shape_meta: Dict,
        zarr_path: str
    ) -> None:
        super().__init__(env)
        self.shape_meta = shape_meta
        self.obs_keys = list(self.shape_meta['obs'].keys())
        # Create Zarr datasets
        self.root = zarr.open(zarr_path, mode="a")
        for key in self.obs_keys:
            shape = self.shape_meta['obs'][key]
            dtype = np.float32
            if key.endswith("image"):
                dtype = np.uint8
            elif key.endswith("mask"):
                dtype = bool
            self.root.create_dataset(
                key,
                shape=(0, *shape),
                chunks=(10, *shape),
                dtype=dtype
            )
        self.root.create_dataset(
            'action',
            shape=(0, *self.shape_meta['action']),
            chunks=(10, *self.shape_meta['action']),
            dtype=np.float32
        )
        self.is_done = False
        self.data_buffer = {key: [] for key in self.obs_keys + ['action']}

    def reset(self, *args, **kwargs) -> Dict:
        if self.is_done:
            # Remove the last observations to align with action sequences
            for key in self.obs_keys:
                self.data_buffer[key].pop()
            self.save()
            self.clear_buffer()
        obs = super().reset(*args, **kwargs)
        for key in self.obs_keys:
            self.data_buffer[key].append(obs[key])
        return obs
    
    def step(self, action: np.ndarray) -> Tuple:
        self.data_buffer['action'].append(action)
        outputs = super().step(action)
        obs = outputs[0]
        for key in self.obs_keys:
            self.data_buffer[key].append(obs[key])
        self.is_done = outputs[2]
        return outputs

    def save(self) -> None:
        for key in self.obs_keys + ['action']:
            dataset = self.root[key]
            data = np.stack(self.data_buffer[key])
            start = dataset.shape[0]
            end = start + data.shape[0]
            dataset.resize(end, *data.shape[1:])
            dataset[start: end] = data
        
    def clear_buffer(self) -> None:
        for key in self.data_buffer.keys():
            self.data_buffer[key].clear()


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
        use_pcd_obs=True,
        use_mask_obs=True,
    )
    env = DataRecordingWrapper(
        env,
        {
            'obs': {
                'frontview_image': [84, 84, 3],
                'robot0_eye_in_hand_image_mask': [84, 84, 1],
                "robot0_eef_pos": [3]
            },
            'action': [7]
        },
        "test.zarr"
    )
    obs = env.reset()
    for i in range(5):
        obs, rew, done, info = env.step(np.random.rand(7))
    env.is_done = True
    env.reset()

if __name__ == "__main__":
    test()