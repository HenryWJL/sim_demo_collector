import zarr
import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict, List


class DataRecordingWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        shape_meta: Dict,
        zarr_path: str
    ) -> None:
        super().__init__(env)
        self.obs_keys = list(shape_meta['obs'].keys())
        # Create Zarr datasets
        self.root = zarr.open(zarr_path, mode="a")
        for key in self.obs_keys:
            shape = shape_meta['obs'][key]
            dtype = np.float32
            if key.endswith("image"):
                dtype = np.uint8
            elif key.endswith("mask"):
                dtype = bool
            self.root.create_dataset(
                key,
                shape=(0, *shape),
                chunks=(10, *shape),
                dtype=dtype,
                maxshape=(None, *shape)
            )
        self.root.create_dataset(
            'action',
            shape=(0, *shape_meta['action']),
            chunks=(10, *shape_meta['action']),
            dtype=np.float32,
            maxshape=(None, *shape_meta['action'])
        )
        self.is_done = False
        self.data_buffer = {key: [] for key in self.obs_keys + ['action']}

    def reset(self, *args, **kwargs) -> Dict:
        if self.is_done:
            self.save()
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
        old_n = img_ds.shape[0]
        new_n = old_n + image_data.shape[0]

        # Resize and append
        img_ds.resize(new_n, axis=0)
        img_ds[old_n:new_n] = image_data
        
    def clear(self) -> None:
        for key in self.data_buffer.keys():
            self.data_buffer[key].clear()