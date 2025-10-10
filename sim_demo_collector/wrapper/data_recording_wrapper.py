import zarr
import numpy as np
import gymnasium as gym
from pathlib import Path
from typing import Optional, Tuple, Dict, List


class DataRecordingWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        obs_keys: List,
        save_dir: str
    ) -> None:
        super().__init__(env)
        self.global_step = 0
        self.is_done = False
        self.data = {key: [] for key in obs_keys}
        self.obs_keys = obs_keys
        self.save_dir = Path(save_dir).expanduser().absolute()
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, *args, **kwargs) -> Dict:
        if self.is_done:

        obs = super().reset(*args, **kwargs)
        self.global_step += 1
        self.data.append()
        return obs
    
    def step(self, action: np.ndarray) -> Tuple:
        outputs = super().step(action)
        if self.record_videos:
            frame = self.env.render()
            assert frame.dtype == np.uint8
            self.video_recoder.append_data(frame)
        return outputs

    def save() -> None:
        with zarr
        
    def close(self) -> None:
        super().close()