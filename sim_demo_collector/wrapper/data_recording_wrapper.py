import imageio
import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict
from sim_demo_collector.util.file_util import str2path, mkdir


class DataRecordingWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        video_save_dir: Optional[str] = None
    ) -> None:
        super().__init__(env)
        self.global_step = 0
        if video_save_dir is None:
            self.record_videos = False
        else:
            self.record_videos = True
            self.video_save_dir = mkdir(str2path(video_save_dir), parents=True, exist_ok=True)
            self.frames = list()
            
    def reset(self, *args, **kwargs) -> Dict:
        obs = super().reset(*args, **kwargs)
        self.global_step += 1
        if self.record_videos:
            self.frames = list()
            self.video_recoder = imageio.get_writer(self.video_save_dir.joinpath(f"episode_{str(self.global_step).zfill(3)}.mp4"), fps=24)
        return obs
    
    def step(self, action: np.ndarray) -> Tuple:
        outputs = super().step(action)
        if self.record_videos:
            frame = self.env.render()
            assert frame.dtype == np.uint8
            self.video_recoder.append_data(frame)
        return outputs
    
    def close(self) -> None:
        super().close()