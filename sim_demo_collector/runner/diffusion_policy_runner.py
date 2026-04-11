import dill
import tqdm
import hydra
import torch
import numpy as np
import gymnasium as gym
import torchvision.transforms.functional as F
from typing import Optional, Dict
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class DiffusionPolicyRunner:

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: Optional[int] = None,
        num_episodes: Optional[int] = 50,
        device: Optional[str] = "cpu"
    ) -> None:
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.num_episodes = num_episodes
        self.device = torch.device(device)
    
    def _extract_obs(self, raw_obs: Dict) -> Dict:
        """
        Extract observations required for policy rollout from gym environments.
        """
        obs = dict()
        for key in self.shape_meta['obs'].keys():
            if key.endswith("image"):
                image = torch.from_numpy(raw_obs[key]).permute(0, 3, 1, 2) / 255.0
                if list(image.shape[2:]) != list(self.shape_meta['obs'][key]['shape'][1:]):
                    image = F.resize(image, list(self.shape_meta['obs'][key]['shape'][1:]))
                obs[key] = image.unsqueeze(0).to(self.device)
            else:
                obs[key] = torch.from_numpy(raw_obs[key]).float().unsqueeze(0).to(self.device)
        return obs
    
    def run(self, checkpoint: str, init_seed: Optional[int] = None) -> None:
        # Load checkpoints
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        # Get shape information
        self.shape_meta = cfg.shape_meta
        # Get policy from workspace
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        if cfg.training.use_ema:
            policy = workspace.ema_model
        else:
            policy = workspace.model
        policy.to(self.device)
        policy.eval()

        seed = init_seed
        success_cnt = 0
        while True:
            obs = self.env.reset(seed=seed)
            self.env.render()
            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Collecting episode [{success_cnt + 1}/{self.num_episodes}]", 
                leave=False,
                mininterval=5.0
            )
            done = False
            while not done:
                obs = self._extract_obs(obs)
                with torch.no_grad():
                    action = policy.predict_action(obs)['action']
                action = action.detach().cpu().squeeze(0).numpy()
                obs, _, done, info = self.env.step(action)
                self.env.render()
                done = np.all(done)
                is_success = np.any(info['is_success'])
                if is_success:
                    success_cnt += 1
                pbar.update(action.shape[1])
            pbar.close()
            if success_cnt == self.num_episodes:
                break
            if seed is not None:
                seed += 1
        # Clear out data buffer
        self.env.reset()