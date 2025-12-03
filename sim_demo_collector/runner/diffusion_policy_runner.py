import dill
import tqdm
import hydra
import torch
import numpy as np
import gymnasium as gym
import torchvision.transforms.functional as F
from typing import Optional, Dict
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from sim_demo_collector.util.transform_util import RotationTransformer


class DiffusionPolicyRunner:

    def __init__(
        self,
        env: gym.Env,
        policy: BaseImagePolicy,
        rotation_transformer: RotationTransformer,
        shape_meta: Dict,
        max_episode_steps: Optional[int] = None,
        num_episodes: Optional[int] = 50,
        device: Optional[str] = "cpu"
    ) -> None:
        self.env = env
        self.policy = policy
        self.rotation_transformer = rotation_transformer
        self.shape_meta = shape_meta
        self.max_episode_steps = max_episode_steps
        self.num_episodes = num_episodes
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.policy.eval()
    
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
    
    def run(self, checkpoint: Optional[str] = None, init_seed: Optional[int] = None) -> None:
        if checkpoint is not None:
            payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
            cfg = payload['cfg']
            cls = hydra.utils.get_class(cfg._target_)
            workspace: BaseWorkspace = cls(cfg)
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            # Get policy from workspace
            self.policy = workspace.model
            if cfg.training.use_ema:
                self.policy = workspace.ema_model
            self.policy.to(self.device)
            self.policy.eval()

        seed_increment = 0
        success_cnt = 0
        while True:
            seed = None
            if init_seed is not None:
                seed = init_seed + seed_increment
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
                    action = self.policy.predict_action(obs)['action']
                action = action.detach().cpu().squeeze(0).numpy()
                if action.shape[1] == 10:
                    # When using absolute actions, diffusion policy returns 6d rotations.
                    # Here, we need to transform 6d rotations into the rotation type that
                    # is accepted by the environment.
                    pos = action[..., :3]
                    rot = action[..., 3:9]
                    gripper = action[..., [-1]]
                    rot = self.rotation_transformer.forward(rot)
                    action = np.concatenate([pos, rot, gripper], axis=-1)
                obs, _, done, info = self.env.step(action)
                self.env.render()
                done = np.all(done)
                is_success = np.any(info['is_success'])
                if is_success:
                    success_cnt += 1
                pbar.update(action.shape[1])
            pbar.close()
            seed_increment += 1
            if success_cnt == self.num_episodes:
                break
        # Clear out data buffer
        self.env.reset()