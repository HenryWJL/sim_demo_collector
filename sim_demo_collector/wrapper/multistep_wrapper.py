
"""
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Literal, Dict, Tuple, List, Any
from collections import defaultdict, deque


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)


def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )


def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method='max'):
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/gym_util/multistep_wrapper.py#L67
class MultiStepWrapper(gym.Wrapper):

    def __init__(
        self,     
        env: gym.Env, 
        obs_horizon: int, 
        action_horizon: int, 
        max_episode_steps: Optional[int] = None,
        reward_agg_method: Literal['max', 'min', 'mean', 'sum'] = 'max',
        enable_temporal_ensemble: Optional[bool] = True
    ) -> None:
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, action_horizon)
        self._observation_space = repeated_space(env.observation_space, obs_horizon)
        self.obs = deque(maxlen=obs_horizon + 1)
        if enable_temporal_ensemble:
            assert max_episode_steps is not None
            T = max_episode_steps
            H = action_horizon
            D = int(env.action_space.shape[0])
            self.all_time_actions = np.zeros((T, T + H, D), dtype=np.float32)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=obs_horizon + 1))
        self.global_step = 0
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.max_episode_steps = max_episode_steps
        self.reward_agg_method = reward_agg_method
        self.enable_temporal_ensemble = enable_temporal_ensemble

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs = super().reset(seed=seed, options=options)
        self.obs = deque([obs], maxlen=self.obs_horizon + 1)
        if self.enable_temporal_ensemble:
            self.all_time_actions = np.zeros_like(self.all_time_actions, dtype=np.float32)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.obs_horizon + 1))
        self.global_step = 0
        obs = self._get_obs(self.obs_horizon)
        return obs

    def step(self, actions: np.ndarray) -> Tuple:
        """
        Args:
            actions (np.ndarray): (action_horizon, action_shape).
        """
        if self.enable_temporal_ensemble:
            t = self.global_step
            H = self.action_horizon
            self.all_time_actions[[t], t: t + H] = actions
            current_actions = self.all_time_actions[:, t]
            actions_populated = np.all(current_actions != 0, axis=1)
            current_actions = current_actions[actions_populated]
            exp_weights = np.exp(-0.01 * np.arange(len(current_actions)))
            exp_weights = (exp_weights / exp_weights.sum())[..., np.newaxis]
            action = (current_actions * exp_weights).sum(axis=0)
            obs, reward, done, info = super().step(action)
            self.obs.append(obs)
            self.reward.append(reward)
            if len(self.reward) >= self.max_episode_steps:
                done = True
            self.done.append(done)
            self._add_info(info)
            self.global_step += 1
        else:
            for action in actions:
                if len(self.done) > 0 and self.done[-1]:
                    break
                obs, reward, done, info = super().step(action)
                self.obs.append(obs)
                self.reward.append(reward)
                if (self.max_episode_steps is not None) and (len(self.reward) >= self.max_episode_steps):
                    done = True
                self.done.append(done)
                self._add_info(info)

        obs = self._get_obs(self.obs_horizon)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.obs_horizon)
        return obs, reward, done, info

    def _get_obs(self, n_steps: Optional[int] = 1) -> Dict:
        """
        Returns:
            obs (dict): each element with shape (obs_horizon, ...).
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info: Dict) -> None:
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self) -> List:
        return self.reward
    
    def get_attr(self, name: str) -> Any:
        return getattr(self, name)
    
    def get_infos(self) -> Dict:
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
    
    def close(self) -> None:
        super().close()
