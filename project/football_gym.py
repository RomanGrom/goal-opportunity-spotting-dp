import gfootball.env as football_env
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv)

import cv2


class FootballGym(gym.Env):
    spec = None
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config=None, frame_skip=1):
        super(FootballGym, self).__init__()
        #env_name = 'academy_single_goal_versus_lazy'
        env_name = '11_vs_11_hard_stochastic'
        #env_name = 'academy_pass_and_shoot_with_keeper'
        rewards = "scoring"

        if config is not None:
            env_name = config.get("env_name", env_name)
            rewards = config.get("rewards", rewards)

        self.env = football_env.create_environment(
            env_name=env_name,
            stacked=True,
            representation="pixels",
            rewards=rewards,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=True,
            write_video=False,
            dump_frequency=1,
            logdir=".",
            extra_players=None,
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0,
            channel_dimensions=[224, 224 , 3],
            other_config_options={"action_set": "v2"} )
        
        self.action_space = gym.spaces.discrete.Discrete(19)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(6, 224, 224), dtype=np.uint8)

        self.frame_skip = frame_skip
        self.reward_range = (-1, 1)
        self.n_step = 0

    def reset(self, seed=None):
        obs = self.env.reset()
        self.n_step = 0
        obs = np.moveaxis(obs, -1, 0)

        return self._get_stacked_obs(obs), {}
    
    def step(self, action):
        self.n_step += self.frame_skip

        total_reward = 0
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step([19])
            total_reward += reward

            if done:
                break

        obs = np.moveaxis(obs, -1, 0)

        if self.n_step >= 1500:
            done = True  # end episode when first half ends

        terminated = done
        truncated = False
        
        return self._get_stacked_obs(obs), float(total_reward), terminated, truncated, info

    
    def render(self, mode='rgb_array'):
        print("BEziiim")
        return self.env.render(mode)
    
    def _get_stacked_obs(self, obs):

        # Last frame to RGB channels
        r_last = obs[9]
        g_last = obs[10]
        b_last = obs[11]

        # Base grey
        gray_base = 128

        grayscale_diffs = []

        # Last frame to grayscale diffs
        grayscale_diffs.append(r_last[..., np.newaxis])  # R kanál
        grayscale_diffs.append(g_last[..., np.newaxis])  # G kanál
        grayscale_diffs.append(b_last[..., np.newaxis])  # B kanál
        
        # Last frames differences
        compare_frame = np.stack((r_last, g_last, b_last), axis=-1)
        compare_frame = cv2.cvtColor(compare_frame, cv2.COLOR_RGB2GRAY)

        for i in range(6, -1, -3):
            frame = np.stack((obs[i], obs[i+1], obs[i+2]), axis=-1)
            # grayscale
            frame  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            diff = compare_frame.astype(np.int16) - frame.astype(np.int16)
            diff = np.clip(diff + gray_base, 0, 255).astype(np.uint8)

            # add difference
            grayscale_diffs.append(diff[..., np.newaxis])

            compare_frame = frame

        # stack all frames into one observattion
        stacked_obs = np.concatenate(grayscale_diffs, axis=-1)
        stacked_obs = np.transpose(stacked_obs, (2, 0, 1))
        
        return stacked_obs




# Create environment
def make_football_env(rank):
    def _init():
        env = Monitor(FootballGym())

        return env
    return _init





