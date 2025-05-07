from collections import deque
import gym
import numpy as np
import cv2

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    NoopResetEnv)


class CustomFrameStack(gym.Wrapper):
    def __init__(self, env, num_stack=4, frame_skip=2):
        super(CustomFrameStack, self).__init__(env)
        self.num_stack = num_stack
        self.frame_skip = frame_skip
        self.frames = deque(maxlen=num_stack)
        self.death_penalty = -1.0
        self.lives = None

        # Úprava observation space: 3 kanály pre RGB a 3 kanály pre rozdiely v grayscale
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(w, h, 4), dtype=np.uint8)

    def reset(self, seed, options):
        obs, _ = self.env.reset()
        # Inicializácia frames deque s prvým frame-om (RGB)
        for _ in range(self.num_stack):
            self.frames.append(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY))

        self.lives = 3

        return self._get_stacked_obs(), {}

    def step(self, action):
        total_reward = 0
        for _ in range(self.frame_skip):
            # Vykonaj akciu na preskočenie frameov a pridaj reward
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward

            self.frames.append(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY))

            if done or truncated:
                break

            if done or truncated:
                break

        # Reward processing normujeme na male cislo
        if (total_reward > 0): total_reward = 1
        if (total_reward > 100): total_reward = 10       # Bonus za mothership!

        # Mensi penalty za nic nerobenie
        #total_reward = total_reward - 0.01

        return self._get_stacked_obs(), total_reward, done, truncated, info

    def _get_stacked_obs(self):
        # Posledný frame
        last_frame = self.frames[-1]

        # Konštantná šedá hodnota (napr. 128 pre strednú šedú)
        gray_base = 128

        grayscale_diffs = []
        grayscale_diffs.append(last_frame[..., np.newaxis])
    
        compare_frame = last_frame
        for frame in reversed(list(self.frames)[:-1]):  # Vynecháme posledný frame
            # Rozdiel oproti poslednému frame, kde sa zachovajú pozitívne aj negatívne zmeny
            diff = compare_frame.astype(np.int16) - frame.astype(np.int16)
            diff = np.clip(diff + gray_base, 0, 255).astype(np.uint8)  # Uisti sa, že hodnoty sú v rozsahu 0-255

            grayscale_diffs.append(diff[..., np.newaxis])  # Pridaj novú dimenziu pre channel

            compare_frame = frame

        # Stack všetky rozdielové frames a posledný frame do jedného pozorovania
        stacked_obs = np.concatenate(grayscale_diffs, axis=-1)
        stacked_obs = np.transpose(stacked_obs, (1, 0, 2))

        return stacked_obs


# Použitie tohto wrapperu vo funkcii na vytváranie prostredia
def make_atari_env(rank):
    def _init():
        env = gym.make("SpaceInvaders-v4")
        env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
        env = gym.wrappers.ResizeObservation(env, (224, 224))
        env = CustomFrameStack(env)  # Nastav wrapper, ktorý prispôsobí pozorovanie
        env.seed(rank)
        return env
    return _init