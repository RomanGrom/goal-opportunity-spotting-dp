import matplotlib.pyplot as plt
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from football_gym import make_football_env
import csv

# Create the environment
env = DummyVecEnv([lambda: make_football_env(0)()])

# Load the model
model = PPO.load(".scratch/logs/only_AI/4/last_model.zip", env=env)
model.policy.set_training_mode(False)

# open csv file for writing values
with open(".scratch/gfootball_eval.csv", mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    goal_count = 0

    # play game
    while True:
        obs = env.reset()
        done = False
        step = 0
        values = []
        goal = 0
        side = False

        while not done:
            step += 1
            #print(step)

            if goal != 0 and step == goal + 10:
                writer.writerow([side] + values[goal - 20 :])
                csvfile.flush()
                print("zapisujem")
                goal = 0

            # Deterministic action and value prediction
            action, _ = model.predict(obs, deterministic=True)
            output = model.policy.forward(model.policy.obs_to_tensor(obs)[0], deterministic=True, eval=True)
            value = output[1].item()  # Extrahovanie hodnoty stavu
            values.append(value)

            # env step
            obs, reward, done, info = env.step(action)

            if abs(reward) == 1:
                goal = step
                goal_count += 1
                print(goal_count)
                if reward == 1:
                    side = "right"
                else:
                    side = "left"

        