import gfootball.env as football_env
import numpy as np
import time

# Vytvorenie prostredia
env = football_env.create_environment(
    env_name='11_vs_11_stochastic',
    other_config_options={"action_set": "v2"} 
)

obs = env.reset()
done = False
total_reward = 0

while not done:
    # Vyber náhodnú akciu
    action = env.action_space.sample()

    # Pošli ju do prostredia
    obs, reward, done, info = env.step([19])
    
    # Sčítaj reward
    total_reward += reward

    # Voliteľne zobraz info
    print(f"Action: {action}, Reward: {reward}, Done: {done}")

    # Mierne spomalenie, aby si stíhal pozerať hru
    time.sleep(0.03)

env.close()
print(f"Celkový reward: {total_reward}")
