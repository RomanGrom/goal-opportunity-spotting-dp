import gfootball.env as football_env
import numpy as np
import cv2  # OpenCV na uloženie obrázka

# Vytvorenie prostredia
env = football_env.create_environment(env_name="11_vs_11_stochastic", render=True, representation="pixels", logdir="/workspace/goal_spotting", channel_dimensions=[224,224])
obs = env.reset()

env.step(1)

# Ak má pozorovanie viaceré rozmery, zobrazíme len prvý
if isinstance(obs, list):  
    obs = obs[0]

import matplotlib.pyplot as plt

plt.imshow(obs)
plt.axis("off")  # Skryje osy, aby nerobilo čierny rámik
plt.savefig("gfootball_obs_nopad.png", bbox_inches="tight", pad_inches=0)

print("Pozorovanie uložené ako gfootball_obs.png")
