
import gfootball.env as football_env
import matplotlib.pyplot as plt



env = football_env.create_environment(
            env_name="11_vs_11_stochastic",
            representation="pixels",
            rewards='scoring',
            render=True,
            channel_dimensions=[224, 224]
            )






#env = make_football_env(0)()
frames = env.reset()

#frames = env.render()

# Odstránenie batch dimenzie -> (6, 224, 224)
#frames = frames.squeeze(0)  
print("New shape:", frames.shape)  # Očakávame (6, 224, 224)

# Uloženie RGB obrázka
plt.figure(figsize=(5, 5))
plt.imshow(frames)
plt.axis("off")
plt.title("RGB Frame")
plt.savefig("rgb_frame.png", bbox_inches="tight", dpi=300)
plt.close()
