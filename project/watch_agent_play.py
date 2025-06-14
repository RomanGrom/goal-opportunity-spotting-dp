import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from football_gym import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import csv
import os




# Function to play the game and save it to a video with sliding window graph
def play_trained_model_with_sliding_graph(model_path, num_episodes=1, video_path="gameplay_with_graph.mp4", fps=30, frame_dir=".scratch/frames"):
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from football_gym import make_football_env

    # Create the environment
    env = DummyVecEnv([lambda: make_football_env(0)()])

    # Load the model
    model = PPO.load(model_path, env=env)
    model.policy.set_training_mode(False)

    # Define video writer
    frame_shape = (224, 224)
    graph_width = 400
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_shape[1] + graph_width, frame_shape[0]))

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        # Initialize value storage
        values = []
        window_size = 200  # Fixed window size for the X-axis

        # Create directory for current episode frames
        episode_dir = os.path.join(frame_dir, f"episode_{episode + 1}")
        os.makedirs(episode_dir, exist_ok=True)

        while not done:
            step += 1
            print(step)
            # Predict action and get value
            action, _ = model.predict(obs, deterministic=True)
            output = model.policy.forward(model.policy.obs_to_tensor(obs)[0], deterministic=True, eval=True)
            value = output[1].item()
            values.append(value)

            # Keep only the last `window_size` values for the sliding window
            if len(values) > window_size:
                values = values[-window_size:]

            # Environment step
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Save the frame with all 6 channels to .npy
            combined_frame = obs[0, :, :, :]  # Shape: (6, H, W)
            combined_frame = np.transpose(combined_frame, (1, 2, 0))  # Transpose to (H, W, C)
            frame_path = os.path.join(episode_dir, f"frame_{step:04d}.npy")
            np.save(frame_path, combined_frame)

            # Prepare the RGB frame
            color_frame = obs[0, :3, :, :]
            frame = np.transpose(color_frame, (1, 2, 0)).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Increase brightness
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            hsv[..., 2] = cv2.add(hsv[..., 2], 50)
            frame_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Create the value progression plot
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(values, label="Value Progression", color="blue")
            ax.axvline(x=len(values) - 1, color="red", linestyle="--", label="Current Frame")
            ax.set_xlim(0, window_size)  # Fixed X-axis window size
            ax.set_ylim(-1.0, 1.0)  # Fixed Y-axis range
            ax.set_xlabel("Frame (Sliding Window)")
            ax.set_ylabel("Value")
            ax.set_title("Critic Value Progression")
            ax.legend()

            # Convert Matplotlib figure to image
            fig.canvas.draw()
            graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            graph_img = cv2.resize(graph_img, (graph_width, frame_shape[0]), interpolation=cv2.INTER_AREA)
            plt.close(fig)

            # Combine the frame and the graph
            combined_frame = np.hstack((frame_bright, graph_img))

            # Write the combined frame to the video
            out.write(combined_frame)

        print(f"Episode {episode + 1} finished. Total reward: {total_reward}")

    # Release resources
    out.release()
    env.close()






def play_for_stats(model_path, num_episodes=1, csv_path="action_values.csv"):
    # Vytvorenie prostredia
    env = DummyVecEnv([lambda: make_football_env(0)()])  # Jedno prostredie na prehrávanie

    # Načítanie modelu a prepnutie do evaluačného režimu
    model = PPO.load(model_path, env=env)
    model.policy.set_training_mode(False)

    # Otvorenie CSV súboru na zapisovanie hodnôt
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "Step", "Action", "Value"])  # Hlavička CSV súboru

        # Prehranie hry pre požadovaný počet epizód
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            step = 0

            while not done:
                print(step)
                step += 1
                # Predikcia akcie modelom (deterministická pre evaluáciu)
                action, _ = model.predict(obs, deterministic=True)
                output = model.policy.forward(model.policy.obs_to_tensor(obs)[0], deterministic=True, eval=True)
                value = output[1].item()  # Extrahovanie hodnoty stavu
                
                # Prostredie vykoná akciu
                obs, reward, done, info = env.step(action)

                # Zapísanie dát do CSV
                writer.writerow([episode + 1, step, action[0], value])

            print(f"Episode {episode + 1} finished.")

    # Uzatvorenie prostredia
    env.close()








# Script to run the game and save it with a sliding window graph
if __name__ == "__main__":
    model_path = ".scratch/logs/only_AI/4/last_model.zip"
    video_path = ".scratch/gameplay.mp4"
    play_trained_model_with_sliding_graph(model_path, num_episodes=1, video_path=video_path, fps=10)
    #play_for_stats(model_path, 10, "action_values_last4.csv")