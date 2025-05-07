import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from stable_baselines3 import PPO
import csv

# --- Input parameters ---
input_dirs = [Path(".scratch/actions_dataset/goal"), Path(".scratch/actions_dataset/kick-off")]
output_file = ".scratch/all_predictions_goals.csv"
model_path = ".scratch/last_model.zip"


# outpur dir
frames_output_dir = Path(".scratch/predictions_frames_simple")
frames_output_dir.mkdir(parents=True, exist_ok=True)

# constants
stacked_frames = 4
frame_skip = 2
gray_base = 128

# --- Load the model ---
def load_model(model_path):
    model = PPO.load(model_path)
    model.policy.set_training_mode(False)
    return model

# --- Augmentation ---
def augment_frame(frame, sat=0.6, val=1, contrast=0.6, blur=4, noise_std=0.0):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=10)
    if blur > 0:
        frame = cv2.GaussianBlur(frame, (2 * blur + 1, 2 * blur + 1), 0)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
    return frame

# --- Process frame ---
def process_frame(obs):
    stacked = np.concatenate([frame for frame in obs], axis=-1)
    obs = np.transpose(stacked, (2, 0, 1))

    r_last, g_last, b_last = obs[9], obs[10], obs[11]
    compare_frame = cv2.cvtColor(np.stack((r_last, g_last, b_last), axis=-1), cv2.COLOR_RGB2GRAY)

    grayscale_diffs = [
        r_last[..., np.newaxis],
        g_last[..., np.newaxis],
        b_last[..., np.newaxis],
    ]

    for i in range(6, -1, -3):
        frame = np.stack((obs[i], obs[i+1], obs[i+2]), axis=-1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        diff = np.clip(compare_frame.astype(np.int16) - frame_gray.astype(np.int16) + gray_base, 0, 255).astype(np.uint8)
        grayscale_diffs.append(diff[..., np.newaxis])
        compare_frame = frame_gray

    stacked_obs = np.concatenate(grayscale_diffs, axis=-1)
    stacked_obs = np.transpose(stacked_obs, (2, 0, 1)).astype(np.float32)
    stacked_obs = np.expand_dims(stacked_obs, axis=0)
    return stacked_obs

# --- Main function --- 
def process_dataset(input_dirs, output_file, model):
    max_actions_per_type = 300

    with open(output_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["filename", "frame_id", "chance", "chance_id", "value"])

        chance_id = 0
        for input_dir in input_dirs:
            is_chance = "goal" in input_dir.name.lower()
            label = "chance" if is_chance else "no_chance"

            processed_matches = 0
            for match_dir in tqdm(input_dir.glob("*"), desc=f"Processing {label}"):
                if processed_matches >= max_actions_per_type:
                    break

                frames = sorted(match_dir.glob("*.png"))
                if len(frames) < stacked_frames * frame_skip:
                    continue

                for i in range(50, len(frames) - stacked_frames + 1):
                    stacked_obs = []
                    for j in range(0, stacked_frames * frame_skip, frame_skip):
                        idx = i + j
                        idx = idx if idx < len(frames) else i
                        frame = cv2.imread(str(frames[idx]))
                        frame = cv2.resize(frame, (224, 224))
                        frame_aug = augment_frame(frame)
                        stacked_obs.append(frame_aug)

                    obs = process_frame(stacked_obs)
                    output = model.policy.forward(model.policy.obs_to_tensor(obs)[0], deterministic=True, eval=True)
                    value = output[1].item()

                    filename = f"{match_dir.name}_{i}.png"
                    csv_writer.writerow([filename, i, label, chance_id, round(value, 5)])

                chance_id += 1
                processed_matches += 1



if __name__ == "__main__":
    model = load_model(model_path)
    process_dataset(input_dirs, output_file, model)
