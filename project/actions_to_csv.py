import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from stable_baselines3 import PPO
import csv

# --- Input variables ---
input_dirs = [Path(p) for p in [#".scratch/chances_dataset/chance/right/true",
                                #".scratch/chances_dataset/chance/right/false",
                                #".scratch/chances_dataset/chance/left/true",
                                #".scratch/chances_dataset/chance/left/false",
                                #".scratch/chances_dataset/no_chance/right/true",
                                ".scratch/chances_dataset/no_chance/right/false",
                                ".scratch/chances_dataset/no_chance/left/true",
                                ".scratch/chances_dataset/no_chance/left/false"
                                ]]

# Output csv file
output_file = ".scratch/test.csv"

# Path to the model
model_path = ".scratch/logs/only_AI/1/last_model.zip"


# Constants
stacked_frames = 4
frame_skip = 2
gray_base = 128


# --- Loading of the model ---
def load_model(model_path):
    model = PPO.load(model_path)
    model.policy.set_training_mode(False)
    return model

# --- Augmentation ---
def augment_frame(frame, sat=0.6, val=1, contrast=0.6, blur=4, noise_std=0.0):
    # 1. Adjust saturation and brightness using HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)  # Saturation
    hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)  # Brightness (Value)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. Adjust contrast using linear transformation
    # frame = np.clip(contrast * (frame - 128) + 128, 0, 255).astype(np.uint8)
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=10)

    # 3. Apply Gaussian blur
    if blur > 0:
        #frame = cv2.GaussianBlur(frame, (2 * blur + 1, 2 * blur + 1), 0)
        frame = cv2.GaussianBlur(frame, (9, 9), 0)

    # 4. Add noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

    return frame

# --- Flip ---
def flip_frame(frame):
    return cv2.flip(frame, 1)



# --- Observation processing ---
def process_frame(obs):
    # Stack all frames along the last channel
    stacked = np.concatenate([frame for frame in obs], axis=-1)
    obs = np.transpose(stacked, (2, 0, 1))  # Change shape to (C, H, W)

    # Get the last frame and convert to grayscale
    r_last, g_last, b_last = obs[9], obs[10], obs[11]
    compare_frame = cv2.cvtColor(np.stack((r_last, g_last, b_last), axis=-1), cv2.COLOR_RGB2GRAY)

    # Initialize grayscale diffs with the last RGB frame
    grayscale_diffs = [
        r_last[..., np.newaxis],
        g_last[..., np.newaxis],
        b_last[..., np.newaxis],
    ]

    # Compute grayscale differences from earlier frames
    for i in range(6, -1, -3):
        frame = np.stack((obs[i], obs[i+1], obs[i+2]), axis=-1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        diff = np.clip(compare_frame.astype(np.int16) - frame_gray.astype(np.int16) + gray_base, 0, 255).astype(np.uint8)
        grayscale_diffs.append(diff[..., np.newaxis])
        compare_frame = frame_gray

    # Stack all channels and return as float32 tensor
    stacked_obs = np.concatenate(grayscale_diffs, axis=-1)
    stacked_obs = np.transpose(stacked_obs, (2, 0, 1)).astype(np.float32)
    stacked_obs = np.expand_dims(stacked_obs, axis=0)
    return stacked_obs

import re



# Get the number from the filename
def extract_frame_number(path):
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


# --- Main function ---
def process_dataset(input_dirs, output_file, model):
    with open(output_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["filename", "frame_id", "chance", "side", "bool", "value", "flipped"])

        for input_dir in input_dirs:
            action_id = 0
            for match_dir in tqdm(input_dir.glob("*"), desc=f"Processing {input_dir.name}"):
                print(match_dir)
                frames = sorted(match_dir.glob("*.png"), key=extract_frame_number)
                if len(frames) < stacked_frames * frame_skip:
                    continue  # Not enough frames

                for i in range(75, len(frames) - stacked_frames + 1):
                    stacked_obs, stacked_obs_flipped = [], []

                    for j in range(0, stacked_frames * frame_skip, frame_skip):
                        idx = i + j
                        idx = idx if idx < len(frames) else i
                        frame = cv2.imread(str(frames[idx]))
                        frame = cv2.resize(frame, (224, 224))
                        frame_aug = augment_frame(frame)
                        #frame_flip = flip_frame(frame_aug)
                        stacked_obs.append(frame_aug)
                        #stacked_obs_flipped.append(frame_flip)

                    obs_normal = process_frame(stacked_obs)
                    #obs_flipped = process_frame(stacked_obs_flipped)


                    np.save("data/chance_frame.npy", obs_normal)   



                    # Normal observation
                    output = model.policy.forward(model.policy.obs_to_tensor(obs_normal)[0], deterministic=True, eval=True)
                    dist, value = output[3], output[1].item()
                    action_probs = dist.distribution.probs.cpu().detach().numpy().flatten()

                    # Flipped observation
                    #output_f = model.policy.forward(model.policy.obs_to_tensor(obs_flipped)[0], deterministic=True, eval=True)
                    #dist_f, value_f = output_f[3], output_f[1].item()
                    #action_probs_f = dist_f.distribution.probs.cpu().detach().numpy().flatten()

                    # Write to CSV
                    # --- Extract info from input path ---
                    path_str = str(input_dir).lower()
                    chance = "chance" in path_str and "no_chance" not in path_str
                    side = "right" if "right" in path_str else "left"
                    label = True if "true" in path_str else False

                    # Unique filename (for later matching)
                    filename = f"{match_dir.name}_{i}.png"

                    # --- Write row to CSV ---
                    csv_writer.writerow([
                        filename, i, chance, side, label, round(value, 5), round(value, 5)
                    ])

                action_id += 1
                print(f"Processed action {action_id} from {input_dir.name}")

# --- Run ---
if __name__ == "__main__":
    model = load_model(model_path)
    process_dataset(input_dirs, output_file, model)

