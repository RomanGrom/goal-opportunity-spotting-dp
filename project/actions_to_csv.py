import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from stable_baselines3 import PPO
import csv

# --- Vstupné premenné ---
input_dirs = [Path(p) for p in [".scratch/chances_dataset/chance/right/true",
                                ".scratch/chances_dataset/chance/right/false",
                                ".scratch/chances_dataset/chance/left/true",
                                ".scratch/chances_dataset/chance/left/false",
                                ".scratch/chances_dataset/no_chance/right/true",
                                ".scratch/chances_dataset/no_chance/right/false",
                                ".scratch/chances_dataset/no_chance/left/true",
                                ".scratch/chances_dataset/no_chance/left/false"
                                ]]
output_file = ".scratch/final_onlyai_2_blur9.csv"
model_path = ".scratch/logs/only_AI/2/last_model.zip"
frames_output_dir = Path(".scratch/predictions_frames")  # <- Výstup pre obrázky
frames_output_dir.mkdir(parents=True, exist_ok=True)

# Konštanty
stacked_frames = 4
frame_skip = 2
gray_base = 128

# --- Načítanie modelu ---
def load_model(model_path):
    model = PPO.load(model_path)
    model.policy.set_training_mode(False)
    print(model.device)
    return model

# --- Augmentácia ---
def augment_frame(frame, sat=0.6, val=1, contrast=0.6, blur=4, noise_std=0.0):
    # 1. Úprava saturácie a jasu pomocou HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)  # Saturation
    hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)  # Value
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. Úprava kontrastu pomocou lineárnej transformácie
    #frame = np.clip(contrast * (frame - 128) + 128, 0, 255).astype(np.uint8)
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=10)

    # 3. Rozmazanie (Gaussian blur)
    if blur > 0:
        frame = cv2.GaussianBlur(frame, (2 * blur + 1, 2 * blur + 1), 0)
        #frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # 4. Pridanie šumu
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

    return frame

# --- Flip ---
def flip_frame(frame):
    return cv2.flip(frame, 1)



# --- Spracovanie pozorovania ---
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

import re

def extract_frame_number(path):
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1

# --- Hlavná funkcia ---
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
                    continue  # Príliš málo snímok

                for i in range(75, len(frames) - stacked_frames + 1):
                    stacked_obs, stacked_obs_flipped = [], []

                    for j in range(0, stacked_frames * frame_skip, frame_skip):
                        idx = i + j
                        idx = idx if idx < len(frames) else i
                        frame = cv2.imread(str(frames[idx]))
                        frame = cv2.resize(frame, (224, 224))
                        frame_aug = augment_frame(frame)
                        frame_flip = flip_frame(frame_aug)
                        stacked_obs.append(frame_aug)
                        stacked_obs_flipped.append(frame_flip)

                    obs_normal = process_frame(stacked_obs)
                    obs_flipped = process_frame(stacked_obs_flipped)

                    # Normálne
                    #action, _ = model.predict(obs_normal, deterministic=True)
                    output = model.policy.forward(model.policy.obs_to_tensor(obs_normal)[0], deterministic=True, eval=True)
                    dist, value = output[3], output[1].item()
                    action_probs = dist.distribution.probs.cpu().detach().numpy().flatten()

                    # Flipnuté
                    #action_f, _ = model.predict(obs_flipped, deterministic=True)
                    output_f = model.policy.forward(model.policy.obs_to_tensor(obs_flipped)[0], deterministic=True, eval=True)
                    dist_f, value_f = output_f[3], output_f[1].item()
                    action_probs_f = dist_f.distribution.probs.cpu().detach().numpy().flatten()



                    # Zápis do CSV
                    # --- Získaj info zo vstupnej cesty ---
                    path_str = str(input_dir).lower()
                    chance = "chance" in path_str and "no_chance" not in path_str
                    side = "right" if "right" in path_str else "left"
                    label = True if "true" in path_str else False

                    # Unikátny názov súboru (napr. pre neskoršie spárovanie)
                    filename = f"{match_dir.name}_{i}.png"

                    # --- Zápis do CSV ---
                    csv_writer.writerow([
                        filename, i, chance, side, label, round(value, 5), round(value_f, 5)
                    ])


                action_id += 1
                print(f"Processed action {action_id} from {input_dir.name}")

# --- Spustenie ---
if __name__ == "__main__":
    model = load_model(model_path)
    process_dataset(input_dirs, output_file, model)
