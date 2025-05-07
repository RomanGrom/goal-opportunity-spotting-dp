import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch

# --- Cesty k dátam ---
png_dir = Path(".scratch/frames_no_names")  # cesta k priečinku so .png obrázkami
npy_dir = Path(".scratch/frames_no_name_npy")  # kde sa uloží .npy súbory
npy_dir.mkdir(parents=True, exist_ok=True)

frame_shape = (224, 224)

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
        diff = np.clip(compare_frame.astype(np.int16) - frame_gray.astype(np.int16) + 128, 0, 255).astype(np.uint8)
        grayscale_diffs.append(diff[..., np.newaxis])
        compare_frame = frame_gray

    stacked_obs = np.concatenate(grayscale_diffs, axis=-1)
    stacked_obs = np.transpose(stacked_obs, (2, 0, 1)).astype(np.float32)
    stacked_obs = np.expand_dims(stacked_obs, axis=0)
    return stacked_obs


# Funkcia na spracovanie obrázkov a uloženie .npy súborov
def process_images_and_save():
    png_files = sorted(png_dir.glob("*.png"))
    images = []  # Tu sa budú ukladať snímky pred tým, ako sa vytvoria rozdiely

    for i in tqdm(range(0, len(png_files) - 3)):  # zabezpečujeme, že máme minimálne 4 obrázky
        # Načítanie aktuálneho obrázku (RGB) a predchádzajúcich troch obrázkov
        for j in range(i, i+4):  # Tento cyklus teraz načíta 4 obrázky (1 RGB + 3 rozdielové)
            img = Image.open(png_files[j]).convert("RGB")
            img = np.array(img)  # Premena na numpy array
            images.append(img)

        # Vytvorenie stacknutého pozorovania
        stacked_obs = process_frame(images)

        # Uloženie do .npy
        npy_file = npy_dir / f"frame_{i}.npy"
        np.save(npy_file, stacked_obs)

        # Vymažeme zoznam, aby sme pracovali so správnym segmentom obrázkov
        images = []  # Vyprázdnenie zoznamu na ďalší segment obrázkov


# --- Spustenie funkcie ---
#process_images_and_save()

import re

def extract_frame_number(path):
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from stable_baselines3 import PPO

def compare_models_on_saved_observations(
    model1_path,
    obs_dir1,
    obs_dir2,
    video_path="comparison.mp4",
    fps=10
):
    # Načítaj modely
    model = PPO.load(model1_path)
    model.policy.set_training_mode(False)

    # Získaj všetky súbory (musí byť rovnaký počet)
    files1 = sorted(Path(obs_dir1).glob("*.npy"), key=extract_frame_number)
    files2 = sorted(Path(obs_dir2).glob("*.npy"), key=extract_frame_number)
    assert len(files1) == len(files2), "Nesúlad počtu snímok v priečinkoch"

    graph_width = 400
    frame_shape = (224, 224)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_shape[1] + graph_width, frame_shape[0]))

    values1, values2 = [], []
    window_size = 200

    for idx, (file1, file2) in enumerate(tqdm(zip(files1, files2), total=len(files1))):
        obs1 = np.load(file1)  # (H, W, C)
        obs2 = np.load(file2)
        
        # Prevod na (1, C, H, W)
        obs1_tensor = model.policy.obs_to_tensor(obs1.transpose(2, 0, 1)[None])[0]
        obs2_tensor = model.policy.obs_to_tensor(obs2)[0]

        # Získaj value
        value1 = model.policy.forward(obs1_tensor, deterministic=True, eval=True)[1].item()
        value2 = model.policy.forward(obs2_tensor, deterministic=True, eval=True)[1].item()
        values1.append(value1)
        values2.append(value2)

        if len(values1) > window_size:
            values1 = values1[-window_size:]
            values2 = values2[-window_size:]

        # RGB frame (predpokladáme, že prvé 3 kanály sú RGB)
        rgb_frame = obs1[:, :, :3].astype(np.uint8)
        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Zvýšenie jasu
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = cv2.add(hsv[..., 2], 50)
        frame_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Graf
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(values1, label="Model 1", color="blue")
        ax.plot(values2, label="Model 2", color="green")
        ax.axvline(x=len(values1)-1, color="red", linestyle="--")
        ax.set_xlim(0, window_size)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title("Value Progression")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")
        ax.legend()

        fig.canvas.draw()
        graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        graph_img = cv2.resize(graph_img, (graph_width, frame_shape[0]))
        plt.close(fig)

        combined = np.hstack((frame_bright, graph_img))
        out.write(combined)

    out.release()
    print(f"Video uložené do {video_path}")





compare_models_on_saved_observations(
    model1_path=".scratch/logs/only_AI/4/last_model.zip",
    obs_dir1=".scratch/frames_with_names/",
    obs_dir2=".scratch/frames_no_name_npy/",
    video_path="comparison.mp4"
)