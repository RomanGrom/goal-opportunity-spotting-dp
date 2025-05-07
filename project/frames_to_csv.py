import os
import numpy as np
import torch
import csv
from stable_baselines3 import PPO

# Cesta k uloženým framom
frame_dir = ".scratch/frames/episode_1"

# Cesta k modelu a CSV
model_path = ".scratch/logs/gfootball_eva/7/last_model.zip"
output_csv_path = ".scratch/frame_predictions.csv"

# Funkcia na načítanie framov
def load_frames(frame_dir):
    frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.npy')])
    frames = [np.load(file) for file in frame_files]
    return frames

# Funkcia na spracovanie framov na Tensor
def preprocess_frames(frames):
    frames_tensor = torch.tensor([np.transpose(frame, (2, 0, 1)) for frame in frames], dtype=torch.float32)
    return frames_tensor

# Načítanie modelu
def load_model(model_path):
    model = PPO.load(model_path)
    model.policy.set_training_mode(False)  # Eval mode
    return model

# Funkcia na predikciu a uloženie do CSV
def predict_and_save(frames_tensor, model, output_csv_path, batch_size=32, device="cuda"):
    # Presuň model na GPU, ak je k dispozícii
    frames_tensor = frames_tensor.to(device)

    # Otvoriť CSV súbor pre zapisovanie výsledkov
    with open(output_csv_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame_Index", "Action", "Value"] + [f"Action_Prob_{i}" for i in range(19)])

        # Predikcie v dávkach
        num_frames = frames_tensor.shape[0]
        for start_idx in range(0, num_frames, batch_size):
            end_idx = min(start_idx + batch_size, num_frames)
            frame_batch = frames_tensor[start_idx:end_idx]

            # Predikcia modelom
            with torch.no_grad():
                output = model.policy.forward(frame_batch, deterministic=True, eval=True)
                dist = output[3]  # Distribúcia akcií
                value = output[1]  # Hodnota stavu

                action_probs = dist.distribution.probs.cpu().numpy()
                values = value.cpu().numpy()

            # Uložiť výsledky do CSV
            for i, probs in enumerate(action_probs):
                csv_writer.writerow(
                    [start_idx + i, 
                     probs.argmax(),  # Vybraná akcia
                     values[i].item()] + list(probs)
                )

# Hlavný kód
if __name__ == "__main__":
    # 1. Načítanie framov
    frames = load_frames(frame_dir)

    # 2. Spracovanie na Tensor
    frames_tensor = preprocess_frames(frames)

    # 3. Načítanie modelu
    model = load_model(model_path)

    # 4. Predikcia a uloženie výsledkov
    predict_and_save(frames_tensor, model, output_csv_path)

    print(f"Predictions saved to {output_csv_path}")
