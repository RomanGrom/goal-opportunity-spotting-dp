import os
import cv2
import numpy as np
import torch

class ActionsDataLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.action_indexes = {
            "goal": 1,
            "kick-off": 1,
            "penalty": 1,
            "shots_on_target": 1,
            "substitution": 1
        }

    def get_action(self, action_type):
        action_path = os.path.join(self.root_dir, action_type, f"action_{self.action_indexes[action_type]}")

        if os.path.exists(action_path):
            frames = sorted([os.path.join(action_path, f) for f in os.listdir(action_path)])
            loaded_frames = [cv2.imread(frame) for frame in frames if cv2.imread(frame) is not None]
            if len(loaded_frames) == 0:
                return None

            frames_array = np.stack(loaded_frames, axis=0)
            frames_array = np.transpose(frames_array, (0, 3, 1, 2))  # (počet_framov, kanály, výška, šírka)
            self.action_indexes[action_type] += 1
            return frames_array

        return None

    def __len__(self, action_type):
        action_path = os.path.join(self.root_dir, action_type)
        return len(os.listdir(action_path)) if os.path.exists(action_path) else 0


# Funkcia na úpravu framov rovnako ako v gfootball
def preprocess_frame(frame, target_height=224, target_width=224):
    # Zmena veľkosti frame
    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    # Normalizácia pixelových hodnôt (do rozsahu [0, 1])
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    return normalized_frame


# Evaluácia kritika na reálnych observations
def evaluate_critic(model, data_loader, action_type, device='cpu'):
    # Prenos modelu na zvolené zariadenie (CPU/GPU)
    model.to(device)
    model.eval()

    total_values = []
    total_observations = 0

    while True:
        # Načítanie ďalšej akcie
        action_frames = data_loader.get_action(action_type)
        if action_frames is None:
            break

        # Preprocessing framov
        preprocessed_frames = np.array([preprocess_frame(frame) for frame in action_frames])
        preprocessed_frames = np.transpose(preprocessed_frames, (0, 3, 1, 2))  # (počet_framov, kanály, výška, šírka)

        # Konverzia na Tensor
        obs_tensor = torch.tensor(preprocessed_frames, dtype=torch.float32, device=device)

        # Evaluácia kritika
        with torch.no_grad():
            values = model(obs_tensor)
        
        # Ukladanie hodnôt
        total_values.append(values.cpu().numpy())
        total_observations += len(values)

    # Spočítanie priemernej hodnoty
    all_values = np.concatenate(total_values, axis=0)
    average_value = np.mean(all_values)
    
    return average_value, total_observations


# Použitie
if __name__ == "__main__":
    root_dir = ".scratch/actions_dataset"
    loader = ActionsDataLoader(root_dir)

    # Nahrať model kritika (predpokladá sa torch model)
    model_path = "path_to_trained_critic_model.pth"
    critic_model = torch.load(model_path)

    # Evaluácia pre "goal"
    action_type = "goal"
    average_value, total_observations = evaluate_critic(critic_model, loader, action_type, device='cpu')

    print(f"Evaluated {total_observations} observations for action '{action_type}'.")
    print(f"Average Critic Value: {average_value:.4f}")
