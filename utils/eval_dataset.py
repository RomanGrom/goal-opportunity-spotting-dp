import os
import cv2
import numpy as np

class ActionsDataLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Počítadlá pre jednotlivé akcie
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
            # Načítanie framov z daného priečinka
            frames = sorted([os.path.join(action_path, f) for f in os.listdir(action_path)])
            
            # Načítanie framov a konverzia do NumPy poľa
            loaded_frames = [cv2.imread(frame) for frame in frames]
            if len(loaded_frames) == 0:
                return None

            # Konverzia do formátu (počet_framov, výška, šírka, kanály)
            frames_array = np.stack(loaded_frames, axis=0)
            frames_array = np.transpose(frames_array, (0, 3, 1, 2))
            
            # Posunutie indexu na ďalšiu akciu
            self.action_indexes[action_type] += 1
            
            return frames_array

        else:
            return None
        
        

    def __len__(self, action_type):
        action_path = os.path.join(self.root_dir, action_type)

        if os.path.exists(action_path):
            return os.listdir(action_path)
        return 0

# Použitie:
root_dir = ".scratch/actions_dataset"
loader = ActionsDataLoader(root_dir)
print(loader.get_action("goal").shape)
print(len(loader.__len__("goal")))