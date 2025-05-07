import os
import numpy as np
import cv2

# Cesty
input_folder = ".scratch/frames/episode_1"     # priečinok s .npy súbormi
output_folder = ".scratch/game_pngs"     # kam uložiť .png obrázky
os.makedirs(output_folder, exist_ok=True)

# Prejdi všetky .npy súbory v priečinku
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".npy"):
        filepath = os.path.join(input_folder, filename)
        array = np.load(filepath)  # Načítanie 6-kanálového obrázku

        # Vyber len prvé tri kanály (RGB)
        rgb_image = array[:, :, :3]

        # Ak sú hodnoty v [0, 1], preveď na [0, 255] a uint8
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        else:
            rgb_image = rgb_image.astype(np.uint8)

        # Uloženie ako .png
        output_path = os.path.join(output_folder, filename.replace(".npy", ".png"))
        cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))  # OpenCV používa BGR
