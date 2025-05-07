import os
import json
import cv2
from datetime import datetime

def time_to_frame(time_str, fps):
    """Konvertuje čas v tvare 'MM:SS' na číslo framu."""
    minutes, seconds = map(int, time_str.split(':'))
    total_seconds = minutes * 60 + seconds
    return int(total_seconds * fps)

def process_videos(annotation_dir, video_dir, output_dir, fps=25):
    """Prejde všetky anotácie a vystrihne framy z videí podľa nich."""
    counter = {}
    
    for annotation_file in os.listdir(annotation_dir):
        if not annotation_file.endswith(".json"):
            continue
        
        annotation_path = os.path.join(annotation_dir, annotation_file)
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        game_name = data["gameName"]
        game_folder = os.path.join(video_dir, game_name)
        
        for ann in data["annotations"]:
            half = ann["half"]
            video_filename = f"{half}_224p.mkv"
            video_path = os.path.join(game_folder, video_filename)
            
            if not os.path.exists(video_path):
                print(f"Súbor {video_path} neexistuje, preskakujem...")
                continue
            
            frame_number = time_to_frame(ann["gameTime"], fps)
            start_frame = max(0, frame_number - 150)
            end_frame = frame_number + 50
            
            label = ann["label"]
            side = ann["side"]
            bool_value = "true" if ann["bool"] else "false"
            
            key = (label, side, bool_value)
            if key not in counter:
                counter[key] = 0
            counter[key] += 1
            
            instance_folder = f"instance_{counter[key]:04d}"
            output_path = os.path.join(output_dir, label, side, bool_value, instance_folder)
            os.makedirs(output_path, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for i in range(start_frame, end_frame + 1):
                success, frame = cap.read()
                if not success:
                    break
                frame_filename = f"frame{i}.png"
                cv2.imwrite(os.path.join(output_path, frame_filename), frame)
            
            print(f"Uložena sanca: {output_path}")
            
            cap.release()

# Príklady cesty k dátam
annotation_dir = ".scratch/annotations"
video_dir = ".scratch/videos"
output_dir = ".scratch/chances_dataset"

process_videos(annotation_dir, video_dir, output_dir)