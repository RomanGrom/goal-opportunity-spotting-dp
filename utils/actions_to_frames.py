import os
import json
import cv2
import ffmpeg

# Cesty k datasetu
DATA_DIR = '.scratch/SoccerNet'  # Cesta k adresáru, kde máte uložený SoccerNet dataset
OUTPUT_DIR = '.scratch/actions_dataset'  # Cesta k adresáru, kde chcete uložiť vystrihnuté akcie

# Funkcia na extrahovanie času akcie a prepočet na počet snímok
def get_action_frames(time, fps=25):
    minutes, seconds = map(int, time.split(':'))
    frame_number = (minutes * 60 + seconds) * fps
    start_frame = frame_number - 75
    end_frame = frame_number + 75
    return start_frame, end_frame

# Funkcia na vystrihnutie akcie z videa a uloženie ako PNG
def extract_action_to_png(video_path, start_frame, end_frame, output_path):
    video_cap = cv2.VideoCapture(video_path)
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = 1  # Začiatok číslovania frameov od 1

    while frame_idx <= (end_frame - start_frame + 1):
        ret, frame = video_cap.read()
        if not ret:
            break

        # Ulož frame ako PNG
        output_filename = os.path.join(output_path, f'frame_{frame_idx}.png')
        cv2.imwrite(output_filename, frame)
        frame_idx += 1

    video_cap.release()


actions = ['Goal', 'Penalty','Kick-off', 'Substitution', 'Shots on target']
    
action_counts = {
    'Goal': 0,
    'Penalty': 0,
    'Shots on target': 0,
    'Kick-off': 0,
    'Substitution': 0
}

action_counter = 1


# Funkcia na spracovanie adresára s videami a labelmi
def process_match(data_path, match_idx):
    labels_path = os.path.join(data_path, 'Labels-v2.json')
    half1_path = os.path.join(data_path, '1_720p.mkv')
    half2_path = os.path.join(data_path, '2_720p.mkv')

    if not os.path.isfile(labels_path):
        return

    with open(labels_path, 'r') as file:
        labels = json.load(file)

    for annotation in labels['annotations']:
        if annotation['label'] in actions:  # Vyberte akcie, ktoré chcete extrahovať
            print(annotation['label'], action_counter)

            time = annotation['gameTime'].split(' - ')
            half = time[0]
            action_time = time[1]

            if half == '1':
                video_path = half1_path
            else:
                video_path = half2_path

            start_frame, end_frame = get_action_frames(action_time)

            action_counts[annotation['label']] += 1

            # Nastavenie výstupnej cesty pre uloženie akcie
            action_label = annotation['label'].replace(' ', '_').lower()
            output_path = os.path.join(OUTPUT_DIR, action_label, f"action_{action_counts[annotation['label']]}")
            os.makedirs(output_path, exist_ok=True)

            # Extrahovanie akcie a uloženie ako PNG
            extract_action_to_png(video_path, start_frame, end_frame, output_path)

# Prejsť celý dataset a spracovať každý zápas
match_idx = 1  # Počítadlo pre číslovanie zápasov
for root, dirs, files in os.walk(DATA_DIR):
    if '1_720p.mkv' in files and '2_720p.mkv' in files:
        print(match_idx)
        process_match(root, match_idx)
        match_idx += 1