import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pathlib import Path
from tqdm import tqdm


# --- Vstupné premenné ---
input_dirs = [Path(p) for p in [
    #".scratch/chances_dataset/chance/right/true",
    #".scratch/chances_dataset/chance/right/false",
    #".scratch/chances_dataset/chance/left/true",
    #".scratch/chances_dataset/chance/left/false",
    #".scratch/chances_dataset/no_chance/right/true",
    #".scratch/chances_dataset/no_chance/right/false",
    #".scratch/chances_dataset/no_chance/left/true",
    ".scratch/chances_dataset/no_chance/left/false"
]]

csv_path = ".scratch/all_predictions_fs2_lr_gamma_up.csv"
output_dir = Path(".scratch/video_values")
output_dir.mkdir(parents=True, exist_ok=True)


# --- CSV načítanie ---
df = pd.read_csv(csv_path)
df["base_filename"] = df["filename"].str.extract(r"(instance_\d+)_")
df["chance_id"] = df['chance'].astype(str) + '_' + df["base_filename"] + "_" + df["bool"].astype(str) + "_" + df["side"]

# --- Pomocná funkcia ---
def create_video_with_graph(frames_dir: Path, frame_data: pd.DataFrame, out_path: Path, fps=25, min_frame_id=75):
    # Extrahovanie informácií z cesty k adresáru
    parts = frames_dir.parts[-4:]  # Predpokladáme, že cesta bude mať štruktúru ako v príklade
    chance, side, bool_value = parts[0], parts[1], parts[2] == 'true'

    # Rozdelenie pôvodného názvu na názov súboru a príponu
    file_name = out_path.stem  # Bez prípony, napr. 'left_true_instance_0017'
    file_extension = out_path.suffix  # Prípona, napr. '.mp4'

    # Vytvorenie nového názvu videa so zahrnutím chance
    out_path = out_path.with_name(f"{file_name}_{chance}{file_extension}")

    base_filename = parts[3]  # 'instance' na začiatku názvu súboru
    if chance == "chance":
        chance = True
    else:
        chance = False

    # Filtrovanie podľa chance, bool, side a instance
    frame_data['base_filename'] = frame_data['filename'].str.extract(r"(instance_\d+)_")
    frame_data['chance_id'] = frame_data['chance'].astype(str) + '_' + frame_data["base_filename"] + "_" + frame_data["bool"].astype(str) + "_" + frame_data["side"]

    # Filtrujeme frame_data podľa chance_id
    chance_id = f"{chance}_{base_filename}_{bool_value}_{side}"
    filtered = frame_data[frame_data['chance_id'] == chance_id]

    # Ak je filtered prázdny, vypíšeme varovanie a vrátime sa
    if filtered.empty:
        print(f"⚠️  Preskakujem {frames_dir}, žiadne platné súbory pre {chance_id}.")
        return

    # Filtrovanie podľa frame_id (číslo framu extrahované z názvu súboru)
    filtered['frame_id'] = filtered['filename'].str.extract(r'_(\d+)\.png')[0].astype(int)
    filtered = filtered[filtered['frame_id'] >= min_frame_id]
    
    # Ak po filtrovaní zostane prázdny DataFrame, vypíšeme varovanie
    if filtered.empty:
        print(f"⚠️  Preskakujem {frames_dir}, žiadne platné framy od {min_frame_id}.")
        return

    #filtered = filtered.sort_values("filename")

    values = filtered['value'].values
     # Získanie všetkých súborov vo vnútri priečinka podľa base_filename

    frames = sorted(frames_dir.glob("*.png"))

    # Vynechanie prvých 74 rámov
    frame_files = frames[min_frame_id - 1:]

    # Získame rozmery videa
    sample_img = cv2.imread(str(frame_files[0]))
    height, width, _ = sample_img.shape

    # Zväčšíme šírku videa, aby sme pridali graf
    graph_width = width  # šírka grafu, prispôsobíme podľa potreby
    output_width = width + graph_width  # celková šírka výstupu (video + graf)

    # Inicializácia VideoWriter s upravenými rozmermi
    video_writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (output_width, height)  # Celkové rozmery videa (video + graf)
    )

    for i, frame_path in enumerate(tqdm(frame_files, desc=f"📹 {frames_dir.name}")):
        img = cv2.imread(str(frame_path))

        # Graf hodnoty
        fig, ax = plt.subplots(figsize=(6, 3), dpi=200)  # Zväčšenie grafu a zvyšovanie DPI pre ostrosť
        ax.plot(values[:i+1], color='blue')

        # Pridanie mierky na osy
        ax.set_xlim(0, len(values))  # Osa X zodpovedá počtu hodnôt
        ax.set_ylim(min(values), max(values))  # Osa Y zodpovedá rozsahu hodnôt
        ax.set_xlabel('Frame Index')  # Pridanie popisku na X os
        ax.set_ylabel('Value')  # Pridanie popisku na Y os
        ax.grid(True)  # Zobraziť mriežku
        ax.axis('on')  # Zapnutie osí

        # Získanie grafu do obrazu
        canvas = FigureCanvas(fig)
        canvas.draw()
        graph = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        graph = graph.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Zmena veľkosti grafu
        graph_resized = cv2.resize(graph, (width, height))  # Graf je veľký a zmenšíme ho na výšku videa

        # Kombinujeme obrázok a graf: video naľavo, graf napravo
        combined = np.hstack((img, graph_resized))  # Spojíme video a graf horizontálne

        # Uistíme sa, že výsledné rozmery sú správne
        video_writer.write(combined)  # Zápis skombinovaného obrázka do videa



    video_writer.release()
    print(f"✅ Uložené video: {out_path}")




# --- Hlavný cyklus ---
for dir_path in input_dirs:
    # Prejdeme rekurzívne všetky podadresáre a vytvoríme názov videa na základe posledných častí cesty
    for instance_dir in dir_path.rglob('instance_*'):  # Pre každý podadresár 'instance_*'
        instance_name = "_".join(instance_dir.parts[-3:])  # napr. instance_0017_right_true
        output_video_path = output_dir / f"{instance_name}.mp4"
        
        # Vytvorte video pre každý instance adresár
        create_video_with_graph(instance_dir, df, output_video_path)

