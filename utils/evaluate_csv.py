import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cosine, cityblock
from scipy.stats import pearsonr
from fastdtw import fastdtw
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Výpočtové funkcie ---
def second_derivative(values):
    return np.diff(np.diff(values))

def moving_average_derivative(values, window_size=5):
    return np.diff(np.convolve(values, np.ones(window_size)/window_size, mode='valid'))

def second_moving_average_derivative(values, window_size=5):
    moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    return np.diff(np.diff(moving_avg))

def max_moving_average(values, window_size=5):
    moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    return np.max(np.abs(moving_avg))

def compute_metrics(df):
    metrics = []
    for chance_id, group in df.groupby("chance_id"):
        values = group["value"].values
        metrics.append({
            "chance_id": chance_id,
            "max_abs_value": np.max(np.abs(values)),
            "top_10_avg": np.mean(np.sort(np.abs(values))[-10:]),
            "min_max_diff": np.abs(np.max(values) - np.min(values)),
            "max_derivative": np.max(np.abs(np.diff(values))),
            "max_second_derivative": np.max(np.abs(second_derivative(values))),
            "max_moving_derivative": np.max(np.abs(moving_average_derivative(values))),
            "max_second_moving_derivative": np.max(np.abs(second_moving_average_derivative(values))),
            "max_moving_value": np.abs(max_moving_average(values)),
            "true_label": group["chance"].iloc[0]
        })
    return pd.DataFrame(metrics)

def compare_to_avg_all_metrics(avg_sanca, avg_nesanca, df):
    results = []
    grouped = df.groupby("chance_id")["abs_value"].apply(lambda x: x.values)
    for chance_id, values in grouped.items():
        l2_score = euclidean(values, avg_nesanca) - euclidean(values, avg_sanca)
        l1_score = cityblock(values, avg_nesanca) - cityblock(values, avg_sanca)
        pearson_score = pearsonr(values, avg_sanca)[0] - pearsonr(values, avg_nesanca)[0]
        cosine_score = (1 - cosine(values, avg_sanca)) - (1 - cosine(values, avg_nesanca))
        dtw_score = fastdtw(values, avg_nesanca)[0] - fastdtw(values, avg_sanca)[0]
        true_label = df[df["chance_id"] == chance_id]["chance"].iloc[0]

        results.append({
            "chance_id": chance_id,
            "true_label": true_label,
            "l2_score": l2_score,
            "l1_score": l1_score,
            "pearson_score": pearson_score,
            "cosine_score": cosine_score,
            "dtw_score": dtw_score
        })
    return pd.DataFrame(results)

def find_threshold_equal_fpr_fnr(df_scores, score_column):
    thresholds = np.linspace(df_scores[score_column].min(), df_scores[score_column].max(), 500)
    best_threshold = None
    min_diff = float('inf')

    y_true = df_scores["true_label"].values

    for threshold in thresholds:
        y_pred = (df_scores[score_column] > threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        diff = abs(fpr - fnr)

        if diff < min_diff:
            min_diff = diff
            best_threshold = threshold

    # Výsledky pre najlepší threshold
    y_pred_best = (df_scores[score_column] > best_threshold).astype(int)

    acc = accuracy_score(y_true, y_pred_best)
    prec = precision_score(y_true, y_pred_best, zero_division=0)
    rec = recall_score(y_true, y_pred_best, zero_division=0)
    f1 = f1_score(y_true, y_pred_best, zero_division=0)

    print(f"\n=== {score_column.upper()} ===")
    print(f"Threshold (FPR ≈ FNR): {best_threshold:.3f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 score:  {f1:.4f}")








# === 1. Načítanie a spracovanie dát ===
df = pd.read_csv("data/onlyai_best.csv")

df["base_filename"] = df["filename"].str.extract(r"(instance_\d+)_")
df["chance_id"] = df['chance'].astype(str) + '_' + df["base_filename"] + "_" + df["bool"].astype(str) + "_" + df["side"]
df["abs_value"] = df["value"].abs()

# special cases
#df1 = df1[(df1["side"] == "left") & (df1["bool"] == False)]

# === 2. Priemerné hodnoty pre šancu a nešancu ===
sanca_matrix = df[df["chance"] == True].groupby("chance_id")["abs_value"].apply(lambda x: x.values).to_list()
sanca_matrix = np.vstack(sanca_matrix)
avg_sanca = np.mean(sanca_matrix, axis=0)

nesanca_matrix = df[df["chance"] == False].groupby("chance_id")["abs_value"].apply(lambda x: x.values).to_list()
nesanca_matrix = np.vstack(nesanca_matrix)
avg_nesanca = np.mean(nesanca_matrix, axis=0)

# === 3. Výpočet similarity metrík ===
df_scores = compare_to_avg_all_metrics(avg_sanca, avg_nesanca, df)

# === 4. Výpočet interných metrík ===
df_internal = compute_metrics(df)

# === 5. Výpis výsledkov pre všetky metriky ===
print("\n\n### Similarity metriky ###")
for col in ["l2_score", "l1_score", "pearson_score", "cosine_score", "dtw_score"]:
    find_threshold_equal_fpr_fnr(df_scores, col)

print("\n\n### Interné metriky ###")
for col in ["max_abs_value", "top_10_avg", "min_max_diff", "max_derivative",
            "max_second_derivative", "max_moving_derivative",
            "max_second_moving_derivative", "max_moving_value"]:
    find_threshold_equal_fpr_fnr(df_internal, col)
