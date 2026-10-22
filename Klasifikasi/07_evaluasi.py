"""TAHAP 7 - EVALUASI: Confusion Matrix, Accuracy, Precision, Recall, F1-Score (kelas positif = 1 / churn)."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, ConfusionMatrixDisplay)
from config import *

d = pd.read_csv(p("prediksi_testing.csv")); y = d["y_true"]
MODELS = ["Naive Bayes", "SVM", "Decision Tree"]; SKEN = ["Semua Fitur", "Chi-Square", "RFE"]

print("=" * 60)
print("TAHAP 7 - EVALUASI (kelas positif = 1 / churn)")
print("=" * 60)
rows = []
for m in MODELS:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    print(f"\n######## {m} ########")
    for a, s in zip(ax, SKEN):
        pred = d[f"{m} | {s}"]
        cm = confusion_matrix(y, pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y, pred); pre = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0); f1 = f1_score(y, pred, zero_division=0)
        rows.append([m, s, tn, fp, fn, tp, acc, pre, rec, f1])
        print(f"\n[{m} - {s}]")
        print("Confusion Matrix (baris = aktual, kolom = prediksi):")
        print(pd.DataFrame(cm, index=["Aktual 0", "Aktual 1"], columns=["Prediksi 0", "Prediksi 1"]).to_string())
        print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(f"Accuracy  = {acc:.4f}")
        print(f"Precision = {pre:.4f}")
        print(f"Recall    = {rec:.4f}")
        print(f"F1-Score  = {f1:.4f}")
        ConfusionMatrixDisplay(cm, display_labels=["Tidak Churn", "Churn"]).plot(ax=a, cmap="Blues", colorbar=False, values_format="d")
        a.set_title(f"{m} - {s}\nAccuracy {acc:.4f}", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, f"cm_{m.replace(' ', '_')}.png"), dpi=130); plt.close()

hasil = pd.DataFrame(rows, columns=["Model", "Skenario", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "F1-Score"])
hasil.to_csv(p("hasil_evaluasi.csv"), index=False)
print("\nTersimpan: output/hasil_evaluasi.csv dan figures/cm_*.png")
