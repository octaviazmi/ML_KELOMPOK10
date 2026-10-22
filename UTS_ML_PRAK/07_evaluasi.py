"""TAHAP 7 - EVALUASI: Confusion Matrix, Akurasi, Presisi, Recall, F1-Score.
Kelas positif = 1 (churn)."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, ConfusionMatrixDisplay)
from config import *

d = pd.read_csv(p("prediksi_testing.csv")); y = d["y_true"]
skenario = ["Semua Fitur", "Chi-Square", "RFE"]

print("=" * 60)
print("TAHAP 7 - EVALUASI (kelas positif = 1 / churn)")
print("=" * 60)
rows = []
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, nama in zip(ax, skenario):
    pred = d[nama]
    cm = confusion_matrix(y, pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y, pred); pre = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred); f1 = f1_score(y, pred)
    rows.append([nama, tn, fp, fn, tp, acc, pre, rec, f1])
    print(f"\n[{nama}]")
    print("Confusion Matrix (baris = aktual, kolom = prediksi):")
    print(pd.DataFrame(cm, index=["Aktual 0", "Aktual 1"], columns=["Prediksi 0", "Prediksi 1"]).to_string())
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"Akurasi  = (TP+TN)/Total = ({tp}+{tn})/{len(y)} = {acc:.4f}")
    print(f"Presisi  = TP/(TP+FP)    = {tp}/({tp}+{fp}) = {pre:.4f}")
    print(f"Recall   = TP/(TP+FN)    = {tp}/({tp}+{fn}) = {rec:.4f}")
    print(f"F1-Score = 2*P*R/(P+R)   = {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y, pred, target_names=["Tidak Churn (0)", "Churn (1)"], zero_division=0, digits=4))
    ConfusionMatrixDisplay(cm, display_labels=["Tidak Churn", "Churn"]).plot(ax=a, cmap="Blues", colorbar=False, values_format="d")
    a.set_title(f"{nama}\nAkurasi {acc:.4f}")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "05_confusion_matrix.png"), dpi=130); plt.close()

hasil = pd.DataFrame(rows, columns=["Skenario", "TN", "FP", "FN", "TP", "Akurasi", "Presisi", "Recall", "F1-Score"])
hasil.to_csv(p("hasil_evaluasi.csv"), index=False)
print("\n" + "=" * 60 + "\nPERBANDINGAN KINERJA\n" + "=" * 60)
print(hasil.round(4).to_string(index=False))

hasil.set_index("Skenario")[["Akurasi", "Presisi", "Recall", "F1-Score"]].plot(kind="bar", figsize=(8, 4.5), rot=0, colormap="Set2")
plt.title("Perbandingan Kinerja Decision Tree"); plt.ylim(0, 1); plt.ylabel("Nilai"); plt.legend(loc="upper right", ncol=4, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "06_perbandingan_metrik.png"), dpi=130); plt.close()
print("\nTersimpan: output/hasil_evaluasi.csv, figures/05_confusion_matrix.png, figures/06_perbandingan_metrik.png")
