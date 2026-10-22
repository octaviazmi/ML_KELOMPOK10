"""TAHAP 8 - PERBANDINGAN Naive Bayes vs SVM vs Decision Tree dan pemilihan model untuk UI."""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

h = pd.read_csv(p("hasil_evaluasi.csv"))
print("=" * 60)
print("TAHAP 8 - PERBANDINGAN KINERJA MODEL")
print("=" * 60)
print(h[["Model", "Skenario", "Accuracy", "Precision", "Recall", "F1-Score"]].round(4).to_string(index=False))

for metrik in ["Accuracy", "F1-Score"]:
    print(f"\nPeringkat berdasarkan {metrik}:")
    print(h.sort_values(metrik, ascending=False)[["Model", "Skenario", metrik]].round(4).head(5).to_string(index=False))

# Model terbaik berdasarkan ACCURACY (sesuai ketentuan tugas).
# Jika ada yang seri: urutkan lagi berdasarkan F1-Score, Recall, Precision, lalu jumlah fitur paling sedikit.
tertinggi = h[h.Accuracy == h.Accuracy.max()]
print(f"\nKombinasi dengan accuracy tertinggi ({h.Accuracy.max():.4f}) ada {len(tertinggi)}:")
print(tertinggi[["Model", "Skenario", "Accuracy", "Recall", "F1-Score"]].round(4).to_string(index=False))
n_pred = (tertinggi.TP + tertinggi.FP)
print(f"Jumlah data yang diprediksi churn oleh kombinasi tersebut: {n_pred.tolist()}")
fitur_n = {"Semua Fitur": 28, "Chi-Square": 10, "RFE": 10}
h["_nfitur"] = h.Skenario.map(fitur_n)
urut = h.sort_values(["Accuracy", "F1-Score", "Recall", "Precision", "_nfitur"], ascending=[False, False, False, False, True], kind="stable")
best_acc = urut.iloc[0]
best_f1 = h.sort_values("F1-Score", ascending=False).iloc[0]
print(f"\nModel untuk UI (accuracy tertinggi): {best_acc.Model} - {best_acc.Skenario} (accuracy {best_acc.Accuracy:.4f}, F1 {best_acc['F1-Score']:.4f})")
print(f"F1-Score tertinggi (pembanding)    : {best_f1.Model} - {best_f1.Skenario} (F1 {best_f1['F1-Score']:.4f}, accuracy {best_f1.Accuracy:.4f})")
suf = {"Semua Fitur": "trans", "Chi-Square": "chi2", "RFE": "rfe"}
json.dump({"model": best_acc.Model, "skenario": best_acc.Skenario, "suffix": suf[best_acc.Skenario],
           "file": f"model_{best_acc.Model.replace(' ', '_')}_{suf[best_acc.Skenario]}.joblib",
           "accuracy": float(best_acc.Accuracy), "precision": float(best_acc.Precision),
           "recall": float(best_acc.Recall), "f1": float(best_acc["F1-Score"])},
          open(p("model_terpilih.json"), "w"), indent=2)
print("Model untuk UI disimpan di output/model_terpilih.json")

# grafik: metrik per model pada tiap skenario
fig, ax = plt.subplots(1, 4, figsize=(17, 4.3), sharey=True)
for a, met in zip(ax, ["Accuracy", "Precision", "Recall", "F1-Score"]):
    pv = h.pivot(index="Skenario", columns="Model", values=met).loc[["Semua Fitur", "Chi-Square", "RFE"], ["Naive Bayes", "SVM", "Decision Tree"]]
    pv.plot(kind="bar", ax=a, rot=15, width=0.8, legend=False, colormap="Set2"); a.set_title(met); a.set_ylim(0, 1); a.set_xlabel("")
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "perbandingan_model.png"), dpi=130); plt.close()
print("Tersimpan: figures/perbandingan_model.png")
