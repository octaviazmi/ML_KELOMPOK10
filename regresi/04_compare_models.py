import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

# KONFIGURASI
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_FILES = {
    "XGBoost": {
        "metrics": RESULTS_DIR / "xgboost_metrics.json",
        "model": MODELS_DIR / "xgboost_model.pkl",
    },
    "LightGBM": {
        "metrics": RESULTS_DIR / "lightgbm_metrics.json",
        "model": MODELS_DIR / "lightgbm_model.pkl",
    },
}

# Metrik utama penentu model terbaik: RMSE pada data UJI (testing), semakin
# kecil semakin baik. RMSE dipilih (bukan MAE/MAPE) karena mengembalikan nilai
# ke skala unit yang sama dengan harga (USD) dan memberi penalti lebih besar
# pada kesalahan prediksi yang ekstrem -- relevan untuk estimasi harga rumah.
METRIK_PENENTU = "RMSE"


def judul(teks):
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


# 1. MEMUAT METRIK SETIAP MODEL
judul("PERBANDINGAN MODEL - XGBOOST vs LIGHTGBM")

hasil_model = {}
for nama, paths in MODEL_FILES.items():
    if not paths["metrics"].exists():
        raise FileNotFoundError(
            f"File metrik {paths['metrics']} tidak ditemukan. "
            f"Jalankan dulu program training untuk {nama} "
            f"(02_xgboost.py / 03_lightgbm.py)."
        )
    with open(paths["metrics"], encoding="utf-8") as f:
        hasil_model[nama] = json.load(f)

# 2. TABEL PERBANDINGAN
rows = []
for nama, data in hasil_model.items():
    tr, te = data["train_metrics"], data["test_metrics"]
    rows.append({
        "Model": nama,
        "Train_MAE": tr["MAE"], "Train_MAPE": tr["MAPE"], "Train_RMSE": tr["RMSE"],
        "Train_MSE": tr["MSE"], "Train_R2": tr["R2"],
        "Test_MAE": te["MAE"], "Test_MAPE": te["MAPE"], "Test_RMSE": te["RMSE"],
        "Test_MSE": te["MSE"], "Test_R2": te["R2"],
        "Waktu_Training_detik": data.get("training_time_seconds"),
    })
comparison_df = pd.DataFrame(rows)
comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

print("\nRingkasan metrik pada DATA UJI (testing):")
print(
    comparison_df[["Model", "Test_MAE", "Test_MAPE", "Test_RMSE", "Test_MSE", "Test_R2"]]
    .round(4)
    .to_string(index=False)
)

# 3. MENENTUKAN MODEL TERBAIK
kolom_penentu = f"Test_{METRIK_PENENTU}"
idx_terbaik = comparison_df[kolom_penentu].idxmin()
nama_terbaik = comparison_df.loc[idx_terbaik, "Model"]
rmse_terbaik = comparison_df.loc[idx_terbaik, kolom_penentu]

print(f"\nModel terbaik berdasarkan {METRIK_PENENTU} (data uji) terendah: {nama_terbaik}")
print(f"{METRIK_PENENTU} data uji: {rmse_terbaik:,.2f}")

for _, row in comparison_df.iterrows():
    tanda = " <- TERBAIK" if row["Model"] == nama_terbaik else ""
    print(
        f"  {row['Model']:<10} | Test RMSE = {row['Test_RMSE']:>14,.2f} | "
        f"Test MAPE = {row['Test_MAPE']:>6.2f}% | Test R2 = {row['Test_R2']:.4f}{tanda}"
    )

# 4. MENYIMPAN MODEL TERBAIK UNTUK APLIKASI STREAMLIT
model_terbaik_path = MODEL_FILES[nama_terbaik]["model"]
if not model_terbaik_path.exists():
    raise FileNotFoundError(f"File model {model_terbaik_path} tidak ditemukan.")

shutil.copy(model_terbaik_path, MODELS_DIR / "best_model.pkl")

best_info = {
    "model_name": nama_terbaik,
    "selection_metric": METRIK_PENENTU,
    "train_metrics": hasil_model[nama_terbaik]["train_metrics"],
    "test_metrics": hasil_model[nama_terbaik]["test_metrics"],
    "training_time_seconds": hasil_model[nama_terbaik].get("training_time_seconds"),
    "compared_against": {
        nama: {
            "test_metrics": data["test_metrics"],
        }
        for nama, data in hasil_model.items() if nama != nama_terbaik
    },
}
with open(MODELS_DIR / "best_model_info.json", "w", encoding="utf-8") as f:
    json.dump(best_info, f, indent=2, default=str)

print(f"\n[SIMPAN] Model terbaik disalin ke: models/best_model.pkl ({nama_terbaik})")
print("[SIMPAN] Info model terbaik       : models/best_model_info.json")
print("[SIMPAN] Tabel perbandingan       : results/model_comparison.csv")

# 5. VISUALISASI PERBANDINGAN METRIK
metrik_list = ["MAE", "MAPE", "RMSE", "MSE", "R2"]
fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
nama_models = comparison_df["Model"].tolist()
warna = ["tab:orange", "tab:blue"]

for ax, m in zip(axes, metrik_list):
    nilai = comparison_df[f"Test_{m}"].values
    bars = ax.bar(nama_models, nilai, color=warna[: len(nama_models)])
    ax.set_title(m)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, nilai):
        label = f"{v:.3f}" if m in ("R2",) else f"{v:,.0f}" if m != "MAPE" else f"{v:.2f}%"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), label,
                ha="center", va="bottom", fontsize=8)

fig.suptitle("Perbandingan Metrik Evaluasi (Data Uji): XGBoost vs LightGBM", y=1.03)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "model_comparison_metrics.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Grafik perbandingan garis aktual vs prediksi kedua model berdampingan
pred_files = {
    "XGBoost": RESULTS_DIR / "xgboost_predictions.csv",
    "LightGBM": RESULTS_DIR / "lightgbm_predictions.csv",
}
fig, ax = plt.subplots(figsize=(11, 5))
loaded = False
for nama, p in pred_files.items():
    if p.exists():
        dfp = pd.read_csv(p).head(80)
        if not loaded:
            ax.plot(range(len(dfp)), dfp["actual"], marker="o", ms=3, label="Aktual", color="black")
            loaded = True
        ax.plot(range(len(dfp)), dfp["predicted"], marker="x", ms=4, label=f"Prediksi {nama}")
ax.set_title("Perbandingan Prediksi XGBoost vs LightGBM (80 sampel pertama data uji)")
ax.set_xlabel("Indeks sampel uji")
ax.set_ylabel("Harga (USD)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "model_comparison_predictions.png", dpi=150)
plt.close(fig)

print("[SIMPAN] Grafik perbandingan metrik     : results/model_comparison_metrics.png")
print("[SIMPAN] Grafik perbandingan prediksi   : results/model_comparison_predictions.png")

judul("SELESAI")
print(f"KESIMPULAN: Model {nama_terbaik} dipilih sebagai model terbaik dan digunakan pada")
print("aplikasi Streamlit (app.py), karena memiliki RMSE data uji paling rendah di antara")
print("kedua model yang dibandingkan.")
