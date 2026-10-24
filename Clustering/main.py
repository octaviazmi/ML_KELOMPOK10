"""
main.py  --  Program utama Tugas Clustering (Modul 13)

Menggabungkan tiga bagian kelompok menjadi satu alur:
  1. eda.py           (Anggota 1) - EDA dan penanganan data awal
  2. kmodes_model.py   (Anggota 1) - Clustering K-Modes (fitur kategorikal)
  3. kmeans_model.py   (Anggota 2) - Clustering K-Means (fitur numerik)

Dataset dibaca SEKALI di sini lewat eda.load_dataset(), lalu DataFrame yang
sama (tidak diubah) dioper ke run_eda, run_kmodes, dan run_kmeans. Setiap
fungsi bekerja pada df.copy() masing-masing sehingga ketiga bagian tidak
saling bergantung satu sama lain.

Jalankan dengan:  python main.py
"""

import time
from pathlib import Path

import pandas as pd

from eda import load_dataset, run_eda
from kmodes_model import run_kmodes
from kmeans_model import run_kmeans

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


def _print_title(text):
    print("\n" + "#" * 70)
    print(text)
    print("#" * 70)


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _print_title("TUGAS CLUSTERING - MODUL 13")
    print(f"Folder output: {OUTPUT_DIR}")

    # ---- 1. Baca dataset sekali, pakai bersama ------------------------------
    df = load_dataset()
    print(f"Dataset terbaca: {df.shape[0]} baris x {df.shape[1]} kolom")

    # ---- 2. EDA (Anggota 1) -------------------------------------------------
    _print_title("TAHAP 1 - EXPLORATORY DATA ANALYSIS")
    t0 = time.time()
    run_eda(df, output_dir=OUTPUT_DIR)
    print(f"\n[Waktu EDA: {time.time() - t0:.2f} detik]")

    # ---- 3. K-Modes (Anggota 1, kelompok ganjil) -----------------------------
    _print_title("TAHAP 2 - CLUSTERING K-MODES (FITUR KATEGORIKAL)")
    t0 = time.time()
    kmodes_result = run_kmodes(df, output_dir=OUTPUT_DIR)
    print(f"\n[Waktu K-Modes: {time.time() - t0:.2f} detik]")

    # ---- 4. K-Means (Anggota 2) ----------------------------------------------
    _print_title("TAHAP 3 - CLUSTERING K-MEANS (FITUR NUMERIK)")
    t0 = time.time()
    kmeans_result = run_kmeans(df, output_dir=OUTPUT_DIR)
    print(f"\n[Waktu K-Means: {time.time() - t0:.2f} detik]")

    # ---- 5. Ringkasan perbandingan dua metode --------------------------------
    _print_title("TAHAP 4 - RINGKASAN PERBANDINGAN K-MODES VS K-MEANS")
    ringkasan = pd.DataFrame([
        {
            "Metode": "K-Modes",
            "Jenis fitur": "Kategorikal",
            "Jumlah fitur": kmodes_result["kmodes_data"].shape[1],
            "Jumlah cluster": kmodes_result["labels"].nunique(),
            "Metrik utama": "Cost",
            "Nilai metrik": round(kmodes_result["model"].cost_, 1),
        },
        {
            "Metode": "K-Means",
            "Jenis fitur": "Numerik",
            "Jumlah fitur": kmeans_result["X_scaled"].shape[1],
            "Jumlah cluster": kmeans_result["labels"].nunique(),
            "Metrik utama": "Silhouette Score",
            "Nilai metrik": round(kmeans_result["silhouette"], 4),
        },
    ])
    print(ringkasan.to_string(index=False))
    print("\nCatatan: Cost (K-Modes) dan Silhouette Score (K-Means) tidak bisa "
          "dibandingkan secara langsung karena mengukur hal yang berbeda "
          "(K-Modes hanya memakai fitur kategorikal, K-Means hanya memakai "
          "fitur numerik). Tabel ini sekadar ringkasan hasil akhir kedua "
          "metode, bukan perbandingan 'mana yang lebih baik'.")
    ringkasan.to_csv(OUTPUT_DIR / "ringkasan_perbandingan.csv", index=False)

    elapsed = time.time() - start
    _print_title(f"SELESAI - total waktu: {elapsed:.2f} detik")
    print(f"Semua file tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
