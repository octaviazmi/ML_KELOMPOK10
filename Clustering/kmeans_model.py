"""
kmeans_model.py  --  K-Means (TANGGUNG JAWAB ANGGOTA 2)

STATUS: KERANGKA KOSONG. Belum ada implementasi K-Means.

Tanggung jawab Anggota 2 di file ini:
  1. Menyiapkan fitur numerik untuk K-Means.
  2. Menangani missing values yang relevan dengan fitur K-Means yang dipilih
     (di dataset ini: kolom Income memiliki 24 nilai kosong).
  3. Scaling fitur jika diperlukan.
  4. Elbow Method, lalu menentukan jumlah cluster berdasarkan hasilnya.
  5. Melatih model K-Means.
  6. Menghasilkan label cluster dan centroid.
  7. Visualisasi hasil clustering.
  8. Silhouette Score dan analisis karakteristik cluster.

Aturan batas tanggung jawab:
  - Fungsi menerima DataFrame ASLI dari eda.load_dataset() dan TIDAK mengubahnya
    (kerjakan pada salinan: df.copy()).
  - Jangan bergantung pada hasil K-Modes, dan jangan mengubah file eda.py / kmodes_model.py.
  - Simpan semua grafik & tabel ke folder output/ (awali nama file dengan "kmeans_"
    agar tidak tertukar dengan file EDA dan K-Modes).
  - Dataset TIDAK perlu dibaca ulang di sini; terima `df` dari main.py.

Referensi: Modul 13 - Clustering (sklearn.cluster.KMeans, Elbow Method, silhouette_score).
"""

from pathlib import Path

# TODO (Anggota 2): tambahkan import yang dibutuhkan, misalnya:
#   import numpy as np, pandas as pd, matplotlib.pyplot as plt
#   from sklearn.cluster import KMeans
#   from sklearn.preprocessing import StandardScaler
#   from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

# TODO (Anggota 2): tentukan konfigurasi di sini (daftar fitur numerik, rentang k untuk
# Elbow, random_state, dll.) supaya mudah diubah.


# ----------------------------------------------------------------------------
# 1. PERSIAPAN DATA K-MEANS
# ----------------------------------------------------------------------------
def prepare_kmeans_data(df):
    """
    TODO (Anggota 2):
      - Pilih fitur numerik yang masuk akal untuk K-Means (jangan pakai ID).
      - Tangani missing values (mis. Income) pada salinan data.
      - Lakukan scaling jika diperlukan.
    Return yang disarankan: data siap pakai (dan scaler bila perlu).
    """
    raise NotImplementedError("TODO Anggota 2: persiapan data K-Means")


# ----------------------------------------------------------------------------
# 2. ELBOW METHOD
# ----------------------------------------------------------------------------
def run_elbow(X, output_dir=None):
    """
    TODO (Anggota 2):
      - Hitung inertia untuk beberapa k, buat grafik Elbow, simpan ke output/.
      - Tentukan jumlah cluster berdasarkan hasil grafik (jelaskan alasannya).
    """
    raise NotImplementedError("TODO Anggota 2: Elbow Method")


# ----------------------------------------------------------------------------
# 3. TRAINING MODEL K-MEANS
# ----------------------------------------------------------------------------
def train_kmeans(X, n_clusters):
    """
    TODO (Anggota 2):
      - Latih model K-Means final.
      - Hasilkan label cluster dan centroid.
    """
    raise NotImplementedError("TODO Anggota 2: training K-Means")


# ----------------------------------------------------------------------------
# 4. VISUALISASI, EVALUASI, DAN PROFIL CLUSTER
# ----------------------------------------------------------------------------
def evaluate_and_profile(df, X, labels, output_dir=None):
    """
    TODO (Anggota 2):
      - Visualisasi hasil clustering (simpan ke output/).
      - Hitung Silhouette Score.
      - Analisis karakteristik tiap cluster.
    """
    raise NotImplementedError("TODO Anggota 2: visualisasi, Silhouette Score, profil cluster")


# ----------------------------------------------------------------------------
# 5. FUNGSI UTAMA K-MEANS (dipanggil oleh main.py)
# ----------------------------------------------------------------------------
def run_kmeans(df, output_dir=None):
    """
    TODO (Anggota 2): rangkai langkah 1-4 di atas menjadi satu fungsi.

    Antarmuka yang disepakati dengan eda.py & kmodes_model.py:
      - input : df (DataFrame asli dari load_dataset(), jangan diubah)
      - input : output_dir (default: folder output/)
      - return: dict berisi hasil (mis. labels, centroid, silhouette, profil cluster)
    """
    raise NotImplementedError("TODO Anggota 2: run_kmeans")


if __name__ == "__main__":
    print("kmeans_model.py masih berupa kerangka. Implementasi dikerjakan Anggota 2.")
