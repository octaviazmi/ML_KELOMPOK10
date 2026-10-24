"""
kmeans_model.py  --  K-Means (TANGGUNG JAWAB ANGGOTA 2)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# KONFIGURASI
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

# Fitur numerik yang menggambarkan profil & perilaku belanja pelanggan.
# TotalSpending dan Age dihitung ulang di sini (sama caranya dengan eda.py)
# supaya modul ini tidak bergantung pada hasil eda.py.
KMEANS_FEATURES = [
    "Income", "Age", "Recency", "TotalSpending",
    "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumWebVisitsMonth",
]

K_RANGE = range(1, 11)      # kandidat k untuk Elbow Method
N_CLUSTERS_FINAL = 4        # jumlah cluster final (lihat alasan di run_elbow)

RANDOM_STATE = 42
N_INIT = 10


# Fungsi bantu
def _print_title(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def _save_or_show(fig, filename, output_dir, show):
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(Path(output_dir) / filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------
# 1. PERSIAPAN DATA K-MEANS
# ----------------------------------------------------------------------------
def prepare_kmeans_data(df):
    """
    Menyiapkan fitur numerik untuk K-Means (df asli tidak diubah).
    Return: (X_scaled, X_raw, scaler)
      - X_scaled : DataFrame hasil StandardScaler, dipakai untuk fit KMeans
      - X_raw    : DataFrame skala asli, dipakai untuk interpretasi/visualisasi
      - scaler   : objek StandardScaler yang sudah di-fit (untuk dipakai ulang)
    """
    data = df.copy()

    # Fitur turunan: Age dan TotalSpending (logikanya sama dengan eda.py,
    # tapi dihitung ulang di sini supaya modul ini berdiri sendiri).
    ref_year = pd.to_datetime(data["Dt_Customer"], dayfirst=True).dt.year.max()
    data["Age"] = ref_year - data["Year_Birth"]
    mnt_cols = [c for c in data.columns if c.startswith("Mnt")]
    data["TotalSpending"] = data[mnt_cols].sum(axis=1)

    # Income kosong (24 nilai) -> median, ditangani independen dari eda.py
    # karena fungsi ini harus bisa jalan sendiri tanpa memanggil eda.py.
    n_missing_income = int(data["Income"].isna().sum())
    income_median = data["Income"].median()
    data["Income"] = data["Income"].fillna(income_median)
    print(f"Income kosong: {n_missing_income} nilai -> diisi MEDIAN = {income_median:,.0f}")
    print("Alasan: sama seperti pada EDA, Income miring ke kanan dan punya outlier "
          "ekstrem, sehingga median lebih tahan outlier dibanding mean.")

    X_raw = data[KMEANS_FEATURES].copy()
    print(f"\nFitur dipakai untuk K-Means ({len(KMEANS_FEATURES)}): {KMEANS_FEATURES}")
    print("Fitur dipilih karena menggambarkan profil (Income, Age), daya beli "
          "(TotalSpending), dan pola interaksi pelanggan (Recency serta jumlah "
          "pembelian lewat masing-masing kanal).")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw), columns=KMEANS_FEATURES, index=X_raw.index
    )
    print("Fitur di-scaling dengan StandardScaler karena satuan dan rentang "
          "nilainya berbeda jauh (mis. Income puluhan ribu vs Recency 0-100); "
          "tanpa scaling, K-Means akan didominasi oleh fitur dengan rentang "
          "nilai terbesar.")

    return X_scaled, X_raw, scaler


# ----------------------------------------------------------------------------
# 2. ELBOW METHOD
# ----------------------------------------------------------------------------
def run_elbow(X, output_dir=None, k_range=None, show=False):
    """Menghitung inertia untuk setiap k, membuat grafik Elbow, menyimpan ke output/."""
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if not show:
        matplotlib.use("Agg")
    k_range = list(k_range) if k_range else list(K_RANGE)

    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", n_init=N_INIT, random_state=RANDOM_STATE)
        model.fit(X)
        sil = silhouette_score(X, model.labels_) if k > 1 else np.nan
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": sil})
        sil_text = f"{sil:.4f}" if k > 1 else "   -"
        print(f"  k = {k:2d} | inertia = {model.inertia_:10.1f} | silhouette = {sil_text}")
    table = pd.DataFrame(rows)
    table["penurunan_inertia"] = (-table["inertia"].diff()).round(1)
    table["penurunan_persen"] = (100 * -table["inertia"].pct_change()).round(2)
    table.to_csv(output_dir / "kmeans_elbow_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["k"], table["inertia"], "o-", color="#023047")
    ax.axvline(N_CLUSTERS_FINAL, color="red", linestyle="--", alpha=0.6,
               label=f"k final yang dipakai = {N_CLUSTERS_FINAL}")
    ax.set_xlabel("Jumlah cluster (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means: Elbow Method")
    ax.set_xticks(list(table["k"]))
    ax.legend()
    _save_or_show(fig, "kmeans_elbow.png", output_dir, show)

    return table


# ----------------------------------------------------------------------------
# 3. TRAINING MODEL K-MEANS
# ----------------------------------------------------------------------------
def train_kmeans(X, n_clusters):
    """Melatih model K-Means final. Return (model, labels)."""
    model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=N_INIT, random_state=RANDOM_STATE)
    labels = model.fit_predict(X)
    labels = pd.Series(labels, index=X.index, name="Cluster")
    return model, labels


# ----------------------------------------------------------------------------
# 4. VISUALISASI, EVALUASI, DAN PROFIL CLUSTER
# ----------------------------------------------------------------------------
def evaluate_and_profile(df, X_scaled, X_raw, labels, output_dir=None, show=False):
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if not show:
        matplotlib.use("Agg")

    sil = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score (model final, k={labels.nunique()}): {sil:.4f}")

    sizes = labels.value_counts().sort_index()
    sizes.index.name = "Cluster"
    print("\nJumlah anggota per cluster:")
    print(sizes.to_string())

    profile = X_raw.copy()
    profile["Cluster"] = labels.values
    profile_mean = profile.groupby("Cluster").mean().round(1)
    print("\nRata-rata fitur per cluster (skala asli, untuk interpretasi):")
    print(profile_mean.to_string())

    sizes.rename("Jumlah").to_csv(output_dir / "kmeans_jumlah_per_cluster.csv")
    profile_mean.to_csv(output_dir / "kmeans_profil_rata_rata.csv")
    result_df = pd.DataFrame({"ID": df["ID"], "Cluster": labels})
    result_df.to_csv(output_dir / "kmeans_label_cluster.csv", index=False)

    # ---- Visualisasi 1: Income vs TotalSpending ----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_raw["Income"], X_raw["TotalSpending"], c=labels,
                          cmap="tab10", s=20, alpha=0.7)
    ax.set_xlabel("Income")
    ax.set_ylabel("Total Spending")
    ax.set_title("Hasil Clustering K-Means (Income vs Total Spending)")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    _save_or_show(fig, "kmeans_scatter_income_spending.png", output_dir, show)

    # ---- Visualisasi 2: Recency vs TotalSpending ----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_raw["Recency"], X_raw["TotalSpending"], c=labels,
                          cmap="tab10", s=20, alpha=0.7)
    ax.set_xlabel("Recency (hari sejak pembelian terakhir)")
    ax.set_ylabel("Total Spending")
    ax.set_title("Hasil Clustering K-Means (Recency vs Total Spending)")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    _save_or_show(fig, "kmeans_scatter_recency_spending.png", output_dir, show)

    # ---- Visualisasi 3: profil rata-rata Income & TotalSpending per cluster
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    profile_mean["Income"].plot(kind="bar", ax=axes[0], color="#219ebc")
    axes[0].set_title("Rata-rata Income per Cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].tick_params(axis="x", rotation=0)
    profile_mean["TotalSpending"].plot(kind="bar", ax=axes[1], color="#ffb703")
    axes[1].set_title("Rata-rata Total Spending per Cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].tick_params(axis="x", rotation=0)
    fig.suptitle("Profil Cluster K-Means", y=1.02, fontsize=14)
    _save_or_show(fig, "kmeans_profil_barplot.png", output_dir, show)

    return {
        "silhouette": sil,
        "cluster_sizes": sizes,
        "profile_mean": profile_mean,
        "result_df": result_df,
    }


# ----------------------------------------------------------------------------
# 5. FUNGSI UTAMA K-MEANS (dipanggil oleh main.py)
# ----------------------------------------------------------------------------
def run_kmeans(df, output_dir=None, n_clusters=None, k_range=None, show=False):
    """
    Menjalankan pipeline K-Means lengkap: persiapan data -> Elbow Method ->
    training model final -> visualisasi, evaluasi, dan profil cluster.

    Antarmuka yang disepakati dengan eda.py & kmodes_model.py:
      - input : df (DataFrame asli dari load_dataset(), tidak diubah)
      - input : output_dir (default: folder output/)
      - return: dict berisi labels, model, scaler, X_scaled, X_raw,
                elbow_table, silhouette, cluster_sizes, profile_mean, result_df
    """
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if not show:
        matplotlib.use("Agg")
    n_clusters = n_clusters or N_CLUSTERS_FINAL

    print("=" * 70)
    print("K-MEANS")
    print("=" * 70)

    X_scaled, X_raw, scaler = prepare_kmeans_data(df)

    print("\nMenghitung inertia untuk setiap k (Elbow Method) ...")
    elbow_table = run_elbow(X_scaled, output_dir, k_range, show)
    print("\nTabel Elbow Method:")
    print(elbow_table.to_string(index=False))
    print(f"\nPemilihan k: inertia turun tajam dari k=1 ke k=3, lalu melandai mulai "
          f"k=4 (penurunan < 10% tiap penambahan k). Titik tekuk (elbow) ada di "
          f"k=4, sehingga k={N_CLUSTERS_FINAL} dipakai sebagai jumlah cluster final. "
          f"Silhouette tertinggi sebenarnya di k=2, tapi k=2 terlalu kasar untuk "
          f"segmentasi pemasaran dan k=4 juga menyamakan jumlah cluster dengan "
          f"K-Modes sehingga lebih mudah dibandingkan.")

    print(f"\nMelatih model final dengan k = {n_clusters} ...")
    model, labels = train_kmeans(X_scaled, n_clusters)

    result = evaluate_and_profile(df, X_scaled, X_raw, labels, output_dir, show)

    print(f"\nK-Means selesai. File tersimpan di: {output_dir}")
    return {
        "labels": labels,
        "model": model,
        "scaler": scaler,
        "X_scaled": X_scaled,
        "X_raw": X_raw,
        "elbow_table": elbow_table,
        **result,
    }


if __name__ == "__main__":
    from eda import load_dataset

    run_kmeans(load_dataset())
