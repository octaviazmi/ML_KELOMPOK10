"""
06_dbscan_marketing_campaign.py
DBSCAN Clustering - Dataset Marketing Campaign (Anggota 3)

Jalankan dari folder mana pun:
    python 06_dbscan_marketing_campaign.py
Output tersimpan di: output/marketing_campaign/dbscan/
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# 1. KONFIGURASI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "marketing_campaign.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "marketing_campaign", "dbscan")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# Fitur sama dengan Fuzzy C-Means (05_fuzzy_cmeans_marketing_campaign.py)
# supaya hasil kedua metode bisa dibandingkan langsung.
FEATURES = [
    "Income",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
]

# min_samples disarankan >= 2 x jumlah fitur untuk data berdimensi tinggi
# (aturan umum Sander et al.), di sini 2 x 13 = 26, jadi kandidat dimulai
# dari sekitar nilai itu.
MIN_SAMPLES_CANDIDATES = [15, 20, 26, 35, 50]
EPS_CANDIDATES = [round(x, 1) for x in np.arange(1.0, 4.1, 0.25)]

sns.set_theme(style="whitegrid")


def judul(teks):
    print("\n" + "=" * 65)
    print(teks)
    print("=" * 65)


# 2. MEMUAT DAN MEMERIKSA DATA
judul("DBSCAN - MARKETING CAMPAIGN")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=None, engine="python")

print("\nInformasi dataset:")
print(f"Jumlah baris  : {df.shape[0]}")
print(f"Jumlah kolom  : {df.shape[1]}")
print("\nLima data pertama:")
print(df.head())

missing_features = [c for c in FEATURES if c not in df.columns]
if missing_features:
    raise ValueError(f"Kolom berikut tidak ditemukan dalam dataset: {missing_features}")

# 3. PREPROCESSING
judul("PREPROCESSING")

data = df[FEATURES].copy()
for col in FEATURES:
    data[col] = pd.to_numeric(data[col], errors="coerce")

print("Missing values sebelum imputasi:")
print(data.isnull().sum())

imputer = SimpleImputer(strategy="median")
data_clean = pd.DataFrame(imputer.fit_transform(data), columns=FEATURES)
data_clean = data_clean.replace([np.inf, -np.inf], np.nan).dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_clean)

print("\nFitur yang digunakan:", FEATURES)
print("Jumlah fitur:", len(FEATURES))
print("Jumlah data setelah preprocessing:", len(X_scaled))
print("Fitur di-scaling dengan StandardScaler agar DBSCAN (berbasis jarak")
print("Euclidean) tidak didominasi fitur dengan rentang nilai terbesar")
print("(mis. MntWines puluhan-ribuan vs NumWebVisitsMonth 0-20).")

# 4. K-DISTANCE GRAPH UNTUK MEMPERKIRAKAN EPS
judul("K-DISTANCE GRAPH (ESTIMASI EPS)")

K_FOR_KDIST = 26  # ~ 2 x jumlah fitur, acuan awal min_samples
nbrs = NearestNeighbors(n_neighbors=K_FOR_KDIST).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_distances, color="#2a6f97")
ax.set_xlabel("Titik data (diurutkan)")
ax.set_ylabel(f"Jarak ke tetangga ke-{K_FOR_KDIST}")
ax.set_title("K-Distance Graph - Marketing Campaign")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dbscan_kdistance_graph.png"), dpi=150)
plt.close(fig)
print(f"Grafik k-distance (k={K_FOR_KDIST}) disimpan sebagai perkiraan awal eps.")

# 5. GRID SEARCH EPS & MIN_SAMPLES
judul("PENGUJIAN KOMBINASI eps DAN min_samples")

grid_rows = []
for min_samples in MIN_SAMPLES_CANDIDATES:
    for eps in EPS_CANDIDATES:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        noise_pct = round(100 * n_noise / len(labels), 2)

        if n_clusters >= 2 and n_noise < len(labels):
            mask = labels != -1
            if len(set(labels[mask])) >= 2:
                sil = silhouette_score(X_scaled[mask], labels[mask])
            else:
                sil = np.nan
        else:
            sil = np.nan

        grid_rows.append({
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": noise_pct,
            "silhouette": sil,
        })

grid_df = pd.DataFrame(grid_rows)
grid_df.to_csv(os.path.join(OUTPUT_DIR, "dbscan_grid_search.csv"), index=False)

print(f"Total kombinasi diuji: {len(grid_df)}")
print("\nRingkasan: jumlah cluster yang dihasilkan tiap kombinasi (agar")
print("kelihatan pola umum pada data berdimensi tinggi ini):")
print(grid_df.pivot(index="min_samples", columns="eps", values="n_clusters").to_string())
print("\n10 kombinasi dengan Silhouette Score tertinggi (abaikan yang NaN):")
print(
    grid_df.dropna(subset=["silhouette"])
    .sort_values("silhouette", ascending=False)
    .head(10)
    .to_string(index=False)
)

# 6. MEMILIH PARAMETER FINAL
judul("PARAMETER FINAL")

valid = grid_df.dropna(subset=["silhouette"])
valid_wajar = valid[(valid["noise_pct"] < 30) & (valid["n_clusters"].between(2, 8))]
kandidat = valid_wajar if len(valid_wajar) > 0 else valid

if len(kandidat) > 0:
    best_row = kandidat.sort_values("silhouette", ascending=False).iloc[0]
    FINAL_EPS = float(best_row["eps"])
    FINAL_MIN_SAMPLES = int(best_row["min_samples"])
    print(f"eps terpilih         : {FINAL_EPS}")
    print(f"min_samples terpilih : {FINAL_MIN_SAMPLES}")
    print(f"Alasan: kombinasi ini memberi Silhouette Score tertinggi "
          f"({best_row['silhouette']:.4f}) di antara kombinasi dengan jumlah "
          f"cluster masuk akal (2-8) dan noise di bawah 30%. Ambang noise "
          f"dilonggarkan dibanding dataset Mall Customers karena pada 13 "
          f"fitur numerik (dimensi tinggi), titik data jadi lebih tersebar "
          f"(curse of dimensionality) sehingga DBSCAN wajar membuang lebih "
          f"banyak titik sebagai noise.")
else:
    # fallback: tidak ada kombinasi yang membentuk >=2 cluster valid sama sekali
    FINAL_EPS = float(grid_df.loc[grid_df["n_clusters"].idxmax(), "eps"])
    FINAL_MIN_SAMPLES = int(grid_df.loc[grid_df["n_clusters"].idxmax(), "min_samples"])
    print("Tidak ada kombinasi yang menghasilkan cluster dengan noise wajar.")
    print(f"Dipakai kombinasi dengan jumlah cluster terbanyak sebagai fallback: "
          f"eps={FINAL_EPS}, min_samples={FINAL_MIN_SAMPLES}")

# 7. MODEL FINAL
judul("MODEL FINAL DBSCAN")

dbscan_final = DBSCAN(eps=FINAL_EPS, min_samples=FINAL_MIN_SAMPLES)
labels = dbscan_final.fit_predict(X_scaled)

n_clusters_final = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_final = int((labels == -1).sum())

print(f"Jumlah cluster (tanpa noise) : {n_clusters_final}")
print(f"Jumlah titik noise           : {n_noise_final} "
      f"({100 * n_noise_final / len(labels):.2f}% dari data)")

result = data_clean.copy()
result["Cluster"] = labels
if "ID" in df.columns:
    result.insert(0, "ID", df.loc[data_clean.index, "ID"].values)

sizes = (
    result["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Jumlah_Anggota")
)
sizes["Keterangan"] = sizes["Cluster"].apply(lambda c: "Noise" if c == -1 else "Cluster")
sizes["Persentase"] = (sizes["Jumlah_Anggota"] / len(result) * 100).round(2)
sizes.to_csv(os.path.join(OUTPUT_DIR, "dbscan_cluster_sizes.csv"), index=False)
print("\nJumlah anggota per cluster (Cluster -1 = noise):")
print(sizes.to_string(index=False))

label_cols = (["ID"] if "ID" in result.columns else []) + ["Cluster"]
result[label_cols].to_csv(os.path.join(OUTPUT_DIR, "dbscan_cluster_labels.csv"), index=False)

# 8. EVALUASI METRIK (titik noise dikeluarkan)
judul("EVALUASI METRIK")

mask = labels != -1
if n_clusters_final >= 2:
    sil = silhouette_score(X_scaled[mask], labels[mask])
    dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
    ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
    print(f"Silhouette Score       : {sil:.4f}")
    print(f"Davies-Bouldin Index   : {dbi:.4f}")
    print(f"Calinski-Harabasz      : {ch:.4f}")
    print("(Dihitung hanya dari titik yang masuk cluster, titik noise dikeluarkan.)")
else:
    sil = dbi = ch = np.nan
    print("Cluster valid kurang dari 2, metrik evaluasi tidak dapat dihitung.")

# 9. PROFIL CLUSTER
judul("PROFIL CLUSTER")

profile = result.groupby("Cluster")[FEATURES].mean().round(2)
profile.to_csv(os.path.join(OUTPUT_DIR, "dbscan_cluster_profile.csv"))
print("Rata-rata fitur per cluster (Cluster -1 = noise):")
print(profile.to_string())

# 10. VISUALISASI (reduksi ke 2D dengan PCA karena 13 fitur)
judul("VISUALISASI (PCA 2D)")

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_.sum() * 100
print(f"Dataset punya {len(FEATURES)} fitur sehingga tidak bisa divisualisasikan")
print(f"langsung; PCA dipakai HANYA untuk visualisasi (bukan untuk clustering),")
print(f"2 komponen utama menjelaskan {var_explained:.1f}% variansi data.")

viz = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
viz["Cluster"] = labels

fig, ax = plt.subplots(figsize=(8, 6))
is_noise = viz["Cluster"] == -1
sns.scatterplot(
    data=viz[~is_noise], x="PC1", y="PC2",
    hue="Cluster", palette="viridis", s=40, alpha=0.8, ax=ax,
)
ax.scatter(
    viz.loc[is_noise, "PC1"], viz.loc[is_noise, "PC2"],
    c="lightgray", marker="x", s=25, alpha=0.6, label="Noise",
)
ax.set_title(f"DBSCAN - Marketing Campaign, proyeksi PCA 2D\n"
             f"(eps={FINAL_EPS}, min_samples={FINAL_MIN_SAMPLES})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variansi)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variansi)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dbscan_cluster_visualization_pca.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=sizes, x="Cluster", y="Jumlah_Anggota", hue="Cluster",
            palette="viridis", legend=False, ax=ax)
for i, v in enumerate(sizes["Jumlah_Anggota"]):
    ax.text(i, v + 2, str(v), ha="center", fontsize=8)
ax.set_title("Distribusi Anggota Cluster DBSCAN (Marketing Campaign)")
ax.set_ylabel("Jumlah pelanggan")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dbscan_cluster_distribution.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------
# 11. PERBANDINGAN DENGAN K-MEANS
# ------------------------------------------------------------
judul("PERBANDINGAN DENGAN K-MEANS")

kmeans_scores = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    km_labels = km.fit_predict(X_scaled)
    kmeans_scores.append({
        "k": k,
        "silhouette": silhouette_score(X_scaled, km_labels),
        "dbi": davies_bouldin_score(X_scaled, km_labels),
        "ch": calinski_harabasz_score(X_scaled, km_labels),
    })
kmeans_df = pd.DataFrame(kmeans_scores)
best_kmeans = kmeans_df.loc[kmeans_df["silhouette"].idxmax()]

comparison = pd.DataFrame([
    {
        "Metode": "K-Means (baseline)",
        "Jumlah cluster": int(best_kmeans["k"]),
        "Silhouette Score": round(best_kmeans["silhouette"], 4),
        "Davies-Bouldin Index": round(best_kmeans["dbi"], 4),
        "Calinski-Harabasz": round(best_kmeans["ch"], 1),
        "Noise": "-",
    },
    {
        "Metode": "DBSCAN",
        "Jumlah cluster": n_clusters_final,
        "Silhouette Score": round(sil, 4) if n_clusters_final >= 2 else None,
        "Davies-Bouldin Index": round(dbi, 4) if n_clusters_final >= 2 else None,
        "Calinski-Harabasz": round(ch, 1) if n_clusters_final >= 2 else None,
        "Noise": f"{n_noise_final} ({100 * n_noise_final / len(labels):.1f}%)",
    },
])
comparison.to_csv(os.path.join(OUTPUT_DIR, "dbscan_vs_kmeans.csv"), index=False)
print(comparison.to_string(index=False))

cluster_sizes_only = sizes[sizes["Cluster"] != -1]["Jumlah_Anggota"]
cluster_tidak_seimbang = (
    n_clusters_final >= 2 and cluster_sizes_only.min() / cluster_sizes_only.sum() < 0.05
)

if n_clusters_final >= 2:
    if sil > best_kmeans["silhouette"] and not cluster_tidak_seimbang:
        kesimpulan = (
            f"DBSCAN (Silhouette {sil:.4f}) LEBIH BAIK dari K-Means "
            f"(Silhouette {best_kmeans['silhouette']:.4f}) pada dataset ini, "
            f"meski membuang {n_noise_final} titik sebagai noise "
            f"({100 * n_noise_final / len(labels):.1f}%)."
        )
    elif sil > best_kmeans["silhouette"] and cluster_tidak_seimbang:
        terkecil = int(cluster_sizes_only.min())
        kesimpulan = (
            f"Secara angka, Silhouette Score DBSCAN ({sil:.4f}) lebih tinggi "
            f"dari K-Means ({best_kmeans['silhouette']:.4f}), TAPI hasil ini "
            f"menyesatkan jika dibaca apa adanya: dari {n_clusters_final} "
            f"cluster yang terbentuk, cluster terkecil cuma berisi {terkecil} "
            f"pelanggan ({100*terkecil/cluster_sizes_only.sum():.1f}% dari data "
            f"yang ter-cluster), sementara DBSCAN juga membuang "
            f"{n_noise_final} titik ({100 * n_noise_final / len(labels):.1f}%) "
            f"sebagai noise. Silhouette tinggi di sini lebih disebabkan oleh "
            f"cluster kecil yang sangat rapat, bukan karena DBSCAN menemukan "
            f"segmentasi pelanggan yang lebih bermakna. Untuk kebutuhan "
            f"segmentasi pemasaran yang perlu kelompok berukuran wajar, "
            f"K-Means lebih praktis dipakai pada data 13 fitur ini."
        )
    else:
        kesimpulan = (
            f"K-Means (Silhouette {best_kmeans['silhouette']:.4f}) LEBIH BAIK "
            f"dari DBSCAN (Silhouette {sil:.4f}) pada dataset ini. Dengan 13 "
            f"fitur numerik, kerapatan titik data jadi tidak merata (curse of "
            f"dimensionality), sehingga sulit bagi DBSCAN menemukan parameter "
            f"eps tunggal yang cocok untuk seluruh ruang fitur; K-Means yang "
            f"tidak bergantung pada kerapatan lokal lebih stabil di kondisi ini."
        )
else:
    kesimpulan = (
        "DBSCAN tidak berhasil membentuk >=2 cluster yang valid pada 13 fitur "
        "ini walau sudah diuji banyak kombinasi eps dan min_samples. Ini "
        "contoh nyata curse of dimensionality: pada data berdimensi tinggi, "
        "jarak antar titik cenderung menjadi hampir seragam sehingga DBSCAN "
        "sulit membedakan wilayah padat dan jarang. K-Means tetap berhasil "
        "membentuk cluster (lihat tabel perbandingan) karena tidak "
        "bergantung pada kerapatan lokal."
    )
print("\nKesimpulan perbandingan:")
print(kesimpulan)

with open(os.path.join(OUTPUT_DIR, "dbscan_kesimpulan.txt"), "w", encoding="utf-8") as f:
    f.write(kesimpulan)

judul("SELESAI")
print(f"Semua output tersimpan di: {OUTPUT_DIR}")
for fp in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", fp)
