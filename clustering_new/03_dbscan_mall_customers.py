"""
03_dbscan_mall_customers.py
DBSCAN Clustering - Dataset Mall Customers (Anggota 3)

Jalankan dari folder mana pun:
    python 03_dbscan_mall_customers.py
Output tersimpan di: output/mall_customers/dbscan/
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ------------------------------------------------------------
# 1. KONFIGURASI
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Mall_Customers.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "mall_customers", "dbscan")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# Fitur sama dengan Fuzzy C-Means (02_fuzzy_cmeans_mall_customers.py)
# supaya hasil kedua metode bisa dibandingkan langsung.
FEATURES = [
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

MIN_SAMPLES_CANDIDATES = [3, 5, 8, 10]
EPS_CANDIDATES = [round(x, 2) for x in np.arange(0.2, 1.21, 0.1)]

sns.set_theme(style="whitegrid")


def judul(teks):
    print("\n" + "=" * 65)
    print(teks)
    print("=" * 65)


# ------------------------------------------------------------
# 2. MEMUAT DAN MEMERIKSA DATA
# ------------------------------------------------------------
judul("DBSCAN - MALL CUSTOMERS")

df = pd.read_csv(DATA_PATH)

print("\nInformasi dataset:")
print("Jumlah baris:", len(df))
print("Jumlah kolom:", len(df.columns))
print("\nLima data pertama:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 3. PREPROCESSING
# ------------------------------------------------------------
judul("PREPROCESSING")

data = df[FEATURES].copy()
for col in FEATURES:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data = data.replace([np.inf, -np.inf], np.nan)
for col in FEATURES:
    data[col] = data[col].fillna(data[col].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

print("Fitur yang digunakan:", FEATURES)
print("Fitur di-scaling dengan StandardScaler karena DBSCAN berbasis jarak")
print("(Euclidean), sehingga satuan Income (puluhan ribu) dan Spending")
print("Score (1-100) harus disamakan rentangnya dulu.")
print("Jumlah data:", X_scaled.shape[0])

# ------------------------------------------------------------
# 4. K-DISTANCE GRAPH UNTUK MEMPERKIRAKAN EPS
# ------------------------------------------------------------
judul("K-DISTANCE GRAPH (ESTIMASI EPS)")

K_FOR_KDIST = 5  # = calon min_samples yang umum dipakai sebagai acuan awal
nbrs = NearestNeighbors(n_neighbors=K_FOR_KDIST).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_distances, color="#2a6f97")
ax.set_xlabel("Titik data (diurutkan)")
ax.set_ylabel(f"Jarak ke tetangga ke-{K_FOR_KDIST}")
ax.set_title("K-Distance Graph - Mall Customers")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dbscan_kdistance_graph.png"), dpi=150)
plt.close(fig)

print(f"Grafik k-distance (k={K_FOR_KDIST}) disimpan.")
print("Titik 'tekuk' (knee) pada grafik ini jadi perkiraan awal nilai eps;")
print("nilai pasti tetap ditentukan lewat grid search di langkah berikut.")

# ------------------------------------------------------------
# 5. GRID SEARCH EPS & MIN_SAMPLES
# ------------------------------------------------------------
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
            # Silhouette butuh >=2 label unik pada titik non-noise
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
print("\n10 kombinasi dengan Silhouette Score tertinggi (abaikan yang NaN):")
print(
    grid_df.dropna(subset=["silhouette"])
    .sort_values("silhouette", ascending=False)
    .head(10)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 6. MEMILIH PARAMETER FINAL
# ------------------------------------------------------------
judul("PARAMETER FINAL")

valid = grid_df.dropna(subset=["silhouette"])
# Dari kombinasi dengan silhouette tertinggi, utamakan yang punya noise
# wajar (<15%) dan jumlah cluster masuk akal (2-8) supaya hasilnya
# bisa diinterpretasi, bukan sekadar memecah data jadi cluster kecil-kecil.
valid_wajar = valid[(valid["noise_pct"] < 15) & (valid["n_clusters"].between(2, 8))]
kandidat = valid_wajar if len(valid_wajar) > 0 else valid
best_row = kandidat.sort_values("silhouette", ascending=False).iloc[0]

FINAL_EPS = float(best_row["eps"])
FINAL_MIN_SAMPLES = int(best_row["min_samples"])

print(f"eps terpilih         : {FINAL_EPS}")
print(f"min_samples terpilih : {FINAL_MIN_SAMPLES}")
print(f"Alasan: kombinasi ini memberi Silhouette Score tertinggi "
      f"({best_row['silhouette']:.4f}) di antara kombinasi dengan jumlah "
      f"cluster yang masuk akal (2-8) dan persentase noise di bawah 15%, "
      f"sehingga hasil cluster cukup bermakna untuk diinterpretasikan "
      f"dan tidak didominasi oleh titik yang dibuang sebagai noise.")

# ------------------------------------------------------------
# 7. MODEL FINAL
# ------------------------------------------------------------
judul("MODEL FINAL DBSCAN")

dbscan_final = DBSCAN(eps=FINAL_EPS, min_samples=FINAL_MIN_SAMPLES)
labels = dbscan_final.fit_predict(X_scaled)

n_clusters_final = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_final = int((labels == -1).sum())

print(f"Jumlah cluster (tanpa noise) : {n_clusters_final}")
print(f"Jumlah titik noise           : {n_noise_final} "
      f"({100 * n_noise_final / len(labels):.2f}% dari data)")

result = df.copy()
result["Cluster"] = labels

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

result[["CustomerID", "Cluster"]].to_csv(
    os.path.join(OUTPUT_DIR, "dbscan_cluster_labels.csv"), index=False
)

# ------------------------------------------------------------
# 8. EVALUASI METRIK (titik noise dikeluarkan)
# ------------------------------------------------------------
judul("EVALUASI METRIK")

mask = labels != -1
if n_clusters_final >= 2:
    sil = silhouette_score(X_scaled[mask], labels[mask])
    dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
    ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
    print(f"Silhouette Score       : {sil:.4f}")
    print(f"Davies-Bouldin Index   : {dbi:.4f}")
    print(f"Calinski-Harabasz      : {ch:.4f}")
    print("(Dihitung hanya dari titik yang masuk cluster, titik noise dikeluarkan,")
    print(" karena ketiga metrik ini tidak didefinisikan untuk label noise.)")
else:
    sil = dbi = ch = np.nan
    print("Cluster valid kurang dari 2, metrik evaluasi tidak dapat dihitung.")

# ------------------------------------------------------------
# 9. PROFIL CLUSTER
# ------------------------------------------------------------
judul("PROFIL CLUSTER")

profile = result.groupby("Cluster")[FEATURES + ["Age"]].mean().round(2)
profile.to_csv(os.path.join(OUTPUT_DIR, "dbscan_cluster_profile.csv"))
print("Rata-rata fitur per cluster (Cluster -1 = noise):")
print(profile.to_string())

# ------------------------------------------------------------
# 10. VISUALISASI
# ------------------------------------------------------------
judul("VISUALISASI")

fig, ax = plt.subplots(figsize=(8, 6))
is_noise = result["Cluster"] == -1
sns.scatterplot(
    data=result[~is_noise], x=FEATURES[0], y=FEATURES[1],
    hue="Cluster", palette="viridis", s=60, ax=ax,
)
ax.scatter(
    result.loc[is_noise, FEATURES[0]], result.loc[is_noise, FEATURES[1]],
    c="lightgray", marker="x", s=50, label="Noise",
)
ax.set_title(f"DBSCAN - Mall Customers (eps={FINAL_EPS}, min_samples={FINAL_MIN_SAMPLES})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dbscan_cluster_visualization.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=sizes, x="Cluster", y="Jumlah_Anggota", hue="Cluster",
            palette="viridis", legend=False, ax=ax)
for i, v in enumerate(sizes["Jumlah_Anggota"]):
    ax.text(i, v + 0.5, str(v), ha="center")
ax.set_title("Distribusi Anggota Cluster DBSCAN (Mall Customers)")
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
km_best = KMeans(n_clusters=int(best_kmeans["k"]), init="k-means++", n_init=10,
                  random_state=RANDOM_STATE)
km_labels_best = km_best.fit_predict(X_scaled)

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

if n_clusters_final >= 2:
    if sil > best_kmeans["silhouette"]:
        kesimpulan = (
            f"DBSCAN (Silhouette {sil:.4f}) sedikit LEBIH BAIK dari K-Means "
            f"(Silhouette {best_kmeans['silhouette']:.4f}) pada dataset ini, "
            f"tapi DBSCAN juga membuang {n_noise_final} titik sebagai noise "
            f"({100 * n_noise_final / len(labels):.1f}%), sementara K-Means "
            f"mengelompokkan seluruh data tanpa terkecuali."
        )
    else:
        kesimpulan = (
            f"K-Means (Silhouette {best_kmeans['silhouette']:.4f}) LEBIH BAIK "
            f"dari DBSCAN (Silhouette {sil:.4f}) pada dataset ini. Pada data "
            f"dengan kerapatan antar cluster yang cukup merata seperti Mall "
            f"Customers, K-Means yang mengasumsikan cluster berbentuk bulat "
            f"cenderung lebih cocok dibanding DBSCAN yang mengandalkan "
            f"perbedaan kerapatan."
        )
else:
    kesimpulan = "DBSCAN tidak berhasil membentuk >=2 cluster yang valid pada parameter yang diuji, sehingga tidak dapat dibandingkan secara numerik dengan K-Means."
print("\nKesimpulan perbandingan:")
print(kesimpulan)

with open(os.path.join(OUTPUT_DIR, "dbscan_kesimpulan.txt"), "w", encoding="utf-8") as f:
    f.write(kesimpulan)

judul("SELESAI")
print(f"Semua output tersimpan di: {OUTPUT_DIR}")
for fp in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", fp)
