"""
01_kmodes_mall_customers.py
K-Modes Clustering - Dataset Mall Customers (Anggota A)

Jalankan dari folder mana pun:
    python 01_kmodes_mall_customers.py
Output tersimpan di: output/mall_customers/kmodes/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from kmodes.kmodes import KModes

# ----------------------------------------------------------------------------
# PARAMETER (nanti yeaahhhhh)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# PATH 
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Mall_Customers.csv"
OUT_DIR = BASE_DIR / "output" / "mall_customers" / "kmodes"
OUT_DIR.mkdir(parents=True, exist_ok=True)  
sns.set_theme(style="whitegrid")


def judul(teks):
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


# ----------------------------------------------------------------------------
# 1. BACA DATA & PERIKSA STRUKTUR
# ----------------------------------------------------------------------------
judul("1. STRUKTUR DATA")
df = pd.read_csv(DATA_PATH)  # dataset asli tidak diubah
print(f"Jumlah baris, kolom : {df.shape}")
print("\nTipe data:")
print(df.dtypes)
print("\n5 baris pertama:")
print(df.head())

judul("2. MISSING VALUES")
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing values: {int(missing.sum())}")
print(f"Baris duplikat      : {int(df.duplicated().sum())}")

# ----------------------------------------------------------------------------
# 3. PILIH FITUR & KATEGORISASI
# ----------------------------------------------------------------------------
judul("3. KATEGORISASI FITUR NUMERIK")

data_cat = pd.DataFrame(index=df.index)
data_cat["Gender"] = df["Gender"].astype(str)

# Usia: Muda (<=25), Dewasa Muda (26-35), Dewasa (36-50), Senior (>50)
data_cat["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 50, 200],
    labels=["Muda (<=25)", "Dewasa Muda (26-35)", "Dewasa (36-50)", "Senior (>50)"],
).astype(str)

# Pendapatan tahunan (k$): Rendah (<40), Sedang (40-70), Tinggi (>70)
data_cat["Income_Level"] = pd.cut(
    df["Annual Income (k$)"],
    bins=[0, 39, 70, 1000],
    labels=["Rendah (<40k)", "Sedang (40-70k)", "Tinggi (>70k)"],
).astype(str)

# Skor belanja (1-100): Rendah (1-33), Sedang (34-66), Tinggi (67-100)
data_cat["Spending_Level"] = pd.cut(
    df["Spending Score (1-100)"],
    bins=[0, 33, 66, 100],
    labels=["Rendah (1-33)", "Sedang (34-66)", "Tinggi (67-100)"],
).astype(str)

FEATURES = list(data_cat.columns)
print(f"Fitur untuk K-Modes: {FEATURES}")
for col in FEATURES:
    print(f"\n{col}:")
    print(data_cat[col].value_counts())

# ----------------------------------------------------------------------------
# 4. K-MODES UNTUK BEBERAPA KANDIDAT K -> HITUNG COST
# ----------------------------------------------------------------------------
judul("4. COST K-MODES UNTUK k = %d..%d" % (min(K_RANGE), max(K_RANGE)))
X = data_cat[FEATURES].values

cost_rows = []
for k in K_RANGE:
    km = KModes(
        n_clusters=k,
        init=INIT_METHOD,
        n_init=N_INIT,
        random_state=RANDOM_STATE,
    )
    km.fit(X)
    cost_rows.append({"k": k, "cost": km.cost_})
    print(f"k = {k:2d} | cost = {km.cost_:.0f}")

cost_df = pd.DataFrame(cost_rows)
cost_df["penurunan_cost"] = -cost_df["cost"].diff()
cost_df.to_csv(OUT_DIR / "kmodes_cost.csv", index=False)

# Grafik cost vs jumlah cluster (elbow)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cost_df["k"], cost_df["cost"], marker="o", color="#2a6f97")
ax.axvline(FINAL_K, color="#c1121f", linestyle="--", label=f"k final = {FINAL_K}")
ax.set_xlabel("Jumlah cluster (k)")
ax.set_ylabel("Cost K-Modes")
ax.set_title("K-Modes - Cost terhadap Jumlah Cluster (Mall Customers)")
ax.set_xticks(list(K_RANGE))
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "kmodes_cost_plot.png", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

# ----------------------------------------------------------------------------
# 5. MODEL FINAL
# ----------------------------------------------------------------------------
judul(f"5. MODEL FINAL (k = {FINAL_K})")
km_final = KModes(
    n_clusters=FINAL_K,
    init=INIT_METHOD,
    n_init=N_INIT,
    random_state=RANDOM_STATE,
)
labels = km_final.fit_predict(X)
print(f"Cost model final: {km_final.cost_:.0f}")

result = df.copy()
result["Cluster"] = labels
result_cat = data_cat.copy()
result_cat["Cluster"] = labels

# Label cluster per pelanggan (data asli + versi kategori)
labels_out = pd.concat(
    [df[["CustomerID"]], data_cat[FEATURES], result[["Cluster"]]], axis=1
)
labels_out.to_csv(OUT_DIR / "kmodes_cluster_labels.csv", index=False)

# Jumlah anggota tiap cluster
sizes = (
    result["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Jumlah_Anggota")
)
sizes["Persentase"] = (sizes["Jumlah_Anggota"] / len(result) * 100).round(2)
sizes.to_csv(OUT_DIR / "kmodes_cluster_sizes.csv", index=False)
print("\nJumlah anggota per cluster:")
print(sizes.to_string(index=False))

# Centroid (modus) tiap cluster
centroids = pd.DataFrame(km_final.cluster_centroids_, columns=FEATURES)
centroids.insert(0, "Cluster", range(FINAL_K))
centroids.to_csv(OUT_DIR / "kmodes_centroids.csv", index=False)
print("\nCentroid (modus) tiap cluster:")
print(centroids.to_string(index=False))

# ----------------------------------------------------------------------------
# 6. PROFIL KATEGORI TIAP CLUSTER
# ----------------------------------------------------------------------------
judul("6. PROFIL KATEGORI TIAP CLUSTER")
profile_rows = []
for col in FEATURES:
    counts = pd.crosstab(result_cat["Cluster"], result_cat[col])
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    for cl in counts.index:
        for cat in counts.columns:
            profile_rows.append(
                {
                    "Cluster": cl,
                    "Fitur": col,
                    "Kategori": cat,
                    "Jumlah": int(counts.loc[cl, cat]),
                    "Persen_dalam_Cluster": round(pct.loc[cl, cat], 2),
                }
            )
profile = pd.DataFrame(profile_rows)
profile.to_csv(OUT_DIR / "kmodes_cluster_profile.csv", index=False)

# Ringkasan numerik asli per cluster (hanya untuk referensi profil)
num_summary = (
    result.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    .agg(["mean", "min", "max"])
    .round(2)
)
num_summary.to_csv(OUT_DIR / "kmodes_cluster_numeric_summary.csv")
print(num_summary)

# ----------------------------------------------------------------------------
# 7. GRAFIK DISTRIBUSI CLUSTER
# ----------------------------------------------------------------------------
# (a) jumlah anggota per cluster
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=sizes, x="Cluster", y="Jumlah_Anggota", hue="Cluster",
            palette="viridis", legend=False, ax=ax)
for i, v in enumerate(sizes["Jumlah_Anggota"]):
    ax.text(i, v + 0.5, str(v), ha="center")
ax.set_title("Distribusi Anggota Cluster K-Modes (Mall Customers)")
ax.set_ylabel("Jumlah pelanggan")
fig.tight_layout()
fig.savefig(OUT_DIR / "kmodes_cluster_distribution.png", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

# (b) komposisi kategori tiap fitur per cluster
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, col in zip(axes.ravel(), FEATURES):
    ct = pd.crosstab(result_cat["Cluster"], result_cat[col], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Persen (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.suptitle("Profil Kategori per Cluster - K-Modes (Mall Customers)")
fig.tight_layout()
fig.savefig(OUT_DIR / "kmodes_cluster_profile.png", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

# (c) sebaran Income vs Spending Score berwarna menurut cluster (referensi visual)
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=result, x="Annual Income (k$)", y="Spending Score (1-100)",
                hue="Cluster", palette="viridis", s=60, ax=ax)
ax.set_title("Sebaran Income vs Spending Score per Cluster K-Modes")
fig.tight_layout()
fig.savefig(OUT_DIR / "kmodes_scatter_income_spending.png", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

judul("SELESAI")
print(f"Semua output tersimpan di: {OUT_DIR}")
for f in sorted(OUT_DIR.iterdir()):
    print(" -", f.name)
