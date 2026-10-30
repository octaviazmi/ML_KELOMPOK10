
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "marketing_campaign.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "output", "marketing_campaign", "fuzzy_cmeans"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fitur yang digunakan untuk segmentasi pelanggan
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
    "NumWebVisitsMonth"
]

# ============================================================
# 2. MEMBACA DATASET
# ============================================================

print("=" * 65)
print("FUZZY C-MEANS - MARKETING CAMPAIGN")
print("=" * 65)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH, sep=None, engine="python")

print("\nInformasi dataset:")
print(f"Jumlah baris  : {df.shape[0]}")
print(f"Jumlah kolom  : {df.shape[1]}")

print("\nLima data pertama:")
print(df.head())

# ============================================================
# 3. PEMILIHAN FITUR DAN PREPROCESSING
# ============================================================

missing_features = [
    col for col in FEATURES if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Kolom berikut tidak ditemukan dalam dataset: {missing_features}"
    )

data = df[FEATURES].copy()

# Mengubah fitur menjadi numerik
for col in FEATURES:
    data[col] = pd.to_numeric(data[col], errors="coerce")

print("\nMissing values sebelum imputasi:")
print(data.isnull().sum())

# Mengganti nilai tidak valid dan missing dengan median
imputer = SimpleImputer(strategy="median")
data_imputed = imputer.fit_transform(data)

data_clean = pd.DataFrame(
    data_imputed,
    columns=FEATURES
)

# Menghapus baris dengan nilai infinity jika ada
data_clean = data_clean.replace(
    [np.inf, -np.inf], np.nan
).dropna()

print("\nJumlah data setelah preprocessing:", len(data_clean))

# Standardisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_clean)

print("\nPreprocessing selesai.")
print("Fitur yang digunakan:", FEATURES)
print("Jumlah fitur:", len(FEATURES))
print("Jumlah data:", len(X_scaled))

# ============================================================
# 4. PENGUJIAN JUMLAH CLUSTER
# ============================================================

print("\n" + "=" * 65)
print("PENGUJIAN JUMLAH CLUSTER")
print("=" * 65)

cluster_range = range(2, 7)
evaluation_results = []
model_results = {}

X = X_scaled.T

for k in cluster_range:

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=X,
        c=k,
        m=2,
        error=0.005,
        maxiter=1000,
        init=None,
        seed=42
    )

    labels = np.argmax(u, axis=0)

    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)

    evaluation_results.append({
        "Jumlah Cluster": k,
        "FPC": fpc,
        "Silhouette Score": sil,
        "Davies-Bouldin Index": dbi,
        "Calinski-Harabasz Score": ch,
        "Objective Function": jm[-1]
    })

    model_results[k] = {
        "centers": cntr,
        "membership": u,
        "labels": labels,
        "fpc": fpc,
        "objective": jm
    }

    print(f"\nk = {k}")
    print(f"FPC                    : {fpc:.4f}")
    print(f"Silhouette Score       : {sil:.4f}")
    print(f"Davies-Bouldin Index   : {dbi:.4f}")
    print(f"Calinski-Harabasz      : {ch:.4f}")
    print(f"Objective Function     : {jm[-1]:.4f}")

evaluation_df = pd.DataFrame(evaluation_results)

print("\nTabel evaluasi:")
print(evaluation_df.to_string(index=False))

# Memilih k berdasarkan Silhouette Score tertinggi
best_k = int(
    evaluation_df.loc[
        evaluation_df["Silhouette Score"].idxmax(),
        "Jumlah Cluster"
    ]
)

best_model = model_results[best_k]

print(f"\nJumlah cluster terpilih: {best_k}")

# ============================================================
# 5. MEMBERSHIP DEGREE
# ============================================================

membership = best_model["membership"]
labels = best_model["labels"]

membership_columns = [
    f"Membership_Cluster_{i}" for i in range(best_k)
]

membership_df = pd.DataFrame(
    membership.T,
    columns=membership_columns
)

membership_df.insert(
    0,
    "Cluster_Label",
    labels
)

if "ID" in df.columns:
    membership_df.insert(0, "ID", df["ID"].values)
else:
    membership_df.insert(
        0, "Data_Index", np.arange(len(membership_df))
    )

print("\nContoh membership degree:")
print(membership_df.head(10).to_string(index=False))

# ============================================================
# 6. PROFIL CLUSTER
# ============================================================

profile_data = data_clean.copy()
profile_data["Cluster"] = labels

cluster_profile = profile_data.groupby("Cluster").agg(
    Jumlah_Pelanggan=("Income", "count"),
    Rata_Rata_Income=("Income", "mean"),
    Rata_Rata_Recency=("Recency", "mean"),
    Rata_Rata_Wines=("MntWines", "mean"),
    Rata_Rata_Meat=("MntMeatProducts", "mean"),
    Rata_Rata_Web_Purchases=("NumWebPurchases", "mean"),
    Rata_Rata_Store_Purchases=("NumStorePurchases", "mean")
).round(2)

print("\nProfil cluster:")
print(cluster_profile)

# ============================================================
# 7. VISUALISASI EVALUASI
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = [
    ("FPC", "Fuzzy Partition Coefficient"),
    ("Silhouette Score", "Silhouette Score"),
    ("Davies-Bouldin Index", "Davies-Bouldin Index"),
    ("Calinski-Harabasz Score", "Calinski-Harabasz Score")
]

for ax, (column, title) in zip(axes.flatten(), metrics):
    sns.lineplot(
        data=evaluation_df,
        x="Jumlah Cluster",
        y=column,
        marker="o",
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Jumlah Cluster")
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "Evaluasi Fuzzy C-Means - Marketing Campaign",
    fontsize=15
)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "fcm_evaluation_plot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# 8. VISUALISASI CLUSTER
# ============================================================

# Visualisasi menggunakan dua fitur utama.
# Model tetap dilatih menggunakan seluruh fitur.
income_index = FEATURES.index("Income")
spending_index = FEATURES.index("MntWines")

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_scaled[:, income_index],
    X_scaled[:, spending_index],
    c=labels,
    cmap="viridis",
    s=35,
    alpha=0.7
)

centers = best_model["centers"]

plt.scatter(
    centers[:, income_index],
    centers[:, spending_index],
    marker="X",
    s=250,
    c="red",
    edgecolors="black",
    label="Cluster Center"
)

plt.xlabel("Income (Standardized)")
plt.ylabel("MntWines (Standardized)")
plt.title(
    f"Fuzzy C-Means Marketing Campaign (k={best_k})"
)
plt.legend()
plt.colorbar(scatter, label="Cluster")
plt.grid(alpha=0.3)

plt.savefig(
    os.path.join(OUTPUT_DIR, "fcm_cluster_visualization.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# 9. MENYIMPAN HASIL
# ============================================================

evaluation_df.to_csv(
    os.path.join(OUTPUT_DIR, "fcm_evaluation.csv"),
    index=False
)

membership_df.to_csv(
    os.path.join(OUTPUT_DIR, "fcm_membership_degree.csv"),
    index=False
)

cluster_profile.to_csv(
    os.path.join(OUTPUT_DIR, "fcm_cluster_profile.csv")
)

result_df = data_clean.copy()
result_df["Cluster_Label"] = labels

if "ID" in df.columns:
    result_df.insert(0, "ID", df["ID"].values)

result_df.to_csv(
    os.path.join(OUTPUT_DIR, "fcm_cluster_labels.csv"),
    index=False
)

# ============================================================
# 10. RINGKASAN
# ============================================================

print("\n" + "=" * 65)
print("RINGKASAN FUZZY C-MEANS MARKETING CAMPAIGN")
print("=" * 65)

print(f"Jumlah cluster final : {best_k}")
print(f"Jumlah pelanggan     : {len(data_clean)}")
print(f"Jumlah fitur         : {len(FEATURES)}")
print(f"FPC                  : {best_model['fpc']:.4f}")

best_row = evaluation_df[
    evaluation_df["Jumlah Cluster"] == best_k
].iloc[0]

print(
    f"Silhouette Score     : "
    f"{best_row['Silhouette Score']:.4f}"
)
print(
    f"Davies-Bouldin Index : "
    f"{best_row['Davies-Bouldin Index']:.4f}"
)

print("\nDistribusi cluster:")
print(pd.Series(labels).value_counts().sort_index())

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nPROGRAM FUZZY C-MEANS MARKETING CAMPAIGN SELESAI!")
