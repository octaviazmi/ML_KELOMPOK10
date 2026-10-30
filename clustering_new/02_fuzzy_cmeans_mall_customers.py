import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. KONFIGURASI
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Mall_Customers.csv")

OUTPUT_DIR = os.path.join(
    BASE_DIR, "output", "mall_customers", "fuzzy_cmeans"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
FUZZINESS = 2
CLUSTER_RANGE = range(2, 7)

FEATURES = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# 2. MEMUAT DAN MEMERIKSA DATA
# ------------------------------------------------------------

print("=" * 65)
print("FUZZY C-MEANS - MALL CUSTOMERS")
print("=" * 65)

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

data = df[FEATURES].copy()

for column in FEATURES:
    data[column] = pd.to_numeric(
        data[column], errors="coerce"
    )

data = data.replace([np.inf, -np.inf], np.nan)

# Mengisi missing values dengan median
for column in FEATURES:
    data[column] = data[column].fillna(
        data[column].median()
    )

# Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Fuzzy C-Means membutuhkan format fitur x sampel
X_fcm = X_scaled.T

print("\nPreprocessing selesai.")
print("Fitur yang digunakan:", FEATURES)
print("Jumlah data:", X_scaled.shape[0])


# ------------------------------------------------------------
# 4. PENGUJIAN JUMLAH CLUSTER
# ------------------------------------------------------------

evaluation_results = []

print("\n" + "=" * 65)
print("PENGUJIAN JUMLAH CLUSTER")
print("=" * 65)

for k in CLUSTER_RANGE:

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=X_fcm,
        c=k,
        m=FUZZINESS,
        error=0.005,
        maxiter=1000,
        init=None,
        seed=RANDOM_STATE
    )

    labels = np.argmax(u, axis=0)

    silhouette = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    evaluation_results.append({
        "Jumlah Cluster": k,
        "FPC": fpc,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": dbi,
        "Calinski-Harabasz Score": ch_score,
        "Objective Function": jm[-1]
    })

    print(f"\nk = {k}")
    print(f"FPC                    : {fpc:.4f}")
    print(f"Silhouette Score       : {silhouette:.4f}")
    print(f"Davies-Bouldin Index   : {dbi:.4f}")
    print(f"Calinski-Harabasz      : {ch_score:.4f}")
    print(f"Objective Function     : {jm[-1]:.4f}")


# ------------------------------------------------------------
# 5. MENYIMPAN HASIL EVALUASI
# ------------------------------------------------------------

evaluation_df = pd.DataFrame(evaluation_results)

evaluation_path = os.path.join(
    OUTPUT_DIR, "fcm_evaluation.csv"
)

evaluation_df.to_csv(evaluation_path, index=False)

print("\nTabel evaluasi:")
print(evaluation_df.to_string(index=False))


# ------------------------------------------------------------
# 6. MENENTUKAN MODEL FINAL
# ------------------------------------------------------------

# Model dipilih berdasarkan Silhouette Score tertinggi
best_index = evaluation_df["Silhouette Score"].idxmax()
best_k = int(evaluation_df.loc[best_index, "Jumlah Cluster"])

print("\nJumlah cluster terpilih:", best_k)

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data=X_fcm,
    c=best_k,
    m=FUZZINESS,
    error=0.005,
    maxiter=1000,
    init=None,
    seed=RANDOM_STATE
)

labels = np.argmax(u, axis=0)


# ------------------------------------------------------------
# 7. MENYIMPAN MEMBERSHIP DEGREE
# ------------------------------------------------------------

membership_df = pd.DataFrame(
    u.T,
    columns=[f"Membership_Cluster_{i}" for i in range(best_k)]
)

membership_df.insert(
    0, "CustomerID", df["CustomerID"].values
)

membership_df["Cluster_Label"] = labels

membership_path = os.path.join(
    OUTPUT_DIR, "fcm_membership_degree.csv"
)

membership_df.to_csv(membership_path, index=False)

print("\nContoh membership degree:")
print(membership_df.head(10).to_string(index=False))


# ------------------------------------------------------------
# 8. PROFIL CLUSTER
# ------------------------------------------------------------

profile_data = data.copy()
profile_data["Cluster"] = labels

cluster_profile = profile_data.groupby("Cluster").agg(
    Jumlah_Pelanggan=("Annual Income (k$)", "size"),
    Rata_Rata_Income=("Annual Income (k$)", "mean"),
    Rata_Rata_Spending=("Spending Score (1-100)", "mean")
).round(2)

profile_path = os.path.join(
    OUTPUT_DIR, "fcm_cluster_profile.csv"
)

cluster_profile.to_csv(profile_path)

print("\nProfil cluster:")
print(cluster_profile)


# ------------------------------------------------------------
# 9. VISUALISASI EVALUASI
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

sns.lineplot(
    data=evaluation_df,
    x="Jumlah Cluster",
    y="Silhouette Score",
    marker="o",
    ax=axes[0]
)
axes[0].set_title("Silhouette Score")

sns.lineplot(
    data=evaluation_df,
    x="Jumlah Cluster",
    y="Davies-Bouldin Index",
    marker="o",
    ax=axes[1]
)
axes[1].set_title("Davies-Bouldin Index")

sns.lineplot(
    data=evaluation_df,
    x="Jumlah Cluster",
    y="FPC",
    marker="o",
    ax=axes[2]
)
axes[2].set_title("Fuzzy Partition Coefficient")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "fcm_evaluation_plot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ------------------------------------------------------------
# 10. VISUALISASI CLUSTER FINAL
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis",
    s=60,
    alpha=0.75
)

plt.scatter(
    cntr[:, 0],
    cntr[:, 1],
    marker="X",
    s=250,
    c="red",
    edgecolors="black",
    label="Cluster Center"
)

plt.title(f"Fuzzy C-Means Mall Customers (k={best_k})")
plt.xlabel("Annual Income (Standardized)")
plt.ylabel("Spending Score (Standardized)")
plt.legend()
plt.colorbar(scatter, label="Cluster")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "fcm_cluster_visualization.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ------------------------------------------------------------
# 11. MENYIMPAN LABEL CLUSTER
# ------------------------------------------------------------

result_df = df.copy()
result_df["FCM_Cluster"] = labels

result_df.to_csv(
    os.path.join(OUTPUT_DIR, "fcm_cluster_labels.csv"),
    index=False
)


# ------------------------------------------------------------
# 12. RINGKASAN AKHIR
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("RINGKASAN HASIL FUZZY C-MEANS")
print("=" * 65)

print("Jumlah cluster final:", best_k)
print("Jumlah pelanggan:", len(df))
print("FPC:", round(fpc, 4))
print(
    "Silhouette Score:",
    round(silhouette_score(X_scaled, labels), 4)
)
print(
    "Davies-Bouldin Index:",
    round(davies_bouldin_score(X_scaled, labels), 4)
)

print("\nDistribusi pelanggan:")
print(pd.Series(labels).value_counts().sort_index())

print("\nSemua hasil tersimpan di:")
print(OUTPUT_DIR)

print("\nPROGRAM FUZZY C-MEANS SELESAI!")
