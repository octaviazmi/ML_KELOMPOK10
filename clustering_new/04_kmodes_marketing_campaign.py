"""
04_kmodes_marketing_campaign.py
K-Modes Clustering - Dataset Marketing Campaign (Anggota A)

Jalankan dari folder mana pun:
    python 04_kmodes_marketing_campaign.py
Output tersimpan di: output/marketing_campaign/kmodes/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from kmodes.kmodes import KModes

K_RANGE = range(2, 11)
FINAL_K = 4
INIT_METHOD = "Huang"    # "Huang" atau "Cao"
N_INIT = 5
RANDOM_STATE = 42
SHOW_PLOTS = False

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "marketing_campaign.csv"
OUT_DIR = BASE_DIR / "output" / "marketing_campaign" / "kmodes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def judul(teks):
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


judul("1. STRUKTUR DATA")
df = pd.read_csv(DATA_PATH, sep="\t")
print(f"Jumlah baris, kolom : {df.shape}")
print("\n5 baris pertama (sebagian kolom):")
print(df[["ID", "Year_Birth", "Education", "Marital_Status", "Income"]].head())

judul("2. MISSING VALUES (seluruh kolom)")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Tidak ada missing values")
print(f"\nTotal missing values: {int(missing.sum())}")
print(f"Baris duplikat (tanpa ID): {int(df.drop(columns='ID').duplicated().sum())}")

print("\nKategori Marital_Status (asli):")
print(df["Marital_Status"].value_counts())
print("\nKategori Education (asli):")
print(df["Education"].value_counts())

# ID tidak dipakai sebagai fitur. Complain, Z_CostContact, Z_Revenue
# tidak dipakai karena hampir/seluruhnya konstan.
judul("3. PEMILIHAN FITUR & PENANGANAN MISSING VALUES")

data_cat = pd.DataFrame(index=df.index)

data_cat["Education"] = df["Education"].astype(str)
data_cat["Marital_Status"] = (
    df["Marital_Status"]
    .replace({"Alone": "Single", "Absurd": "Single", "YOLO": "Single",
              "Together": "Partner", "Married": "Partner"})
    .astype(str)
)

data_cat["Kidhome"] = df["Kidhome"].astype(str)
data_cat["Teenhome"] = df["Teenhome"].astype(str)

REF_YEAR = 2014
usia = REF_YEAR - df["Year_Birth"]
data_cat["Age_Group"] = pd.cut(
    usia,
    bins=[0, 35, 50, 65, 200],
    labels=["<=35", "36-50", "51-65", ">65"],
).astype(str)

income_level = pd.qcut(df["Income"], q=3, labels=["Rendah", "Sedang", "Tinggi"])
data_cat["Income_Level"] = income_level.astype(object)

campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                 "AcceptedCmp4", "AcceptedCmp5"]
n_acc = df[campaign_cols].sum(axis=1)
data_cat["Campaign_Accepted"] = n_acc.clip(upper=2).map(
    {0: "0", 1: "1", 2: "2+"}
)
data_cat["Response"] = df["Response"].astype(str)

FEATURES = list(data_cat.columns)

print("Missing values pada fitur terpilih (sebelum penanganan):")
print(data_cat.isnull().sum())

# Missing value (hanya Income_Level) diisi modus.
for col in FEATURES:
    if data_cat[col].isnull().any():
        modus = data_cat[col].mode()[0]
        n = int(data_cat[col].isnull().sum())
        data_cat[col] = data_cat[col].fillna(modus)
        print(f"-> {col}: {n} missing diisi modus '{modus}'")
data_cat = data_cat.astype(str)

print("\nMissing values pada fitur terpilih (setelah penanganan):")
print(data_cat.isnull().sum().sum())
print(f"\nFitur untuk K-Modes: {FEATURES}")
for col in FEATURES:
    print(f"\n{col}:")
    print(data_cat[col].value_counts())

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

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cost_df["k"], cost_df["cost"], marker="o", color="#2a6f97")
ax.axvline(FINAL_K, color="#c1121f", linestyle="--", label=f"k final = {FINAL_K}")
ax.set_xlabel("Jumlah cluster (k)")
ax.set_ylabel("Cost K-Modes")
ax.set_title("K-Modes - Cost terhadap Jumlah Cluster (Marketing Campaign)")
ax.set_xticks(list(K_RANGE))
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "kmodes_cost_plot.png", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

judul(f"5. MODEL FINAL (k = {FINAL_K})")
km_final = KModes(
    n_clusters=FINAL_K,
    init=INIT_METHOD,
    n_init=N_INIT,
    random_state=RANDOM_STATE,
)
labels = km_final.fit_predict(X)
print(f"Cost model final: {km_final.cost_:.0f}")

result_cat = data_cat.copy()
result_cat["Cluster"] = labels

labels_out = pd.concat(
    [df[["ID"]], data_cat[FEATURES], result_cat[["Cluster"]]], axis=1
)
labels_out.to_csv(OUT_DIR / "kmodes_cluster_labels.csv", index=False)

sizes = (
    result_cat["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Jumlah_Anggota")
)
sizes["Persentase"] = (sizes["Jumlah_Anggota"] / len(result_cat) * 100).round(2)
sizes.to_csv(OUT_DIR / "kmodes_cluster_sizes.csv", index=False)
print("\nJumlah anggota per cluster:")
print(sizes.to_string(index=False))

centroids = pd.DataFrame(km_final.cluster_centroids_, columns=FEATURES)
centroids.insert(0, "Cluster", range(FINAL_K))
centroids.to_csv(OUT_DIR / "kmodes_centroids.csv", index=False)
print("\nCentroid (modus) tiap cluster:")
print(centroids.to_string(index=False))

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

# Tampilkan kategori dominan tiap fitur per cluster
dominan = (
    profile.sort_values("Persen_dalam_Cluster", ascending=False)
    .groupby(["Cluster", "Fitur"], as_index=False)
    .first()
    .pivot(index="Cluster", columns="Fitur", values="Kategori")[FEATURES]
)
print("Kategori dominan per cluster:")
print(dominan.to_string())

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=sizes, x="Cluster", y="Jumlah_Anggota", hue="Cluster",
            palette="viridis", legend=False, ax=ax)
for i, v in enumerate(sizes["Jumlah_Anggota"]):
    ax.text(i, v + 5, str(v), ha="center")
ax.set_title("Distribusi Anggota Cluster K-Modes (Marketing Campaign)")
ax.set_ylabel("Jumlah pelanggan")
fig.tight_layout()
fig.savefig(OUT_DIR / "kmodes_cluster_distribution.png", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

ncol = 3
nrow = -(-len(FEATURES) // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(18, 5 * nrow))
axes = axes.ravel()
for ax, col in zip(axes, FEATURES):
    ct = pd.crosstab(result_cat["Cluster"], result_cat[col], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Persen (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
for ax in axes[len(FEATURES):]:
    ax.axis("off")
fig.suptitle("Profil Kategori per Cluster - K-Modes (Marketing Campaign)")
fig.tight_layout()
fig.savefig(OUT_DIR / "kmodes_cluster_profile.png", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

judul("SELESAI")
print(f"Semua output tersimpan di: {OUT_DIR}")
for f in sorted(OUT_DIR.iterdir()):
    print(" -", f.name)