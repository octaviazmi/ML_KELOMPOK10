"""
kmodes_model.py  --  Persiapan data kategorikal + K-Modes
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmodes.kmodes import KModes

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

KMODES_FEATURES = [
    "Education", "Marital_Status",
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
    "Complain", "Response",
    "Kidhome", "Teenhome",
]

K_RANGE = range(2, 11)          # kandidat k = 2 sampai 10
N_CLUSTERS_FINAL = 4            # jumlah cluster final

INIT_METHOD = "Huang"           # "Huang" atau "Cao"
N_INIT = 5
MAX_ITER = 100
RANDOM_STATE = 42

# Penggabungan nilai Marital_Status yang jarang/tidak wajar (hanya pada salinan data)
MARITAL_STATUS_MAP = {
    "Alone": "Single",
    "Absurd": "Other",
    "YOLO": "Other",
}


# ----------------------------------------------------------------------------
# 1. PERSIAPAN DATA
# ----------------------------------------------------------------------------
def prepare_kmodes_data(df, features=None):
    """Menyiapkan DataFrame kategorikal untuk K-Modes (df asli tidak diubah)."""
    features = list(features) if features else list(KMODES_FEATURES)

    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Kolom tidak ditemukan di dataset: {missing_cols}")
    if "ID" in features:
        raise ValueError("ID tidak boleh dipakai sebagai fitur clustering.")

    data = df[features].copy()

    if "Marital_Status" in data.columns:
        data["Marital_Status"] = data["Marital_Status"].replace(MARITAL_STATUS_MAP)

    # Missing values diisi modus
    for col in data.columns:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].mode().iloc[0])

    # Semua fitur diperlakukan sebagai kategori (string)
    return data.astype(str)


# ----------------------------------------------------------------------------
# 2. FUNGSI BANTU
# ----------------------------------------------------------------------------
def _fit_kmodes(X, n_clusters):
    model = KModes(
        n_clusters=n_clusters, init=INIT_METHOD, n_init=N_INIT,
        max_iter=MAX_ITER, random_state=RANDOM_STATE, verbose=0,
    )
    model.fit(X)
    return model


def compute_cost_table(X, k_range=None):
    k_range = list(k_range) if k_range else list(K_RANGE)
    rows = []
    for k in k_range:
        model = _fit_kmodes(X, k)
        rows.append({"k": k, "cost": float(model.cost_), "iterasi": int(model.n_iter_)})
        print(f"  k = {k:2d} | cost = {model.cost_:10.1f}")
    table = pd.DataFrame(rows)
    table["penurunan_cost"] = (-table["cost"].diff()).round(1)
    table["penurunan_persen"] = (100 * -table["cost"].pct_change()).round(2)
    return table


def _plot_cost(cost_table, n_final, output_dir, show):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_table["k"], cost_table["cost"], "o-", color="#023047")
    for _, r in cost_table.iterrows():
        ax.annotate(f"{r['cost']:.0f}", (r["k"], r["cost"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.axvline(n_final, color="red", linestyle="--", alpha=0.6,
               label=f"k final yang dipakai = {n_final}")
    ax.set_xlabel("Jumlah cluster (k)")
    ax.set_ylabel("Cost K-Modes")
    ax.set_title("K-Modes: Jumlah Cluster vs Cost")
    ax.set_xticks(list(cost_table["k"]))
    ax.margins(y=0.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "kmodes_cost_vs_k.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_cluster_sizes(sizes, output_dir, show):
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(sizes.index.astype(str), sizes.values, color="#219ebc")
    for b, v in zip(bars, sizes.values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah pelanggan")
    ax.set_title("Jumlah Anggota per Cluster (K-Modes)")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "kmodes_jumlah_per_cluster.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_cluster_profile(X, labels):
    """Kategori dominan per cluster dan proporsi semua kategori."""
    dom_rows, share_rows = [], []
    for c in sorted(np.unique(labels)):
        sub = X[labels == c]
        dom = {"Cluster": c, "Jumlah": len(sub)}
        for col in X.columns:
            vc = sub[col].value_counts(normalize=True)
            dom[col] = f"{vc.index[0]} ({100 * vc.iloc[0]:.0f}%)"
            for cat, p in vc.items():
                share_rows.append({"Cluster": c, "Fitur": col, "Kategori": cat,
                                   "Persen": round(100 * p, 2)})
        dom_rows.append(dom)
    return pd.DataFrame(dom_rows), pd.DataFrame(share_rows)


def build_binary_rate(X, labels):
    """Persentase nilai 1 per cluster untuk fitur biner."""
    binary_cols = [c for c in X.columns if set(X[c].unique()) <= {"0", "1"}]
    rate = (X[binary_cols] == "1").groupby(labels).mean() * 100
    rate.index.name = "Cluster"
    return rate.round(1)


def build_numeric_profile(df, labels):
    """Median numerik per cluster, hanya untuk interpretasi (tidak dipakai clustering)."""
    tmp = pd.DataFrame(index=df.index)
    tmp["Cluster"] = labels
    tmp["Income"] = df["Income"]
    tmp["Total_Spending"] = df[[c for c in df.columns if c.startswith("Mnt")]].sum(axis=1)
    tmp["Recency"] = df["Recency"]
    tmp["Year_Birth"] = df["Year_Birth"]
    return tmp.groupby("Cluster").median().round(1)


# ----------------------------------------------------------------------------
# 3. FUNGSI UTAMA
# ----------------------------------------------------------------------------
def run_kmodes(df, output_dir=None, n_clusters=None, k_range=None, show=False):
    """
    Menjalankan pipeline K-Modes. df asli tidak diubah.

    Return (dict): labels, model, kmodes_data, cost_table, cluster_sizes,
    profile_dominant, profile_share, profile_binary_rate, profile_numeric, result_df.
    """
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if not show:
        matplotlib.use("Agg")
    n_clusters = n_clusters or N_CLUSTERS_FINAL

    print("=" * 70)
    print("K-MODES")
    print("=" * 70)
    X = prepare_kmodes_data(df)
    print(f"Fitur dipakai ({X.shape[1]}): {list(X.columns)}")
    print(f"Jumlah data: {len(X)} | Missing values: {int(X.isna().sum().sum())}")
    print("\nJumlah kategori per fitur:")
    print(X.nunique().to_string())

    print("\nMenghitung cost untuk setiap k ...")
    cost_table = compute_cost_table(X, k_range)
    print("\nTabel cost:")
    print(cost_table.to_string(index=False))
    cost_table.to_csv(output_dir / "kmodes_cost_table.csv", index=False)
    _plot_cost(cost_table, n_clusters, output_dir, show)

    print(f"\nMelatih model final dengan k = {n_clusters} ...")
    model = _fit_kmodes(X, n_clusters)
    labels = pd.Series(model.labels_, index=df.index, name="Cluster")
    print(f"Cost model final: {model.cost_:.1f} | iterasi: {model.n_iter_}")

    sizes = labels.value_counts().sort_index()
    sizes.index.name = "Cluster"
    print("\nJumlah anggota per cluster:")
    print(sizes.to_string())
    _plot_cluster_sizes(sizes, output_dir, show)

    dominant, share = build_cluster_profile(X, labels.values)
    binary_rate = build_binary_rate(X, labels)
    numeric_profile = build_numeric_profile(df, labels)
    print("\nKategori dominan tiap cluster (persen dalam cluster):")
    print(dominant.to_string(index=False))
    print("\nPersentase nilai 1 per cluster (fitur biner, dalam %):")
    print(binary_rate.to_string())
    print("\nRingkasan numerik per cluster (median, hanya untuk interpretasi):")
    print(numeric_profile.to_string())

    sizes.rename("Jumlah").to_csv(output_dir / "kmodes_jumlah_per_cluster.csv")
    dominant.to_csv(output_dir / "kmodes_profil_dominan.csv", index=False)
    share.to_csv(output_dir / "kmodes_profil_proporsi.csv", index=False)
    binary_rate.to_csv(output_dir / "kmodes_profil_biner_persen.csv")
    numeric_profile.to_csv(output_dir / "kmodes_profil_numerik.csv")
    result_df = pd.DataFrame({"ID": df["ID"], "Cluster": labels})
    result_df.to_csv(output_dir / "kmodes_label_cluster.csv", index=False)

    print(f"\nK-Modes selesai. File tersimpan di: {output_dir}")
    return {
        "labels": labels, "model": model, "kmodes_data": X,
        "cost_table": cost_table, "cluster_sizes": sizes,
        "profile_dominant": dominant, "profile_share": share,
        "profile_binary_rate": binary_rate, "profile_numeric": numeric_profile,
        "result_df": result_df,
    }


if __name__ == "__main__":
    from eda import load_dataset

    run_kmodes(load_dataset())
