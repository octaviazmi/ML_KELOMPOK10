"""
eda.py  --  Exploratory Data Analysis 
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "marketing_campaign.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

# Dataset ini dipisahkan oleh TAB (bukan koma) -> sep="\t"
CSV_SEPARATOR = "\t"

# Kolom ID bukan fitur; Z_CostContact & Z_Revenue bernilai konstan (tidak informatif).
NON_FEATURE_COLUMNS = ["ID", "Z_CostContact", "Z_Revenue"]

# Kolom bertipe angka tetapi sebenarnya KATEGORI (0/1 atau jumlah kecil)
CATEGORICAL_NUMERIC_COLUMNS = [
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
    "Complain", "Response", "Kidhome", "Teenhome",
]
CATEGORICAL_TEXT_COLUMNS = ["Education", "Marital_Status"]


# ----------------------------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------------------------
def load_dataset(path=None):
    """Membaca dataset asli. Panggil SEKALI saja, lalu oper DataFrame-nya ke fungsi lain."""
    path = Path(path) if path else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path, sep=CSV_SEPARATOR)
    return df


# ----------------------------------------------------------------------------
# Fungsi bantu
# ----------------------------------------------------------------------------
def _get_numeric_columns(df):
    """Fitur numerik kontinu/hitungan (tanpa ID, konstanta, dan kolom kategori berkode angka)."""
    exclude = set(NON_FEATURE_COLUMNS) | set(CATEGORICAL_NUMERIC_COLUMNS)
    return [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]


def _iqr_outlier_table(df, columns):
    """Hitung outlier dengan metode IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR)."""
    rows = []
    for col in columns:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < low) | (s > high)).sum())
        rows.append({
            "Fitur": col, "Q1": q1, "Q3": q3, "IQR": iqr,
            "Batas_Bawah": low, "Batas_Atas": high,
            "Min": s.min(), "Max": s.max(),
            "Jumlah_Outlier": n_out,
            "Persen_Outlier": round(100 * n_out / len(s), 2),
        })
    return pd.DataFrame(rows)


def _save_or_show(fig, filename, output_dir, show):
    """Simpan figure ke output_dir (jika ada), tampilkan jika show=True, lalu tutup."""
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(Path(output_dir) / filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _print_title(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ----------------------------------------------------------------------------
# 2. EDA UTAMA
# ----------------------------------------------------------------------------
def run_eda(df, output_dir=None, show=False):
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if not show:
        matplotlib.use("Agg")  # backend tanpa jendela, aman dijalankan di mana saja

    sns.set_theme(style="whitegrid")
    results = {}

    # ---- 2.1 Gambaran umum --------------------------------------------------
    _print_title("1. GAMBARAN UMUM DATASET")
    print("5 baris pertama:")
    print(df.head().to_string())
    print(f"\nDimensi dataset: {df.shape[0]} baris x {df.shape[1]} kolom")
    print("\nTipe data & info:")
    df.info()

    dtype_table = pd.DataFrame({
        "Tipe_Data": df.dtypes.astype(str),
        "Non_Null": df.notna().sum(),
        "Jumlah_Unik": df.nunique(),
    })
    dtype_table.index.name = "Kolom"
    dtype_table.to_csv(output_dir / "eda_info_kolom.csv")
    results["dtype_table"] = dtype_table

    # ---- 2.2 Statistik deskriptif ------------------------------------------
    _print_title("2. STATISTIK DESKRIPTIF")
    desc_num = df.describe().T
    print("Fitur numerik:")
    print(desc_num.round(2).to_string())
    desc_num.round(2).to_csv(output_dir / "eda_statistik_deskriptif_numerik.csv")

    desc_cat = df[CATEGORICAL_TEXT_COLUMNS].describe().T
    print("\nFitur kategorikal (teks):")
    print(desc_cat.to_string())
    desc_cat.to_csv(output_dir / "eda_statistik_deskriptif_kategorikal.csv")
    results["describe_numeric"] = desc_num
    results["describe_categorical"] = desc_cat

    # ---- 2.3 Missing values & duplikat -------------------------------------
    _print_title("3. MISSING VALUES DAN DUPLIKAT")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("Tidak ada missing values.")
    else:
        print("Kolom dengan missing values:")
        print(missing.to_string())
    n_dup_full = int(df.duplicated().sum())
    n_dup_no_id = int(df.drop(columns="ID").duplicated().sum())
    n_dup_id = int(df["ID"].duplicated().sum())
    print(f"\nDuplikat baris penuh (semua kolom)   : {n_dup_full}")
    print(f"Duplikat ID                          : {n_dup_id}")
    print(f"Duplikat jika kolom ID diabaikan     : {n_dup_no_id}")
    print("Catatan: baris yang hanya berbeda ID dianggap pelanggan berbeda "
          "(profil sama), sehingga TIDAK dihapus.")
    results["missing"] = missing
    results["n_duplicates_full"] = n_dup_full
    results["n_duplicates_ignoring_id"] = n_dup_no_id

    # ---- 2.4 Salinan data untuk analisis (data asli tidak diubah) ----------
    _print_title("4. PENANGANAN DATA PADA SALINAN (df_eda)")
    df_eda = df.copy()

    # (a) Missing Income -> imputasi MEDIAN
    n_missing_income = int(df_eda["Income"].isna().sum())
    income_median = df_eda["Income"].median()
    df_eda["Income"] = df_eda["Income"].fillna(income_median)
    print(f"Income kosong: {n_missing_income} nilai -> diisi MEDIAN = {income_median:,.0f}")
    print("Alasan: distribusi Income miring ke kanan dan memiliki outlier ekstrem "
          "(maks 666.666), sehingga median lebih tahan outlier dibanding mean.")

    # (b) Fitur turunan sederhana untuk membantu visualisasi
    ref_year = pd.to_datetime(df_eda["Dt_Customer"], dayfirst=True).dt.year.max()
    df_eda["Age"] = ref_year - df_eda["Year_Birth"]
    mnt_cols = [c for c in df_eda.columns if c.startswith("Mnt")]
    df_eda["TotalSpending"] = df_eda[mnt_cols].sum(axis=1)
    print(f"Fitur turunan: Age = {ref_year} - Year_Birth ; TotalSpending = jumlah kolom Mnt*")

    # (c) Kolom kategori berkode angka diperlakukan sebagai kategori
    for col in CATEGORICAL_NUMERIC_COLUMNS:
        df_eda[col] = df_eda[col].astype("category")
    print("Kolom berkode angka yang diperlakukan sebagai kategori: "
          + ", ".join(CATEGORICAL_NUMERIC_COLUMNS))

    results["df_eda"] = df_eda
    results["income_median"] = income_median

    # ---- 2.5 Outlier (metode IQR) ------------------------------------------
    _print_title("5. DETEKSI OUTLIER (METODE IQR)")
    num_cols = _get_numeric_columns(df_eda)  # sudah termasuk Age & TotalSpending
    outlier_table = _iqr_outlier_table(df_eda, num_cols)
    print(outlier_table.round(2).to_string(index=False))
    outlier_table.round(2).to_csv(output_dir / "eda_outlier_iqr.csv", index=False)
    results["outlier_table"] = outlier_table
    print("\nContoh nilai tidak wajar:")
    print(f"  Year_Birth terkecil = {int(df['Year_Birth'].min())} "
          f"(Age {int(df_eda['Age'].max())} tahun)")
    print(f"  Income terbesar     = {df['Income'].max():,.0f}")
    print("Catatan: outlier hanya DILAPORKAN, tidak dihapus dari data.")

    # Boxplot
    n = len(num_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
    for ax, col in zip(axes.ravel(), num_cols):
        sns.boxplot(y=df_eda[col], ax=ax, color="#8ecae6")
        ax.set_title(col)
        ax.set_ylabel("")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Boxplot Fitur Numerik (deteksi outlier)", y=1.0, fontsize=14)
    _save_or_show(fig, "eda_boxplot_numerik.png", output_dir, show)

    # ---- 2.6 Histogram fitur numerik ---------------------------------------
    _print_title("6. VISUALISASI FITUR NUMERIK")
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
    for ax, col in zip(axes.ravel(), num_cols):
        sns.histplot(df_eda[col], kde=True, bins=30, ax=ax, color="#219ebc")
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Histogram Fitur Numerik", y=1.0, fontsize=14)
    _save_or_show(fig, "eda_histogram_numerik.png", output_dir, show)

    # Heatmap korelasi
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df_eda[num_cols].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".1f",
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Korelasi Antar Fitur Numerik")
    _save_or_show(fig, "eda_korelasi_numerik.png", output_dir, show)
    results["correlation"] = corr

    # Hubungan fitur yang relevan: Income vs TotalSpending (Income dibatasi
    # < 200.000 hanya untuk tampilan agar outlier 666.666 tidak merusak skala)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_df = df_eda[df_eda["Income"] < 200000]
    sns.scatterplot(data=plot_df, x="Income", y="TotalSpending",
                    hue="Response", alpha=0.6, ax=axes[0])
    axes[0].set_title("Income vs TotalSpending (Income < 200.000, hanya untuk tampilan)")
    sns.scatterplot(data=df_eda, x="Age", y="TotalSpending",
                    hue="Response", alpha=0.6, ax=axes[1])
    axes[1].set_title("Age vs TotalSpending")
    _save_or_show(fig, "eda_hubungan_fitur.png", output_dir, show)

    # ---- 2.7 Fitur kategorikal ---------------------------------------------
    _print_title("7. VISUALISASI FITUR KATEGORIKAL")
    cat_cols = CATEGORICAL_TEXT_COLUMNS + CATEGORICAL_NUMERIC_COLUMNS
    cat_tables = {}
    for col in cat_cols:
        vc = df[col].value_counts()
        cat_tables[col] = vc
        print(f"\n{col}:")
        print(vc.to_string())
    pd.concat(
        {c: t.rename_axis("Kategori").rename("Jumlah").reset_index() for c, t in cat_tables.items()}
    ).reset_index(level=0).rename(columns={"level_0": "Fitur"}) \
        .to_csv(output_dir / "eda_distribusi_kategorikal.csv", index=False)
    results["categorical_counts"] = cat_tables

    ncols_c = 3
    nrows_c = int(np.ceil(len(cat_cols) / ncols_c))
    fig, axes = plt.subplots(nrows_c, ncols_c, figsize=(16, 3.6 * nrows_c))
    for ax, col in zip(axes.ravel(), cat_cols):
        order = df[col].value_counts().index
        sns.countplot(x=df[col].astype(str), order=order.astype(str), ax=ax, color="#ffb703")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    for ax in axes.ravel()[len(cat_cols):]:
        ax.axis("off")
    fig.suptitle("Distribusi Fitur Kategorikal", y=1.0, fontsize=14)
    _save_or_show(fig, "eda_kategorikal.png", output_dir, show)

    results["output_files"] = sorted(p.name for p in output_dir.glob("eda_*"))
    print(f"\nEDA selesai. File tersimpan di: {output_dir}")
    return results

if __name__ == "__main__":
    data = load_dataset()
    run_eda(data)
