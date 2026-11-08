
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. PENGATURAN PATH DAN KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_PATH = BASE_DIR / "data" / "raw" / "kc_house_data_NaN.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "price"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ============================================================
# 2. MEMBACA DATASET
# ============================================================

print("=" * 65)
print("TAHAP 1 - MEMBACA DATASET")
print("=" * 65)

if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {RAW_PATH}\n"
        "Pastikan file berada di folder data/raw/."
    )

df = pd.read_csv(RAW_PATH)

print(f"Lokasi dataset : {RAW_PATH}")
print(f"Ukuran dataset : {df.shape[0]:,} baris x {df.shape[1]} kolom")

print("\n5 data pertama:")
print(df.head())

print("\nInformasi tipe data:")
df.info()

if TARGET not in df.columns:
    raise ValueError(
        f"Kolom target '{TARGET}' tidak ditemukan dalam dataset."
    )

# ============================================================
# 3. DATA UNDERSTANDING DAN EDA
# ============================================================

print("\n" + "=" * 65)
print("TAHAP 2 - DATA UNDERSTANDING DAN EDA")
print("=" * 65)

print("\nStatistik deskriptif:")
print(df.describe(include="all").T)

# Memeriksa missing values
print("\nJumlah missing values setiap kolom:")

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))

# Memeriksa dan menangani data duplikat
jumlah_duplikat = int(df.duplicated().sum())

print(f"\nJumlah baris duplikat: {jumlah_duplikat:,}")

if jumlah_duplikat > 0:
    df = df.drop_duplicates().copy()
    print(
        f"Jumlah baris setelah penghapusan duplikat: "
        f"{len(df):,}"
    )
else:
    print("Tidak ditemukan data duplikat.")

# Menghapus baris dengan target kosong
if df[TARGET].isna().any():
    jumlah_target_kosong = int(df[TARGET].isna().sum())

    print(
        f"\nMenghapus {jumlah_target_kosong} baris "
        "karena target price kosong."
    )

    df = df.dropna(subset=[TARGET]).copy()

# Memastikan target berupa angka
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

if df[TARGET].isna().any():
    jumlah_target_invalid = int(df[TARGET].isna().sum())

    print(
        f"Menghapus {jumlah_target_invalid} baris "
        "karena target price bukan angka."
    )

    df = df.dropna(subset=[TARGET]).copy()

# Menghapus harga yang tidak valid
if (df[TARGET] <= 0).any():
    jumlah_harga_tidak_valid = int(
        (df[TARGET] <= 0).sum()
    )

    print(
        f"Menghapus {jumlah_harga_tidak_valid} baris "
        "karena price <= 0."
    )

    df = df.loc[df[TARGET] > 0].copy()

if df.empty:
    raise ValueError("Dataset kosong setelah proses pembersihan.")

# Visualisasi distribusi harga
plt.figure(figsize=(10, 5))

sns.histplot(
    df[TARGET],
    bins=50,
    kde=True
)

plt.title("Distribusi Harga Rumah")
plt.xlabel("Harga Rumah (USD)")
plt.ylabel("Frekuensi")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "eda_distribusi_harga.png",
    dpi=300
)

plt.close()

# Boxplot harga
plt.figure(figsize=(10, 4))

sns.boxplot(x=df[TARGET])

plt.title("Boxplot Harga Rumah")
plt.xlabel("Harga Rumah (USD)")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "eda_boxplot_harga.png",
    dpi=300
)

plt.close()

# Korelasi fitur numerik
numeric_for_corr = df.select_dtypes(include=np.number)

plt.figure(figsize=(15, 12))

sns.heatmap(
    numeric_for_corr.corr(),
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title("Matriks Korelasi Fitur Numerik")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "eda_korelasi.png",
    dpi=300
)

plt.close()

# ============================================================
# 4. PEMERIKSAAN OUTLIER DENGAN METODE IQR
# ============================================================

print("\n" + "=" * 65)
print("TAHAP 3 - PEMERIKSAAN OUTLIER")
print("=" * 65)

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

outlier_rows = []

for column in numeric_columns:

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    )

    outlier_rows.append({
        "kolom": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "batas_bawah": lower_bound,
        "batas_atas": upper_bound,
        "jumlah_outlier": int(mask.sum()),
        "persentase_outlier": float(mask.mean() * 100)
    })

outlier_summary = pd.DataFrame(outlier_rows)

outlier_summary.to_csv(
    RESULTS_DIR / "outlier_summary.csv",
    index=False
)

print(
    outlier_summary[
        ["kolom", "jumlah_outlier", "persentase_outlier"]
    ].to_string(index=False)
)

print(
    "\nCatatan: Outlier hanya diidentifikasi dan dicatat. "
    "Data ekstrem tidak dihapus otomatis karena dapat "
    "merupakan karakteristik rumah yang valid."
)

# ============================================================
# 5. MENENTUKAN FITUR (X) DAN TARGET (y)
# ============================================================

print("\n" + "=" * 65)
print("TAHAP 4 - MENENTUKAN FITUR DAN TARGET")
print("=" * 65)

# Kolom yang tidak digunakan sebagai fitur prediksi
columns_to_drop = [
    column
    for column in ["Unnamed: 0", "id", "date"]
    if column in df.columns
]

df_model = df.drop(
    columns=columns_to_drop
).copy()

X = df_model.drop(
    columns=[TARGET]
)

y = df_model[TARGET].copy()

# Memastikan seluruh fitur bersifat numerik
non_numeric_columns = X.select_dtypes(
    exclude=np.number
).columns.tolist()

if non_numeric_columns:
    raise ValueError(
        "Terdapat fitur non-numerik yang perlu ditangani: "
        f"{non_numeric_columns}"
    )

if X.shape[1] == 0:
    raise ValueError("Tidak terdapat fitur untuk pemodelan.")

print(f"Kolom yang dihapus : {columns_to_drop}")
print(f"Jumlah fitur       : {X.shape[1]}")
print(f"Nama fitur         : {X.columns.tolist()}")
print(f"Nama target        : {TARGET}")

# ============================================================
# 6. MEMBAGI DATA TRAINING DAN TESTING (80:20)
# ============================================================

print("\n" + "=" * 65)
print("TAHAP 5 - TRAIN TEST SPLIT")
print("=" * 65)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Data awal     : {len(X):,} baris")
print(
    f"Data training : {len(X_train_raw):,} baris "
    f"({(1 - TEST_SIZE) * 100:.0f}%)"
)
print(
    f"Data testing  : {len(X_test_raw):,} baris "
    f"({TEST_SIZE * 100:.0f}%)"
)

# ============================================================
# 7. MENANGANI MISSING VALUES
# ============================================================

print("\n" + "=" * 65)
print("TAHAP 6 - MENANGANI MISSING VALUES")
print("=" * 65)

missing_before = X_train_raw.isnull().sum()

print("Missing values pada data training sebelum imputasi:")
print(missing_before[missing_before > 0])

# Memastikan tidak ada fitur yang seluruh nilainya kosong
all_missing_columns = X_train_raw.columns[
    X_train_raw.isnull().all()
].tolist()

if all_missing_columns:
    raise ValueError(
        "Terdapat fitur yang seluruh nilainya kosong pada "
        f"data training: {all_missing_columns}"
    )

# Imputasi menggunakan median
# Imputer hanya di-fit pada data training
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train_raw),
    columns=X_train_raw.columns,
    index=X_train_raw.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test_raw),
    columns=X_test_raw.columns,
    index=X_test_raw.index
)

print("\nMissing values setelah imputasi:")
print(f"Training: {int(X_train.isnull().sum().sum())}")
print(f"Testing : {int(X_test.isnull().sum().sum())}")

# ============================================================
# 8. VALIDASI HASIL PREPROCESSING
# ============================================================

print("\n" + "=" * 65)
print("TAHAP 7 - VALIDASI HASIL PREPROCESSING")
print("=" * 65)

assert X_train.isnull().sum().sum() == 0
assert X_test.isnull().sum().sum() == 0

assert y_train.isnull().sum() == 0
assert y_test.isnull().sum() == 0

assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

assert X_train.columns.tolist() == X_test.columns.tolist()

assert np.isfinite(X_train.to_numpy()).all()
assert np.isfinite(X_test.to_numpy()).all()

assert np.isfinite(y_train.to_numpy()).all()
assert np.isfinite(y_test.to_numpy()).all()

print("Validasi missing values : LULUS")
print("Validasi jumlah data    : LULUS")
print("Validasi nama fitur     : LULUS")
print("Validasi nilai numerik  : LULUS")

# ============================================================
# 9. MENYIMPAN DATA HASIL PREPROCESSING
# ============================================================

print("\n" + "=" * 65)
print("TAHAP 8 - MENYIMPAN DATA PREPROCESSING")
print("=" * 65)

# Menyimpan fitur training dan testing
X_train.to_csv(
    PROCESSED_DIR / "X_train.csv",
    index=False
)

X_test.to_csv(
    PROCESSED_DIR / "X_test.csv",
    index=False
)

# Menyimpan target training dan testing
y_train.to_frame(name=TARGET).to_csv(
    PROCESSED_DIR / "y_train.csv",
    index=False
)

y_test.to_frame(name=TARGET).to_csv(
    PROCESSED_DIR / "y_test.csv",
    index=False
)

# Menyimpan nilai median hasil imputasi
imputer_metadata = {
    column: float(median)
    for column, median in zip(
        X_train_raw.columns,
        imputer.statistics_
    )
}

# Menyimpan metadata preprocessing
metadata = {
    "dataset": RAW_PATH.name,
    "target": TARGET,
    "feature_columns": X.columns.tolist(),
    "dropped_columns": columns_to_drop,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "imputation_strategy": "median",
    "imputation_values": imputer_metadata,
    "outlier_method": "IQR (identified, not automatically removed)",
    "duplicate_rows_removed": jumlah_duplikat,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "number_of_features": int(X_train.shape[1])
}

with open(
    PROCESSED_DIR / "preprocessing_metadata.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
        ensure_ascii=False
    )

# ============================================================
# 10. RINGKASAN AKHIR
# ============================================================

print("\n" + "=" * 65)
print("PREPROCESSING SELESAI")
print("=" * 65)

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : {y_train.shape}")
print(f"y_test  : {y_test.shape}")

print("\nJumlah fitur:", X_train.shape[1])

print("\nFile preprocessing berhasil disimpan di:")
print(PROCESSED_DIR)

print("\nVisualisasi dan laporan outlier disimpan di:")
print(RESULTS_DIR)

print("\nDaftar file preprocessing:")
for file_path in sorted(PROCESSED_DIR.iterdir()):
    if file_path.is_file():
        print(f"- {file_path.name}")

print("\nSEMUA TAHAP PREPROCESSING BERHASIL DIJALANKAN!")
