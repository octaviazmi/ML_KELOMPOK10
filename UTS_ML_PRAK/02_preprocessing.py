"""TAHAP 2 - PREPROCESSING: deteksi + handling missing value, duplikasi, outlier."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

df = pd.read_csv(DATA_PATH)
print("=" * 60)
print("TAHAP 2 - PREPROCESSING")
print("=" * 60)

# ---------------------------------------------------------------
# 2.1 DETEKSI & HANDLING KOLOM TIDAK RELEVAN
# ---------------------------------------------------------------
print(f"\n[2.1] Drop kolom tidak relevan: {DROP_COLS}")
df = df.drop(columns=DROP_COLS)
print(f"      Bentuk data setelah drop kolom: {df.shape}")

kat = df.select_dtypes(exclude="number").columns.tolist()
num = [c for c in df.select_dtypes(include="number").columns if c != TARGET]

# ---------------------------------------------------------------
# 2.2 MISSING VALUE
# ---------------------------------------------------------------
print("\n[2.2] DETEKSI MISSING VALUE")
mv = df.isna().sum()
mv_tbl = pd.DataFrame({"jumlah_missing": mv, "persen (%)": (mv / len(df) * 100).round(2)})
print(mv_tbl[mv_tbl.jumlah_missing > 0].to_string())
print(f"Total sel kosong: {int(mv.sum())}")

plt.figure(figsize=(8, 4))
mm = mv_tbl[mv_tbl.jumlah_missing > 0].sort_values("jumlah_missing")
plt.barh(mm.index, mm.jumlah_missing, color="#4C72B0")
for i, v in enumerate(mm.jumlah_missing):
    plt.text(v + 5, i, str(v), va="center")
plt.title("Jumlah Missing Value per Kolom (sebelum handling)")
plt.xlabel("Jumlah")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "01_missing_value.png"), dpi=130); plt.close()

# HANDLING: numerik -> median, kategorikal -> modus
print("\n      HANDLING: numerik diisi MEDIAN, kategorikal diisi MODUS")
for c in num:
    if df[c].isna().any():
        med = df[c].median()
        df[c] = df[c].fillna(med)
        print(f"      - {c:<16}: diisi median = {med:.2f}")
for c in kat:
    if df[c].isna().any():
        mod = df[c].mode()[0]
        df[c] = df[c].fillna(mod)
        print(f"      - {c:<16}: diisi modus  = {mod}")
print(f"      Total missing value setelah handling: {int(df.isna().sum().sum())}")

# ---------------------------------------------------------------
# 2.3 DUPLIKASI DATA
# ---------------------------------------------------------------
print("\n[2.3] DETEKSI DUPLIKASI DATA")
n_dup = int(df.duplicated().sum())
print(f"      Jumlah baris duplikat: {n_dup}")
if n_dup > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"      HANDLING: baris duplikat dihapus -> bentuk data {df.shape}")
else:
    print("      HANDLING: tidak diperlukan (tidak ada duplikat)")

# ---------------------------------------------------------------
# 2.4 OUTLIER (metode IQR)
# ---------------------------------------------------------------
print("\n[2.4] DETEKSI OUTLIER (metode IQR: Q1-1.5*IQR s/d Q3+1.5*IQR)")
rows, bounds = [], {}
for c in num:
    q1, q3 = df[c].quantile([0.25, 0.75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
    bounds[c] = (lo, hi)
    rows.append([c, round(lo, 2), round(hi, 2), n_out, round(n_out / len(df) * 100, 2)])
out_tbl = pd.DataFrame(rows, columns=["kolom", "batas_bawah", "batas_atas", "jumlah_outlier", "persen (%)"])
print(out_tbl.to_string(index=False))

# Kolom diskrit (hitungan 0-5) tidak di-capping: rentang nilainya valid & mayoritas bernilai 0,
# sehingga IQR menjadi 0 dan nilai sah (mis. late_payments = 1..3) salah terdeteksi sebagai outlier.
DISKRIT = ["support_tickets", "late_payments"]
KONTINU = [c for c in num if c not in DISKRIT]
print(f"\n      Kolom diskrit (tidak di-capping): {DISKRIT}")

# boxplot sebelum
fig, ax = plt.subplots(1, len(KONTINU), figsize=(3 * len(KONTINU), 4))
for a, c in zip(ax, KONTINU):
    a.boxplot(df[c], patch_artist=True, boxprops=dict(facecolor="#DD8452")); a.set_title(c, fontsize=9)
fig.suptitle("Boxplot SEBELUM handling outlier"); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_boxplot_sebelum.png"), dpi=130); plt.close()

# HANDLING: capping (winsorizing) ke batas IQR
print("      HANDLING: capping nilai outlier ke batas atas/bawah IQR (data tidak dibuang)")
for c in KONTINU:
    lo, hi = bounds[c]
    df[c] = df[c].clip(lower=lo, upper=hi)

fig, ax = plt.subplots(1, len(KONTINU), figsize=(3 * len(KONTINU), 4))
for a, c in zip(ax, KONTINU):
    a.boxplot(df[c], patch_artist=True, boxprops=dict(facecolor="#55A868")); a.set_title(c, fontsize=9)
fig.suptitle("Boxplot SESUDAH handling outlier"); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_boxplot_sesudah.png"), dpi=130); plt.close()

# Verifikasi
sisa = 0
for c in KONTINU:
    q1, q3 = df[c].quantile([0.25, 0.75]); iqr = q3 - q1
    sisa += int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
print(f"      Sisa outlier pada kolom kontinu setelah capping: {sisa}")

df.to_csv(p("data_preprocessed.csv"), index=False)
print(f"\nData bersih disimpan: output/data_preprocessed.csv  (bentuk {df.shape})")
