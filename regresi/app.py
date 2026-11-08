"""
TM11 - Regresi | Estimasi Harga Rumah King County (USA)
Kelompok 5 - Aplikasi Streamlit (UI/UX dari model dengan performa terbaik)

Menjalankan (dari folder yang berisi app.py):
    streamlit run app.py

=====================================================================
ALUR INTEGRASI PROYEK (kontrak antar file)
=====================================================================
01_preprocessing.py  ->  data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
                         data/processed/preprocessing_metadata.json
                           (feature_columns, imputation_values, train_rows, test_rows)
02_xgboost.py        ->  models/xgboost_model.pkl
                         results/xgboost_metrics.json      (train_metrics, test_metrics, params, ...)
                         results/xgboost_predictions.csv   (kolom: actual, predicted, residual)
03_lightgbm.py       ->  models/lightgbm_model.pkl
                         results/lightgbm_metrics.json     (struktur SAMA dengan xgboost_metrics.json)
                         results/lightgbm_predictions.csv  (kolom: actual, predicted, residual)
04_compare_models.py ->  results/model_comparison.csv      (tabel perbandingan, kolom bebas)
                         models/best_model.pkl             (salinan model terbaik, joblib)
                         models/best_model_info.json       (lihat skema di bawah)

FORMAT MODEL TERBAIK (yang dibaca aplikasi ini)
  * models/best_model.pkl : hasil joblib.dump() dari XGBRegressor / LGBMRegressor
    yang sudah di-fit memakai DataFrame (nama kolom fitur tersimpan di model).
    Model menerima DataFrame 1 baris berisi fitur yang sama dengan data latih dan
    mengembalikan harga dalam USD (tanpa transformasi log).
  * models/best_model_info.json (disarankan; bila tidak ada, aplikasi menebak nama
    model dari kelas model lalu membaca results/<nama>_metrics.json):
        {
          "model_name": "XGBoost" atau "LightGBM",
          "selection_metric": "RMSE",                 (opsional, dasar pemilihan)
          "train_metrics": {"MAE":..., "MAPE":..., "RMSE":..., "MSE":..., "R2":...},
          "test_metrics":  {"MAE":..., "MAPE":..., "RMSE":..., "MSE":..., "R2":...},
          "params": {...}, "n_train": ..., "n_test": ...,
          "target_transform": "none"                  (opsional; "log1p" bila target di-log)
        }
    Cara termudah di 04_compare_models.py: salin isi results/<model_terbaik>_metrics.json,
    tambahkan "model_name" dan "selection_metric", lalu simpan sebagai best_model_info.json.

Aplikasi TIDAK memiliki angka evaluasi atau prediksi bawaan. Semua angka berasal dari
file hasil eksperimen di atas. Bila file belum ada, aplikasi tetap terbuka dan bagian
terkait menampilkan petunjuk.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Path
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
BEST_INFO_PATH = MODELS_DIR / "best_model_info.json"
PREPROC_META_PATH = PROCESSED_DIR / "preprocessing_metadata.json"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"

PIPELINE_FILES = [
    ("01_preprocessing.py", PROCESSED_DIR / "X_train.csv", "data/processed/X_train.csv"),
    ("01_preprocessing.py", PREPROC_META_PATH, "data/processed/preprocessing_metadata.json"),
    ("02_xgboost.py", MODELS_DIR / "xgboost_model.pkl", "models/xgboost_model.pkl"),
    ("02_xgboost.py", RESULTS_DIR / "xgboost_metrics.json", "results/xgboost_metrics.json"),
    ("03_lightgbm.py", MODELS_DIR / "lightgbm_model.pkl", "models/lightgbm_model.pkl"),
    ("03_lightgbm.py", RESULTS_DIR / "lightgbm_metrics.json", "results/lightgbm_metrics.json"),
    ("04_compare_models.py", COMPARISON_PATH, "results/model_comparison.csv"),
    ("04_compare_models.py", BEST_MODEL_PATH, "models/best_model.pkl"),
    ("04_compare_models.py", BEST_INFO_PATH, "models/best_model_info.json"),
]

# ----------------------------------------------------------------------------
# Konfigurasi fitur (rentang input = batas kewajaran data King County)
# ----------------------------------------------------------------------------
# Hanya dipakai bila berkas metadata belum ada. Nilai sama dengan imputation_values
# pada data/processed/preprocessing_metadata.json (median data latih).
FALLBACK_DEFAULTS = {
    "bedrooms": 3.0, "bathrooms": 2.25, "sqft_living": 1910.0, "sqft_lot": 7600.0,
    "floors": 1.5, "waterfront": 0.0, "view": 0.0, "condition": 3.0, "grade": 7.0,
    "sqft_above": 1560.0, "sqft_basement": 0.0, "yr_built": 1975.0, "yr_renovated": 0.0,
    "zipcode": 98065.0, "lat": 47.5718, "long": -122.231, "sqft_living15": 1840.0,
    "sqft_lot15": 7615.0, "sale_year": 2014.0, "sale_month": 6.0,
}
FALLBACK_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
    "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
    "zipcode", "lat", "long", "sqft_living15", "sqft_lot15",
]

VIEW_LABELS = {0: "0 - Tidak ada", 1: "1 - Cukup", 2: "2 - Sedang", 3: "3 - Baik", 4: "4 - Sangat baik"}
CONDITION_LABELS = {1: "1 - Buruk", 2: "2 - Kurang", 3: "3 - Rata-rata", 4: "4 - Baik", 5: "5 - Sangat baik"}

# kind: int | float | slider | select | toggle
SPECS = {
    "zipcode": dict(label="Kode pos (zipcode)", help="Kode pos wilayah di King County, Washington."),
    "lat": dict(label="Latitude", kind="float", min=47.0, max=48.0, step=0.001, fmt="%.4f",
                help="Koordinat lintang. Terisi otomatis dari kode pos, boleh diubah."),
    "long": dict(label="Longitude", kind="float", min=-123.0, max=-121.0, step=0.001, fmt="%.4f",
                 help="Koordinat bujur. Terisi otomatis dari kode pos, boleh diubah."),
    "bedrooms": dict(label="Kamar tidur", kind="int", min=1, max=10, step=1, help="Jumlah kamar tidur."),
    "bathrooms": dict(label="Kamar mandi", kind="float", min=0.5, max=8.0, step=0.25, fmt="%.2f",
                      help="Jumlah kamar mandi (0,5 = toilet tanpa mandi)."),
    "floors": dict(label="Lantai", kind="float", min=1.0, max=3.5, step=0.5, fmt="%.1f", help="Jumlah lantai."),
    "sqft_living": dict(label="Luas bangunan (sqft)", kind="int", min=290, max=14000, step=10,
                        help="Luas ruang tinggal dalam square feet (1 sqft = 0,0929 m²)."),
    "sqft_basement": dict(label="Luas basement (sqft)", kind="int", min=0, max=5000, step=10,
                          help="Luas basement. Isi 0 bila tidak ada."),
    "sqft_above": dict(label="Luas di atas tanah (sqft)", kind="int", min=290, max=10000, step=10,
                       help="Luas bangunan di atas permukaan tanah."),
    "sqft_lot": dict(label="Luas tanah (sqft)", kind="int", min=500, max=1700000, step=100, help="Luas kavling tanah."),
    "grade": dict(label="Grade konstruksi (1-13)", kind="slider", min=1, max=13,
                  help="Kualitas konstruksi dan desain menurut sistem penilaian King County (13 = tertinggi)."),
    "condition": dict(label="Kondisi bangunan", kind="select", options=CONDITION_LABELS, help="Kondisi fisik bangunan (1-5)."),
    "view": dict(label="Kualitas pemandangan", kind="select", options=VIEW_LABELS, help="Indeks pemandangan (0-4)."),
    "waterfront": dict(label="Tepi air", kind="toggle", help="Waterfront: aktifkan bila rumah berada di tepi air."),
    "yr_built": dict(label="Tahun dibangun", kind="int", min=1900, max=2015, step=1, help="Tahun rumah dibangun."),
    "yr_renovated": dict(label="Tahun renovasi (0 = belum)", kind="int", min=0, max=2015, step=1,
                         help="Tahun renovasi terakhir. Isi 0 bila belum pernah direnovasi."),
    "sqft_living15": dict(label="Luas bangunan tetangga (sqft)", kind="int", min=300, max=7000, step=10,
                          help="Rata-rata luas bangunan 15 rumah tetangga terdekat."),
    "sqft_lot15": dict(label="Luas tanah tetangga (sqft)", kind="int", min=600, max=900000, step=100,
                       help="Rata-rata luas tanah 15 rumah tetangga terdekat."),
    "sale_year": dict(label="Tahun penjualan", kind="int", min=2014, max=2015, step=1, help="Tahun transaksi."),
    "sale_month": dict(label="Bulan penjualan", kind="int", min=1, max=12, step=1, help="Bulan transaksi (1-12)."),
}

GROUPS = [
    ("📍 Lokasi", ["zipcode", "lat", "long"]),
    ("📐 Ukuran & Tata Ruang", ["bedrooms", "bathrooms", "floors", "sqft_living", "sqft_basement", "sqft_lot"]),
    ("⭐ Kualitas & Fitur", ["grade", "condition", "view", "waterfront"]),
    ("🗓️ Tahun", ["yr_built", "yr_renovated", "sale_year", "sale_month"]),
    ("🏘️ Lingkungan Sekitar", ["sqft_living15", "sqft_lot15"]),
]

# Ringkasan karakteristik algoritma (sumber: modul TM11 Regresi, bagian 1.3 dan tabel perbandingan)
ALGO_INFO = {
    "xgboost": {
        "title": "XGBoost (Extreme Gradient Boosting)",
        "summary": "Gradient Boosting teroptimasi yang membangun pohon keputusan secara berurutan; "
                   "setiap pohon baru memperbaiki kesalahan (residu) pohon sebelumnya.",
        "rows": [
            ("Pendekatan utama", "Gradient Boosting teroptimasi dengan regularisasi"),
            ("Pertumbuhan pohon", "Level-wise / depth-wise (melebar per level)"),
            ("Kecepatan & efisiensi", "Sangat cepat dan efisien dalam memori"),
            ("Pencegahan overfitting", "Regularisasi L1 (Lasso) dan L2 (Ridge) bawaan"),
            ("Missing value", "Ditangani otomatis secara internal"),
            ("Skala data ideal", "Data tabular sedang hingga besar"),
        ],
    },
    "lightgbm": {
        "title": "LightGBM (Light Gradient Boosting Machine)",
        "summary": "Gradient Boosting berbasis histogram dari Microsoft yang berfokus pada kecepatan "
                   "dan efisiensi memori.",
        "rows": [
            ("Pendekatan utama", "Gradient Boosting berbasis histogram"),
            ("Pertumbuhan pohon", "Leaf-wise (memilih daun dengan penurunan loss terbesar)"),
            ("Kecepatan & efisiensi", "Sangat cepat dan paling hemat memori pada data besar"),
            ("Pencegahan overfitting", "Pembatasan max_depth / num_leaves"),
            ("Missing value", "Ditangani otomatis secara internal"),
            ("Skala data ideal", "Data sangat besar (large-scale data)"),
        ],
    },
}
PARAM_WHITELIST = [
    "n_estimators", "learning_rate", "max_depth", "num_leaves", "min_child_weight",
    "min_child_samples", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "random_state",
]
METRIC_ORDER = ["MAE", "MAPE", "RMSE", "MSE", "R2"]

# ----------------------------------------------------------------------------
# Halaman & gaya
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Estimasi Harga Rumah King County", page_icon="🏠", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1240px;}
    .hero {background: linear-gradient(120deg, #0b2545 0%, #13487f 55%, #2b78c2 100%);
           color: #fff; padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1rem;
           display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;}
    .hero h1 {margin: 0; font-size: 1.8rem; font-weight: 750; color: #fff; line-height: 1.2;}
    .hero p {margin: .35rem 0 0 0; opacity: .88; font-size: .97rem; color: #fff;}
    .pill {display: inline-block; padding: .35rem .85rem; border-radius: 999px; font-size: .84rem;
           font-weight: 650; border: 1px solid rgba(255,255,255,.45); background: rgba(255,255,255,.14); color: #fff;}
    .pill.ok {background: rgba(46,204,113,.25); border-color: rgba(46,204,113,.7);}
    .pill.wait {background: rgba(255,193,7,.22); border-color: rgba(255,193,7,.7);}
    .section-title {font-size: 1.02rem; font-weight: 700; color: #3b82c4; border-left: 4px solid #2b78c2;
                    padding-left: .6rem; margin: .9rem 0 .5rem 0;}
    .result-card {background: linear-gradient(135deg, #f4f9ff 0%, #e3effb 100%); border: 1px solid #c5dcf3;
                  border-radius: 16px; padding: 1.5rem 1.2rem; text-align: center;}
    .result-label {color: #4a6785; font-size: .85rem; letter-spacing: .08em; text-transform: uppercase;}
    .result-price {color: #0b2545; font-size: 2.9rem; font-weight: 800; margin: .15rem 0; line-height: 1.15;}
    .result-sub {color: #4a6785; font-size: .93rem;}
    .placeholder-card {border: 2px dashed rgba(100,140,190,.6); border-radius: 16px; padding: 2.2rem 1rem;
                       text-align: center; opacity: .85;}
    .info-card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 1rem 1.2rem;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .6rem .9rem;}
    .footer {text-align: center; opacity: .6; font-size: .8rem; margin-top: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.45rem;}
    button[data-testid="stBaseButton-primary"] {background-color: #1d4f8c; border-color: #1d4f8c; color: #fff;}
    button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {background-color: #13487f; border-color: #13487f;}
    button[data-testid="stBaseButton-primary"]:disabled {background-color: #1d4f8c; border-color: #1d4f8c; opacity: .45;}
    [data-testid="stTab"][aria-selected="true"], [data-testid="stTab"][aria-selected="true"] p {color: #3b82c4 !important;}
    [role="tablist"] div:empty {background-color: #3b82c4 !important;}
    div[data-testid="stSlider"] {filter: hue-rotate(220deg);}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Pemuatan berkas (dengan cache yang ikut berubah bila berkas diperbarui)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float):
    return pd.read_csv(path)


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return _read_json(str(path), path.stat().st_mtime)
    except Exception:
        return None


def read_csv(path: Path):
    if not path.exists():
        return None
    try:
        return _read_csv(str(path), path.stat().st_mtime)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_model(path: str, mtime: float):
    return joblib.load(path)


def load_best_model():
    """Mengembalikan (model, pesan_error). Keduanya None bila best_model.pkl belum ada."""
    if not BEST_MODEL_PATH.exists():
        return None, None
    try:
        return _load_model(str(BEST_MODEL_PATH), BEST_MODEL_PATH.stat().st_mtime), None
    except Exception as exc:  # file rusak, atau xgboost/lightgbm belum terpasang
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False)
def _reference_stats(x_path: str, y_path: str, mtime: float):
    """Statistik referensi dari data latih HASIL PREPROCESSING (bukan angka karangan)."""
    X = pd.read_csv(x_path)
    stats = {
        "median": X.median(numeric_only=True).to_dict(),
        "min": X.min(numeric_only=True).to_dict(),
        "max": X.max(numeric_only=True).to_dict(),
        "zip_table": {},
        "prices": None,
    }
    if {"zipcode", "lat", "long"}.issubset(X.columns):
        zt = X.groupby("zipcode")[["lat", "long"]].median().round(5)
        stats["zip_table"] = {str(int(z)): {"lat": float(r["lat"]), "long": float(r["long"])} for z, r in zt.iterrows()}
    try:
        y = pd.read_csv(y_path)
        col = "price" if "price" in y.columns else y.columns[0]
        stats["prices"] = np.sort(y[col].to_numpy(dtype=float))
    except Exception:
        pass
    return stats


def load_reference():
    xp, yp = PROCESSED_DIR / "X_train.csv", PROCESSED_DIR / "y_train.csv"
    if not xp.exists():
        return None
    try:
        return _reference_stats(str(xp), str(yp), xp.stat().st_mtime)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Utilitas model
# ----------------------------------------------------------------------------
def model_feature_names(model) -> list[str]:
    """Nama & urutan fitur diambil dari model itu sendiri agar selalu cocok."""
    for attr in ("feature_names_in_", "feature_name_"):
        try:
            v = getattr(model, attr, None)
            if v is not None and len(v):
                return [str(x) for x in list(v)]
        except Exception:
            pass
    try:
        names = model.get_booster().feature_names
        if names:
            return [str(x) for x in names]
    except Exception:
        pass
    return []


def detect_model_name(model) -> str:
    cls = type(model).__name__
    return {"XGBRegressor": "XGBoost", "LGBMRegressor": "LightGBM"}.get(cls, cls)


def slug(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_info(model) -> dict:
    """Gabungkan best_model_info.json, results/<model>_metrics.json, dan parameter model."""
    raw = read_json(BEST_INFO_PATH) or {}
    name = raw.get("model_name") or raw.get("model") or detect_model_name(model)
    key = slug(name)
    from_file = read_json(RESULTS_DIR / f"{key}_metrics.json") or {}
    try:
        params = {k: v for k, v in model.get_params().items() if v is not None}
    except Exception:
        params = raw.get("params") or from_file.get("params") or {}
    return {
        "name": name,
        "key": key,
        "selection_metric": raw.get("selection_metric"),
        "test": raw.get("test_metrics") or from_file.get("test_metrics") or {},
        "train": raw.get("train_metrics") or from_file.get("train_metrics") or {},
        "params": params,
        "n_train": raw.get("n_train") or from_file.get("n_train"),
        "n_test": raw.get("n_test") or from_file.get("n_test"),
        "target_transform": raw.get("target_transform", "none"),
    }


def predict_price(model, X: pd.DataFrame, target_transform: str = "none") -> float:
    y = float(np.asarray(model.predict(X)).ravel()[0])
    if target_transform == "log1p":
        y = float(np.expm1(y))
    return max(y, 0.0)


def usd(x, d=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"-${abs(x):,.{d}f}" if x < 0 else f"${x:,.{d}f}"


def fmt_metric(name: str, v):
    if v is None:
        return "-"
    try:
        v = float(v)
    except Exception:
        return "-"
    if name == "MAPE":
        return f"{v:.2f}%"
    if name == "R2":
        return f"{v:.4f}"
    if name == "MSE":
        return f"{v:.2e}"
    return usd(v, 0)


# ----------------------------------------------------------------------------
# Muat sumber data aplikasi
# ----------------------------------------------------------------------------
model, model_error = load_best_model()
info = resolve_info(model) if model is not None else None
meta = read_json(PREPROC_META_PATH) or {}
ref = load_reference()

feature_columns = (model_feature_names(model) if model is not None else []) \
    or meta.get("feature_columns") or FALLBACK_FEATURES


def default_of(name: str) -> float:
    if ref and name in ref["median"]:
        return float(ref["median"][name])
    iv = meta.get("imputation_values") or {}
    if name in iv:
        return float(iv[name])
    return float(FALLBACK_DEFAULTS.get(name, 0.0))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏠 King County House Price")
    st.caption("Tugas TM11 - Regresi | Kelompok 5")
    st.divider()
    st.markdown("**Status model terbaik**")
    if model is not None:
        st.success(f"Aktif: {info['name']}")
        t = info["test"]
        if t:
            st.markdown(f"R² uji: **{fmt_metric('R2', t.get('R2'))}**  \nMAPE uji: **{fmt_metric('MAPE', t.get('MAPE'))}**")
        st.caption(f"{len(feature_columns)} fitur input")
    elif model_error:
        st.error("best_model.pkl gagal dimuat.")
        st.caption(model_error)
    else:
        st.warning("best_model.pkl belum tersedia.")
        st.caption("Dibuat oleh 04_compare_models.py setelah XGBoost dan LightGBM dibandingkan.")
    st.divider()
    st.markdown(
        "**Tentang**  \nEstimasi harga rumah berdasarkan data penjualan King County, Washington, USA, "
        "menggunakan model gradient boosting terbaik hasil perbandingan XGBoost dan LightGBM."
    )

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
pill = (f'<span class="pill ok">Model aktif: {info["name"]}</span>' if model is not None
        else '<span class="pill wait">Model terbaik belum tersedia</span>')
st.markdown(
    f"""
    <div class="hero">
      <div>
        <h1>🏠 Estimasi Harga Rumah - King County, USA</h1>
        <p>Masukkan karakteristik rumah dan dapatkan estimasi harga jual dalam USD dari model machine learning.</p>
      </div>
      <div>{pill}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_model, tab_about = st.tabs(["🔮 Prediksi Harga", "🏆 Model Terbaik", "📘 Tentang Proyek"])


# ----------------------------------------------------------------------------
# Komponen input
# ----------------------------------------------------------------------------
def render_input(name: str, zip_value=None):
    """Menampilkan satu widget input dan mengembalikan nilainya."""
    spec = SPECS.get(name, dict(label=name, kind="float", min=-1e9, max=1e9, step=1.0, fmt="%.4f", help=""))
    kind = spec.get("kind")
    key = f"in_{name}"
    helptext = spec.get("help")
    d = default_of(name)

    if name == "zipcode":
        if ref and ref["zip_table"]:
            opts = sorted(ref["zip_table"].keys())
            dz = str(int(d))
            idx = opts.index(dz) if dz in opts else 0
            return int(st.selectbox(spec["label"], opts, index=idx, key=key, help=helptext))
        return int(st.number_input(spec["label"], 98001, 98199, int(clamp(d, 98001, 98199)), 1, key=key, help=helptext))

    if name in ("lat", "long"):
        if ref and ref["zip_table"] and zip_value is not None and str(zip_value) in ref["zip_table"]:
            d = ref["zip_table"][str(zip_value)][name]
        return float(st.number_input(spec["label"], float(spec["min"]), float(spec["max"]),
                                     float(clamp(d, spec["min"], spec["max"])), float(spec["step"]),
                                     format=spec["fmt"], key=f"{key}_{zip_value}", help=helptext))

    if kind == "int":
        return int(st.number_input(spec["label"], int(spec["min"]), int(spec["max"]),
                                   int(clamp(round(d), spec["min"], spec["max"])), int(spec["step"]),
                                   key=key, help=helptext))
    if kind == "float":
        return float(st.number_input(spec["label"], float(spec["min"]), float(spec["max"]),
                                     float(clamp(d, spec["min"], spec["max"])), float(spec["step"]),
                                     format=spec.get("fmt", "%.2f"), key=key, help=helptext))
    if kind == "slider":
        return int(st.slider(spec["label"], int(spec["min"]), int(spec["max"]),
                             int(clamp(round(d), spec["min"], spec["max"])), key=key, help=helptext))
    if kind == "select":
        opts = list(spec["options"])
        dv = int(round(d)) if int(round(d)) in opts else opts[0]
        return int(st.select_slider(spec["label"], options=opts, value=dv, key=key,
                                    format_func=lambda v, o=spec["options"]: o[v], help=helptext))
    if kind == "toggle":
        return int(st.toggle(spec["label"], value=bool(round(d)), key=key, help=helptext))
    return float(st.number_input(name, value=float(d), key=key))


# ============================================================================
# TAB 1 - PREDIKSI
# ============================================================================
with tab_predict:
    if model is None:
        if model_error:
            st.error(f"Model terbaik tidak dapat dimuat. {model_error}")
        st.warning(
            "**Model terbaik belum tersedia.** Formulir tetap dapat dicoba, tetapi tombol prediksi aktif "
            "setelah `models/best_model.pkl` dibuat. Jalankan berurutan: `02_xgboost.py`, `03_lightgbm.py`, "
            "lalu `04_compare_models.py`."
        )

    col_form, col_result = st.columns([1.4, 1], gap="large")
    values: dict[str, float] = {}

    with col_form:
        shown = set()
        for title, names in GROUPS:
            present = [n for n in names if n in feature_columns]
            extra_above = "sqft_above" in feature_columns and title.startswith("📐")
            if not present and not extra_above:
                continue
            st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
            ncol = 3 if len(present) >= 3 else max(len(present), 1)
            cols = st.columns(ncol)
            for i, n in enumerate(present):
                with cols[i % ncol]:
                    values[n] = render_input(n, zip_value=values.get("zipcode"))
                shown.add(n)
            # sqft_above: dihitung otomatis bila luas bangunan & basement tersedia sebagai input
            if extra_above:
                if "sqft_living" in values and "sqft_basement" in values:
                    values["sqft_above"] = max(int(values["sqft_living"]) - int(values["sqft_basement"]), 0)
                    st.caption(f"Luas di atas tanah (`sqft_above`) dihitung otomatis: "
                               f"**{values['sqft_above']:,} sqft** (bangunan - basement).")
                else:
                    values["sqft_above"] = render_input("sqft_above")
                shown.add("sqft_above")

        # Fitur lain yang ada pada model tetapi tidak dikenal oleh formulir
        others = [n for n in feature_columns if n not in shown]
        if others:
            with st.expander("Fitur tambahan model"):
                for n in others:
                    values[n] = render_input(n)

        # Validasi ringan (tidak memblokir)
        if values.get("yr_renovated") and values.get("yr_built") and values["yr_renovated"] < values["yr_built"]:
            st.warning("Tahun renovasi lebih awal daripada tahun dibangun. Periksa kembali input.")
        if values.get("sqft_basement", 0) > values.get("sqft_living", 10**9):
            st.warning("Luas basement melebihi luas bangunan. Periksa kembali input.")
        if ref:
            outside = [n for n in ("sqft_living", "sqft_lot", "sqft_living15", "sqft_lot15", "bathrooms")
                       if n in values and n in ref["min"] and not (ref["min"][n] <= values[n] <= ref["max"][n])]
            if outside:
                st.info("Nilai " + ", ".join(f"`{n}`" for n in outside) +
                        " berada di luar rentang data latih sehingga estimasi kurang andal.")

        predict_clicked = st.button("🔮 Prediksi Harga", type="primary", width="stretch", disabled=(model is None))

    with col_result:
        st.markdown('<div class="section-title">💰 Hasil Estimasi</div>', unsafe_allow_html=True)

        if predict_clicked and model is not None:
            try:
                row = {n: values.get(n, default_of(n)) for n in feature_columns}
                X_input = pd.DataFrame([row], columns=feature_columns)
                price = predict_price(model, X_input, info["target_transform"])
                st.session_state["result"] = {"price": price, "inputs": row, "model": info["name"]}
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"Prediksi gagal: {type(exc).__name__}: {exc}")

        result = st.session_state.get("result")
        if result and model is not None:
            price = result["price"]
            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-label">Estimasi harga rumah</div>
                  <div class="result-price">{usd(price)}</div>
                  <div class="result-sub">Model: {result['model']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            current = {n: values.get(n, default_of(n)) for n in feature_columns}
            if current != result["inputs"]:
                st.caption("⚠️ Input telah diubah sejak prediksi terakhir. Tekan **Prediksi Harga** untuk memperbarui.")

            mape = (info["test"] or {}).get("MAPE")
            if mape is not None:
                lo, hi = price * (1 - mape / 100), price * (1 + mape / 100)
                st.info(f"Rentang indikatif (± MAPE data uji {mape:.2f}%): **{usd(lo)} - {usd(hi)}**")

            m1, m2 = st.columns(2)
            sq = result["inputs"].get("sqft_living")
            if sq:
                m1.metric("Harga per sqft", usd(price / max(sq, 1)))
                m2.metric("Luas bangunan (sqft)", f"{int(sq):,}")
            if ref and ref.get("prices") is not None and len(ref["prices"]):
                prices = ref["prices"]
                pct = float((prices < price).mean() * 100)
                st.markdown(
                    f"Posisi harga: lebih tinggi daripada **{pct:.0f}%** rumah pada data latih "
                    f"(median data latih {usd(float(np.median(prices)))})."
                )
            with st.expander("Lihat input yang dikirim ke model"):
                st.dataframe(pd.DataFrame(result["inputs"], index=["nilai"]).T, width="stretch")
            st.caption("Estimasi bersifat indikatif berdasarkan data penjualan 2014-2015 dan bukan penilaian (appraisal) resmi.")
        else:
            st.markdown(
                """
                <div class="placeholder-card">
                  Isi karakteristik rumah di sebelah kiri,<br>lalu tekan <b>Prediksi Harga</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================================
# TAB 2 - MODEL TERBAIK
# ============================================================================
def render_pipeline_status():
    rows = [{"Dibuat oleh": who, "Berkas": label, "Status": "✅ Tersedia" if p.exists() else "⏳ Belum ada"}
            for who, p, label in PIPELINE_FILES]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


with tab_model:
    if model is None:
        st.markdown('<div class="section-title">Model terbaik belum tersedia</div>', unsafe_allow_html=True)
        st.markdown(
            "Halaman ini akan menampilkan **algoritma, hiperparameter, metrik evaluasi, perbandingan model, "
            "dan grafik** dari model terbaik secara otomatis setelah `models/best_model.pkl` dibuat oleh "
            "`04_compare_models.py`. Berikut status berkas pada alur proyek:"
        )
        render_pipeline_status()
        with st.expander("Format berkas yang dibaca aplikasi"):
            st.code(
                'models/best_model.pkl        -> joblib.dump(model_terbaik, ...)  # XGBRegressor / LGBMRegressor\n'
                'models/best_model_info.json  -> {"model_name": "XGBoost" | "LightGBM",\n'
                '                                 "selection_metric": "RMSE",\n'
                '                                 "train_metrics": {MAE, MAPE, RMSE, MSE, R2},\n'
                '                                 "test_metrics":  {MAE, MAPE, RMSE, MSE, R2},\n'
                '                                 "n_train": ..., "n_test": ...}\n'
                'results/model_comparison.csv -> tabel perbandingan XGBoost vs LightGBM\n'
                'results/<model>_predictions.csv -> kolom actual, predicted, residual',
                language="text",
            )
    else:
        algo = ALGO_INFO.get(info["key"])
        st.markdown(f'<div class="section-title">🏆 Model terbaik: {info["name"]}</div>', unsafe_allow_html=True)
        if info["selection_metric"]:
            st.caption(f"Dipilih berdasarkan metrik {info['selection_metric']} pada data uji.")

        # --- metrik uji ---
        test_m = info["test"]
        if test_m:
            st.markdown("**Performa pada data uji**" + (f" ({int(info['n_test']):,} rumah)" if info["n_test"] else ""))
            cols = st.columns(5)
            for c, mname in zip(cols, METRIC_ORDER):
                c.metric(mname if mname != "R2" else "R²", fmt_metric(mname, test_m.get(mname)))
            if info["train"]:
                tbl = pd.DataFrame({
                    "Metrik": [m if m != "R2" else "R²" for m in METRIC_ORDER],
                    "Data latih": [fmt_metric(m, info["train"].get(m)) for m in METRIC_ORDER],
                    "Data uji": [fmt_metric(m, test_m.get(m)) for m in METRIC_ORDER],
                })
                st.dataframe(tbl, width="stretch", hide_index=True)
        else:
            st.info("Metrik evaluasi belum ditemukan. Sediakan `models/best_model_info.json` atau "
                    f"`results/{info['key']}_metrics.json`.")

        # --- algoritma & hiperparameter ---
        st.markdown('<div class="section-title">Algoritma dan konfigurasi</div>', unsafe_allow_html=True)
        ca, cb = st.columns([1.2, 1], gap="large")
        with ca:
            if algo:
                st.markdown(f"**{algo['title']}**")
                st.write(algo["summary"])
                st.dataframe(pd.DataFrame(algo["rows"], columns=["Karakteristik", "Keterangan"]),
                             width="stretch", hide_index=True)
                st.caption("Sumber: modul TM11 Regresi.")
            else:
                st.write(f"Algoritma: **{info['name']}** (kelas `{type(model).__name__}`).")
        with cb:
            st.markdown("**Hiperparameter model**")
            prm = [(k, str(info["params"][k])) for k in PARAM_WHITELIST if k in info["params"]]
            if prm:
                st.dataframe(pd.DataFrame(prm, columns=["Parameter", "Nilai"]), width="stretch", hide_index=True)
            if info["n_train"]:
                st.caption(f"Dilatih pada {int(info['n_train']):,} data; {len(feature_columns)} fitur.")

        # --- perbandingan model ---
        comp = read_csv(COMPARISON_PATH)
        if comp is not None and len(comp):
            st.markdown('<div class="section-title">Perbandingan model</div>', unsafe_allow_html=True)
            st.dataframe(comp, width="stretch", hide_index=True)

        # --- grafik ---
        st.markdown('<div class="section-title">Visualisasi hasil prediksi</div>', unsafe_allow_html=True)
        preds = read_csv(RESULTS_DIR / f"{info['key']}_predictions.csv")
        g1, g2 = st.columns(2, gap="large")
        if preds is not None and {"actual", "predicted"}.issubset(preds.columns):
            p = preds.copy()
            if "residual" not in p.columns:
                p["residual"] = p["actual"] - p["predicted"]
            lim = float(max(p["actual"].max(), p["predicted"].max()))
            r_lo, r_hi = (float(v) for v in p["residual"].quantile([0.01, 0.99]))
            with g1:
                st.markdown("**Harga aktual vs prediksi (data uji)**")
                pts = alt.Chart(p).mark_circle(size=14, opacity=0.35).encode(
                    x=alt.X("actual:Q", title="Harga aktual (USD)"),
                    y=alt.Y("predicted:Q", title="Harga prediksi (USD)"),
                    tooltip=[alt.Tooltip("actual:Q", format=",.0f"), alt.Tooltip("predicted:Q", format=",.0f")],
                )
                line = alt.Chart(pd.DataFrame({"v": [0, lim]})).mark_line(color="red").encode(x="v:Q", y="v:Q")
                st.altair_chart(pts + line, width="stretch")
            with g2:
                st.markdown("**Distribusi residual (aktual - prediksi)**")
                pc = p[(p["residual"] >= r_lo) & (p["residual"] <= r_hi)]
                hist = alt.Chart(pc).mark_bar(color="#2b78c2").encode(
                    x=alt.X("residual:Q", bin=alt.Bin(maxbins=50), title="Residual (USD)"),
                    y=alt.Y("count():Q", title="Frekuensi"),
                )
                st.altair_chart(hist, width="stretch")
                st.caption(f"Ditampilkan untuk residual persentil 1-99 ({usd(r_lo)} sampai {usd(r_hi)}) agar bentuk distribusi terlihat.")
        else:
            st.info(f"Grafik aktual vs prediksi akan tampil setelah `results/{info['key']}_predictions.csv` tersedia.")

        try:
            imp = np.asarray(model.feature_importances_, dtype=float)
            if len(imp) == len(feature_columns) and imp.sum() > 0:
                fi = pd.DataFrame({"Fitur": feature_columns, "Kepentingan (%)": imp / imp.sum() * 100})
                fi = fi.sort_values("Kepentingan (%)", ascending=False).head(10)
                st.markdown("**10 fitur paling berpengaruh pada model**")
                bar = alt.Chart(fi).mark_bar(color="#13487f").encode(
                    x=alt.X("Kepentingan (%):Q"), y=alt.Y("Fitur:N", sort="-x"),
                    tooltip=["Fitur", alt.Tooltip("Kepentingan (%):Q", format=".2f")],
                )
                st.altair_chart(bar, width="stretch")
                st.caption("Nilai kepentingan menunjukkan seberapa besar fitur dipakai model, bukan hubungan sebab-akibat.")
        except Exception:
            pass


# ============================================================================
# TAB 3 - TENTANG
# ============================================================================
with tab_about:
    st.markdown('<div class="section-title">Alur proyek</div>', unsafe_allow_html=True)
    st.markdown(
        "1. **Preprocessing** (`01_preprocessing.py`): pembersihan data, imputasi median, dan pembagian 80% latih : 20% uji.\n"
        "2. **XGBoost** (`02_xgboost.py`) dan **LightGBM** (`03_lightgbm.py`): dilatih pada data hasil preprocessing yang sama.\n"
        "3. **Perbandingan** (`04_compare_models.py`): metrik MAE, MAPE, RMSE, MSE, dan R² dibandingkan; model terbaik disimpan sebagai `best_model.pkl`.\n"
        "4. **Aplikasi** (`app.py`): memuat model terbaik untuk estimasi harga."
    )
    n_rows = None
    if meta.get("train_rows") and meta.get("test_rows"):
        n_rows = int(meta["train_rows"]) + int(meta["test_rows"])
    st.markdown('<div class="section-title">Data</div>', unsafe_allow_html=True)
    st.write(
        "Dataset penjualan rumah King County, Washington, USA. Target prediksi adalah `price` (USD)."
        + (f" Setelah preprocessing terdapat {n_rows:,} data ({int(meta['train_rows']):,} latih dan {int(meta['test_rows']):,} uji)." if n_rows else "")
    )
    desc_rows = [(n, SPECS[n]["label"], SPECS[n].get("help", "")) for n in feature_columns if n in SPECS]
    if desc_rows:
        st.markdown("**Fitur yang digunakan model**")
        st.dataframe(pd.DataFrame(desc_rows, columns=["Fitur", "Label pada formulir", "Keterangan"]),
                     width="stretch", hide_index=True)
    st.markdown('<div class="section-title">Catatan penggunaan</div>', unsafe_allow_html=True)
    st.markdown(
        "- Estimasi berasal dari pola data penjualan 2014-2015 dan bukan penilaian resmi.\n"
        "- Model pohon keputusan tidak mengekstrapolasi dengan baik; input di luar rentang data latih kurang andal.\n"
        "- Rentang indikatif memakai MAPE data uji sebagai perkiraan kasar, bukan interval kepercayaan statistik."
    )

st.markdown('<div class="footer">TM11 Regresi - Kelompok 5 | Dataset: King County House Sales</div>', unsafe_allow_html=True)