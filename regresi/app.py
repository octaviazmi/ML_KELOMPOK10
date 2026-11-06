
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Path & konfigurasi
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
META_PATH = MODELS_DIR / "preprocessing_meta.json"
BEST_INFO_PATH = MODELS_DIR / "best_model_info.json"  # opsional (dibuat oleh 04)

# Urutan fitur & nilai bawaan (median data latih). Dipakai hanya jika
# models/preprocessing_meta.json belum ada; bila ada, isi file itu yang dipakai.
FALLBACK_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
    "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built",
    "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15",
    "sale_year", "sale_month",
]
FALLBACK_DEFAULTS = {
    "bedrooms": 3, "bathrooms": 2.25, "sqft_living": 1910, "sqft_lot": 7618,
    "floors": 1.5, "waterfront": 0, "view": 0, "condition": 3, "grade": 7,
    "sqft_above": 1560, "sqft_basement": 0, "yr_built": 1975, "yr_renovated": 0,
    "zipcode": 98065, "lat": 47.5718, "long": -122.230, "sqft_living15": 1840,
    "sqft_lot15": 7620, "sale_year": 2015, "sale_month": 5,
}

VIEW_LABELS = {0: "0 - Tidak ada", 1: "1 - Cukup", 2: "2 - Sedang", 3: "3 - Baik", 4: "4 - Sangat baik"}
CONDITION_LABELS = {1: "1 - Buruk", 2: "2 - Kurang", 3: "3 - Rata-rata", 4: "4 - Baik", 5: "5 - Sangat baik"}

st.set_page_config(
    page_title="Estimasi Harga Rumah King County",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Tampilan (CSS)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1200px;}
    .hero {
        background: linear-gradient(120deg, #0f2a4a 0%, #1d4f8c 60%, #2b78c2 100%);
        color: #ffffff; padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 1.9rem; font-weight: 700; color: #ffffff;}
    .hero p {margin: .35rem 0 0 0; opacity: .88; font-size: .98rem;}
    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #1d4f8c;
        border-left: 4px solid #2b78c2; padding-left: .6rem; margin: .4rem 0 .6rem 0;
    }
    .result-card {
        background: linear-gradient(135deg, #f4f9ff 0%, #e6f1fc 100%);
        border: 1px solid #c5dcf3; border-radius: 16px; padding: 1.6rem 1.4rem;
        text-align: center; margin-top: .4rem;
    }
    .result-label {color: #4a6785; font-size: .9rem; letter-spacing: .06em; text-transform: uppercase;}
    .result-price {color: #0f2a4a; font-size: 2.9rem; font-weight: 800; margin: .2rem 0;}
    .result-range {color: #4a6785; font-size: .95rem;}
    .placeholder-card {
        border: 2px dashed #c5dcf3; border-radius: 16px; padding: 2.2rem 1rem;
        text-align: center; color: #6b819a; background: #fafcff;
    }
    .footer {text-align: center; color: #8a99ab; font-size: .82rem; margin-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Pemuatan model & metadata
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path_str, mtime):
    """mtime ikut sebagai argumen agar cache diperbarui jika file model berganti."""
    return joblib.load(path_str)


def get_model():
    if not BEST_MODEL_PATH.exists():
        return None, None
    try:
        return load_model(str(BEST_MODEL_PATH), BEST_MODEL_PATH.stat().st_mtime), None
    except Exception as exc:  # file rusak / versi library berbeda
        return None, str(exc)


@st.cache_data(show_spinner=False)
def load_json(path_str, mtime):
    with open(path_str, encoding="utf-8") as f:
        return json.load(f)


def read_json_if_exists(path):
    if path.exists():
        try:
            return load_json(str(path), path.stat().st_mtime)
        except Exception:
            return None
    return None


meta = read_json_if_exists(META_PATH)
best_info = read_json_if_exists(BEST_INFO_PATH)
model, model_error = get_model()

feature_columns = meta["feature_columns"] if meta else FALLBACK_FEATURES
defaults = meta["defaults"] if meta else FALLBACK_DEFAULTS
zip_table = meta["zipcode_latlong"] if meta else {}


def model_display_name():
    if best_info and best_info.get("model_name"):
        return best_info["model_name"]
    if model is not None:
        return type(model).__name__
    return "-"


def get_mape():
    """MAPE (%) data uji model terbaik, bila info tersedia."""
    if not best_info:
        return None
    m = best_info.get("test_metrics") or best_info.get("metrics") or {}
    return m.get("MAPE")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏠 King County House Price")
    st.caption("Tugas TM11 - Regresi | Kelompok 5")
    st.divider()
    st.markdown("**Status model**")
    if model is not None:
        st.success(f"Model aktif: {model_display_name()}")
    elif model_error:
        st.error("best_model.pkl gagal dimuat.")
    else:
        st.warning("best_model.pkl belum tersedia.")
    if not meta:
        st.info("preprocessing_meta.json belum ada; memakai nilai bawaan.")
    st.divider()
    st.markdown(
        "**Tentang**  \n"
        "Estimasi harga rumah berdasarkan data penjualan King County, "
        "Washington, USA (Mei 2014 - Mei 2015) menggunakan model "
        "regresi berbasis gradient boosting."
    )

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏠 Estimasi Harga Rumah - King County, USA</h1>
        <p>Masukkan karakteristik rumah, lalu dapatkan estimasi harga jual dalam USD
        dari model machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_perf = st.tabs(["🔮 Prediksi Harga", "📊 Performa Model"])

# ============================================================================
# TAB 1 - PREDIKSI
# ============================================================================
with tab_predict:
    if model is None:
        if model_error:
            st.error(f"Model terbaik tidak bisa dimuat: {model_error}")
        st.warning(
            "**Model terbaik belum tersedia.** Form di bawah tetap bisa dicoba, "
            "tetapi tombol prediksi aktif setelah `models/best_model.pkl` dibuat. "
            "Jalankan berurutan: `02_xgboost.py`, `03_lightgbm.py`, lalu "
            "`04_compare_models.py`."
        )

    col_form, col_result = st.columns([1.35, 1], gap="large")

    with col_form:
        # --- Lokasi ---
        st.markdown('<div class="section-title">📍 Lokasi</div>', unsafe_allow_html=True)
        if zip_table:
            zip_options = sorted(zip_table.keys())
            default_zip = str(int(defaults.get("zipcode", 98065)))
            zip_index = zip_options.index(default_zip) if default_zip in zip_options else 0
            zipcode = int(st.selectbox("Kode pos (zipcode)", zip_options, index=zip_index))
            lat_default = float(zip_table[str(zipcode)]["lat"])
            long_default = float(zip_table[str(zipcode)]["long"])
        else:
            zipcode = int(st.number_input("Kode pos (zipcode)", 98001, 98199, int(defaults["zipcode"]), 1))
            lat_default, long_default = float(defaults["lat"]), float(defaults["long"])

        c1, c2 = st.columns(2)
        # key memuat zipcode agar nilai bawaan lat/long ikut berubah saat zipcode diganti
        lat = c1.number_input("Latitude", 47.0, 48.0, lat_default, 0.001, format="%.4f", key=f"lat_{zipcode}")
        long = c2.number_input("Longitude", -123.0, -121.0, long_default, 0.001, format="%.4f", key=f"long_{zipcode}")
        st.caption("Lat/long terisi otomatis dari kode pos (median data latih); boleh diubah.")

        # --- Ukuran & tata ruang ---
        st.markdown('<div class="section-title">📐 Ukuran & Tata Ruang</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        bedrooms = c1.number_input("Kamar tidur", 1, 20, int(defaults["bedrooms"]), 1)
        bathrooms = c2.number_input("Kamar mandi", 0.5, 8.0, float(defaults["bathrooms"]), 0.25)
        floors = c3.number_input("Lantai", 1.0, 3.5, float(defaults["floors"]), 0.5)

        c1, c2, c3 = st.columns(3)
        sqft_living = c1.number_input("Luas bangunan (sqft)", 290, 14000, int(defaults["sqft_living"]), 10)
        sqft_basement = c2.number_input("Luas basement (sqft)", 0, 5000, int(defaults["sqft_basement"]), 10)
        sqft_lot = c3.number_input("Luas tanah (sqft)", 500, 1_700_000, int(defaults["sqft_lot"]), 100)

        sqft_above = max(int(sqft_living) - int(sqft_basement), 0)
        st.caption(f"Luas di atas tanah (sqft_above) dihitung otomatis: **{sqft_above:,} sqft** (bangunan - basement).")

        # --- Kualitas ---
        st.markdown('<div class="section-title">⭐ Kualitas & Fitur</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        grade = c1.slider("Grade konstruksi (1-13)", 1, 13, int(defaults["grade"]))
        condition = c2.select_slider(
            "Kondisi", options=list(CONDITION_LABELS), value=int(defaults["condition"]),
            format_func=lambda v: CONDITION_LABELS[v],
        )
        view = c3.select_slider(
            "Pemandangan", options=list(VIEW_LABELS), value=int(defaults["view"]),
            format_func=lambda v: VIEW_LABELS[v],
        )
        waterfront = int(st.toggle("Tepi air (waterfront)", value=bool(defaults["waterfront"])))

        # --- Tahun ---
        st.markdown('<div class="section-title">🗓️ Tahun & Transaksi</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        yr_built = c1.number_input("Tahun dibangun", 1900, 2015, int(defaults["yr_built"]), 1)
        yr_renovated = c2.number_input("Tahun renovasi (0 = belum)", 0, 2015, int(defaults["yr_renovated"]), 1)
        sale_year = c3.selectbox("Tahun jual", [2014, 2015], index=1)
        sale_month = c4.selectbox("Bulan jual", list(range(1, 13)), index=int(defaults["sale_month"]) - 1)

        # --- Lingkungan ---
        with st.expander("🏘️ Lingkungan sekitar (rata-rata 15 tetangga terdekat)"):
            c1, c2 = st.columns(2)
            sqft_living15 = c1.number_input("Luas bangunan tetangga (sqft)", 300, 7000, int(defaults["sqft_living15"]), 10)
            sqft_lot15 = c2.number_input("Luas tanah tetangga (sqft)", 600, 900_000, int(defaults["sqft_lot15"]), 100)

        # Peringatan validasi ringan (tidak memblokir)
        if yr_renovated and yr_renovated < yr_built:
            st.warning("Tahun renovasi lebih awal dari tahun dibangun. Periksa kembali input.")
        if sqft_basement > sqft_living:
            st.warning("Luas basement melebihi luas bangunan. Periksa kembali input.")

        predict_clicked = st.button(
            "🔮 Prediksi Harga", type="primary", width="stretch", disabled=(model is None)
        )

    # ------------------------------------------------------------------
    # Hasil prediksi
    # ------------------------------------------------------------------
    with col_result:
        st.markdown('<div class="section-title">💰 Hasil Estimasi</div>', unsafe_allow_html=True)

        input_values = {
            "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living,
            "sqft_lot": sqft_lot, "floors": floors, "waterfront": waterfront,
            "view": view, "condition": condition, "grade": grade,
            "sqft_above": sqft_above, "sqft_basement": sqft_basement,
            "yr_built": yr_built, "yr_renovated": yr_renovated, "zipcode": zipcode,
            "lat": lat, "long": long, "sqft_living15": sqft_living15,
            "sqft_lot15": sqft_lot15, "sale_year": sale_year, "sale_month": sale_month,
        }

        if predict_clicked and model is not None:
            try:
                X_input = pd.DataFrame([input_values])[feature_columns]
                price = float(model.predict(X_input)[0])
                price = max(price, 0.0)
                st.session_state["last_result"] = {
                    "price": price,
                    "inputs": input_values,
                    "model": model_display_name(),
                }
            except Exception as exc:
                st.session_state.pop("last_result", None)
                st.error(f"Prediksi gagal: {exc}")

        result = st.session_state.get("last_result")
        if result:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimasi harga rumah</div>
                    <div class="result-price">${result['price']:,.0f}</div>
                    <div class="result-range">Model: {result['model']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            mape = get_mape()
            if mape is not None:
                low = result["price"] * (1 - mape / 100)
                high = result["price"] * (1 + mape / 100)
                st.info(
                    f"Rentang indikatif (±MAPE data uji {mape:.2f}%): "
                    f"**${low:,.0f} - ${high:,.0f}**"
                )
            else:
                st.caption("Rentang kesalahan akan tampil setelah info model terbaik tersedia.")

            m1, m2 = st.columns(2)
            sqft = max(result["inputs"]["sqft_living"], 1)
            m1.metric("Harga per sqft", f"${result['price'] / sqft:,.0f}")
            m2.metric("Luas bangunan", f"{result['inputs']['sqft_living']:,} sqft")

            with st.expander("Lihat input yang dipakai model"):
                st.dataframe(
                    pd.DataFrame(result["inputs"], index=["nilai"]).T,
                    width="stretch",
                )
            st.caption(
                "Estimasi bersifat indikatif berdasarkan data historis 2014-2015 "
                "dan bukan penilaian (appraisal) resmi."
            )
        else:
            st.markdown(
                """
                <div class="placeholder-card">
                    Isi karakteristik rumah di sebelah kiri,<br>
                    lalu tekan <b>Prediksi Harga</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================================
# TAB 2 - PERFORMA MODEL
# ============================================================================
with tab_perf:
    st.markdown('<div class="section-title">Ringkasan Performa (Data Uji 20%)</div>', unsafe_allow_html=True)

    shown_any = False

    comparison_csv = RESULTS_DIR / "model_comparison.csv"
    if comparison_csv.exists():
        try:
            st.markdown("**Perbandingan model**")
            st.dataframe(pd.read_csv(comparison_csv), width="stretch", hide_index=True)
            shown_any = True
        except Exception:
            st.warning("model_comparison.csv tidak dapat dibaca.")

    # Metrik tiap model yang sudah dilatih (dibaca dari hasil nyata, tidak diketik manual)
    for label, fname in [("XGBoost", "xgboost_metrics.json"), ("LightGBM", "lightgbm_metrics.json")]:
        data = read_json_if_exists(RESULTS_DIR / fname)
        if not data:
            continue
        tm = data.get("test_metrics", {})
        st.markdown(f"**{label}**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("MAE", f"${tm.get('MAE', float('nan')):,.0f}")
        c2.metric("MAPE", f"{tm.get('MAPE', float('nan')):.2f}%")
        c3.metric("RMSE", f"${tm.get('RMSE', float('nan')):,.0f}")
        c4.metric("MSE", f"{tm.get('MSE', float('nan')):,.2e}")
        c5.metric("R²", f"{tm.get('R2', float('nan')):.4f}")
        shown_any = True

    images = [
        ("Aktual vs Prediksi (XGBoost)", RESULTS_DIR / "xgboost_actual_vs_pred.png"),
        ("Fitur terpenting (XGBoost)", RESULTS_DIR / "xgboost_feature_importance.png"),
        ("Aktual vs Prediksi (LightGBM)", RESULTS_DIR / "lightgbm_actual_vs_pred.png"),
        ("Fitur terpenting (LightGBM)", RESULTS_DIR / "lightgbm_feature_importance.png"),
    ]
    existing = [(t, p) for t, p in images if p.exists()]
    if existing:
        st.markdown("**Visualisasi**")
        cols = st.columns(2)
        for i, (title, path) in enumerate(existing):
            cols[i % 2].image(str(path), caption=title, width="stretch")
        shown_any = True

    if not shown_any:
        st.info(
            "Hasil evaluasi belum tersedia. Jalankan `02_xgboost.py` (dan `03_lightgbm.py`) "
            "terlebih dahulu; metrik dan grafik akan muncul di sini otomatis."
        )

st.markdown(
    '<div class="footer">TM11 Regresi - Kelompok 5 | Dataset: King County House Sales</div>',
    unsafe_allow_html=True,
)
