"""UI Streamlit - Prediksi Churn Pelanggan."""
import json, os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
FIG = os.path.join(BASE, "figures")
SUF = {"Semua Fitur": "trans", "Chi-Square": "chi2", "RFE": "rfe"}

st.set_page_config(page_title="Prediksi Churn Pelanggan", page_icon="📉", layout="wide")


@st.cache_resource
def muat_artefak():
    tr = joblib.load(os.path.join(OUT, "transformer.joblib"))
    params = json.load(open(os.path.join(OUT, "preprocess_params.json")))
    terpilih = json.load(open(os.path.join(OUT, "model_terpilih.json")))
    hasil = pd.read_csv(os.path.join(OUT, "hasil_evaluasi.csv"))
    data = pd.read_csv(os.path.join(OUT, "data_preprocessed.csv"))
    return tr, params, terpilih, hasil, data


@st.cache_resource
def muat_model(nama_model, skenario):
    suf = SUF[skenario]
    model = joblib.load(os.path.join(OUT, f"model_{nama_model.replace(' ', '_')}_{suf}.joblib"))
    kolom = pd.read_csv(os.path.join(OUT, f"X_train_{suf}.csv"), nrows=0).columns.tolist()
    return model, kolom


def siapkan_fitur(tr, params, raw, kolom_model):
    """Mengulang preprocessing + transformation yang sama dengan program (capping outlier, Min-Max, One-Hot, pilih fitur)."""
    df = pd.DataFrame([raw])
    for c, (lo, hi) in params["bounds"].items():       # capping outlier (batas IQR dari data)
        df[c] = df[c].clip(lo, hi)
    a = pd.DataFrame(tr["scaler"].transform(df[tr["num"]]), columns=tr["num"])
    b = pd.DataFrame(tr["ohe"].transform(df[tr["kat"]]), columns=tr["ohe"].get_feature_names_out(tr["kat"]))
    return pd.concat([a, b], axis=1)[kolom_model]


try:
    tr, params, terpilih, hasil, data = muat_artefak()
except Exception as e:
    st.error("File hasil program belum ada. Jalankan dulu `python run_all.py`, lalu buka aplikasi ini lagi.")
    st.exception(e)
    st.stop()

st.title("📉 Prediksi Churn Pelanggan")
st.caption("Tugas Klasifikasi - Praktikum Pembelajaran Mesin | Dataset: Intermediate Core ML Assignment Dataset")

# ---------------- sidebar: pilih model ----------------
st.sidebar.header("Pengaturan Model")
daftar = [(m, s) for m in ["Naive Bayes", "SVM", "Decision Tree"] for s in ["Semua Fitur", "Chi-Square", "RFE"]]
label = [f"{m} - {s}" for m, s in daftar]
idx_default = daftar.index((terpilih["model"], terpilih["skenario"]))
pilih = st.sidebar.selectbox("Model", label, index=idx_default)
nama_model, skenario = daftar[label.index(pilih)]
st.sidebar.info(f"Model default (accuracy tertinggi): **{terpilih['model']} - {terpilih['skenario']}**")

row = hasil[(hasil.Model == nama_model) & (hasil.Skenario == skenario)].iloc[0]
st.sidebar.subheader("Kinerja pada data testing")
st.sidebar.metric("Accuracy", f"{row.Accuracy:.4f}")
c1, c2, c3 = st.sidebar.columns(3)
c1.metric("Precision", f"{row.Precision:.3f}"); c2.metric("Recall", f"{row.Recall:.3f}"); c3.metric("F1", f"{row['F1-Score']:.3f}")
if row.Recall == 0:
    st.sidebar.warning("Model ini tidak pernah memprediksi churn pada data testing (recall = 0). "
                       "Accuracy-nya sama dengan proporsi kelas 'Tidak Churn'.")

tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Perbandingan Model", "ℹ️ Tentang"])

# ---------------- tab prediksi ----------------
with tab1:
    st.subheader("Masukkan data pelanggan")
    med = data.median(numeric_only=True)
    cats = {c: list(k) for c, k in zip(tr["kat"], tr["ohe"].categories_)}
    with st.form("form_prediksi"):
        k1, k2, k3 = st.columns(3)
        with k1:
            age = st.number_input("Umur (age)", 18, 65, int(med["age"]))
            tenure = st.number_input("Lama berlangganan, bulan (tenure_months)", 0, 120, int(med["tenure_months"]))
            income = st.number_input("Pendapatan (income)", 0.0, 200000.0, float(round(med["income"], 2)))
            credit = st.number_input("Credit score", 300, 850, int(med["credit_score"]))
        with k2:
            charges = st.number_input("Tagihan bulanan (monthly_charges)", 0.0, 10000.0, float(round(med["monthly_charges"], 2)))
            tickets = st.number_input("Jumlah tiket bantuan (support_tickets)", 0, 20, int(med["support_tickets"]))
            session = st.number_input("Rata-rata durasi sesi (avg_session_time)", 0.0, 100.0, float(round(med["avg_session_time"], 2)))
            late = st.number_input("Keterlambatan bayar (late_payments)", 0, 20, int(med["late_payments"]))
        with k3:
            gender = st.selectbox("Gender", cats["gender"])
            city = st.selectbox("City tier", cats["city_tier"])
            plan = st.selectbox("Plan type", cats["plan_type"])
            pay = st.selectbox("Payment method", cats["payment_method"])
            dev = st.selectbox("Device type", cats["device_type"])
            intl = st.selectbox("Has international", cats["has_international"])
            stud = st.selectbox("Is student", cats["is_student"])
        kirim = st.form_submit_button("Prediksi", type="primary")

    if kirim:
        raw = {"age": age, "tenure_months": tenure, "income": income, "credit_score": credit, "monthly_charges": charges,
               "support_tickets": tickets, "avg_session_time": session, "late_payments": late, "gender": gender,
               "city_tier": city, "plan_type": plan, "payment_method": pay, "device_type": dev,
               "has_international": intl, "is_student": stud}
        model, kolom = muat_model(nama_model, skenario)
        X = siapkan_fitur(tr, params, raw, kolom)
        pred = int(model.predict(X)[0])
        st.markdown(f"**Model:** {nama_model} - {skenario} ({len(kolom)} fitur)")
        if pred == 1:
            st.error("⚠️ Prediksi: pelanggan **CHURN** (berpotensi berhenti berlangganan)")
        else:
            st.success("✅ Prediksi: pelanggan **TIDAK CHURN**")
        if hasattr(model, "predict_proba"):
            pr = model.predict_proba(X)[0]
            st.write(f"Probabilitas tidak churn: **{pr[0]:.2%}** | Probabilitas churn: **{pr[1]:.2%}**")
        if row.Recall == 0:
            st.info("Catatan: model yang dipilih tidak pernah memprediksi churn pada data testing, sehingga hasilnya hampir pasti 'Tidak Churn'. "
                    "Coba pilih model lain di sidebar (misalnya Decision Tree - RFE) untuk perbandingan.")
        with st.expander("Lihat fitur yang masuk ke model"):
            st.dataframe(X.T.rename(columns={0: "nilai (sudah Min-Max/One-Hot)"}))

# ---------------- tab perbandingan ----------------
with tab2:
    st.subheader("Hasil evaluasi seluruh model (data testing)")
    tampil = hasil[["Model", "Skenario", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "F1-Score"]]
    st.dataframe(tampil.style.format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"}),
                 width="stretch", hide_index=True)
    for f, cap in [("perbandingan_model.png", "Perbandingan metrik"), (f"cm_{nama_model.replace(' ', '_')}.png", f"Confusion matrix {nama_model}")]:
        pth = os.path.join(FIG, f)
        if os.path.exists(pth):
            st.image(pth, caption=cap)

with tab3:
    st.markdown("""
**Alur program:** Input → Preprocessing (missing value, duplikasi, outlier) → Split 80:20 → Transformation (Min-Max + One-Hot)
→ Seleksi Fitur (Chi-Square dan RFE) → Training → Testing → Evaluasi.

**Model:** Naive Bayes (GaussianNB), SVM (SVC, kernel RBF), dan Decision Tree.

Input yang Anda masukkan diproses dengan tahapan yang sama seperti pada program (capping outlier, Min-Max Scaling, One-Hot Encoding,
dan pemilihan fitur sesuai skenario) sebelum diberikan ke model.
""")
