# TM11 - Regresi: Prediksi Harga Rumah King County (USA)

Tugas Machine Learning TM11 (Regresi) - **Kelompok 5 (Ganjil: XGBoost & LightGBM)**.
Memprediksi harga rumah (`price`, USD) dari dataset penjualan rumah King County,
Washington, USA (Mei 2014 - Mei 2015).

## Struktur Proyek

```
TM11_Kelompok_5/
├── data/
│   ├── raw/                    # kc_house_data_NaN.csv (data mentah)
│   └── processed/              # X_train, X_test, y_train, y_test (dibuat otomatis)
├── models/                     # xgboost_model.pkl, preprocessing_meta.json, best_model.pkl (nanti)
├── results/                    # metrik (.json), prediksi (.csv), grafik (.png)
├── 01_preprocessing.py         # placeholder
├── 02_xgboost.py               # LENGKAP: preprocessing bersama + XGBoost
├── 03_lightgbm.py              # placeholder
├── 04_compare_models.py        # placeholder
├── app.py                      # LENGKAP: aplikasi Streamlit
├── requirements.txt
├── README.md
└── .gitignore
```

## Instalasi (VS Code)

Butuh Python 3.9+ (disarankan 3.10 ke atas).

1. Ekstrak ZIP, lalu buka foldernya: **File > Open Folder > `TM11_Kelompok_5`**.
2. Buka terminal VS Code (**Terminal > New Terminal**) dan buat virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Aktifkan environment:

   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
     (jika diblokir: `Set-ExecutionPolicy -Scope Process Bypass`)
   - Windows (CMD): `.venv\Scripts\activate.bat`
   - macOS / Linux: `source .venv/bin/activate`

4. Pasang dependensi:

   ```bash
   pip install -r requirements.txt
   ```

5. Pilih interpreter `.venv` di VS Code (**Ctrl+Shift+P > Python: Select Interpreter**).

## Menjalankan Program

### 1. Latih & evaluasi XGBoost

```bash
python 02_xgboost.py
```

Langkah yang dijalankan (mengikuti alur modul TM11):

1. Membaca `data/raw/kc_house_data_NaN.csv` dan menyiapkan fitur `X` serta target `y = price`.
2. Split data **80% training / 20% testing** (`random_state=42`).
3. Training dengan `XGBRegressor.fit()`.
4. Testing dengan `.predict()`, lalu menghitung **MAE, MAPE, RMSE, MSE, R²**
   (data latih dan data uji).
5. Membuat visualisasi aktual vs prediksi, residual, dan feature importance.
6. Menyimpan model dan hasil evaluasi.

Opsi `--rebuild` memaksa preprocessing diulang dari data mentah:

```bash
python 02_xgboost.py --rebuild
```

Keluaran:

| File | Isi |
|------|-----|
| `models/xgboost_model.pkl` | Model XGBoost terlatih |
| `models/preprocessing_meta.json` | Daftar fitur, nilai bawaan, tabel zipcode -> lat/long |
| `data/processed/*.csv` | Data latih/uji hasil preprocessing (dipakai bersama LightGBM) |
| `results/xgboost_metrics.json` | Metrik data latih & uji, parameter, waktu training |
| `results/xgboost_predictions.csv` | Aktual, prediksi, residual data uji |
| `results/xgboost_actual_vs_pred.png` | Scatter aktual vs prediksi |
| `results/xgboost_actual_vs_pred_line.png` | Grafik garis 100 sampel uji |
| `results/xgboost_residuals.png` | Residual vs prediksi + histogram residual |
| `results/xgboost_feature_importance.png` | 15 fitur terpenting |

### 2. Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

Browser terbuka otomatis (biasanya di `http://localhost:8501`).

Aplikasi memuat **`models/best_model.pkl`**. File ini baru ada setelah
`04_compare_models.py` selesai dikembangkan dan memilih model terbaik.
**Sebelum itu aplikasi tetap bisa dibuka**: form tampil normal, status sidebar
menunjukkan "best_model.pkl belum tersedia", dan tombol prediksi dinonaktifkan.

## Preprocessing (dipakai bersama XGBoost & LightGBM)

Dilakukan oleh fungsi `prepare_data()` di `02_xgboost.py`, hasilnya disimpan di
`data/processed/`:

- Kolom `Unnamed: 0` (nomor indeks) dan `id` dibuang.
- `date` (format `20141013T000000`) diubah menjadi `sale_year` dan `sale_month`.
- 1 baris dengan `bedrooms = 33` (salah input; rumah 1.620 sqft) dibuang.
- Split 80/20 dilakukan **sebelum** imputasi.
- NaN pada `bedrooms` dan `bathrooms` diisi **median data latih** (tanpa data leakage).
- Fitur akhir: 20 kolom (18 fitur asli + `sale_year`, `sale_month`).
- Target: `price` (USD, tanpa transformasi).

`03_lightgbm.py` cukup membaca file di `data/processed/` (`X_train.csv`, `X_test.csv`,
`y_train.csv`, `y_test.csv`) agar data latih/uji **identik** dengan XGBoost. Karena nama
`02_xgboost` diawali angka, fungsi `load_processed()` tidak bisa di-`import` biasa; bila
ingin memakainya, gunakan `importlib.import_module("02_xgboost")`.

## Menghubungkan Model Terbaik ke Aplikasi

Agar `app.py` berfungsi penuh, `04_compare_models.py` perlu menyimpan:

- `models/best_model.pkl` - model terbaik, disimpan dengan `joblib.dump(model, ...)`.
  Model harus menerima DataFrame dengan 20 kolom sesuai urutan di
  `models/preprocessing_meta.json` (`feature_columns`).
- *(opsional)* `models/best_model_info.json`, contoh format:
  `{"model_name": "XGBoost", "test_metrics": {"MAE": 0, "MAPE": 0, "RMSE": 0, "MSE": 0, "R2": 0}}`
  (isi dengan angka hasil evaluasi nyata). Bila ada, nama model dan rentang ±MAPE tampil di aplikasi.
- *(opsional)* `results/model_comparison.csv` - tabel perbandingan untuk tab "Performa Model".
- *(opsional)* `results/lightgbm_metrics.json`, `results/lightgbm_actual_vs_pred.png`, dan
  `results/lightgbm_feature_importance.png` - ditampilkan otomatis bila ada.

## Catatan

- Dataset berisi 177 `id` rumah yang muncul lebih dari sekali (dijual ulang pada tanggal
  berbeda). Baris tersebut dipertahankan karena merupakan transaksi berbeda, sehingga
  split acak bisa menempatkan rumah yang sama di data latih dan uji. Hal ini perlu
  disebutkan di laporan.
- Estimasi aplikasi bersifat indikatif berdasarkan data 2014-2015, bukan penilaian resmi.
