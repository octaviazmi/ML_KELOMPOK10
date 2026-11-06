import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                                          
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

                                                                              
                              
                                                                              
BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "raw" / "kc_house_data_NaN.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

TARGET = "price"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_VALID_BEDROOMS = 20                                             

                                         
                                                    
                                                             
                                                          
DROP_COLUMNS = ["Unnamed: 0", "id", "date"]

                                                                              
                                                                         
                                                                              
def prepare_data():

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {RAW_PATH}. "
            "Letakkan kc_house_data_NaN.csv di folder data/raw/."
        )

    df = pd.read_csv(RAW_PATH)
    n_awal = len(df)

                                                                       
    tanggal = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
    df["sale_year"] = tanggal.dt.year
    df["sale_month"] = tanggal.dt.month

                                                                                 
    mask_valid = df["bedrooms"].isna() | (df["bedrooms"] <= MAX_VALID_BEDROOMS)
    n_outlier = int((~mask_valid).sum())
    df = df[mask_valid].reset_index(drop=True)

                               
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    y = df[TARGET].astype(float)
    X = df.drop(columns=DROP_COLUMNS + [TARGET])

                                                                                      
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

                                                                       
    medians = X_train.median(numeric_only=True)
    nan_train = int(X_train.isna().sum().sum())
    nan_test = int(X_test.isna().sum().sum())
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

                                                                                  
    zip_table = (
        X_train.groupby("zipcode")[["lat", "long"]].median().round(5).to_dict("index")
    )
    zip_table = {str(int(k)): v for k, v in zip_table.items()}

    meta = {
        "target": TARGET,
        "feature_columns": list(X_train.columns),
        "defaults": {c: float(medians[c]) for c in X_train.columns},
        "ranges": {
            c: [float(X_train[c].min()), float(X_train[c].max())]
            for c in X_train.columns
        },
        "zipcode_latlong": zip_table,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_raw_rows": int(n_awal),
            "n_outlier_removed": n_outlier,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "nan_filled_train": nan_train,
            "nan_filled_test": nan_test,
        },
    }

                                                                      
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    with open(MODELS_DIR / "preprocessing_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return X_train, X_test, y_train, y_test, meta

def load_processed():

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")[TARGET]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")[TARGET]
    with open(MODELS_DIR / "preprocessing_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return X_train, X_test, y_train, y_test, meta

def get_data(rebuild=False):

    needed = [
        PROCESSED_DIR / "X_train.csv",
        PROCESSED_DIR / "X_test.csv",
        PROCESSED_DIR / "y_train.csv",
        PROCESSED_DIR / "y_test.csv",
        MODELS_DIR / "preprocessing_meta.json",
    ]
    if rebuild or not all(p.exists() for p in needed):
        print("[INFO] Menjalankan preprocessing dari data mentah ...")
        return prepare_data()
    print("[INFO] Memuat data hasil preprocessing dari data/processed/ ...")
    return load_processed()

                                                                              
                                                                   
                                                                              
def hitung_metrik(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)                
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "MAPE": mape,
        "RMSE": rmse,
        "MSE": float(mse),
        "R2": float(r2),
    }

def cetak_metrik(judul, m):
    print(f"\n=== {judul} ===")
    print(f"Mean Absolute Error (MAE)            : {m['MAE']:,.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {m['MAPE']:.2f}%")
    print(f"Root Mean Squared Error (RMSE)       : {m['RMSE']:,.2f}")
    print(f"Mean Squared Error (MSE)             : {m['MSE']:,.2f}")
    print(f"R2 Score                             : {m['R2']:.4f}")

                                                                              
                
                                                                              
def plot_actual_vs_pred(y_test, y_pred, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred, s=10, alpha=0.35, color="tab:blue", label="Data uji")
    batas = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(batas, batas, color="red", linewidth=2, label="Prediksi sempurna (y = x)")
    ax.set_title("XGBoost: Harga Aktual vs Prediksi (Data Uji)")
    ax.set_xlabel("Harga aktual (USD)")
    ax.set_ylabel("Harga prediksi (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_line_sample(y_test, y_pred, path, n=100):

    n = min(n, len(y_test))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(n), np.asarray(y_test)[:n], marker="o", ms=3, label="Aktual")
    ax.plot(range(n), np.asarray(y_pred)[:n], marker="x", ms=4, label="Prediksi XGBoost")
    ax.set_title(f"XGBoost: Aktual vs Prediksi ({n} sampel pertama data uji)")
    ax.set_xlabel("Indeks sampel uji")
    ax.set_ylabel("Harga (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_residuals(y_test, y_pred, path):
    residu = np.asarray(y_test) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residu, s=10, alpha=0.35, color="tab:purple")
    axes[0].axhline(0, color="red", linewidth=2)
    axes[0].set_title("Residual vs Prediksi")
    axes[0].set_xlabel("Harga prediksi (USD)")
    axes[0].set_ylabel("Residual = aktual - prediksi (USD)")
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(residu, bins=60, color="tab:green", alpha=0.8)
    axes[1].set_title("Distribusi Residual")
    axes[1].set_xlabel("Residual (USD)")
    axes[1].set_ylabel("Frekuensi")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_feature_importance(model, feature_names, path, top=15):
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, color="tab:orange")
    ax.set_title(f"XGBoost: {top} Fitur Terpenting")
    ax.set_xlabel("Feature importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

                                                                              
               
                                                                              
def main(rebuild=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

                                       
    X_train, X_test, y_train, y_test, meta = get_data(rebuild=rebuild)
    s = meta["split"]
    print(
        f"[DATA] Baris mentah: {s['n_raw_rows']} | outlier dibuang: {s['n_outlier_removed']} "
        f"| train: {len(X_train)} | test: {len(X_test)}"
    )
    print(f"[DATA] Jumlah fitur: {X_train.shape[1]} -> {list(X_train.columns)}")

                                                 
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,                    
        reg_lambda=1.0,                   
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("\n[TRAINING] Melatih XGBoost Regressor ...")
    t0 = time.time()
    model.fit(X_train, y_train)
    waktu_latih = time.time() - t0
    print(f"[TRAINING] Selesai dalam {waktu_latih:.2f} detik")

                                                          
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)

    metrik_train = hitung_metrik(y_train, y_pred_train)
    metrik_test = hitung_metrik(y_test, y_pred)
    cetak_metrik("Evaluasi Data LATIH (training)", metrik_train)
    cetak_metrik("Evaluasi Data UJI (testing)", metrik_test)

                    
    plot_actual_vs_pred(y_test, y_pred, RESULTS_DIR / "xgboost_actual_vs_pred.png")
    plot_line_sample(y_test, y_pred, RESULTS_DIR / "xgboost_actual_vs_pred_line.png")
    plot_residuals(y_test, y_pred, RESULTS_DIR / "xgboost_residuals.png")
    plot_feature_importance(
        model, X_train.columns, RESULTS_DIR / "xgboost_feature_importance.png"
    )

                                                                           
    contoh = X_test.iloc[[0]]
    pred_contoh = float(model.predict(contoh)[0])
    print(
        f"\n[CONTOH] Rumah uji ke-1 -> aktual: ${y_test.iloc[0]:,.0f} | "
        f"prediksi: ${pred_contoh:,.0f}"
    )

                                      
    dump(model, MODELS_DIR / "xgboost_model.pkl")

    hasil = {
        "model": "XGBoost",
        "train_metrics": metrik_train,
        "test_metrics": metrik_test,
        "training_time_seconds": round(waktu_latih, 3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(X_train.columns),
        "params": {k: v for k, v in model.get_params().items() if v is not None},
    }
    with open(RESULTS_DIR / "xgboost_metrics.json", "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, default=str)

    pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": y_pred,
            "residual": y_test.values - y_pred,
        }
    ).to_csv(RESULTS_DIR / "xgboost_predictions.csv", index=False)

    print("\n[SIMPAN] Model   : models/xgboost_model.pkl")
    print("[SIMPAN] Metrik  : results/xgboost_metrics.json")
    print("[SIMPAN] Prediksi: results/xgboost_predictions.csv")
    print("[SIMPAN] Grafik  : results/xgboost_*.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TM11 - XGBoost Regressor harga rumah")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="paksa preprocessing ulang dari data mentah",
    )
    args = parser.parse_args()
    main(rebuild=args.rebuild)
