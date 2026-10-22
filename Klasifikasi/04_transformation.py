"""TAHAP 4 - TRANSFORMASI: Min-Max Scaling (numerik) + One-Hot Encoding (kategorikal).
Fit HANYA pada data training, lalu diterapkan ke data training & testing (mencegah data leakage)."""
import pandas as pd, numpy as np, joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from config import *

X_train = pd.read_csv(p("X_train_raw.csv")); X_test = pd.read_csv(p("X_test_raw.csv"))
kat = X_train.select_dtypes(exclude="number").columns.tolist()
num = X_train.select_dtypes(include="number").columns.tolist()

print("=" * 60)
print("TAHAP 4 - TRANSFORMASI (Min-Max + One-Hot Encoding)")
print("=" * 60)
print(f"Kolom numerik     -> Min-Max Scaling : {num}")
print(f"Kolom kategorikal -> One-Hot Encoding: {kat}")

scaler = MinMaxScaler().fit(X_train[num])
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_train[kat])

joblib.dump({"scaler": scaler, "ohe": ohe, "num": num, "kat": kat}, p("transformer.joblib"))  # dipakai aplikasi Streamlit

def transform(X):
    a = pd.DataFrame(scaler.transform(X[num]), columns=num, index=X.index)
    b = pd.DataFrame(ohe.transform(X[kat]), columns=ohe.get_feature_names_out(kat), index=X.index)
    return pd.concat([a, b], axis=1)

Xtr, Xte = transform(X_train), transform(X_test)
print(f"\nJumlah fitur sebelum transformasi: {X_train.shape[1]}")
print(f"Jumlah fitur sesudah transformasi: {Xtr.shape[1]}")
print("\nContoh data training SEBELUM (kolom numerik):")
print(X_train[num].head(3).round(2).to_string())
print("\nContoh data training SESUDAH Min-Max (rentang 0-1):")
print(Xtr[num].head(3).round(4).to_string())
print("\nNilai min/max kolom numerik data training (harus 0 dan 1):")
print(Xtr[num].agg(["min", "max"]).round(2).to_string())
print("\nContoh hasil One-Hot Encoding:")
print(Xtr[[c for c in Xtr.columns if c not in num]].head(3).astype(int).to_string())

Xtr.to_csv(p("X_train_trans.csv"), index=False); Xte.to_csv(p("X_test_trans.csv"), index=False)
print("\nTersimpan: output/X_train_trans.csv, X_test_trans.csv")
