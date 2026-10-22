"""TAHAP 6 - TRAINING & TESTING: klasifikasi Decision Tree.
Tiga skenario dibandingkan dengan parameter model yang SAMA:
  (A) Semua fitur (tanpa seleksi) | (B) Chi-Square | (C) RFE"""
import pandas as pd, numpy as np, joblib
from sklearn.tree import DecisionTreeClassifier
from config import *

ytr = pd.read_csv(p("y_train.csv")).squeeze(); yte = pd.read_csv(p("y_test.csv")).squeeze()
SKENARIO = {"Semua Fitur": "trans", "Chi-Square": "chi2", "RFE": "rfe"}

print("=" * 60)
print("TAHAP 6 - TRAINING & TESTING (Decision Tree)")
print("=" * 60)
hasil = {}
for nama, suf in SKENARIO.items():
    Xtr = pd.read_csv(p(f"X_train_{suf}.csv")); Xte = pd.read_csv(p(f"X_test_{suf}.csv"))
    model = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)                       # TRAINING
    pred = model.predict(Xte)                 # TESTING
    hasil[nama] = pred
    print(f"\n[{nama}]")
    print(f"  Jumlah fitur        : {Xtr.shape[1]}")
    print(f"  Kedalaman pohon     : {model.get_depth()}")
    print(f"  Jumlah daun         : {model.get_n_leaves()}")
    print(f"  Akurasi training    : {model.score(Xtr, ytr):.4f}")
    print(f"  Akurasi testing     : {model.score(Xte, yte):.4f}")
    print(f"  Prediksi churn (1)  : {int(pred.sum())} dari {len(pred)} data testing")
    joblib.dump(model, p(f"model_{suf}.joblib"))

pd.DataFrame(hasil).assign(y_true=yte.values).to_csv(p("prediksi_testing.csv"), index=False)
print("\nTersimpan: output/prediksi_testing.csv dan model_*.joblib")
