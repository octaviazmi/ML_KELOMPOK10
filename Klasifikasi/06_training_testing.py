"""TAHAP 6 - TRAINING & TESTING: Naive Bayes, SVM, dan Decision Tree.
Setiap model dicoba pada 3 skenario fitur (sama seperti UTS):
  Semua Fitur | Chi-Square | RFE """
import time, joblib
import pandas as pd, numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from config import *

ytr = pd.read_csv(p("y_train.csv")).squeeze(); yte = pd.read_csv(p("y_test.csv")).squeeze()
SKENARIO = {"Semua Fitur": "trans", "Chi-Square": "chi2", "RFE": "rfe"}
MODEL = {
    "Naive Bayes":   lambda: GaussianNB(),
    "SVM":           lambda: SVC(kernel="rbf", random_state=RANDOM_STATE),
    "Decision Tree": lambda: DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE),
}

print("=" * 60)
print("TAHAP 6 - TRAINING & TESTING (Naive Bayes, SVM, Decision Tree)")
print("=" * 60)
pred_all = {"y_true": yte.values}
ringkas = []
for m_nama, buat in MODEL.items():
    print(f"\n######## {m_nama} ########")
    for s_nama, suf in SKENARIO.items():
        Xtr = pd.read_csv(p(f"X_train_{suf}.csv")); Xte = pd.read_csv(p(f"X_test_{suf}.csv"))
        model = buat()
        t0 = time.time()
        model.fit(Xtr, ytr)                         # TRAINING
        t_train = time.time() - t0
        pred = model.predict(Xte)                   # TESTING
        acc_tr = model.score(Xtr, ytr); acc_te = (pred == yte.values).mean()
        pred_all[f"{m_nama} | {s_nama}"] = pred
        ringkas.append([m_nama, s_nama, Xtr.shape[1], acc_tr, acc_te, int(pred.sum()), round(t_train, 2)])
        print(f"[{s_nama:<11}] fitur={Xtr.shape[1]:>2} | akurasi training={acc_tr:.4f} | akurasi testing={acc_te:.4f} "
              f"| prediksi churn={int(pred.sum()):>4}/{len(pred)} | waktu training={t_train:.2f} dtk")
        joblib.dump(model, p(f"model_{m_nama.replace(' ', '_')}_{suf}.joblib"))

pd.DataFrame(pred_all).to_csv(p("prediksi_testing.csv"), index=False)
pd.DataFrame(ringkas, columns=["Model", "Skenario", "Jumlah Fitur", "Akurasi Training", "Akurasi Testing",
                               "Prediksi Churn", "Waktu Training (dtk)"]).to_csv(p("ringkasan_training.csv"), index=False)
print("\nTersimpan: output/prediksi_testing.csv, ringkasan_training.csv, dan model_*.joblib")
