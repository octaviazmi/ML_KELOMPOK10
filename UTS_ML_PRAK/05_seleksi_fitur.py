"""TAHAP 5 - SELEKSI FITUR: 2 metode seleksi fitur.
  Metode 1 : Chi-Square (SelectKBest)        -> metode FILTER  (statistik, tidak butuh model)
  Metode 2 : RFE + Decision Tree             -> metode WRAPPER (memakai model untuk memilih fitur)
Seleksi di-fit HANYA pada data training."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.tree import DecisionTreeClassifier
from config import *

Xtr = pd.read_csv(p("X_train_trans.csv")); Xte = pd.read_csv(p("X_test_trans.csv"))
ytr = pd.read_csv(p("y_train.csv")).squeeze()
fitur = Xtr.columns.tolist()

print("=" * 60)
print(f"TAHAP 5 - SELEKSI FITUR (memilih {K_FEATURES} dari {len(fitur)} fitur)")
print("=" * 60)

# ---------------- Metode 1: Chi-Square ----------------
print("\n[Metode 1] CHI-SQUARE (SelectKBest, k=%d)" % K_FEATURES)
chi = SelectKBest(score_func=chi2, k=K_FEATURES).fit(Xtr, ytr)
tbl1 = pd.DataFrame({"fitur": fitur, "skor_chi2": chi.scores_, "p_value": chi.pvalues_,
                     "terpilih": chi.get_support()}).sort_values("skor_chi2", ascending=False).reset_index(drop=True)
tbl1.index += 1
print(tbl1.round(4).to_string())
fitur_chi = tbl1[tbl1.terpilih].fitur.tolist()
print(f"\nFitur terpilih Chi-Square: {fitur_chi}")

# ---------------- Metode 2: RFE ----------------
print("\n[Metode 2] RFE (Recursive Feature Elimination) + Decision Tree, n_features=%d" % K_FEATURES)
rfe = RFE(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
          n_features_to_select=K_FEATURES, step=1).fit(Xtr, ytr)
tbl2 = pd.DataFrame({"fitur": fitur, "ranking_RFE": rfe.ranking_, "terpilih": rfe.support_}
                    ).sort_values("ranking_RFE").reset_index(drop=True)
tbl2.index += 1
print(tbl2.to_string())
fitur_rfe = tbl2[tbl2.terpilih].fitur.tolist()
print(f"\nFitur terpilih RFE: {fitur_rfe}")

# ---------------- Perbandingan ----------------
irisan = sorted(set(fitur_chi) & set(fitur_rfe))
print(f"\nFitur yang dipilih KEDUA metode ({len(irisan)}): {irisan}")

# Visualisasi
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
t = tbl1.sort_values("skor_chi2")
ax[0].barh(t.fitur, t.skor_chi2, color=["#C44E52" if s else "#BBBBBB" for s in t.terpilih])
ax[0].set_title("Chi-Square: skor fitur (merah = terpilih)"); ax[0].set_xlabel("Skor chi2")
imp = pd.Series(DecisionTreeClassifier(random_state=RANDOM_STATE).fit(Xtr, ytr).feature_importances_, index=fitur)
r = tbl2.set_index("fitur").ranking_RFE.reindex(imp.index)
t2 = pd.DataFrame({"imp": imp, "terpilih": r == 1}).sort_values("imp")
ax[1].barh(t2.index, t2.imp, color=["#4C72B0" if s else "#BBBBBB" for s in t2.terpilih])
ax[1].set_title("RFE: importance Decision Tree (biru = terpilih)"); ax[1].set_xlabel("Importance")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "04_seleksi_fitur.png"), dpi=130); plt.close()

# Simpan dataset hasil seleksi
Xtr[fitur_chi].to_csv(p("X_train_chi2.csv"), index=False); Xte[fitur_chi].to_csv(p("X_test_chi2.csv"), index=False)
Xtr[fitur_rfe].to_csv(p("X_train_rfe.csv"), index=False);  Xte[fitur_rfe].to_csv(p("X_test_rfe.csv"), index=False)
pd.DataFrame({"chi2": pd.Series(fitur_chi), "rfe": pd.Series(fitur_rfe)}).to_csv(p("fitur_terpilih.csv"), index=False)
print("\nTersimpan: output/X_train_chi2.csv, X_test_chi2.csv, X_train_rfe.csv, X_test_rfe.csv, fitur_terpilih.csv")
