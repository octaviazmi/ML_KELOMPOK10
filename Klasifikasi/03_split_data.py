"""TAHAP 3 - SPLIT DATA: 80% training, 20% testing (stratified)."""
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *

df = pd.read_csv(p("data_preprocessed.csv"))
X, y = df.drop(columns=TARGET), df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

print("=" * 60)
print("TAHAP 3 - SPLIT DATA (80% : 20%)")
print("=" * 60)
print(f"Total data   : {len(df)}")
print(f"Data training: {len(X_train)} ({len(X_train)/len(df)*100:.0f}%)")
print(f"Data testing : {len(X_test)} ({len(X_test)/len(df)*100:.0f}%)")
print("\nProporsi kelas churn (stratified, menjaga rasio kelas):")
print(pd.DataFrame({"training (%)": (y_train.value_counts(normalize=True) * 100).round(2),
                    "testing (%)": (y_test.value_counts(normalize=True) * 100).round(2)}).to_string())

X_train.to_csv(p("X_train_raw.csv"), index=False); X_test.to_csv(p("X_test_raw.csv"), index=False)
y_train.to_csv(p("y_train.csv"), index=False);     y_test.to_csv(p("y_test.csv"), index=False)
print("\nTersimpan: output/X_train_raw.csv, X_test_raw.csv, y_train.csv, y_test.csv")
