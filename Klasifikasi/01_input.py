"""TAHAP 1 - INPUT: membaca dataset dan melihat gambaran umum data."""
import pandas as pd
from config import *

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("TAHAP 1 - INPUT DATA")
print("=" * 60)
print(f"Jumlah baris  : {df.shape[0]}")
print(f"Jumlah kolom  : {df.shape[1]}")
print("\n5 data teratas:")
print(df.head().to_string())
print("\nTipe data tiap kolom:")
print(df.dtypes.to_string())

kat = df.select_dtypes(exclude="number").columns.tolist()
num = df.select_dtypes(include="number").columns.tolist()
print(f"\nAtribut kategorikal ({len(kat)}): {kat}")
print(f"Atribut numerik     ({len(num)}): {num}")

print("\nDistribusi target (churn):")
dist = df[TARGET].value_counts()
print(pd.concat([dist, (df[TARGET].value_counts(normalize=True) * 100).round(2)],
                axis=1, keys=["jumlah", "persen (%)"]).to_string())

print("\nStatistik deskriptif (numerik):")
print(df[num].describe().round(2).T.to_string())
