"""Konfigurasi bersama untuk seluruh tahap pipeline (Soal E)."""
import os
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "intermediate_coreml_assignment_dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TARGET = "churn"
# customer_id = identitas unik (tidak punya makna prediktif)
# target_reg  = target untuk kasus regresi (bukan fitur untuk klasifikasi churn)
DROP_COLS = ["customer_id", "target_reg"]

RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80% training : 20% testing
K_FEATURES = 10           # jumlah fitur yang dipilih oleh tiap metode seleksi fitur

def p(*parts):
    """Path helper ke folder output."""
    return os.path.join(OUT_DIR, *parts)
