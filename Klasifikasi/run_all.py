"""Menjalankan seluruh tahap berurutan."""
import subprocess, sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
TAHAP = ["01_input.py", "02_preprocessing.py", "03_split_data.py", "04_transformation.py",
         "05_seleksi_fitur.py", "06_training_testing.py", "07_evaluasi.py", "08_perbandingan.py"]
os.makedirs(os.path.join(BASE, "output"), exist_ok=True)
with open(os.path.join(BASE, "output", "log_program.txt"), "w", encoding="utf-8") as log:
    for t in TAHAP:
        r = subprocess.run([sys.executable, os.path.join(BASE, t)], cwd=BASE, capture_output=True, text=True)
        print(r.stdout); log.write(r.stdout)
        if r.returncode != 0:
            print(r.stderr); sys.exit(f"Gagal di tahap {t}")
        with open(os.path.join(BASE, "output", t.replace(".py", ".txt")), "w", encoding="utf-8") as f:
            f.write(r.stdout)
