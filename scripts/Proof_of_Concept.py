from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
POC_DIR = Path(__file__).resolve().parents[1] / "data" / "POC"
PROC_DIR.mkdir(parents=True, exist_ok=True)

csv_path = POC_DIR / "31185.csv"

df = pd.read_csv(csv_path, header=None, names=["x","y"])
x, y = df.x.to_numpy(), df.y.to_numpy()

p = PchipInterpolator(x, y)
#xs = np.linspace(x.min(), x.max(), 500)
#pd.DataFrame({"x": xs, "y_pchip": p(xs)}).to_csv("31185_pchip.csv", index=False)

xs = np.array([float(a) for a in sys.argv[1:]], dtype=float)
ys = p(xs)
for xv, yv in zip(xs, ys):
    print(f"{xv}, {yv}")


#x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
#y = np.array([0, 0.5, 1.2, 1.8, 2.0, 2.1], dtype=float)  # monotone-ish data

#p = PchipInterpolator(x, y)     # build interpolant
#xs = np.linspace(x.min(), x.max(), 200)
#ys = p(xs)                      # evaluate anywhere
