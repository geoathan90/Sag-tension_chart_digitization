from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import sys

#RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
#PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
#PROC_DIR.mkdir(parents=True, exist_ok=True)

POC_DIR = Path(__file__).resolve().parents[1] / "data" / "POC"
csv_path = POC_DIR / "31185.csv"

df = pd.read_csv(csv_path, header=None, names=["x","y"])
x, y = df.x.to_numpy(), df.y.to_numpy()

p = PchipInterpolator(x, y)
#xs = np.linspace(x.min(), x.max(), 500) #or any number of points to plot
#pd.DataFrame({"x": xs, "y_pchip": p(xs)}).to_csv("31185_pchip.csv", index=False)

xs = np.array([float(a) for a in sys.argv[1:]], dtype=float)
ys = p(xs)
for xv, yv in zip(xs, ys):
    print(f"{xv}, {yv}")

# run python eval.py 150 200 300.5 | or any number of values - no commas in between


