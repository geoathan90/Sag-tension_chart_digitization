from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

csv_path = PROC_DIR / "sag_tables_tidy.csv"

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["series_label"] = df["series_label"].astype(str)  
    return df

def fit_series(df, label, xcol="span_m", ycol="sag_m"):
    s = (df[df["series_label"] == str(label)]
         .copy().sort_values(xcol))
    x, y = s[xcol].to_numpy(), s[ycol].to_numpy()
    coeffs = np.polyfit(x, y, 2)
    return np.poly1d(coeffs)  

def sag_from_span(span_m, poly):
    return float(poly(span_m))

def main():
    df = load_data(csv_path)
    p0 = fit_series(df, "0")
    print(sag_from_span(320, p0))
    return df, p0

if __name__ == "__main__":
    df, p0 = main()