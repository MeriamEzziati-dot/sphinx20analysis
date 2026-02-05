import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# USER INPUT (ONLY CHANGE THIS)
# =========================
DATA_DIR = "/home/mezziati/Documents/IAP/SPHINX20/data/"
FILENAME = "Leakers - LACES.csv"
REDSHIFT = 3.1

# =========================
# Load data
# =========================
file_path = Path(DATA_DIR) / FILENAME
df = pd.read_csv(file_path)
for col in ["V", "R", "i'", "z'"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print(df.dtypes)# =========================
# Effective wavelengths [Å]
# =========================
bands = {
    "V": 5500,
    "R": 6500,
    "i'": 7700,
    "z'": 9000
}

# =========================
# Helper function
# =========================
def compute_beta(z, w_obs, mag):
    c = 2.998e18  # Å/s

    lambda_rest = w_obs / (1 + z)

    f_nu = 10**(-(mag + 48.6) / 2.5)
    f_lambda = f_nu * c / w_obs**2

    x = np.log10(lambda_rest)
    y = np.log10(f_lambda)

    beta, intercept = np.polyfit(x, y, 1)

    return beta

beta_list = []

for idx, row in df.iterrows():

    mags, waves = [], []

    for b, w in bands.items():
        mag_col = b

        if mag_col not in df.columns:
            continue

        mag = row[mag_col]
        print(mag)
        if isinstance(mag, str):
            continue
        if not np.isfinite(mag):
            continue

        mags.append(mag)
        waves.append(w)

    if len(mags) >= 3:
        beta = compute_beta(
            REDSHIFT,
            np.array(waves),
            np.array(mags)
        )
    else:
        beta = np.nan

    beta_list.append(beta)

df["beta_UV"] = beta_list

out_file = Path(DATA_DIR) / "beta_results.csv"
df.to_csv(out_file, index=False)

print(f"Saved β results to: {out_file}")
