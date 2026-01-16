import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filepath='/home/mezziati/Documents/IAP/SPHINX20/data/previous_Lyc_measumenets.csv'
print("Loading SPHINX20 data...")
df = pd.read_csv(filepath)
print(f"Loaded {len(df)} galaxies")
columns=df.columns.tolist()
print(columns)