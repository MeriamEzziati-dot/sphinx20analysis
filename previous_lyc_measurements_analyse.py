import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fletcher_path='/home/mezziati/Documents/IAP/SPHINX20/data/fletcher.csv'
print("Loading SPHINX20 data...")
fletcher_data = pd.read_csv(fletcher_path)
print(f"Loaded {len(fletcher_data)} galaxies")
columns=fletcher_data.columns.tolist()
print(columns[:10])

fesc=fletcher_data['f_esc']
print(fesc)

