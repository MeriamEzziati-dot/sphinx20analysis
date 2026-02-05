sim_catalogue_dir = '/home/mezziati/Documents/IAP/SPHINX20/data/all_basic_data.csv'
obs_catalogue_dir = '/home/mezziati/Documents/IAP/SPHINX20/data/flury.csv'
laces_catalogue_dir = '/home/mezziati/Documents/IAP/SPHINX20/data/LeakersLACES.csv'

output_dir =  '/home/mezziati/Documents/IAP/SPHINX20/sphinx_analysis/outputs/'

import pandas as pd
sim_catalogue = pd.read_csv(sim_catalogue_dir)
obs_catalogue = pd.read_csv(obs_catalogue_dir)
laces_catalogue = pd.read_csv(laces_catalogue_dir)
print(sim_catalogue.columns)
laces_catalogue['E(B - V)'] = pd.to_numeric(laces_catalogue['E(B - V)'], errors="coerce")
print(laces_catalogue.dtypes)# =========================

ebv_dir_cols = [col for col in sim_catalogue.columns if col.startswith('ebmv_dir_')]
if ebv_dir_cols:
    sim_catalogue['ebmv_dir_mean'] = sim_catalogue[ebv_dir_cols].mean(axis=1)
    sim_catalogue['ebmv_dir_std'] = sim_catalogue[ebv_dir_cols].std(axis=1)

# Beta (UV slope) directional
beta_dir_cols = [col for col in sim_catalogue.columns if col.startswith('beta_dir_') and col.endswith('_sn')]
if beta_dir_cols:
    sim_catalogue['beta_dir_mean'] = sim_catalogue[beta_dir_cols].mean(axis=1)
    sim_catalogue['beta_dir_std'] = sim_catalogue[beta_dir_cols].std(axis=1)
    
    
beta_sim=sim_catalogue['beta_dir_mean']
beta_obs=obs_catalogue['UV-beta']
beta_laces=laces_catalogue['beta_UV']
ebmv_sim=sim_catalogue['ebmv_dir_mean']
ebmv_obs=obs_catalogue['E(B-V)_uv']
embv_laces=laces_catalogue['E(B - V)']
mask_laces = ~(pd.isna(beta_laces) | pd.isna(embv_laces))
beta_laces_clean = beta_laces[mask_laces]
ebmv_laces_clean = embv_laces[mask_laces]


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


def M1500_to_L(M1500, wavelength=1500, delta_lambda=None):
    """
    Convert absolute UV magnitude M1500 to luminosity.

    Parameters:
    -----------
    M1500 : float
        Absolute AB magnitude at 1500 Å.
    wavelength : float
        Wavelength in Angstroms (default 1500 Å)
    delta_lambda : float or None
        Bandwidth in Angstroms for total luminosity. If None, returns monochromatic luminosity.

    Returns:
    --------
    L_nu : float
        Monochromatic luminosity in erg/s/Hz
    L_lambda : float
        Monochromatic luminosity in erg/s/Å
    L_total : float or None
        Total luminosity in erg/s over delta_lambda, if delta_lambda is given
    """
    import numpy as np

    # Speed of light in Å/s
    c = 2.998e18  # Å/s

    # Monochromatic luminosity per Hz
    L_nu = 10 ** (0.4 * (51.60 - M1500))

    # Convert to per Å
    L_lambda = L_nu * c / wavelength ** 2

    # Total luminosity over delta_lambda
    L_total = L_lambda * delta_lambda if delta_lambda is not None else None

    return  L_total


# Example usage:
M1500 = sim_catalogue['MAB_1500_int']
delta_lambda = 1  # e.g., 100 Å filter width
L_total = M1500_to_L(M1500, delta_lambda=delta_lambda)

sim_catalogue['log_halpha_l1500'] = np.log10(sim_catalogue['HI_6562.8_int'] / L_total)
fesc_sim=sim_catalogue['f_esc']
print(sim_catalogue['log_halpha_l1500'])

# Create cut at 1.6
sim_catalogue['above_threshold'] = sim_catalogue['log_halpha_l1500'] > 1.6

# Extract data
beta_sim = sim_catalogue['beta_dir_mean']
ebmv_sim = sim_catalogue['ebmv_dir_mean']
beta_obs = obs_catalogue['UV-beta']
ebmv_obs = obs_catalogue['E(B-V)_uv']
beta_laces = laces_catalogue['beta_UV']
ebmv_laces = laces_catalogue['E(B - V)']



mask_sim_above = ~(pd.isna(beta_sim) | pd.isna(ebmv_sim)) & sim_catalogue['above_threshold']
mask_sim_below = ~(pd.isna(beta_sim) | pd.isna(ebmv_sim)) & ~sim_catalogue['above_threshold']

beta_sim_above = beta_sim[mask_sim_above]
ebmv_sim_above = ebmv_sim[mask_sim_above]
beta_sim_below = beta_sim[mask_sim_below]
ebmv_sim_below = ebmv_sim[mask_sim_below]

# LzLCS
mask_obs = ~(pd.isna(beta_obs) | pd.isna(ebmv_obs))
beta_obs_clean = beta_obs[mask_obs]
ebmv_obs_clean = ebmv_obs[mask_obs]

# LACES
mask_laces = ~(pd.isna(beta_laces) | pd.isna(ebmv_laces))
beta_laces_clean = beta_laces[mask_laces]
ebmv_laces_clean = ebmv_laces[mask_laces]

# Print info
print(f"SPHINX20 (log(Hα/L1500) > 1.6): {len(beta_sim_above)} points")
print(f"SPHINX20 (log(Hα/L1500) ≤ 1.6): {len(beta_sim_below)} points")
print(f"LzLCS: {len(beta_obs_clean)} valid points")
print(f"LACES: {len(beta_laces_clean)} valid points")


# Create the figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the three populations




scatter=ax.scatter(beta_sim, np.log10(ebmv_sim), alpha=0.6, s=50,
           label='SPHINX20', c=(10**fesc_sim)*100,cmap='viridis'
           , linewidth=0.5)

ax.scatter(beta_sim[mask_sim_above], np.log10(ebmv_sim[mask_sim_above]),marker='x',color='white',s=200)
ax.scatter(beta_obs, np.log10(ebmv_obs), alpha=0.7, s=150, label='LzLCS', color='red', marker='s', edgecolors='black', linewidth=0.5)
ax.scatter(beta_laces_clean, np.log10(ebmv_laces_clean), alpha=0.7, s=200, label='LACES', color='green', marker='^', edgecolors='black', linewidth=0.5)

# Plot the relation line: E(B-V) = -1.1*beta - 3.3
beta_line = np.linspace(-3, 0, 100)
ebmv_line = 10**(-1.1 * beta_line - 3.3)
ax.plot(beta_line, np.log10(ebmv_line), 'k--', linewidth=2, label=r'E(B-V) = -1.1$\beta$ - 3.3', zorder=5)

# Labels and formatting
ax.set_xlabel(r'UV Slope ($\beta$)', fontsize=14)
ax.set_ylabel(r'E(B-V)', fontsize=14)
ax.set_title(r'Choustikov+24 criteria', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('f_esc', fontsize=12)
plt.tight_layout()

# Save the figure
output_path = output_dir + 'beta_vs_ebmv_comparison_after_withlaphal1500.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

plt.show()

