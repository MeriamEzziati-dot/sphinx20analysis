#!/usr/bin/env python3
"""
Example Usage: LyC Data Analysis
=================================
Demonstrates how to use LyCDataManager and LyCDiagnostics classes
for analyzing SPHINX20 simulations vs LzLCS observations.

Author: Meriam Ezziati
Date: January 2026
"""

import sys

from sphinx_analysis.catalogues.CatalogueManager import CatalogueManager
from sphinx_analysis.analyser.LyCDiagnostics import LyCDiagnostics

sys.path.insert(0, '/home/mezziati/')  # Adjust this path as needed



# Define paths
sim_catalogue = '/home/mezziati/Documents/IAP/SPHINX20/data/all_basic_data.csv'
obs_catalogue = '/home/mezziati/Documents/IAP/SPHINX20/data/flury.csv'
output_dir =  '/home/mezziati/Documents/IAP/SPHINX20/sphinx_analysis/outputs/'


# ============================================================
# SETUP
# ============================================================



# ============================================================
# INITIALIZE DATA MANAGER
# ============================================================

print("Initializing Data Manager...")
data = CatalogueManager(sim_catalogue, obs_catalogue)

# Print summary
data.summary()

# List all available parameters
data.list_available_parameters()

# ============================================================
# INITIALIZE DIAGNOSTICS
# ============================================================

print("Initializing Diagnostics...")
diag = LyCDiagnostics(data)

# ============================================================
# EXAMPLE 1: Basic 2D Comparison
# ============================================================
diag.plot_2d_all_directions(
    'age_star',           # Gets f_esc_dir_0 to f_esc_dir_9
    'beta',            # Gets M_UV for all directions
    color_param='f_esc',
    save_path=output_dir + 'age_beta_fesc_all_directions.png'
)
diag.plot_2d_all_directions(
    'log10(zeta_ISM)',           # Gets f_esc_dir_0 to f_esc_dir_9
    'beta',            # Gets M_UV for all directions
    color_param='f_esc',
    save_path=output_dir + 'zeta_beta_fesc_all_directions.png'
)
diag.plot_2d_all_directions(
    'log10(zeta_ISM)',
    'E(B-V)',
    color_param='f_esc',
    save_path=output_dir + 'zeta_E(B-V)_fesc_all_directions.png'
)

print("\n" + "="*70)
print("EXAMPLE 1: M_UV vs f_esc colored by redshift")
print("="*70)

diag.plot_2d_comparison(
    x_param='M_UV',
    y_param='f_esc',
    color_param='redshift',
    figsize=(10, 8),
    save_path=output_dir + 'muv_fesc_z.png'
)

# ============================================================
# EXAMPLE 2: UV Slope vs Attenuation
# ============================================================

print("\n" + "="*70)
print("EXAMPLE 2: Beta vs E(B-V)")
print("="*70)

diag.plot_2d_comparison(
    x_param='beta',
    y_param='E(B-V)',
    color_param='f_esc',
    figsize=(10, 8),
    save_path=output_dir + 'beta_ebv_fesc.png'
)

# ============================================================
# EXAMPLE 3: Stellar Mass - SFR Relation
# ============================================================

print("\n" + "="*70)
print("EXAMPLE 3: Stellar Mass vs SFR")
print("="*70)

diag.plot_2d_comparison(
    x_param='stellar_mass',
    y_param='SFR',
    color_param='f_esc',
    log_x=False,
    log_y=True,
    figsize=(10, 8),
    save_path=output_dir + 'mass_sfr.png'
)

# ============================================================
# EXAMPLE 4: Histogram Comparison
# ============================================================

print("\n" + "="*70)
print("EXAMPLE 4: Distribution of f_esc")
print("="*70)

diag.plot_histogram_comparison(
    param='f_esc',
    bins=50,
    figsize=(10, 6),
    save_path=output_dir + 'fesc_histogram.png'
)

# ============================================================
# EXAMPLE 5: BPT Diagram
# ============================================================

print("\n" + "="*70)
print("EXAMPLE 5: BPT Diagram")
print("="*70)

diag.plot_bpt_diagram(
    figsize=(10, 8),
    save_path=output_dir + 'bpt_diagram.png'
)

# ============================================================
# EXAMPLE 6: Directional Analysis
# ============================================================

print("\n" + "="*70)
print("EXAMPLE 6: Directional Analysis of f_esc")
print("="*70)

diag.plot_directional_comparison(
    x_param='M_UV',
    y_param='f_esc_dir',
    figsize=(14, 8),
    save_path=output_dir + 'directional_muv_fesc.png'
)

# ============================================================
# EXAMPLE 7: Multiparameter Grid
# ============================================================

print("\n" + "="*70)
print("EXAMPLE 7: Multiparameter Grid")
print("="*70)

params_to_compare = ['f_esc', 'M_UV', 'stellar_mass', 'beta', 'metallicity']

diag.plot_multiparameter_grid(
    params=params_to_compare,
    figsize=(15, 15),
    save_path=output_dir + 'multiparameter_grid.png'
)

# ============================================================
# EXAMPLE 8: Correlation Analysis
# ============================================================

print("\n" + "="*70)
print("EXAMPLE 8: Correlation Analysis")
print("="*70)

correlation_params = [
    'M_UV', 'beta', 'stellar_mass', 'SFR', 'sSFR',
    'metallicity', 'xi_ion', 'age_star', 'E(B-V)'
]

corr_results = diag.correlations_summary(
    param_list=correlation_params,
    reference_param='f_esc'
)

# Save correlation results
corr_results.to_csv(output_dir + 'fesc_correlations.csv', index=False)

# ============================================================
# EXAMPLE 9: Direct Data Access
# ============================================================

print("\n" + "="*70)
print("EXAMPLE 9: Direct Data Access for Custom Analysis")
print("="*70)

# Get specific parameters
muv_sim, muv_obs = data.get_parameter('M_UV', 'both')
fesc_sim, fesc_obs = data.get_parameter('f_esc', 'both')

# Get mask for valid data
mask_sim = data.get_valid_mask('M_UV', 'f_esc', dataset='sim')
mask_obs = data.get_valid_mask('M_UV', 'f_esc', dataset='obs')

print(f"\nValid simulation points: {mask_sim.sum()}")
print(f"Valid observation points: {mask_obs.sum()}")

# Do custom analysis
import numpy as np
print(f"\nSimulation f_esc statistics:")
print(f"  Mean: {fesc_sim[mask_sim].mean():.2f}%")
print(f"  Median: {fesc_sim[mask_sim].median():.2f}%")
print(f"  Std: {fesc_sim[mask_sim].std():.2f}%")

# ============================================================
# EXAMPLE 10: Directional Parameter with Specific Direction
# ============================================================

print("\n" + "="*70)
print("EXAMPLE 10: Specific Direction Analysis")
print("="*70)

# Compare f_esc in direction 0 vs direction 5
diag.plot_2d_comparison(
    x_param='M_UV',
    y_param='f_esc_dir',
    direction=0,
    figsize=(10, 8),
    save_path=output_dir + 'muv_fesc_dir0.png'
)

diag.plot_2d_comparison(
    x_param='M_UV',
    y_param='f_esc_dir',
    direction=5,
    figsize=(10, 8),
    save_path=output_dir + 'muv_fesc_dir5.png'
)

# ============================================================
# EXAMPLE 11: Emission Line Diagnostics
# ============================================================

print("\n" + "="*70)
print("EXAMPLE 11: O32 vs Metallicity")
print("="*70)

diag.plot_2d_comparison(
    x_param='metallicity',
    y_param='O32',
    color_param='f_esc',
    figsize=(10, 8),
    save_path=output_dir + 'o32_metallicity.png'
)


diag.plot_2d_comparison(
    x_param='log10(zeta_ISM)',
    y_param='beta',
    color_param='f_esc',
    figsize=(10, 8),
    save_path=output_dir + 'zeta_ism_beta_f_esc.png'
)


# ============================================================
# EXAMPLE 12: Multiple Histogram Comparisons
# ============================================================

print("\n" + "="*70)
print("EXAMPLE 12: Multiple Distributions")
print("="*70)

for param in ['f_esc', 'beta', 'metallicity', 'xi_ion']:
    diag.plot_histogram_comparison(
        param=param,
        bins=30,
        save_path=output_dir + f'{param}_hist.png'
    )

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)