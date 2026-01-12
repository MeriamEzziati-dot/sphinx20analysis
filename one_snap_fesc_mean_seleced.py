import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

import argparse 

parser = argparse.ArgumentParser(description='Analyze fesc for a specific snapshot')
parser.add_argument('snapshot_index', type=int, help='Snapshot index (0-216)')
args = parser.parse_args()

snapshot_index = args.snapshot_index 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)




c = 2.99792458e10                     # cm/s
pc = 3.085677581e18                   # cm
prefactor = 4 * np.pi * (10 * pc)**2 

def Llambda_to_M1500(Llambda):
    """Convert Llambda at 1500Å (erg/s/Å) to absolute magnitude M1500"""
    # Step 1: Convert luminosity to flux at 10 pc
    flux_lambda = Llambda / prefactor  # erg/s/Å/cm²
    
    lambda_1500_cm = 1500e-8  
    flux_nu = flux_lambda * lambda_1500_cm**2 / c  # erg/s/Hz/cm²
    
    M_AB = -2.5 * np.log10(flux_nu) - 48.60
    return M_AB


file = h5py.File('haloprops_all.h5', 'r')

logging.info("Available snapshots:")
snapshots = list(file.keys())

snapshot = snapshots[snapshot_index]
logging.info(f"Using snapshot: {snapshot} ({snapshot_index} / 216 index)")

grp = file[snapshot]

fesc = grp['LyC_fesc_dir'][:]  # Shape: (N_halos, 10)
lum = grp['lum'][:]
lum1500 = grp['lum1500'][:]
minit = grp['minit'][:]
mstar = grp['mstar'][:]
mvir = grp['mvir'][:]
sfr1 = grp['sfr1'][:]
sfr10 = grp['sfr10'][:]
sfr100 = grp['sfr100'][:]
sfr3 = grp['sfr3'][:]
stellar_Zlw = grp['stellar_Zlw'][:]
stellar_Zmw = grp['stellar_Zmw'][:]
stellar_age = grp['stellar_age'][:]
file.close()



# Average fesc over the 10 directions
fesc_avg = np.mean(fesc, axis=1)  # Shape: (N_halos,)

logging.info(f"fesc averaged over 10 directions: shape = {fesc_avg.shape}")

# ============================================================================
# OBSERVATIONAL DATA
# ============================================================================

obs_data = {
    'log10_Mstar': [8.75, 9.20, 9.11, 8.76, 9.15, 9.54, 8.47, 8.35, 9.56, 8.51, 
                    10.26, 9.64, 8.83, 9.12, 8.70, 9.05, 8.89, 9.31, 8.50, 8.78,
                    9.17, 8.72, 10.76, 9.41, 8.96, 9.11, 8.79, 10.45, 9.62, 9.74,
                    9.80, 8.28, 8.52, 8.36, 9.00, 9.59, 10.04, 9.45, 8.89, 10.25,
                    9.73, 8.56, 8.21, 9.59, 8.9, 8.98, 9.8, 9.55, 8.38, 9.17, 8.68,
                    8.5, 8.7, 9.26, 8.2, 7.4, 9, 8.22, 7.51, 8.2, 7.79, 7.44, 8.8, 7.2],
    
    'fesc_LyC': [0.0291, 0.0132, 0.022, 0.0376, 0.0069, 0.0037, 0.5838, 0.0198, 
                 0.0075, 0.0089, 0.0198, 0.016, 0.0129, 0.007, 0.0025, 0.0045,
                 0.016, 0.0231, 0.022, 0.0094, 0.0189, 0.0163, 0.0248, 0.0044,
                 0.0473, 0.0205, 0.0144, 0.0549, 0.0421, 0.0658, 0.1607, 0.2659,
                 0.0917, 0.1777, 0.0405, 0.0376, 0.0198, 0.1053, 0.0309, 0.0127,
                 0.0055, 0.0421, 0.1197, 0.0333, 0.0003, 0.0005, 0.0009, 0.0011,
                 0.0473, 0.0258, 0.0903, 0.06, 0.028, 0.0333, 0.889, 0.6247, 
                 0.1767, 0.3052, 0.1281, 0.0917, 0.1121, 0.4333, 0.028, 0.06],
    
    'M_1500': [-18.527, -20.732, -20.268, -20.143, -20.439, -19.907, -18.461,
               -19.516, -20.928, -19.612, -19.806, -20.516, -19.999, -19.403, 
               -20.055, -20.059, -19.122, -19.943, -19.541, -20.337, -20.334,
               -19.735, -20.703, -20.340, -19.453, -20.137, -20.023, -20.590,
               -21.278, -21.033, -20.869, -18.580, -19.085, -18.286, -20.517,
               -21.508, -21.333, -20.102, -21.178, -20.341, -20.561, -19.485,
               -18.865, -21.048, -18.733, -19.021, -19.751, -18.405, -18.484,
               -19.610, -20.097, -18.654, -19.752, -19.752, -19.221, -18.857,
               -20.970, -21.194, -19.209, -19.566, -18.864, -18.530, -18.899, -18.529],
    
    'log10_SFR': [1.181, 1.314, 0.642, 0.934, 1.119, 0.927, 0.697, 0.555, 1.201,
                  0.816, 0.834, 1.038, 1.466, 0.765, 0.839, 0.671, 1.074, 1.607,
                  1.018, 0.853, 1.468, 1.143, 1.445, 1.351, 0.765, 1.037, 0.905,
                  1.397, 1.404, 1.327, 1.353, 0.547, 0.476, -0.173, 1.293, 1.438,
                  1.606, 1.330, 1.553, 1.239, 1.572, 0.833, 1.240, 1.220, 0.831,
                  0.655, 0.585, 0.894, 0.547, 1.257, 1.088, 1.356, 1.433, 0.676,
                  0.655, 0.476, 1.496, 1.167, 0.950, 0.818, 0.560, 0.160, 1.125, 0.676],
    
    'OH_12': [7.781, 8.029, 8.329, 7.798, 8.411, 8.169, 7.463, 7.938, 8.059,
              8.304, 8.182, 8.181, 8.398, 8.242, 8.266, 8.215, 8.163, 8.458, 8.251,
              8.315, 8.172, 8.045, 8.164, 8.397, 8.050, 8.164, 8.307, 8.214, 8.206,
              8.251, 8.060, 8.314, 8.382, 8.359, 8.354, 8.024, 8.037, 8.317, 8.205,
              8.396, 8.334, 8.266, 8.097, 8.272, 7.950, 7.672, 8.022, 7.848, 7.910,
              8.102, 7.940, 8.540, 8.550, 8.360, 7.670, 8.320, 8.020, 7.930, 7.755,
              8.045, 7.872, 8.315, 7.930, 8.090],
}

obs_mstar = 10**np.array(obs_data['log10_Mstar']) 
obs_fesc = np.array(obs_data['fesc_LyC'])
obs_M1500 = np.array(obs_data['M_1500'])
obs_sfr = 10**np.array(obs_data['log10_SFR'])
obs_oh12 = np.array(obs_data['OH_12'])
obs_Z = 10**(obs_oh12 - 12)  # Convert 12+log(O/H) to metallicity Z

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'LyC_fesc_dir vs Halo Properties - {snapshot} (with Observations)', 
             fontsize=16, fontweight='bold')




# Calculate M1500 for all halos (handle zeros)
M1500_all = np.full(len(lum1500), np.nan)  # Initialize with NaN
mask_nonzero = lum1500 > 0
M1500_all[mask_nonzero] = Llambda_to_M1500(lum1500[mask_nonzero])

# Create selection mask
mask_M1500 = (M1500_all > -22) & (M1500_all < -18)  # Assuming you want -22 < M1500 < -18?
mask_metallicity = (stellar_Zlw >= 2e-5) & (stellar_Zlw <= 4e-4)
mask_mass = (mstar >= 1e7)&(lum1500>0)
mask_sfr = (sfr10 >= 3e-1) &(lum1500>0) # Change to sfr1, sfr3, etc. if needed

# Combine all masks
selection_mask = mask_M1500 & mask_metallicity & mask_mass & mask_sfr 

logging.info(f"\nSelection criteria applied:")
logging.info(f"  M1500 > -22 and < -18: {np.sum(mask_M1500)} halos")
logging.info(f"  Metallicity 2e-5 to 4e-4: {np.sum(mask_metallicity)} halos")
logging.info(f"  Mass > 1e7: {np.sum(mask_mass)} halos")
logging.info(f"  SFR > 0.3: {np.sum(mask_sfr)} halos")
logging.info(f"  Total selected halos: {np.sum(selection_mask)} / {len(lum)}")


mstar_sel = mstar[mstar>=1e7]
mvir_sel = mvir[mvir>=1e7]
minit_sel = minit[minit>=1e7]
sfr1_sel = sfr1[sfr1 >= 3e-1]
sfr3_sel = sfr3[sfr3 >= 3e-1]
sfr10_sel = sfr10[sfr10 >= 3e-1]
sfr100_sel = sfr100[sfr100 >= 3e-1]
stellar_Zlw_sel = stellar_Zlw[ (stellar_Zlw >= 2e-5) & (stellar_Zlw <= 4e-4)]
stellar_Zmw_sel = stellar_Zmw[ (stellar_Zmw >= 2e-5) & (stellar_Zmw <= 4e-4)]

m1500_sel=M1500_all[mask_M1500]
# Subplot 1: Luminosity
ax1 = axes[0, 0]


ax1.scatter(m1500_sel, fesc_avg[mask_M1500], 
            alpha=0.3, s=20, label='M1450 SPHINX (sim)', color='cyan')

ax1.scatter(obs_M1500, obs_fesc, alpha=0.8, s=100, label='lum1500 (obs)', 
           color='red', marker='*', edgecolors='black', linewidths=1)
ax1.set_xlabel('M1500')
ax1.set_ylabel('LyC_fesc_dir (averaged)')
ax1.set_title('Magnitudes')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Masses
ax2 = axes[0, 1]
ax2.scatter(mstar_sel, fesc_avg[mstar>=1e7], alpha=0.3, s=20, label='mstar (sim)', color='red')
ax2.scatter(mvir_sel, fesc_avg[mvir>=1e7], alpha=0.3, s=20, label='mvir (sim)', color='orange')
ax2.scatter(minit_sel, fesc_avg[minit>=1e7], alpha=0.3, s=20, label='minit (sim)', color='pink')

ax2.scatter(obs_mstar, obs_fesc, alpha=0.8, s=100, label='mstar (obs)', 
           color='darkred', marker='*', edgecolors='black', linewidths=1)
ax2.set_xlabel('Mass')
ax2.set_ylabel('LyC_fesc_dir (averaged)')
ax2.set_xscale('log')
ax2.set_title('Masses')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Metallicities 
ax3 = axes[1, 0]
ax3.scatter(stellar_Zlw_sel, fesc_avg[ (stellar_Zlw >= 2e-5) & (stellar_Zlw <= 4e-4)], alpha=0.3, s=20, 
           label='stellar_Zlw (sim)', color='green')
ax3.scatter(stellar_Zmw_sel, fesc_avg[ (stellar_Zmw >= 2e-5) & (stellar_Zmw <= 4e-4)], alpha=0.3, s=20, 
           label='stellar_Zmw (sim)', color='lime')

ax3.scatter(obs_Z, obs_fesc, alpha=0.8, s=100, label='Z (obs)', 
           color='darkgreen', marker='*', edgecolors='black', linewidths=1)
ax3.set_xlabel('Metallicity (Z)')
ax3.set_ylabel('LyC_fesc_dir (averaged)')
ax3.set_xscale('log')
ax3.set_title('Metallicities')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Star Formation Rates
ax4 = axes[1, 1]
ax4.scatter(sfr1_sel, fesc_avg[sfr1 >= 3e-1], alpha=0.3, s=20, label='sfr1 (sim)', color='purple')
ax4.scatter(sfr3_sel, fesc_avg[sfr3 >= 3e-1], alpha=0.3, s=20, label='sfr3 (sim)', color='violet')
ax4.scatter(sfr10_sel, fesc_avg[sfr10 >= 3e-1], alpha=0.3, s=20, label='sfr10 (sim)', color='magenta')
ax4.scatter(sfr100_sel, fesc_avg[sfr100 >= 3e-1], alpha=0.3, s=20, label='sfr100 (sim)', color='orchid')

ax4.scatter(obs_sfr, obs_fesc, alpha=0.8, s=100, label='SFR (obs)', 
           color='darkviolet', marker='*', edgecolors='black', linewidths=1)
ax4.set_xlabel('Star Formation Rate [M☉/yr]')
ax4.set_ylabel('LyC_fesc_dir (averaged)')
ax4.set_xscale('log')
ax4.set_title('Star Formation Rates')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'SELECTED_AVG_fesc_analysis_with_obs_{snapshot}.png', dpi=300, bbox_inches='tight')
logging.info(f"\nPlot saved as: SELECTED_AVG_fesc_analysis_with_obs_{snapshot}.png")

logging.info(f"\nSimulation Summary Statistics for {snapshot}:")
logging.info(f"  Number of halos: {len(lum)}")
logging.info(f"  LyC_fesc_dir (averaged) range: [{np.min(fesc_avg):.4f}, {np.max(fesc_avg):.4f}]")
logging.info(f"  LyC_fesc_dir (averaged) mean: {np.mean(fesc_avg):.4f}")
logging.info(f"  LyC_fesc_dir (averaged) median: {np.median(fesc_avg):.4f}")

logging.info(f"\nObservational Data Summary:")
logging.info(f"  Number of observations: {len(obs_fesc)}")
logging.info(f"  fesc_LyC range: [{np.min(obs_fesc):.4f}, {np.max(obs_fesc):.4f}]")
logging.info(f"  fesc_LyC mean: {np.mean(obs_fesc):.4f}")
logging.info(f"  fesc_LyC median: {np.median(obs_fesc):.4f}")
