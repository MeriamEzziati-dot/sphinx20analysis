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

# Calculate M1500 for all halos (handle zeros)
M1500_all = np.full(len(lum1500), np.nan)  # Initialize with NaN
mask_nonzero = lum1500 > 0
M1500_all[mask_nonzero] = Llambda_to_M1500(lum1500[mask_nonzero])

# Average fesc over the 10 directions
fesc_avg = np.mean(fesc, axis=1)  # Shape: (N_halos,)

logging.info(f"fesc averaged over 10 directions: shape = {fesc_avg.shape}")

mask=fesc_avg==1.0


logging.info(f"Number of haloes with fesc=1.0: shape = {fesc_avg[mask].shape}")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))




# Subplot 1: Luminosity
ax1 = axes[0, 0]
ax1.hist(M1500_all[mask],bins=10)
ax1.set_xlabel('M1500')


# Subplot 2: Masses
ax2 = axes[0, 1]
ax2.set_xlabel('Masses')
ax2.set_xscale('log')
ax2.hist(mstar[mask],bins=10)
# Subplot 3: Metallicities 
ax3 = axes[1, 0]

ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.hist(stellar_Zlw[mask],bins=10)
ax3.set_xlabel('Metallicity')
# Subplot 4: Star Formation Rates
ax4 = axes[1, 1]
ax4.set_xlabel('SFR')
ax4.set_xscale('log')
ax4.hist(sfr10[mask],bins=10,label='10Myr')
ax4.hist(sfr1[mask],bins=10,label='1Myr')
ax4.hist(sfr3[mask],bins=10,label='3Myr')
ax4.hist(sfr100[mask],bins=10,label='100Myr')
ax4.legend()
plt.tight_layout()
plt.savefig(f'SELECTED_AVG_fesc_equa1_{snapshot}.png', dpi=300, bbox_inches='tight')
logging.info(f"\nPlot saved ")

