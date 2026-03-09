#!/usr/bin/env python3
"""
SPHINX20 LyC Escape Fraction Analysis Script
=============================================
Focuses on analyzing Lyman Continuum (LyC) escape fractions and their
correlations with galaxy properties in the SPHINX20 data release.

Author: Meriam Ezziati 05 dec 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
data_path='/home/mezziati/Documents/IAP/SPHINX20/data/'
home_path='/home/mezziati/Documents/IAP/SPHINX20/sphinx_analysis/outputs/'
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MERMosaicError(Exception):
    pass


class SPHINXLyCAnalyzer:
    """Analyzer for SPHINX20 LyC escape fraction data."""

    def __init__(self, filepath,observations):
        """Load SPHINX20 data."""
        print("Loading SPHINX20 data...")
        self.df = pd.read_csv(filepath)
        self.observations = pd.read_csv(observations)
        # self.control = pd.read_csv(control)

        print(f"Loaded {len(self.df)} galaxies")


        # Convert log values to linear
        self.df['f_esc_linear'] = 10 ** self.df['f_esc']
        self.df['stellar_mass_linear'] = self.df['stellar_mass']
        self.df['ionizing_luminosity_linear'] = 10 ** self.df['ionizing_luminosity']
        # Same for the control sample
        # self.control['A_SFR_l'] = 10 ** self.control['A_SFR']
        # self.control['A_SFR_error_l'] = 10 ** self.control['A_SFR_error']


        # Calculate mean directional escape fraction (total of dir=10)
        fesc_dir_cols = [col for col in self.df.columns if col.startswith('fesc_dir_')]
        if fesc_dir_cols:
            self.df['fesc_dir_mean'] = self.df[fesc_dir_cols].mean(axis=1)
            self.df['fesc_dir_std'] = self.df[fesc_dir_cols].std(axis=1)

    def save_column_names(self, output_path=home_path+'column_names.txt'):
        """Extract and save column names from both simulation and observation dataframes."""

        # Extract column names
        simu_columns = self.df.columns.tolist()
        obs_columns = self.observations.columns.tolist()

        # Save to text file
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SIMULATION FILE COLUMNS\n")
            f.write("=" * 60 + "\n\n")
            for i, col in enumerate(simu_columns, 1):
                f.write(f"{i}. {col}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("OBSERVATIONS FILE COLUMNS\n")
            f.write("=" * 60 + "\n\n")
            for i, col in enumerate(obs_columns, 1):
                f.write(f"{i}. {col}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Total simulation columns: {len(simu_columns)}\n")
            f.write(f"Total observation columns: {len(obs_columns)}\n")
            f.write("=" * 60 + "\n")

        print(f"Column names saved to {output_path}")
        return output_path



    def plot_multiparameter_comparison(self, figsize=(16, 10), save_path=None):
        """
        Create a comprehensive multi-panel comparison plot.
        Shows: mass-fesc, SFR-fesc, metallicity-fesc, and xi_ion-fesc
        """
        fletcher_path = '/home/mezziati/Documents/IAP/SPHINX20/data/fletcher.csv'
        print("Loading SPHINX20 data...")
        fletcher_data = pd.read_csv(fletcher_path)
        print(f"Loaded {len(fletcher_data)} galaxies")
        columns = fletcher_data.columns.tolist()
        print(columns[:10])

        fesc = fletcher_data['f_esc']
        print(fesc)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        mask = 10 ** (self.df['f_esc']) > 0.1
        mask01 = self.observations['f_esc(LyC)-Hbeta']>0.1
        # 1. Mass vs f_esc
        ax1 = fig.add_subplot(gs[0, 0])
        scatter1 = ax1.scatter(
            self.df['stellar_mass'][mask],
            self.df['f_esc'][mask],
            alpha=0.4, s=30, label='SPHINX20',
            c=self.df['redshift'][mask],  # Don't forget to mask the color too!
            cmap='cividis'
        )
        mask_obs = (self.observations['f_esc(LyC)-Hbeta'].notna() & mask01 &
                    self.observations['log10(Mstar)'].notna())
        mask_fletcher = (fletcher_data['f_esc'].notna() &
                    fletcher_data['M_star'].notna())


        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])
        obs_mass = self.observations.loc[mask_obs, 'log10(Mstar)']

        fletcher_fesc = np.log10(fletcher_data.loc[mask_fletcher, 'f_esc'])
        fletcher_mass = fletcher_data.loc[mask_fletcher, 'M_star']
        ax1.scatter(obs_mass, obs_fesc,
                    alpha=0.7, s=100, label='LzLCS',
                    color='red', marker='s', edgecolors='black', linewidth=0.5)
        ax1.scatter((fletcher_mass), fletcher_fesc,
                    alpha=0.7, s=200, label='LACES',
                    color='green', marker='*', edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('log₁₀(M* / $\mathrm{M_{\odot}}$)', fontsize=11)
        ax1.set_ylabel('log₁₀($\mathrm{f_{esc}}$)', fontsize=11)
        ax1.set_title('Stellar Mass vs $\mathrm{f_{esc}}$', fontsize=12, fontweight='bold')
        #ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. SFR vs f_esc
        ax2 = fig.add_subplot(gs[0, 1])
        scatter2=ax2.scatter(self.df['sfr_100'][mask], self.df['f_esc'][mask],
                    alpha=0.4, s=30, label='SPHINX20', c=self.df['redshift'][mask],cmap='cividis')

        # Observation masks
        mask_fletcher = (fletcher_data['f_esc'].notna() &
                    fletcher_data['A_SFR'].notna())

        mask_obs = (self.observations['f_esc(LyC)-Hbeta'].notna() & mask01 &
                    self.observations['log10(SFR)-UV'].notna())
        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])
        obs_sfr = self.observations.loc[mask_obs, 'log10(SFR)-UV']

        #TODO: include fletcher data  in the class contructor as an **args

        fletcher_fesc = np.log10(fletcher_data.loc[mask_fletcher, 'f_esc'])
        fletcher_sfr = fletcher_data.loc[mask_fletcher, 'A_SFR']

        ax2.scatter(np.log10(fletcher_sfr), fletcher_fesc,
                    alpha=0.7, s=200, label='LACES',
                    color='green', marker='*', edgecolors='black', linewidth=0.5)

        ax2.scatter(obs_sfr, obs_fesc,
                    alpha=0.7, s=100, label='LzLCS',
                    color='red', marker='s', edgecolors='black', linewidth=0.5)

        ax2.set_xlabel('log10(SFR)($\mathrm{M_{\odot}}$ /yr)', fontsize=11)
        ax2.set_ylabel('log10($\mathrm{f_{esc}}$)', fontsize=11)
        ax2.set_title('Star Formation Rate vs $\mathrm{f_{esc}}$, xrange[0,5]', fontsize=12, fontweight='bold')
        # ax2.legend()
        ax2.set_xlim(0,5)
        ax2.grid(True, alpha=0.3)

        # 3. Metallicity vs f_esc - CORRECTED WITH CONVERSION
        ax3 = fig.add_subplot(gs[1, 0])
        # Convert simulation metallicity from log(Z/Z☉) to 12+log(O/H)
        solar_oh = 8.69

        # OBS
        O2_obs = self.observations['O2_3726A']+ self.observations['O2_3729A']
        O3_obs = self.observations['O3_4959A'] + self.observations['O3_5007A']
        O32_obs=O3_obs/O2_obs

        #SIM

        O2_sim = self.df['O__2_3726.03A_int'] +self.df['O__2_3728.81A_int']
        O3_sim = self.df['O__3_4958.91A_int']+ self.df['O__3_5006.84A_int']
        O32 = O3_sim/O2_sim



        scatter3=ax3.scatter(O32[mask], self.df['f_esc'][mask],
                    alpha=0.4, s=30, label='SPHINX20', c=self.df['redshift'][mask],cmap='cividis')

        mask_obs = (self.observations['f_esc(LyC)-Hbeta'].notna() & mask01 &
                    O32_obs.notna())

        mask_fletcher = (fletcher_data['f_esc'].notna() &
                    O32_obs.notna())




        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])
        obs_metal = O32_obs[mask_obs]  # Already in 12+log(O/H)

        fletcher_fesc = np.log10(fletcher_data.loc[mask_fletcher, 'f_esc'])
        fletcher_metal = fletcher_data.loc[mask_fletcher, 'O3/02']  # Already in 12+log(O/H)


        ax3.scatter(obs_metal, obs_fesc,
                    alpha=0.7, s=100, label='LzLCS',
                    color='red', marker='s', edgecolors='black', linewidth=0.5)
        ax3.scatter(fletcher_metal, fletcher_fesc,
                    alpha=0.7, s=200, label='LACES',
                    color='green', marker='*', edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('OIII/OII', fontsize=11)
        ax3.set_xlim(0,40)

        ax3.set_ylabel('log₁₀($\mathrm{f_{esc}}$)', fontsize=11)
        ax3.set_title('OIII/OII vs $\mathrm{f_{esc}}$, xrange [0,40]', fontsize=12, fontweight='bold')
        # ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. xi_ion vs f_esc
        ax4 = fig.add_subplot(gs[1, 1])
        scatter4=ax4.scatter(self.df['xi_ion'][mask], self.df['f_esc'][mask],
                    alpha=0.4, s=30, label='SPHINX20', c=self.df['redshift'][mask],cmap='cividis')
        mask_obs = (self.observations['f_esc(LyC)-Hbeta'].notna() & mask01 &
                    self.observations['xi-ion'].notna())

        mask_fletcher = (fletcher_data['f_esc'].notna()  & fletcher_data['log10(Xi-ion)'].notna())
        fletcher_fesc = np.log10(fletcher_data.loc[mask_fletcher, 'f_esc'])
        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])

        obs_xi = self.observations.loc[mask_obs, 'xi-ion']
        fletcher_xi = np.log10(10**(fletcher_data.loc[mask_fletcher, 'log10(Xi-ion)'])/(1-fletcher_fesc))

        ax4.scatter(obs_xi, obs_fesc,
                    alpha=0.7, s=100, label='LzLCS',
                    color='red', marker='s', edgecolors='black', linewidth=0.5)
        ax4.scatter(fletcher_xi, fletcher_fesc,
                    alpha=0.7, s=200, label='LACES',
                    color='green', marker='*', edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('log₁₀($ξ_{\mathrm{ion}}$ / Hz erg)', fontsize=11)
        ax4.set_ylabel('log₁₀($\mathrm{f_{esc}}$)', fontsize=11)
        ax4.set_title('$ξ_{\mathrm{ion}}$ vs $\mathrm{f_{esc}}$', fontsize=12, fontweight='bold')
        # ax4.legend()
        ax4.grid(True, alpha=0.3)


        cbar1 = fig.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Redshift', fontsize=9)

        cbar2 = fig.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Redshift', fontsize=9)

        cbar3 = fig.colorbar(scatter3, ax=ax3)
        cbar3.set_label('Redshift', fontsize=9)

        cbar4 = fig.colorbar(scatter4, ax=ax4)
        cbar4.set_label('Redshift', fontsize=9)

        plt.suptitle(r'$f_{\mathrm{esc}}>10\%$' + ' Multi-Parameter Comparison: Simulations vs Observations',
                     fontsize=15, fontweight='bold', y=0.995)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='SPHINX20',
                   markerfacecolor='gray', markersize=8, alpha=0.6),
            Line2D([0], [0], marker='s', color='w', label='LzLCS',
                   markerfacecolor='red', markersize=10, markeredgecolor='black', markeredgewidth=0.5),
            Line2D([0], [0], marker='*', color='w', label='LACES',
                   markerfacecolor='green', markersize=15, markeredgecolor='black', markeredgewidth=0.5)
        ]

        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=3, fontsize=12, frameon=True,
                   bbox_to_anchor=(0.5, 0.02))
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# INDIRECT f_esc INDICATOR PLOTS
# ============================================================================

    def plot_ebv_beta_relation(self, figsize=(12, 8), save_path=None):
        """
        Create E(B-V) vs Beta slope plot with analytical relation.
        Includes SPHINX20 simulations and LzLCS observations.
        Only observations color-coded by f_esc.
        Includes side histogram for f_esc > 5% simulations.
        Prints detailed statistics for both datasets.
        """
        # Create figure with gridspec for main plot and histogram
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharey=ax_main)

        print("\n" + "=" * 60)
        print("STATISTICS FOR SIMULATIONS (SPHINX20)")
        print("=" * 60)

        # Total galaxies
        print(f"Total galaxies: {len(self.df)}")

        # Base mask (all columns present)
        mask_sim_base = (self.df['f_esc'].notna() &
                         self.df['H__1_6562.80A_int'].notna() &  # Halpha
                         self.df['cont_1500_int'].notna() &  # L1500
                         self.df['beta_int_s'].notna() &  # beta slope
                         self.df['redshift'].notna() &
                         self.df['ebmv_dir_0'].notna() &
                         self.df['ebmv_dir_1'].notna() &
                         self.df['ebmv_dir_2'].notna() &
                         self.df['ebmv_dir_3'].notna() &
                         self.df['ebmv_dir_4'].notna() &
                         self.df['ebmv_dir_5'].notna() &
                         self.df['ebmv_dir_6'].notna() &
                         self.df['ebmv_dir_7'].notna() &
                         self.df['ebmv_dir_8'].notna() &
                         self.df['ebmv_dir_9'].notna())

        print(f"Galaxies with all required data: {mask_sim_base.sum()}")

        # Calculate mean E(B-V) across all 10 directions
        ebv_columns = [f'ebmv_dir_{i}' for i in range(10)]
        self.df['ebmv_mean'] = self.df[ebv_columns].mean(axis=1)

        # Calculate Halpha/L1500 ratio
        self.df['halpha_l1500_ratio'] = np.log10(self.df['H__1_6562.80A_int'] / self.df['cont_1500_int'])

        # Individual cuts
        mask_fesc = mask_sim_base & (10 ** self.df['f_esc'] > 0.05)
        mask_ratio = mask_sim_base & (self.df['halpha_l1500_ratio'] < 1.6)

        print(f"  → After f_esc > 5% cut: {mask_fesc.sum()}")
        print(f"  → After Hα/L1500 < 1.6 cut: {mask_ratio.sum()}")

        # Combined cuts
        mask_sim = mask_sim_base & (10 ** self.df['f_esc'] > 0.05) & (self.df['halpha_l1500_ratio'] < 1.6)
        print(f"  → After BOTH cuts: {mask_sim.sum()}")

        print("\n" + "=" * 60)
        print("STATISTICS FOR OBSERVATIONS (LzLCS)")
        print("=" * 60)

        # Total observations
        print(f"Total galaxies: {len(self.observations)}")

        # Convert M_1500 to L_1500 for observations
        # L_1500 / L_sun = 10^(-0.4 * (M_1500 - M_sun_1500))
        # Using M_sun_1500 ≈ 15.5
        M_sun_1500 = 15.5
        self.observations['L_1500'] = 10 ** (-0.4 * (self.observations['M_1500'] - M_sun_1500))

        # Calculate Halpha/L1500 ratio for observations
        # Need to convert Halpha flux to luminosity first
        # L(Halpha) is already in the observations as L(H1r_6563A)
        # But we need flux ratio, so use flux directly
        # Actually, for consistent comparison, let's use the flux ratio
        self.observations['halpha_l1500_ratio'] = np.log10(
            self.observations['H1r_6563A'] / self.observations['L_1500'])

        # Base mask (all columns present)
        mask_obs_base = (self.observations['UV-beta'].notna() &
                         self.observations['E(B-V)_uv'].notna() &
                         self.observations['H1r_6563A'].notna() &
                         self.observations['f_esc(LyC)-UVfit'].notna() &
                         self.observations['M_1500'].notna())

        print(f"Galaxies with all required data: {mask_obs_base.sum()}")

        # Individual cuts
        mask_obs_fesc = mask_obs_base & (self.observations['f_esc(LyC)-UVfit'] > 0.05)
        mask_obs_ratio = mask_obs_base & (self.observations['halpha_l1500_ratio'] < 1.6)

        print(f"  → After f_esc > 5% cut: {mask_obs_fesc.sum()}")
        print(f"  → After Hα/L1500 < 1.6 cut: {mask_obs_ratio.sum()}")

        # Combined cuts
        mask_obs = mask_obs_base & (self.observations['f_esc(LyC)-UVfit'] > 0.05) & (
                    self.observations['halpha_l1500_ratio'] < 1.6)
        print(f"  → After BOTH cuts: {mask_obs.sum()}")
        print("=" * 60 + "\n")

        # Plot simulations (SPHINX20) - color-coded by f_esc
        scatter = ax_main.scatter(
            self.df.loc[mask_sim, 'beta_int_s'],
            np.log10(self.df.loc[mask_sim, 'ebmv_mean']),
            c=10 ** self.df.loc[mask_sim, 'f_esc'] * 100,
            cmap='cividis',
            alpha=0.4,
            s=30,
            label='SPHINX20',
            edgecolors='none'
        )

        # Plot observations (LzLCS) - red squares
        if mask_obs.any():
            scatter_obs = ax_main.scatter(
                self.observations.loc[mask_obs, 'UV-beta'],
                np.log10(self.observations.loc[mask_obs, 'E(B-V)_uv']),
                alpha=0.7,
                s=100,
                label='LzLCS',
                color='red',
                marker='s',
                edgecolors='black',
                linewidth=0.5,
                zorder=5
            )

        # Add colorbar for f_esc (simulations)
        cbar = plt.colorbar(scatter, ax=ax_main)
        cbar.set_label('$f_{esc}$ (%)', fontsize=12)

        # Plot analytical relation: log10(E(B-V)) = -1.1 * beta - 3.3
        beta_range = np.linspace(-3, 0.5, 100)
        log_ebv_analytical = -1.1 * beta_range - 3.3
        ax_main.plot(beta_range, log_ebv_analytical,
                     'k--', linewidth=2, label='log₁₀(E(B-V)) = -1.1β - 3.3',
                     zorder=10)

        # Labels and formatting for main plot
        ax_main.set_xlabel('UV β slope', fontsize=13)
        ax_main.set_ylabel('log₁₀(E(B-V))', fontsize=13)
        ax_main.set_title('UV Beta Slope vs E(B-V)',
                          fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3)

        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='SPHINX20',
                   markerfacecolor='gray', markersize=8, alpha=0.6),
            Line2D([0], [0], marker='s', color='w', label='LzLCS',
                   markerfacecolor='red', markersize=10,
                   markeredgecolor='black', markeredgewidth=0.5),
            Line2D([0], [0], color='black', linestyle='--', linewidth=2,
                   label='log₁₀(E(B-V)) = -1.1β - 3.3')
        ]
        ax_main.legend(handles=legend_elements, loc='best', fontsize=11, frameon=True)

        # Side histogram for simulations with f_esc > 5% and ratio cut
        if mask_sim.any():
            log_ebv_high_fesc = np.log10(self.df.loc[mask_sim, 'ebmv_mean'])
            ax_hist.hist(log_ebv_high_fesc, bins=30, orientation='horizontal',
                         color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax_hist.set_xlabel('Count', fontsize=11)
            ax_hist.set_title(f'SPHINX20\n$f_{{esc}}$ > 5%\nHα/L1500 < 1.6\n(n={mask_sim.sum()})',
                              fontsize=10, fontweight='bold')
            ax_hist.grid(True, alpha=0.3, axis='x')

        # Remove y-axis labels from histogram (shared with main plot)
        ax_hist.tick_params(labelleft=False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()


    def plot_muv_fesc_comparison(self, figsize=(10, 8), save_path=None):
        """
        Create M_UV vs f_esc comparison plot between SPHINX20 simulations and LzLCS observations.
        Simulations color-coded by redshift, observations shown as red squares.
        """
        fig, ax = plt.subplots(figsize=figsize)

        print("\n" + "=" * 60)
        print("M_UV vs f_esc COMPARISON")
        print("=" * 60)

        # ============================================================
        # SIMULATIONS (SPHINX20)
        # ============================================================
        print("\nSIMULATIONS (SPHINX20):")

        # Base mask for simulations
        mask_sim = (self.df['f_esc'].notna() &
                    self.df['cont_1500_int'].notna() &
                    self.df['redshift'].notna())

        print(f"Total galaxies with f_esc, L_UV, and redshift: {mask_sim.sum()}")


        # Now calculate M_UV using AB magnitude system
        self.df['M_UV'] = self.df['MAB_1500_int']
        # Get f_esc in linear scale (percentage)
        f_esc_sim = 10 ** self.df.loc[mask_sim, 'f_esc'] * 100

        # Plot simulations color-coded by redshift
        scatter_sim = ax.scatter(
            self.df.loc[mask_sim, 'M_UV'],
            f_esc_sim,
            c=self.df.loc[mask_sim, 'redshift'],
            cmap='viridis',
            alpha=0.6,
            s=50,
            label='SPHINX20',
            edgecolors='none'
        )

        # Add colorbar for redshift
        cbar = plt.colorbar(scatter_sim, ax=ax)
        cbar.set_label('Redshift', fontsize=12)

        print(f"M_UV range: [{self.df.loc[mask_sim, 'M_UV'].min():.2f}, {self.df.loc[mask_sim, 'M_UV'].max():.2f}]")
        print(f"f_esc range: [{f_esc_sim.min():.2f}, {f_esc_sim.max():.2f}] %")
        print(
            f"Redshift range: [{self.df.loc[mask_sim, 'redshift'].min():.2f}, {self.df.loc[mask_sim, 'redshift'].max():.2f}]")

        # ============================================================
        # OBSERVATIONS (LzLCS)
        # ============================================================
        print("\nOBSERVATIONS (LzLCS):")

        # Base mask for observations
        mask_obs = (self.observations['M_1500'].notna() &
                    self.observations['f_esc(LyC)-UVfit'].notna())
        mask_obs2 = (self.observations['M_1500'].notna() &
                    self.observations['f_esc(LyC)-UVfit'].notna()&
                    self.observations['f_esc(LyC)-UVfit']>0)
        print(f"Total galaxies with M_UV and f_esc: {mask_obs.sum()}")

        # Get f_esc in percentage
        f_esc_obs = self.observations.loc[mask_obs, 'f_esc(LyC)-UVfit'] * 100
        f_esc_obs2 = self.observations.loc[mask_obs2, 'f_esc(LyC)-UVfit'] * 100

        # Plot observations as red squares
        if mask_obs.any():
            ax.scatter(
                self.observations.loc[mask_obs, 'M_1500'],
                f_esc_obs,
                alpha=0.8,
                s=150,
                label='LzLCS',
                color='red',
                marker='s',
                edgecolors='black',
                linewidth=1,
                zorder=5
            )


            print(
                f"M_UV range: [{self.observations.loc[mask_obs, 'M_1500'].min():.2f}, {self.observations.loc[mask_obs, 'M_1500'].max():.2f}]")
            print(f"f_esc range: [{f_esc_obs.min():.2f}, {f_esc_obs.max():.2f}] %")

        print("=" * 60 + "\n")

        # Labels and formatting
        ax.set_xlabel('$M_{UV}$', fontsize=14)
        ax.set_ylabel('$f_{esc}$ (%)', fontsize=14)
        ax.set_title('UV Magnitude vs LyC Escape Fraction', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Set log scale for y-axis (optional, uncomment if desired)
        # ax.set_yscale('log')
        # ax.set_ylabel('$f_{esc}$ (%) [log scale]', fontsize=14)

        # Invert x-axis (brighter objects have more negative magnitudes)
        ax.invert_xaxis()

        # Legend
        ax.legend(loc='best', fontsize=12, frameon=True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

def main():
    """Main analysis pipeline."""

    # Initialize analyser
    analyzer = SPHINXLyCAnalyzer(data_path+'all_basic_data.csv', data_path+'flury.csv')


    analyzer.plot_multiparameter_comparison(save_path=home_path+'comprehensive_comparison_all_F011.png')

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()