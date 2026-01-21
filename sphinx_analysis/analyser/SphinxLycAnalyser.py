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

    def summary_statistics(self):
        """Print summary statistics for LyC escape fractions."""
        print("\n" + "=" * 60)
        print("LyC ESCAPE FRACTION SUMMARY STATISTICS")
        print("=" * 60)

        print(f"\nTotal number of galaxies: {len(self.df)}")
        print(f"Redshift range: {self.df['redshift'].min():.2f} - {self.df['redshift'].max():.2f}")

        print("\n--- f_esc (log10 LyC escape fraction) ---")
        print(f"Mean: {self.df['f_esc'].mean():.3f}")
        print(f"Median: {self.df['f_esc'].median():.3f}")
        print(f"Std: {self.df['f_esc'].std():.3f}")
        print(f"Min: {self.df['f_esc'].min():.3f}")
        print(f"Max: {self.df['f_esc'].max():.3f}")

        print("\n--- f_esc (linear, %) ---")
        print(f"Mean: {self.df['f_esc_linear'].mean() * 100:.2f}%")
        print(f"Median: {self.df['f_esc_linear'].median() * 100:.2f}%")

        # Directional escape fractions
        if 'fesc_dir_mean' in self.df.columns:
            print("\n--- Directional f_esc (mean across viewing angles) ---")
            print(f"Mean: {self.df['fesc_dir_mean'].mean() * 100:.2f}%")
            print(f"Median: {self.df['fesc_dir_mean'].median() * 100:.2f}%")
            print(f"Mean anisotropy (std): {self.df['fesc_dir_std'].mean() * 100:.2f}%")

        # Galaxies with high escape fractions
        high_esc = (self.df['f_esc_linear'] > 0.1).sum()
        print(f"\nGalaxies with f_esc > 10%: {high_esc} ({high_esc / len(self.df) * 100:.1f}%)")

        very_high_esc = (self.df['f_esc_linear'] > 0.2).sum()
        print(f"Galaxies with f_esc > 20%: {very_high_esc} ({very_high_esc / len(self.df) * 100:.1f}%)")

    def redshift_evolution(self):
        """Analyze LyC escape fraction evolution with redshift."""
        print("\n" + "=" * 60)
        print("REDSHIFT EVOLUTION")
        print("=" * 60)

        redshifts = sorted(self.df['redshift'].unique())
        print(f"\nAvailable redshifts: {redshifts}")

        for z in redshifts:
            z_data = self.df[self.df['redshift'] == z]
            mean_fesc = z_data['f_esc_linear'].mean() * 100
            median_fesc = z_data['f_esc_linear'].median() * 100
            print(f"z = {z:.1f}: N = {len(z_data):4d}, "
                  f"<f_esc> = {mean_fesc:.2f}%, "
                  f"median = {median_fesc:.2f}%")

    def correlations(self):
        """Calculate correlations between f_esc and galaxy properties."""
        print("\n" + "=" * 60)
        print("CORRELATIONS WITH f_esc")
        print("=" * 60)

        # Properties to correlate with f_esc
        properties = {
            'stellar_mass': 'Stellar Mass (log M☉)',
            'mvir': 'Halo Mass (log M☉)',
            'sfr_10': 'SFR (10 Myr avg)',
            'ionizing_luminosity': 'Ionizing Luminosity (log)',
            'gas_metallicity': 'Gas Metallicity (log Z☉)',
            'stellar_metallicity': 'Stellar Metallicity (log Z☉)',
            'mean_stellar_age_mass': 'Mean Stellar Age (Myr)',
        }

        print("\nSpearman correlation coefficients with f_esc:")
        print("-" * 60)

        results = []
        for prop, name in properties.items():
            if prop in self.df.columns:
                # Remove NaN values
                mask = ~(self.df[prop].isna() | self.df['f_esc'].isna())
                if mask.sum() > 10:  # Need at least 10 points
                    corr, pval = stats.spearmanr(self.df.loc[mask, prop],
                                                 self.df.loc[mask, 'f_esc'])
                    results.append({
                        'Property': name,
                        'Correlation': corr,
                        'P-value': pval,
                        'Significant': 'Yes' if pval < 0.05 else 'No'
                    })
                    print(f"{name:30s}: r = {corr:6.3f}, p = {pval:.3e}")

        return pd.DataFrame(results)

    def plot_overview(self, output_file='sphinx_lyc_overview.png'):
        """Create overview plots of LyC escape fractions."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('SPHINX20 LyC Escape Fraction Overview',
                     fontsize=16, fontweight='bold')

        # 1. Distribution of f_esc
        ax = axes[0, 0]
        ax.hist(self.df['f_esc_linear'] * 100, bins=50,
                edgecolor='black', alpha=0.7)
        ax.set_xlabel('f_esc (%)')
        ax.set_ylabel('Number of Galaxies')
        ax.set_title('Distribution of LyC Escape Fraction')
        ax.axvline(self.df['f_esc_linear'].mean() * 100,
                   color='red', linestyle='--', label='Mean')
        ax.axvline(self.df['f_esc_linear'].median() * 100,
                   color='black', linestyle='--', label='Median')
        ax.legend()

        # 2. f_esc vs Stellar Mass
        ax = axes[0, 1]
        scatter = ax.scatter(self.df['stellar_mass'],
                             self.df['f_esc_linear'] * 100,
                             c=self.df['redshift'],
                             alpha=0.6, s=20, cmap='viridis')
        ax.set_xlabel('Stellar Mass (log M☉)')
        ax.set_ylabel('f_esc (%)')
        ax.set_title('f_esc vs Stellar Mass')
        ax.set_yscale('log')
        plt.colorbar(scatter, ax=ax, label='Redshift')

        # 3. f_esc vs SFR
        ax = axes[0, 2]
        mask = self.df['sfr_10'] > 0
        ax.scatter(self.df.loc[mask, 'sfr_10'],
                   self.df.loc[mask, 'f_esc_linear'] * 100,
                   alpha=0.6, s=20)
        ax.set_xlabel('SFR (10 Myr avg, M☉/yr)')
        ax.set_ylabel('f_esc (%)')
        ax.set_title('f_esc vs Star Formation Rate')
        ax.set_xscale('log')
        ax.set_yscale('log')

        # 4. f_esc vs Gas Metallicity
        ax = axes[1, 0]
        ax.scatter(self.df['gas_metallicity'],
                   self.df['f_esc_linear'] * 100,
                   alpha=0.6, s=20)
        ax.set_xlabel('Gas Metallicity (log Z☉)')
        ax.set_ylabel('f_esc (%)')
        ax.set_title('f_esc vs Gas Metallicity')
        ax.set_yscale('log')

        # 5. Redshift Evolution
        ax = axes[1, 1]
        redshifts = sorted(self.df['redshift'].unique())
        mean_fesc = [self.df[self.df['redshift'] == z]['f_esc_linear'].mean() * 100
                     for z in redshifts]
        median_fesc = [self.df[self.df['redshift'] == z]['f_esc_linear'].median() * 100
                       for z in redshifts]

        ax.plot(redshifts, mean_fesc, 'o-', label='Mean', markersize=8)
        ax.plot(redshifts, median_fesc, 's-', label='Median', markersize=8)
        ax.set_xlabel('Redshift')
        ax.set_ylabel('f_esc (%)')
        ax.set_title('f_esc Evolution with Redshift')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Directional variance
        if 'fesc_dir_std' in self.df.columns:
            ax = axes[1, 2]
            ax.scatter(self.df['fesc_dir_mean'] * 100,
                       self.df['fesc_dir_std'] * 100,
                       alpha=0.6, s=20)
            ax.set_xlabel('Mean Directional f_esc (%)')
            ax.set_ylabel('Std Dev of Directional f_esc (%)')
            ax.set_title('Anisotropy in LyC Escape')
            ax.plot([0, 100], [0, 0], 'k--', alpha=0.3)
        else:
            ax = axes[1, 2]
            ax.text(0.5, 0.5, 'Directional f_esc\ndata not available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved overview plot to {output_file}")
        return fig

    def plot_detailed_correlations(self, output_file='sphinx_lyc_correlations.png'):
        """Create detailed correlation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('SPHINX20: f_esc Correlations',
                     fontsize=16, fontweight='bold')

        # 1. f_esc vs Ionizing Luminosity
        ax = axes[0, 0]
        ax.scatter(self.df['ionizing_luminosity'],
                   self.df['f_esc_linear'] * 100,
                   alpha=0.5, s=20)
        ax.set_xlabel('Ionizing Luminosity (log phot/s)')
        ax.set_ylabel('f_esc (%)')
        ax.set_yscale('log')
        ax.set_title('f_esc vs Ionizing Luminosity')

        # Calculate and display correlation
        mask = ~(self.df['ionizing_luminosity'].isna() | self.df['f_esc'].isna())
        if mask.sum() > 10:
            corr, pval = stats.spearmanr(
                self.df.loc[mask, 'ionizing_luminosity'],
                self.df.loc[mask, 'f_esc']
            )
            ax.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {pval:.3e}',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. f_esc vs Stellar Age
        ax = axes[0, 1]
        mask = self.df['mean_stellar_age_mass'] > 0
        ax.scatter(self.df.loc[mask, 'mean_stellar_age_mass'],
                   self.df.loc[mask, 'f_esc_linear'] * 100,
                   alpha=0.5, s=20)
        ax.set_xlabel('Mean Stellar Age (Myr)')
        ax.set_ylabel('f_esc (%)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('f_esc vs Stellar Age')

        # 3. f_esc vs Halo Mass
        ax = axes[1, 0]
        ax.scatter(self.df['mvir'],
                   self.df['f_esc_linear'] * 100,
                   alpha=0.5, s=20)
        ax.set_xlabel('Halo Mass (log M☉)')
        ax.set_ylabel('f_esc (%)')
        ax.set_yscale('log')
        ax.set_title('f_esc vs Halo Mass')

        # 4. Stellar Mass vs Halo Mass colored by f_esc
        ax = axes[1, 1]
        scatter = ax.scatter(self.df['mvir'],
                             self.df['stellar_mass'],
                             c=np.log10(self.df['f_esc_linear'] * 100),
                             alpha=0.6, s=20, cmap='RdYlBu_r')
        ax.set_xlabel('Halo Mass (log M☉)')
        ax.set_ylabel('Stellar Mass (log M☉)')
        ax.set_title('Galaxy Mass Relation (colored by f_esc)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log(f_esc %)')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved correlation plot to {output_file}")
        return fig

    def analyze_high_escapers(self, threshold=0.1):
        """Analyze properties of high LyC escape fraction galaxies."""
        print("\n" + "=" * 60)
        print(f"HIGH ESCAPER ANALYSIS (f_esc > {threshold * 100}%)")
        print("=" * 60)

        high_esc = self.df[self.df['f_esc_linear'] > threshold]
        low_esc = self.df[self.df['f_esc_linear'] <= threshold]

        print(f"\nHigh escapers: {len(high_esc)} galaxies")
        print(f"Low escapers: {len(low_esc)} galaxies")

        properties = ['stellar_mass', 'mvir', 'sfr_10', 'gas_metallicity',
                      'stellar_metallicity', 'ionizing_luminosity']

        print("\nProperty comparison:")
        print("-" * 60)
        for prop in properties:
            if prop in self.df.columns:
                high_mean = high_esc[prop].mean()
                low_mean = low_esc[prop].mean()
                print(f"{prop:25s}: High = {high_mean:8.3f}, Low = {low_mean:8.3f}")

        return high_esc, low_esc

    def export_summary(self, output_file='sphinx_lyc_summary.txt'):
        """Export summary statistics to text file."""
        with open(output_file, 'w') as f:
            f.write("SPHINX20 LyC ESCAPE FRACTION ANALYSIS\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Total galaxies: {len(self.df)}\n")
            f.write(f"Redshift range: {self.df['redshift'].min():.2f} - "
                    f"{self.df['redshift'].max():.2f}\n\n")

            f.write("ESCAPE FRACTION STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean f_esc: {self.df['f_esc_linear'].mean() * 100:.2f}%\n")
            f.write(f"Median f_esc: {self.df['f_esc_linear'].median() * 100:.2f}%\n")
            f.write(f"Std f_esc: {self.df['f_esc_linear'].std() * 100:.2f}%\n\n")

            f.write("REDSHIFT EVOLUTION\n")
            f.write("-" * 70 + "\n")
            for z in sorted(self.df['redshift'].unique()):
                z_data = self.df[self.df['redshift'] == z]
                f.write(f"z = {z:.1f}: N = {len(z_data)}, "
                        f"<f_esc> = {z_data['f_esc_linear'].mean() * 100:.2f}%\n")

        print(f"\nExported summary to {output_file}")

    def plot_fesc_vs_stellar_mass(self, figsize=(14, 6), save_path=None):
        """
        Plot escape fraction vs stellar mass for both simulations and observations.

        Parameters:
        -----------
        figsize : tuple, default (14, 6)
        save_path : str, optional path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Simulation data
        ax = axes[0]
        scatter = ax.scatter(self.df['stellar_mass'],
                             self.df['f_esc'],
                             c=self.df['redshift'],
                             cmap='viridis',
                             alpha=0.6,
                             s=50)
        ax.set_xlabel('log₁₀(M* / M☉)', fontsize=12)
        ax.set_ylabel('log₁₀(f_esc)', fontsize=12)
        ax.set_title('SPHINX20 Simulations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Redshift', fontsize=11)

        # Observation data
        ax = axes[1]
        # Using f_esc(LyC)-Hbeta as the escape fraction
        mask_obs = (self.observations['f_esc(LyC)-Hbeta'].notna() &
                    self.observations['log10(Mstar)'].notna())

        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])
        obs_mass = self.observations.loc[mask_obs, 'log10(Mstar)']
        obs_z = self.observations.loc[mask_obs, 'z']

        scatter = ax.scatter(obs_mass, obs_fesc,
                             c=obs_z,
                             cmap='viridis',
                             alpha=0.6,
                             s=100,
                             marker='s',
                             edgecolors='black',
                             linewidth=0.5)
        ax.set_xlabel('log₁₀(M* / M☉)', fontsize=12)
        ax.set_ylabel('log₁₀(f_esc)', fontsize=12)
        ax.set_title('Observations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Redshift', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_fesc_vs_sfr(self, figsize=(14, 6), save_path=None):
        """
        Plot escape fraction vs star formation rate.
        Color-coded by stellar mass.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Simulation data (using SFR_100)
        ax = axes[0]
        scatter = ax.scatter(self.df['sfr_100'],
                             self.df['f_esc'],
                             c=self.df['stellar_mass'],
                             cmap='plasma',
                             alpha=0.6,
                             s=50)
        ax.set_xlabel('log₁₀(SFR / M☉ yr⁻¹)', fontsize=12)
        ax.set_ylabel('log₁₀(f_esc)', fontsize=12)
        ax.set_title('SPHINX20 Simulations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(M* / M☉)', fontsize=11)

        # Observation data
        ax = axes[1]
        mask_obs = (self.observations['f_esc(LyC)-Hbeta'].notna() &
                    self.observations['log10(SFR)-UV'].notna() &
                    self.observations['log10(Mstar)'].notna())

        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])
        obs_sfr = self.observations.loc[mask_obs, 'log10(SFR)-UV']
        obs_mass = self.observations.loc[mask_obs, 'log10(Mstar)']

        scatter = ax.scatter(obs_sfr, obs_fesc,
                             c=obs_mass,
                             cmap='plasma',
                             alpha=0.6,
                             s=100,
                             marker='s',
                             edgecolors='black',
                             linewidth=0.5)
        ax.set_xlabel('log₁₀(SFR / M☉ yr⁻¹)', fontsize=12)
        ax.set_ylabel('log₁₀(f_esc)', fontsize=12)
        ax.set_title('Observations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(M* / M☉)', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_fesc_vs_metallicity(self, figsize=(14, 6), save_path=None):
        """
        Plot escape fraction vs gas metallicity.
        Color-coded by stellar mass.
        Converts simulation metallicity to 12+log(O/H) scale for proper comparison.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Simulation data - CONVERT to 12+log(O/H) scale
        # log(Z/Z☉) → 12+log(O/H) conversion:
        # 12+log(O/H) = [12+log(O/H)☉] + log(Z/Z☉)
        # Using solar value: 12+log(O/H)☉ ≈ 8.69
        solar_oh = 8.69
        sim_metallicity = self.df['gas_metallicity'] + solar_oh

        ax = axes[0]
        scatter = ax.scatter(sim_metallicity,
                             self.df['f_esc'],
                             c=self.df['stellar_mass'],
                             cmap='coolwarm',
                             alpha=0.6,
                             s=50)
        ax.set_xlabel('12 + log₁₀(O/H)', fontsize=12)
        ax.set_ylabel('log₁₀(f_esc)', fontsize=12)
        ax.set_title('SPHINX20 Simulations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(M* / M☉)', fontsize=11)

        # Observation data (already in 12+log(O/H) scale)
        ax = axes[1]
        mask_obs = (self.observations['f_esc(LyC)-Hbeta'].notna() &
                    self.observations['OH_12'].notna() &
                    self.observations['log10(Mstar)'].notna())

        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])
        obs_metallicity = self.observations.loc[mask_obs, 'OH_12']  # Already 12+log(O/H)
        obs_mass = self.observations.loc[mask_obs, 'log10(Mstar)']

        scatter = ax.scatter(obs_metallicity, obs_fesc,
                             c=obs_mass,
                             cmap='coolwarm',
                             alpha=0.6,
                             s=100,
                             marker='s',
                             edgecolors='black',
                             linewidth=0.5)
        ax.set_xlabel('12 + log₁₀(O/H)', fontsize=12)
        ax.set_ylabel('log₁₀(f_esc)', fontsize=12)
        ax.set_title('Observations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(M* / M☉)', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    def plot_fesc_vs_uvbeta(self, figsize=(14, 6), save_path=None):
        """
        Plot escape fraction vs UV slope (beta).
        Color-coded by redshift.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Simulation data (using beta_int_sn)
        ax = axes[0]
        mask_sim = self.df['beta_int_sn'].notna()
        scatter = ax.scatter(self.df.loc[mask_sim, 'beta_int_sn'],
                             self.df.loc[mask_sim, 'f_esc'],
                             c=self.df.loc[mask_sim, 'redshift'],
                             cmap='viridis',
                             alpha=0.6,
                             s=50)
        ax.set_xlabel('β (UV slope)', fontsize=12)
        ax.set_ylabel('log₁₀(f_esc)', fontsize=12)
        ax.set_title('SPHINX20 Simulations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Redshift', fontsize=11)

        # Observation data
        ax = axes[1]
        mask_obs = (self.observations['f_esc(LyC)-Hbeta'].notna() &
                    self.observations['UV-beta'].notna())

        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])
        obs_beta = self.observations.loc[mask_obs, 'UV-beta']
        obs_z = self.observations.loc[mask_obs, 'z']

        scatter = ax.scatter(obs_beta, obs_fesc,
                             c=obs_z,
                             cmap='viridis',
                             alpha=0.6,
                             s=100,
                             marker='s',
                             edgecolors='black',
                             linewidth=0.5)
        ax.set_xlabel('β (UV slope)', fontsize=12)
        ax.set_ylabel('log₁₀(f_esc)', fontsize=12)
        ax.set_title('Observations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Redshift', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')


    def plot_fesc_vs_xi_ion(self, figsize=(14, 6), save_path=None):
        """
        Plot escape fraction vs ionizing photon production efficiency (xi_ion).
        Color-coded by stellar mass.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Simulation data
        ax = axes[0]
        scatter = ax.scatter(self.df['xi_ion'],
                             self.df['f_esc'],
                             c=self.df['stellar_mass'],
                             cmap='magma',
                             alpha=0.6,
                             s=50)
        ax.set_xlabel('log₁₀(ξ_ion / Hz erg⁻¹)', fontsize=12)
        ax.set_ylabel('log₁₀(f_esc)', fontsize=12)
        ax.set_title('SPHINX20 Simulations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(M* / M☉)', fontsize=11)

        # Observation data
        ax = axes[1]
        mask_obs = (self.observations['f_esc(LyC)-Hbeta'].notna() &
                    self.observations['xi-ion'].notna() &
                    self.observations['log10(Mstar)'].notna())

        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])
        obs_xi = self.observations.loc[mask_obs, 'xi-ion']
        obs_mass = self.observations.loc[mask_obs, 'log10(Mstar)']

        scatter = ax.scatter(obs_xi, obs_fesc,
                             c=obs_mass,
                             cmap='magma',
                             alpha=0.6,
                             s=100,
                             marker='s',
                             edgecolors='black',
                             linewidth=0.5)
        ax.set_xlabel('log₁₀(ξ_ion / Hz erg⁻¹)', fontsize=12)
        ax.set_ylabel('log₁₀(f_esc)', fontsize=12)
        ax.set_title('Observations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(M* / M☉)', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_fesc_histograms(self, figsize=(10, 6), save_path=None):
        """
        Compare distributions of escape fractions between simulations and observations.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Simulation data
        ax.hist(self.df['f_esc'], bins=30, alpha=0.5,
                label='SPHINX20 Simulations', color='blue', edgecolor='black')

        # Observation data
        mask_obs = self.observations['f_esc(LyC)-Hbeta'].notna()
        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])
        ax.hist(obs_fesc, bins=20, alpha=0.5,
                label='LzLCS', color='red', edgecolor='black')

        ax.set_xlabel('log₁₀(f_esc)', fontsize=13)
        ax.set_ylabel('Number of Galaxies', fontsize=13)
        ax.set_title('Escape Fraction Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

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
        ax1.set_xlabel('log₁₀(M* / Msol)', fontsize=11)
        ax1.set_ylabel('log₁₀(f_esc)', fontsize=11)
        ax1.set_title('Stellar Mass vs f_esc', fontsize=12, fontweight='bold')
        ax1.legend()
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

        ax2.set_xlabel('log10(SFR)(Msol /yr)', fontsize=11)
        ax2.set_ylabel('log10(f_esc)', fontsize=11)
        ax2.set_title('Star Formation Rate vs LyC escape fraction, xlim[0,5]', fontsize=12, fontweight='bold')
        ax2.legend()
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

        ax3.set_ylabel('log₁₀(f_esc)', fontsize=11)
        ax3.set_title('OIII/OII vs f_esc, xrange [0,40]', fontsize=12, fontweight='bold')
        ax3.legend()
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
        ax4.set_xlabel('log₁₀(ξ_ion / Hz erg)', fontsize=11)
        ax4.set_ylabel('log₁₀(f_esc)', fontsize=11)
        ax4.set_title('ξ_ion vs f_esc', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)


        cbar1 = fig.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Redshift', fontsize=9)

        cbar2 = fig.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Redshift', fontsize=9)

        cbar3 = fig.colorbar(scatter3, ax=ax3)
        cbar3.set_label('Redshift', fontsize=9)

        cbar4 = fig.colorbar(scatter4, ax=ax4)
        cbar4.set_label('Redshift', fontsize=9)

        plt.suptitle(' Multi-Parameter Comparison: Simulations vs Observations',
                     fontsize=15, fontweight='bold', y=0.995)


        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# INDIRECT f_esc INDICATOR PLOTS
# ============================================================================

    def plot_o32_vs_metallicity(self, figsize=(14, 6), save_path=None):
        """
        Plot O32 ([OIII]/[OII]) vs metallicity.
        High O32 is a strong indicator of high ionization and f_esc.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Simulation data
        ax = axes[0]
        # Calculate O32 = [OIII]5007 / [OII]3727
        sim_o32 = np.log10(self.df['OIII_5006.84_int'] / self.df['OII_3726.03_int'])
        solar_oh = 8.69
        sim_metallicity = self.df['gas_metallicity'] + solar_oh

        scatter = ax.scatter(sim_metallicity, sim_o32,
                             c=self.df['f_esc'],
                             cmap='plasma',
                             alpha=0.6,
                             s=50,
                             vmin=-3, vmax=0)
        ax.set_xlabel('12 + log₁₀(O/H)', fontsize=12)
        ax.set_ylabel('log₁₀(O32) = log([OIII]/[OII])', fontsize=12)
        ax.set_title('SPHINX20 Simulations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=np.log10(3), color='red', linestyle='--', alpha=0.5, label='O32=3 (leaker threshold)')
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(f_esc)', fontsize=11)

        # Observation data
        ax = axes[1]
        mask_obs = (self.observations['O3_5007A'].notna() &
                    self.observations['O2_3726A'].notna() &
                    self.observations['OH_12'].notna() &
                    self.observations['f_esc(LyC)-Hbeta'].notna())

        obs_o32 = np.log10(self.observations.loc[mask_obs, 'O3_5007A'] /
                           self.observations.loc[mask_obs, 'O2_3726A'])
        obs_metallicity = self.observations.loc[mask_obs, 'OH_12']
        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])

        # Size by f_esc (larger = higher f_esc)
        sizes = 50 + 200 * (obs_fesc - obs_fesc.min()) / (obs_fesc.max() - obs_fesc.min())

        scatter = ax.scatter(obs_metallicity, obs_o32,
                             c=obs_fesc,
                             cmap='plasma',
                             alpha=0.6,
                             s=sizes,
                             marker='s',
                             edgecolors='black',
                             linewidth=0.5,
                             vmin=-3, vmax=0)
        ax.set_xlabel('12 + log₁₀(O/H)', fontsize=12)
        ax.set_ylabel('log₁₀(O32) = log([OIII]/[OII])', fontsize=12)
        ax.set_title('Observations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=np.log10(3), color='red', linestyle='--', alpha=0.5, label='O32=3 (leaker threshold)')
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(f_esc)', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


    def plot_ew_hbeta_vs_mass(self, figsize=(14, 6), save_path=None):
        """
        Plot Hbeta equivalent width vs stellar mass.
        High EW indicates young, bursty SF => high f_esc.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Simulation data - calculate EW(Hβ)
        ax = axes[0]
        # EW = Line flux / continuum
        sim_ew_hb = self.df['HI_4861.32_int'] / self.df['cont_4861_int']

        scatter = ax.scatter(self.df['stellar_mass'],
                             np.log10(sim_ew_hb),
                             c=self.df['f_esc'],
                             cmap='coolwarm',
                             alpha=0.6,
                             s=50,
                             vmin=-3, vmax=0)
        ax.set_xlabel('log₁₀(M* / M☉)', fontsize=12)
        ax.set_ylabel('log₁₀(EW(Hβ) / Å)', fontsize=12)
        ax.set_title('SPHINX20 Simulations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=np.log10(100), color='red', linestyle='--', alpha=0.5, label='EW=100Å')
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(f_esc)', fontsize=11)

        # Observation data
        ax = axes[1]
        mask_obs = (self.observations['EW(H1r_4861)'].notna() &
                    self.observations['log10(Mstar)'].notna() &
                    self.observations['f_esc(LyC)-Hbeta'].notna())

        obs_ew = self.observations.loc[mask_obs, 'EW(H1r_4861)']
        obs_mass = self.observations.loc[mask_obs, 'log10(Mstar)']
        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])

        scatter = ax.scatter(obs_mass, np.log10(obs_ew),
                             c=obs_fesc,
                             cmap='coolwarm',
                             alpha=0.6,
                             s=100,
                             marker='s',
                             edgecolors='black',
                             linewidth=0.5,
                             vmin=-3, vmax=0)
        ax.set_xlabel('log₁₀(M* / M☉)', fontsize=12)
        ax.set_ylabel('log₁₀(EW(Hβ) / Å)', fontsize=12)
        ax.set_title('Observations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=np.log10(100), color='red', linestyle='--', alpha=0.5, label='EW=100Å')
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(f_esc)', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


    def plot_uv_slope_vs_metallicity(self, figsize=(14, 6), save_path=None):
        """
        Plot UV slope (β) vs metallicity.
        Blue UV slope (β < -2) + low metallicity → high f_esc.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Simulation data
        ax = axes[0]
        mask_sim = self.df['beta_int_sn'].notna()
        solar_oh = 8.69
        sim_metallicity = self.df.loc[mask_sim, 'gas_metallicity'] + solar_oh

        scatter = ax.scatter(sim_metallicity,
                             self.df.loc[mask_sim, 'beta_int_sn'],
                             c=self.df.loc[mask_sim, 'f_esc'],
                             cmap='viridis',
                             alpha=0.6,
                             s=50,
                             vmin=-3, vmax=0)
        ax.set_xlabel('12 + log₁₀(O/H)', fontsize=12)
        ax.set_ylabel('β (UV slope)', fontsize=12)
        ax.set_title('SPHINX20 Simulations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5, label='β=-2 (dust-free)')
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(f_esc)', fontsize=11)

        # Observation data
        ax = axes[1]
        mask_obs = (self.observations['UV-beta'].notna() &
                    self.observations['OH_12'].notna() &
                    self.observations['f_esc(LyC)-Hbeta'].notna())

        obs_beta = self.observations.loc[mask_obs, 'UV-beta']
        obs_metallicity = self.observations.loc[mask_obs, 'OH_12']
        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])

        scatter = ax.scatter(obs_metallicity, obs_beta,
                             c=obs_fesc,
                             cmap='viridis',
                             alpha=0.6,
                             s=100,
                             marker='s',
                             edgecolors='black',
                             linewidth=0.5,
                             vmin=-3, vmax=0)
        ax.set_xlabel('12 + log₁₀(O/H)', fontsize=12)
        ax.set_ylabel('β (UV slope)', fontsize=12)
        ax.set_title('Observations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5, label='β=-2 (dust-free)')
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(f_esc)', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


    def plot_ssfr_vs_mass(self, figsize=(14, 6), save_path=None):
        """
        Plot specific star formation rate (sSFR) vs stellar mass.
        High sSFR indicates bursty SF → feedback → high f_esc.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Simulation data
        ax = axes[0]
        # sSFR = SFR / M* (both in log, so subtract)
        sim_ssfr = self.df['sfr_100'] - self.df['stellar_mass']

        scatter = ax.scatter(self.df['stellar_mass'], sim_ssfr,
                             c=self.df['f_esc'],
                             cmap='plasma',
                             alpha=0.6,
                             s=50,
                             vmin=-3, vmax=0)
        ax.set_xlabel('log₁₀(M* / M☉)', fontsize=12)
        ax.set_ylabel('log₁₀(sSFR / yr⁻¹)', fontsize=12)
        ax.set_title('SPHINX20 Simulations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=-8, color='red', linestyle='--', alpha=0.5, label='sSFR=10⁻⁸ yr⁻¹')
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(f_esc)', fontsize=11)

        # Observation data
        ax = axes[1]
        mask_obs = (self.observations['log10(SFR)-UV'].notna() &
                    self.observations['log10(Mstar)'].notna() &
                    self.observations['f_esc(LyC)-Hbeta'].notna())

        obs_ssfr = (self.observations.loc[mask_obs, 'log10(SFR)-UV'] -
                    self.observations.loc[mask_obs, 'log10(Mstar)'])
        obs_mass = self.observations.loc[mask_obs, 'log10(Mstar)']
        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])

        scatter = ax.scatter(obs_mass, obs_ssfr,
                             c=obs_fesc,
                             cmap='plasma',
                             alpha=0.6,
                             s=100,
                             marker='s',
                             edgecolors='black',
                             linewidth=0.5,
                             vmin=-3, vmax=0)
        ax.set_xlabel('log₁₀(M* / M☉)', fontsize=12)
        ax.set_ylabel('log₁₀(sSFR / yr⁻¹)', fontsize=12)
        ax.set_title('Observations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=-8, color='red', linestyle='--', alpha=0.5, label='sSFR=10⁻⁸ yr⁻¹')
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(f_esc)', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


    def plot_o32_vs_ew_hbeta(self, figsize=(14, 6), save_path=None):
        """
        Classic LyC leaker diagnostic: O32 vs EW(Hβ).
        High O32 + high EW → strong LyC leaker.
        *** THIS IS THE MOST IMPORTANT DIAGNOSTIC PLOT ***
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Simulation data
        ax = axes[0]
        sim_o32 = np.log10(self.df['OIII_5006.84_int'] / self.df['OII_3726.03_int'])
        sim_ew_hb = self.df['HI_4861.32_int'] / self.df['cont_4861_int']

        scatter = ax.scatter(np.log10(sim_ew_hb), sim_o32,
                             c=self.df['f_esc'],
                             cmap='magma',
                             alpha=0.6,
                             s=50,
                             vmin=-3, vmax=0)
        ax.set_xlabel('log₁₀(EW(Hβ) / Å)', fontsize=12)
        ax.set_ylabel('log₁₀(O32)', fontsize=12)
        ax.set_title('SPHINX20 Simulations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add leaker region
        ax.axhline(y=np.log10(3), color='red', linestyle='--', alpha=0.3)
        ax.axvline(x=np.log10(100), color='red', linestyle='--', alpha=0.3)
        ax.text(np.log10(150), np.log10(5), 'Leaker\nRegion',
                fontsize=10, color='red', alpha=0.7, fontweight='bold')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(f_esc)', fontsize=11)

        # Observation data
        ax = axes[1]
        mask_obs = (self.observations['EW(H1r_4861)'].notna() &
                    self.observations['O3_5007A'].notna() &
                    self.observations['O2_3726A'].notna() &
                    self.observations['f_esc(LyC)-Hbeta'].notna())

        obs_ew = self.observations.loc[mask_obs, 'EW(H1r_4861)']
        obs_o32 = np.log10(self.observations.loc[mask_obs, 'O3_5007A'] /
                           self.observations.loc[mask_obs, 'O2_3726A'])
        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])

        scatter = ax.scatter(np.log10(obs_ew), obs_o32,
                             c=obs_fesc,
                             cmap='magma',
                             alpha=0.6,
                             s=100,
                             marker='s',
                             edgecolors='black',
                             linewidth=0.5,
                             vmin=-3, vmax=0)
        ax.set_xlabel('log₁₀(EW(Hβ) / Å)', fontsize=12)
        ax.set_ylabel('log₁₀(O32)', fontsize=12)
        ax.set_title('Observations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add leaker region
        ax.axhline(y=np.log10(3), color='red', linestyle='--', alpha=0.3)
        ax.axvline(x=np.log10(100), color='red', linestyle='--', alpha=0.3)
        ax.text(np.log10(150), np.log10(5), 'Leaker\nRegion',
                fontsize=10, color='red', alpha=0.7, fontweight='bold')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(f_esc)', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


    def plot_compactness_diagnostic(self, figsize=(14, 6), save_path=None):
        """
        Plot galaxy size vs stellar mass (compactness).
        Compact galaxies → higher surface brightness → higher f_esc.
        Note: Only observations have size measurements.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Observation data only
        mask_obs = (self.observations['r_50_phys'].notna() &
                    self.observations['log10(Mstar)'].notna() &
                    self.observations['f_esc(LyC)-Hbeta'].notna())

        obs_size = np.log10(self.observations.loc[mask_obs, 'r_50_phys'])
        obs_mass = self.observations.loc[mask_obs, 'log10(Mstar)']
        obs_fesc = np.log10(self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta'])

        scatter = ax.scatter(obs_mass, obs_size,
                             c=obs_fesc,
                             cmap='coolwarm',
                             alpha=0.7,
                             s=150,
                             marker='s',
                             edgecolors='black',
                             linewidth=0.5,
                             vmin=-3, vmax=0)
        ax.set_xlabel('log₁₀(M* / M☉)', fontsize=13)
        ax.set_ylabel('log₁₀(r₅₀ / kpc)', fontsize=13)
        ax.set_title('Galaxy Compactness: Size vs Mass (Observations)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add typical size-mass relation line (for reference)
        mass_ref = np.linspace(7, 10, 50)
        size_ref = 0.14 * (mass_ref - 9) + np.log10(2)  # Approximate relation
        ax.plot(mass_ref, size_ref, 'k--', alpha=0.3, label='Typical size-mass relation')
        ax.legend()

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(f_esc)', fontsize=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


    def plot_lya_lyc_comparison(self, figsize=(10, 8), save_path=None):
        """
        Compare Lyα and LyC escape fractions (observations only).
        Tests if Lyα escape can predict LyC escape.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Observation data only
        mask_obs = (self.observations['f_esc(LyA)'].notna() &
                    self.observations['f_esc(LyC)-Hbeta'].notna() &
                    self.observations['log10(Mstar)'].notna())

        obs_lya = self.observations.loc[mask_obs, 'f_esc(LyA)']
        obs_lyc = self.observations.loc[mask_obs, 'f_esc(LyC)-Hbeta']
        obs_mass = self.observations.loc[mask_obs, 'log10(Mstar)']

        scatter = ax.scatter(obs_lya, obs_lyc,
                             c=obs_mass,
                             cmap='viridis',
                             alpha=0.7,
                             s=150,
                             marker='o',
                             edgecolors='black',
                             linewidth=0.5)

        ax.set_xlabel('f_esc(Lyα)', fontsize=13)
        ax.set_ylabel('f_esc(LyC)', fontsize=13)
        ax.set_title('Lyα vs LyC Escape Fraction Correlation',
                     fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')

        # Add 1:1 line
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
               max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, 'k--', alpha=0.3, label='1:1 relation')
        ax.legend()

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(M* / M☉)', fontsize=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main analysis pipeline."""

    # Initialize analyser
    analyzer = SPHINXLyCAnalyzer(data_path+'all_basic_data.csv', data_path+'flury.csv')

    # Run analyses
    # analyser.summary_statistics()
    #analyser.save_column_names()
    # analyser.redshift_evolution()
    # corr_df = analyser.correlations()

    # analyser.plot_overview(home_path+'sphinx_analysis/outputs/sphinx_lyc_overview.png')
    # analyser.plot_detailed_correlations(home_path+'sphinx_analysis/outputs/sphinx_lyc_correlations.png')

    # Analyze high escapers
    high_esc, low_esc = analyzer.analyze_high_escapers(threshold=0.1)

    # analyser.export_summary(home_path+'sphinx_analysis/outputs/sphinx_lyc_summary.txt')

    # analyser.plot_fesc_vs_stellar_mass(save_path=home_path+'sphinx_analysis/outputs/fesc_mass.png')
    # analyser.plot_fesc_vs_sfr()
    # analyser.plot_fesc_vs_metallicity(save_path=home_path+'sphinx_analysis/outputs/fesc_Z.png')
    # analyser.plot_fesc_vs_uvbeta()
    # analyser.plot_fesc_vs_xi_ion()
    # analyser.plot_fesc_histograms()
    analyzer.plot_multiparameter_comparison(save_path=home_path+'comprehensive_comparison_all.png')

    # Call the new indirect indicator plots:
    # analyzer.plot_o32_vs_metallicity(save_path=home_path + 'o32_metallicity.png')
    # analyzer.plot_ew_hbeta_vs_mass(save_path=home_path + 'ew_mass.png')
    # analyzer.plot_uv_slope_vs_metallicity(save_path=home_path + 'beta_metallicity.png')
    # analyzer.plot_ssfr_vs_mass(save_path=home_path + 'ssfr_mass.png')
    # analyzer.plot_o32_vs_ew_hbeta(save_path=home_path + 'o32_ew_diagnostic.png')
    # analyzer.plot_compactness_diagnostic(save_path=home_path + 'compactness.png')
    # analyzer.plot_lya_lyc_comparison(save_path=home_path + 'lya_lyc.png')

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()