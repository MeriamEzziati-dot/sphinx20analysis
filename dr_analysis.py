#!/usr/bin/env python3
"""
SPHINX20 LyC Escape Fraction Analysis Script
=============================================
Focuses on analyzing Lyman Continuum (LyC) escape fractions and their
correlations with galaxy properties in the SPHINX20 data release.

Author: Analysis script for SPHINX20 data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

home_path='/home/mezziati/Documents/IAP/SPHINX20/'
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SPHINXLyCAnalyzer:
    """Analyzer for SPHINX20 LyC escape fraction data."""

    def __init__(self, filepath):
        """Load SPHINX20 data."""
        print("Loading SPHINX20 data...")
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df)} galaxies")

        # Convert log values to linear where needed
        self.df['f_esc_linear'] = 10 ** self.df['f_esc']
        self.df['stellar_mass_linear'] = 10 ** self.df['stellar_mass']
        self.df['ionizing_luminosity_linear'] = 10 ** self.df['ionizing_luminosity']

        # Calculate mean directional escape fraction
        fesc_dir_cols = [col for col in self.df.columns if col.startswith('fesc_dir_')]
        if fesc_dir_cols:
            self.df['fesc_dir_mean'] = self.df[fesc_dir_cols].mean(axis=1)
            self.df['fesc_dir_std'] = self.df[fesc_dir_cols].std(axis=1)

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
                   color='orange', linestyle='--', label='Median')
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


def main():
    """Main analysis pipeline."""

    # Initialize analyzer
    analyzer = SPHINXLyCAnalyzer(home_path+'all_basic_data.csv')

    # Run analyses
    analyzer.summary_statistics()
    analyzer.redshift_evolution()
    corr_df = analyzer.correlations()

    # Generate plots
    analyzer.plot_overview(home_path+'sphinx_analysis/outputs/sphinx_lyc_overview.png')
    analyzer.plot_detailed_correlations(home_path+'sphinx_analysis/outputs/sphinx_lyc_correlations.png')

    # Analyze high escapers
    high_esc, low_esc = analyzer.analyze_high_escapers(threshold=0.1)

    # Export summary
    analyzer.export_summary(home_path+'sphinx_analysis/outputs/sphinx_lyc_summary.txt')

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - sphinx_lyc_overview.png")
    print("  - sphinx_lyc_correlations.png")
    print("  - sphinx_lyc_summary.txt")


if __name__ == "__main__":
    main()