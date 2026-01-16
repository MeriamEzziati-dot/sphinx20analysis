"""
Simulation catalogue class for SPHINX20 data.
"""

import numpy as np
from .base import BaseCatalogue


class SimulationCatalogue(BaseCatalogue):
    """Catalogue for SPHINX20 simulation data."""

    def _preprocess(self):
        """Preprocess simulation data: convert log values, calculate derived quantities."""

        # Convert log values to linear
        if 'f_esc' in self.df.columns:
            self.df['f_esc_linear'] = 10 ** self.df['f_esc']

        if 'stellar_mass' in self.df.columns:
            self.df['stellar_mass_linear'] = 10 ** self.df['stellar_mass']

        if 'ionizing_luminosity' in self.df.columns:
            self.df['ionizing_luminosity_linear'] = 10 ** self.df['ionizing_luminosity']

        # Calculate mean directional escape fraction (average over viewing angles)
        fesc_dir_cols = [col for col in self.df.columns if col.startswith('fesc_dir_')]
        if fesc_dir_cols:
            self.df['fesc_dir_mean'] = self.df[fesc_dir_cols].mean(axis=1)
            self.df['fesc_dir_std'] = self.df[fesc_dir_cols].std(axis=1)

    def get_redshift_bins(self):
        """Return available redshifts in the simulation."""
        if 'redshift' not in self.df.columns:
            raise KeyError("No redshift column found")
        return sorted(self.df['redshift'].unique())

    def get_at_redshift(self, z):
        """
        Get galaxies at a specific redshift.

        Parameters
        ----------
        z : float
            Redshift value

        Returns
        -------
        SimulationCatalogue
            New catalogue with galaxies at specified redshift
        """
        return self.filter(self.df['redshift'] == z)

    def summary(self):
        """Print summary statistics specific to simulation data."""
        super().summary()

        if 'f_esc' in self.df.columns:
            print(f"\n--- f_esc (log10 LyC escape fraction) ---")
            print(f"Mean: {self.df['f_esc'].mean():.3f}")
            print(f"Median: {self.df['f_esc'].median():.3f}")
            print(f"Std: {self.df['f_esc'].std():.3f}")
            print(f"Min: {self.df['f_esc'].min():.3f}")
            print(f"Max: {self.df['f_esc'].max():.3f}")

        if 'f_esc_linear' in self.df.columns:
            print(f"\n--- f_esc (linear, %) ---")
            print(f"Mean: {self.df['f_esc_linear'].mean() * 100:.2f}%")
            print(f"Median: {self.df['f_esc_linear'].median() * 100:.2f}%")

            # High escaper statistics
            high_esc = (self.df['f_esc_linear'] > 0.1).sum()
            print(f"\nGalaxies with f_esc > 10%: {high_esc} ({high_esc / len(self) * 100:.1f}%)")

            very_high_esc = (self.df['f_esc_linear'] > 0.2).sum()
            print(f"Galaxies with f_esc > 20%: {very_high_esc} ({very_high_esc / len(self) * 100:.1f}%)")

        if 'fesc_dir_mean' in self.df.columns:
            print(f"\n--- Directional f_esc (mean across viewing angles) ---")
            print(f"Mean: {self.df['fesc_dir_mean'].mean() * 100:.2f}%")
            print(f"Median: {self.df['fesc_dir_mean'].median() * 100:.2f}%")
            print(f"Mean anisotropy (std): {self.df['fesc_dir_std'].mean() * 100:.2f}%")