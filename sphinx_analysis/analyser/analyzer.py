"""
Main analyzer class for comparing simulation and observation catalogues.
"""

import numpy as np
import pandas as pd
from ..catalogues import SimulationCatalogue, ObservationCatalogue
from ..comparison.statistics  import (
    calculate_correlations,
    redshift_evolution_summary,
    analyze_high_escapers,
    compare_distributions
)


class LyCEscapeAnalyzer:
    """
    Main analyzer for comparing SPHINX20 simulations with observations.
    """

    def __init__(self, sim_filepath=None, obs_filepath=None,
                 sim_catalogue=None, obs_catalogue=None):
        """
        Initialize analyzer with catalogues.

        Parameters
        ----------
        sim_filepath : str, optional
            Path to simulation CSV file
        obs_filepath : str, optional
            Path to observation CSV file
        sim_catalogue : SimulationCatalogue, optional
            Pre-loaded simulation catalogue
        obs_catalogue : ObservationCatalogue, optional
            Pre-loaded observation catalogue
        """
        if sim_catalogue is not None:
            self.sim = sim_catalogue
        elif sim_filepath is not None:
            print("Loading SPHINX20 simulation data...")
            self.sim = SimulationCatalogue(sim_filepath)
            print(f"Loaded {len(self.sim)} galaxies")
        else:
            raise ValueError("Must provide either sim_filepath or sim_catalogue")

        if obs_catalogue is not None:
            self.obs = obs_catalogue
        elif obs_filepath is not None:
            print("Loading observation data...")
            self.obs = ObservationCatalogue(obs_filepath)
            print(f"Loaded {len(self.obs)} observed objects")
        else:
            raise ValueError("Must provide either obs_filepath or obs_catalogue")

    def summary_statistics(self):
        """Print summary statistics for both catalogues."""
        print("\n" + "=" * 60)
        print("SIMULATION CATALOGUE")
        print("=" * 60)
        self.sim.summary()

        print("\n" + "=" * 60)
        print("OBSERVATION CATALOGUE")
        print("=" * 60)
        self.obs.summary()

    def redshift_evolution(self, target_column='f_esc_linear'):
        """
        Analyze redshift evolution of escape fractions.

        Parameters
        ----------
        target_column : str
            Column to analyze (default: 'f_esc_linear')

        Returns
        -------
        pd.DataFrame
            Summary by redshift
        """
        return redshift_evolution_summary(self.sim, target_column)

    def correlations(self, target_column='f_esc'):
        """
        Calculate correlations between escape fraction and galaxy properties.

        Parameters
        ----------
        target_column : str
            Target column for correlation (default: 'f_esc')

        Returns
        -------
        pd.DataFrame
            Correlation results
        """
        print("\n" + "=" * 60)
        print(f"CORRELATIONS WITH {target_column}")
        print("=" * 60)

        # Define properties to correlate
        properties = {
            'stellar_mass': 'Stellar Mass (log M☉)',
            'mvir': 'Halo Mass (log M☉)',
            'sfr_10': 'SFR (10 Myr avg)',
            'ionizing_luminosity': 'Ionizing Luminosity (log)',
            'gas_metallicity': 'Gas Metallicity (log Z☉)',
            'stellar_metallicity': 'Stellar Metallicity (log Z☉)',
            'mean_stellar_age_mass': 'Mean Stellar Age (Myr)',
        }

        return calculate_correlations(self.sim, target_column, properties)

    def analyze_high_escapers(self, threshold=0.1):
        """
        Separate and analyze high vs low escapers.

        Parameters
        ----------
        threshold : float
            Escape fraction threshold (default: 0.1 = 10%)

        Returns
        -------
        tuple
            (high_escapers, low_escapers) as SimulationCatalogue objects
        """
        return analyze_high_escapers(self.sim, threshold)

    def compare_property_distributions(self, property_name,
                                       sim_column=None, obs_column=None):
        """
        Compare distributions of a property between sim and obs.

        Parameters
        ----------
        property_name : str
            Name of the property for reporting
        sim_column : str, optional
            Column name in simulation (defaults to property_name)
        obs_column : str, optional
            Column name in observations (defaults to property_name)

        Returns
        -------
        dict
            Comparison statistics
        """
        sim_col = sim_column or property_name
        obs_col = obs_column or property_name

        if sim_col not in self.sim.df.columns:
            print(f"Warning: {sim_col} not found in simulation catalogue")
            return None
        if obs_col not in self.obs.df.columns:
            print(f"Warning: {obs_col} not found in observation catalogue")
            return None

        sim_data = self.sim.df[sim_col].values
        obs_data = self.obs.df[obs_col].values

        return compare_distributions(sim_data, obs_data, property_name)