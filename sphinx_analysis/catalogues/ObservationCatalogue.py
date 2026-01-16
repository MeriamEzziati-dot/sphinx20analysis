"""
Observation catalogue class for observational data (e.g., Flury et al.).
"""

import numpy as np
import pandas as pd
from .base import BaseCatalogue


class ObservationCatalogue(BaseCatalogue):
    """Catalogue for observational data."""

    def _preprocess(self):
        """Preprocess observation data."""
        # Add any observational data preprocessing here
        # For now, just pass through
        pass

    def has_column(self, column_name):
        """Check if a column exists in the catalogue."""
        return column_name in self.df.columns

    def get_valid_mask(self, *column_names):
        """
        Get mask for rows with valid (non-NaN) values in specified columns.

        Parameters
        ----------
        *column_names : str
            Names of columns to check

        Returns
        -------
        pd.Series (bool)
            Boolean mask
        """
        mask = pd.Series(True, index=self.df.index)
        for col in column_names:
            if col in self.df.columns:
                mask &= self.df[col].notna()
        return mask

    def summary(self):
        """Print summary statistics specific to observational data."""
        super().summary()

        # Check for common observational columns
        obs_columns = {
            'f_esc(LyC)-Hbeta': 'LyC escape fraction',
            'f_esc(LyA)': 'Lyα escape fraction',
            'EW(H1r_4861)': 'Hβ equivalent width',
            'log10(Mstar)': 'Stellar mass',
            'r_50_phys': 'Half-light radius'
        }

        print(f"\n--- Available Observables ---")
        for col, desc in obs_columns.items():
            if col in self.df.columns:
                n_valid = self.df[col].notna().sum()
                print(f"{desc}: {n_valid} measurements")