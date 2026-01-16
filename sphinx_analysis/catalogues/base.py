"""
Base catalogue class defining common interface for all catalogues.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class MERMosaicError(Exception):
    """Custom exception for catalogue operations."""
    pass


class BaseCatalogue(ABC):
    """Abstract base class for all catalogue types."""

    def __init__(self, filepath=None, data=None):
        """
        Initialize catalogue from file or DataFrame.

        Parameters
        ----------
        filepath : str, optional
            Path to CSV file containing catalogue data
        data : pd.DataFrame, optional
            DataFrame containing catalogue data
        """
        #TODO: make it include all catalogues file formats
        if filepath is not None:
            self.df = pd.read_csv(filepath)
            self.filepath = filepath
        elif data is not None:
            self.df = data
            self.filepath = None
        else:
            raise ValueError("Must provide either filepath or data")

        self._preprocess()

    @abstractmethod
    def _preprocess(self):
        """Preprocess data (convert units, calculate derived quantities, etc.)"""
        pass

    def filter(self, condition):
        """
        Filter catalogue based on a condition.

        Parameters
        ----------
        condition : pd.Series (bool)
            Boolean mask for filtering

        Returns
        -------
        BaseCatalogue
            New catalogue instance with filtered data
        """
        return self.__class__(data=self.df[condition].copy())

    def get_column(self, column_name):
        """
        Get a column from the catalogue.

        Parameters
        ----------
        column_name : str
            Name of column to retrieve

        Returns
        -------
        pd.Series
            Column data
        """
        if column_name not in self.df.columns:
            raise KeyError(f"Column '{column_name}' not found in catalogue")
        return self.df[column_name]

    def get_columns(self):
        """Return list of all column names."""
        return self.df.columns.tolist()

    def __len__(self):
        """Return number of objects in catalogue."""
        return len(self.df)

    def __repr__(self):
        """String representation of catalogue."""
        return f"{self.__class__.__name__}(n_objects={len(self)})"

    def summary(self):
        """Print summary statistics of the catalogue."""
        print(f"\n{'=' * 60}")
        print(f"{self.__class__.__name__} Summary")
        print(f"{'=' * 60}")
        print(f"Number of objects: {len(self)}")
        if 'redshift' in self.df.columns:
            print(f"Redshift range: {self.df['redshift'].min():.2f} - {self.df['redshift'].max():.2f}")