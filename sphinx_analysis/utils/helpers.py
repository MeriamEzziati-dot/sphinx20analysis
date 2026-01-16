"""
Utility functions for the SPHINX analysis package.
"""

import numpy as np


def setup_matplotlib_style():
    """Set up matplotlib style for consistent plotting."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings

    warnings.filterwarnings('ignore')
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")


def log_to_linear(log_value):
    """Convert log10 value to linear."""
    return 10 ** log_value


def linear_to_log(linear_value):
    """Convert linear value to log10."""
    return np.log10(linear_value)


def calculate_specific_sfr(sfr, stellar_mass):
    """
    Calculate specific star formation rate.

    Parameters
    ----------
    sfr : array-like
        Star formation rate in M☉/yr
    stellar_mass : array-like
        Stellar mass in M☉ (can be log or linear)

    Returns
    -------
    array-like
        Specific SFR in yr^-1
    """
    # If stellar mass looks like log values (typical range 7-11)
    if np.median(stellar_mass) < 15:
        stellar_mass = 10 ** stellar_mass

    return sfr / stellar_mass


def safe_log10(values, fill_value=-np.inf):
    """
    Safely take log10, handling zeros and negative values.

    Parameters
    ----------
    values : array-like
        Input values
    fill_value : float
        Value to use for non-positive numbers

    Returns
    -------
    array-like
        Log10 of values
    """
    result = np.full_like(values, fill_value, dtype=float)
    positive_mask = values > 0
    result[positive_mask] = np.log10(values[positive_mask])
    return result