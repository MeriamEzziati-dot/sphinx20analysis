"""
SPHINX20 LyC Escape Fraction Plotting Package
=============================================
A package for Plotting results of the analysis of the Lyman Continuum (LyC) escape fractions and their
correlations with galaxy properties in the SPHINX20 data release.
Date: Jan 2026
Author: Meriam Ezziati
"""

from .core import plot_overview, plot_detailed_correlations, plot_fesc_histograms
from .comparison import plot_fesc_vs_stellar_mass, plot_multiparameter_comparison

__version__ = '0.1.0'
__all__ = ['plot_overview', 'plot_detailed_correlations', 'plot_fesc_histograms','plot_fesc_vs_stellar_mass','plot_multiparameter_comparison']
