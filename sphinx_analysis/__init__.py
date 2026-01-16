"""
SPHINX20 LyC Escape Fraction Analysis Package
=============================================
A package for analyzing Lyman Continuum (LyC) escape fractions and their
correlations with galaxy properties in the SPHINX20 data release.
Date: Jan 2026
Author: Meriam Ezziati
"""

from .catalogues.base import BaseCatalogue
from .catalogues.SimulationCatalogue import SimulationCatalogue
from .catalogues.ObservationCatalogue import ObservationCatalogue
from .analyser.analyzer import LyCEscapeAnalyzer

__version__ = '0.1.0'
__all__ = ['BaseCatalogue', 'SimulationCatalogue', 'ObservationCatalogue', 'LyCEscapeAnalyzer']
