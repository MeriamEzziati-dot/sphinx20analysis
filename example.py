#!/usr/bin/env python3
"""
Example usage of the SPHINX analysis package.

This script demonstrates how to use the refactored package to analyze
SPHINX20 simulation data and compare with observations.
"""


import sys

sys.path.insert(0, '/home/mezziati/')  # Adjust this path as needed

from sphinx_analysis import SimulationCatalogue, ObservationCatalogue, LyCEscapeAnalyzer
from sphinx_analysis.plotting import (
    plot_overview,
    plot_detailed_correlations,
    plot_fesc_histograms,
    plot_fesc_vs_stellar_mass
)
from sphinx_analysis.io import save_column_names, export_summary

# Set your data paths
DATA_PATH = '/home/mezziati/Documents/IAP/SPHINX20/data/'
OUTPUT_PATH = '/home/mezziati/Documents/IAP/SPHINX20/sphinx_analysis/outputs/'

def main():
    """Main analysis pipeline using the new package structure."""

    print("=" * 70)
    print("SPHINX20 LyC Escape Fraction Analysis")
    print("=" * 70)

    # 1. Load data using the new catalogue classes
    print("\n1. Loading catalogues...")
    sim_cat = SimulationCatalogue(DATA_PATH + 'all_basic_data.csv')
    obs_cat = ObservationCatalogue(DATA_PATH + 'flury.csv')
    obs_old = ObservationCatalogue(DATA_PATH + 'previous_Lyc_measurements.csv')

    # 2. Create analyzer
    print("\n2. Creating analyzer...")
    analyzer = LyCEscapeAnalyzer(sim_catalogue=sim_cat, obs_catalogue=obs_cat)

    # 3. Print summary statistics
    print("\n3. Summary statistics...")
    analyzer.summary_statistics()

    # 4. Analyze redshift evolution
    print("\n4. Redshift evolution...")
    z_evolution = analyzer.redshift_evolution()
    print(z_evolution)

    # 5. Calculate correlations
    print("\n5. Calculating correlations...")
    correlations = analyzer.correlations()
    print("\nCorrelation results:")
    print(correlations)

    # 6. Analyze high escapers
    print("\n6. Analyzing high escapers...")
    high_esc, low_esc = analyzer.analyze_high_escapers(threshold=0.1)

    # 7. Create plots
    print("\n7. Creating plots...")

    # Overview plot
    plot_overview(sim_cat, save_path=OUTPUT_PATH + 'overview.png')

    # Detailed correlations
    plot_detailed_correlations(sim_cat, save_path=OUTPUT_PATH + 'correlations.png')

    # Histogram comparison
    plot_fesc_histograms(sim_cat, obs_cat, save_path=OUTPUT_PATH + 'histograms.png')

    # Mass comparison
    plot_fesc_vs_stellar_mass(sim_cat, obs_cat, save_path=OUTPUT_PATH + 'mass_comparison.png')

    # 8. Export results
    print("\n8. Exporting results...")
    save_column_names(sim_cat, obs_cat, OUTPUT_PATH + 'column_names.txt')
    export_summary(sim_cat, obs_cat, correlations, OUTPUT_PATH + 'summary.txt')

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()