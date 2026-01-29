#!/usr/bin/env python3
"""
Demo: Parameter Availability Report
====================================
Shows how to generate a comprehensive parameter availability report.
"""

from sphinx_analysis.catalogues.CatalogueManager import CatalogueManager

output_dir =  '/home/mezziati/Documents/IAP/SPHINX20/sphinx_analysis/outputs/'


data = CatalogueManager(sim_catalogue = '/home/mezziati/Documents/IAP/SPHINX20/data/all_basic_data.csv',
                        obs_catalogue = '/home/mezziati/Documents/IAP/SPHINX20/data/flury.csv'
)

# Generate and save parameter availability report
# This creates a detailed file showing which parameters exist in which datasets
output_file = data.save_parameter_availability(output_dir + 'parameter_availability.txt')

print(f"\n✓ Report saved to: {output_file}")
print("\nThe report includes:")
print("  • Complete list of all available parameters")
print("  • Which parameters exist in BOTH datasets")
print("  • Which parameters exist in SIMULATIONS only")
print("  • Which parameters exist in OBSERVATIONS only")
print("  • Column names in each dataset")
print("  • Usage examples")