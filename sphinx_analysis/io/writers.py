"""
Input/output utilities for catalogues.
"""

import pandas as pd


def save_column_names(sim_catalogue, obs_catalogue, output_path):
    """
    Extract and save column names from both simulation and observation catalogues.

    Parameters
    ----------
    sim_catalogue : SimulationCatalogue
        Simulation catalogue
    obs_catalogue : ObservationCatalogue
        Observation catalogue
    output_path : str
        Path to save the column names
    """
    simu_columns = sim_catalogue.get_columns()
    obs_columns = obs_catalogue.get_columns()

    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SIMULATION FILE COLUMNS\n")
        f.write("=" * 60 + "\n\n")
        for i, col in enumerate(simu_columns, 1):
            f.write(f"{i}. {col}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("OBSERVATIONS FILE COLUMNS\n")
        f.write("=" * 60 + "\n\n")
        for i, col in enumerate(obs_columns, 1):
            f.write(f"{i}. {col}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Total simulation columns: {len(simu_columns)}\n")
        f.write(f"Total observation columns: {len(obs_columns)}\n")
        f.write("=" * 60 + "\n")

    print(f"Column names saved to {output_path}")
    return output_path


def export_summary(sim_catalogue, obs_catalogue, correlations_df, output_path):
    """
    Export text summary of analysis results.

    Parameters
    ----------
    sim_catalogue : SimulationCatalogue
        Simulation catalogue
    obs_catalogue : ObservationCatalogue
        Observation catalogue
    correlations_df : pd.DataFrame
        Correlation results
    output_path : str
        Path to save summary
    """
    with open(output_path, 'w') as f:
        f.write("SPHINX20 LyC Escape Fraction Analysis Summary\n")
        f.write("=" * 60 + "\n\n")

        # Simulation summary
        f.write("SIMULATION DATA\n")
        f.write("-" * 60 + "\n")
        f.write(f"Number of galaxies: {len(sim_catalogue)}\n")
        if 'redshift' in sim_catalogue.df.columns:
            redshifts = sim_catalogue.get_redshift_bins()
            f.write(f"Redshifts: {redshifts}\n")

        # Observation summary
        f.write("\n\nOBSERVATION DATA\n")
        f.write("-" * 60 + "\n")
        f.write(f"Number of objects: {len(obs_catalogue)}\n")

        # Correlations
        if correlations_df is not None and not correlations_df.empty:
            f.write("\n\nCORRELATIONS\n")
            f.write("-" * 60 + "\n")
            f.write(correlations_df.to_string())

    print(f"Summary exported to {output_path}")