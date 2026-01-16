"""
Statistical comparison methods for catalogues.
"""

import numpy as np
import pandas as pd
from scipy import stats


def calculate_correlations(catalogue, target_column, properties_dict):
    """
    Calculate Spearman correlations between target and multiple properties.

    Parameters
    ----------
    catalogue : BaseCatalogue
        Catalogue containing the data
    target_column : str
        Name of target column (e.g., 'f_esc')
    properties_dict : dict
        Dictionary mapping column names to descriptive names

    Returns
    -------
    pd.DataFrame
        DataFrame with correlation results
    """
    print(f"\nSpearman correlation coefficients with {target_column}:")
    print("-" * 60)

    results = []
    for prop, name in properties_dict.items():
        if prop in catalogue.df.columns:
            # Remove NaN values
            mask = ~(catalogue.df[prop].isna() | catalogue.df[target_column].isna())
            if mask.sum() > 10:  # Need at least 10 points
                corr, pval = stats.spearmanr(
                    catalogue.df.loc[mask, prop],
                    catalogue.df.loc[mask, target_column]
                )
                results.append({
                    'Property': name,
                    'Correlation': corr,
                    'P-value': pval,
                    'Significant': 'Yes' if pval < 0.05 else 'No',
                    'N_points': mask.sum()
                })
                print(f"{name:30s}: r = {corr:6.3f}, p = {pval:.3e}, N = {mask.sum()}")

    return pd.DataFrame(results)


def redshift_evolution_summary(sim_catalogue, target_column='f_esc_linear'):
    """
    Analyze evolution of a property with redshift.

    Parameters
    ----------
    sim_catalogue : SimulationCatalogue
        Simulation catalogue
    target_column : str
        Column to analyze (default: 'f_esc_linear')

    Returns
    -------
    pd.DataFrame
        Summary statistics by redshift
    """
    print("\n" + "=" * 60)
    print("REDSHIFT EVOLUTION")
    print("=" * 60)

    redshifts = sim_catalogue.get_redshift_bins()
    print(f"\nAvailable redshifts: {redshifts}")

    results = []
    for z in redshifts:
        z_cat = sim_catalogue.get_at_redshift(z)
        if target_column in z_cat.df.columns:
            mean_val = z_cat.df[target_column].mean()
            median_val = z_cat.df[target_column].median()
            std_val = z_cat.df[target_column].std()

            # Convert to percentage if it's f_esc
            if 'f_esc' in target_column and 'linear' in target_column:
                mean_val *= 100
                median_val *= 100
                std_val *= 100
                unit = '%'
            else:
                unit = ''

            results.append({
                'Redshift': z,
                'N_galaxies': len(z_cat),
                'Mean': mean_val,
                'Median': median_val,
                'Std': std_val
            })

            print(f"z = {z:.1f}: N = {len(z_cat):4d}, "
                  f"<{target_column}> = {mean_val:.2f}{unit}, "
                  f"median = {median_val:.2f}{unit}")

    return pd.DataFrame(results)


def analyze_high_escapers(sim_catalogue, threshold=0.1):
    """
    Separate high and low escapers based on threshold.

    Parameters
    ----------
    sim_catalogue : SimulationCatalogue
        Simulation catalogue
    threshold : float
        Escape fraction threshold (default: 0.1 = 10%)

    Returns
    -------
    tuple
        (high_escapers, low_escapers) as SimulationCatalogue objects
    """
    if 'f_esc_linear' not in sim_catalogue.df.columns:
        raise KeyError("f_esc_linear column not found in catalogue")

    high_mask = sim_catalogue.df['f_esc_linear'] > threshold
    low_mask = ~high_mask

    high_esc = sim_catalogue.filter(high_mask)
    low_esc = sim_catalogue.filter(low_mask)

    print(f"\n{'=' * 60}")
    print(f"HIGH vs LOW ESCAPERS (threshold = {threshold * 100:.0f}%)")
    print(f"{'=' * 60}")
    print(f"High escapers: {len(high_esc)} ({len(high_esc) / len(sim_catalogue) * 100:.1f}%)")
    print(f"Low escapers: {len(low_esc)} ({len(low_esc) / len(sim_catalogue) * 100:.1f}%)")

    return high_esc, low_esc


def compare_distributions(sim_data, obs_data, column_name):
    """
    Compare distributions between simulation and observations using KS test.

    Parameters
    ----------
    sim_data : array-like
        Simulation data
    obs_data : array-like
        Observation data
    column_name : str
        Name of the property being compared

    Returns
    -------
    dict
        Dictionary with test statistics
    """
    # Remove NaN values
    sim_clean = sim_data[~np.isnan(sim_data)]
    obs_clean = obs_data[~np.isnan(obs_data)]

    if len(sim_clean) < 3 or len(obs_clean) < 3:
        return None

    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(sim_clean, obs_clean)

    # Anderson-Darling test
    try:
        ad_result = stats.anderson_ksamp([sim_clean, obs_clean])
        ad_stat = ad_result.statistic
        ad_pval = ad_result.pvalue
    except:
        ad_stat, ad_pval = np.nan, np.nan

    results = {
        'property': column_name,
        'n_sim': len(sim_clean),
        'n_obs': len(obs_clean),
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'ad_statistic': ad_stat,
        'ad_pvalue': ad_pval,
        'distributions_match': ks_pval > 0.05
    }

    return results