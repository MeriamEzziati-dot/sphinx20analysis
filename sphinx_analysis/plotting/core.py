"""
Core plotting functions for LyC escape fraction analysis.

This module contains the main plotting functions extracted from the original
SPHINXLyCAnalyzer class, now refactored to work with the new package structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_overview(sim_catalogue, save_path=None):
    """
    Create overview plots of LyC escape fractions from simulation data.

    Parameters
    ----------
    sim_catalogue : SimulationCatalogue
        Simulation catalogue
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    df = sim_catalogue.df

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SPHINX20 LyC Escape Fraction Overview',
                 fontsize=16, fontweight='bold')

    # 1. Distribution of f_esc
    ax = axes[0, 0]
    ax.hist(df['f_esc_linear'] * 100, bins=50,
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('f_esc (%)')
    ax.set_ylabel('Number of Galaxies')
    ax.set_title('Distribution of LyC Escape Fraction')
    ax.axvline(df['f_esc_linear'].mean() * 100,
               color='red', linestyle='--', label='Mean')
    ax.axvline(df['f_esc_linear'].median() * 100,
               color='orange', linestyle='--', label='Median')
    ax.legend()

    # 2. f_esc vs Stellar Mass
    ax = axes[0, 1]
    scatter = ax.scatter(df['stellar_mass'],
                         df['f_esc_linear'] * 100,
                         c=df['redshift'],
                         alpha=0.6, s=20, cmap='viridis')
    ax.set_xlabel('Stellar Mass (log M☉)')
    ax.set_ylabel('f_esc (%)')
    ax.set_title('f_esc vs Stellar Mass')
    ax.set_yscale('log')
    plt.colorbar(scatter, ax=ax, label='Redshift')

    # 3. f_esc vs SFR
    ax = axes[0, 2]
    mask = df['sfr_10'] > 0
    ax.scatter(df.loc[mask, 'sfr_10'],
               df.loc[mask, 'f_esc_linear'] * 100,
               alpha=0.6, s=20)
    ax.set_xlabel('SFR (10 Myr avg, M☉/yr)')
    ax.set_ylabel('f_esc (%)')
    ax.set_title('f_esc vs Star Formation Rate')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 4. f_esc vs Gas Metallicity
    ax = axes[1, 0]
    ax.scatter(df['gas_metallicity'],
               df['f_esc_linear'] * 100,
               alpha=0.6, s=20)
    ax.set_xlabel('Gas Metallicity (log Z☉)')
    ax.set_ylabel('f_esc (%)')
    ax.set_title('f_esc vs Gas Metallicity')
    ax.set_yscale('log')

    # 5. Redshift Evolution
    ax = axes[1, 1]
    redshifts = sorted(df['redshift'].unique())
    mean_fesc = [df[df['redshift'] == z]['f_esc_linear'].mean() * 100
                 for z in redshifts]
    median_fesc = [df[df['redshift'] == z]['f_esc_linear'].median() * 100
                   for z in redshifts]

    ax.plot(redshifts, mean_fesc, 'o-', label='Mean', markersize=8)
    ax.plot(redshifts, median_fesc, 's-', label='Median', markersize=8)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('f_esc (%)')
    ax.set_title('f_esc Evolution with Redshift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Directional variance
    if 'fesc_dir_std' in df.columns:
        ax = axes[1, 2]
        ax.scatter(df['fesc_dir_mean'] * 100,
                   df['fesc_dir_std'] * 100,
                   alpha=0.6, s=20)
        ax.set_xlabel('Mean Directional f_esc (%)')
        ax.set_ylabel('Std Dev of Directional f_esc (%)')
        ax.set_title('Anisotropy in LyC Escape')
        ax.plot([0, 100], [0, 0], 'k--', alpha=0.3)
    else:
        ax = axes[1, 2]
        ax.text(0.5, 0.5, 'Directional f_esc\ndata not available',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved overview plot to {save_path}")
    return fig


def plot_detailed_correlations(sim_catalogue, save_path=None):
    """
    Create detailed correlation plots for simulation data.

    Parameters
    ----------
    sim_catalogue : SimulationCatalogue
        Simulation catalogue
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    df = sim_catalogue.df

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SPHINX20: f_esc Correlations',
                 fontsize=16, fontweight='bold')

    # 1. f_esc vs Ionizing Luminosity
    ax = axes[0, 0]
    ax.scatter(df['ionizing_luminosity'],
               df['f_esc_linear'] * 100,
               alpha=0.5, s=20)
    ax.set_xlabel('Ionizing Luminosity (log phot/s)')
    ax.set_ylabel('f_esc (%)')
    ax.set_yscale('log')
    ax.set_title('f_esc vs Ionizing Luminosity')

    # Calculate and display correlation
    mask = ~(df['ionizing_luminosity'].isna() | df['f_esc'].isna())
    if mask.sum() > 10:
        corr, pval = stats.spearmanr(
            df.loc[mask, 'ionizing_luminosity'],
            df.loc[mask, 'f_esc']
        )
        ax.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {pval:.3e}',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. f_esc vs Stellar Age
    ax = axes[0, 1]
    mask = df['mean_stellar_age_mass'] > 0
    ax.scatter(df.loc[mask, 'mean_stellar_age_mass'],
               df.loc[mask, 'f_esc_linear'] * 100,
               alpha=0.5, s=20)
    ax.set_xlabel('Mean Stellar Age (Myr)')
    ax.set_ylabel('f_esc (%)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('f_esc vs Stellar Age')

    # 3. f_esc vs Halo Mass
    ax = axes[1, 0]
    ax.scatter(df['mvir'],
               df['f_esc_linear'] * 100,
               alpha=0.5, s=20)
    ax.set_xlabel('Halo Mass (log M☉)')
    ax.set_ylabel('f_esc (%)')
    ax.set_yscale('log')
    ax.set_title('f_esc vs Halo Mass')

    # 4. Stellar Mass vs Halo Mass colored by f_esc
    ax = axes[1, 1]
    scatter = ax.scatter(df['mvir'],
                         df['stellar_mass'],
                         c=np.log10(df['f_esc_linear'] * 100),
                         alpha=0.6, s=20, cmap='RdYlBu_r')
    ax.set_xlabel('Halo Mass (log M☉)')
    ax.set_ylabel('Stellar Mass (log M☉)')
    ax.set_title('Galaxy Mass Relation (colored by f_esc)')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log(f_esc %)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation plot to {save_path}")
    return fig


def plot_fesc_histograms(sim_catalogue, obs_catalogue, save_path=None):
    """
    Plot histograms comparing f_esc distributions in simulations and observations.

    Parameters
    ----------
    sim_catalogue : SimulationCatalogue
        Simulation catalogue
    obs_catalogue : ObservationCatalogue
        Observation catalogue
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Simulation data
    ax.hist(np.log10(sim_catalogue.df['f_esc_linear']),
            bins=30,
            alpha=0.5,
            label='SPHINX20 Simulations',
            color='blue',
            edgecolor='black')

    # Observation data
    if 'f_esc(LyC)-Hbeta' in obs_catalogue.df.columns:
        obs_fesc = obs_catalogue.df['f_esc(LyC)-Hbeta'].dropna()
        ax.hist(np.log10(obs_fesc),
                bins=15,
                alpha=0.5,
                label='Observations (Flury+21)',
                color='red',
                edgecolor='black')

    ax.set_xlabel('log₁₀(f_esc)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of LyC Escape Fractions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved histogram to {save_path}")
    return fig