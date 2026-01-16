"""
Comparison plotting functions between simulations and observations.

This module contains plots that directly compare simulation predictions
with observational data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_fesc_vs_stellar_mass(sim_catalogue, obs_catalogue, save_path=None):
    """
    Compare f_esc vs stellar mass for simulations and observations.

    Parameters
    ----------
    sim_catalogue : SimulationCatalogue
        Simulation catalogue
    obs_catalogue : ObservationCatalogue
        Observation catalogue
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Simulation data
    ax = axes[0]
    scatter = ax.scatter(sim_catalogue.df['stellar_mass'],
                         sim_catalogue.df['f_esc'],
                         c=sim_catalogue.df['redshift'],
                         cmap='viridis',
                         alpha=0.6,
                         s=50)
    ax.set_xlabel('log₁₀(M* / M☉)', fontsize=12)
    ax.set_ylabel('log₁₀(f_esc)', fontsize=12)
    ax.set_title('SPHINX20 Simulations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Redshift', fontsize=11)

    # Observation data
    ax = axes[1]
    mask_obs = (obs_catalogue.df['log10(Mstar)'].notna() &
                obs_catalogue.df['f_esc(LyC)-Hbeta'].notna())

    obs_mass = obs_catalogue.df.loc[mask_obs, 'log10(Mstar)']
    obs_fesc = np.log10(obs_catalogue.df.loc[mask_obs, 'f_esc(LyC)-Hbeta'])

    ax.scatter(obs_mass, obs_fesc,
               c='red',
               alpha=0.6,
               s=100,
               marker='s',
               edgecolors='black',
               linewidth=0.5)
    ax.set_xlabel('log₁₀(M* / M☉)', fontsize=12)
    ax.set_ylabel('log₁₀(f_esc)', fontsize=12)
    ax.set_title('Observations (Flury+21)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_multiparameter_comparison(sim_catalogue, obs_catalogue, save_path=None):
    """
    Create a multi-panel comparison of key properties.

    Parameters
    ----------
    sim_catalogue : SimulationCatalogue
        Simulation catalogue
    obs_catalogue : ObservationCatalogue
        Observation catalogue
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('SPHINX20 vs Observations: Multi-Parameter Comparison',
                 fontsize=16, fontweight='bold')

    # Plot 1: f_esc vs M* (already defined above, simplified here)
    ax = axes[0, 0]
    ax.scatter(sim_catalogue.df['stellar_mass'],
               sim_catalogue.df['f_esc'],
               alpha=0.4, s=20, label='SPHINX20', color='blue')

    if 'log10(Mstar)' in obs_catalogue.df.columns:
        mask = obs_catalogue.df['f_esc(LyC)-Hbeta'].notna()
        ax.scatter(obs_catalogue.df.loc[mask, 'log10(Mstar)'],
                   np.log10(obs_catalogue.df.loc[mask, 'f_esc(LyC)-Hbeta']),
                   alpha=0.6, s=80, label='Observations',
                   color='red', marker='s', edgecolors='black', linewidth=0.5)

    ax.set_xlabel('log₁₀(M* / M☉)')
    ax.set_ylabel('log₁₀(f_esc)')
    ax.set_title('f_esc vs Stellar Mass')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Additional panels would go here
    # For now, showing the structure

    for i in range(1, 6):
        ax = axes.flatten()[i]
        ax.text(0.5, 0.5, f'Panel {i + 1}\n(add specific comparison)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.set_title(f'Comparison {i + 1}')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-parameter plot to {save_path}")
    plt.show()

