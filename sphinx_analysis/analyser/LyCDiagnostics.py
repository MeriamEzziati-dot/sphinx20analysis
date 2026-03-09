#!/usr/bin/env python3
"""
LyC Diagnostics
===============
Diagnostic and plotting tools for SPHINX20 vs LzLCS comparison.
Uses LyCDataManager for all data access.

Author: Meriam Ezziati
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union
from ..catalogues.CatalogueManager import CatalogueManager

class LyCDiagnosticsError(Exception):
    pass


class LyCDiagnostics:
    """
    Diagnostic and visualization tools for LyC escape fraction analysis.
    Fetches data from LyCDataManager and creates comparison plots.
    """

    def __init__(self, data_manager: CatalogueManager):
        """
        Initialize diagnostics with a data manager.

        Parameters:
        -----------
        data_manager : LyCDataManager
            Initialized data manager instance
        """
        self.data = data_manager
        print("✓ LyCDiagnostics initialized with DataManager")

    def plot_2d_comparison(self, x_param: str, y_param: str,
                           color_param: Optional[str] = None,
                           direction: Optional[Union[int, str]] = None,
                           figsize=(10, 8),
                           log_x=False,
                           log_y=False,
                           xlim=None,
                           ylim=None,
                           save_path=None):
        """
        Create 2D scatter plot comparing simulations and observations.

        Parameters:
        -----------
        x_param : str
            Parameter for x-axis
        y_param : str
            Parameter for y-axis
        color_param : str, optional
            Parameter for color coding (if None, uses single color)
        direction : int or str, optional
            For directional parameters
        figsize : tuple
            Figure size
        log_x, log_y : bool
            Use log scale for axes
        xlim, ylim : tuple, optional
            Axis limits
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        print("\n" + "=" * 70)
        if color_param:
            print(f"{x_param} vs {y_param} (colored by {color_param})")
        else:
            print(f"{x_param} vs {y_param}")
        print("=" * 70)

        # Extract parameters
        try:
            x_sim, x_obs = self.data.get_parameter(x_param, 'both', direction=direction)
            y_sim, y_obs = self.data.get_parameter(y_param, 'both', direction=direction)

            if color_param:
                c_sim, c_obs = self.data.get_parameter(color_param, 'both', direction=direction)
            else:
                c_sim, c_obs = None, None

        except Exception as e:
            print(f"✗ Error extracting parameters: {e}")
            return

        # Create masks for valid data
        if c_sim is not None:
            mask_sim = (x_sim.notna() & y_sim.notna() & c_sim.notna())
        else:
            mask_sim = (x_sim.notna() & y_sim.notna())

        if x_obs is not None and y_obs is not None:
            if c_obs is not None:
                mask_obs = (x_obs.notna() & y_obs.notna() & c_obs.notna())
            else:
                mask_obs = (x_obs.notna() & y_obs.notna())
        else:
            mask_obs = None

        print(f"  • Simulations: {mask_sim.sum()} valid points")
        if mask_obs is not None:
            print(f"  • Observations: {mask_obs.sum()} valid points")

        # Plot simulations
        if color_param and c_sim is not None:
            scatter_sim = ax.scatter(
                x_sim[mask_sim], y_sim[mask_sim],
                c=c_sim[mask_sim],
                cmap='viridis',
                alpha=0.6,
                s=50,
                label='SPHINX20',
                edgecolors='none'
            )
            cbar = plt.colorbar(scatter_sim, ax=ax)
            cbar.set_label(color_param, fontsize=12)
        else:
            ax.scatter(
                x_sim[mask_sim], y_sim[mask_sim],
                alpha=0.6,
                s=50,
                color='steelblue',
                label='SPHINX20',
                edgecolors='none'
            )

        # Plot observations if available
        if mask_obs is not None and mask_obs.any():
            if color_param and c_obs is not None:
                ax.scatter(
                    x_obs[mask_obs], y_obs[mask_obs],
                    c=c_obs[mask_obs],
                    cmap='Reds',
                    alpha=0.8,
                    s=150,
                    marker='s',
                    edgecolors='black',
                    linewidth=1,
                    label='LzLCS',
                    zorder=5
                )
            else:
                ax.scatter(
                    x_obs[mask_obs], y_obs[mask_obs],
                    alpha=0.8,
                    s=150,
                    color='red',
                    marker='s',
                    edgecolors='black',
                    linewidth=1,
                    label='LzLCS',
                    zorder=5
                )

        # Formatting
        ax.set_xlabel(x_param, fontsize=14)
        ax.set_ylabel(y_param, fontsize=14)

        title = f'{x_param} vs {y_param}'
        if color_param:
            title += f' (colored by {color_param})'
        ax.set_title(title, fontsize=15, fontweight='bold')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=12, frameon=True)

        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Figure saved to {save_path}")

        plt.show()
        print("=" * 70 + "\n")

    def plot_2d_all_directions(self, x_param: str, y_param: str,
                               color_param: Optional[str] = None,
                               figsize=(10, 8),
                               log_x=False,
                               log_y=False,
                               xlim=None,
                               ylim=None,
                               save_path=None):
        """
        Create 2D scatter plot where each direction is treated as a separate point.
        Instead of averaging over directions, this plots all 10 directions as individual points.

        If a parameter has directions (e.g., f_esc_dir_0 to f_esc_dir_9), each direction
        becomes one point. If a parameter has no directions (e.g., stellar_mass), it's
        replicated 10 times to match the directional parameter.

        Parameters:
        -----------
        x_param : str
            Parameter for x-axis
        y_param : str
            Parameter for y-axis
        color_param : str, optional
            Parameter for color coding
        figsize : tuple
            Figure size
        log_x, log_y : bool
            Use log scale for axes
        xlim, ylim : tuple, optional
            Axis limits
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        print("\n" + "=" * 70)
        if color_param:
            print(f"{x_param} vs {y_param} (colored by {color_param}) - ALL DIRECTIONS")
        else:
            print(f"{x_param} vs {y_param} - ALL DIRECTIONS")
        print("=" * 70)

        # Collect all directional data
        x_all = []
        y_all = []
        c_all = []

        try:
            # Try to get all 10 directions for each parameter
            for direction in range(10):
                try:
                    # Get x data for this direction
                    x_sim_dir = self.data.get_parameter(x_param, 'sim', direction=direction)
                    if x_sim_dir is None:
                        # Parameter has no directions, get the base value
                        x_sim_dir = self.data.get_parameter(x_param, 'sim')

                    # Get y data for this direction
                    y_sim_dir = self.data.get_parameter(y_param, 'sim', direction=direction)
                    if y_sim_dir is None:
                        # Parameter has no directions, get the base value
                        y_sim_dir = self.data.get_parameter(y_param, 'sim')

                    # Get color data for this direction if specified
                    if color_param:
                        c_sim_dir = self.data.get_parameter(color_param, 'sim', direction=direction)
                        if c_sim_dir is None:
                            # Parameter has no directions, get the base value
                            c_sim_dir = self.data.get_parameter(color_param, 'sim')
                    else:
                        c_sim_dir = None

                    # Append to lists
                    if x_sim_dir is not None and y_sim_dir is not None:
                        x_all.append(x_sim_dir)
                        y_all.append(y_sim_dir)
                        if c_sim_dir is not None:
                            c_all.append(c_sim_dir)

                except Exception as e:
                    print(f"  ⚠ Warning: Could not get direction {direction}: {e}")
                    continue

            if not x_all or not y_all:
                print("  ✗ Error: No valid data found")
                return

            # Concatenate all directions
            x_data = pd.concat(x_all, ignore_index=True)
            y_data = pd.concat(y_all, ignore_index=True)

            if c_all:
                c_data = pd.concat(c_all, ignore_index=True)
            else:
                c_data = None

            # Create mask for valid data
            if c_data is not None:
                mask = (x_data.notna() & y_data.notna() & c_data.notna())
            else:
                mask = (x_data.notna() & y_data.notna())

            print(f"  • Total points across all directions: {mask.sum()}")
            print(f"  • Points per direction (avg): {mask.sum() / 10:.1f}")

            # Plot
            if color_param and c_data is not None:
                scatter = ax.scatter(
                    x_data[mask], y_data[mask],
                    c=(c_data[mask]),
                    cmap='viridis',
                    alpha=0.5,
                    s=30,
                    edgecolors='none'
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_param, fontsize=12)
            else:
                ax.scatter(
                    x_data[mask], y_data[mask],
                    alpha=0.5,
                    s=30,
                    color='steelblue',
                    edgecolors='none'
                )

            # Formatting
            ax.set_xlabel(x_param, fontsize=14)
            ax.set_ylabel(y_param, fontsize=14)

            title = f'{x_param} vs {y_param} - All Directions'
            if color_param:
                title += f' (colored by {color_param})'
            ax.set_title(title, fontsize=15, fontweight='bold')

            ax.grid(True, alpha=0.3)

            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')

            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  ✓ Figure saved to {save_path}")

            plt.show()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return

        print("=" * 70 + "\n")

    def plot_directional_comparison(self, x_param: str, y_param: str,
                                    figsize=(12, 8), save_path=None):
        """
        Plot comparison across all 10 viewing directions.

        Parameters:
        -----------
        x_param : str
            Parameter for x-axis (must support directional extraction)
        y_param : str
            Parameter for y-axis (must support directional extraction)
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 5, figsize=figsize, sharex=True, sharey=True)
        axes = axes.flatten()

        print("\n" + "=" * 70)
        print(f"DIRECTIONAL COMPARISON: {x_param} vs {y_param}")
        print("=" * 70)

        for direction in range(10):
            ax = axes[direction]

            try:
                x_data = self.data.get_parameter(x_param, 'sim', direction=direction)
                y_data = self.data.get_parameter(y_param, 'sim', direction=direction)

                mask = (x_data.notna() & y_data.notna())

                ax.scatter(x_data[mask], y_data[mask],
                           alpha=0.5, s=20, color='steelblue')
                ax.set_title(f'Direction {direction}', fontsize=10)
                ax.grid(True, alpha=0.3)

            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:20]}',
                        ha='center', va='center', transform=ax.transAxes)

        fig.text(0.5, 0.04, x_param, ha='center', fontsize=14)
        fig.text(0.04, 0.5, y_param, va='center', rotation='vertical', fontsize=14)
        fig.suptitle(f'Directional Analysis: {x_param} vs {y_param}',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Figure saved to {save_path}")

        plt.show()
        print("=" * 70 + "\n")

    def plot_histogram_comparison(self, param: str,
                                  bins=30,
                                  log_scale=False,
                                  figsize=(10, 6),
                                  save_path=None):
        """
        Plot histogram comparison between simulations and observations.

        Parameters:
        -----------
        param : str
            Parameter to plot
        bins : int
            Number of bins
        log_scale : bool
            Use log scale for x-axis
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        print("\n" + "=" * 70)
        print(f"HISTOGRAM: {param}")
        print("=" * 70)

        try:
            sim_data, obs_data = self.data.get_parameter(param, 'both')

            # Plot simulations
            sim_valid = sim_data.dropna()
            ax.hist(sim_valid, bins=bins, alpha=0.6,
                    label=f'SPHINX20 (n={len(sim_valid)})',
                    color='steelblue', edgecolor='black', linewidth=0.5)

            # Plot observations if available
            if obs_data is not None:
                obs_valid = obs_data.dropna()
                if len(obs_valid) > 0:
                    ax.hist(obs_valid, bins=bins, alpha=0.6,
                            label=f'LzLCS (n={len(obs_valid)})',
                            color='red', edgecolor='black', linewidth=0.5)

            ax.set_xlabel(param, fontsize=14)
            ax.set_ylabel('Number of Galaxies', fontsize=14)
            ax.set_title(f'Distribution of {param}', fontsize=15, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            if log_scale:
                ax.set_xscale('log')

            # Add statistics
            print(f"  • Simulations: mean={sim_valid.mean():.3f}, median={sim_valid.median():.3f}")
            if obs_data is not None and len(obs_valid) > 0:
                print(f"  • Observations: mean={obs_valid.mean():.3f}, median={obs_valid.median():.3f}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Figure saved to {save_path}")

        plt.show()
        print("=" * 70 + "\n")

    def plot_bpt_diagram(self, figsize=(10, 8), save_path=None):
        """
        Plot BPT (Baldwin-Phillips-Terlevich) diagram.
        """
        fig, ax = plt.subplots(figsize=figsize)

        print("\n" + "=" * 70)
        print("BPT DIAGRAM")
        print("=" * 70)

        try:
            # Get BPT coordinates
            x_sim, x_obs = self.data.get_parameter('log_NII_Ha', 'both')
            y_sim, y_obs = self.data.get_parameter('log_OIII_Hb', 'both')

            # Simulations
            mask_sim = (x_sim.notna() & y_sim.notna())
            ax.scatter(x_sim[mask_sim], y_sim[mask_sim],
                       alpha=0.5, s=30, color='steelblue', label='SPHINX20')

            # Observations
            if x_obs is not None and y_obs is not None:
                mask_obs = (x_obs.notna() & y_obs.notna())
                if mask_obs.any():
                    ax.scatter(x_obs[mask_obs], y_obs[mask_obs],
                               alpha=0.8, s=150, color='red', marker='s',
                               edgecolors='black', linewidth=1, label='LzLCS', zorder=5)

            # Add demarcation lines
            # Kauffmann+03 line
            x_line = np.linspace(-2, 0.05, 100)
            y_kauff = 0.61 / (x_line - 0.05) + 1.3
            ax.plot(x_line, y_kauff, 'k--', linewidth=2, label='Kauffmann+03', zorder=10)

            # Kewley+01 line
            y_kew = 0.61 / (x_line - 0.47) + 1.19
            ax.plot(x_line, y_kew, 'k-', linewidth=2, label='Kewley+01', zorder=10)

            ax.set_xlabel('log([NII]λ6583/Hα)', fontsize=14)
            ax.set_ylabel('log([OIII]λ5007/Hβ)', fontsize=14)
            ax.set_title('BPT Diagram', fontsize=16, fontweight='bold')
            ax.set_xlim(-2, 0.5)
            ax.set_ylim(-1.5, 1.5)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            print(f"  • Simulations: {mask_sim.sum()} galaxies")
            if x_obs is not None:
                print(f"  • Observations: {mask_obs.sum()} galaxies")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Figure saved to {save_path}")

        plt.show()
        print("=" * 70 + "\n")

    def plot_multiparameter_grid(self, params: list,
                                 figsize=(15, 15),
                                 save_path=None):
        """
        Create grid of scatter plots for multiple parameter combinations.

        Parameters:
        -----------
        params : list
            List of parameter names to include in grid
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        n_params = len(params)
        fig, axes = plt.subplots(n_params, n_params, figsize=figsize)

        print("\n" + "=" * 70)
        print(f"MULTIPARAMETER GRID: {n_params}x{n_params}")
        print("=" * 70)

        for i, param_y in enumerate(params):
            for j, param_x in enumerate(params):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histograms
                    try:
                        sim_data = self.data.get_parameter(param_x, 'sim')
                        if sim_data is not None:
                            sim_valid = sim_data.dropna()
                            ax.hist(sim_valid, bins=20, alpha=0.7, color='steelblue')
                            ax.set_yticks([])
                    except:
                        pass

                elif i > j:
                    # Lower triangle: scatter plots
                    try:
                        x_sim, x_obs = self.data.get_parameter(param_x, 'both')
                        y_sim, y_obs = self.data.get_parameter(param_y, 'both')

                        mask_sim = (x_sim.notna() & y_sim.notna())
                        ax.scatter(x_sim[mask_sim], y_sim[mask_sim],
                                   alpha=0.3, s=10, color='steelblue')

                        if x_obs is not None and y_obs is not None:
                            mask_obs = (x_obs.notna() & y_obs.notna())
                            if mask_obs.any():
                                ax.scatter(x_obs[mask_obs], y_obs[mask_obs],
                                           alpha=0.8, s=50, color='red', marker='s',
                                           edgecolors='black', linewidth=0.5, zorder=5)
                    except:
                        pass

                else:
                    # Upper triangle: empty or could add correlation coefficients
                    ax.axis('off')

                # Labels
                if i == n_params - 1:
                    ax.set_xlabel(param_x, fontsize=9)
                else:
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(param_y, fontsize=9)
                else:
                    ax.set_yticklabels([])

                ax.grid(True, alpha=0.2)

        plt.suptitle('Multiparameter Comparison Grid', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Figure saved to {save_path}")

        plt.show()
        print("=" * 70 + "\n")

    def correlations_summary(self, param_list: list, reference_param: str = 'f_esc'):
        """
        Calculate and display correlations between a reference parameter and others.

        Parameters:
        -----------
        param_list : list
            List of parameters to correlate with reference
        reference_param : str
            Reference parameter (default: f_esc)
        """
        from scipy.stats import spearmanr, pearsonr

        print("\n" + "=" * 70)
        print(f"CORRELATIONS WITH {reference_param}")
        print("=" * 70)

        ref_sim = self.data.get_parameter(reference_param, 'sim')

        results = []
        for param in param_list:
            try:
                param_sim = self.data.get_parameter(param, 'sim')
                mask = (ref_sim.notna() & param_sim.notna())

                if mask.sum() > 10:
                    r_spearman, p_spearman = spearmanr(ref_sim[mask], param_sim[mask])
                    r_pearson, p_pearson = pearsonr(ref_sim[mask], param_sim[mask])

                    results.append({
                        'Parameter': param,
                        'N': mask.sum(),
                        'Spearman r': r_spearman,
                        'Spearman p': p_spearman,
                        'Pearson r': r_pearson,
                        'Pearson p': p_pearson,
                        'Significant': 'Yes' if p_spearman < 0.05 else 'No'
                    })
            except Exception as e:
                print(f"  ✗ Error with {param}: {e}")

        df_results = pd.DataFrame(results)
        print(f"\n{df_results.to_string(index=False)}")
        print("=" * 70 + "\n")

        return df_results

    def plot_2d_all_directions_all(
            self,
            x_param: str,
            y_param: str,
            color_param=None,
            obs_threshold: float = 0.1,
            figsize=(10, 8),
            log_x: bool = False,
            log_y: bool = False,
            xlim=None,
            ylim=None,
            save_path=None,
            extra_obs_paths: dict = None,
    ):
        """
        Create 2D scatter plot where each direction is treated as a separate point.

        Overlays additional observational catalogues when their paths are supplied.

        Parameters
        ----------
        x_param, y_param : str
            Parameters for x and y axes.
        color_param : str, optional
            Parameter for color coding simulations; drives filled/hollow split
            for LzLCS observations.
        obs_threshold : float
            Threshold on color_param for LzLCS filled vs hollow markers.
        figsize, log_x, log_y, xlim, ylim, save_path : same as before.
        extra_obs_paths : dict, optional
            Paths to the additional observational tables. Keys: 'sxds', 'bassett',
            'laces', 'highz'. Any key can be omitted; that catalogue is skipped.

            Example::

                extra_obs_paths={
                    'sxds':    'path/to/SXDS.csv',
                    'bassett': 'path/to/bassett.csv',
                    'laces':   'path/to/LeakersLACES.csv',
                    'highz':   'path/to/misc_highz.csv',
                }
        """
        fig, ax = plt.subplots(figsize=figsize)

        print("\n" + "=" * 70)
        if color_param:
            print(f"{x_param} vs {y_param} (colored by {color_param}) - ALL DIRECTIONS")
        else:
            print(f"{x_param} vs {y_param} - ALL DIRECTIONS")
        print("=" * 70)

        # ------------------------------------------------------------------
        # 1. Simulation data across all directions
        # ------------------------------------------------------------------
        x_sim_all, y_sim_all, c_sim_all = [], [], []

        try:
            for direction in range(10):
                try:
                    x_sim_dir = self.data.get_parameter(x_param, 'sim', direction=direction)
                    if x_sim_dir is None:
                        x_sim_dir = self.data.get_parameter(x_param, 'sim')

                    y_sim_dir = self.data.get_parameter(y_param, 'sim', direction=direction)
                    if y_sim_dir is None:
                        y_sim_dir = self.data.get_parameter(y_param, 'sim')

                    if color_param:
                        c_sim_dir = self.data.get_parameter(color_param, 'sim', direction=direction)
                        if c_sim_dir is None:
                            c_sim_dir = self.data.get_parameter(color_param, 'sim')
                    else:
                        c_sim_dir = None

                    if x_sim_dir is not None and y_sim_dir is not None:
                        x_sim_all.append(x_sim_dir)
                        y_sim_all.append(y_sim_dir)
                        if c_sim_dir is not None:
                            c_sim_all.append(c_sim_dir)

                except Exception as e:
                    print(f"  ⚠ Warning: Could not get direction {direction}: {e}")
                    continue

            if not x_sim_all or not y_sim_all:
                print("  ✗ Error: No valid simulation data found")
                return

            x_sim = pd.concat(x_sim_all, ignore_index=True)
            y_sim = pd.concat(y_sim_all, ignore_index=True)
            c_sim = pd.concat(c_sim_all, ignore_index=True) if c_sim_all else None

            mask_sim = x_sim.notna() & y_sim.notna()
            if c_sim is not None:
                mask_sim &= c_sim.notna()

            print(f"  • Simulation points across all directions: {mask_sim.sum()}")
            print(f"  • Points per direction (avg): {mask_sim.sum() / 10:.1f}")

            # ------------------------------------------------------------------
            # 2. Plot simulations
            # ------------------------------------------------------------------
            if color_param and c_sim is not None:
                scatter_sim = ax.scatter(
                    x_sim[mask_sim], y_sim[mask_sim],
                    c=c_sim[mask_sim],
                    cmap='viridis', alpha=0.5, s=30,
                    edgecolors='none', label='SPHINX20', zorder=2,
                )
                cbar = plt.colorbar(scatter_sim, ax=ax)
                cbar.set_label(color_param, fontsize=12)
            else:
                ax.scatter(
                    x_sim[mask_sim], y_sim[mask_sim],
                    alpha=0.5, s=30, color='steelblue',
                    edgecolors='none', label='SPHINX20', zorder=2,
                )

            # ------------------------------------------------------------------
            # 3. LzLCS observations
            # ------------------------------------------------------------------
            try:
                _, x_obs = self.data.get_parameter(x_param, 'both')
                _, y_obs = self.data.get_parameter(y_param, 'both')
                _, c_obs = (self.data.get_parameter(color_param, 'both')
                            if color_param else (None, None))
            except Exception:
                x_obs = y_obs = c_obs = None

            has_obs = (x_obs is not None and y_obs is not None
                       and len(x_obs) > 0 and len(y_obs) > 0)

            if has_obs:
                if color_param and c_obs is not None:
                    mask_obs_base = x_obs.notna() & y_obs.notna() & c_obs.notna()
                    above = mask_obs_base & (c_obs >= np.log10(obs_threshold))
                    below = mask_obs_base & (c_obs < np.log10(obs_threshold))

                    print(f"  • LzLCS above threshold ({obs_threshold}): {above.sum()}")
                    print(f"  • LzLCS below threshold ({obs_threshold}): {below.sum()}")

                    obs_kw = dict(s=150, marker='s', linewidth=1.2, zorder=5)
                    if above.any():
                        ax.scatter(x_obs[above], y_obs[above],
                                   c=c_obs[above], cmap='Reds', alpha=0.9,
                                   edgecolors='black',
                                   label=f'LzLCS (≥{obs_threshold})', **obs_kw)
                    if below.any():
                        ax.scatter(x_obs[below], y_obs[below],
                                   cmap='Reds', alpha=0.9,
                                   edgecolors='black', facecolors='none',
                                   label=f'LzLCS (<{obs_threshold})', **obs_kw)
                else:
                    mask_obs_base = x_obs.notna() & y_obs.notna()
                    print(f"  • LzLCS: {mask_obs_base.sum()} points")
                    ax.scatter(x_obs[mask_obs_base], y_obs[mask_obs_base],
                               alpha=0.9, s=150, color='red', marker='s',
                               edgecolors='black', linewidth=1.2,
                               label='LzLCS', zorder=5)
            else:
                print("  • LzLCS: not available for this parameter combination")

            # ------------------------------------------------------------------
            # 4. Additional observational catalogues
            # ------------------------------------------------------------------
            if extra_obs_paths:
                x_key = _PARAM_MAP.get(x_param)
                y_key = _PARAM_MAP.get(y_param)

                if x_key is None or y_key is None:
                    print(f"  ⚠ Extra catalogues: no param mapping for "
                          f"'{x_param}' or '{y_param}' – skipping.")
                else:
                    for cat_key, path in extra_obs_paths.items():
                        if cat_key not in _EXTRA_CATALOGS:
                            print(f"  ⚠ Unknown catalogue key '{cat_key}' – skipping.")
                            continue

                        style = _EXTRA_CATALOGS[cat_key]
                        loader = style['loader']

                        try:
                            data = loader(path)
                        except Exception as e:
                            print(f"  ⚠ Could not load '{cat_key}' from {path}: {e}")
                            continue

                        x_cat = data.get(x_key)
                        y_cat = data.get(y_key)
                        z_cat = data.get('redshift')

                        if x_cat is None or y_cat is None:
                            print(f"  • {style['label']}: missing '{x_key}' or "
                                  f"'{y_key}' – skipped.")
                            continue

                        mask = x_cat.notna() & y_cat.notna()
                        n_valid = int(mask.sum())

                        if n_valid == 0:
                            print(f"  • {style['label']}: no valid points for "
                                  f"({x_key}, {y_key}) – skipped.")
                            continue

                        print(f"  • {style['label']}: {n_valid} valid points")

                        # For misc_highz: give special marker to z~10 source
                        if cat_key == 'highz' and z_cat is not None:
                            is_highz  = mask & (z_cat >= _HIGHZ_THRESHOLD)
                            is_normal = mask & (z_cat < _HIGHZ_THRESHOLD)

                            if is_normal.any():
                                ax.scatter(
                                    x_cat[is_normal], y_cat[is_normal],
                                    marker=style['marker'], color=style['color'],
                                    s=style['size'], edgecolors='black',
                                    linewidth=0.8, alpha=0.9,
                                    label=f"{style['label']}+21,22", zorder=style['zorder'],
                                )
                            if is_highz.any():
                                ax.scatter(
                                    x_cat[is_highz], y_cat[is_highz],
                                    label=f"{style['label']}+26 (z~10.2)",
                                    **_HIGHZ_SPECIAL_STYLE,
                                )
                                print(f"    ↳ {int(is_highz.sum())} source(s) at "
                                      f"z≥{_HIGHZ_THRESHOLD} plotted with special marker")
                        elif cat_key == 'bassett':
                            # Filled triangle: f_esc >= 0.1
                            # Hollow triangle: f_esc < 0.1 or f_esc not measured
                            f_esc_cat = data.get('f_esc')
                            if f_esc_cat is not None:
                                filled = mask & f_esc_cat.notna() & (f_esc_cat >= 0.1)
                                hollow = mask & (f_esc_cat.isna() | (f_esc_cat < 0.1))
                            else:
                                # No f_esc column at all – all hollow
                                filled = mask & False
                                hollow = mask

                            n_filled = int(filled.sum())
                            n_hollow = int(hollow.sum())
                            print(f"  • {style['label']}: {n_filled} filled, "
                                  f"{n_hollow} hollow (f_esc<0.1 or missing)")

                            common_kw = dict(
                                marker=style['marker'], s=style['size'],
                                linewidth=0.8, alpha=0.9, zorder=style['zorder'],
                            )
                            if filled.any():
                                ax.scatter(
                                    x_cat[filled], y_cat[filled],
                                    color=style['color'], edgecolors='black',
                                    label=style['label'],
                                    **common_kw,
                                )
                            if hollow.any():
                                ax.scatter(
                                    x_cat[hollow], y_cat[hollow],
                                    facecolors='none', edgecolors=style['color'],
                                    label=f"{style['label']} ($f_{{esc}}<0.1$)",
                                    **common_kw,
                                )
                        else:
                            ax.scatter(
                                x_cat[mask], y_cat[mask],
                                marker=style['marker'], color=style['color'],
                                s=style['size'], edgecolors='black',
                                linewidth=0.8, alpha=0.9,
                                label=style['label'], zorder=style['zorder'],
                            )

            # ------------------------------------------------------------------
            # 5. Formatting
            # ------------------------------------------------------------------
            # Parameters that are stored/plotted as log10 values
            _LOG_PARAMS = {'O32'}

            def _axis_label(param):
                """Return 'log10(param)' for log-scaled params, else 'param'."""
                if param in _LOG_PARAMS:
                    return f'log$_{{10}}$({param})'
                return param

            ax.set_xlabel(_axis_label(x_param), fontsize=14)
            ax.set_ylabel(_axis_label(y_param), fontsize=14)

            title = f'{_axis_label(x_param)} vs {_axis_label(y_param)}'
            if color_param:
                title += f' (colored by {_axis_label(color_param)})'
            ax.set_title(title, fontsize=15, fontweight='bold')

            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=11, frameon=True)

            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  ✓ Figure saved to {save_path}")

            plt.show()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return

        print("=" * 70 + "\n")
# end of LyCDiagnostics class

# ===========================================================================
# Module-level helpers for additional observational catalogues
# ===========================================================================

def _to_float(series):
    """
    Convert a Series to float, coercing strings like '<0.003', '>6.2',
    '≤0.67' to NaN so they are cleanly excluded from plots.
    """
    def _safe(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        for prefix in ('<', '>', '≤', '≥', '~'):
            s = s.lstrip(prefix).strip()
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.apply(_safe)


def _load_sxds(path: str) -> dict:
    df = pd.read_csv(path)
    return {
        'M_UV':     _to_float(df['M_UV']),
        'E(B-V)':   (_to_float(df['E_BV'])),
        'f_esc':    _to_float(df['f_abs_c']),
        'O32':      pd.Series([np.nan] * len(df)),
        'redshift': _to_float(df['redshift']),
    }


def _load_bassett(path: str) -> dict:
    df = pd.read_csv(path)
    oiii = _to_float(df['[O III] (5007)'])
    oii  = _to_float(df['[O II] (λλ3727)'])
    with np.errstate(divide='ignore', invalid='ignore'):
        o32 = np.where(
            (oii > 0) & oiii.notna() & oii.notna(),
            np.log10(oiii / oii), np.nan
        )
    z = _to_float(df['z_spec']).fillna(_to_float(df['z']))
    return {
        'M_UV':     pd.Series([np.nan] * len(df)),
        'E(B-V)':   (_to_float(df['E(B-V)'])),
        'f_esc':    _to_float(df['f_esc abs']),
        'O32':      pd.Series(o32),
        'redshift': z,
    }


def _load_laces(path: str) -> dict:
    df = pd.read_csv(path)
    return {
        'M_UV':     _to_float(df['M_UV']),
        'E(B-V)':   (_to_float(df['E(B - V)'])),
        'f_esc':    _to_float(df['f_esc']),
        'O32':      np.log10(_to_float(df['O3/02'])),
        'redshift': _to_float(df['z_sys']),
    }


def _load_highz(path: str) -> dict:
    df = pd.read_csv(path)
    df['Source'] = df['Source'].str.strip()
    return {
        'M_UV':     _to_float(df['M_UV']),
        'E(B-V)':   (_to_float(df['E_BV'])),
        'f_esc':    _to_float(df['fesc_LyC_abs']),
        'O32':      pd.Series([np.nan] * len(df)),
        'redshift': _to_float(df['z']),
    }


# Redshift threshold above which the special high-z marker is used
_HIGHZ_THRESHOLD = 8.0

# Style per extra catalogue
_EXTRA_CATALOGS = {
    'sxds':    dict(label='SXDS Liu+23',    marker='D', color='purple', size=120, zorder=6, loader=_load_sxds),
    'bassett': dict(label='Bassett+19', marker='^', color='magenta', size=130, zorder=6, loader=_load_bassett),
    'laces':   dict(label='LACES Fletcher+19',   marker='o', color='olive', size=120, zorder=6, loader=_load_laces),
    'highz':   dict(label='Marques-Chaves',   marker='*', color='cyan', size=200, zorder=7, loader=_load_highz),
}

# Special style for the z~10 source inside misc_highz
_HIGHZ_SPECIAL_STYLE = dict(
    marker='*', color='black', edgecolors='red',
    linewidth=1.5, s=350, zorder=8,
)

# Mapping from CatalogueManager param names -> extra-catalogue dict keys
_PARAM_MAP = {
    'M_UV':   'M_UV',
    'E(B-V)': 'E(B-V)',
    'f_esc':  'f_esc',
    'O32':    'O32',
}


# ===========================================================================
# Patch plot_2d_all_directions_all onto LyCDiagnostics
# ===========================================================================