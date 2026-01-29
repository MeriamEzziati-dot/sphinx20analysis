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

