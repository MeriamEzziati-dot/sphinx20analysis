#!/usr/bin/env python3
"""
LyC Data Manager
================
Handles data loading and parameter extraction for SPHINX20 simulations and LzLCS observations.
This class is responsible for all data access and parameter mapping logic.

Author: Meriam Ezziati
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union

class CatalogueManagerError(Exception):
    pass
class CatalogueManager:
    """
    Data manager for SPHINX20 simulations and LzLCS observations.
    Handles all data loading, preprocessing, and parameter extraction.
    """

    def __init__(self, sim_catalogue: str, obs_catalogue: str):
        """
        Load SPHINX20 simulation and LzLCS observation data.

        Parameters:
        -----------
        sim_catalogue : str
            Path to simulation CSV file
        obs_catalogue : str
            Path to observations CSV file
        """
        print("=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        self.df = pd.read_csv(sim_catalogue)
        self.observations = pd.read_csv(obs_catalogue)

        print(f"✓ SPHINX20 simulations: {len(self.df)} galaxies")
        print(f"✓ LzLCS observations:   {len(self.observations)} galaxies")

        # Pre-calculate derived quantities
        self._calculate_directional_means()
        self._calculate_derived_quantities()

        print("✓ Calculated directional means and derived quantities")
        print("=" * 70 + "\n")

    def _calculate_directional_means(self):
        """Calculate mean and std for all directional parameters."""

        # f_esc directional (dir_0 to dir_9)
        fesc_dir_cols = [col for col in self.df.columns if col.startswith('fesc_dir_')]
        if fesc_dir_cols:
            self.df['fesc_dir_mean'] = self.df[fesc_dir_cols].mean(axis=1)
            self.df['fesc_dir_std'] = self.df[fesc_dir_cols].std(axis=1)

        # E(B-V) directional
        ebv_dir_cols = [col for col in self.df.columns if col.startswith('ebmv_dir_')]
        if ebv_dir_cols:
            self.df['ebmv_dir_mean'] = self.df[ebv_dir_cols].mean(axis=1)
            self.df['ebmv_dir_std'] = self.df[ebv_dir_cols].std(axis=1)

        # Beta (UV slope) directional
        beta_dir_cols = [col for col in self.df.columns if col.startswith('beta_dir_') and col.endswith('_sn')]
        if beta_dir_cols:
            self.df['beta_dir_mean'] = self.df[beta_dir_cols].mean(axis=1)
            self.df['beta_dir_std'] = self.df[beta_dir_cols].std(axis=1)

    def _calculate_derived_quantities(self):
        """Calculate common derived quantities."""

        # Linear quantities if not already present
        if 'f_esc_linear' not in self.df.columns:
            self.df['f_esc_linear'] = 10 ** self.df['f_esc']
        if 'stellar_mass_linear' not in self.df.columns:
            self.df['stellar_mass_linear'] = self.df['stellar_mass']
        if 'ionizing_luminosity_linear' not in self.df.columns:
            self.df['ionizing_luminosity_linear'] = 10 ** self.df['ionizing_luminosity']

        # Calculate O32 ratio (log([OIII]/[OII]))
        if 'OIII_5006.84_int' in self.df.columns and 'OII_3726.03_int' in self.df.columns:
            oiii = self.df['OIII_5006.84_int']
            oii = self.df['OII_3726.03_int'] + self.df['OII_3728.81_int']
            self.df['O32'] = np.log10(oiii / oii)

        # Calculate sSFR
        self.df['sSFR'] = np.log10(self.df['sfr_10'] / (10 ** self.df['stellar_mass']))

        # Calculate 12+log(O/H) from gas metallicity (assuming solar O/H = 8.69)
        self.df['12+log(O/H)'] = self.df['gas_metallicity'] + 8.69

        # BPT diagram coordinates
        if 'NII_6583.45_int' in self.df.columns and 'HI_6562.8_int' in self.df.columns:
            self.df['log_NII_Ha'] = np.log10(self.df['NII_6583.45_int'] / self.df['HI_6562.8_int'])
        if 'OIII_5006.84_int' in self.df.columns and 'HI_4861.32_int' in self.df.columns:
            self.df['log_OIII_Hb'] = np.log10(self.df['OIII_5006.84_int'] / self.df['HI_4861.32_int'])

    def get_parameter(self, param_name: str, dataset: str = 'both',
                      direction: Optional[Union[int, str]] = None) -> Union[Tuple[pd.Series, pd.Series], pd.Series]:
        """
        Extract parameter from simulation and/or observation datasets.

        Parameters:
        -----------
        param_name : str
            Standardized parameter name
        dataset : str
            'both', 'sim', or 'obs'
        direction : int or str, optional
            For directional parameters (0-9, 'mean', or 'std')

        Returns:
        --------
        tuple or pd.Series
            If dataset='both': (sim_data, obs_data)
            If dataset='sim': sim_data
            If dataset='obs': obs_data
        """
        sim_data = None
        obs_data = None

        # ============================================================
        # BASIC GALAXY PROPERTIES
        # ============================================================

        if param_name == "halo_id":
            sim_data = self.df['halo_id']
            obs_data = None

        elif param_name in ["redshift", "z"]:
            sim_data = self.df['redshift']
            obs_data = self.observations['z']

        elif param_name == "RA":
            sim_data = None
            obs_data = self.observations['RA']

        elif param_name == "DEC":
            sim_data = None
            obs_data = self.observations['DEC']

        # ============================================================
        # MASS PROPERTIES
        # ============================================================

        elif param_name in ["M_halo", "mvir"]:
            sim_data = self.df['mvir']
            obs_data = None

        elif param_name in ["R_halo", "rvir"]:
            sim_data = self.df['rvir']
            obs_data = None

        elif param_name in ["stellar_mass", "M_star"]:
            sim_data = self.df['stellar_mass']
            obs_data = self.observations['log10(Mstar)']

        # ============================================================
        # STAR FORMATION
        # ============================================================

        elif param_name == "SFR":
            sim_data = self.df['sfr_10']  # Default to 10 Myr
            obs_data = 10 ** self.observations['log10(SFR)-UV']

        elif param_name == "SFR_3Myr":
            sim_data = self.df['sfr_3']
            obs_data = None

        elif param_name == "SFR_10Myr":
            sim_data = self.df['sfr_10']
            obs_data = None

        elif param_name == "SFR_100Myr":
            sim_data = self.df['sfr_100']
            obs_data = None

        elif param_name == "sSFR":
            sim_data = self.df['sSFR']
            obs_data = self.observations['log10(SFR)-UV'] - self.observations['log10(Mstar)']

        # ============================================================
        # LYC ESCAPE FRACTION
        # ============================================================

        elif param_name == "f_esc":
            sim_data = 10 ** self.df['f_esc'] * 100  # Percentage
            obs_data = self.observations['f_esc(LyC)-UVfit'] * 100

        elif param_name == "f_esc_log":
            sim_data = self.df['f_esc']
            obs_data = np.log10(self.observations['f_esc(LyC)-UVfit'])

        elif param_name == "f_esc_dir":
            if direction is None or direction == 'mean':
                sim_data = self.df['fesc_dir_mean'] * 100
            elif direction == 'std':
                sim_data = self.df['fesc_dir_std'] * 100
            elif isinstance(direction, int) and 0 <= direction <= 9:
                sim_data = self.df[f'fesc_dir_{direction}'] * 100
            obs_data = None

        elif param_name == "f_esc_Lya":
            sim_data = None
            obs_data = self.observations['f_esc(LyA)']

        # ============================================================
        # METALLICITY
        # ============================================================

        elif param_name == "Z_gas":
            sim_data = self.df['gas_metallicity']
            obs_data = None

        elif param_name == "Z_star":
            sim_data = self.df['stellar_metallicity']
            obs_data = None

        elif param_name in ["metallicity", "12+log(O/H)"]:
            sim_data = self.df['12+log(O/H)']
            obs_data = self.observations['OH_12']

        # ============================================================
        # STELLAR PROPERTIES
        # ============================================================

        elif param_name == "age_star":
            sim_data = self.df['mean_stellar_age_mass']
            obs_data = None

        # ============================================================
        # IONIZING PROPERTIES
        # ============================================================

        elif param_name == "L_ion":
            sim_data = self.df['ionizing_luminosity']
            obs_data = None

        elif param_name == "xi_ion":
            sim_data = self.df['xi_ion']
            obs_data = self.observations['xi-ion']

        # ============================================================
        # UV PROPERTIES
        # ============================================================

        elif param_name in ["M_UV", "M_1500"]:
            if direction is None:
                sim_data = self.df['MAB_1500_int']
            elif isinstance(direction, int) and 0 <= direction <= 9:
                sim_data = self.df[f'MAB_1500_dir_{direction}']
            obs_data = self.observations['M_1500']

        elif param_name in ["beta", "UV_slope"]:
            if direction is None:
                sim_data = self.df['beta_int_sn']
            elif direction == 'mean':
                sim_data = self.df['beta_dir_mean']
            elif direction == 'std':
                sim_data = self.df['beta_dir_std']
            elif isinstance(direction, int) and 0 <= direction <= 9:
                sim_data = self.df[f'beta_dir_{direction}_sn']
            obs_data = self.observations['UV-beta']

        # ============================================================
        # DUST ATTENUATION
        # ============================================================

        elif param_name == "E(B-V)":
            if direction is None or direction == 'mean':
                sim_data = self.df['ebmv_dir_mean']
            elif direction == 'std':
                sim_data = self.df['ebmv_dir_std']
            elif isinstance(direction, int) and 0 <= direction <= 9:
                sim_data = self.df[f'ebmv_dir_{direction}']
            obs_data = None

        elif param_name == "E(B-V)_nebular":
            sim_data = None
            obs_data = self.observations['E(B-V)_nebular']

        elif param_name == "E(B-V)_uv":
            sim_data = None
            obs_data = self.observations['E(B-V)_uv']

        # ============================================================
        # EMISSION LINES - HYDROGEN
        # ============================================================

        elif param_name in ["Lya", "H_1215"]:
            sim_data = self.df['H__1_1215.67A_int']
            obs_data = self.observations['H1r_1216A']

        elif param_name in ["Ha", "H_6563"]:
            sim_data = self._get_directional_line('HI_6562.8', direction)
            obs_data = self.observations['H1r_6563A']

        elif param_name in ["Hb", "H_4861"]:
            sim_data = self._get_directional_line('HI_4861.32', direction)
            obs_data = self.observations['H1r_4861A']

        elif param_name in ["Hg", "H_4340"]:
            sim_data = self._get_directional_line('HI_4340.46', direction)
            obs_data = self.observations['H1r_4341A']

        elif param_name in ["Hd", "H_4101"]:
            sim_data = self._get_directional_line('HI_4101.73', direction)
            obs_data = self.observations['H1r_4102A']

        # ============================================================
        # EMISSION LINES - OXYGEN
        # ============================================================

        elif param_name == "OII_3726":
            sim_data = self._get_directional_line('OII_3726.03', direction)
            obs_data = self.observations['O2_3726A']

        elif param_name == "OII_3729":
            sim_data = self._get_directional_line('OII_3728.81', direction)
            obs_data = self.observations['O2_3729A']

        elif param_name == "OIII_4363":
            sim_data = self._get_directional_line('OIII_4363.21', direction)
            obs_data = self.observations['O3_4363A']

        elif param_name == "OIII_4959":
            sim_data = self._get_directional_line('OIII_4958.91', direction)
            obs_data = self.observations['O3_4959A']

        elif param_name == "OIII_5007":
            sim_data = self._get_directional_line('OIII_5006.84', direction)
            obs_data = self.observations['O3_5007A']

        elif param_name == "OI_6300":
            sim_data = self.df['O__1_6300.30A_int']
            obs_data = self.observations['O1_6300A']

        # ============================================================
        # EMISSION LINES - NITROGEN
        # ============================================================

        elif param_name == "NII_6548":
            sim_data = self.df['N__2_6548.05A_int']
            obs_data = self.observations['N2_6548A']

        elif param_name == "NII_6583":
            sim_data = self._get_directional_line('NII_6583.45', direction)
            obs_data = self.observations['N2_6584A']

        # ============================================================
        # EMISSION LINES - OTHER
        # ============================================================

        elif param_name == "NeIII_3869":
            sim_data = self._get_directional_line('NeIII_3868.76', direction)
            obs_data = self.observations['Ne3_3869A']

        elif param_name == "HeII_4686":
            sim_data = self.df['He_2_4685.68A_int']
            obs_data = self.observations['He2r_4686A']

        elif param_name == "SII_6716":
            sim_data = self.df['S__2_6716.44A_int']
            obs_data = self.observations['S2_6716A']

        elif param_name == "SII_6731":
            sim_data = self.df['S__2_6730.82A_int']
            obs_data = self.observations['S2_6731A']

        # ============================================================
        # EQUIVALENT WIDTHS
        # ============================================================

        elif param_name in ["EW_Hb", "EW_H4861"]:
            sim_data = None
            obs_data = self.observations['EW(H1r_4861)']

        elif param_name in ["EW_Ha", "EW_H6563"]:
            sim_data = None
            obs_data = self.observations['EW(H1r_6563)']

        elif param_name == "EW_OIII5007":
            sim_data = None
            obs_data = self.observations['EW(O3_5007)']

        elif param_name in ["EW_Lya", "EW_H1216"]:
            sim_data = None
            obs_data = self.observations['EW(H1r_1216)']

        # ============================================================
        # DIAGNOSTIC RATIOS
        # ============================================================

        elif param_name == "O32":
            sim_data = self.df['O32']
            oiii = self.observations['O3_5007A']
            oii = self.observations['O2_3726A'] + self.observations['O2_3729A']
            obs_data = np.log10(oiii / oii)

        elif param_name in ["BPT_x", "log_NII_Ha"]:
            sim_data = self.df['log_NII_Ha']
            obs_data = np.log10(self.observations['N2_6584A'] / self.observations['H1r_6563A'])

        elif param_name in ["BPT_y", "log_OIII_Hb"]:
            sim_data = self.df['log_OIII_Hb']
            obs_data = np.log10(self.observations['O3_5007A'] / self.observations['H1r_4861A'])

        # ============================================================
        # CONTINUUM LUMINOSITIES (wavelength-dependent)
        # ============================================================

        elif param_name.startswith("cont_"):
            # Extract wavelength: cont_1500, cont_1500_dir_3
            sim_data = self._get_continuum(param_name, direction)
            obs_data = None

        # ============================================================
        # SIZE / MORPHOLOGY
        # ============================================================

        elif param_name in ["r50", "half_radius"]:
            sim_data = None
            obs_data = self.observations['r_50_phys']

        # ============================================================
        # LYC DETECTION INFO (observations only)
        # ============================================================

        elif param_name == "LyC_SNR":
            sim_data = None
            obs_data = self.observations['SNR']

        elif param_name == "f_LyC":
            sim_data = None
            obs_data = self.observations['f(LyC)']

        else:
            raise ValueError(
                f"Parameter '{param_name}' not recognized.\n"
                f"Use list_available_parameters() to see all options."
            )

        # Return based on dataset request
        if dataset == 'both':
            return sim_data, obs_data
        elif dataset == 'sim':
            return sim_data
        elif dataset == 'obs':
            return obs_data
        else:
            raise ValueError("dataset must be 'both', 'sim', or 'obs'")

    def _get_directional_line(self, line_base: str, direction: Optional[Union[int, str]] = None) -> pd.Series:
        """
        Helper to get directional line emission.

        Parameters:
        -----------
        line_base : str
            Base name like 'HI_6562.8' or 'OIII_5006.84'
        direction : int or None
            Direction 0-9 or None for integrated
        """
        if direction is None:
            col_name = f"{line_base}_int"
        elif isinstance(direction, int) and 0 <= direction <= 9:
            col_name = f"{line_base}_dir_{direction}"
        else:
            col_name = f"{line_base}_int"

        if col_name in self.df.columns:
            return self.df[col_name]
        else:
            # Try alternate naming
            for col in self.df.columns:
                if line_base.replace('.', '') in col.replace('.', ''):
                    return self.df[col]
            return None

    def _get_continuum(self, param_name: str, direction: Optional[Union[int, str]] = None) -> pd.Series:
        """Helper to get continuum luminosity at various wavelengths."""
        if direction is None:
            col_name = f"{param_name}_int"
        elif isinstance(direction, int) and 0 <= direction <= 9:
            col_name = f"{param_name}_dir_{direction}"
        else:
            col_name = param_name

        if col_name in self.df.columns:
            return self.df[col_name]
        return None

    def list_available_parameters(self):
        """Print comprehensive list of all available parameters organized by category."""

        print("\n" + "=" * 70)
        print("AVAILABLE PARAMETERS FOR EXTRACTION")
        print("=" * 70)

        categories = {
            "BASIC PROPERTIES": [
                "halo_id", "redshift/z", "RA", "DEC"
            ],

            "MASS & SIZE": [
                "M_halo/mvir", "R_halo/rvir", "stellar_mass/M_star", "r50/half_radius"
            ],

            "STAR FORMATION": [
                "SFR", "SFR_3Myr", "SFR_10Myr", "SFR_100Myr", "sSFR"
            ],

            "LYC ESCAPE FRACTION": [
                "f_esc", "f_esc_log", "f_esc_dir (directions: 0-9, mean, std)", "f_esc_Lya"
            ],

            "METALLICITY": [
                "Z_gas", "Z_star", "metallicity/12+log(O/H)"
            ],

            "STELLAR PROPERTIES": [
                "age_star"
            ],

            "IONIZING PROPERTIES": [
                "L_ion", "xi_ion"
            ],

            "UV PROPERTIES": [
                "M_UV/M_1500 (directions: 0-9)", "beta/UV_slope (directions: 0-9, mean, std)"
            ],

            "DUST ATTENUATION": [
                "E(B-V) (directions: 0-9, mean, std)", "E(B-V)_nebular", "E(B-V)_uv"
            ],

            "HYDROGEN LINES (directions: 0-9 or integrated)": [
                "Lya/H_1215", "Ha/H_6563", "Hb/H_4861", "Hg/H_4340", "Hd/H_4101"
            ],

            "OXYGEN LINES (directions: 0-9 or integrated)": [
                "OII_3726", "OII_3729", "OIII_4363", "OIII_4959", "OIII_5007", "OI_6300"
            ],

            "NITROGEN LINES": [
                "NII_6548", "NII_6583 (directions: 0-9)"
            ],

            "OTHER LINES": [
                "NeIII_3869 (directions: 0-9)", "HeII_4686", "SII_6716", "SII_6731"
            ],

            "EQUIVALENT WIDTHS": [
                "EW_Hb/EW_H4861", "EW_Ha/EW_H6563", "EW_OIII5007", "EW_Lya/EW_H1216"
            ],

            "DIAGNOSTIC RATIOS": [
                "O32", "log_NII_Ha/BPT_x", "log_OIII_Hb/BPT_y"
            ],

            "CONTINUUM (wavelengths: 1300-6583Å, directions: 0-9)": [
                "cont_1500", "cont_1600", "cont_3000", "etc."
            ],

            "LYC DETECTION (observations only)": [
                "LyC_SNR", "f_LyC"
            ]
        }

        for category, params in categories.items():
            print(f"\n{category}:")
            print("-" * 70)
            for param in params:
                print(f"  • {param}")

        print("\n" + "=" * 70)
        print("DATASET AVAILABILITY:")
        print("  • BOTH (sim + obs): redshift, stellar_mass, SFR, f_esc, metallicity,")
        print("                      xi_ion, M_UV, beta, hydrogen/oxygen lines, etc.")
        print("  • SIMULATION ONLY:  halo mass, directional parameters, continuum,")
        print("                      Z_gas, Z_star, age_star, etc.")
        print("  • OBSERVATIONS ONLY: RA/DEC, equivalent widths, LyC detection info")
        print("=" * 70 + "\n")

        print("USAGE EXAMPLES:")
        print("  data.get_parameter('M_UV', 'both')")
        print("  data.get_parameter('f_esc_dir', 'sim', direction=3)")
        print("  data.get_parameter('Ha', 'both', direction=5)")
        print("  data.get_parameter('cont_1500', 'sim', direction='mean')")
        print("=" * 70 + "\n")

    def get_valid_mask(self, *param_names, dataset='both', direction=None):
        """
        Get boolean mask for valid (non-NaN) data points across multiple parameters.

        Parameters:
        -----------
        *param_names : str
            Variable number of parameter names
        dataset : str
            'sim' or 'obs'
        direction : int or str, optional
            For directional parameters

        Returns:
        --------
        pd.Series
            Boolean mask
        """
        if dataset == 'sim':
            mask = pd.Series(True, index=self.df.index)
            for param in param_names:
                data = self.get_parameter(param, 'sim', direction=direction)
                if data is not None:
                    mask &= data.notna()
        elif dataset == 'obs':
            mask = pd.Series(True, index=self.observations.index)
            for param in param_names:
                data = self.get_parameter(param, 'obs', direction=direction)
                if data is not None:
                    mask &= data.notna()
        else:
            raise ValueError("dataset must be 'sim' or 'obs' for mask generation")

        return mask

    def summary(self):
        """Print summary statistics of the loaded data."""
        print("\n" + "=" * 70)
        print("DATA SUMMARY")
        print("=" * 70)

        print(f"\nSIMULATIONS:")
        print(f"  • Total galaxies: {len(self.df)}")
        print(f"  • Redshift range: {self.df['redshift'].min():.2f} - {self.df['redshift'].max():.2f}")
        print(
            f"  • Stellar mass range: {self.df['stellar_mass'].min():.2f} - {self.df['stellar_mass'].max():.2f} log(M☉)")
        print(f"  • Mean f_esc: {(10 ** self.df['f_esc'].mean()) * 100:.2f}%")

        print(f"\nOBSERVATIONS:")
        print(f"  • Total galaxies: {len(self.observations)}")
        print(f"  • Redshift range: {self.observations['z'].min():.2f} - {self.observations['z'].max():.2f}")
        if 'log10(Mstar)' in self.observations.columns:
            print(
                f"  • Stellar mass range: {self.observations['log10(Mstar)'].min():.2f} - {self.observations['log10(Mstar)'].max():.2f} log(M☉)")
        if 'f_esc(LyC)-UVfit' in self.observations.columns:
            fesc_valid = self.observations['f_esc(LyC)-UVfit'].dropna()
            print(f"  • Mean f_esc: {fesc_valid.mean() * 100:.2f}% ({len(fesc_valid)} detections)")

        print("=" * 70 + "\n")

    def save_parameter_availability(self, output_path='parameter_availability.txt'):
        """
        Save a comprehensive report of parameter availability across datasets.

        Parameters:
        -----------
        output_path : str
            Path to save the output file

        Returns:
        --------
        str
            Path to the saved file
        """
        # Define all parameters organized by category
        parameter_catalog = {
            'BASIC PROPERTIES': {
                'halo_id': {'sim': 'halo_id', 'obs': None},
                'redshift/z': {'sim': 'redshift', 'obs': 'z'},
                'RA': {'sim': None, 'obs': 'RA'},
                'DEC': {'sim': None, 'obs': 'DEC'},
            },

            'MASS & SIZE': {
                'M_halo/mvir': {'sim': 'mvir', 'obs': None},
                'R_halo/rvir': {'sim': 'rvir', 'obs': None},
                'stellar_mass/M_star': {'sim': 'stellar_mass', 'obs': 'log10(Mstar)'},
                'r50/half_radius': {'sim': None, 'obs': 'r_50_phys'},
            },

            'STAR FORMATION': {
                'SFR': {'sim': 'sfr_10', 'obs': 'log10(SFR)-UV'},
                'SFR_3Myr': {'sim': 'sfr_3', 'obs': None},
                'SFR_5Myr': {'sim': 'sfr_5', 'obs': None},
                'SFR_10Myr': {'sim': 'sfr_10', 'obs': None},
                'SFR_100Myr': {'sim': 'sfr_100', 'obs': None},
                'sSFR': {'sim': 'sSFR (calculated)', 'obs': 'calculated from SFR/M*'},
            },

            'LYC ESCAPE FRACTION': {
                'f_esc': {'sim': 'f_esc', 'obs': 'f_esc(LyC)-UVfit'},
                'f_esc_dir (directions: 0-9, mean, std)': {'sim': 'fesc_dir_0...9', 'obs': None},
                'f_esc_Lya': {'sim': None, 'obs': 'f_esc(LyA)'},
            },

            'METALLICITY': {
                'Z_gas': {'sim': 'gas_metallicity', 'obs': None},
                'Z_star': {'sim': 'stellar_metallicity', 'obs': None},
                'metallicity/12+log(O/H)': {'sim': 'gas_metallicity + 8.69', 'obs': 'OH_12'},
                'Z_gas_Hb': {'sim': 'gas_metallicity_Hb', 'obs': None},
                'Z_gas_5007': {'sim': 'gas_metallicity_5007', 'obs': None},
                'Z_gas_6583': {'sim': 'gas_metallicity_6583', 'obs': None},
                'Z_gas_3727': {'sim': 'gas_metallicity_3727', 'obs': None},
            },

            'STELLAR PROPERTIES': {
                'age_star': {'sim': 'mean_stellar_age_mass', 'obs': None},
                'age_star_lion': {'sim': 'mean_stellar_age_lion', 'obs': None},
                'Z_star_mass': {'sim': 'mean_stellar_metallicity_mass', 'obs': None},
                'Z_star_lion': {'sim': 'mean_stellar_metallicity_lion', 'obs': None},
            },

            'IONIZING PROPERTIES': {
                'L_ion': {'sim': 'ionizing_luminosity', 'obs': None},
                'xi_ion': {'sim': 'xi_ion', 'obs': 'xi-ion'},
            },

            'UV PROPERTIES': {
                'M_UV/M_1500 (directions: 0-9)': {'sim': 'MAB_1500_int, MAB_1500_dir_0...9', 'obs': 'M_1500'},
                'M_FUV': {'sim': None, 'obs': 'M_FUV'},
                'M_NUV': {'sim': None, 'obs': 'M_NUV'},
                'beta/UV_slope (directions: 0-9, mean, std)': {'sim': 'beta_int_sn, beta_dir_0...9_sn',
                                                               'obs': 'UV-beta'},
                'beta1550': {'sim': None, 'obs': 'beta1550'},
            },

            'DUST ATTENUATION': {
                'E(B-V) (directions: 0-9, mean, std)': {'sim': 'ebmv_dir_0...9', 'obs': None},
                'E(B-V)_nebular': {'sim': None, 'obs': 'E(B-V)_nebular'},
                'E(B-V)_uv': {'sim': None, 'obs': 'E(B-V)_uv'},
                'E(B-V)_mw': {'sim': None, 'obs': 'E(B-V)_mw'},
            },

            'HYDROGEN LINES': {
                'Lya/H_1215': {'sim': 'H__1_1215.67A_int', 'obs': 'H1r_1216A'},
                'Ha/H_6563 (directions: 0-9)': {'sim': 'H__1_6562.80A_int, HI_6562.8_dir_0...9', 'obs': 'H1r_6563A'},
                'Hb/H_4861 (directions: 0-9)': {'sim': 'H__1_4861.32A_int, HI_4861.32_dir_0...9', 'obs': 'H1r_4861A'},
                'Hg/H_4340 (directions: 0-9)': {'sim': 'H__1_4340.46A_int, HI_4340.46_dir_0...9', 'obs': 'H1r_4341A'},
                'Hd/H_4101 (directions: 0-9)': {'sim': 'H__1_4101.73A_int, HI_4101.73_dir_0...9', 'obs': 'H1r_4102A'},
            },

            'OXYGEN LINES': {
                'OII_3726 (directions: 0-9)': {'sim': 'O__2_3726.03A_int, OII_3726.03_dir_0...9', 'obs': 'O2_3726A'},
                'OII_3729 (directions: 0-9)': {'sim': 'O__2_3728.81A_int, OII_3728.81_dir_0...9', 'obs': 'O2_3729A'},
                'OIII_4363 (directions: 0-9)': {'sim': 'O__3_4363.21A_int, OIII_4363.21_dir_0...9', 'obs': 'O3_4363A'},
                'OIII_4959 (directions: 0-9)': {'sim': 'O__3_4958.91A_int, OIII_4958.91_dir_0...9', 'obs': 'O3_4959A'},
                'OIII_5007 (directions: 0-9)': {'sim': 'O__3_5006.84A_int, OIII_5006.84_dir_0...9', 'obs': 'O3_5007A'},
                'OI_6300': {'sim': 'O__1_6300.30A_int', 'obs': 'O1_6300A'},
            },

            'NITROGEN LINES': {
                'NII_6548': {'sim': 'N__2_6548.05A_int', 'obs': 'N2_6548A'},
                'NII_6583 (directions: 0-9)': {'sim': 'N__2_6583.45A_int, NII_6583.45_dir_0...9', 'obs': 'N2_6584A'},
            },

            'OTHER EMISSION LINES': {
                'NeIII_3869 (directions: 0-9)': {'sim': 'Ne_3_3868.76A_int, NeIII_3868.76_dir_0...9',
                                                 'obs': 'Ne3_3869A'},
                'HeII_4686': {'sim': 'He_2_4685.68A_int', 'obs': 'He2r_4686A'},
                'SII_6716': {'sim': 'S__2_6716.44A_int', 'obs': 'S2_6716A'},
                'SII_6731': {'sim': 'S__2_6730.82A_int', 'obs': 'S2_6731A'},
                'CIII_1907': {'sim': 'C__3_1906.68A_int', 'obs': None},
                'CIV_1549': {'sim': 'C__4_1548.19A_int', 'obs': None},
            },

            'EQUIVALENT WIDTHS': {
                'EW_Hb/EW_H4861': {'sim': None, 'obs': 'EW(H1r_4861)'},
                'EW_Ha/EW_H6563': {'sim': None, 'obs': 'EW(H1r_6563)'},
                'EW_OIII5007': {'sim': None, 'obs': 'EW(O3_5007)'},
                'EW_Lya/EW_H1216': {'sim': None, 'obs': 'EW(H1r_1216)'},
            },

            'DIAGNOSTIC RATIOS': {
                'O32': {'sim': 'calculated: log(OIII/OII)', 'obs': 'calculated: log(OIII/OII)'},
                'log_NII_Ha/BPT_x': {'sim': 'calculated: log(NII/Ha)', 'obs': 'calculated: log(NII/Ha)'},
                'log_OIII_Hb/BPT_y': {'sim': 'calculated: log(OIII/Hb)', 'obs': 'calculated: log(OIII/Hb)'},
                'R23': {'sim': 'calculated: (OII+OIII)/Hb', 'obs': None},
            },

            'CONTINUUM LUMINOSITIES': {
                'cont_XXXX (wavelengths: 1300-6583Å, directions: 0-9)': {
                    'sim': 'cont_1300...6583_int, cont_XXXX_dir_0...9',
                    'obs': None
                },
            },

            'NEBULAR CONTINUUM': {
                'nebc_XXXX (wavelengths: 1300-6583Å, directions: 0-9)': {
                    'sim': 'nebc_1300...6583_int, nebc_XXXX_dir_0...9',
                    'obs': None
                },
            },

            'JWST FILTERS': {
                'F_XXXW/M (filters: F070W-F480M, directions: 0-9)': {
                    'sim': 'F070W...F480M_int, F_XXXW_dir_0...9',
                    'obs': None
                },
            },

            'PHYSICAL DENSITIES & TEMPERATURES': {
                'gas_density_3727': {'sim': 'gas_density_3727', 'obs': None},
                'gas_density_1908': {'sim': 'gas_density_1908', 'obs': None},
                'n_e_S2': {'sim': None, 'obs': 'ne_S2'},
                'n_e_O2': {'sim': None, 'obs': 'ne_O2'},
                'T_e_O3': {'sim': None, 'obs': 'Te_O3'},
            },

            'DISTANCE MEASURES': {
                'd_lum': {'sim': None, 'obs': 'd_lum'},
                'd_ang': {'sim': None, 'obs': 'd_ang'},
                'd_com': {'sim': None, 'obs': 'd_com'},
            },

            'LYC DETECTION INFO (observations only)': {
                'LyC_SNR': {'sim': None, 'obs': 'SNR'},
                'LyC_significance': {'sim': None, 'obs': 'significance'},
                'f_LyC': {'sim': None, 'obs': 'f(LyC)'},
                'L_LyC': {'sim': None, 'obs': 'L(LyC)'},
            },
        }

        # Write to file
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PARAMETER AVAILABILITY ACROSS DATASETS\n")
            f.write("SPHINX20 Simulations vs LzLCS Observations\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"Simulation file: {len(self.df)} galaxies, {len(self.df.columns)} columns\n")
            f.write(f"Observation file: {len(self.observations)} galaxies, {len(self.observations.columns)} columns\n")
            f.write("\n" + "=" * 80 + "\n\n")

            # Count statistics
            both_count = 0
            sim_only_count = 0
            obs_only_count = 0

            # Categorized parameters
            for category, params in parameter_catalog.items():
                f.write(f"\n{'─' * 80}\n")
                f.write(f"{category}\n")
                f.write(f"{'─' * 80}\n\n")

                for param_name, availability in params.items():
                    sim_col = availability['sim']
                    obs_col = availability['obs']

                    # Determine availability
                    if sim_col and obs_col:
                        status = "BOTH"
                        both_count += 1
                        marker = "✓✓"
                    elif sim_col:
                        status = "SIM ONLY"
                        sim_only_count += 1
                        marker = "✓─"
                    elif obs_col:
                        status = "OBS ONLY"
                        obs_only_count += 1
                        marker = "─✓"
                    else:
                        status = "NEITHER"
                        marker = "──"

                    f.write(f"{marker} {param_name:<50} [{status}]\n")
                    if sim_col:
                        f.write(f"     → Simulation: {sim_col}\n")
                    if obs_col:
                        f.write(f"     → Observation: {obs_col}\n")
                    f.write("\n")

            # Summary statistics
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            total = both_count + sim_only_count + obs_only_count

            f.write(f"Total parameter groups: {total}\n\n")
            f.write(f"Available in BOTH datasets:        {both_count:3d} ({both_count / total * 100:.1f}%)\n")
            f.write(f"Available in SIMULATIONS ONLY:     {sim_only_count:3d} ({sim_only_count / total * 100:.1f}%)\n")
            f.write(f"Available in OBSERVATIONS ONLY:    {obs_only_count:3d} ({obs_only_count / total * 100:.1f}%)\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("LEGEND\n")
            f.write("=" * 80 + "\n\n")
            f.write("✓✓  Available in both simulations and observations\n")
            f.write("✓─  Available in simulations only\n")
            f.write("─✓  Available in observations only\n")
            f.write("──  Not available (placeholder)\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("NOTES\n")
            f.write("=" * 80 + "\n\n")
            f.write("• Directional parameters (directions: 0-9) are available for many emission\n")
            f.write("  lines and physical properties in simulations.\n")
            f.write("• Mean and std across directions are automatically calculated.\n")
            f.write("• Some parameters are derived/calculated (e.g., O32, BPT ratios, sSFR).\n")
            f.write("• Continuum luminosities available at wavelengths: 1300, 1400, 1500, 1600,\n")
            f.write("  1700, 1800, 1900, 2000, 2500, 3000, 3727, 3869, 4102, 4341, 4363, 4861,\n")
            f.write("  4959, 5008, 6563, 6583 Å (both integrated and directional).\n")
            f.write("• JWST filters: F070W, F090W, F115W, F140M, F150W, F162M, F182M, F200W,\n")
            f.write("  F210M, F250M, F277W, F300M, F335M, F356W, F360M, F410M, F430M, F444W,\n")
            f.write("  F460M, F480M (both integrated and directional).\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("USAGE EXAMPLES\n")
            f.write("=" * 80 + "\n\n")
            f.write("# Get parameter from both datasets\n")
            f.write("muv_sim, muv_obs = data.get_parameter('M_UV', 'both')\n\n")
            f.write("# Get directional parameter\n")
            f.write("fesc_dir3 = data.get_parameter('f_esc_dir', 'sim', direction=3)\n\n")
            f.write("# Get mean across directions\n")
            f.write("beta_mean = data.get_parameter('beta', 'sim', direction='mean')\n\n")
            f.write("# Get specific emission line\n")
            f.write("ha_sim, ha_obs = data.get_parameter('Ha', 'both', direction=5)\n\n")
            f.write("# Get continuum at specific wavelength\n")
            f.write("cont_1500 = data.get_parameter('cont_1500', 'sim', direction=0)\n\n")

            f.write("=" * 80 + "\n")

        print(f"✓ Parameter availability report saved to: {output_path}")
        print(f"  • Parameters in BOTH datasets: {both_count}")
        print(f"  • Parameters in SIMULATIONS only: {sim_only_count}")
        print(f"  • Parameters in OBSERVATIONS only: {obs_only_count}")

        return output_path