#!/usr/bin/env python3
"""
Bulk Modulus Analysis Script for VASP Calculations (No Outlier Removal)

This script analyzes bulk modulus calculations from VASP OUTCAR files.
It extracts volume and energy data, fits to Birch-Murnaghan EOS,
and generates comprehensive plots. All data is plotted without removing outliers.

Key Parameters:
- B0: Bulk modulus (GPa)
- V0: Equilibrium volume (Å³)
- B0': Pressure derivative of bulk modulus (dimensionless)
- E0: Equilibrium energy (eV)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from collections import defaultdict


class BulkModulusAnalyzer:
    """Analyze bulk modulus calculations from VASP OUTCAR files."""
    
    def __init__(self, base_dir):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing bulk modulus calculations
        """
        self.base_dir = Path(base_dir)
        self.data = defaultdict(dict)  # {compound: {volume: {energy, volume}}}
        
    def extract_volume_energy(self, outcar_path):
        """
        Extract volume, energy, max atomic force, and hydrostatic pressure
        (external pressure) from OUTCAR file.

        Parameters:
        -----------
        outcar_path : str or Path
            Path to OUTCAR file

        Returns:
        --------
        tuple : (volume, energy, max_force, pressure_gpa) or (None, None, None, None)
            if volume/energy extraction fails. max_force is the largest per-atom
            force magnitude (eV/Angst) in the final ionic step; pressure_gpa is
            the "external pressure" (GPa) from the final ionic step's stress
            tensor. Either can be None if it could not be parsed.
        """
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()

            # Extract volume - look for "volume of cell :      XXX.XX"
            vol_pattern = r'volume of cell\s*:\s*([\d.]+)'
            vol_match = re.findall(vol_pattern, content)
            if vol_match:
                volume = float(vol_match[-1])  # Take the last occurrence (final volume)
            else:
                return None, None, None, None

            # Extract energy - look for final "free energy    TOTEN  =     -XXXX.XXXXX eV"
            energy_pattern = r'free\s+energy\s+TOTEN\s+=\s+([\d.-]+)\s+eV'
            energy_matches = re.findall(energy_pattern, content)
            if energy_matches:
                energy = float(energy_matches[-1])  # Take the last occurrence (final energy)
            else:
                return None, None, None, None

            max_force = self.extract_max_force(content)
            pressure_gpa = self.extract_pressure(content)

            return volume, energy, max_force, pressure_gpa

        except Exception as e:
            print(f"Error reading {outcar_path}: {e}")
            return None, None, None, None

    def extract_max_force(self, content):
        """
        Extract the max per-atom force magnitude from the final ionic step's
        TOTAL-FORCE block in OUTCAR content.

        Parameters:
        -----------
        content : str
            Full text content of an OUTCAR file

        Returns:
        --------
        float or None : max |force| (eV/Angst) in the last TOTAL-FORCE block,
            or None if the block could not be parsed.
        """
        blocks = content.split('TOTAL-FORCE (eV/Angst)')
        if len(blocks) < 2:
            return None

        last_block = blocks[-1]
        forces = []
        started = False
        for line in last_block.splitlines():
            stripped = line.strip()
            if stripped and set(stripped) <= {'-'}:
                if started:
                    break
                started = True
                continue
            if not started:
                continue
            parts = stripped.split()
            if len(parts) != 6:
                break
            try:
                fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
            except ValueError:
                break
            forces.append((fx, fy, fz))

        if not forces:
            return None

        forces = np.array(forces)
        return float(np.max(np.linalg.norm(forces, axis=1)))

    def extract_pressure(self, content):
        """
        Extract the hydrostatic "external pressure" from the final ionic
        step's stress tensor in OUTCAR content.

        Parameters:
        -----------
        content : str
            Full text content of an OUTCAR file

        Returns:
        --------
        float or None : external pressure (GPa) at the final ionic step,
            or None if it could not be parsed. VASP reports this in kB
            (1 kB = 0.1 GPa); positive means compressive.
        """
        pressure_pattern = r'external pressure\s*=\s*([-\d.]+)\s*kB'
        matches = re.findall(pressure_pattern, content)
        if not matches:
            return None
        pressure_kb = float(matches[-1])  # Final ionic step
        return pressure_kb * 0.1  # kB -> GPa

    def parse_directory_structure(self):
        """Parse directory structure and extract data from all OUTCAR files."""
        print("Scanning directory structure...")
        
        # Find all OUTCAR files
        outcar_files = list(self.base_dir.rglob('OUTCAR'))
        print(f"Found {len(outcar_files)} OUTCAR files")
        
        for outcar_path in outcar_files:
            # Extract compound name and volume from path
            # Path structure: base_dir/compound/.../V_XX.X%/OUTCAR
            parts = outcar_path.parts
            
            # Find volume directory (V_XX.X%)
            vol_dir = None
            compound = None
            
            for i, part in enumerate(parts):
                if part.startswith('V_'):
                    vol_dir = part
                    # Compound is typically 2-3 levels up
                    if i >= 2:
                        compound = parts[i-1]
                    elif i >= 1:
                        compound = parts[i-1]
                    break
            
            if not vol_dir or not compound:
                # Try alternative: look for compound name in path
                for part in parts:
                    if part not in ['optimization', 'neutral', 'monkhorst-pack_calculated', 
                                   'unitcells', 'isif3'] and not part.startswith('V_'):
                        compound = part
                        break
                if not vol_dir:
                    continue
            
            # Extract volume percentage
            vol_pct_match = re.search(r'V_([+-]?[\d.]+)%', vol_dir)
            if not vol_pct_match:
                continue
            vol_pct = float(vol_pct_match.group(1))
            
            # Extract volume, energy, max atomic force, and pressure
            volume, energy, max_force, pressure_gpa = self.extract_volume_energy(outcar_path)

            if volume is not None and energy is not None:
                if compound not in self.data:
                    self.data[compound] = {}
                self.data[compound][vol_pct] = {
                    'volume': volume,
                    'energy': energy,
                    'max_force': max_force,
                    'pressure_gpa': pressure_gpa,
                    'path': str(outcar_path)
                }
        
        print(f"Extracted data for {len(self.data)} compounds")
        for compound, vols in self.data.items():
            print(f"  {compound}: {len(vols)} volume points")
    
    def birch_murnaghan_eos(self, V, E0, V0, B0, B0_prime):
        """
        Birch-Murnaghan equation of state.
        
        Parameters:
        -----------
        V : array-like
            Volume (Å³)
        E0 : float
            Equilibrium energy (eV)
        V0 : float
            Equilibrium volume (Å³)
        B0 : float
            Bulk modulus (eV/Å³)
        B0_prime : float
            Pressure derivative of bulk modulus (dimensionless)
            
        Returns:
        --------
        array-like : Energy (eV)
        """
        V = np.array(V)
        eta = (V0 / V) ** (1/3)
        
        # Birch-Murnaghan EOS (3rd order)
        E = E0 + (9 * V0 * B0 / 16) * (
            (eta**2 - 1)**3 * B0_prime +
            (eta**2 - 1)**2 * (6 - 4 * eta**2)
        )
        
        return E
    
    def pressure_from_eos(self, V, E0, V0, B0, B0_prime):
        """
        Calculate pressure from Birch-Murnaghan EOS.
        
        P = -dE/dV
        """
        V = np.array(V)
        eta = (V0 / V) ** (1/3)
        
        # Derivative of Birch-Murnaghan EOS
        dE_dV = (3 * B0 / (2 * V)) * (
            (eta**7 - eta**5) * (1 + 3/4 * (B0_prime - 4) * (eta**2 - 1))
        )
        
        P = -dE_dV  # Pressure = -dE/dV
        
        # Convert from eV/Å³ to GPa
        # 1 eV/Å³ = 160.21766208 GPa
        P_GPa = P * 160.21766208
        
        return P_GPa
    
    def fit_eos(self, volumes, energies):
        """
        Fit energy-volume data to Birch-Murnaghan EOS.
        
        Parameters:
        -----------
        volumes : array-like
            Volumes (Å³)
        energies : array-like
            Energies (eV)
            
        Returns:
        --------
        dict : Fitted parameters and statistics
        """
        volumes = np.array(volumes)
        energies = np.array(energies)
        
        # Need at least 4 points for 4-parameter fit
        if len(volumes) < 4:
            print(f"Warning: Only {len(volumes)} data points, need at least 4 for EOS fit")
            return None
        
        # Initial guesses
        E0_guess = np.min(energies)
        V0_guess = volumes[np.argmin(energies)]
        
        # Estimate B0 from curvature around minimum
        # Use polynomial fit to estimate second derivative
        if len(volumes) >= 5:
            # Fit a quadratic around the minimum
            idx_min = np.argmin(energies)
            idx_range = max(2, min(3, len(volumes) // 3))
            start_idx = max(0, idx_min - idx_range)
            end_idx = min(len(volumes), idx_min + idx_range + 1)
            V_local = volumes[start_idx:end_idx]
            E_local = energies[start_idx:end_idx]
            
            # Fit quadratic: E = a*V^2 + b*V + c
            coeffs = np.polyfit(V_local, E_local, 2)
            # B0 ≈ V0 * d²E/dV² at V0
            # d²E/dV² = 2*a
            # B0 = V0 * 2*a (in eV/Å³)
            B0_guess = abs(V0_guess * 2 * coeffs[0])
            B0_guess = max(0.1, min(B0_guess, 10.0))  # Reasonable bounds
        else:
            B0_guess = 1.0  # Default guess
        
        B0_prime_guess = 4.0  # Typical value
        
        # Set bounds for parameters
        # E0: within range of energies
        E0_bounds = (np.min(energies) - 100, np.max(energies) + 100)
        # V0: within range of volumes
        V0_bounds = (volumes.min() * 0.8, volumes.max() * 1.2)
        # B0: reasonable range (0.01 to 20 eV/Å³)
        B0_bounds = (0.01, 20.0)
        # B0_prime: typically 2-8
        B0_prime_bounds = (1.0, 10.0)
        
        bounds = ([E0_bounds[0], V0_bounds[0], B0_bounds[0], B0_prime_bounds[0]],
                  [E0_bounds[1], V0_bounds[1], B0_bounds[1], B0_prime_bounds[1]])
        
        # Fit
        try:
            popt, pcov = curve_fit(
                self.birch_murnaghan_eos,
                volumes,
                energies,
                p0=[E0_guess, V0_guess, B0_guess, B0_prime_guess],
                bounds=bounds,
                maxfev=20000,
                method='trf'  # Trust Region Reflective algorithm
            )
            
            E0, V0, B0, B0_prime = popt
            
            # Calculate errors
            perr = np.sqrt(np.diag(pcov))
            
            # Convert B0 to GPa
            B0_GPa = B0 * 160.21766208
            
            # Calculate R-squared
            energies_fit = self.birch_murnaghan_eos(volumes, *popt)
            ss_res = np.sum((energies - energies_fit)**2)
            ss_tot = np.sum((energies - np.mean(energies))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'E0': E0,
                'V0': V0,
                'B0': B0,
                'B0_GPa': B0_GPa,
                'B0_prime': B0_prime,
                'E0_err': perr[0],
                'V0_err': perr[1],
                'B0_err': perr[2] * 160.21766208,  # Convert to GPa
                'B0_prime_err': perr[3],
                'r_squared': r_squared,
                'popt': popt
            }
            
        except Exception as e:
            print(f"Error fitting EOS: {e}")
            return None
    
    def plot_all_compounds(self):
        """Plot all compounds on a single Energy vs Volume plot with EOS fits.
        Energies are normalized (shifted) so each compound's minimum energy is at 0 eV.
        All data is plotted without removing any outliers.
        """
        if not self.data:
            print("No data to plot")
            return
        
        n_compounds = len(self.data)
        fig, (ax, ax_force, ax_stress) = plt.subplots(3, 1, figsize=(14, 22), constrained_layout=True)
        
        # Generate colors for each compound
        colors = plt.cm.tab20(np.linspace(0, 1, n_compounds))
        
        # Collect all normalized energies
        all_normalized_energies = []
        plot_data = []
        
        # Process all data: normalize and prepare for plotting
        for i, (compound, data) in enumerate(self.data.items()):
            vol_pcts = sorted(data.keys())
            volumes = np.array([data[v]['volume'] for v in vol_pcts])
            energies = np.array([data[v]['energy'] for v in vol_pcts])
            forces_raw = [data[v].get('max_force') for v in vol_pcts]
            force_mask = np.array([f is not None for f in forces_raw])
            forces = np.array([f for f in forces_raw if f is not None])
            force_volumes = volumes[force_mask]

            pressures_raw = [data[v].get('pressure_gpa') for v in vol_pcts]
            pressure_mask = np.array([p is not None for p in pressures_raw])
            pressures = np.array([p for p in pressures_raw if p is not None])
            pressure_volumes = volumes[pressure_mask]

            # Fit EOS to get E0 (equilibrium energy)
            fit_result = self.fit_eos(volumes, energies)

            if fit_result:
                E0 = fit_result['E0']
                energies_normalized = energies - E0
                all_normalized_energies.extend(energies_normalized.tolist())

                # Store plot data
                V_fit = np.linspace(volumes.min() * 0.95, volumes.max() * 1.05, 200)
                E_fit = self.birch_murnaghan_eos(V_fit, *fit_result['popt'])
                E_fit_normalized = E_fit - E0

                plot_data.append({
                    'compound': compound,
                    'volumes': volumes,
                    'energies_norm': energies_normalized,
                    'V_fit': V_fit,
                    'E_fit_norm': E_fit_normalized,
                    'color': colors[i],
                    'fit_result': fit_result,
                    'force_volumes': force_volumes,
                    'forces': forces,
                    'pressure_volumes': pressure_volumes,
                    'pressures': pressures
                })
            else:
                # If fit failed, just normalize by minimum energy
                E0 = np.min(energies)
                energies_normalized = energies - E0
                all_normalized_energies.extend(energies_normalized.tolist())

                plot_data.append({
                    'compound': compound,
                    'volumes': volumes,
                    'energies_norm': energies_normalized,
                    'V_fit': None,
                    'E_fit_norm': None,
                    'color': colors[i],
                    'fit_result': None,
                    'force_volumes': force_volumes,
                    'forces': forces,
                    'pressure_volumes': pressure_volumes,
                    'pressures': pressures
                })
        
        # Calculate y-axis limits from all data (use full range to show all points)
        all_normalized_energies = np.array(all_normalized_energies)
        if len(all_normalized_energies) > 0:
            y_min = np.min(all_normalized_energies)
            y_max = np.max(all_normalized_energies)
            y_range = max(abs(y_min), abs(y_max))
            y_limit = y_range * 1.1  # Add 10% padding
        else:
            y_limit = 10.0  # Fallback
        
        # Plot each compound (all data, no filtering)
        print(f"\nPlotting {len(plot_data)} compound(s):")
        for data in plot_data:
            compound = data['compound']
            volumes = data['volumes']
            energies_norm = data['energies_norm']
            color = data['color']
            
            print(f"  '{compound}': {len(volumes)} points, energy range: [{np.min(energies_norm):.3f}, {np.max(energies_norm):.3f}] eV")
            
            # Plot normalized data points
            # Use larger markers and different style for compounds without EOS fit to make them more visible
            if data['V_fit'] is not None and data['E_fit_norm'] is not None:
                # Standard markers for compounds with EOS fit
                ax.scatter(volumes, energies_norm, s=80, alpha=0.7, 
                          label=compound, color=color, zorder=3)
            else:
                # Larger, edge-highlighted markers for compounds without EOS fit
                ax.scatter(volumes, energies_norm, s=120, alpha=0.8, 
                          label=compound, color=color, zorder=3,
                          edgecolors='black', linewidths=1.5, marker='s')
            
            # Plot normalized fitted curve if available
            if data['V_fit'] is not None and data['E_fit_norm'] is not None:
                V_fit = data['V_fit']
                E_fit_norm = data['E_fit_norm']
                ax.plot(V_fit, E_fit_norm, '-', linewidth=2, color=color,
                       alpha=0.7, zorder=2)
                print(f"    -> Fitted curve plotted")
            else:
                print(f"    -> No fitted curve (insufficient points or fit failed)")

            # Plot max atomic force vs volume on the second subplot
            force_volumes = data['force_volumes']
            forces = data['forces']
            if len(forces) > 0:
                ax_force.plot(force_volumes, forces, 'o-', linewidth=1.5, markersize=6,
                              alpha=0.7, color=color, label=compound, zorder=3)
            else:
                print(f"    -> No force data (could not parse TOTAL-FORCE block)")

            # Plot DFT external pressure (stress) vs volume
            pressure_volumes = data['pressure_volumes']
            pressures = data['pressures']
            if len(pressures) > 0:
                ax_stress.plot(pressure_volumes, pressures, 'o-', linewidth=1.5, markersize=6,
                               alpha=0.7, color=color, label=compound, zorder=3)
            else:
                print(f"    -> No pressure data (could not parse 'external pressure')")

        # Set y-axis limits (symmetric around zero)
        ax.set_ylim(-y_limit, y_limit)

        ax.set_xlabel('Volume (Å³)', fontsize=14)
        ax.set_ylabel('Energy - E₀ (eV)', fontsize=14)
        ax.set_title(
            f'Bulk Modulus: Normalized Energy vs Volume\n'
            f'(All data included, no outliers removed; y-range ±{y_limit:.1f} eV, '
            f'{n_compounds} compound(s))',
            fontsize=14, fontweight='bold'
        )
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                 borderaxespad=0, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

        ax_force.set_xlabel('Volume (Å³)', fontsize=14)
        ax_force.set_ylabel('Max atomic force (eV/Å, log scale)', fontsize=14)
        ax_force.set_title('Max Atomic Force vs Volume', fontsize=14, fontweight='bold')
        ax_force.set_yscale('log')
        ax_force.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                        borderaxespad=0, framealpha=0.9)
        ax_force.grid(True, alpha=0.3, which='both')

        ax_stress.set_xlabel('Volume (Å³)', fontsize=14)
        ax_stress.set_ylabel('External pressure (GPa, symlog scale)', fontsize=14)
        ax_stress.set_title('Stress (External Pressure) vs Volume', fontsize=14, fontweight='bold')
        # symlog: pressure goes negative in tension (expansion) and spans orders
        # of magnitude in compression, so linear/log alone can't show both.
        ax_stress.set_yscale('symlog', linthresh=10)
        ax_stress.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                         borderaxespad=0, framealpha=0.9)
        ax_stress.grid(True, alpha=0.3, which='both')
        ax_stress.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

        # Legends live outside the axes (right side); constrained_layout
        # reserves space for them and for wrapped titles automatically,
        # so nothing gets cropped regardless of title length or compound count.
        plt.show()
    
    def generate_summary_table(self):
        """Generate a summary table of all fitted parameters."""
        results = []
        
        for compound, data in self.data.items():
            vol_pcts = sorted(data.keys())
            volumes = np.array([data[v]['volume'] for v in vol_pcts])
            energies = np.array([data[v]['energy'] for v in vol_pcts])
            fit_result = self.fit_eos(volumes, energies)
            
            if fit_result:
                results.append({
                    'Compound': compound,
                    'B₀ (GPa)': f"{fit_result['B0_GPa']:.2f} ± {fit_result['B0_err']:.2f}",
                    'V₀ (Å³)': f"{fit_result['V0']:.2f} ± {fit_result['V0_err']:.2f}",
                    "B₀'": f"{fit_result['B0_prime']:.2f} ± {fit_result['B0_prime_err']:.2f}",
                    'E₀ (eV)': f"{fit_result['E0']:.2f} ± {fit_result['E0_err']:.2f}",
                    'R²': f"{fit_result['r_squared']:.4f}",
                    'N_points': len(volumes)
                })
        
        # Print table
        print("\n" + "="*100)
        print("BULK MODULUS ANALYSIS SUMMARY")
        print("="*100)
        print(f"{'Compound':<20} {'B₀ (GPa)':<20} {'V₀ (Å³)':<15} {'B₀\'':<15} {'E₀ (eV)':<15} {'R²':<10} {'N':<5}")
        print("-"*100)
        
        for r in results:
            print(f"{r['Compound']:<20} {r['B₀ (GPa)']:<20} {r['V₀ (Å³)']:<15} "
                  f"{r['B₀\'']:<15} {r['E₀ (eV)']:<15} {r['R²']:<10} {r['N_points']:<5}")
        
        print("="*100 + "\n")
        
        return results


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze bulk modulus calculations from VASP OUTCAR files (no outlier removal)'
    )
    parser.add_argument(
        'base_dir',
        type=str,
        help='Base directory containing bulk modulus calculations'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = BulkModulusAnalyzer(args.base_dir)
    
    # Parse directory structure
    analyzer.parse_directory_structure()
    
    if not analyzer.data:
        print("No data found! Check your directory structure.")
        return
    
    # Generate summary
    analyzer.generate_summary_table()
    
    # Plot all compounds on single plot
    analyzer.plot_all_compounds()


if __name__ == '__main__':
    main()

