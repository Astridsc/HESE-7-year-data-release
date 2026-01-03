"""
Script to scan over E_cutoff values and plot LLH or TS vs E_cutoff.

This script runs HESE_fit.py multiple times with different fixed E_cutoff values
and collects the results to create a plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import argparse
import json
import re
import time as time_module

def run_fit_at_cutoff(E_cutoff, model="cutoff", fix_cutoff=True, **kwargs):
    """
    Run HESE_fit.py for a specific E_cutoff value.
    
    Parameters:
    -----------
    E_cutoff : float
        Cutoff energy value in GeV
    model : str
        Model to use ("cutoff" or "spl")
    fix_cutoff : bool
        Whether to fix cutoff_energy parameter
    **kwargs : dict
        Additional arguments to pass to HESE_fit.py
    
    Returns:
    --------
    best_llh : float
        Best-fit negative log-likelihood (overall best)
    best_params : dict
        Best-fit parameters (overall best)
    fit_results : list
        List of both fit results (for both epsilon_DOM intervals)
        Each element is a dict with keys: 'llh', 'params', 'interval', 'epsilon_range'
    fit_time : float
        Time taken for the fit in seconds (None if not found)
    """
    # Build command
    cmd = ["python", "HESE_fit.py", "--model", model, "--cutoff_energy", str(E_cutoff)]
    
    if fix_cutoff:
        cmd.append("--fix_cutoff_energy")
    
    # Add any additional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.append(f"--{key}")
                cmd.append(str(value))
    
    # Run the fit
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output to extract best-fit LLH and both interval fits
        output_lines = result.stdout.split('\n')
        best_llh = None
        best_params = {}
        fit_results = []
        fit_time = None
        
        # Parse the overall best fit
        in_best_fit_params = False
        for i, line in enumerate(output_lines):
            if "Best Fit -LLH:" in line:
                best_llh = float(line.split("Best Fit -LLH:")[1].strip())
                in_best_fit_params = True
            elif in_best_fit_params and "Best Fit Paramters:" in line:
                continue
            elif in_best_fit_params and "\t" in line and ":" in line:
                # Parse parameter lines like "\tastro_gamma: \t2.5"
                parts = line.strip().split(":")
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    try:
                        param_value = float(parts[1].strip())
                        best_params[param_name] = param_value
                    except ValueError:
                        pass
            elif in_best_fit_params and line.strip() == "":
                # Empty line after parameters, stop parsing best fit
                in_best_fit_params = False
        
        # Parse both epsilon_DOM interval fits
        in_interval_fit = False
        current_fit = None
        for i, line in enumerate(output_lines):
            if "Fit " in line and "epsilon_DOM" in line:
                # Save previous fit if exists
                if current_fit and current_fit['llh'] is not None:
                    fit_results.append(current_fit)
                # Start of a new interval fit
                in_interval_fit = True
                current_fit = {
                    'llh': None,
                    'params': {},
                    'interval': None,
                    'epsilon_range': None
                }
                # Extract interval info
                if "low" in line.lower():
                    current_fit['interval'] = 'low'
                elif "high" in line.lower():
                    current_fit['interval'] = 'high'
                # Extract epsilon range
                if "[0.8, 0.99]" in line:
                    current_fit['epsilon_range'] = "[0.8, 0.99]"
                elif "[0.99, 1.25]" in line:
                    current_fit['epsilon_range'] = "[0.99, 1.25]"
            elif in_interval_fit and "  -LLH:" in line:
                try:
                    current_fit['llh'] = float(line.split("  -LLH:")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif in_interval_fit and "  Parameters:" in line:
                continue
            elif in_interval_fit and "\t" in line and ":" in line and "Interval:" not in line and "Epsilon_DOM range:" not in line:
                # Parse parameter lines (but skip the interval/range lines)
                parts = line.strip().split(":")
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    try:
                        param_value = float(parts[1].strip())
                        current_fit['params'][param_name] = param_value
                    except ValueError:
                        pass
            elif in_interval_fit and ("  Interval:" in line or "  Epsilon_DOM range:" in line):
                # Skip these lines, already extracted from header
                continue
            elif in_interval_fit and ("===" in line or "LLH list:" in line):
                # End of interval fits section, save current fit
                if current_fit and current_fit['llh'] is not None:
                    fit_results.append(current_fit)
                in_interval_fit = False
                current_fit = None
        
        # Save last fit if we're still in one
        if current_fit and current_fit['llh'] is not None:
            fit_results.append(current_fit)
        
        # If we didn't find interval fits (e.g., epsilon_DOM is fixed), create a single result
        if len(fit_results) == 0:
            fit_results = [{
                'llh': best_llh,
                'params': best_params,
                'interval': 'single',
                'epsilon_range': 'N/A'
            }]
        
        # Parse fit time
        for line in output_lines:
            if "Fit took" in line and "seconds" in line:
                try:
                    # Extract number from "Fit took X seconds" or "Fit took X.XX seconds"
                    match = re.search(r'Fit took\s+([\d.]+)\s+seconds', line)
                    if match:
                        fit_time = float(match.group(1))
                except (ValueError, AttributeError):
                    pass
                break
        
        return best_llh, best_params, fit_results, fit_time
        
    except subprocess.CalledProcessError as e:
        print(f"Error running fit for E_cutoff={E_cutoff:.2e} GeV:")
        print(e.stderr)
        return None, None, [], None


def main():
    parser = argparse.ArgumentParser(
        description="Scan over E_cutoff values and plot LLH or TS vs E_cutoff"
    )
    parser.add_argument(
        "--E_cutoff_min", type=float, default=1e5,
        help="Minimum E_cutoff value in GeV (default: 1e5)"
    )
    parser.add_argument(
        "--E_cutoff_max", type=float, default=1e7,
        help="Maximum E_cutoff value in GeV (default: 1e7)"
    )
    parser.add_argument(
        "--npoints", type=int, default=20,
        help="Number of points to scan (default: 20)"
    )
    parser.add_argument(
        "--log_space", action="store_true", default=True,
        help="Use log-spaced E_cutoff values (default: True)"
    )
    parser.add_argument(
        "--plot_ts", action="store_true",
        help="Plot TS (test statistic) instead of LLH. Requires SPL best-fit LLH."
    )
    parser.add_argument(
        "--spl_llh", type=float, default=None,
        help="SPL best-fit -LLH value (required if --plot_ts). If not provided, will run SPL fit first."
    )
    parser.add_argument(
        "--output", type=str, default="cutoff_scan.png",
        help="Output plot filename (default: cutoff_scan.png)"
    )
    parser.add_argument(
        "--save_data", type=str, default=None,
        help="Save scan data to JSON file (optional)"
    )
    
    args = parser.parse_args()
    
    # Generate E_cutoff values
    if args.log_space:
        E_cutoff_values = np.logspace(
            np.log10(args.E_cutoff_min),
            np.log10(args.E_cutoff_max),
            args.npoints
        )
    else:
        E_cutoff_values = np.linspace(
            args.E_cutoff_min,
            args.E_cutoff_max,
            args.npoints
        )
    
    print(f"Scanning {len(E_cutoff_values)} E_cutoff values from {args.E_cutoff_min:.2e} to {args.E_cutoff_max:.2e} GeV")
    if args.log_space:
        print("Using log-spaced values")
    print()
    
    # If plotting TS, get SPL best-fit LLH first
    if args.plot_ts:
        if args.spl_llh is None:
            print("Running SPL fit to get reference LLH...")
            spl_llh, _, _, spl_time = run_fit_at_cutoff(1e5, model="spl", fix_cutoff=True)
            if spl_llh is None:
                print("Error: Could not get SPL best-fit LLH")
                sys.exit(1)
            print(f"SPL best-fit -LLH: {spl_llh:.6f}")
            if spl_time is not None:
                print(f"SPL fit took {spl_time:.1f} seconds ({spl_time/60:.1f} minutes)")
            print()
        else:
            spl_llh = args.spl_llh
    
    # Scan over E_cutoff values
    llh_values = []
    ts_values = []
    all_params = []
    all_fit_results = []  # Store both interval fits for each point
    fit_times = []  # Store fit times
    
    total_start_time = time_module.time()
    
    for i, E_cutoff in enumerate(E_cutoff_values):
        print(f"[{i+1}/{len(E_cutoff_values)}] Running fit for E_cutoff = {E_cutoff:.2e} GeV...")
        point_start_time = time_module.time()
        llh, params, fit_results, fit_time = run_fit_at_cutoff(E_cutoff, model="cutoff", fix_cutoff=True)
        point_elapsed = time_module.time() - point_start_time
        
        if llh is not None:
            llh_values.append(llh)
            all_params.append(params)
            all_fit_results.append(fit_results)
            fit_times.append(fit_time if fit_time is not None else point_elapsed)
            
            if args.plot_ts:
                # TS = 2 * (LLH_cutoff - LLH_spl)
                ts = 2.0 * (llh - spl_llh)
                ts_values.append(ts)
                print(f"  -LLH: {llh:.6f}, TS: {ts:.6f}")
            else:
                print(f"  -LLH: {llh:.6f}")
            
            # Print timing information
            if fit_time is not None:
                print(f"  Fit took {fit_time:.1f} seconds ({fit_time/60:.1f} minutes)")
            else:
                print(f"  Elapsed time: {point_elapsed:.1f} seconds ({point_elapsed/60:.1f} minutes)")
            
            # Print both interval fits if available
            if len(fit_results) == 2:
                print(f"  Interval fits:")
                for fit in fit_results:
                    print(f"    {fit['interval']} ({fit['epsilon_range']}): -LLH = {fit['llh']:.6f}")
            
            # Print estimated time remaining
            if i > 0:
                avg_time = np.mean(fit_times)
                remaining_points = len(E_cutoff_values) - (i + 1)
                estimated_remaining = avg_time * remaining_points
                print(f"  Estimated time remaining: {estimated_remaining/60:.1f} minutes ({estimated_remaining/3600:.1f} hours)")
        else:
            print(f"  Failed to get result")
            llh_values.append(np.nan)
            all_fit_results.append([])
            fit_times.append(point_elapsed)
            if args.plot_ts:
                ts_values.append(np.nan)
        print()
    
    total_elapsed = time_module.time() - total_start_time
    print(f"\nTotal scan time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    if fit_times:
        print(f"Average time per point: {np.mean(fit_times)/60:.1f} minutes")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if args.plot_ts:
        ax.plot(E_cutoff_values, ts_values, 'o-', linewidth=2, markersize=6)
        ax.set_ylabel('Test Statistic TS', fontsize=12)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='TS=0 (SPL)')
        ax.legend()
    else:
        ax.plot(E_cutoff_values, llh_values, 'o-', linewidth=2, markersize=6)
        ax.set_ylabel('Best-fit -LLH', fontsize=12)
    
    ax.set_xlabel('E_cutoff (GeV)', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Profile Likelihood Scan: Cutoff Energy', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    
    # Save data if requested
    if args.save_data:
        data = {
            'E_cutoff_values': E_cutoff_values.tolist(),
            'llh_values': llh_values,
            'all_params': all_params,
            'all_fit_results': all_fit_results,  # Both interval fits for each point
            'fit_times': fit_times  # Time taken for each fit in seconds
        }
        if args.plot_ts:
            data['ts_values'] = ts_values
            data['spl_llh'] = spl_llh
        
        with open(args.save_data, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {args.save_data}")


if __name__ == "__main__":
    main()

