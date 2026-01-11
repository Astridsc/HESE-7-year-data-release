#!/usr/bin/env python3
"""
Plot correlations between any two parameters from results.jsonl files.

This script reads results.jsonl files from Cluster_package/HESE/ directory
(including subdirectories) and allows you to plot any two parameters against
each other, showing all documented points. You can also plot any parameter
against the LLH (log-likelihood) value.

Usage:
    python plot_parameter_correlations.py [options]
    
Examples:
    # Interactive mode - choose parameters from menu
    python plot_parameter_correlations.py
    
    # Plot specific parameters
    python plot_parameter_correlations.py --param1 astro_gamma --param2 astro_norm
    
    # Plot parameter against LLH
    python plot_parameter_correlations.py --param1 astro_gamma --param2 llh
    
    # Plot with custom output file
    python plot_parameter_correlations.py --param1 Mphi --param2 g --output correlation_Mphi_g.png
    
    # Filter out failed fits (inf LLH)
    python plot_parameter_correlations.py --param1 astro_gamma --param2 astro_norm --exclude_inf
    
    # Color by LLH value (when not plotting LLH on an axis)
    python plot_parameter_correlations.py --param1 Mphi --param2 g --color_by_llh
"""

import json
import ast
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from glob import glob
from collections import defaultdict


def find_results_files(search_dir, no_subdirs=False):
    """Find all .jsonl files in directory and optionally subdirectories."""
    results_files = []
    search_path = Path(search_dir)
    
    if not search_path.exists():
        return results_files
    
    # Search for all .jsonl files in the directory itself
    if no_subdirs:
        for jsonl_file in search_path.glob("*.jsonl"):
            results_files.append(str(jsonl_file))
    else:
        # Search in subdirectories too
        for jsonl_file in search_path.rglob("*.jsonl"):
            results_files.append(str(jsonl_file))
    
    return sorted(set(results_files))  # Remove duplicates and sort


def load_results(results_files):
    """Load all results from JSONL files."""
    all_results = []
    
    for results_file in results_files:
        if not os.path.exists(results_file):
            print(f"Warning: {results_file} not found, skipping")
            continue
        
        if not os.path.isfile(results_file):
            print(f"Warning: {results_file} is not a file, skipping")
            continue
        
        print(f"Loading {results_file}...")
        try:
            with open(results_file, "r") as f:
                file_results = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        result = json.loads(line)
                        result["_source_file"] = results_file  # Track source
                        all_results.append(result)
                        file_results += 1
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON in {results_file}, line {line_num}: {e}")
                print(f"  Loaded {file_results} results from {os.path.basename(results_file)}")
        except Exception as e:
            print(f"Error reading {results_file}: {e}")
            continue
    
    print(f"Loaded {len(all_results)} total results from {len(results_files)} file(s)")
    return all_results


def extract_parameters(results):
    """Extract all unique parameter names from results."""
    param_set = set()
    
    for result in results:
        if "params" in result and isinstance(result["params"], dict):
            param_set.update(result["params"].keys())
    
    # Add "llh" as a special parameter option
    param_set.add("llh")
    
    return sorted(param_set)


def get_parameter_values(results, param_name, exclude_inf=False):
    """Extract values for a specific parameter from all results.
    
    Special case: if param_name is "llh", extracts LLH values directly.
    """
    values = []
    llh_values = []
    valid_indices = []
    
    # Special handling for LLH parameter
    if param_name == "llh":
        for idx, result in enumerate(results):
            llh = result.get("llh", None)
            if llh is None:
                continue
            
            # Convert "inf" string to np.inf
            if llh == "inf" or (isinstance(llh, str) and llh.lower() == "inf"):
                if exclude_inf:
                    continue
                values.append(np.inf)
            elif isinstance(llh, (int, float)):
                if exclude_inf and np.isinf(llh):
                    continue
                values.append(float(llh))
            else:
                continue
            
            llh_values.append(llh)
            valid_indices.append(idx)
        
        return np.array(values), llh_values, valid_indices
    
    # Regular parameter extraction
    for idx, result in enumerate(results):
        # Check if this result has valid LLH (if filtering)
        if exclude_inf:
            llh = result.get("llh", None)
            if llh == "inf" or (isinstance(llh, (int, float)) and np.isinf(llh)):
                continue
        
        # Get parameter value
        if "params" in result and isinstance(result["params"], dict):
            if param_name in result["params"]:
                value = result["params"][param_name]
                if value is not None:
                    try:
                        values.append(float(value))
                        llh_values.append(result.get("llh", None))
                        valid_indices.append(idx)
                    except (ValueError, TypeError):
                        continue
    
    return np.array(values), llh_values, valid_indices


def plot_correlation(results, param1_name, param2_name, exclude_inf=False, 
                     color_by_llh=False, output_file=None, show_plot=True,
                     title=None, xlabel=None, ylabel=None, fill=False, y1_axis=None, y2_axis=None, color='mediumaquamarine', best_point=None, label_best_fit=None):
    """Plot correlation between two parameters.
    
    Args:
        best_point: If None, no best point is marked. If a dict, should contain:
            - 'p1_name': parameter 1 name (str)
            - 'p1_val': parameter 1 value
            - 'p2_name': parameter 2 name (str)
            - 'p2_val': parameter 2 value
            - 'llh': LLH value
            - 'label': label string for the best fit point
            If True (for backward compatibility), automatically calculates from data.
    
    When best_point is not None:
    - If color_by_llh=True, uses test statistic TS = 2*(LLH - min(LLH)) for colorbar
    - Marks the best fit point on the plot with a red star
    """
    
    # Extract values
    param1_values, llh1, idx1 = get_parameter_values(results, param1_name, exclude_inf)
    param2_values, llh2, idx2 = get_parameter_values(results, param2_name, exclude_inf)
    
    # Find common indices (results that have both parameters)
    common_indices = set(idx1) & set(idx2)
    
    if len(common_indices) == 0:
        print(f"Error: No results found with both {param1_name} and {param2_name}")
        return
    
    # Filter to common indices
    param1_filtered = []
    param2_filtered = []
    llh_filtered = []
    
    for idx in common_indices:
        # Find position in param1 array
        if idx in idx1:
            pos1 = idx1.index(idx)
            param1_filtered.append(param1_values[pos1])
            llh_filtered.append(llh1[pos1])
        # Find position in param2 array
        if idx in idx2:
            pos2 = idx2.index(idx)
            param2_filtered.append(param2_values[pos2])
    
    param1_array = np.array(param1_filtered)
    param2_array = np.array(param2_filtered)
    
    # Handle LLH values (convert "inf" strings to np.inf for filtering)
    llh_array = []
    for llh in llh_filtered:
        if llh == "inf" or (isinstance(llh, str) and llh.lower() == "inf"):
            llh_array.append(np.inf)
        elif isinstance(llh, (int, float)):
            llh_array.append(float(llh))
        else:
            llh_array.append(np.nan)
    llh_array = np.array(llh_array)
    
    # Determine if one axis is LLH
    is_llh_plot = (param1_name == "llh" or param2_name == "llh")
    
    # Filter out inf LLH values if one axis is LLH (can't plot inf)
    if is_llh_plot:
        if param1_name == "llh":
            finite_mask = np.isfinite(param1_array)
            if not np.all(finite_mask):
                print(f"Note: {np.sum(~finite_mask)} points with inf LLH excluded from plot")
        else:  # param2_name == "llh"
            finite_mask = np.isfinite(param2_array)
            if not np.all(finite_mask):
                print(f"Note: {np.sum(~finite_mask)} points with inf LLH excluded from plot")
        
        if not np.all(finite_mask):
            param1_array = param1_array[finite_mask]
            param2_array = param2_array[finite_mask]
            llh_array = llh_array[finite_mask]
    elif exclude_inf:
        # Filter out inf if requested (but not if one of the parameters IS llh)
        valid_mask = np.isfinite(llh_array)
        param1_array = param1_array[valid_mask]
        param2_array = param2_array[valid_mask]
        llh_array = llh_array[valid_mask]
    
    if len(param1_array) == 0:
        print(f"Error: No valid data points after filtering")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Don't color by LLH if we're already plotting LLH on one axis
    if color_by_llh and not is_llh_plot and len(llh_array) > 0 and np.any(np.isfinite(llh_array)):
        # Color by LLH or TS (test statistic)
        finite_mask = np.isfinite(llh_array)
        if np.any(finite_mask):
            # Always calculate minimum LLH for TS calculation
            
            
            # Calculate values for coloring
            if best_point is not None:
                # Use test statistic: TS = 2*(LLH - min(LLH))
                # If best_point is a dict, use its llh value; otherwise use min from data
                if isinstance(best_point, dict) and 'llh' in best_point:
                    min_llh = best_point['llh']
                #color_values = 2.0 * (llh_array[finite_mask] - min_llh)
                #colorbar_label = 'TS = 2Δ(-LLH)'
            else:
                min_llh = np.min(llh_array[finite_mask])
                # Use test statistic even when best_point is None
                # Use minimum LLH from the data
            color_values = 2.0 * (llh_array[finite_mask] - min_llh)
            colorbar_label = r'$TS = 2\Delta(LLH-LLH_{min})$'
            
            scatter = ax.scatter(
                param1_array[finite_mask],
                param2_array[finite_mask],
                c=color_values,
                cmap='viridis',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(colorbar_label, rotation=90, labelpad=20)
            
            # Also plot infinite LLH points in gray
            inf_mask = ~finite_mask
            if np.any(inf_mask):
                ax.scatter(
                    param1_array[inf_mask],
                    param2_array[inf_mask],
                    c='gray',
                    s=50,
                    alpha=0.3,
                    edgecolors='black',
                    linewidths=0.5,
                    label='Failed fits (inf LLH)'
                )
                ax.legend()
        else:
            # All inf, just plot in gray
            ax.scatter(
                param1_array,
                param2_array,
                c='gray',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5
            )
    else:
        # Simple scatter plot
        ax.scatter(
            param1_array,
            param2_array,
            s=50,
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5
        )
    
    if fill:
        # Use axhspan to fill the entire x-axis between y1_axis and y2_axis
        ax.axhspan(y1_axis, y2_axis, color=color, alpha=0.3, zorder=0)
    
    # Mark best fit point if requested
    if best_point is not None:
        # If best_point is True, calculate from data
        if best_point is True:
            if len(llh_array) > 0 and np.any(np.isfinite(llh_array)):
                finite_mask = np.isfinite(llh_array)
                if np.any(finite_mask):
                    best_idx = np.argmin(llh_array[finite_mask])
                    best_point = {
                        'p1_name': str(param1_name),
                        'p1_val': param1_array[finite_mask][best_idx],
                        'p2_name': str(param2_name),
                        'p2_val': param2_array[finite_mask][best_idx],
                        'llh': llh_array[finite_mask][best_idx],
                        'label': f'Best fit: {param1_name} = {param1_array[finite_mask][best_idx]:.4g}, {param2_name} = {param2_array[finite_mask][best_idx]:.4g}, -LLH = {llh_array[finite_mask][best_idx]:.4g}'
                    }
        
        # If best_point is a dict, use its values
        if isinstance(best_point, dict):
            p1_val = best_point.get('p1_val')
            p2_val = best_point.get('p2_val')
            
            # Check if best_point dict has label_best_fit key (overrides parameter)
            if 'label_best_fit' in best_point:
                label_best_fit = best_point['label_best_fit']
            
            if label_best_fit is None:
                # Check if best_point dict has a 'label' key
                dict_label = best_point.get('label')
                if dict_label is False:
                    label_best_fit = None
                elif dict_label is not None:
                    label_best_fit = dict_label
                else:
                    label_best_fit = f"Best fit: {best_point.get('p1_name', param1_name)} = {p1_val:.4g}, {best_point.get('p2_name', param2_name)} = {p2_val:.4g}"
            elif label_best_fit is False:
                label_best_fit = None
            
            # Only add label if it's not None
            #if label_best_fit is not None:
            ax.scatter(p1_val, p2_val, color='red', s=100, label=label_best_fit, marker='*', linewidths=0.5, zorder=10)
            """    ax.legend()
            else:
                # Plot without label (won't appear in legend)
                ax.scatter(p1_val, p2_val, color='red', s=100, marker='*', linewidths=0.5, zorder=10)"""
    # Set labels - use custom labels if provided, otherwise use defaults
    if xlabel is None:
        xlabel = "-LLH" if param1_name == "llh" else param1_name
    if ylabel is None:
        ylabel = "-LLH" if param2_name == "llh" else param2_name
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Title - use custom title if provided, otherwise use default
    if title is None:
        if is_llh_plot:
            other_param = param2_name if param1_name == "llh" else param1_name
            title = f'Parameter vs -LLH: {other_param}'
        else:
            title = f'Correlation: {param1_name} vs {param2_name}'
    
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3)
    
    # Use log scale if values span large range
    # For LLH axis, don't use log scale (LLH is typically linear)
    if param1_name != "llh" and len(param1_array) > 0 and param1_array.min() > 0:
        if param1_array.max() / param1_array.min() > 100:
            ax.set_xscale('log')
    if param2_name != "llh" and len(param2_array) > 0 and param2_array.min() > 0:
        if param2_array.max() / param2_array.min() > 100:
            ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def interactive_mode(results):
    """Interactive mode to select parameters from menu."""
    params = extract_parameters(results)
    
    if len(params) == 0:
        print("Error: No parameters found in results")
        return
    
    print("\nAvailable parameters:")
    for i, param in enumerate(params, 1):
        print(f"  {i:2d}. {param}")
    
    print("\nSelect two parameters to plot:")
    
    try:
        choice1 = int(input(f"First parameter (1-{len(params)}): "))
        if choice1 < 1 or choice1 > len(params):
            print("Invalid choice")
            return
        param1 = params[choice1 - 1]
        
        choice2 = int(input(f"Second parameter (1-{len(params)}): "))
        if choice2 < 1 or choice2 > len(params):
            print("Invalid choice")
            return
        param2 = params[choice2 - 1]
        
        exclude_inf = input("Exclude failed fits (inf LLH)? [y/N]: ").lower().startswith('y')
        color_by_llh = input("Color by LLH value? [y/N]: ").lower().startswith('y')
        
        best_point = input("Mark best fit point and use TS = 2*(LLH - min(LLH)) for colorbar (if coloring by LLH)? [y/N]: ").lower().startswith('y')
        
        output_file = input("Output filename (press Enter for default): ").strip()
        if not output_file:
            output_file = f"correlation_{param1}_vs_{param2}.png"
        
        """save = input("Save plot to file? [Y/n]: ").strip().lower()
        save = save != 'n'  # Default to True unless user explicitly says 'n'
        """
        # Optional custom labels and title
        title = input("Custom title (press Enter for default, supports LaTeX math like r'$M_\\phi$ vs $m_{tot}$'): ").strip()
        if not title:
            title = None
        
        xlabel = input("Custom x-axis label (press Enter for default, supports LaTeX math): ").strip()
        if not xlabel:
            xlabel = None
        
        ylabel = input("Custom y-axis label (press Enter for default, supports LaTeX math): ").strip()
        if not ylabel:
            ylabel = None
        
        # Convert boolean to appropriate value for best_point
        best_point_val = True if best_point else None
        plot_correlation(results, param1, param2, exclude_inf=exclude_inf,
                        color_by_llh=color_by_llh, output_file=output_file,
                        show_plot=True, title=title, xlabel=xlabel, ylabel=ylabel,
                        best_point=best_point_val)
        
    except (ValueError, KeyboardInterrupt):
        print("\nCancelled")


def main():
    parser = argparse.ArgumentParser(
        description="Plot correlations between parameters from results.jsonl files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--search_dir",
        type=str,
        default=None,
        help="Directory to search for results.jsonl files (default: current directory). If specified, only searches in that directory unless --include_subdirs is used."
    )
    
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Specific results.jsonl file to use (overrides --search_dir)"
    )
    
    parser.add_argument(
        "--include_subdirs",
        action="store_true",
        help="When using --search_dir, also search in subdirectories (default: only search in specified directory)"
    )
    
    parser.add_argument(
        "--param1",
        type=str,
        default=None,
        help="First parameter name (if not provided, interactive mode)"
    )
    
    parser.add_argument(
        "--param2",
        type=str,
        default=None,
        help="Second parameter name (if not provided, interactive mode)"
    )
    
    parser.add_argument(
        "--exclude_inf",
        action="store_true",
        help="Exclude failed fits (inf LLH) from plot"
    )
    
    parser.add_argument(
        "--color_by_llh",
        action="store_true",
        help="Color points by LLH value"
    )
    
    parser.add_argument(
        "--best_point",
        type=str,
        nargs='?',
        const=True,
        default=None,
        help="Mark best fit point on plot. Use --best_point to auto-calculate, or --best_point '{\"p1_name\":\"mntot\",\"p1_val\":0.066,\"p2_name\":\"Mphi\",\"p2_val\":2.5,\"llh\":120.6}' to provide values (JSON format)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output filename for plot (default: auto-generated or show interactively)"
    )
    
    parser.add_argument(
        "--list_params",
        action="store_true",
        help="List all available parameters and exit"
    )
    
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't display plot (only save to file)"
    )
    
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title (supports LaTeX math, e.g., r'$M_\\phi$ vs $m_{tot}$')"
    )
    
    parser.add_argument(
        "--xlabel",
        type=str,
        default=None,
        help="Custom x-axis label (supports LaTeX math, e.g., r'$M_\\phi$ (GeV)')"
    )
    
    parser.add_argument(
        "--ylabel",
        type=str,
        default=None,
        help="Custom y-axis label (supports LaTeX math, e.g., r'$m_{tot}$ (eV)')"
    )
    
    parser.add_argument(
        "--fill",
        action="store_true",
        default=False,
        help="Fill area between y1_axis and y2_axis"
    )
    
    parser.add_argument(
        "--y1_axis",
        type=float,
        default=None,
        help="Lower y-axis value for fill area"
    )
    
    parser.add_argument(
        "--y2_axis",
        type=float,
        default=None,
        help="Upper y-axis value for fill area"
    )
    
    parser.add_argument(
        "--label_best_fit",
        type=str,
        default=None,
        help="Label for best fit point. Use 'False' (as string) to hide the label, or provide a custom label string"
    )
    
    args = parser.parse_args()
    
    # Find and load results files
    if args.results_file:
        # Use specific file if provided
        # If search_dir is also specified and results_file is relative, try it in search_dir first
        results_file_path = None
        
        if args.search_dir and not os.path.isabs(args.results_file):
            # Try relative to search_dir first
            search_dir_abs = os.path.abspath(args.search_dir)
            candidate_path = os.path.join(search_dir_abs, args.results_file)
            if os.path.exists(candidate_path):
                results_file_path = os.path.abspath(candidate_path)
        
        # If not found yet, try as-is
        if results_file_path is None and os.path.exists(args.results_file):
            results_file_path = os.path.abspath(args.results_file)
        
        # If still not found, try relative to current directory
        if results_file_path is None and os.path.exists(os.path.join(".", args.results_file)):
            results_file_path = os.path.abspath(os.path.join(".", args.results_file))
        
        # Last resort: try as absolute path (will fail if doesn't exist)
        if results_file_path is None:
            results_file_path = os.path.abspath(args.results_file)
        
        if not os.path.exists(results_file_path):
            print(f"Error: Results file not found: {args.results_file}")
            print(f"  Tried paths:")
            if args.search_dir:
                candidate = os.path.join(os.path.abspath(args.search_dir), args.results_file)
                print(f"    - {candidate} (exists: {os.path.exists(candidate)})")
            abs_path = os.path.abspath(args.results_file)
            print(f"    - {abs_path} (exists: {os.path.exists(abs_path)})")
            rel_path = os.path.join(os.getcwd(), args.results_file)
            print(f"    - {rel_path} (exists: {os.path.exists(rel_path)})")
            print(f"  Current working directory: {os.getcwd()}")
            if args.search_dir:
                print(f"  Search directory: {os.path.abspath(args.search_dir)}")
                print(f"  Search directory exists: {os.path.exists(os.path.abspath(args.search_dir))}")
            return
        
        results_files = [results_file_path]
        print(f"Using specified results file: {results_file_path}")
    else:
        # Search for files
        if args.search_dir is None:
            search_dir = os.path.abspath(".")
        else:
            search_dir = os.path.abspath(args.search_dir)
        
        # If search_dir is specified, default to no_subdirs unless explicitly requested
        no_subdirs = not args.include_subdirs if args.search_dir is not None else False
        
        results_files = find_results_files(search_dir, no_subdirs=no_subdirs)
        
        if len(results_files) == 0:
            print(f"Error: No .jsonl files found in {search_dir}")
            if no_subdirs:
                print("  (Note: Only searching in the specified directory. Use --include_subdirs to search subdirectories)")
            return
        
        print(f"Found {len(results_files)} results.jsonl file(s):")
        for f in results_files:
            print(f"  - {f}")
    
    results = load_results(results_files)
    
    if len(results) == 0:
        print("Error: No results loaded")
        return
    
    # List parameters if requested
    if args.list_params:
        params = extract_parameters(results)
        print("\nAvailable parameters:")
        for param in params:
            print(f"  - {param}")
        return
    
    # Plot or interactive mode
    if args.param1 and args.param2:
        # Prompt for options not provided via command line
        color_by_llh = args.color_by_llh
        if not args.color_by_llh and not args.no_show:
            # Only prompt if not in no_show mode (batch mode)
            try:
                color_input = input("Color by LLH value? [y/N]: ").strip().lower()
                color_by_llh = color_input.startswith('y') if color_input else False
            except (EOFError, KeyboardInterrupt):
                color_by_llh = False
        
        output_file = args.output_file
        if not output_file and not args.no_show:
            # Only prompt if not in no_show mode (batch mode)
            try:
                output_input = input("Output filename (press Enter for default): ").strip()
                if not output_input:
                    output_file = f"correlation_{args.param1}_vs_{args.param2}.png"
                else:
                    output_file = output_input
            except (EOFError, KeyboardInterrupt):
                output_file = f"correlation_{args.param1}_vs_{args.param2}.png"
        elif not output_file:
            # In batch mode (no_show), use default filename
            output_file = f"correlation_{args.param1}_vs_{args.param2}.png"
        
        # Parse best_point argument
        best_point_val = None
        if args.best_point:
            if args.best_point is True:
                # Flag used without value - auto-calculate
                best_point_val = True
            else:
                # Try to parse as JSON dictionary first, then as Python literal
                best_point_dict = None
                try:
                    best_point_dict = json.loads(args.best_point)
                except json.JSONDecodeError:
                    # Try parsing as Python literal (handles single quotes)
                    try:
                        best_point_dict = ast.literal_eval(args.best_point)
                    except (ValueError, SyntaxError):
                        pass
                
                if best_point_dict is not None and isinstance(best_point_dict, dict):
                    # Map parameter names from the dict keys to p1_name, p2_name format
                    # If keys are parameter names directly, use them
                    if args.param1 in best_point_dict and args.param2 in best_point_dict:
                        llh_val = best_point_dict.get('llh', best_point_dict.get('-llh', None))
                        llh_str = f"{llh_val:.4g}" if llh_val is not None else "N/A"
                        best_point_val = {
                            'p1_name': str(args.param1),
                            'p1_val': best_point_dict[args.param1],
                            'p2_name': str(args.param2),
                            'p2_val': best_point_dict[args.param2],
                            'llh': llh_val,
                            'label': f"Best fit: {args.param1} = {best_point_dict[args.param1]:.4g}, {args.param2} = {best_point_dict[args.param2]:.4g}, -LLH = {llh_str}"
                        }
                    else:
                        # Assume it's already in the correct format
                        best_point_val = best_point_dict
                else:
                    print(f"Warning: Could not parse --best_point as dictionary: {args.best_point}")
                    print("Using auto-calculation instead")
                    best_point_val = True
        
        plot_correlation(
            results,
            args.param1,
            args.param2,
            exclude_inf=args.exclude_inf,
            color_by_llh=color_by_llh,
            output_file=output_file,
            show_plot=not args.no_show,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            best_point=best_point_val,
            fill=args.fill,
            y1_axis=args.y1_axis,
            y2_axis=args.y2_axis,
            label_best_fit=False if args.label_best_fit and args.label_best_fit.lower() == 'false' else args.label_best_fit
        )
    else:
        interactive_mode(results)


if __name__ == "__main__":
    main()

