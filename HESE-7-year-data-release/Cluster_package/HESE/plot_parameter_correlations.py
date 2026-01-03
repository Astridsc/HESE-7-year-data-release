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
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from glob import glob
from collections import defaultdict


def find_results_files(search_dir, no_subdirs=False):
    """Find all results.jsonl files in directory and optionally subdirectories."""
    results_files = []
    search_path = Path(search_dir)
    
    # Search in the directory itself
    if (search_path / "results.jsonl").exists():
        results_files.append(str(search_path / "results.jsonl"))
    
    # Search in subdirectories if requested
    if not no_subdirs:
        for jsonl_file in search_path.rglob("results.jsonl"):
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
                     color_by_llh=False, output_file=None, show_plot=True):
    """Plot correlation between two parameters."""
    
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
        # Color by LLH
        finite_mask = np.isfinite(llh_array)
        if np.any(finite_mask):
            scatter = ax.scatter(
                param1_array[finite_mask],
                param2_array[finite_mask],
                c=llh_array[finite_mask],
                cmap='viridis',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('-LLH', rotation=270, labelpad=20)
            
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
    
    # Set labels - use proper label for LLH
    xlabel = "-LLH" if param1_name == "llh" else param1_name
    ylabel = "-LLH" if param2_name == "llh" else param2_name
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Title
    if is_llh_plot:
        other_param = param2_name if param1_name == "llh" else param1_name
        ax.set_title(f'Parameter vs -LLH: {other_param}\n({len(param1_array)} points)', 
                     fontsize=14)
    else:
        ax.set_title(f'Correlation: {param1_name} vs {param2_name}\n({len(param1_array)} points)', 
                     fontsize=14)
    
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
        
        output_file = input("Output filename (press Enter for default): ").strip()
        if not output_file:
            output_file = f"correlation_{param1}_vs_{param2}.png"
        
        plot_correlation(results, param1, param2, exclude_inf=exclude_inf,
                        color_by_llh=color_by_llh, output_file=output_file)
        
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
        "--output",
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
                print(f"    - {os.path.join(os.path.abspath(args.search_dir), args.results_file)}")
            print(f"    - {os.path.abspath(args.results_file)}")
            print(f"    - {os.path.join(os.getcwd(), args.results_file)}")
            print(f"  Current working directory: {os.getcwd()}")
            if args.search_dir:
                print(f"  Search directory: {os.path.abspath(args.search_dir)}")
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
            print(f"Error: No results.jsonl files found in {search_dir}")
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
    
    # Determine output filename
    output_file = args.output
    if args.param1 and args.param2 and not output_file:
        output_file = f"correlation_{args.param1}_vs_{args.param2}.png"
    
    # Plot or interactive mode
    if args.param1 and args.param2:
        plot_correlation(
            results,
            args.param1,
            args.param2,
            exclude_inf=args.exclude_inf,
            color_by_llh=args.color_by_llh,
            output_file=output_file,
            show_plot=not args.no_show
        )
    else:
        interactive_mode(results)


if __name__ == "__main__":
    main()

