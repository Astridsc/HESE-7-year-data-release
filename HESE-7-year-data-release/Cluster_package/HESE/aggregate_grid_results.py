"""
Aggregate results from parallel grid scan and create visualization.

Run this after all cluster jobs complete to combine results and plot.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import glob


def aggregate_results(output_dir):
    """Aggregate all result files from cluster run."""
    result_files = sorted(glob.glob(str(Path(output_dir) / "result_*.json")))
    
    if not result_files:
        print(f"No result files found in {output_dir}")
        return None
    
    # Read first file to get grid structure
    with open(result_files[0], "r") as f:
        first_result = json.load(f)
    
    param1_name = first_result["param1_name"]
    param2_name = first_result["param2_name"]
    
    # Collect all results
    results = []
    for result_file in result_files:
        with open(result_file, "r") as f:
            results.append(json.load(f))
    
    # Find grid dimensions
    max_i = max(r["grid_index"][0] for r in results)
    max_j = max(r["grid_index"][1] for r in results)
    n1 = max_i + 1
    n2 = max_j + 1
    
    # Build grid arrays
    param1_values = sorted(set(r["param1_value"] for r in results))
    param2_values = sorted(set(r["param2_value"] for r in results))
    
    # Create LLH grid
    llh_grid = np.full((n1, n2), np.nan)
    
    for result in results:
        i, j = result["grid_index"]
        if result["llh"] is not None:
            llh_grid[i, j] = result["llh"]
    
    return {
        "param1_name": param1_name,
        "param1_values": param1_values,
        "param2_name": param2_name,
        "param2_values": param2_values,
        "llh_grid": llh_grid,
        "results": results,
    }


def plot_grid(data, output_file="grid_scan_2d.png"):
    """Plot 2D grid scan results."""
    param1_values = np.array(data["param1_values"])
    param2_values = np.array(data["param2_values"])
    llh_grid = np.array(data["llh_grid"])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid for plotting
    P1, P2 = np.meshgrid(param1_values, param2_values, indexing='ij')
    llh_spl = 123.05300548196891
    # Plot contour
    contour = ax.contourf(P1, P2, llh_grid-llh_spl, levels=20, cmap='viridis')
    ax.contour(P1, P2, llh_grid-llh_spl, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    
    # Find and mark best fit
    best_idx = np.unravel_index(np.nanargmin(llh_grid), llh_grid.shape)
    best_p1 = param1_values[best_idx[0]]
    best_p2 = param2_values[best_idx[1]]
    best_llh = llh_grid[best_idx]
    
    ax.plot(best_p1, best_p2, 'r*', markersize=20, label=f'Best fit: -LLH={best_llh:.2f}')
    
    ax.set_xlabel(data["param1_name"], fontsize=12)
    ax.set_ylabel(data["param2_name"], fontsize=12)
    ax.set_title("2D Grid Scan: -LLH", fontsize=14)
    ax.legend()
    
    # Use log scale if values span orders of magnitude
    if param1_values.max() / param1_values.min() > 10:
        ax.set_xscale('log')
    if param2_values.max() / param2_values.min() > 10:
        ax.set_yscale('log')
    
    plt.colorbar(contour, ax=ax, label='-LLH')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")
    
    print(f"\nBest fit:")
    print(f"  {data['param1_name']} = {best_p1:.4g}")
    print(f"  {data['param2_name']} = {best_p2:.4g}")
    print(f"  -LLH = {best_llh:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot grid scan results")
    parser.add_argument("--output_dir", type=str, default="grid_scan_results",
                       help="Directory containing result files")
    parser.add_argument("--plot", type=str, default="grid_scan_2d.png",
                       help="Output plot filename")
    
    args = parser.parse_args()
    
    print(f"Aggregating results from {args.output_dir}...")
    data = aggregate_results(args.output_dir)
    
    if data is None:
        return
    
    # Save aggregated summary
    summary_file = Path(args.output_dir) / "summary.json"
    summary = {
        "param1_name": data["param1_name"],
        "param1_values": data["param1_values"],
        "param2_name": data["param2_name"],
        "param2_values": data["param2_values"],
        "llh_grid": data["llh_grid"].tolist(),
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {summary_file}")
    
    # Plot
    plot_file = Path(args.output_dir) / args.plot
    plot_grid(data, str(plot_file))


if __name__ == "__main__":
    main()

