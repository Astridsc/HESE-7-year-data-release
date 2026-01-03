#!/usr/bin/env python3
"""
Combine results from cluster grid scan into a single summary.json file.

This script reads results.jsonl (created by scan_2d_grid.py in cluster mode)
and creates a summary.json file with the full llh_grid.
"""

import json
import numpy as np
import argparse
import os
from pathlib import Path


def combine_results(results_file, output_dir, n1, n2):
    """Read JSONL results and create summary.json with llh_grid."""
    
    results_file = os.path.join(output_dir, "results.jsonl")
    summary_file = os.path.join(output_dir, "summary.json")
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        return
    
    # Read all results from JSONL file
    results = []
    with open(results_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    
    if not results:
        print(f"Error: No results found in {results_file}")
        return
    
    print(f"Found {len(results)} results")
    
    # Get grid parameters from first result
    first_result = results[0]
    param1_name = first_result["param1_name"]
    param2_name = first_result["param2_name"]
    model = first_result["model"]
    
    # Initialize grid arrays
    llh_grid = np.full((n1, n2), np.inf)
    fit_times = np.full((n1, n2), np.nan)
    retry_info_grid = {}  # Store retry info by grid index
    
    # Extract parameter values (assuming they match the grid)
    param1_values = []
    param2_values = []
    
    # Fill in the grid
    for result in results:
        job_idx = result["job_index"]
        i, j = result["grid_index"]
        grid_key = (i, j)
        
        # Convert "inf" string back to np.inf for processing
        llh_val = result["llh"]
        if llh_val == "inf" or (isinstance(llh_val, float) and np.isinf(llh_val)):
            llh_grid[i, j] = np.inf
        else:
            llh_grid[i, j] = float(llh_val)
        
        fit_times[i, j] = result["fit_time"]
        
        # Store retry info if available
        if "retry_info" in result:
            retry_info_grid[grid_key] = result["retry_info"]
        
        # Collect unique parameter values
        p1_val = result["param1_value"]
        p2_val = result["param2_value"]
        if p1_val not in param1_values:
            param1_values.append(p1_val)
        if p2_val not in param2_values:
            param2_values.append(p2_val)
    
    # Sort parameter values
    param1_values = sorted(param1_values)
    param2_values = sorted(param2_values)
    
    # Convert np.inf to "inf" string for JSON compatibility
    llh_grid_json = [[("inf" if np.isinf(val) else val) for val in row] for row in llh_grid.tolist()]
    
    # Convert fit_times (nan to None for JSON)
    fit_times_json = [[(None if np.isnan(val) else val) for val in row] for row in fit_times.tolist()]
    
    # Create retry info summary
    retry_summary = {
        "retry_attempted": [[retry_info_grid.get((i, j), {}).get("retry_attempted", False) 
                             for j in range(n2)] for i in range(n1)],
        "retry_succeeded": [[retry_info_grid.get((i, j), {}).get("retry_succeeded", False) 
                            for j in range(n2)] for i in range(n1)],
        "retry_attempt": [[retry_info_grid.get((i, j), {}).get("retry_attempt", 0) 
                          for j in range(n2)] for i in range(n1)],
    }
    
    # Create summary
    summary = {
        "param1_name": param1_name,
        "param1_values": param1_values,
        "param2_name": param2_name,
        "param2_values": param2_values,
        "llh_grid": llh_grid_json,
        "fit_times": fit_times_json,
        "model": model,
        "total_points": n1 * n2,
        "completed_points": len(results),
        "retry_info": retry_summary,
    }
    
    # Save summary
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {summary_file}")
    print(f"  Grid size: {n1} x {n2} = {n1 * n2} points")
    print(f"  Completed: {len(results)} points")
    print(f"  Missing: {n1 * n2 - len(results)} points")
    
    # Count successful vs failed fits
    successful = np.sum(~np.isinf(llh_grid))
    failed = np.sum(np.isinf(llh_grid))
    print(f"  Successful fits: {successful}")
    print(f"  Failed fits: {failed}")
    
    # Count retry statistics
    if retry_info_grid:
        retry_attempted = sum(1 for info in retry_info_grid.values() if info.get("retry_attempted", False))
        retry_succeeded = sum(1 for info in retry_info_grid.values() if info.get("retry_succeeded", False))
        print(f"  Points that required retry: {retry_attempted}")
        print(f"  Points where retry succeeded: {retry_succeeded}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine grid scan results into summary.json"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing results.jsonl"
    )
    parser.add_argument(
        "--n1",
        type=int,
        required=True,
        help="Number of points for parameter 1"
    )
    parser.add_argument(
        "--n2",
        type=int,
        required=True,
        help="Number of points for parameter 2"
    )
    
    args = parser.parse_args()
    
    results_file = os.path.join(args.results_dir, "results.jsonl")
    combine_results(results_file, args.results_dir, args.n1, args.n2)


if __name__ == "__main__":
    main()

