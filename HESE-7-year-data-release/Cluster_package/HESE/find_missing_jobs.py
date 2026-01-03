#!/usr/bin/env python3
"""
Find missing grid points in results.jsonl and generate commands to re-run them.

Usage:
    python find_missing_jobs.py --results_dir . --n1 25 --n2 25
"""

import json
import argparse
import os
import numpy as np


def find_missing_jobs(results_file, n1, n2):
    """Find missing grid points and return their job indices."""
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        return []
    
    # Read all results
    results = []
    with open(results_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    
    # Get all job indices that were completed
    completed_job_indices = set(r["job_index"] for r in results)
    
    # Calculate all expected job indices
    total_points = n1 * n2
    all_job_indices = set(range(total_points))
    
    # Find missing job indices
    missing_job_indices = sorted(all_job_indices - completed_job_indices)
    
    # Also get grid indices for missing points
    missing_grid_indices = []
    for job_idx in missing_job_indices:
        i, j = np.unravel_index(job_idx, (n1, n2))
        missing_grid_indices.append((job_idx, i, j))
    
    return missing_job_indices, missing_grid_indices


def main():
    parser = argparse.ArgumentParser(
        description="Find missing grid points in results.jsonl"
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
    parser.add_argument(
        "--generate_commands",
        action="store_true",
        help="Generate SLURM job submission commands for missing jobs"
    )
    
    args = parser.parse_args()
    
    results_file = os.path.join(args.results_dir, "results.jsonl")
    missing_job_indices, missing_grid_indices = find_missing_jobs(
        results_file, args.n1, args.n2
    )
    
    if not missing_job_indices:
        print("No missing jobs found!")
        return
    
    print(f"Found {len(missing_job_indices)} missing jobs:")
    print(f"  Missing job indices: {missing_job_indices}")
    print("\nMissing grid points:")
    for job_idx, i, j in missing_grid_indices:
        print(f"  Job {job_idx:3d}: grid[{i:2d},{j:2d}]")
    
    if args.generate_commands:
        print("\n" + "="*70)
        print("To re-run these jobs, you can:")
        print("="*70)
        print("\n1. Submit individual jobs using SLURM array:")
        print(f"   sbatch --array={','.join(map(str, missing_job_indices))} submit_grid_scan_tetralith.sh")
        print("\n2. Or run them individually:")
        for job_idx, i, j in missing_grid_indices:
            print(f"   python scan_2d_grid.py --param1 Mphi --p1min 0.03 --p1max 100.0 --n1 {args.n1} --log1 \\")
            print(f"       --param2 g --p2min 0.0001 --p2max 1.0 --n2 {args.n2} --log2 \\")
            print(f"       --model nusiprop --output_dir grid_scan_results --cluster_mode --no_retry --job_index {job_idx}")


if __name__ == "__main__":
    main()

