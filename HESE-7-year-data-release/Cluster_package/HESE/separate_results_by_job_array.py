#!/usr/bin/env python3
"""
Separate results.jsonl entries by job array ID based on LLH matching from .out files.

This script matches results from results.jsonl to job arrays by:
1. Convert grid_index [i,j] to task_index = i * n2 + j
2. Find all .out files with that task_index across all job arrays
3. Extract LLH from each of those .out files
4. Match the result's LLH to one of those LLH values
5. Assign the result to the matching job array

Each .out file corresponds to one job array task, and contains:
- Job array ID in filename: grid_scan_<JOB_ID>_<TASK_ID>.out
- Grid point in output: "Running grid point [i,j]"
- LLH value in output: "Best Fit -LLH: <value>"

Each result in results.jsonl has:
- grid_index: [i, j] coordinates
- llh: the likelihood value

Usage:
    python separate_results_by_job_array.py --results_file results.jsonl --out_dir . --out_prefix results --n2 25
"""

import argparse
import json
import os
import re
import numpy as np
from collections import defaultdict
from pathlib import Path


def extract_data_from_out_file(out_file):
    """Extract grid point, job array ID, task index, and LLH from .out file."""
    try:
        with open(out_file, 'r') as f:
            lines = f.readlines()
        
        # Extract job array ID and task index from filename
        # Format: grid_scan_<JOB_ID>_<TASK_ID>.out
        filename = os.path.basename(out_file)
        match = re.match(r'grid_scan_(\d+)_(\d+)\.out', filename)
        if not match:
            return None
        
        job_array_id = int(match.group(1))
        task_index = int(match.group(2))
        
        # Extract grid point from first lines
        # Format: "Running grid point [3,20] (Mphi=0.0827, g=0.2154)"
        grid_point = None
        for line in lines[:10]:  # Check first 10 lines
            if 'Running grid point' in line:
                # Extract grid coordinates
                grid_match = re.search(r'\[(\d+),(\d+)\]', line)
                if grid_match:
                    grid_point = tuple([int(grid_match.group(1)), int(grid_match.group(2))])
                    break
        
        if grid_point is None:
            print(f"Warning: Could not extract grid point from {out_file}")
            return None
        
        # Extract LLH value from output
        # Format: "Best Fit -LLH: 165.33695592344876" or "Best Fit -LLH: inf (all fits failed)"
        llh = None
        for line in lines:
            if 'Best Fit -LLH:' in line:
                llh_str = line.split('Best Fit -LLH:')[1].strip()
                # Handle "inf (all fits failed)" or just "inf"
                if 'inf' in llh_str.lower():
                    llh = float('inf')
                else:
                    try:
                        llh = float(llh_str)
                    except ValueError:
                        llh = None
                break
        
        return {
            'job_array_id': job_array_id,
            'task_index': task_index,
            'grid_point': grid_point,
            'llh': llh,
        }
    except Exception as e:
        print(f"Error reading {out_file}: {e}")
        return None


def load_results_jsonl(results_file):
    """Load all results from JSONL file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line: {line[:100]}...")
    return results


def main():
    print("Separating results.jsonl by job array ID using grid_index from .out files")
    parser = argparse.ArgumentParser(
        description="Separate results.jsonl by job array ID using grid_index from .out files"
    )
    parser.add_argument("--results_file", type=str, required=True,
                       help="Path to results.jsonl file")
    parser.add_argument("--out_dir", type=str, default=".",
                       help="Directory containing .out files (default: current directory)")
    parser.add_argument("--out_prefix", type=str, default="results",
                       help="Prefix for output files (default: results)")
    parser.add_argument("--n2", type=int, required=True,
                       help="Number of points in second dimension (e.g., 25 for 25x25 grid)")
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    # Find all .out files
    out_dir = Path(args.out_dir)
    out_files = list(out_dir.glob("grid_scan_*.out"))
    
    if not out_files:
        print(f"Error: No grid_scan_*.out files found in {args.out_dir}")
        return
    
    print(f"Found {len(out_files)} .out files")
    
    # Build mapping: task_index -> list of (job_array_id, llh) tuples
    # For each task_index, we'll have multiple entries (one per job array)
    task_to_job_arrays = defaultdict(list)  # task_index -> [(job_array_id, llh), ...]
    job_arrays_seen = set()
    
    print("Building mapping from task_index to job arrays and LLH values...")
    for out_file in out_files:
        data = extract_data_from_out_file(out_file)
        if data:
            job_array_id = data['job_array_id']
            task_index = data['task_index']
            llh = data.get('llh')
            
            task_to_job_arrays[task_index].append((job_array_id, llh))
            job_arrays_seen.add(job_array_id)
    
    print(f"Found {len(job_arrays_seen)} unique job arrays: {sorted(job_arrays_seen)}")
    print(f"Built mapping for {len(task_to_job_arrays)} task indices")
    
    # Verify that each task_index appears in all job arrays
    if len(job_arrays_seen) > 0:
        expected_job_arrays = len(job_arrays_seen)
        tasks_with_all_arrays = sum(1 for job_list in task_to_job_arrays.values() 
                                    if len(job_list) == expected_job_arrays)
        print(f"Task indices appearing in all {expected_job_arrays} job arrays: {tasks_with_all_arrays}/{len(task_to_job_arrays)}")
    
    # Load results.jsonl
    print(f"\nLoading results from {args.results_file}...")
    results = load_results_jsonl(args.results_file)
    print(f"Loaded {len(results)} results")
    
    # Match results to job arrays using LLH matching
    # For each result:
    # 1. Convert grid_index [i,j] to task_index = i * n2 + j
    # 2. Find all .out files with that task_index (across all job arrays)
    # 3. Extract LLH from each of those .out files
    # 4. Match the result's LLH to one of those LLH values
    # 5. Assign the result to the matching job array
    # IMPORTANT: Each (job_array_id, grid_index) combination should only have ONE result
    matched_results = defaultdict(list)
    unmatched_results = []
    llh_mismatches = []  # Track cases where no LLH match is found
    
    # Track which (job_array_id, grid_index) combinations have already been assigned
    # This prevents duplicate assignments
    assigned_combinations = set()  # (job_array_id, tuple(grid_index))
    
    for result in results:
        grid_index = result.get('grid_index')
        result_llh = result.get('llh')
        
        if grid_index is None or result_llh is None:
            unmatched_results.append(result)
            continue
        
        # Convert grid_index [i,j] to task_index = i * n2 + j
        i, j = grid_index[0], grid_index[1]
        task_index = i * args.n2 + j
        grid_index_tuple = tuple(grid_index)
        
        # Find all job arrays that have this task_index
        if task_index not in task_to_job_arrays:
            unmatched_results.append(result)
            continue
        
        # Get list of (job_array_id, llh) for this task_index
        job_array_candidates = task_to_job_arrays[task_index]
        
        # Convert result_llh to float (handle "inf" string)
        if isinstance(result_llh, str) and result_llh == "inf":
            result_llh_float = float('inf')
        else:
            try:
                result_llh_float = float(result_llh)
            except (ValueError, TypeError):
                result_llh_float = None
        
        if result_llh_float is None:
            unmatched_results.append(result)
            continue
        
        # Find the job array whose LLH matches the result's LLH
        # But skip if we've already assigned a result for this (job_array_id, grid_index) combination
        matched = False
        for job_array_id, out_llh in job_array_candidates:
            # Check if we've already assigned a result for this combination
            combination_key = (job_array_id, grid_index_tuple)
            if combination_key in assigned_combinations:
                continue  # Skip this job array, already has a result for this grid point
            
            if out_llh is None:
                continue
            
            # Compare LLH values (with tolerance for floating point, handle inf cases)
            if np.isinf(result_llh_float) and np.isinf(out_llh):
                # Both are inf, that's a match
                matched_results[job_array_id].append(result)
                assigned_combinations.add(combination_key)
                matched = True
                break
            elif np.isinf(result_llh_float) or np.isinf(out_llh):
                # One is inf, one isn't - no match
                continue
            elif abs(result_llh_float - out_llh) < 1e-6:
                # Both are finite and match (within tolerance)
                matched_results[job_array_id].append(result)
                assigned_combinations.add(combination_key)
                matched = True
                break
        
        if not matched:
            # No matching LLH found (or all candidates already assigned)
            llh_mismatches.append({
                'grid_index': grid_index,
                'task_index': task_index,
                'result_llh': result_llh,
                'candidate_llhs': [(job_id, llh) for job_id, llh in job_array_candidates],
            })
            unmatched_results.append(result)
    
    # Print summary
    print(f"\nMatching summary:")
    for job_id in sorted(matched_results.keys()):
        print(f"  Job array {job_id}: {len(matched_results[job_id])} results")
    if unmatched_results:
        print(f"  Unmatched: {len(unmatched_results)} results")
        print(f"  (These may be from job arrays not found in .out files)")
    if llh_mismatches:
        print(f"  Warning: {len(llh_mismatches)} results could not be matched by LLH")
        if len(llh_mismatches) <= 10:
            for mismatch in llh_mismatches[:10]:
                print(f"    Grid {mismatch['grid_index']}, task {mismatch['task_index']}: "
                      f"result LLH={mismatch['result_llh']}")
                print(f"      Candidate LLHs: {mismatch['candidate_llhs']}")
    
    # Write separated results
    output_dir = Path(args.out_dir)
    for job_id, job_results in matched_results.items():
        output_file = output_dir / f"{args.out_prefix}_job_{job_id}.jsonl"
        with open(output_file, 'w') as f:
            for result in job_results:
                f.write(json.dumps(result) + "\n")
        print(f"Wrote {len(job_results)} results to {output_file}")
    
    if unmatched_results:
        output_file = output_dir / f"{args.out_prefix}_unmatched.jsonl"
        with open(output_file, 'w') as f:
            for result in unmatched_results:
                f.write(json.dumps(result) + "\n")
        print(f"Wrote {len(unmatched_results)} unmatched results to {output_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()