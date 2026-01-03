#!/usr/bin/env python3
"""
Separate results.jsonl entries by job array ID based on matching LLH values from .out files.

Usage:
    python separate_results_by_job_array.py --results_file results.jsonl --out_dir . --out_prefix results
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


def extract_llh_from_out_file(out_file):
    """Extract final LLH value and grid point from .out file."""
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
        
        # Extract grid point from first line
        # Format: "Running grid point [3,20] (Mphi=0.0827, g=0.2154)"
        grid_point = None
        param1_value = None
        param2_value = None
        for line in lines[:10]:  # Check first 10 lines
            if 'Running grid point' in line:
                # Extract grid coordinates
                grid_match = re.search(r'\[(\d+),(\d+)\]', line)
                if grid_match:
                    grid_point = [int(grid_match.group(1)), int(grid_match.group(2))]
                
                # Extract parameter values
                param1_match = re.search(r'(\w+)=([\d.]+)', line)
                param2_match = re.search(r'(\w+)=([\d.]+)', line)
                if param1_match and param2_match:
                    # Get both matches
                    matches = re.findall(r'(\w+)=([\d.]+)', line)
                    if len(matches) >= 2:
                        param1_value = float(matches[0][1])
                        param2_value = float(matches[1][1])
                break
        
        # Extract final LLH from "Best Fit -LLH:" line
        final_llh = None
        for line in reversed(lines):  # Search from end
            if 'Best Fit -LLH:' in line:
                # Extract number after "Best Fit -LLH:"
                llh_match = re.search(r'Best Fit -LLH:\s+([\d.]+)', line)
                if llh_match:
                    final_llh = float(llh_match.group(1))
                    break
        
        if final_llh is None:
            # Try alternative format: "  -LLH: 165.336956" (near end)
            for line in reversed(lines[-20:]):  # Check last 20 lines
                if '-LLH:' in line and 'Best Fit' not in line:
                    llh_match = re.search(r'-LLH:\s+([\d.]+)', line)
                    if llh_match:
                        final_llh = float(llh_match.group(1))
                        break
        
        if final_llh is None:
            print(f"Warning: Could not extract LLH from {out_file}")
            return None
        
        return {
            'job_array_id': job_array_id,
            'task_index': task_index,
            'llh': final_llh,
            'grid_point': grid_point,
            'param1_value': param1_value,
            'param2_value': param2_value,
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


def match_result_to_out(result, out_data, llh_tolerance=1e-6):
    """Match a result entry to an .out file entry."""
    # Match by task index and LLH value
    if result.get('job_index') == out_data['task_index']:
        result_llh = result.get('llh')
        if result_llh == "inf":
            return False
        if isinstance(result_llh, (int, float)):
            if abs(result_llh - out_data['llh']) < llh_tolerance:
                return True
        
        # Also try matching by parameter values if available
        if out_data['param1_value'] is not None and out_data['param2_value'] is not None:
            result_param1 = result.get('param1_value')
            result_param2 = result.get('param2_value')
            if (result_param1 is not None and result_param2 is not None and
                abs(result_param1 - out_data['param1_value']) < 1e-6 and
                abs(result_param2 - out_data['param2_value']) < 1e-6):
                return True
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Separate results.jsonl by job array ID using .out files"
    )
    parser.add_argument("--results_file", type=str, required=True,
                       help="Path to results.jsonl file")
    parser.add_argument("--out_dir", type=str, default=".",
                       help="Directory containing .out files (default: current directory)")
    parser.add_argument("--out_prefix", type=str, default="results",
                       help="Prefix for output files (default: results)")
    parser.add_argument("--llh_tolerance", type=float, default=1e-6,
                       help="Tolerance for LLH matching (default: 1e-6)")
    
    args = parser.parse_args()
    
    # Find all .out files
    out_dir = Path(args.out_dir)
    out_files = list(out_dir.glob("grid_scan_*.out"))
    
    if not out_files:
        print(f"Error: No grid_scan_*.out files found in {args.out_dir}")
        return
    
    print(f"Found {len(out_files)} .out files")
    
    # Extract data from .out files
    out_data_list = []
    for out_file in out_files:
        data = extract_llh_from_out_file(out_file)
        if data:
            out_data_list.append(data)
    
    print(f"Successfully extracted data from {len(out_data_list)} .out files")
    
    # Group by job array ID
    job_arrays = defaultdict(list)
    for data in out_data_list:
        job_arrays[data['job_array_id']].append(data)
    
    print(f"Found {len(job_arrays)} unique job arrays:")
    for job_id, data_list in sorted(job_arrays.items()):
        print(f"  Job array {job_id}: {len(data_list)} tasks")
    
    # Load results.jsonl
    print(f"\nLoading results from {args.results_file}...")
    results = load_results_jsonl(args.results_file)
    print(f"Loaded {len(results)} results")
    
    # Match results to job arrays
    matched_results = defaultdict(list)
    unmatched_results = []
    
    for result in results:
        matched = False
        for job_id, out_data_list in job_arrays.items():
            for out_data in out_data_list:
                if match_result_to_out(result, out_data, args.llh_tolerance):
                    matched_results[job_id].append(result)
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            unmatched_results.append(result)
    
    # Print summary
    print(f"\nMatching summary:")
    for job_id in sorted(matched_results.keys()):
        print(f"  Job array {job_id}: {len(matched_results[job_id])} results")
    if unmatched_results:
        print(f"  Unmatched: {len(unmatched_results)} results")
    
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

