#!/usr/bin/env python3
"""
Find missing job indices and recover results from output files.

This script works for both 2D grid scans (default) and 1D parameter scans (with --scan_1d).
It:
1. Identifies missing job indices by comparing expected vs. found results
2. Reads the corresponding .out files to extract fit results
3. Parses the HESE_fit.py output to extract LLH and parameters
4. Adds the recovered results to the results.jsonl file

Usage for 2D scan (default):
    python recover_missing_jobs.py \
        --results_file grid_scan_results/results.jsonl \
        --n1 25 --n2 25 \
        --output_pattern "grid_scan_*_%d.out" \
        --output_dir .
    
Usage for 1D scan:
    python recover_missing_jobs.py \
        --scan_1d \
        --results_file 1d_parameter_scan/BPL_Ebreak/results_Ebreak_4e4_to_2e5.jsonl \
        --npoints 15 \
        --output_pattern "grid_scan_*_%d.out" \
        --output_dir .
"""

import json
import argparse
import os
import re
import glob
import numpy as np


def parse_hese_fit_output(output_text):
    """
    Parse the stdout from HESE_fit.py to extract:
      - overall best-fit LLH and parameters
      - both epsilon_DOM interval fits (if present)
      - total fit time
    """
    output_lines = output_text.split("\n")

    best_llh = None
    best_params = {}
    fit_results = []
    fit_time = None

    # Parse the overall best fit
    in_best_fit_params = False
    for line in output_lines:
        if "Best Fit -LLH:" in line:
            try:
                llh_str = line.split("Best Fit -LLH:")[1].strip()
                # Handle "inf (all fits failed)" or just "inf"
                if "inf" in llh_str.lower():
                    best_llh = float('inf')
                else:
                    best_llh = float(llh_str)
            except ValueError:
                best_llh = None
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
    for line in output_lines:
        if "Fit " in line and "epsilon_DOM" in line:
            # Save previous fit if exists
            if current_fit and current_fit["llh"] is not None:
                fit_results.append(current_fit)
            # Start of a new interval fit
            in_interval_fit = True
            current_fit = {
                "llh": None,
                "params": {},
                "interval": None,
                "epsilon_range": None,
            }
            # Extract interval info
            if "low" in line.lower():
                current_fit["interval"] = "low"
            elif "high" in line.lower():
                current_fit["interval"] = "high"
            # Extract epsilon range
            if "[0.8, 0.99]" in line:
                current_fit["epsilon_range"] = "[0.8, 0.99]"
            elif "[0.99, 1.25]" in line:
                current_fit["epsilon_range"] = "[0.99, 1.25]"
        elif in_interval_fit and "  -LLH:" in line:
            try:
                llh_str = line.split("  -LLH:")[1].strip()
                if "inf" in llh_str.lower():
                    current_fit["llh"] = float('inf')
                else:
                    current_fit["llh"] = float(llh_str)
            except (ValueError, IndexError):
                pass
        elif in_interval_fit and "  Parameters:" in line:
            continue
        elif (
            in_interval_fit
            and "\t" in line
            and ": " in line
            and "Interval:" not in line
            and "Epsilon_DOM range:" not in line
        ):
            # Parse parameter lines (but skip the interval/range lines)
            parts = line.strip().split(":")
            if len(parts) == 2:
                param_name = parts[0].strip()
                try:
                    param_value = float(parts[1].strip())
                    current_fit["params"][param_name] = param_value
                except ValueError:
                    pass
        elif in_interval_fit and ("  Interval:" in line or "  Epsilon_DOM range:" in line):
            # Skip these lines, already extracted from header
            continue
        elif in_interval_fit and ("===" in line or "LLH list:" in line):
            # End of interval fits section, save current fit
            if current_fit and current_fit["llh"] is not None:
                fit_results.append(current_fit)
            in_interval_fit = False
            current_fit = None

    # Save last fit if we're still in one
    if current_fit and current_fit["llh"] is not None:
        fit_results.append(current_fit)

    # If we didn't find interval fits (e.g., epsilon_DOM is fixed), create a single result
    if len(fit_results) == 0 and best_llh is not None:
        fit_results = [
            {
                "llh": best_llh,
                "params": best_params,
                "interval": "single",
                "epsilon_range": "N/A",
            }
        ]

    # Parse fit time
    for line in output_lines:
        if "Fit took" in line and "seconds" in line:
            try:
                # Extract number from "Fit took X seconds" or "Fit took X.XX seconds"
                match = re.search(r"Fit took\s+([\d.]+)\s+seconds", line)
                if match:
                    fit_time = float(match.group(1))
            except (ValueError, AttributeError):
                pass
            break

    return best_llh, best_params, fit_results, fit_time


def find_missing_jobs_2d(results_file, n1, n2):
    """Find missing job indices in a 2D grid scan."""
    
    if not os.path.exists(results_file):
        print(f"Warning: {results_file} not found. All jobs are considered missing.")
        total_points = n1 * n2
        missing_indices = list(range(total_points))
        missing_grid_indices = [(idx, *np.unravel_index(idx, (n1, n2))) for idx in missing_indices]
        return missing_indices, missing_grid_indices
    
    # Read all results
    completed_job_indices = set()
    with open(results_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    result = json.loads(line)
                    if "job_index" in result:
                        completed_job_indices.add(result["job_index"])
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    
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


def find_missing_jobs_1d(results_file, npoints):
    """Find missing job indices in a 1D parameter scan."""
    
    if not os.path.exists(results_file):
        print(f"Warning: {results_file} not found. All jobs are considered missing.")
        return list(range(npoints))
    
    # Read all results
    completed_job_indices = set()
    with open(results_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    result = json.loads(line)
                    if "job_index" in result:
                        completed_job_indices.add(result["job_index"])
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    
    # Calculate all expected job indices
    all_job_indices = set(range(npoints))
    
    # Find missing job indices
    missing_job_indices = sorted(all_job_indices - completed_job_indices)
    
    return missing_job_indices


def find_output_file(job_index, output_pattern, output_dir, output_files_dict=None):
    """
    Find the output file for a given job index.
    
    output_pattern should be a glob pattern with %d for the job index,
    e.g., "grid_scan_*_%d.out"
    """
    if output_files_dict and job_index in output_files_dict:
        return output_files_dict[job_index]
    
    # Replace %d with the job index
    pattern = output_pattern.replace("%d", str(job_index))
    
    # Search in output_dir
    search_path = os.path.join(output_dir, pattern)
    matches = glob.glob(search_path)
    
    if matches:
        return matches[0]
    
    return None


def extract_param_value_from_output(output_text, param_name):
    """Extract the parameter value being scanned from the output."""
    # Look for lines like "Running parameter scan point [X/Y]: param_name=value"
    pattern = rf"Running parameter scan point.*{re.escape(param_name)}=([\d.eE+-]+)"
    match = re.search(pattern, output_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Also try to find it in the command line arguments
    pattern = rf"--{re.escape(param_name)}\s+([\d.eE+-]+)"
    match = re.search(pattern, output_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    return None


def extract_grid_params_from_output(output_text, param1_name, param2_name):
    """Extract grid parameter values from 2D scan output."""
    param1_value = None
    param2_value = None
    
    # Look for lines like "Running grid point [i,j]: param1=value1, param2=value2"
    pattern = rf"Running grid point.*{re.escape(param1_name)}=([\d.eE+-]+).*{re.escape(param2_name)}=([\d.eE+-]+)"
    match = re.search(pattern, output_text)
    if match:
        try:
            param1_value = float(match.group(1))
            param2_value = float(match.group(2))
            return param1_value, param2_value
        except ValueError:
            pass
    
    # Try to find them separately
    if param1_name:
        param1_value = extract_param_value_from_output(output_text, param1_name)
    if param2_name:
        param2_value = extract_param_value_from_output(output_text, param2_name)
    
    return param1_value, param2_value


def main():
    parser = argparse.ArgumentParser(
        description="Recover missing job results from output files (2D grid scan by default, 1D with --scan_1d)"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the results.jsonl file"
    )
    
    # 2D scan parameters (default)
    parser.add_argument(
        "--n1",
        type=int,
        default=None,
        help="Number of points for parameter 1 (required for 2D scan)"
    )
    parser.add_argument(
        "--n2",
        type=int,
        default=None,
        help="Number of points for parameter 2 (required for 2D scan)"
    )
    
    # 1D scan parameters
    parser.add_argument(
        "--scan_1d",
        action="store_true",
        help="Use 1D parameter scan mode (default is 2D grid scan)"
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=None,
        help="Total number of points in 1D scan (required if --scan_1d)"
    )
    
    # Output file options
    parser.add_argument(
        "--output_pattern",
        type=str,
        default="grid_scan_*_%d.out",
        help="Glob pattern for output files with %%d for job index (default: grid_scan_*_%%d.out)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to search for output files (default: current directory)"
    )
    parser.add_argument(
        "--output_files",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of output files (one per job index, in order)"
    )
    
    # Parameter names (for extracting values from output)
    parser.add_argument(
        "--param_name",
        type=str,
        default=None,
        help="Name of the parameter being scanned in 1D mode (for extracting param_value)"
    )
    parser.add_argument(
        "--param1_name",
        type=str,
        default=None,
        help="Name of parameter 1 in 2D mode (for extracting param1_value)"
    )
    parser.add_argument(
        "--param2_name",
        type=str,
        default=None,
        help="Name of parameter 2 in 2D mode (for extracting param2_value)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't write to results file, just show what would be recovered"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Write recovered results to a new file instead of appending to --results_file"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.scan_1d:
        if args.npoints is None:
            parser.error("--npoints is required for 1D scan (use --scan_1d)")
        if args.n1 is not None or args.n2 is not None:
            print("Warning: --n1 and --n2 are ignored in 1D scan mode")
    else:
        if args.n1 is None or args.n2 is None:
            parser.error("--n1 and --n2 are required for 2D scan (or use --scan_1d --npoints)")
        if args.npoints is not None:
            print("Warning: --npoints is ignored in 2D scan mode")
    
    # Find missing jobs
    if args.scan_1d:
        missing_indices = find_missing_jobs_1d(args.results_file, args.npoints)
        missing_grid_indices = None
        total_points = args.npoints
        print(f"1D scan mode: {args.npoints} total points")
    else:
        missing_indices, missing_grid_indices = find_missing_jobs_2d(args.results_file, args.n1, args.n2)
        total_points = args.n1 * args.n2
        print(f"2D grid scan mode: {args.n1} x {args.n2} = {total_points} total points")
    
    if not missing_indices:
        print("No missing jobs found! All jobs are present in the results file.")
        return
    
    print(f"Found {len(missing_indices)} missing jobs out of {total_points} total:")
    print(f"  Missing job indices: {missing_indices}")
    if missing_grid_indices:
        print("\nMissing grid points:")
        for job_idx, i, j in missing_grid_indices:
            print(f"  Job {job_idx:3d}: grid[{i:2d},{j:2d}]")
    print()
    
    # Build output files dictionary if explicit list provided
    output_files_dict = None
    if args.output_files:
        if len(args.output_files) != total_points:
            print(f"Warning: Number of output files ({len(args.output_files)}) doesn't match total points ({total_points})")
        output_files_dict = {i: f for i, f in enumerate(args.output_files)}
    
    # Try to recover results from output files
    recovered_results = []
    failed_recoveries = []
    
    for job_index in missing_indices:
        output_file = find_output_file(job_index, args.output_pattern, args.output_dir, output_files_dict)
        
        if not output_file:
            print(f"  Job {job_index}: No output file found")
            failed_recoveries.append(job_index)
            continue
        
        if not os.path.exists(output_file):
            print(f"  Job {job_index}: Output file not found: {output_file}")
            failed_recoveries.append(job_index)
            continue
        
        # Read output file
        try:
            with open(output_file, "r") as f:
                output_text = f.read()
        except Exception as e:
            print(f"  Job {job_index}: Error reading {output_file}: {e}")
            failed_recoveries.append(job_index)
            continue
        
        # Parse output
        best_llh, best_params, fit_results, fit_time = parse_hese_fit_output(output_text)
        
        if best_llh is None:
            print(f"  Job {job_index}: Could not parse LLH from {output_file}")
            failed_recoveries.append(job_index)
            continue
        
        # Create result dictionary
        llh_json = float(best_llh) if (best_llh is not None and not (isinstance(best_llh, float) and best_llh == float('inf'))) else "inf"
        
        result = {
            "job_index": job_index,
            "llh": llh_json,
            "params": best_params if best_params else {},
            "fit_results": fit_results if fit_results else [],
            "fit_time": float(fit_time) if fit_time else None,
        }
        
        # Add grid indices for 2D scans
        if not args.scan_1d and missing_grid_indices:
            for job_idx, i, j in missing_grid_indices:
                if job_idx == job_index:
                    result["grid_index"] = [int(i), int(j)]
                    break
        
        # Extract parameter values from output
        if args.scan_1d and args.param_name:
            param_value = extract_param_value_from_output(output_text, args.param_name)
            result["param_name"] = args.param_name
            if param_value is not None:
                result["param_value"] = param_value
        elif not args.scan_1d:
            if args.param1_name or args.param2_name:
                param1_value, param2_value = extract_grid_params_from_output(
                    output_text, args.param1_name or "", args.param2_name or ""
                )
                if args.param1_name and param1_value is not None:
                    result["param1_name"] = args.param1_name
                    result["param1_value"] = param1_value
                if args.param2_name and param2_value is not None:
                    result["param2_name"] = args.param2_name
                    result["param2_value"] = param2_value
        
        recovered_results.append((job_index, result, output_file))
        print(f"  Job {job_index}: Successfully recovered from {output_file}")
        print(f"    -LLH: {llh_json}")
        if not args.scan_1d and "grid_index" in result:
            print(f"    Grid: [{result['grid_index'][0]}, {result['grid_index'][1]}]")
    
    print()
    print(f"Recovered {len(recovered_results)} results")
    if failed_recoveries:
        print(f"Failed to recover {len(failed_recoveries)} results: {failed_recoveries}")
    
    # Write recovered results to file
    if recovered_results and not args.dry_run:
        if args.output_file:
            # Write to new file
            output_file_abs = os.path.abspath(args.output_file)
            print(f"\nWriting {len(recovered_results)} recovered results to new file: {output_file_abs}")
            
            with open(output_file_abs, "w") as f:
                for job_index, result, output_file in recovered_results:
                    f.write(json.dumps(result) + "\n")
                    print(f"  Wrote job {job_index} to new file")
            
            print(f"\nSuccessfully wrote {len(recovered_results)} results to {args.output_file}")
        else:
            # Append to existing file
            results_file_abs = os.path.abspath(args.results_file)
            print(f"\nAppending {len(recovered_results)} recovered results to {results_file_abs}")
            
            with open(results_file_abs, "a") as f:
                for job_index, result, output_file in recovered_results:
                    f.write(json.dumps(result) + "\n")
                    print(f"  Added job {job_index} to results file")
            
            print(f"\nSuccessfully added {len(recovered_results)} results to {args.results_file}")
    elif args.dry_run:
        print("\n[DRY RUN] Would have added the following results:")
        for job_index, result, output_file in recovered_results:
            print(f"  Job {job_index}: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
