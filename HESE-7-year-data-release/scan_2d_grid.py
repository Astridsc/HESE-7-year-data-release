"""
2D grid scan over two parameters (e.g., Mphi and g for nuSIprop) with cluster support.

This script can be used in two modes:
1. Standalone: runs all grid points sequentially
2. Cluster mode: runs a single grid point (for job arrays)

Example usage (standalone):
    python scan_2d_grid.py --param1 Mphi --p1min 0.1 --p1max 100 --n1 10 --log1 \
                           --param2 g --p2min 0.001 --p2max 1.0 --n2 10 --log2 \
                           --model nusiprop --output_dir grid_scan_results

Example usage (cluster, single point):
    python scan_2d_grid.py --param1 Mphi --p1min 0.1 --p1max 100 --n1 10 --log1 \
                           --param2 g --p2min 0.001 --p2max 1.0 --n2 10 --log2 \
                           --model nusiprop --output_dir grid_scan_results \
                           --cluster_mode --job_index 42

For SLURM job arrays:
    #SBATCH --array=0-99
    python scan_2d_grid.py ... --cluster_mode --job_index $SLURM_ARRAY_TASK_ID
"""

import numpy as np
import subprocess
import sys
import argparse
import json
import os
import time
import re
from pathlib import Path


def _build_cmd(param1_name, param1_value, param2_name, param2_value, model="nusiprop",
               python_executable=None, **kwargs):
    """Build command to run HESE_fit.py with two fixed parameters."""
    if python_executable is None:
        python_executable = sys.executable
    
    cmd = [
        python_executable, "HESE_fit.py",
        "--model", model,
        f"--{param1_name}", str(param1_value),
        f"--fix_{param1_name}",
        f"--{param2_name}", str(param2_value),
        f"--fix_{param2_name}",
    ]
    
    # Add any additional arguments
    for key, value in kwargs.items():
        if value is not None:
            flag = f"--{key}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.append(flag)
                cmd.append(str(value))
    
    return cmd


def _parse_hese_fit_output(stdout):
    """Parse HESE_fit.py output to extract LLH and parameters."""
    output_lines = stdout.split("\n")
    
    best_llh = None
    best_params = {}
    fit_time = None
    
    # Parse best fit
    in_best_fit_params = False
    for line in output_lines:
        if "Best Fit -LLH:" in line:
            try:
                best_llh = float(line.split("Best Fit -LLH:")[1].strip())
            except ValueError:
                best_llh = None
            in_best_fit_params = True
        elif in_best_fit_params and "Best Fit Paramters:" in line:
            continue
        elif in_best_fit_params and "\t" in line and ":" in line:
            parts = line.strip().split(":")
            if len(parts) == 2:
                param_name = parts[0].strip()
                try:
                    param_value = float(parts[1].strip())
                    best_params[param_name] = param_value
                except ValueError:
                    pass
        elif in_best_fit_params and line.strip() == "":
            in_best_fit_params = False
    
    # Parse fit time
    for line in output_lines:
        if "Fit took" in line and "seconds" in line:
            try:
                match = re.search(r"Fit took\s+([\d.]+)\s+seconds", line)
                if match:
                    fit_time = float(match.group(1))
            except (ValueError, AttributeError):
                pass
            break
    
    return best_llh, best_params, fit_time


def run_single_point(param1_name, param1_value, param2_name, param2_value,
                     model="nusiprop", python_executable=None, **kwargs):
    """Run HESE_fit.py for a single (param1, param2) point."""
    cmd = _build_cmd(param1_name, param1_value, param2_name, param2_value,
                     model=model, python_executable=python_executable, **kwargs)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return _parse_hese_fit_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running fit for {param1_name}={param1_value:.4g}, "
              f"{param2_name}={param2_value:.4g}:")
        print(e.stderr)
        return None, None, None


def generate_grid(p1min, p1max, n1, log1, p2min, p2max, n2, log2):
    """Generate 2D grid of parameter values."""
    if log1:
        param1_values = np.logspace(np.log10(p1min), np.log10(p1max), n1)
    else:
        param1_values = np.linspace(p1min, p1max, n1)
    
    if log2:
        param2_values = np.logspace(np.log10(p2min), np.log10(p2max), n2)
    else:
        param2_values = np.linspace(p2min, p2max, n2)
    
    # Create meshgrid
    P1, P2 = np.meshgrid(param1_values, param2_values, indexing='ij')
    
    # Flatten to list of (p1, p2) pairs
    grid_points = list(zip(P1.flatten(), P2.flatten()))
    
    return param1_values, param2_values, grid_points


def main():
    parser = argparse.ArgumentParser(
        description="2D grid scan over two parameters with cluster support"
    )
    
    # Parameter 1
    parser.add_argument("--param1", type=str, required=True,
                       help="Name of first parameter (e.g., 'Mphi')")
    parser.add_argument("--p1min", type=float, required=True,
                       help="Minimum value for param1")
    parser.add_argument("--p1max", type=float, required=True,
                       help="Maximum value for param1")
    parser.add_argument("--n1", type=int, required=True,
                       help="Number of points for param1")
    parser.add_argument("--log1", action="store_true",
                       help="Use log spacing for param1")
    
    # Parameter 2
    parser.add_argument("--param2", type=str, required=True,
                       help="Name of second parameter (e.g., 'g')")
    parser.add_argument("--p2min", type=float, required=True,
                       help="Minimum value for param2")
    parser.add_argument("--p2max", type=float, required=True,
                       help="Maximum value for param2")
    parser.add_argument("--n2", type=int, required=True,
                       help="Number of points for param2")
    parser.add_argument("--log2", action="store_true",
                       help="Use log spacing for param2")
    
    # Model and output
    parser.add_argument("--model", type=str, default="nusiprop",
                       choices=["spl", "cutoff", "nusiprop"],
                       help="Model to use")
    parser.add_argument("--output_dir", type=str, default="grid_scan_results",
                       help="Output directory for results")
    
    # Cluster mode
    parser.add_argument("--cluster_mode", action="store_true",
                       help="Run in cluster mode (single point)")
    parser.add_argument("--job_index", type=int, default=None,
                       help="Job index for cluster mode (0-indexed)")
    
    parser.add_argument("--python", type=str, default=None,
                       help="Python executable (default: sys.executable)")
    
    args = parser.parse_args()
    
    # Generate grid
    param1_values, param2_values, grid_points = generate_grid(
        args.p1min, args.p1max, args.n1, args.log1,
        args.p2min, args.p2max, args.n2, args.log2
    )
    
    total_points = len(grid_points)
    
    if args.cluster_mode:
        # Cluster mode: run single point
        if args.job_index is None:
            print("Error: --job_index required in cluster mode")
            sys.exit(1)
        
        if args.job_index >= total_points:
            print(f"Job index {args.job_index} >= total points {total_points}, skipping")
            sys.exit(0)
        
        p1_val, p2_val = grid_points[args.job_index]
        i, j = np.unravel_index(args.job_index, (args.n1, args.n2))
        
        print(f"Running grid point [{i},{j}] ({args.param1}={p1_val:.4g}, "
              f"{args.param2}={p2_val:.4g})")
        
        start_time = time.time()
        llh, params, fit_time = run_single_point(
            args.param1, p1_val, args.param2, p2_val,
            model=args.model, python_executable=args.python
        )
        elapsed = time.time() - start_time
        
        # Save result
        os.makedirs(args.output_dir, exist_ok=True)
        result_file = os.path.join(args.output_dir, f"result_{args.job_index:05d}.json")
        
        result = {
            "job_index": args.job_index,
            "grid_index": [int(i), int(j)],
            "param1_name": args.param1,
            "param1_value": float(p1_val),
            "param2_name": args.param2,
            "param2_value": float(p2_val),
            "llh": float(llh) if llh is not None else None,
            "params": params if params else {},
            "fit_time": float(fit_time) if fit_time else float(elapsed),
            "model": args.model,
        }
        
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Result saved to {result_file}")
        if llh is not None:
            print(f"  -LLH: {llh:.6f}")
        else:
            print("  Fit failed")
        
    else:
        # Standalone mode: run all points sequentially
        print(f"Running 2D grid scan: {args.n1} x {args.n2} = {total_points} points")
        print(f"  {args.param1}: {args.p1min:.4g} to {args.p1max:.4g} ({args.n1} points, "
              f"log={args.log1})")
        print(f"  {args.param2}: {args.p2min:.4g} to {args.p2max:.4g} ({args.n2} points, "
              f"log={args.log2})")
        print()
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize results array
        llh_grid = np.full((args.n1, args.n2), np.nan)
        fit_times = []
        
        total_start = time.time()
        
        for idx, (p1_val, p2_val) in enumerate(grid_points):
            i, j = np.unravel_index(idx, (args.n1, args.n2))
            
            print(f"[{idx+1}/{total_points}] Grid point [{i},{j}]: "
                  f"{args.param1}={p1_val:.4g}, {args.param2}={p2_val:.4g}")
            
            start_time = time.time()
            llh, params, fit_time = run_single_point(
                args.param1, p1_val, args.param2, p2_val,
                model=args.model, python_executable=args.python
            )
            elapsed = time.time() - start_time
            
            if llh is not None:
                llh_grid[i, j] = llh
                print(f"  -LLH: {llh:.6f}")
            else:
                print("  Fit failed")
            
            fit_times.append(fit_time if fit_time else elapsed)
            
            # Save individual result
            result_file = os.path.join(args.output_dir, f"result_{idx:05d}.json")
            result = {
                "job_index": idx,
                "grid_index": [int(i), int(j)],
                "param1_name": args.param1,
                "param1_value": float(p1_val),
                "param2_name": args.param2,
                "param2_value": float(p2_val),
                "llh": float(llh) if llh is not None else None,
                "params": params if params else {},
                "fit_time": float(fit_time) if fit_time else float(elapsed),
                "model": args.model,
            }
            with open(result_file, "w") as w:
                json.dump(result, w, indent=2)
            
            # Progress estimate
            if idx > 0:
                avg_time = np.mean(fit_times)
                remaining = avg_time * (total_points - idx - 1)
                print(f"  Estimated remaining: {remaining/60:.1f} min ({remaining/3600:.1f} h)")
            
            print()
        
        total_elapsed = time.time() - total_start
        print(f"Total time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} h)")
        print(f"Average time per point: {np.mean(fit_times)/60:.1f} min")
        
        # Save summary
        summary = {
            "param1_name": args.param1,
            "param1_values": param1_values.tolist(),
            "param2_name": args.param2,
            "param2_values": param2_values.tolist(),
            "llh_grid": llh_grid.tolist(),
            "fit_times": fit_times,
            "model": args.model,
            "total_points": total_points,
            "total_time": total_elapsed,
        }
        
        summary_file = os.path.join(args.output_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

