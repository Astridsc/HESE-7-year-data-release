
import numpy as np
import subprocess
import sys
import argparse
import json
import os
import time
import re
from pathlib import Path


def _build_cmd_fit_all(model="nusiprop", python_executable=None, fixed_param_name=None, fixed_param_value=None, **kwargs):
    """Build command to run HESE_fit.py with one parameter fixed (or none) and all others free."""
    if python_executable is None:
        python_executable = sys.executable
    
    # Get absolute path to HESE_fit.py (same directory as this script)
    base_path = os.path.dirname(os.path.abspath(__file__))
    hese_fit_path = os.path.join(base_path, "HESE_fit.py")
    
    # Use -u flag for unbuffered output so we see output immediately
    cmd = [
        python_executable, "-u", hese_fit_path,
        "--model", model,
    ]
    
    # If a parameter should be fixed, add it with fix flag
    if fixed_param_name is not None and fixed_param_value is not None:
        cmd.extend([f"--{fixed_param_name}", str(fixed_param_value), f"--fix_{fixed_param_name}"])
    
    # Handle fixed parameters: if a parameter value is provided and fix flag is True, add both
    fixed_params = {
        "mntot": "fix_mntot",
        "Mphi": "fix_Mphi",
        "g": "fix_g",
        "prompt_norm": "fix_prompt_norm",
        "astro_norm": "fix_astro_norm",
        "astro_gamma": "fix_astro_gamma",
    }
    
    # Add any additional arguments
    for key, value in kwargs.items():
        if value is not None:
            # Skip fix flags here, handle them with their corresponding parameter values
            if key.startswith("fix_"):
                continue
            
            flag = f"--{key}"
            # For boolean arguments that need values (like majorana, normal), always pass the value
            if key in ("majorana", "normal"):
                cmd.append(flag)
                cmd.append(str(value))
            elif isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.append(flag)
                cmd.append(str(value))
                
                # If this parameter has a fix flag and it's set, add the fix flag
                if key in fixed_params:
                    fix_key = fixed_params[key]
                    if kwargs.get(fix_key, False):
                        cmd.append(f"--{fix_key}")
    
    # Add cluster_mode and output_dir if provided
    if kwargs.get("cluster_mode", False):
        cmd.append("--cluster_mode")
    if kwargs.get("output_dir") is not None:
        cmd.extend(["--output_dir", str(kwargs["output_dir"])])
    
    return cmd


def _parse_hese_fit_output(stdout):
    """Parse HESE_fit.py output to extract LLH and parameters."""
    output_lines = stdout.split("\n")
    
    best_llh = np.inf  # Default to inf if parsing fails
    best_params = {}
    fit_time = None
    
    # Parse best fit
    in_best_fit_params = False
    for line in output_lines:
        if "Best Fit -LLH:" in line:
            try:
                llh_str = line.split("Best Fit -LLH:")[1].strip()
                # Handle "inf (all fits failed)" or just "inf"
                if "inf" in llh_str.lower():
                    best_llh = np.inf
                else:
                    best_llh = float(llh_str)
            except ValueError:
                best_llh = np.inf  # Set to inf if parsing fails
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


def run_single_fit(fixed_param_name=None, fixed_param_value=None,
                   model="nusiprop", python_executable=None, **kwargs):
    """Run HESE_fit.py with one parameter fixed (or none) and all others free."""
    cmd = _build_cmd_fit_all(model=model, python_executable=python_executable,
                             fixed_param_name=fixed_param_name, fixed_param_value=fixed_param_value, **kwargs)
    
    # Only print command in debug mode (set via environment variable)
    if os.environ.get('DEBUG_SCAN', '').lower() in ('1', 'true', 'yes'):
        print('cmd: ', cmd)
    try:
        # Pass environment variables to subprocess so it can find photospline
        env = os.environ.copy()
        # Set working directory to script directory so relative paths work
        base_path = os.path.dirname(os.path.abspath(__file__))
        #print('env: ', env)
        # Use Popen to read output in real-time and avoid hanging
        # Combine stderr into stdout to avoid deadlock
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            env=env,
            cwd=base_path,
            bufsize=1,  # Line buffered
        )
        
        # Read output line by line in real-time
        stdout_lines = []
        
        # Read combined stdout+stderr
        for line in iter(process.stdout.readline, ''):
            if line:
                stdout_lines.append(line)
                print(line.rstrip())  # Print for debugging
        
        # Wait for process to complete
        returncode = process.wait()
        
        stdout = ''.join(stdout_lines)
        stderr = ''  # Already included in stdout
        
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd, stdout, stderr)
        
        return _parse_hese_fit_output(stdout)
    except subprocess.CalledProcessError as e:
        param_desc = f"{fixed_param_name}={fixed_param_value:.4g}" if fixed_param_name else "all parameters free"
        print(f"Error running fit for {param_desc}:")
        print(e.stderr)
        return np.inf, None, None








def main():
    parser = argparse.ArgumentParser(
        description="Running multiple single jobs, with different specifications"
    )
    
    # Model and output
    parser.add_argument("--model", type=str, default="nusiprop",
                       choices=["spl", "cutoff", "nusiprop"],
                       help="Model to use")
    parser.add_argument("--output_dir", type=str, default="single_job_results",
                       help="Output directory for results")
    
    # Cluster mode
    parser.add_argument("--cluster_mode", action="store_true",
                       help="Run in cluster mode (single point)")
    parser.add_argument("--job_index", type=int, default=None,
                       help="Job index for cluster mode (0-indexed)")
    
    parser.add_argument("--python", type=str, default=None,
                       help="Python executable (default: sys.executable)")
    
    # nuSIprop options
    parser.add_argument("--majorana", type=str, default=None,
                       help="Use Majorana (True) or Dirac (False) neutrinos for nuSIprop (default: True)")
    parser.add_argument("--normal", type=str, default=None,
                       help="Use normal (True) or inverted (False) mass ordering for nuSIprop (default: True)")
    
    # Scan parameters: arrays of values to scan over
    parser.add_argument("--mntot_values", type=str, default=None,
                       help="Comma-separated list of mntot values to scan (e.g., '0.06,0.08,0.10,0.12,0.14')")
    parser.add_argument("--astro_gamma_values", type=str, default=None,
                       help="Comma-separated list of astro_gamma values to scan (e.g., '2.0,2.5,3.0,3.5')")
    
    # Additional parameters to fix (nuSIprop parameters) - for backward compatibility
    parser.add_argument("--mntot", type=float, default=None,
                       help="Fix mntot parameter to this value (requires --fix_mntot)")
    parser.add_argument("--fix_mntot", action="store_true",
                       help="Fix mntot parameter in fit")
    parser.add_argument("--astro_gamma", type=float, default=None,
                       help="Fix astro_gamma parameter to this value (requires --fix_astro_gamma)")
    parser.add_argument("--fix_astro_gamma", action="store_true",
                       help="Fix astro_gamma parameter in fit")
    
    # Optimization parameters
    parser.add_argument("--pgtol", type=float, default=None,
                       help="Gradient tolerance for L-BFGS-B optimizer")
    parser.add_argument("--factr", type=float, default=None,
                       help="Convergence factor for L-BFGS-B optimizer")
    parser.add_argument("--m", type=int, default=None,
                       help="Number of corrections used in L-BFGS-B (default: 30 for nusiprop, 20 otherwise)")
    parser.add_argument("--maxiter", type=int, default=None,
                       help="Maximum number of iterations for L-BFGS-B (default: 500)")
    
    args = parser.parse_args()
    
    # Make output directory path absolute to avoid issues with different working directories on cluster
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)
    
    # Parse value arrays
    mntot_values = []
    if args.mntot_values:
        mntot_values = [float(x.strip()) for x in args.mntot_values.split(',')]
    
    astro_gamma_values = []
    if args.astro_gamma_values:
        astro_gamma_values = [float(x.strip()) for x in args.astro_gamma_values.split(',')]
    
    if not mntot_values and not astro_gamma_values:
        print("Error: Must specify at least one of --mntot_values or --astro_gamma_values")
        sys.exit(1)
    
    # Save metadata once (optimization parameters)
    os.makedirs(args.output_dir, exist_ok=True)
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        metadata = {
            "pgtol": args.pgtol,
            "factr": args.factr,
            "m": args.m,
            "maxiter": args.maxiter,
            "model": args.model,
            "mntot_values": mntot_values,
            "astro_gamma_values": astro_gamma_values,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    # Prepare kwargs for HESE_fit.py arguments
    fit_kwargs = {}
    if args.majorana is not None:
        fit_kwargs["majorana"] = args.majorana
    if args.normal is not None:
        fit_kwargs["normal"] = args.normal
    
    # Add optimization parameters
    if args.pgtol is not None:
        fit_kwargs["pgtol"] = args.pgtol
    if args.factr is not None:
        fit_kwargs["factr"] = args.factr
    if args.m is not None:
        fit_kwargs["m"] = args.m
    if args.maxiter is not None:
        fit_kwargs["maxiter"] = args.maxiter
    
    # Add cluster_mode and output_dir to kwargs (will be passed to HESE_fit.py)
    fit_kwargs["cluster_mode"] = True
    fit_kwargs["output_dir"] = args.output_dir
    
    results_file = os.path.join(args.output_dir, "results.jsonl")
    results_file_abs = os.path.abspath(results_file)
    
    if args.cluster_mode:
        # Cluster mode: run single point based on job_index
        if args.job_index is None:
            print("Error: --job_index required in cluster mode")
            sys.exit(1)
        
        # Calculate total points
        total_points = len(mntot_values) + len(astro_gamma_values)
        
        if args.job_index >= total_points:
            print(f"Job index {args.job_index} >= total points {total_points}, skipping")
            sys.exit(0)
        
        # Determine which scan and which value
        if args.job_index < len(mntot_values):
            # mntot scan
            scan_type = "mntot"
            fixed_param_name = "mntot"
            fixed_param_value = mntot_values[args.job_index]
            scan_index = args.job_index
        else:
            # astro_gamma scan
            scan_type = "astro_gamma"
            fixed_param_name = "astro_gamma"
            fixed_param_value = astro_gamma_values[args.job_index - len(mntot_values)]
            scan_index = args.job_index - len(mntot_values)
        
        print(f"Running {scan_type} scan, index {scan_index}: {fixed_param_name}={fixed_param_value:.6f}")
        
        start_time = time.time()
        llh, params, fit_time = run_single_fit(
            fixed_param_name=fixed_param_name,
            fixed_param_value=fixed_param_value,
            model=args.model,
            python_executable=args.python,
            **fit_kwargs
        )
        elapsed = time.time() - start_time
        
        # Result is already saved by HESE_fit.py in cluster mode, but we can add scan info
        if llh is not None and not np.isinf(llh):
            print(f"  -LLH: {llh:.6f}")
        else:
            print("  Fit failed (set -LLH to inf)")
        
    else:
        # Standalone mode: run all scans sequentially
        total_points = len(mntot_values) + len(astro_gamma_values)
        print(f"Running 1D scans: {len(mntot_values)} mntot values + {len(astro_gamma_values)} astro_gamma values = {total_points} total points")
        if mntot_values:
            print(f"  mntot values: {mntot_values}")
        if astro_gamma_values:
            print(f"  astro_gamma values: {astro_gamma_values}")
        print()
        
        # Initialize results file (clear it if it exists for a fresh run)
        if os.path.exists(results_file_abs):
            os.remove(results_file_abs)  # Start fresh for standalone mode
        
        fit_times = []
        total_start = time.time()
        point_idx = 0
        
        # Scan over mntot values
        for idx, mntot_val in enumerate(mntot_values):
            point_idx += 1
            print(f"[{point_idx}/{total_points}] mntot scan, index {idx}: mntot={mntot_val:.6f}")
            
            start_time = time.time()
            llh, params, fit_time = run_single_fit(
                fixed_param_name="mntot",
                fixed_param_value=mntot_val,
                model=args.model,
                python_executable=args.python,
                **fit_kwargs
            )
            elapsed = time.time() - start_time
            
            if llh is not None and not np.isinf(llh):
                print(f"  -LLH: {llh:.6f}")
            else:
                print("  Fit failed (set -LLH to inf)")
            
            fit_times.append(fit_time if fit_time else elapsed)
            
            # Progress estimate
            if point_idx > 1:
                avg_time = np.mean(fit_times)
                remaining = avg_time * (total_points - point_idx)
                print(f"  Estimated remaining: {remaining/60:.1f} min ({remaining/3600:.1f} h)")
            
            print()
        
        # Scan over astro_gamma values
        for idx, astro_gamma_val in enumerate(astro_gamma_values):
            point_idx += 1
            print(f"[{point_idx}/{total_points}] astro_gamma scan, index {idx}: astro_gamma={astro_gamma_val:.6f}")
            
            start_time = time.time()
            llh, params, fit_time = run_single_fit(
                fixed_param_name="astro_gamma",
                fixed_param_value=astro_gamma_val,
                model=args.model,
                python_executable=args.python,
                **fit_kwargs
            )
            elapsed = time.time() - start_time
            
            if llh is not None and not np.isinf(llh):
                print(f"  -LLH: {llh:.6f}")
            else:
                print("  Fit failed (set -LLH to inf)")
            
            fit_times.append(fit_time if fit_time else elapsed)
            
            # Progress estimate
            if point_idx > 1:
                avg_time = np.mean(fit_times)
                remaining = avg_time * (total_points - point_idx)
                print(f"  Estimated remaining: {remaining/60:.1f} min ({remaining/3600:.1f} h)")
            
            print()
        
        total_elapsed = time.time() - total_start
        print(f"Total time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} h)")
        print(f"Average time per point: {np.mean(fit_times)/60:.1f} min")
        print(f"\nResults saved to {results_file_abs}")


if __name__ == "__main__":
    main()


