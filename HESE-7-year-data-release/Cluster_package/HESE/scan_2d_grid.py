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
    
    # Get absolute path to HESE_fit.py (same directory as this script)
    base_path = os.path.dirname(os.path.abspath(__file__))
    hese_fit_path = os.path.join(base_path, "HESE_fit.py")
    
    # Use -u flag for unbuffered output so we see output immediately
    cmd = [
        python_executable, "-u", hese_fit_path,
        "--model", model,
        f"--{param1_name}", str(param1_value),
        f"--fix_{param1_name}",
        f"--{param2_name}", str(param2_value),
        f"--fix_{param2_name}",
    ]
    
    # Fix cutoff_energy if model is not "cutoff" (to avoid fitting unused parameter)
    #if model != "cutoff":
    #    cmd.append("--fix_cutoff_energy")
    
    # TESTA OM ALLA VÄRDEN BLIR BÄTTRE MED PROMPT_NORM = 0.0
    #cmd.append("--prompt_norm")
    #cmd.append("0.0")
    #cmd.append("--fix_prompt_norm")
    
    # Handle fixed parameters: if a parameter value is provided and fix flag is True, add both
    # This allows fixing additional nuisance parameters beyond the two being scanned
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
            # For boolean arguments that need values (like majorana, normal, nuSI, HESE12), always pass the value
            if key in ("majorana", "normal", "nuSI", "HESE12"):
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


def run_single_point(param1_name, param1_value, param2_name, param2_value,
                     model="nusiprop", python_executable=None, **kwargs):
    """Run HESE_fit.py for a single (param1, param2) point."""
    cmd = _build_cmd(param1_name, param1_value, param2_name, param2_value,
                     model=model, python_executable=python_executable, **kwargs)
    
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
        print(f"Error running fit for {param1_name}={param1_value:.4g}, "
              f"{param2_name}={param2_value:.4g}:")
        print(e.stderr)
        return np.inf, None, None


def run_single_point_with_retry(param1_name, param1_value, param2_name, param2_value,
                                model="nusiprop", python_executable=None,
                                max_retries=3, retry_with_relaxed_tol=True, **kwargs):
    """
    Run HESE_fit.py for a single point with retry mechanism if fit fails.
    
    If the initial fit returns inf, this function will retry with progressively
    more relaxed optimization tolerances:
    - pgtol: multiplied by 10, 100, 1000 (larger = more relaxed)
    - factr: multiplied by 10, 100, 1000 (larger = more relaxed)
    
    Note: Parameter perturbation is not currently implemented - only tolerance
    relaxation is used.
    
    Returns:
    --------
    llh, params, fit_time, retry_info
    where retry_info is a dict with:
        - retry_attempted: bool (whether retries were attempted)
        - retry_succeeded: bool (whether a retry succeeded)
        - retry_attempt: int (which retry attempt succeeded, 0 if first attempt, None if all failed)
        - pgtol_used: float (final pgtol value used)
        - factr_used: float (final factr value used)
    
    Parameters:
    -----------
    max_retries : int
        Maximum number of retry attempts (default: 3)
    retry_with_relaxed_tol : bool
        If True, retry with relaxed tolerances (default: True)
    **kwargs : dict
        Additional arguments passed to HESE_fit.py
    """
    # Apply default tolerances based on model
    default_pgtol = 1e-10 if model == "nusiprop" else 1e-15
    default_factr = 1e7 if model == "nusiprop" else 1e4
    
    # First attempt with default parameters
    llh, params, fit_time = run_single_point(
        param1_name, param1_value, param2_name, param2_value,
        model=model, python_executable=python_executable, **kwargs
    )
    
    retry_info = {
        "retry_attempted": False,
        "retry_succeeded": False,
        "retry_attempt": 0,  # 0 means first attempt succeeded
        "pgtol_used": default_pgtol,
        "factr_used": default_factr,
    }
    
    # If successful, return immediately
    if llh is not None and not np.isinf(llh):
        return llh, params, fit_time, retry_info
    
    # Fit failed, try retries with adjusted parameters
    print(f"Initial fit failed (returned inf). Attempting retries with adjusted parameters...")
    retry_info["retry_attempted"] = True
    
    # Define retry strategies: (pgtol_multiplier, factr_multiplier, description)
    # Note: Both pgtol and factr are made LARGER to relax tolerances:
    #   - pgtol: larger = more relaxed (accepts larger gradients)
    #   - factr: larger = more relaxed (easier convergence criterion)
    retry_strategies = []
    if retry_with_relaxed_tol:
        # Strategy 1: Relax tolerances moderately (10x more relaxed)
        retry_strategies.append((10.0, 10.0, "relaxed tolerances (10x)"))
        # Strategy 2: Relax tolerances more aggressively (100x more relaxed)
        if max_retries >= 2:
            retry_strategies.append((100.0, 100.0, "very relaxed tolerances (100x)"))
        # Strategy 3: Very relaxed tolerances (1000x more relaxed)
        if max_retries >= 3:
            retry_strategies.append((1000.0, 1000.0, "extremely relaxed tolerances (1000x)"))
    
    for retry_idx, (pgtol_mult, factr_mult, description) in enumerate(retry_strategies[:max_retries], 1):
        print(f"\nRetry attempt {retry_idx}/{len(retry_strategies[:max_retries])}: "
              f"trying with {description}...")
        
        # Calculate new tolerances (multiplying makes both LARGER = more relaxed)
        new_pgtol = default_pgtol * pgtol_mult  # Larger pgtol = more relaxed
        new_factr = default_factr * factr_mult   # Larger factr = more relaxed
        
        # Create new kwargs with adjusted tolerances
        retry_kwargs = kwargs.copy()
        retry_kwargs['pgtol'] = new_pgtol
        retry_kwargs['factr'] = new_factr
        
        # Run with adjusted parameters
        retry_llh, retry_params, retry_fit_time = run_single_point(
            param1_name, param1_value, param2_name, param2_value,
            model=model, python_executable=python_executable, **retry_kwargs
        )
        
        # If this retry succeeded, return the result
        if retry_llh is not None and not np.isinf(retry_llh):
            print(f"Retry {retry_idx} succeeded! -LLH: {retry_llh:.6f}")
            retry_info["retry_succeeded"] = True
            retry_info["retry_attempt"] = retry_idx
            retry_info["pgtol_used"] = new_pgtol
            retry_info["factr_used"] = new_factr
            return retry_llh, retry_params, retry_fit_time, retry_info
        else:
            print(f"Retry {retry_idx} also failed (returned inf)")
    
    # All retries failed
    print(f"\nAll {len(retry_strategies[:max_retries])} retry attempts failed. Returning inf.")
    retry_info["retry_attempt"] = None
    return np.inf, None, None, retry_info


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
    parser.add_argument("--model", type=str, default="spl",
                       choices=["spl", "cutoff", "bpl", "lp"],
                       help="Model to use")
    """parser.add_argument("--model_type", type=str, default="nusiprop",
                       choices=["nusiprop", "regular"],
                       help="Model type to use for nuSIprop parameters")"""
    parser.add_argument("--output_dir", type=str, default="grid_scan_results",
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
    
    # Additional parameters to fix (nuSIprop parameters)
    parser.add_argument("--mntot", type=float, default=None,
                       help="Fix mntot parameter to this value (requires --fix_mntot)")
    parser.add_argument("--fix_mntot", action="store_true",
                       help="Fix mntot parameter in fit")
    parser.add_argument("--Mphi", type=float, default=None,
                       help="Fix Mphi parameter to this value (requires --fix_Mphi)")
    parser.add_argument("--fix_Mphi", action="store_true",
                       help="Fix Mphi parameter in fit")
    parser.add_argument("--g", type=float, default=None,
                       help="Fix g parameter to this value (requires --fix_g)")
    parser.add_argument("--fix_g", action="store_true",
                       help="Fix g parameter in fit")
    
    # Additional nuisance parameters
    parser.add_argument("--prompt_norm", type=float, default=None,
                       help="Fix prompt_norm parameter to this value (requires --fix_prompt_norm)")
    parser.add_argument("--fix_prompt_norm", action="store_true",
                       help="Fix prompt_norm parameter in fit")
    parser.add_argument("--astro_norm", type=float, default=None,
                       help="Fix astro_norm parameter to this value (requires --fix_astro_norm)")
    parser.add_argument("--fix_astro_norm", action="store_true",
                       help="Fix astro_norm parameter in fit")
    parser.add_argument("--astro_gamma", type=float, default=None,
                       help="Fix astro_gamma parameter to this value (requires --fix_astro_gamma)")
    parser.add_argument("--fix_astro_gamma", action="store_true",
                       help="Fix astro_gamma parameter in fit")
    parser.add_argument("--nuSI", type=str, default=None,
                       help="Enable nuSIprop secret interactions (True/False, default: True). If False, fixes Mphi, g, mntot and sets g=1e-30")
    parser.add_argument("--HESE12", type=str, default=None,
                       help="Use HESE12 data instead of HESE data (True/False, default: False)")
    
    # Retry options
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum number of retry attempts when fit returns inf (default: 3)")
    parser.add_argument("--no_retry", action="store_true",
                       help="Disable retry mechanism (use original behavior)")
    
    # Optimization parameters
    parser.add_argument("--pgtol", type=float, default=None,
                       help="Gradient tolerance for L-BFGS-B optimizer")
    parser.add_argument("--factr", type=float, default=None,
                       help="Convergence factor for L-BFGS-B optimizer")
    parser.add_argument("--m", type=int, default=None,
                       help="Number of corrections used in L-BFGS-B (default: 30 for nusiprop, 20 otherwise)")
    parser.add_argument("--maxiter", type=int, default=None,
                       help="Maximum number of iterations for L-BFGS-B (default: 500)")
    parser.add_argument("--results_file", type=str, default="results.jsonl",
                       help="Name of the JSONL results file (default: results.jsonl)")
    
    args = parser.parse_args()
    
    # Make output directory path absolute to avoid issues with different working directories on cluster
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)
    
    # Save metadata once (optimization parameters)
    os.makedirs(args.output_dir, exist_ok=True)
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        # Only create metadata file if it doesn't exist (first job in cluster mode, or standalone mode)
        metadata = {
            "pgtol": args.pgtol,
            "factr": args.factr,
            "m": args.m,
            "maxiter": args.maxiter,
            "model": args.model,
            "param1_name": args.param1,
            "param2_name": args.param2,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
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
        
        # Add fixed parameter arguments
        if args.mntot is not None:
            fit_kwargs["mntot"] = args.mntot
        if args.fix_mntot:
            fit_kwargs["fix_mntot"] = True
        if args.Mphi is not None:
            fit_kwargs["Mphi"] = args.Mphi
        if args.fix_Mphi:
            fit_kwargs["fix_Mphi"] = True
        if args.g is not None:
            fit_kwargs["g"] = args.g
        if args.fix_g:
            fit_kwargs["fix_g"] = True
        if args.prompt_norm is not None:
            fit_kwargs["prompt_norm"] = args.prompt_norm
        if args.fix_prompt_norm:
            fit_kwargs["fix_prompt_norm"] = True
        if args.astro_norm is not None:
            fit_kwargs["astro_norm"] = args.astro_norm
        if args.fix_astro_norm:
            fit_kwargs["fix_astro_norm"] = True
        if args.astro_gamma is not None:
            fit_kwargs["astro_gamma"] = args.astro_gamma
        if args.fix_astro_gamma:
            fit_kwargs["fix_astro_gamma"] = True
        if args.nuSI is not None:
            fit_kwargs["nuSI"] = args.nuSI
        if args.HESE12 is not None:
            fit_kwargs["HESE12"] = args.HESE12
        
        # Pass cluster_mode to HESE_fit.py so it uses the correct nuSIprop path
        fit_kwargs["cluster_mode"] = True
        
        if args.no_retry:
            llh, params, fit_time = run_single_point(
                args.param1, p1_val, args.param2, p2_val,
                model=args.model, python_executable=args.python, **fit_kwargs
            )
            retry_info = None
        else:
            llh, params, fit_time, retry_info = run_single_point_with_retry(
                args.param1, p1_val, args.param2, p2_val,
                model=args.model, python_executable=args.python,
                max_retries=args.max_retries, **fit_kwargs
            )
        elapsed = time.time() - start_time
        
        # Save result to single shared file (JSONL format - one JSON object per line)
        # This allows parallel jobs to safely append without file locking
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            results_file = os.path.join(args.output_dir, args.results_file)
            results_file_abs = os.path.abspath(results_file)
        
            # Handle np.inf for JSON (JSON doesn't support infinity, use string)
            llh_json = float(llh) if (llh is not None and not np.isinf(llh)) else "inf"
        
            result = {
                "job_index": args.job_index,
                "grid_index": [int(i), int(j)],
                "param1_name": args.param1,
                "param1_value": float(p1_val),
                "param2_name": args.param2,
                "param2_value": float(p2_val),
                "llh": llh_json,
                "params": params if params else {},
                "fit_time": float(fit_time) if fit_time else float(elapsed),
                "model": args.model,
            }
            # Only include retry_info if retries were enabled
            if retry_info is not None:
                result["retry_info"] = retry_info
        
            # Append to JSONL file (one JSON object per line, safe for parallel writes)
            #model_type_json = args.model_type
            with open(results_file_abs, "a") as f:
                f.write(json.dumps(result) + "\n")
        
            print(f"Result appended to {results_file_abs}")
        except Exception as e:
            print(f"ERROR: Failed to save result to {args.output_dir}: {e}")
            print(f"Current working directory: {os.getcwd()}")
            raise
        if llh is not None and not np.isinf(llh):
            print(f"  -LLH: {llh:.6f}")
        else:
            print("  Fit failed (set -LLH to inf)")
        
    else:
        # Standalone mode: run all points sequentially
        print(f"Running 2D grid scan: {args.n1} x {args.n2} = {total_points} points")
        print(f"  {args.param1}: {args.p1min:.4g} to {args.p1max:.4g} ({args.n1} points, "
              f"log={args.log1})")
        print(f"  {args.param2}: {args.p2min:.4g} to {args.p2max:.4g} ({args.n2} points, "
              f"log={args.log2})")
        print()
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize results file (clear it if it exists for a fresh run)
        results_file = os.path.join(args.output_dir, args.results_file)
        if os.path.exists(results_file):
            os.remove(results_file)  # Start fresh for standalone mode
        
        # Initialize results array
        llh_grid = np.full((args.n1, args.n2), np.inf)
        fit_times = []
        
        total_start = time.time()
        
        for idx, (p1_val, p2_val) in enumerate(grid_points):
            i, j = np.unravel_index(idx, (args.n1, args.n2))
            
            print(f"[{idx+1}/{total_points}] Grid point [{i},{j}]: "
                  f"{args.param1}={p1_val:.4g}, {args.param2}={p2_val:.4g}")
            
            start_time = time.time()
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
            
            # Add fixed parameter arguments
            if args.mntot is not None:
                fit_kwargs["mntot"] = args.mntot
            if args.fix_mntot:
                fit_kwargs["fix_mntot"] = True
            if args.Mphi is not None:
                fit_kwargs["Mphi"] = args.Mphi
            if args.fix_Mphi:
                fit_kwargs["fix_Mphi"] = True
            if args.g is not None:
                fit_kwargs["g"] = args.g
            if args.fix_g:
                fit_kwargs["fix_g"] = True
            if args.prompt_norm is not None:
                fit_kwargs["prompt_norm"] = args.prompt_norm
            if args.fix_prompt_norm:
                fit_kwargs["fix_prompt_norm"] = True
            if args.astro_norm is not None:
                fit_kwargs["astro_norm"] = args.astro_norm
            if args.fix_astro_norm:
                fit_kwargs["fix_astro_norm"] = True
            if args.astro_gamma is not None:
                fit_kwargs["astro_gamma"] = args.astro_gamma
            if args.fix_astro_gamma:
                fit_kwargs["fix_astro_gamma"] = True
            if args.nuSI is not None:
                fit_kwargs["nuSI"] = args.nuSI
            if args.HESE12 is not None:
                fit_kwargs["HESE12"] = args.HESE12
            
            # Pass cluster_mode to HESE_fit.py so it uses the correct nuSIprop path
            fit_kwargs["cluster_mode"] = True
            
            if args.no_retry:
                llh, params, fit_time = run_single_point(
                    args.param1, p1_val, args.param2, p2_val,
                    model=args.model, python_executable=args.python, **fit_kwargs
                )
                retry_info = None
            else:
                llh, params, fit_time, retry_info = run_single_point_with_retry(
                    args.param1, p1_val, args.param2, p2_val,
                    model=args.model, python_executable=args.python,
                    max_retries=args.max_retries, **fit_kwargs
                )
            elapsed = time.time() - start_time
            
            if llh is not None and not np.isinf(llh):
                llh_grid[i, j] = llh
                print(f"  -LLH: {llh:.6f}")
            else:
                llh_grid[i, j] = np.inf
                print("  Fit failed (set -LLH to inf)")
            
            fit_times.append(fit_time if fit_time else elapsed)
            
            # Save result to single shared file (JSONL format - one JSON object per line)
            results_file = os.path.join(args.output_dir, args.results_file)
            # Handle np.inf for JSON (JSON doesn't support infinity, use string)
            llh_json = float(llh) if (llh is not None and not np.isinf(llh)) else "inf"
            
            result = {
                "job_index": idx,
                "grid_index": [int(i), int(j)],
                "param1_name": args.param1,
                "param1_value": float(p1_val),
                "param2_name": args.param2,
                "param2_value": float(p2_val),
                "llh": llh_json,
                "params": params if params else {},
                "fit_time": float(fit_time) if fit_time else float(elapsed),
                "model": args.model,
            }
            # Only include retry_info if retries were enabled
            if retry_info is not None:
                result["retry_info"] = retry_info
            # Append to JSONL file (one JSON object per line)
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")
            
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
        # Convert np.inf to "inf" string for JSON compatibility
        llh_grid_json = [[("inf" if np.isinf(val) else val) for val in row] for row in llh_grid.tolist()]
        
        summary = {
            "param1_name": args.param1,
            "param1_values": param1_values.tolist(),
            "param2_name": args.param2,
            "param2_values": param2_values.tolist(),
            "llh_grid": llh_grid_json,
            "fit_times": fit_times,
            "model": args.model,
            "total_points": total_points,
            "total_time": total_elapsed,
            "pgtol": args.pgtol,
            "factr": args.factr,
            "m": args.m,
            "maxiter": args.maxiter,
        }
        
        summary_file = os.path.join(args.output_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()


