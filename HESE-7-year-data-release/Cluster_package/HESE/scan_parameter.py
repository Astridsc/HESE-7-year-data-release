""" 
Examples of how to run the script:

Scan over cutoff_energy values:

python scan_parameter.py \
    --param cutoff_energy \
    --pmin 1e5 --pmax 1e7 --npoints 20 --log_space \
    --model cutoff \
    --save_data cutoff_scan.json
    
Scan over astro_gamma values for SPL:
python scan_parameter.py \
--param astro_gamma \
--pmin 2.0 --pmax 3.5 --npoints 30 \
--model spl

Scan over Mphi values for nuSIprop:
python scan_parameter.py \
--param Mphi \
--pmin 0.1 --pmax 100 --npoints 20 --log_space \
--model cutoff
    """





"""
Generic script to scan over a single parameter value and compute LLH for each value.

This is a generalization of scan_cutoff_energy.py. It:
  - Runs HESE_fit.py multiple times with different fixed values of a chosen parameter
  - Works for any model / parameter combination that HESE_fit.py understands

This script can be used in two modes:
1. Standalone: runs all parameter values sequentially
2. Cluster mode: runs a single parameter value (for job arrays)

Example usage (standalone):
    python scan_parameter.py --param cutoff_energy --pmin 1e5 --pmax 1e7 \
                              --npoints 20 --log_space --model cutoff

Example usage (cluster, single point):
    python scan_parameter.py --param cutoff_energy --pmin 1e5 --pmax 1e7 \
                              --npoints 20 --log_space --model cutoff \
                              --cluster_mode --job_index 5

For SLURM job arrays:
    #SBATCH --array=0-19
    python scan_parameter.py ... --cluster_mode --job_index $SLURM_ARRAY_TASK_ID
"""

import numpy as np
import subprocess
import sys
import argparse
import json
import re
import time as time_module
import os
from pathlib import Path


def _build_cmd(
    param_name,
    param_value,
    model="cutoff",
    fix_param=True,
    python_executable=None,
    **kwargs,
):
    """
    Build the command-line invocation of HESE_fit.py for a given parameter value.

    Parameters
    ----------
    param_name : str
        Name of the parameter in HESE_fit.py (e.g. 'cutoff_energy', 'astro_gamma', 'Mphi').
    param_value : float
        Value to set for the parameter.
    model : str
        HESE_fit model string (e.g. 'spl', 'cutoff', 'bpl', 'lp').
    fix_param : bool
        If True, add the corresponding '--fix_<param_name>' flag so that the parameter
        is held fixed at 'param_value' in the fit.
    python_executable : str or None
        Python executable to use; if None, uses sys.executable.
    **kwargs :
        Extra keyword arguments converted to CLI flags, passed through to HESE_fit.py.

    Returns
    -------
    cmd : list[str]
        Command list suitable for subprocess.run()
    """
    if python_executable is None:
        python_executable = sys.executable

    # Get absolute path to HESE_fit.py (same directory as this script)
    base_path = os.path.dirname(os.path.abspath(__file__))
    hese_fit_path = os.path.join(base_path, "HESE_fit.py")
    
    # Use -u flag for unbuffered output so we see output immediately
    cmd = [python_executable, "-u", hese_fit_path, "--model", model]
    
    # Format parameter value - convert numpy types to Python native types first
    # to avoid issues with repr() on numpy types
    if hasattr(param_value, 'item'):  # numpy scalar
        param_value = param_value.item()
    elif hasattr(param_value, '__float__'):  # other numeric types
        param_value = float(param_value)
    
    # Use str() which works correctly for all numeric types
    cmd.extend([f"--{param_name}", str(param_value)])

    if fix_param:
        # Only add the fix flag if it exists in HESE_fit.py; if the flag is not defined,
        # HESE_fit.py will raise a normal argparse error.
        cmd.append(f"--fix_{param_name}")

    # Add any additional arguments
    # Skip the parameter we're scanning to avoid overriding it
    for key, value in kwargs.items():
        if value is not None and key != param_name and key != f"fix_{param_name}":
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

    return cmd


def _parse_hese_fit_output(stdout):
    """
    Parse the stdout from HESE_fit.py to extract:
      - overall best-fit LLH and parameters
      - both epsilon_DOM interval fits (if present)
      - total fit time

    This mirrors the parsing logic used in scan_cutoff_energy.py.
    """
    output_lines = stdout.split("\n")

    best_llh = None
    best_params = {}
    fit_results = []
    fit_time = None

    # Parse the overall best fit
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
                current_fit["llh"] = float(line.split("  -LLH:")[1].strip())
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


def run_fit_at_value(
    param_name,
    param_value,
    model="cutoff",
    fix_param=True,
    python_executable=None,
    **kwargs,
):
    """
    Run HESE_fit.py for a specific parameter value.

    Parameters
    ----------
    param_name : str
        Name of the parameter in HESE_fit.py (e.g. 'cutoff_energy', 'astro_gamma', 'Mphi').
    param_value : float
        Parameter value to scan.
    model : str
        Model string passed to HESE_fit.py (e.g. 'spl', 'cutoff', 'bpl', 'lp').
    fix_param : bool
        If True, keep the parameter fixed during the fit via '--fix_<param_name>'.
    python_executable : str or None
        Python executable to use; if None, uses sys.executable.
    **kwargs:
        Extra CLI options to pass through to HESE_fit.py.

    Returns
    -------
    best_llh : float or None
    best_params : dict
    fit_results : list[dict]
    fit_time : float or None
    """
    cmd = _build_cmd(
        param_name,
        param_value,
        model=model,
        fix_param=fix_param,
        python_executable=python_executable,
        **kwargs,
    )

    try:
        # Pass environment variables to subprocess so it can find photospline
        env = os.environ.copy()
        # Set working directory to script directory so relative paths work
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Always print command for debugging (can be disabled with DEBUG_SCAN=0)
        if os.environ.get('DEBUG_SCAN', '1').lower() not in ('0', 'false', 'no'):
            print('Executing command:', ' '.join(cmd))
        
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
        print(
            f"Error running fit for {param_name}={param_value:.4g}, model='{model}':"
        )
        print(e.stderr if e.stderr else e.stdout)
        return None, None, [], None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generic 1D profile-likelihood scan over any HESE_fit.py parameter.\n"
            "Example: scan_parameter.py --param cutoff_energy --pmin 1e5 --pmax 1e7 "
            "--npoints 20 --model cutoff\n"
            "Another example (nuSIprop): scan_parameter.py --param Mphi --pmin 0.1 "
            "--pmax 100 --npoints 15 --log_space --model cutoff"
        )
    )

    parser.add_argument(
        "--param",
        type=str,
        required=True,
        help=(
            "Name of the parameter to scan (must match HESE_fit.py arguments, "
            "e.g. 'cutoff_energy', 'astro_gamma', 'Mphi', 'g', 'mntot')."
        ),
    )
    parser.add_argument(
        "--pmin",
        type=float,
        required=True,
        help="Minimum value of the parameter to scan.",
    )
    parser.add_argument(
        "--pmax",
        type=float,
        required=True,
        help="Maximum value of the parameter to scan.",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=20,
        help="Number of scan points (default: 20).",
    )
    parser.add_argument(
        "--log_space",
        action="store_true",
        help="Use log-spaced parameter values between pmin and pmax.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="spl",
        choices=["spl", "cutoff", "bpl", "lp"],
        help="Astrophysical flux model (passed to HESE_fit.py).",
    )
    parser.add_argument(
        "--save_data",
        type=str,
        default=None,
        help="Save scan data to JSON file (optional).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help=(
            "Python executable to use when calling HESE_fit.py. "
            "Defaults to the current interpreter."
        ),
    )
    
    # Cluster mode
    parser.add_argument(
        "--cluster_mode",
        action="store_true",
        help="Run in cluster mode (single point)",
    )
    parser.add_argument(
        "--job_index",
        type=int,
        default=None,
        help="Job index for cluster mode (0-indexed)",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="parameter_scan_results",
        help="Output directory for results (cluster mode)",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="results.jsonl",
        help="Name of the JSONL results file (default: results.jsonl)",
    )
    
    # nuSIprop options
    parser.add_argument("--majorana", type=str, default=None,
                       help="Use Majorana (True) or Dirac (False) neutrinos for nuSIprop (default: True)")
    parser.add_argument("--normal", type=str, default=None,
                       help="Use normal (True) or inverted (False) mass ordering for nuSIprop (default: True)")
    
    # Optional fixed parameters
    parser.add_argument("--Mphi", type=float, default=None,
                       help="Fix Mphi parameter to this value (requires --fix_Mphi)")
    parser.add_argument("--fix_Mphi", action="store_true",
                       help="Fix Mphi parameter in fit")
    parser.add_argument("--g", type=float, default=None,
                       help="Fix g parameter to this value (requires --fix_g)")
    parser.add_argument("--fix_g", action="store_true",
                       help="Fix g parameter in fit")
    parser.add_argument("--mntot", type=float, default=None,
                       help="Fix mntot parameter to this value (requires --fix_mntot)")
    parser.add_argument("--fix_mntot", action="store_true",
                       help="Fix mntot parameter in fit")
    parser.add_argument("--astro_gamma", type=float, default=None,
                       help="Set initial value for astro_gamma parameter")
    parser.add_argument("--nuSI", type=str, default=None,
                       help="Enable nuSIprop secret interactions (True/False, default: True). If False, fixes Mphi, g, mntot and sets g=1e-30")
    parser.add_argument("--HESE12", type=str, default=None,
                       help="Use HESE12 data instead of HESE data (True/False, default: False)")
    
    # Optimization parameters
    parser.add_argument("--pgtol", type=float, default=None,
                       help="Gradient tolerance for L-BFGS-B optimizer")
    parser.add_argument("--factr", type=float, default=None,
                       help="Convergence factor for L-BFGS-B optimizer")
    parser.add_argument("--m", type=int, default=None,
                       help="Number of corrections used in L-BFGS-B")
    parser.add_argument("--maxiter", type=int, default=None,
                       help="Maximum number of iterations for L-BFGS-B")

    args = parser.parse_args()

    param_name = args.param

    # Generate scan values
    if args.log_space:
        values = np.logspace(np.log10(args.pmin), np.log10(args.pmax), args.npoints)
    else:
        values = np.linspace(args.pmin, args.pmax, args.npoints)

    total_points = len(values)

    if args.cluster_mode:
        # Cluster mode: run single point
        if args.job_index is None:
            print("Error: --job_index required in cluster mode")
            sys.exit(1)
        
        if args.job_index >= total_points:
            print(f"Job index {args.job_index} >= total points {total_points}, skipping")
            sys.exit(0)
        
        value = values[args.job_index]
        
        print(f"Running parameter scan point [{args.job_index}/{total_points-1}]: "
              f"{param_name}={value:.4g} (model='{args.model}')")
        print(f"  Setting {param_name} to {value} and fixing it in the fit")
        
        start_time = time_module.time()
        # Prepare kwargs for HESE_fit.py arguments
        fit_kwargs = {}
        if args.majorana is not None:
            fit_kwargs["majorana"] = args.majorana
        if args.normal is not None:
            fit_kwargs["normal"] = args.normal
        
        # Add optional fixed parameters
        if args.Mphi is not None:
            fit_kwargs["Mphi"] = args.Mphi
        if args.fix_Mphi:
            fit_kwargs["fix_Mphi"] = True
        if args.g is not None:
            fit_kwargs["g"] = args.g
        if args.fix_g:
            fit_kwargs["fix_g"] = True
        if args.mntot is not None:
            fit_kwargs["mntot"] = args.mntot
        if args.fix_mntot:
            fit_kwargs["fix_mntot"] = True
        if args.astro_gamma is not None:
            fit_kwargs["astro_gamma"] = args.astro_gamma
        if args.nuSI is not None:
            fit_kwargs["nuSI"] = args.nuSI
        if args.HESE12 is not None:
            fit_kwargs["HESE12"] = args.HESE12
        
        # Pass cluster_mode to HESE_fit.py so it uses the correct nuSIprop path
        fit_kwargs["cluster_mode"] = True
        
        # Add optimization parameters
        if args.pgtol is not None:
            fit_kwargs["pgtol"] = args.pgtol
        if args.factr is not None:
            fit_kwargs["factr"] = args.factr
        if args.m is not None:
            fit_kwargs["m"] = args.m
        if args.maxiter is not None:
            fit_kwargs["maxiter"] = args.maxiter
        
        llh, params, fit_results, fit_time = run_fit_at_value(
            param_name,
            param_value=value,
            model=args.model,
            fix_param=True,
            python_executable=args.python,
            **fit_kwargs
        )
        elapsed = time_module.time() - start_time
        
        # Save result to single shared file (JSONL format - one JSON object per line)
        # This allows parallel jobs to safely append without file locking
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(args.output_dir, args.results_file)
        
        # Handle None and np.nan for JSON
        llh_json = float(llh) if (llh is not None and not np.isnan(llh)) else None
        
        result = {
            "job_index": args.job_index,
            "param_name": param_name,
            "param_value": float(value),
            "llh": llh_json,
            "params": params if params else {},
            "fit_results": fit_results if fit_results else [],
            "fit_time": float(fit_time) if fit_time else float(elapsed),
            "model": args.model,
        }
        
        # Append to JSONL file (one JSON object per line, safe for parallel writes)
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        
        print(f"Result appended to {results_file}")
        if llh is not None:
            print(f"  -LLH: {llh:.6f}")
        else:
            print("  Fit failed")
        
        sys.exit(0)
    
    # Standalone mode: run all points sequentially
    print(
        f"Scanning {len(values)} values of '{param_name}' from "
        f"{args.pmin:.4g} to {args.pmax:.4g} (logspace={args.log_space}) "
        f"for model '{args.model}'"
    )
    print()

    # Scan
    llh_values = []
    all_params = []
    all_fit_results = []
    fit_times = []

    total_start_time = time_module.time()

    for i, value in enumerate(values):
        print(
            f"[{i+1}/{len(values)}] Running fit for {param_name} = {value:.4g} "
            f"(model='{args.model}')..."
        )
        point_start_time = time_module.time()

        # Prepare kwargs for HESE_fit.py arguments
        fit_kwargs = {}
        if args.majorana is not None:
            fit_kwargs["majorana"] = args.majorana
        if args.normal is not None:
            fit_kwargs["normal"] = args.normal

        # Add optional fixed parameters
        if args.Mphi is not None:
            fit_kwargs["Mphi"] = args.Mphi
        if args.fix_Mphi:
            fit_kwargs["fix_Mphi"] = True
        if args.g is not None:
            fit_kwargs["g"] = args.g
        if args.fix_g:
            fit_kwargs["fix_g"] = True
        if args.mntot is not None:
            fit_kwargs["mntot"] = args.mntot
        if args.fix_mntot:
            fit_kwargs["fix_mntot"] = True
        if args.astro_gamma is not None:
            fit_kwargs["astro_gamma"] = args.astro_gamma
        if args.nuSI is not None:
            fit_kwargs["nuSI"] = args.nuSI
        if args.HESE12 is not None:
            fit_kwargs["HESE12"] = args.HESE12

        # Pass cluster_mode to HESE_fit.py so it uses the correct nuSIprop path
        # (even in standalone mode, we want HESE_fit.py to use cluster paths when called from scan_parameter.py)
        fit_kwargs["cluster_mode"] = True

        # Add optimization parameters
        if args.pgtol is not None:
            fit_kwargs["pgtol"] = args.pgtol
        if args.factr is not None:
            fit_kwargs["factr"] = args.factr
        if args.m is not None:
            fit_kwargs["m"] = args.m
        if args.maxiter is not None:
            fit_kwargs["maxiter"] = args.maxiter

        llh, params, fit_results, fit_time = run_fit_at_value(
            param_name,
            param_value=value,
            model=args.model,
            fix_param=True,
            python_executable=args.python,
            **fit_kwargs
        )
        point_elapsed = time_module.time() - point_start_time

        if llh is not None:
            llh_values.append(llh)
            all_params.append(params)
            all_fit_results.append(fit_results)
            fit_times.append(fit_time if fit_time is not None else point_elapsed)

            print(f"  -LLH: {llh:.6f}")

            # Timing info
            if fit_time is not None:
                print(
                    f"  Fit took {fit_time:.1f} s ({fit_time/60:.1f} min)"
                )
            else:
                print(
                    f"  Elapsed time: {point_elapsed:.1f} s "
                    f"({point_elapsed/60:.1f} min)"
                )

            # Interval fits
            if len(fit_results) == 2:
                print("  Interval fits:")
                for fit in fit_results:
                    print(
                        f"    {fit['interval']} ({fit['epsilon_range']}): "
                        f"-LLH = {fit['llh']:.6f}"
                    )

            # Estimated remaining time
            if i > 0:
                avg_time = np.mean(fit_times)
                remaining_points = len(values) - (i + 1)
                est_remaining = avg_time * remaining_points
                print(
                    f"  Estimated remaining: {est_remaining/60:.1f} min "
                    f"({est_remaining/3600:.1f} h)"
                )
        else:
            print("  Failed to get result.")
            llh_values.append(np.nan)
            all_fit_results.append([])
            fit_times.append(point_elapsed)

        print()

    total_elapsed = time_module.time() - total_start_time
    print(
        f"\nTotal scan time: {total_elapsed/60:.1f} min "
        f"({total_elapsed/3600:.1f} h)"
    )
    if fit_times:
        print(f"Average time per point: {np.mean(fit_times)/60:.1f} min")

    # Save data if requested
    if args.save_data:
        data = {
            "param_name": param_name,
            "values": values.tolist(),
            "llh_values": llh_values,
            "all_params": all_params,
            "all_fit_results": all_fit_results,
            "fit_times": fit_times,
            "model": args.model,
        }

        with open(args.save_data, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {args.save_data}")
    
    # Also save to JSONL format for consistency with cluster mode
    # (useful if you want to combine standalone and cluster results)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(args.output_dir, args.results_file)
        # Clear file if it exists for a fresh run
        if os.path.exists(results_file):
            os.remove(results_file)
        
        for idx, (value, llh_val, params_dict, fit_results_list, fit_time_val) in enumerate(
            zip(values, llh_values, all_params, all_fit_results, fit_times)
        ):
            llh_json = float(llh_val) if (llh_val is not None and not np.isnan(llh_val)) else None
            result = {
                "job_index": idx,
                "param_name": param_name,
                "param_value": float(value),
                "llh": llh_json,
                "params": params_dict if params_dict else {},
                "fit_results": fit_results_list if fit_results_list else [],
                "fit_time": float(fit_time_val) if fit_time_val is not None else None,
                "model": args.model,
            }
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()


