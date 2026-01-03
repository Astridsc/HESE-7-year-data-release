import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def load_results(results_dir):
    """Load all per-point result_*.json files from a grid scan."""
    pattern = os.path.join(results_dir, "result_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No result_*.json files found in {results_dir}")

    results = []
    for path in files:
        with open(path, "r") as f:
            data = json.load(f)
        results.append(data)

    return results


def build_summary(results, n1, n2):
    """Build summary dict with param values and llh_grid from per-point results."""
    # Use first result as template for names / model
    first = results[0]
    param1_name = first["param1_name"]
    param2_name = first["param2_name"]
    model = first.get("model", "nusiprop")

    # Initialize arrays
    param1_values = np.full(n1, np.nan)
    param2_values = np.full(n2, np.nan)
    llh_grid = np.full((n1, n2), np.inf)
    fit_times = np.full((n1, n2), np.nan)

    for res in results:
        i, j = res["grid_index"]
        p1 = float(res["param1_value"])
        p2 = float(res["param2_value"])
        llh = res["llh"]
        # Handle "inf" string from scan_2d_grid.py
        if isinstance(llh, str) and llh.lower() == "inf":
            llh_val = np.inf
        else:
            llh_val = float(llh)

        ft = res.get("fit_time", None)
        ft_val = float(ft) if ft is not None else np.nan

        param1_values[i] = p1
        param2_values[j] = p2
        llh_grid[i, j] = llh_val
        fit_times[i, j] = ft_val

    # Convert np.inf to "inf" for JSON compatibility
    llh_grid_json = [
        [("inf" if np.isinf(v) else v) for v in row] for row in llh_grid.tolist()
    ]

    # Fit times: keep as list of lists (can contain NaN)
    fit_times_json = fit_times.tolist()

    summary = {
        "param1_name": param1_name,
        "param1_values": param1_values.tolist(),
        "param2_name": param2_name,
        "param2_values": param2_values.tolist(),
        "llh_grid": llh_grid_json,
        "fit_times": fit_times_json,
        "model": model,
        "total_points": int(n1 * n2),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-point result_*.json files from a cluster grid scan "
            "into a single summary.json containing the full llh_grid."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="grid_scan_results",
        help="Directory containing result_*.json files",
    )
    parser.add_argument(
        "--n1",
        type=int,
        required=True,
        help="Number of points in param1 (same as used in scan_2d_grid.py)",
    )
    parser.add_argument(
        "--n2",
        type=int,
        required=True,
        help="Number of points in param2 (same as used in scan_2d_grid.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary.json",
        help="Name of combined summary file (written inside results_dir)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(str(results_dir))
    summary = build_summary(results, args.n1, args.n2)

    out_path = results_dir / args.output
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Combined summary written to {out_path}")


if __name__ == "__main__":
    main()



