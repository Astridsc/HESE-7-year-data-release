# 2D Grid Scan for Cluster Computing

This directory contains scripts for running 2D parameter grid scans (e.g., Mphi vs g for nuSIprop) on computer clusters.

## Files

- `scan_2d_grid.py`: Main script for running grid scans (standalone or cluster mode)
- `submit_grid_scan.sh`: Example SLURM submission script
- `aggregate_grid_results.py`: Script to combine results and create plots

## Usage

### Standalone Mode (Sequential)

Run all grid points sequentially on a single machine:

```bash
python scan_2d_grid.py \
    --param1 Mphi \
    --p1min 0.1 \
    --p1max 100 \
    --n1 10 \
    --log1 \
    --param2 g \
    --p2min 0.001 \
    --p2max 1.0 \
    --n2 10 \
    --log2 \
    --model nusiprop \
    --output_dir grid_scan_results
```

### Cluster Mode (Parallel with SLURM)

1. **Edit `submit_grid_scan.sh`**:
   - Set `--array=0-N` where N = (n1 * n2) - 1
   - For 10x10 grid: `--array=0-99`
   - For 20x20 grid: `--array=0-399`
   - Update paths and conda environment
   - Adjust resource requirements (time, memory, etc.)

2. **Submit jobs**:
   ```bash
   sbatch submit_grid_scan.sh
   ```

3. **Monitor jobs**:
   ```bash
   squeue -u $USER
   ```

4. **After all jobs complete, aggregate results**:
   ```bash
   python aggregate_grid_results.py \
       --output_dir grid_scan_results \
       --plot grid_scan_2d.png
   ```

## Example: 10x10 Grid Scan

```bash
# Total points = 10 * 10 = 100
# So use --array=0-99 in submit_grid_scan.sh

python scan_2d_grid.py \
    --param1 Mphi --p1min 0.1 --p1max 100 --n1 10 --log1 \
    --param2 g --p2min 0.001 --p2max 1.0 --n2 10 --log2 \
    --model nusiprop \
    --output_dir grid_scan_results \
    --cluster_mode \
    --job_index $SLURM_ARRAY_TASK_ID
```

## Output Structure

Each grid point produces a JSON file:
- `result_00000.json`, `result_00001.json`, etc.
- Contains: parameter values, LLH, fit parameters, fit time

After aggregation:
- `summary.json`: Combined results with LLH grid
- `grid_scan_2d.png`: 2D contour plot

## Notes

- Each grid point runs independently, perfect for parallelization
- Results are saved incrementally (safe if jobs fail)
- Can resume by checking which `result_*.json` files exist
- Adjust `--n1` and `--n2` to control grid resolution
- Use `--log1` and `--log2` for log-spaced grids (recommended for nuSIprop parameters)

## Other Job Schedulers

For PBS/Torque, modify `submit_grid_scan.sh`:
```bash
#!/bin/bash
#PBS -N grid_scan
#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00
#PBS -t 0-99

python scan_2d_grid.py ... --job_index $PBS_ARRAY_INDEX
```

For SGE:
```bash
#$ -t 0-99
python scan_2d_grid.py ... --job_index $SGE_TASK_ID
```

