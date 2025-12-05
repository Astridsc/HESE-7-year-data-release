#!/bin/bash
#SBATCH --job-name=grid_scan
#SBATCH --output=grid_scan_%A_%a.out
#SBATCH --error=grid_scan_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-99  # Adjust based on total grid points (n1 * n2)

# Example: 10x10 grid = 100 points, so --array=0-99
# For 20x20 grid = 400 points, use --array=0-399

# Activate your conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# Set working directory
cd /path/to/HESE-7-year-data-release/HESE-7-year-data-release

# Run single grid point
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
    --output_dir grid_scan_results \
    --cluster_mode \
    --job_index $SLURM_ARRAY_TASK_ID

