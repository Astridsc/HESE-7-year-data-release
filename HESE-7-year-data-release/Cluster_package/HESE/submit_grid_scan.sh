#!/bin/bash
#SBATCH --job-name=grid_scan
#SBATCH --output=grid_scan_%A_%a.out
#SBATCH --error=grid_scan_%A_%a.err
#SBATCH --account=naiss2025-22-846
#SBATCH --partition=main
#SBATCH --time=02:00:00  # Reduced from 3h - adjust if fits take longer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G  # Reduced from 16G - adjust if needed
#SBATCH --array=0-624  # 25x25 grid = 625 points
# For 2x2 grid = 4 points, use --array=0-3
# For 10x10 grid = 100 points, use --array=0-99
# For 20x20 grid = 400 points, use --array=0-399

# Set working directory
cd /cfs/klemming/home/a/astridsc/HESE_cluster_package/HESE

# Set environment variables
export PYTHONPATH=/cfs/klemming/home/a/astridsc/photospline-installation/lib/python3.11/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/cfs/klemming/home/a/astridsc/photospline-installation/lib64:/cfs/klemming/home/a/astridsc/cfitsio-installation/lib:$LD_LIBRARY_PATH

# Python optimization: disable bytecode generation for .pyc files (saves I/O)
export PYTHONDONTWRITEBYTECODE=1

# Run single grid point
python3 scan_2d_grid.py \
    --param1 Mphi \
    --p1min 0.03 \
    --p1max 100.0 \
    --n1 25 \
    --log1 \
    --param2 g \
    --p2min 0.0001 \
    --p2max 1.0 \
    --n2 25 \
    --log2 \
    --model nusiprop \
    --output_dir grid_scan_results \
    --cluster_mode \
    --job_index $SLURM_ARRAY_TASK_ID

