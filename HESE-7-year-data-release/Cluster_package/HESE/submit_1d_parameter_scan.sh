#!/bin/bash
#SBATCH -J grid_scan
#SBATCH -A naiss2025-22-1669
#SBATCH -t 03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --array=0-14
#SBATCH --output=grid_scan_%A_%a.out
#SBATCH --error=grid_scan_%A_%a.err



cd /home/x_astsc/Cluster_package/HESE

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


python3 scan_parameter.py \
    --model nusiprop \
    --param mntot \
    --pmin 0.06 \
    --pmax 0.15 \
    --npoints 15 \
    --cluster_mode \
    --job_index $SLURM_ARRAY_TASK_ID \
    --output_dir 1d_parameter_scan
    
