#!/bin/bash
#SBATCH -J single_job
#SBATCH -A naiss2025-22-1669
#SBATCH -t 01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G   
#SBATCH --array=0-25
#SBATCH --output=single_job_%A_%a.out
#SBATCH --error=single_job_%A_%a.err

# Note: After all jobs complete, run combine_grid_results.py to create summary.json:
# python3 combine_grid_results.py --results_dir grid_scan_results --n1 25 --n2 25

cd /home/x_astsc/Cluster_package/HESE

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Cluster mode (for job arrays)
python run_multiple_single_jobs.py \
    --mntot_values "0.1,0.098,0.102,0.095,0.105,0.09,0.11,0.085,0.115,0.08,0.12,0.06,0.14" \
    --astro_gamma_values "2.5,2.505,2.495,2.51,2.49,2.55,2.45,2.5025,2.4975,2.6,2.4,2.9,2.1" \
    --model nusiprop \
    --output_dir single_job_results \
    --cluster_mode \
    --job_index $SLURM_ARRAY_TASK_ID
