#!/bin/bash
#SBATCH -J grid_scan
#SBATCH -A naiss2025-22-1669
#SBATCH -t 03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --array=0-624
#SBATCH --output=grid_scan_%A_%a.out
#SBATCH --error=grid_scan_%A_%a.err

# Note: After all jobs complete, run combine_grid_results.py to create summary.json:
# python3 combine_grid_results.py --results_dir grid_scan_results --n1 25 --n2 25

cd /home/x_astsc/Cluster_package/HESE

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

#export PYTHONPATH=/cfs/klemming/home/a/astridsc/photospline-installation/lib/python3.11/site-packages:$PYTHONPATH
#export LD_LIBRARY_PATH=/cfs/klemming/home/a/astridsc/photospline-installation/lib64:/cfs/klemming/home/a/astridsc/cfitsio-installation/lib:$LD_LIBRARY_PATH

python3 scan_2d_grid.py \
    --param1 Mphi \
    --p1min 0.03 \
    --p1max 100 \
    --n1 25 \
    --log1 \
    --param2 g \
    --p2min 0.0001 \
    --p2max 1.0 \
    --n2 25 \
    --log2 \
    --prompt_norm 0.0 \
    --fix_prompt_norm \
    --model nusiprop \
    --majorana True \
    --normal False \
    --pgtol 1e-6 \
    --factr 10 \
    --m 10 \
    --maxiter 400 \
    --output_dir grid_scan_results \
    --results_file results_75MIO_free_mntot.jsonl \
    --cluster_mode \
    --no_retry \
    --job_index $SLURM_ARRAY_TASK_ID
