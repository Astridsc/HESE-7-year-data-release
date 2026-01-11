#!/bin/bash
#SBATCH -J grid_scan
#SBATCH -A naiss2025-22-1669
#SBATCH -t 04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --array=0-676
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
# GLÖM INTE --logl FÖR Mphi och g!

python3 scan_2d_grid.py \
    --model spl \
    --majorana True \
    --normal True \
    --prompt_norm 0.0 \
    --fix_prompt_norm \
    --g 0.03 \
    --param1 Mphi \
    --p1min 0.5 \
    --p1max 20 \
    --n1 26 \
    --log1 \
    --param2 mntot \
    --p2min 0.06 \
    --p2max 0.07 \
    --n2 26 \
    --pgtol 1e-6 \
    --factr 10 \
    --m 10 \
    --maxiter 400 \
    --output_dir grid_scan_results/MNO \
    --results_file results12_Mphi_05_to_20_Mntot_006_to_007.jsonl \
    --cluster_mode \
    --no_retry \
    --job_index $SLURM_ARRAY_TASK_ID
