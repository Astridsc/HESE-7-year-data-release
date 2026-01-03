#!/bin/bash
# Script to submit a large grid scan in batches to avoid concurrent CPU minute limits
# Usage: ./submit_grid_scan_batched.sh

# Grid parameters (25x25 = 625 points)
N1=25
N2=25
TOTAL_POINTS=$((N1 * N2))

# Batch size: how many jobs to submit at once
# Adjust this based on your account's concurrent CPU minute limit
# Start with 50-100 jobs at a time
BATCH_SIZE=50

# Calculate number of batches
NUM_BATCHES=$(( (TOTAL_POINTS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Submitting $TOTAL_POINTS jobs in $NUM_BATCHES batches of ~$BATCH_SIZE jobs each"

# Create batch submission script
BATCH_SCRIPT=$(mktemp)
cat > "$BATCH_SCRIPT" << 'BATCH_EOF'
#!/bin/bash
#SBATCH --job-name=grid_scan
#SBATCH --output=grid_scan_%A_%a.out
#SBATCH --error=grid_scan_%A_%a.err
#SBATCH --account=naiss2025-22-846
#SBATCH --partition=main
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=BATCH_ARRAY_PLACEHOLDER

cd /cfs/klemming/home/a/astridsc/HESE_cluster_package/HESE

export PYTHONPATH=/cfs/klemming/home/a/astridsc/photospline-installation/lib/python3.11/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/cfs/klemming/home/a/astridsc/photospline-installation/lib64:/cfs/klemming/home/a/astridsc/cfitsio-installation/lib:$LD_LIBRARY_PATH

python3 scan_2d_grid.py \
    --param1 Mphi \
    --p1min 0.03 \
    --p1max 100.0 \
    --n1 25 \
    --param2 g \
    --p2min 0.0001 \
    --p2max 1.0 \
    --n2 25 \
    --log2 \
    --model nusiprop \
    --output_dir grid_scan_results \
    --cluster_mode \
    --job_index $SLURM_ARRAY_TASK_ID
BATCH_EOF

# Submit batches
for BATCH in $(seq 0 $((NUM_BATCHES - 1))); do
    START_IDX=$((BATCH * BATCH_SIZE))
    END_IDX=$((START_IDX + BATCH_SIZE - 1))
    
    # Don't exceed total points
    if [ $END_IDX -ge $TOTAL_POINTS ]; then
        END_IDX=$((TOTAL_POINTS - 1))
    fi
    
    # Create batch-specific script
    BATCH_FILE="${BATCH_SCRIPT}_batch${BATCH}.sh"
    sed "s|BATCH_ARRAY_PLACEHOLDER|${START_IDX}-${END_IDX}|g" "$BATCH_SCRIPT" > "$BATCH_FILE"
    
    # Submit the batch
    echo "Submitting batch $((BATCH + 1))/$NUM_BATCHES: jobs $START_IDX-$END_IDX"
    JOB_OUTPUT=$(sbatch "$BATCH_FILE" 2>&1)
    
    if [ $? -eq 0 ]; then
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+$')
        echo "  Batch submitted as job array: $JOB_ID"
    else
        echo "  ERROR: Failed to submit batch: $JOB_OUTPUT"
        echo "  You may need to reduce BATCH_SIZE or wait for running jobs to finish"
        rm -f "$BATCH_FILE"
        break
    fi
    
    # Clean up
    rm -f "$BATCH_FILE"
    
    # Small delay between batches to avoid overwhelming the scheduler
    sleep 2
done

# Clean up template
rm -f "$BATCH_SCRIPT"

echo ""
echo "Done! Submitted $NUM_BATCHES batches."
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Note: If you still get limits, reduce BATCH_SIZE in this script"


