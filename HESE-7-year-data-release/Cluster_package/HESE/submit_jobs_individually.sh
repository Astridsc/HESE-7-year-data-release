#!/bin/bash
# Script to submit individual jobs for each grid point (workaround for job array limits)

# Grid parameters
PARAM1="Mphi"
P1MIN=5.0
P1MAX=10.0
N1=2

PARAM2="g"
P2MIN=0.01
P2MAX=0.1
N2=2
LOG2=true

MODEL="nusiprop"
OUTPUT_DIR="grid_scan_results"

# Cluster paths
WORK_DIR="/cfs/klemming/home/a/astridsc/HESE_cluster_package/HESE"
PYTHONPATH="/cfs/klemming/home/a/astridsc/photospline-installation/lib/python3.11/site-packages"
LD_LIBRARY_PATH="/cfs/klemming/home/a/astridsc/photospline-installation/lib64:/cfs/klemming/home/a/astridsc/cfitsio-installation/lib"

# SLURM account (required on Dardel)
ACCOUNT="naiss2025-22-846"

# Generate grid indices
TOTAL_POINTS=$((N1 * N2))
echo "Submitting $TOTAL_POINTS individual jobs for ${N1}x${N2} grid..."

# Create a temporary submission script template
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --job-name=grid_pt
#SBATCH --output=grid_scan_%j.out
#SBATCH --error=grid_scan_%j.err
#SBATCH --account=ACCOUNT_PLACEHOLDER
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

cd WORK_DIR_PLACEHOLDER

export PYTHONPATH=PYTHONPATH_PLACEHOLDER:$PYTHONPATH
export LD_LIBRARY_PATH=LD_LIBRARY_PATH_PLACEHOLDER:$LD_LIBRARY_PATH

python3 scan_2d_grid.py \
    --param1 PARAM1_PLACEHOLDER \
    --p1min P1MIN_PLACEHOLDER \
    --p1max P1MAX_PLACEHOLDER \
    --n1 N1_PLACEHOLDER \
    --param2 PARAM2_PLACEHOLDER \
    --p2min P2MIN_PLACEHOLDER \
    --p2max P2MAX_PLACEHOLDER \
    --n2 N2_PLACEHOLDER \
    LOG2_FLAG_PLACEHOLDER \
    --model MODEL_PLACEHOLDER \
    --output_dir OUTPUT_DIR_PLACEHOLDER \
    --cluster_mode \
    --job_index JOB_INDEX_PLACEHOLDER
EOF

# Submit each job individually
for JOB_INDEX in $(seq 0 $((TOTAL_POINTS - 1))); do
    # Create job-specific script
    JOB_SCRIPT="${TEMP_SCRIPT}_${JOB_INDEX}.sh"
    sed "s|WORK_DIR_PLACEHOLDER|$WORK_DIR|g; \
         s|PYTHONPATH_PLACEHOLDER|$PYTHONPATH|g; \
         s|LD_LIBRARY_PATH_PLACEHOLDER|$LD_LIBRARY_PATH|g; \
         s|ACCOUNT_PLACEHOLDER|$ACCOUNT|g; \
         s|PARAM1_PLACEHOLDER|$PARAM1|g; \
         s|P1MIN_PLACEHOLDER|$P1MIN|g; \
         s|P1MAX_PLACEHOLDER|$P1MAX|g; \
         s|N1_PLACEHOLDER|$N1|g; \
         s|PARAM2_PLACEHOLDER|$PARAM2|g; \
         s|P2MIN_PLACEHOLDER|$P2MIN|g; \
         s|P2MAX_PLACEHOLDER|$P2MAX|g; \
         s|N2_PLACEHOLDER|$N2|g; \
         s|LOG2_FLAG_PLACEHOLDER|$([ "$LOG2" = true ] && echo "--log2" || echo "")|g; \
         s|MODEL_PLACEHOLDER|$MODEL|g; \
         s|OUTPUT_DIR_PLACEHOLDER|$OUTPUT_DIR|g; \
         s|JOB_INDEX_PLACEHOLDER|$JOB_INDEX|g" "$TEMP_SCRIPT" > "$JOB_SCRIPT"
    
    # Submit the job
    SBATCH_OUTPUT=$(sbatch "$JOB_SCRIPT" 2>&1)
    if [ $? -eq 0 ]; then
        # Extract job ID (format: "Submitted batch job 12345")
        JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oE '[0-9]+$')
        echo "Submitted job $JOB_INDEX -> Job ID: $JOB_ID"
    else
        echo "Failed to submit job $JOB_INDEX: $SBATCH_OUTPUT"
    fi
    
    # Clean up temporary script
    rm -f "$JOB_SCRIPT"
done

# Clean up template
rm -f "$TEMP_SCRIPT"

echo "Done! Submitted $TOTAL_POINTS jobs."
echo "Monitor with: squeue -u \$USER"

