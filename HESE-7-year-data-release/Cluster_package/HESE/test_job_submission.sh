#!/bin/bash
# Simple test to see if ANY job submission works

echo "Testing job submission with minimal resources..."

# Try submitting a very simple job
cat > /tmp/test_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --partition=main
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100M

echo "Test job running successfully!"
hostname
date
sleep 5
echo "Test job completed!"
EOF

echo "Attempting to submit test job..."
sbatch /tmp/test_job.sh

if [ $? -eq 0 ]; then
    echo "SUCCESS: Job submitted! Check with: squeue -u \$USER"
    echo "If this works, the issue is with resource requirements in submit_jobs_individually.sh"
else
    echo "FAILED: Cannot submit any jobs. You may need to:"
    echo "  1. Contact cluster support about your account limits"
    echo "  2. Check available partitions: sinfo"
    echo "  3. Check your account limits: sacctmgr show assoc where user=\$USER"
fi

rm -f /tmp/test_job.sh


