#!/bin/bash
# Script to sync repository to cluster using rsync
# Usage: ./sync_to_cluster.sh user@cluster:/path/to/destination

if [ -z "$1" ]; then
    echo "Usage: $0 user@cluster:/path/to/destination"
    echo "Example: $0 username@cluster.university.edu:/home/username/HESE-7-year-data-release"
    exit 1
fi

DEST="$1"

# Exclude unnecessary files for faster transfer
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='*.png' \
    --exclude='*.pdf' \
    --exclude='*.jpg' \
    --exclude='*.jpeg' \
    --exclude='grid_scan_results/' \
    --exclude='*.out' \
    --exclude='*.err' \
    --exclude='.DS_Store' \
    ./ "$DEST"

echo "Sync complete!"

