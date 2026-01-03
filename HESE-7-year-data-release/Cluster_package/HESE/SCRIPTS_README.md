# Shell Scripts Guide

This directory contains several shell scripts for managing cluster transfers. Here's what each does:

## Essential Scripts (Keep These)

### 1. `package_for_cluster.sh` ⭐ **MAIN SCRIPT**
**Purpose:** Creates a clean package of only the necessary files for cluster transfer.

**What it does:**
- Copies core Python files (uses `cluster_files/` versions when available)
- Copies `resources/` directory
- Creates a `HESE_cluster_package/` directory ready for transfer

**Usage:**
```bash
./package_for_cluster.sh
# Creates HESE_cluster_package/HESE/ with all files
```

**When to use:** Always use this before transferring to cluster. It ensures you only send what's needed.

---

### 2. `submit_grid_scan.sh` ⭐ **FOR CLUSTER**
**Purpose:** SLURM job submission script for running grid scans on the cluster.

**What it does:**
- Submits parallel jobs using SLURM job arrays
- Each job runs one grid point independently

**Usage:** On the cluster, after transferring files:
```bash
# Edit the script with your paths, then:
sbatch submit_grid_scan.sh
```

**When to use:** Only on the cluster, to submit grid scan jobs.

---

---

## Recommended Workflow

### Step 1: Package Files (Local)
```bash
cd HESE-7-year-data-release/HESE-7-year-data-release
./package_for_cluster.sh
```

### Step 2: Transfer to Cluster
```bash
# Option A: Create tarball and transfer
tar -czf HESE_cluster_package.tar.gz HESE_cluster_package/
scp HESE_cluster_package.tar.gz user@cluster:/path/to/destination/

# Option B: Use rsync (if you have SSH access)
rsync -avz HESE_cluster_package/HESE/ user@cluster:/path/to/repo/HESE/
```

### Step 3: On Cluster - Extract and Setup
```bash
# If using tarball:
tar -xzf HESE_cluster_package.tar.gz
cp -r HESE_cluster_package/HESE/* /path/to/repo/HESE/

# Update nuSIprop path if needed
cd /path/to/repo/HESE
python update_nusiprop_path.py HESE_fit.py --nusiprop_path ../nuSIprop
```

### Step 4: On Cluster - Submit Jobs
```bash
# Edit submit_grid_scan.sh with correct paths
sbatch submit_grid_scan.sh
```

---

## Summary

**Essential Scripts (Only 2):**
- ✅ `package_for_cluster.sh` - Main packaging tool (use before transferring)
- ✅ `submit_grid_scan.sh` - For cluster job submission (use on cluster)

That's it! Simple and clean.

