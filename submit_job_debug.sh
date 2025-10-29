#!/bin/bash
#SBATCH --job-name=ptycho_debug
#SBATCH --account=m5073
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu
#SBATCH --output=ptycho_debug_%j.out
#SBATCH --error=ptycho_debug_%j.err

# Load Python module
module load python/3.11

# Navigate to project directory
cd /global/cfs/cdirs/m5073/ptycho_simulation_factory

# Activate virtual environment (for main shell environment)
source .venv/bin/activate

# Run with multiple GPUs across multiple nodes using srun
# 2 nodes × 4 GPUs = 8 GPUs total (8 tasks) for debug testing
# SLURM sets SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID automatically
# The Python code will map these to RANK, WORLD_SIZE, LOCAL_RANK for PyTorch distributed
# MASTER_ADDR and MASTER_PORT are set automatically in parallel.py from SLURM_JOB_NODELIST
srun --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=1 \
     /global/cfs/cdirs/m5073/ptycho_simulation_factory/.venv/bin/python \
     main.py config/batch_simulation.yaml

