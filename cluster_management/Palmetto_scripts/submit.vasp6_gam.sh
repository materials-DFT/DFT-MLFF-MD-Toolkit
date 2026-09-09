#!/bin/bash

#SBATCH --job-name=vasp_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=10-00:00:00
#SBATCH --partition=skystd
#SBATCH --constraint=extension_avx512
#SBATCH --output=job.out
#SBATCH --error=job.err

module purge
module load intel-oneapi-compilers/2024.0.2
module load intel-oneapi-mkl/2024.0.0
module load intel-oneapi-mpi/2021.11.0

export PATH=/software/commercial/vasp_sky/vasp_na/vasp.6.5.1_vtst_cpu/bin:$PATH

cd $SLURM_SUBMIT_DIR

# Fix stack overflow: unlimited stack for main thread + 512 MB per OMP thread
ulimit -s unlimited
export OMP_STACKSIZE=512m
export OMP_NUM_THREADS=1

# Verify vasp_gam is accessible before running
if ! command -v vasp_gam &>/dev/null; then
    echo "ERROR: vasp_gam not found in PATH" >&2
    exit 1
fi

srun vasp_gam
