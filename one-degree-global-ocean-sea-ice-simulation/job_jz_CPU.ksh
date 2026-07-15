#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH -J julia_one-degree
#SBATCH -e julia_one-degree.e%j
#SBATCH -o julia_one-degree.o%j
#SBATCH -A cli@cpu
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --dependency=singleton
#SBATCH --exclusive


source ~/.bashrc
load_julia

export JULIA_DEPOT_PATH=/lustre/fswork/projects/rech/cli/rote001/DEV/julia_myNE
export JULIA_PKG_OFFLINE=true

cd /lustre/fswork/projects/rech/cli/rote001/DEV/git/myNumericalEarth/one-degree-global-ocean-sea-ice-simulation/onenodeCPU

julia run_on_jz_CPU.jl
