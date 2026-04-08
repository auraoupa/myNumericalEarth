#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --hint=nomultithread # One MPI process per physical core (no hyperthreading)
#SBATCH --time=00:10:00
#SBATCH --account=cli@cpu   # GPU partition
#SBATCH --exclusive
#SBATCH -J julia
#SBATCH -e julia.e%j
#SBATCH -o julia.o%j

source ~/.bashrc

load_julia

cd /lustre/fswork/projects/rech/cli/rote001/DEV/julia

julia test_minimal_cpu.jl
