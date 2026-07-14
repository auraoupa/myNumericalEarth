#!/bin/bash

#OAR -n julie_one_degree_gpu
#OAR -l /nodes=1/gpu=1,walltime=22:30:00
#OAR -p gpumodel='V100'
#OAR --stdout julie_one_degree_gpu.out%jobid%
#OAR --stderr julie_one_degree_gpu.err%jobid%
#OAR --project pr-data-ocean

source ~/.bashrc

cd /bettik/alberta/git/myNumericalEarth/one-degree-global-ocean-sea-ice-simulation/GPU_bigfoot
cp /bettik/alberta/git/myNumericalEarth/one-degree-global-ocean-sea-ice-simulation/run_on_bigfoot.jl .

export JULIA_DEPOT_PATH=/bettik/alberta/julia_myNEGPU
/home/alberta/.juliaup/bin/julia run_on_bigfoot.jl
