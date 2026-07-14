#!/bin/bash

#OAR -n julie_near_global_gpu
#OAR -l /nodes=1/gpu=1,walltime=22:30:00
#OAR -p gpumodel='A100'
#OAR --stdout julie_near_global_gpu.out%jobid%
#OAR --stderr julie_near_global_gpu.err%jobid%
#OAR --project pr-data-ocean

source ~/.bashrc

cd /bettik/alberta/git/myNumericalEarth/one-degree-global-ocean-sea-ice-simulation/GPUA100

export JULIA_DEPOT_PATH=/bettik/alberta/julia_myNEGPUA100
/home/alberta/.juliaup/bin/julia run_on_bigfoot.jl
