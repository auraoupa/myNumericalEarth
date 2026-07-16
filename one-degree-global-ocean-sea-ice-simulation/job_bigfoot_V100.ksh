#!/bin/bash

#OAR -n one_degree_V100
#OAR -l /nodes=1/gpu=1,walltime=22:30:00
#OAR -p gpumodel='V100'
#OAR --stdout one_degree_V100.out%jobid%
#OAR --stderr one_degree_V100.err%jobid%
#OAR --project pr-data-ocean

source ~/.bashrc

mkdir -p /bettik/alberta/NumericalEarth/tmpdir-one-degree-global-ocean-sea-ice-simulation-V100

cd /bettik/alberta/NumericalEarth/tmpdir-one-degree-global-ocean-sea-ice-simulation-V100

cp /bettik/alberta/git/myNumericalEarth/one-degree-global-ocean-sea-ice-simulation/run_on_bigfoot_072026.jl .

export JULIA_DEPOT_PATH=/bettik/alberta/julia_myNEGPUV100
/home/alberta/.juliaup/bin/julia run_on_bigfoot_072026.jl
