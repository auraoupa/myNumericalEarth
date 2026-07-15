#!/bin/bash

#OAR -n one_degree_H200
#OAR -l /nodes=1/gpu=1,walltime=02:30:00
#OAR -p gpumodel='H200'
#OAR --stdout one_degree_H200.out%jobid%
#OAR --stderr one_degree_H200.err%jobid%
#OAR --project pr-data-ocean

source ~/.bashrc

mkdir -p /bettik/alberta/NumericalEarth/tmpdir-one-degree-global-ocean-sea-ice-simulation-H200

cd /bettik/alberta/NumericalEarth/tmpdir-one-degree-global-ocean-sea-ice-simulation-H200

cp /bettik/alberta/git/myNumericalEarth/one-degree-global-ocean-sea-ice-simulation/run_on_bigfoot.jl .

export JULIA_DEPOT_PATH=/bettik/alberta/julia_myNEGPU
/bettik/alberta/bin/.juliaup/bin/julia run_on_bigfoot.jl
