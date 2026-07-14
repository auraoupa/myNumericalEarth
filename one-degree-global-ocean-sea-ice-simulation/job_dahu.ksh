#!/bin/bash

#OAR -n julie_one_degree
#OAR -l /nodes=1/core=32,walltime=6:30:00
#OAR --stdout julie_one_degree.out%jobid%
#OAR --stderr julie_one_degree.err%jobid%
#OAR --project pr-data-ocean


cd /bettik/alberta/julia/one-degree
cp /bettik/alberta/git/myNumericalEarth/one-degree-global-ocean-sea-ice-simulation/run_on_dahu.jl .

julia run_on_dahu.jl
