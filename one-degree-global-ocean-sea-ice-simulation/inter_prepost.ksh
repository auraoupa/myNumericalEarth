srun --pty --nodes=1 --cpus-per-task=1 --hint=nomultithread --partition=prepost --time=12:00:00 --account=cli@cpu bash

source ~/.bashrc
load_julia_gpu

cd /lustre/fswork/projects/rech/cli/rote001/DEV/git/myNumericalEarth/one-degree-global-ocean-sea-ice-simulation
export JULIA_DEPOT_PATH=/lustre/fsn1/projects/rech/cli/rote001/julia_prepost

julia
