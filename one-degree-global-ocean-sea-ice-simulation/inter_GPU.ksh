srun --pty --ntasks=4 --hint=nomultithread  --gres=gpu:1 --partition=gpu_p13 --time=02:00:00 --account=cli@v100 bash

source ~/.bashrc
load_julia_gpu

cd /lustre/fswork/projects/rech/cli/rote001/DEV/git/myNumericalEarth/one-degree-global-ocean-sea-ice-simulation/GPU

export JULIA_DEPOT_PATH=/lustre/fswork/projects/rech/cli/rote001/DEV/julia_myNEGPU

julia
