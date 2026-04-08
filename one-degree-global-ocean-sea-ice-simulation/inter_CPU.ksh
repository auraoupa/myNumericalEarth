srun --pty --ntasks=40 --cpus-per-task=1 --hint=nomultithread --partition=compil --time=02:00:00 --account=cli@cpu bash

source ~/.bashrc
load_julia

cd /lustre/fswork/projects/rech/cli/rote001/DEV/julia
julia