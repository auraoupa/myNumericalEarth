using Pkg; Pkg.activate(".")
import Pkg; Pkg.add("NumericalEarth")
import Pkg; Pkg.add("Oceananigans")
import Pkg; Pkg.add("CairoMakie")

using Oceananigans
Base.retry_load_extensions()
using CUDA
CUDA.set_runtime_version!(v"12.4"; local_toolkit=true)
CUDA.precompile_runtime()

