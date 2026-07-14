import Pkg; Pkg.add("CFTime")

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using CairoMakie
using CFTime
using Dates
using Printf
using CUDA

arch = GPU()
Nx = 1440
Ny = 600
Nz = 40

depth = 6000meters
z = ExponentialDiscretization(Nz, -depth, 0, mutable=true)

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             z,
                             latitude  = (-75, 75),
                             longitude = (0, 360))

bottom_height = regrid_bathymetry(grid;
                                  minimum_depth = 10meters,
                                  interpolation_passes = 5,
                                  major_basins = 3)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

h = grid.immersed_boundary.bottom_height

fig, ax, hm = heatmap(h, colormap=:deep, colorrange=(-depth, 0))
Colorbar(fig[0, 1], hm, label="Bottom height (m)", vertical=false)
save("bathymetry.png", fig)

ocean = ocean_simulation(grid)

ocean.model

ENV["ECCO_USERNAME"] = "aureliealbert"
ENV["ECCO_WEBDAV_PASSWORD"] = "nVHMfH7fBea82wSaWhm"

set!(ocean.model, T=Metadatum(:temperature, dataset=ECCO4Monthly()),
                  S=Metadatum(:salinity, dataset=ECCO4Monthly()))

radiation = Radiation(arch)

atmosphere = JRA55PrescribedAtmosphere(arch; backend = JRA55NetCDFBackend(41),
                                       include_rivers_and_icebergs = false)

coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)

simulation = Simulation(coupled_model; Δt=25minutes, stop_time=60days)

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T

    Tmax = maximum(interior(T))
    Tmin = minimum(interior(T))

    umax = (maximum(abs, interior(u)),
            maximum(abs, interior(v)),
            maximum(abs, interior(w)))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg = @sprintf("Iter: %d, time: %s, Δt: %s", iteration(sim), prettytime(sim), prettytime(sim.Δt))
    msg *= @sprintf(", max|u|: (%.2e, %.2e, %.2e) m s⁻¹, extrema(T): (%.2f, %.2f) ᵒC, wall time: %s",
                    umax..., Tmax, Tmin, prettytime(step_time))

    @info msg

    wall_time[] = time_ns()
end

simulation.callbacks[:progress] = Callback(progress, TimeInterval(5days))

outputs = merge(ocean.model.tracers, ocean.model.velocities)
ocean.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                            schedule = TimeInterval(1days),
                                            including = [:grid],
                                            filename = "near_global_surface_fields",
                                            indices = (:, :, grid.Nz),
                                            with_halos = true,
                                            overwrite_existing = true,
                                            array_type = Array{Float32})

flush(stdout)
flush(stderr)

run!(simulation)


flush(stdout)
flush(stderr)

u = FieldTimeSeries("near_global_surface_fields.jld2", "u"; backend = OnDisk())
v = FieldTimeSeries("near_global_surface_fields.jld2", "v"; backend = OnDisk())
T = FieldTimeSeries("near_global_surface_fields.jld2", "T"; backend = OnDisk())
e = FieldTimeSeries("near_global_surface_fields.jld2", "e"; backend = OnDisk())

times = u.times
Nt = length(times)

n = Observable(Nt)

land = interior(T.grid.immersed_boundary.bottom_height) .>= 0

Tn = @lift begin
    Tn = interior(T[$n])
    Tn[land] .= NaN
    view(Tn, :, :, 1)
end

en = @lift begin
    en = interior(e[$n])
    en[land] .= NaN
    view(en, :, :, 1)
end

un = Field{Face, Center, Nothing}(u.grid)
vn = Field{Center, Face, Nothing}(v.grid)

s = @at (Center, Center, Nothing) sqrt(un^2 + vn^2) # compute √(u²+v²) and interpolate back to Center, Center
s = Field(s)

sn = @lift begin
    parent(un) .= parent(u[$n])
    parent(vn) .= parent(v[$n])
    compute!(s)
    sn = interior(s)
    sn[land] .= NaN
    view(sn, :, :, 1)
end

title = @lift string("Near-global 1/4 degree ocean simulation after ",
                     prettytime(times[$n] - times[1]))

λ, φ, _ = nodes(T) # T, e, and s all live on the same grid locations

fig = Figure(size = (1000, 1500))

axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
axT = Axis(fig[2, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
axe = Axis(fig[3, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

hm = heatmap!(axs, λ, φ, sn, colorrange = (0, 0.5), colormap = :deep, nan_color=:lightgray)
Colorbar(fig[1, 2], hm, label = "Surface Speed (m s⁻¹)")

hm = heatmap!(axT, λ, φ, Tn, colorrange = (-1, 30), colormap = :magma, nan_color=:lightgray)
Colorbar(fig[2, 2], hm, label = "Surface Temperature (ᵒC)")

hm = heatmap!(axe, λ, φ, en, colorrange = (0, 1e-3), colormap = :solar, nan_color=:lightgray)
Colorbar(fig[3, 2], hm, label = "Turbulent Kinetic Energy (m² s⁻²)")

Label(fig[0, :], title)

save("snapshot.png", fig)

CairoMakie.record(fig, "near_global_ocean_surface.mp4", 1:Nt, framerate = 8) do nn
    n[] = nn
end


