# Duckietown.jl
Differentiable Duckietown

## Installation
This package is currently not registered. Additionally, it makes use of two more unregistered packages: [GymSpaces.jl](https://github.com/kraftpunk97/GymSpaces.jl) and [RayTracer.jl](https://github.com/avik-pal/RayTracer.jl).Hence, we first need to install them in order to install Duckietown smoothly.
```julia
julia> ]add https://github.com/kraftpunk97/GymSpaces.jl
julia> ]add https://github.com/avik-pal/RayTracer.jl
julia> ]add https://github.com/tejank10/Duckietown.jl
```

## Current State
Texturers are being supported now. To use them, checkout the [`texture`](https://github.com/tejank10/Duckietown.jl/tree/texture) branch.  
So far, we have tested rendering on following maps:  
* [x] straight_road
* [x] small_loop
* [x] 4way
* [x] loop_empty
* [ ] loop_dyn_duckiebots
* [ ] loop_obstacles
* [ ] loop_pedestrians
* [ ] regress_4way_adam
* [ ] regress_4way_drivable
* [ ] small_loop_cw
* [ ] udem1
* [ ] zigzag_dists


## To Do
* [x] Textures
* [ ] Test out all the maps
* [ ] MultiMap mode
* [ ] Recording episode/animation

