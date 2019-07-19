module Duckietown

using YAML
using GymSpaces
using Distributions
using Random
using RayTracer
using Zygote
using LinearAlgebra

include("utils.jl")

include("graphics.jl")
using .Graphics

include("collision.jl")
include("objmesh.jl")
include("objects.jl")
include("map.jl")
include("randomization/randomizer.jl")
include("fixed_params.jl")
include("simulator.jl")

include("envs/duckietown_env.jl")
include("envs/multimap_env.jl")

export Simulator, step!, reset!, render_obs,
       DuckietownEnv, MultiMapEnv

end # module
