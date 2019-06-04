module Duckietown

using Gym
using YAML
using Gym: Space
using Distributions
using Random
using RayTracer
using Images

include("utils.jl")

include("graphics.jl")
using .Graphics

include("collision.jl")
include("objmesh.jl")
include("objects.jl")
include("map.jl")
include("randomization/randomizer.jl")
include("simulator.jl")


export Simulator, step!

end # module
