using Duckietown, Flux, Zygote, Test

function get_map_names(maps_path::String="src/maps/")
    map_names = readdir(maps_path)
    return map_names ∪ ["custom_map"]
end

@testset "Differentiability" begin
    map_names = get_map_names()
    model = Chain(
           	Conv((3, 3), 3=>8, relu, pad = 1),  MaxPool((2, 2)),
           	Conv((3, 3), 8=>16, relu, pad = 1), MaxPool((2, 2)),
           	Conv((3, 3), 16=>32, relu, pad = 1),
           	x -> reshape(x, :, 1),
           	Dense((32 * 32 * 32), 64, relu),
           	Dense(64, 16, relu),
           	Dense(16, 2),
           	x -> reshape(x, 2))
    
    for map_name in map_names
    	sim = Simulator(map_name=map_name, camera_width=128, camera_height=128, raytrace=true, train=true)
    	o0 = render_obs(sim)
	gs = Zygote.gradient (params(model)) do
	         o1, a, r1, done, _ = step!(sim, model(o0))
	         o2, a, r2, done, _ = step!(sim, model(o1))
		 -(r1 + r2)
	     end

	grads = map(x->gs[x], params(model))

	@test !any(isnothing.(grads))
	#@test all(sum.(grads) .!= 0f0)
    end
end
