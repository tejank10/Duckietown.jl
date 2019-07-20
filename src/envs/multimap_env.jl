struct MultiMapEnv
    ##
    #Environment which samples from multiple environments, for
    #multi-taks learning
    ##
	env_list::Vector{String}
	cur_env_idx::Int
	cur_reward_sum::Float32
	cur_num_steps::Int
end

function MultiMapEnv(; kwargs...)
	env_list = Vector{String}()

    maps_dir = get_subdir_path("src/maps")

    #window = nothing

    # Try loading each of the available map files
    for map_file in readdir(maps_dir)
        map_name = split(map_file, '.')[1]

        # Do not load the regression test maps
        startswith(map_name, "regress") && continue

        env = DuckietownEnv(map_name=map_name, kwargs...)

        push!(env_list, env)
	end

    @assert !isempty(env.env_list)

    cur_env_idx = 1
    cur_reward_sum = 0
    cur_num_steps = 0

	MultiMapEnv(env_list, cur_env_idx, cur_reward_sum, cur_num_steps)
end

function reset!(mm_env::MultiMapEnv)
    #self.cur_env_idx = self.np_random.randint(0, len(self.env_list))
    mm_env.cur_env_idx = mm_env.cur_env_idx % length(mm_env.env_list) + 1

    env = mm_env.env_list[mm_env.cur_env_idx]
    return reset!(env)
end

function step!(mm_env::MultiMapEnv, action::Vector{Float32})
    env = mm_env.env_list[mm_env.cur_env_idx]

    obs, a, reward, done, info = step!(env, action)

    # Keep track of the total reward for this episode
    mm_env.cur_reward_sum += reward
    mm_env.cur_num_steps += 1

    # If the episode is done, sample a new environment
    if done
        mm_env.cur_reward_sum = 0
        mm_env.cur_num_steps = 0
	end

    return obs, a, reward, done, info
end

function render(mm_env::MultiMapEnv, mode::Val{:human}, close::Bool=false)
	#=
	env = mm_env.env_list[mm_env.cur_env_idx]

    # Make all environments use the same rendering window
    if isnothing(mm_env.window)
        ret = render(env, mode, close)
        mm_env.window = env.window
    else
        env.window = self.window
        ret = env.render(mode, close)
	end

    return ret
	=#
end

function close(mm_env::MultiMapEnv)
    for env in mm_env.env_list
        close(env)
	end

    mm_env.cur_env_idx = 1
    #mm_env.env_names = nothing
    mm_env.env_list = Vector{String}()
end

step_count(mm_env::MultiMapEnv) = mm_env.env_list[mm_env.cur_env_idx].step_count
