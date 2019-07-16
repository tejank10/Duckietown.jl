export DuckietownEnv, DuckietownLF, DuckietownNav

##
#Wrapper to control the simulator using velocity and steering angle
#instead of differential drive motor velocities
##

struct DuckietownEnv
    sim::Simulator
    gain::Float32
    trim::Float32
    radius::Float32
    k::Float32
    limit::Float32
end

function DuckietownEnv(; gain::Float32=1f0, trim::Float32=0f0, radius::Float32=0.0318f0,
                       k::Float32=27f0, limit::Float32=1f0, kwargs...)
    sim = Simulator(; kwargs...)
    #logger.info('using DuckietownEnv')
    return DuckietownEnv(sim, gain, trim, radius, k, limit)    
end

reset!(dt_env::DuckietownEnv) = reset!(dt_env.sim)

function step!(dt_env::DuckietownEnv, action::Vector{Float32})
     vel, angle = action

     # Distance between the wheels
     baseline = dt_env.sim.fixedparams.wheel_dist

     # assuming same motor constants k for both motors
     k_r = dt_env.k
     k_l = dt_env.k

     # adjusting k by gain and trim
     k_r_inv = (dt_env.gain + dt_env.trim) / k_r
     k_l_inv = (dt_env.gain - dt_env.trim) / k_l

     omega_r = (vel + 0.5f0 * angle * baseline) / dt_env.radius
     omega_l = (vel - 0.5f0 * angle * baseline) / dt_env.radius

     # conversion from motor rotation rate to duty cycle
     u_r = omega_r * k_r_inv
     u_l = omega_l * k_l_inv

     # limiting output to limit, which is 1.0 for the duckiebot
     u_r_limited = max(min(u_r, dt_env.limit), -dt_env.limit)
     u_l_limited = max(min(u_l, dt_env.limit), -dt_env.limit)

     vels = [u_l_limited, u_r_limited]

     obs, v, reward, done, info = step!(dt_env.sim, vels)
     mine = Dict{String,Any}()
     mine["k"] = dt_env.k
     mine["gain"] = dt_env.gain
     mine["train"] = dt_env.trim
     mine["radius"] = dt_env.radius
     mine["omega_r"] = omega_r
     mine["omega_l"] = omega_l
     info["DuckietownEnv"] = mine
     return obs, action, reward, done, info
end

##
#Environment for the Duckietown lane following task with
#and without obstacles (LF and LFV tasks)
##
struct DuckietownLF
    dt_env::DuckietownEnv
end

DuckietownLF(; kwargs...) = DuckietownLF(DuckietownEnv(; kwargs...))

reset!(dt_lf::DuckietownLF) = reset!(dt_lf.dt_env)

step!(dt_lf::DuckietownLF, action::Vector{Float32}) = step!(dt_lf.dt_env, action)

##
#Environment for the Duckietown navigation task (NAV)
##
struct DuckietownNav
    dt_env::DuckietownEnv
    goal_tile::Union{Nothing, Dict{String,Any}
end

function DuckietownNav(; kwargs...)
    goal_tile = nothing
    DuckietownNav(DuckietownEnv(; kwargs...), goal_tile)
end

function reset!(dt_nav::DuckietownNav)
    sim = dt_nav.dt_env.sim
    obs = reset!(sim)

    # Find the tile the agent starts on
    start_tile_pos = get_grid_coords(sim.cur_pos)
    start_tile = self._get_tile(start_tile_pos...)

    # Select a random goal tile to navigate to
    @assert length(_drivable_tiles(sim)) > 1
    while true
        tile_idx = rand(sim.fixedparams.rng, 1:length(_drivable_tiles(sim)))
        self.goal_tile = _drivable_tiles(sim)[tile_idx]
        dt_nav.goal_tile != sim.fixedparams._map.start_tile && break
    end 

    return obs
end

function step!(dt_nav::DuckietownNav, action::Vector{Float32})
    obs, a, reward, done, info = step!(dt_env, action)

    info["goal_tile"] = dt_nav.goal_tile

    # TODO: add term to reward based on distance to goal?

    cur_tile_coords = get_grid_coords(dt_nav.dt_env.sim.cur_pos)
    cur_tile = _get_tile(cur_tile_coords)

    if cur_tile == dt_nav.goal_tile
        done = true
        reward = 1000f0
    end

    return obs, a, reward, done, info
end
