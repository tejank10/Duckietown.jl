mutable struct FixedSimParams <: RayTracer.FixedParams
    rng::MersenneTwister
    _map::Map
    max_steps::Int              # Maximum number of steps per episode
    draw_curve::Bool                # Flag to draw the road curve
    draw_bbox::Bool                 # Flag to draw bounding boxes
    domain_rand::Bool               # Flag to enable/disable domain randomization
    randomizer::Union{Randomizer, Nothing}
    randomization_settings::Union{Dict, Nothing}
    graphics::Bool
    frame_rate::Int                 # Frame rate to run at
    frame_skip::Int                 # Number of frames to skip per action
    delta_time::Float32
    camera_width::Int
    camera_height::Int
    robot_speed::Float32
    action_space::Box
    observation_space::Box
    reward_range::NTuple{2, Int}
    #window                          # Window for displaying the environment to humans
    #img_array::Array               # Array to render the image into (for observation rendering)
    accept_start_angle_deg::Float32    # allowed angle in lane for starting position
    full_transparency::Bool
    user_tile_start                 # Start tile
    distortion::Bool
    randomize_maps_on_reset::Bool
    camera_model::Nothing
    map_names::Union{Vector{String}, Nothing}
    undistort::Bool
    horizon_color::Vec3
    ground_color::Vec3
    wheel_dist::Float32
    cam_height::Float32
    cam_angle::Vector
    cam_fov_y::Float32
    cam_offset::Vector
end

function FixedSimParams(map_name::String=DEFAULT_MAP_NAME,
                        max_steps::Int=DEFAULT_MAX_STEPS;
                        draw_curve::Bool=false,
                        draw_bbox::Bool=false,
                        domain_rand::Bool=true,
                        frame_rate::Int=DEFAULT_FRAMERATE,
                        frame_skip::Int=DEFAULT_FRAME_SKIP,
                        camera_width::Int=DEFAULT_CAMERA_WIDTH,
                        camera_height::Int=DEFAULT_CAMERA_HEIGHT,
                        robot_speed::Float32=DEFAULT_ROBOT_SPEED,
                        accept_start_angle_deg::Real=DEFAULT_ACCEPT_START_ANGLE_DEG,
                        full_transparency::Bool=false,
                        user_tile_start=nothing,
                        seed=nothing,
                        distortion::Bool=false,
                        randomize_maps_on_reset::Bool=false)
    #=
    :param map_name:
    :param max_steps:
    :param draw_curve:
    :param draw_bbox:
    :param domain_rand: If true, applies domain randomization
    :param frame_rate:
    :param frame_skip:
    :param camera_width:
    :param camera_height:
    :param robot_speed:
    :param accept_start_angle_deg:
    :param full_transparency:   # If true, then we publish all transparency information
    :param user_tile_start: If None, sample randomly. Otherwise (i,j). Overrides map start tile
    :param seed:
    :param distortion: If true, distorts the image with fish-eye approximation
    :param randomize_maps_on_reset: If true, randomizes the map on reset (Slows down training)
    =#

    # first initialize the RNG
    rng = MersenneTwister(seed)

    _map = Map(map_name, domain_rand)

    randomizer, randomization_settings = nothing, nothing
    if domain_rand
        randomizer = Randomizer()
        randomization_settings = randomize(randomizer)
    end

    delta_time = 1f0 / frame_rate

    # Produce graphical output
    graphics = true

    # Two-tuple of wheel torques, each in the range [-1, 1]
    action_space = Box(-1, 1, (2,), Float32)

    # We observe an RGB image with pixels in [0, 255]
    # Note: the pixels are in UInt8 format because this is more compact
    # than Float32 if sent over the network or stored in a dataset
    observation_space = Box(0, 255, (camera_height, camera_width, 3), UInt8)

    reward_range = (-1000, 1000)

    # Distortion params, if so, load the library, only if not bbox mode
    distortion = distortion && !draw_bbox
    camera_model = nothing
    if !draw_bbox && distortion
        if distortion
            throw(error("Currently not supporting distortion!"))
            #include("distortion.jl")
            #camera_model = Distortion()
        end
    end

    map_names = nothing
    if randomize_maps_on_reset
        map_names = readdir("src/maps")
        map_names = map(mapfile->replace(mapfile, ".yaml"=>""), map_names)
    end

    # Used by the UndistortWrapper, always initialized to False
    undistort = false

    # Robot's current speed
    speed = 0f0

    horizon_color = BLUE_SKY_COLOR
    ground_color = GROUND_COLOR

    # Distance between the robot's wheels
    wheel_dist = WHEEL_DIST

    # Distance bewteen camera and ground
    cam_height = CAMERA_FLOOR_DIST

    # Angle at which the camera is rotated
    cam_angle = [CAMERA_ANGLE, 0, 0]

    # Field of view angle of the camera
    cam_fov_y = CAMERA_FOV_Y

    # Camera offset for use in free camera mode
    cam_offset = zeros(Float32, 3)

    FixedSimParams(rng, _map, max_steps, draw_curve, draw_bbox, domain_rand,
                    randomizer, randomization_settings, graphics, frame_rate,
                    frame_skip, delta_time, camera_width, camera_height,
                    robot_speed, action_space, observation_space, reward_range,
                    accept_start_angle_deg, full_transparency, user_tile_start,
                    distortion, randomize_maps_on_reset, camera_model,
                    map_names, undistort, horizon_color, ground_color, wheel_dist, cam_height,
                    cam_angle, cam_fov_y, cam_offset)
end

function reset!(fsp::FixedSimParams)
    # Step count since episode start
    if fsp.randomize_maps_on_reset
        map_name = rand(fsp.map_names)
        fsp._map = Map(map_name, fsp._map.map_file_path)
    end

    #TODO: Randomizer
    if fsp.domain_rand
        fsp.randomization_settings = randomize(fsp.randomizer)
    end

    # Horizon color
    # Note: we explicitly sample white and grey/black because
    # these colors are easily confused for road and lane markings
    if fsp.domain_rand
        horz_mode = fsp.randomization_settings["horz_mode"]
        if horz_mode == 0
            fsp.horizon_color = _perturb(fsp, BLUE_SKY_COLOR)
        elseif horz_mode == 1
            fsp.horizon_color = _perturb(fsp, WALL_COLOR)
        elseif horz_mode == 2
            fsp.horizon_color = _perturb(fsp, ones(Float32, 3)*0.15f0, 0.4f0)
        elseif horz_mode == 3
            fsp.horizon_color = _perturb(fsp, ones(Float32, 3)*0.9f0, 0.4f0)
        end
    end

    # Ground color
    fsp.ground_color = _perturb(fsp, GROUND_COLOR, 0.3f0)

    # Distance between the robot's wheels
    fsp.wheel_dist = _perturb(fsp, WHEEL_DIST)

    # Distance between camera and ground
    fsp.cam_height = _perturb(fsp, CAMERA_FLOOR_DIST, 0.08f0)

    # Angle at which the camera is rotated
    fsp.cam_angle = vcat(_perturb(fsp, CAMERA_ANGLE, 0.2f0), 0, 0)

    # Field of view angle of the camera
    fsp.cam_fov_y = _perturb(fsp, CAMERA_FOV_Y, 0.2f0)

    # Camera offset for use in free camera mode
    fsp.cam_offset = zeros(Float32, 3)

    # Create the vertex list for the ground/noise triangles
    # These are distractors, junk on the floor
    numTris = 12
    verts = []
    colors = []
    for _ in 0 : 3numTris
        p = [rand(fsp.rng, Uniform(-20f0, 20f0)),
             rand(fsp.rng, Uniform(-0.6f0, -0.3f0)),
             rand(fsp.rng, Uniform(-20f0, 20f0))]
        c = Float32(rand(fsp.rng, Uniform(0f0, 0.9f0)))
        c = _perturb(fsp, ones(Float32, 3)*c, 0.1f0)
        verts = vcat(verts, p)
        colors = vcat(colors, c)
    end

    #=
    import pyglet
    self.tri_vlist = pyglet.graphics.vertex_list(3 * numTris, ('v3f', verts), ('c3f', colors))
    =#

    # Randomize tile parameters
    for tile in _grid(fsp)._grid
        rng = fsp.domain_rand ? fsp.rng : nothing
        # Randomize the tile texture
        # comment till textures are implemented
        tile["texture"] = Graphics.get(tile["kind"], fsp.rng)

        # Random tile color multiplier
        tile["color"] = _perturb(fsp, Vec3([1f0]), 0.2f0)
    end

    # Randomize object parameters
    for obj in _objects(fsp)
        # Randomize the object color
        _set_color!(obj, _perturb(fsp, Vec3([1f0]), 0.3f0))

        # Randomize whether the object is visible or not
        if _optional(obj) && fsp.domain_rand
            _set_visible!(obj, rand(fsp.rng, 0:1) == 0)
        else
            _set_visible!(obj, true)
        end
    end

    # If the map specifies a starting tile
    if !isnothing(fsp.user_tile_start)
        #logger.info('using user tile start: %s' % self.user_tile_start)
        i, j = fsp.user_tile_start
        tile = _get_tile(_grid(fsp), i, j)
        if isnothing(tile)
            msg = "The tile specified does not exist."
            throw(error(msg))
        end
        #logger.debug('tile: %s' % tile)
    else
        if !isnothing(_start_tile(fsp))
            tile = _start_tile(fsp)
        else
            # Select a random drivable tile to start on
            tile_idx = rand(fsp.rng, 1:length(_drivable_tiles(fsp)))
            tile = _drivable_tiles(fsp)[tile_idx]
        end
    end

    propose_pos, propose_angle = nothing, nothing

    for iter in 1:MAX_SPAWN_ATTEMPTS
        i, j = tile["coords"]

        # Choose a random position on this tile
        x = rand(fsp.rng, Uniform(i-1f0, i)) * _road_tile_size(fsp)
        z = rand(fsp.rng, Uniform(j-1f0, j)) * _road_tile_size(fsp)
        propose_pos = Float32.([x, 0f0, z])

        # Choose a random direction
        propose_angle = Float32(rand(fsp.rng, Uniform(0f0, 2π)))

        # logger.debug('Sampled %s %s angle %s' % (propose_pos[0],
        #                                          propose_pos[1],
        #                                          np.rad2deg(propose_angle)))

        # If this is too close to an object or not a valid pose, retry
        inconvenient = _inconvenient_spawn(_objects(fsp), propose_pos)

        inconvenient && continue

        invalid = !_valid_pose(fsp, propose_pos, propose_angle, 1.3f0)
        invalid && continue

        # If the angle is too far away from the driving direction, retry
        lp = get_lane_pos2(fsp, propose_pos, propose_angle)

        !is_inlane(lp) && continue

        M = fsp.accept_start_angle_deg
        ok = -M < lp.angle_deg < +M
        if !ok
            if iter == MAX_SPAWN_ATTEMPTS
                msg = "Could not find a valid starting pose after $MAX_SPAWN_ATTEMPTS attempts"
                throw(error(msg))
            end
            continue
        end
        # Found a valid initial pose
        break
    end

    return propose_pos, propose_angle

end

function _perturb(fsp::FixedSimParams, val::Vec3, scale=0.1f0)
    val = [val.x[1], val.y[1], val.z[1]]
    val = _perturb(fsp, val, scale)
    return Vec3(val...)
end

function _perturb(fsp::FixedSimParams, val, scale=0.1f0)
    ##
    #Add noise to a value. This is used for domain randomization.
    ##
    #@assert 0f0 ≤ scale < 1f0

    !fsp.domain_rand && (return val)

    noise = Float32.(rand(fsp.rng, Uniform(1f0-scale, 1f0+scale), size(val)...))

    return val .* noise
end


function _inconvenient_spawn(objects::Vector, pos)
    ##
    #Check that agent spawn is not too close to any object
    ##

    cond(x) = norm(_pos(x) .- pos) <
               maximum(_max_coords(x)) * 0.5f0 * _scale(x) + MIN_SPAWN_OBJ_DIST
    arr = filter(x->_visible(x), objects)
    results = map(x->cond(x), arr)

    return any(results)
end

function _valid_pose(fsp::FixedSimParams, pos, angle, safety_factor=1f0)
    ##
    #    Check that the agent is in a valid pose
    #
    #    safety_factor = minimum distance
    ##

    # Compute the coordinates of the base of both wheels
    pos = _actual_center(pos, angle)
    f_vec = get_dir_vec(angle)
    r_vec = get_right_vec(angle)

    l_pos = pos .- (safety_factor * 0.5f0 * ROBOT_WIDTH) .* r_vec
    r_pos = pos .+ (safety_factor * 0.5f0 * ROBOT_WIDTH) .* r_vec
    f_pos = pos .+ (safety_factor * 0.5f0 * ROBOT_LENGTH) .* f_vec

    # Check that the center position and
    # both wheels are on drivable tiles and no collisions

    all_drivable = (_drivable_pos(_grid(fsp), pos) &&
                    _drivable_pos(_grid(fsp), l_pos) &&
                    _drivable_pos(_grid(fsp), r_pos) &&
                    _drivable_pos(_grid(fsp), f_pos))


    # Recompute the bounding boxes (BB) for the agent
    agent_corners = get_agent_corners(pos, angle)
    no_collision = !_collision(fsp, agent_corners)

    res = (no_collision && all_drivable)

    if !res
        #logger.debug(f'Invalid pose. Collision free: {no_collision} On drivable area: {all_drivable}')
        #logger.debug(f'safety_factor: {safety_factor}')
        #logger.debug(f'pos: {pos}')
        #logger.debug(f'l_pos: {l_pos}')
        #logger.debug(f'r_pos: {r_pos}')
        #logger.debug(f'f_pos: {f_pos}')
    end

    return res
end

Zygote.@nograd _valid_pose

function _collision(fsp::FixedSimParams, agent_corners)
    ##
    #Tensor-based OBB Collision detection
    ##

    # If there are no objects to collide against, stop
    length(_collidable_corners(fsp)) == 0 && (return false)

    # Generate the norms corresponding to each face of BB
    agent_norm = generate_norm(agent_corners)

    # Check collisions with static objects
    collision = intersects(
            agent_corners,
            _collidable_corners(fsp),
            agent_norm,
            _collidable_norms(fsp)
    )

    collision && (return true)

    # Check collisions with Dynamic Objects
    for obj in _objects(fsp)
        check_collision(obj, agent_corners, agent_norm) && (return true)
    end

    # No collision with any object
    return false
end

# FIXME: this does not follow the same signature as WorldOb
# NOTE: This is actually function meant for DuckiebotObj, defined here to break
#       cyclic dependency of types
function step!(db_obj::DuckiebotObj, fp::FixedSimParams, delta_time, closest_curve_point, objects)
    ##
    #Take a step, implemented as a PID controller
    ##

    # Find the curve point closest to the agent, and the tangent at that point
    closest_point, closest_tangent = closest_curve_point(fp, db_obj.wobj.pos, db_obj.wobj.angle)

    iterations = 0

    lookup_distance = db_obj.follow_dist
    curve_point = nothing
    while iterations < db_obj.max_iterations
        # Project a point ahead along the curve tangent,
        # then find the closest point to to that
        follow_point = closest_point .+ closest_tangent * lookup_distance
        curve_point, _ = closest_curve_point(fp, follow_point, db_obj.wobj.angle)

        # If we have a valid point on the curve, stop
        isnothing(curve_point) && break

        iterations += 1
        lookup_distance *= 0.5f0
    end

    # Compute a normalized vector to the curve point
    point_vec = curve_point .- db_obj.wobj.pos
    point_vec = point_vec ./ norm(point_vec)

    dot = dot(get_right_vec(db_obj, db_obj.wobj.angle), point_vec)
    steering = db_obj.gain * (-dot)

    _update_pos(db_obj, [db_obj.velocity, steering], delta_time)
end
