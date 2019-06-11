using RayTracer: FixedParams

struct DoneRewardInfo
    done::Bool
    done_why::String
    reward::Float32
    done_code::String
end

# Randomization code

# Rendering window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Camera image size
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480

# Blue sky horizon color
BLUE_SKY_COLOR = Vec3([0.45f0], [0.82f0], [1f0])

# Color meant to approximate interior walls
WALL_COLOR = Vec3([0.64f0], [0.71f0], [0.28f0])

# Ground/floor color
GROUND_COLOR = Vec3([0.15f0])

# Angle at which the camera is pitched downwards
CAMERA_ANGLE = 20f0

# Camera field of view angle in the Y direction
# Note: robot uses Raspberri Pi camera module V1.3
# https://www.raspberrypi.org/documentation/hardware/camera/README.md
CAMERA_FOV_Y = 42f0

# Distance from camera to floor (10.8cm)
CAMERA_FLOOR_DIST = 0.108f0

# Forward distance between the camera (at the front)
# and the center of rotation (6.6cm)
CAMERA_FORWARD_DIST = 0.066f0

# Distance (diameter) between the center of the robot wheels (10.2cm)
WHEEL_DIST = 0.102f0

# Total robot width at wheel base, used for collision detection
# Note: the actual robot width is 13cm, but we add a litte bit of buffer
#       to faciliate sim-to-real transfer.
ROBOT_WIDTH = 0.13f0 + 0.02f0

# Total robot length
# Note: the center of rotation (between the wheels) is not at the
#       geometric center see CAMERA_FORWARD_DIST
ROBOT_LENGTH = 0.18f0

# Height of the robot, used for scaling
ROBOT_HEIGHT = 0.12f0

# Safety radius multiplier
SAFETY_RAD_MULT = 1.8f0

# Robot safety circle radius
AGENT_SAFETY_RAD = (max(ROBOT_LENGTH, ROBOT_WIDTH) / 2) * SAFETY_RAD_MULT

# Minimum distance spawn position needs to be from all objects
MIN_SPAWN_OBJ_DIST = 0.25f0

# Road tile dimensions (2ft x 2ft, 61cm wide)
# self.road_tile_size = 0.61

# Maximum forward robot speed in meters/second
DEFAULT_ROBOT_SPEED = 1.2f0
# approx 2 tiles/second

DEFAULT_FRAMERATE = 30

DEFAULT_MAX_STEPS = 1500

DEFAULT_MAP_NAME = "udem1"

DEFAULT_FRAME_SKIP = 1

DEFAULT_ACCEPT_START_ANGLE_DEG = 60

REWARD_INVALID_POSE = -1000f0

MAX_SPAWN_ATTEMPTS = 5000

struct LanePosition
    dist
    dot_dir
    angle_deg
    angle_rad
end

function as_json_dict(lp::LanePosition)
    ### Serialization-friendly format. ###
    return Dict([:dist=>lp.dist,
                 :dot_dir=>lp.dot_dir,
                 :angle_deg=>lp.angle_deg,
                 :angle_rad=>lp.angle_rad])
end


# Raised when the Duckiebot is not in a lane. #
struct NotInLane <: Exception end


mutable struct Simulator
    ##
    #Simple road simulator to test RL training.
    #Draws a road with turns using RayTracer, and simulates
    #basic differential-drive dynamics.
    ##
    last_action::Vector{Float32}
    wheelVels::Vector{Float32}
    speed::Float32
    cur_pos::Union{Vector{Float32}, Nothing}
    cur_angle::Union{Float32, Nothing}
    step_count::Int
    timestamp::Float32
    fixedparams::FixedSimParams
end

function Simulator(
        map_name::String=DEFAULT_MAP_NAME,
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
        randomize_maps_on_reset::Bool=false
)
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

    randomizer = domain_rand ? Randomizer() : nothing
    randomization_settings = nothing

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

    last_action = zeros(Float32, 2)
    wheelVels = zeros(Float32, 2)

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

    step_count = 0
    timestamp = 0f0

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

    fp = FixedSimParams(rng, _map, max_steps, draw_curve, draw_bbox, domain_rand,
                        randomizer, randomization_settings, graphics, frame_rate,
                        frame_skip, delta_time, camera_width, camera_height,
                        robot_speed, action_space, observation_space, reward_range,
                        accept_start_angle_deg, full_transparency, user_tile_start,
                        distortion, randomize_maps_on_reset, camera_model,
                        map_names, undistort, horizon_color, ground_color, wheel_dist, cam_height,
                        cam_angle, cam_fov_y, cam_offset)

    cur_pos, cur_angle = reset!(fp)

    # Robot's current speed
    speed = 0f0
    sim = Simulator(last_action, wheelVels, speed, cur_pos, cur_angle,
                    step_count, timestamp, fp)

    # Initialize the state
    return sim
end

function Base.display(io::IO, sim::Simulator)
    print("Simulator($(sim.fixedparams._map.map_name))")
end

function reset!(sim::Simulator)
    ##
    #Reset the simulation at the start of a new episode
    #This also randomizes many environment parameters (domain randomization)
    ##

    sim.cur_pos, sim.cur_angle = reset!(sim.fixedparams)

    # Robot's current speed
    sim.speed = 0f0

    #logger.info('Starting at %s %s' % (sim.cur_pos, sim.cur_angle))

    # Generate the first camera image
    obs = render_obs(sim)
end

_objects(grid::Grid) = grid.obj_data.objects
_objects(map::Map) = _objects(map._grid)
_objects(fp::FixedSimParams) = _objects(fp._map)
_objects(sim::Simulator) = _objects(sim.fixedparams)

_grid(map::Map) = map._grid
_grid(fp::FixedSimParams) = _grid(fp._map)
_grid(sim::Simulator) = _grid(sim.fixedparams)

_drivable_tiles(map::Map) = map._grid.drivable_tiles
_drivable_tiles(fp::FixedSimParams) = _drivable_tiles(fp._map)
_drivable_tiles(sim::Simulator) = _drivable_tiles(sim.fixedparams)

_start_tile(map::Map) = map._grid.start_tile
_start_tile(fp::FixedSimParams) = _start_tile(fp._map)
_start_tile(sim::Simulator) = _start_tile(sim.fixedparams)

_road_tile_size(grid::Grid) = grid.road_tile_size
_road_tile_size(map::Map) = _road_tile_size(map._grid)
_road_tile_size(fp::FixedSimParams) = _road_tile_size(fp._map)
_road_tile_size(sim::Simulator) = _road_tile_size(sim.fixedparams)

_collidable_corners(grid::Grid) = grid.obj_data.collidable_corners
_collidable_corners(map::Map) = _collidable_corners(map._grid)
_collidable_corners(fp::FixedSimParams) = _collidable_corners(fp._map)
_collidable_corners(sim::Simulator) = _collidable_corners(sim.fixedparams)


_collidable_norms(grid::Grid) = grid.obj_data.collidable_norms
_collidable_norms(map::Map) = _collidable_norms(map._grid)
_collidable_norms(fp::FixedSimParams) = _collidable_norms(fp._map)
_collidable_norms(sim::Simulator) = _collidable_norms(sim.fixedparams)

_collidable_centers(grid::Grid) = grid.obj_data.collidable_corners
_collidable_centers(map::Map) = _collidable_centers(map._grid)
_collidable_centers(fp::FixedSimParams) = _collidable_centers(fp._map)
_collidable_centers(sim::Simulator) = _collidable_centers(sim.fixedparams)

_collidable_safety_radii(fp::FixedSimParams) = fp.collidable_safety_radii
_collidable_safety_radii(sim::Simulator) = _collidable_safety_radii(sim.fixedparams)

_delta_time(fp::FixedSimParams) = fp.delta_time
_delta_time(sim::Simulator) = _delta_time(sim.fixedparams)

_robot_speed(fp::FixedSimParams) = fp.robot_speed
_robot_speed(sim::Simulator) = _robot_speed(sim.fixedparams)

_max_steps(fp::FixedSimParams) = fp.max_steps
_max_steps(sim::Simulator) = _max_steps(sim.fixedparams)

_wheel_dist(fp::FixedSimParams) = fp.wheel_dist
_wheel_dist(sim::Simulator) = _wheel_dist(sim.fixedparams)

_full_transparency(fp::FixedSimParams) = fp.full_transparency
_full_transparency(sim::Simulator) = _full_transparency(sim.fixedparams)


function _init_vlists(road_tile_size)
    # Create the vertex list for our road quad
    # Note: the vertices are centered around the origin so we can easily
    # rotate the tiles about their center
    half_size = road_tile_size / 2f0
    verts = [
        -half_size 0f0 -half_size;
         half_size 0f0 -half_size;
         half_size 0f0  half_size;
        -half_size 0f0  half_size
    ]
    texCoords = [
        1f0 0f0;
        0f0 0f0;
        0f0 1f0;
        1f0 1f0
    ]

    road_vlist = verts
    road_tlist = texCoords

    # Create the vertex list for the ground quad
    verts = [
        -1 -0.8f0  1;
        -1 -0.8f0 -1;
         1 -0.8f0 -1;
         1 -0.8f0  1
    ]

    ground_vlist = verts

    return road_vlist, road_tlist, ground_vlist
end

function closest_curve_point(fp::FixedSimParams, pos, angle)
    ##
    #    Get the closest point on the curve to a given point
    #    Also returns the tangent at that point.
    #
    #    Returns nothing, nothing if not in a lane.
    ##

    i, j = get_grid_coords(_road_tile_size(fp), pos)
    tile = _get_tile(_grid(fp), i, j)

    (isnothing(tile) || !tile["drivable"]) && return nothing, nothing

    # Find curve with largest dotproduct with heading
    curves = _get_tile(_grid(fp), i, j)["curves"]
    curve_headings = curves[end, :, :] .- curves[1, :, :]
    curve_headings = curve_headings / norm(curve_headings)
    dir_vec = get_dir_vec(angle)

    dot_prods = [dot(curve_headings[:, i], dir_vec) for i in 1:size(curve_headings, 2)]

    # Closest curve = one with largest dotprod
    max_idx = argmax(dot_prods)
    cps = curves[:, :, max_idx]

    # Find closest point and tangent to this curve
    t = bezier_closest(cps, pos)
    point = bezier_point(cps, t)
    tangent = bezier_tangent(cps, t)

    return point, tangent
end

function get_lane_pos2(fp::FixedSimParams, pos, angle)
    ##
    #Get the position of the agent relative to the center of the right lane
    #
    #Raises NotInLane if the Duckiebot is not in a lane.
    ##

    # Get the closest point along the right lane's Bezier curve,
    # and the tangent at that point
    point, tangent = closest_curve_point(fp, pos, angle)
    if isnothing(point)
        msg = "Point not in lane: $pos"
        throw(NotInLane(msg))
    end

    @assert !isnothing(point)

    # Compute the alignment of the agent direction with the curve tangent
    dirVec = get_dir_vec(angle)
    dotDir = dot(dirVec, tangent)
    dotDir = max(-1f0, min(1f0, dotDir))

    # Compute the signed distance to the curve
    # Right of the curve is negative, left is positive
    posVec = pos .- point
    upVec = [0f0, 1f0, 0f0]
    rightVec = cross(tangent, upVec)
    signedDist = dot(posVec, rightVec)

    # Compute the signed angle between the direction and curve tangent
    # Right of the tangent is negative, left is positive
    angle_rad = acos(dotDir)

    if dot(dirVec, rightVec) < 0
        angle_rad *= -1f0
    end

    angle_deg = rad2deg(angle_rad)
    # return signedDist, dotDir, angle_deg

    return LanePosition(signedDist, dotDir, angle_deg, angle_rad)
end


function _drivable_pos(grid::Grid, pos)
    ##
    #Check that the given (x,y,z) position is on a drivable tile
    ##

    coords = get_grid_coords(grid.road_tile_size, pos)
    tile = _get_tile(grid, coords...)
    if isnothing(tile)
        msg = "No tile found at $pos $coords"
        #logger.debug(msg)
        return false
    end

    if !tile["drivable"]
        msg = "$pos corresponds to tile at $coords which is not drivable: $tile"
        #logger.debug(msg)
        return false
    end

    return true
end

function Base.show(io::IO, sim::Simulator)
    map_name = sim.fixedparams._map.map_name
    print("Simulator($map_name)")
end

function close(sim::Simulator) end

function _collidable_object(grid, grid_width, grid_height, obj_corners, obj_norm,
                            possible_tiles, road_tile_size)
    ##
    #A function to check if an object intersects with any
    #drivable tiles, which would mean our agent could run into them.
    #Helps optimize collision checking with agent during runtime
    ##

    size(possible_tiles) == (0,) && (return false)
    drivable_tiles = []
    for c in possible_tiles
        tile = _get_tile(grid, c..., grid_width, grid_height)
        if !ismissing(tile) && tile["drivable"]
            push!(drivable_tiles, (c...))
        end
    end
    isempty(drivable_tiles) && (return false)

    # Tiles are axis aligned, so add normal vectors in bulk
    tile_norms = repeat([1f0 0f0; 0f0 1f0],  length(drivable_tiles))

    # Find the corners for each candidate tile
    drivable_tiles = [permutedims(
        tile_corners(
                _get_tile(grid, pt..., grid_width, grid_height)["coords"],
                road_tile_size
        )) for pt in drivable_tiles
    ]

    # Stack doesn't do anything if there's only one object,
    # So we add an extra dimension to avoid shape errors later
    if ndims(tile_norms) == 2
        tile_norms = [tile_norms]
    end
    # Only add it if one of the vertices is on a drivable tile
    return intersects(obj_corners, drivable_tiles, obj_norm, tile_norms)
end

function get_grid_coords(road_tile_size, abs_pos)
    ##
    #Compute the tile indices (i,j) for a given (x,_,z) world position
    #
    #x-axis maps to increasing i indices
    #z-axis maps to increasing j indices
    #
    #Note: may return coordinates outside of the grid if the
    #position entered is outside of the grid.
    ##

    x, _, z = abs_pos
    i = floor(x / road_tile_size)
    j = floor(z / road_tile_size)

    return Int.((i, j)) .+ 1
end

get_dir_vec(sim::Simulator) = get_dir_vec(sim.cur_angle)
get_dir_vec(sim::Simulator, ::Nothing) = get_dir_vec(sim.cur_angle)

function get_dir_vec(angle)
    ##
    #Vector pointing in the direction the agent is looking
    ##
    x = cos(angle)
    z = -sin(angle)
    return [x, 0f0, z]
end

get_right_vec(sim::Simulator) = get_right_vec(sim.cur_angle)
get_right_vec(sim::Simulator, ::Nothing) = get_right_vec(sim.cur_angle)

function get_right_vec(angle)
    ##
    #Vector pointing to the right of the agent
    ##

    x = sin(angle)
    z = cos(angle)
    return [x, 0f0, z]
end

function _proximity_penalty2(sim::Simulator, pos, angle)
    ##
    #Calculates a 'safe driving penalty' (used as negative rew.)
    #as described in Issue #24 of gym-duckietown
    #
    #Describes the amount of overlap between the "safety circles" (circles
    #that extend further out than BBoxes, giving an earlier collision 'signal'
    #The number is max(0, prox.penalty), where a lower (more negative) penalty
    #means that more of the circles are overlapping
    ##

    pos = _actual_center(pos, angle)
    if length(_collidable_centers(sim)) == 0
        static_dist = 0
    # Find safety penalty w.r.t static obstacles
    else
        d = norm(_collidable_centers(sim) .- pos, dims=2)

        if !safety_circle_intersection(d, AGENT_SAFETY_RAD, _collidable_safety_radii(sim))
            static_dist = 0
        else
            static_dist = safety_circle_overlap(d, AGENT_SAFETY_RAD, _collidable_safety_radii(sim))
        end
    end
    total_safety_pen = static_dist
    for obj in _objects(sim)
        # Find safety penalty w.r.t dynamic obstacles
        total_safety_pen = total_safety_pen + proximity(obj, pos, AGENT_SAFETY_RAD)
    end
    return total_safety_pen
end


update_physics(sim::Simulator, action) = update_physics(sim, action, _delta_time(sim))
update_physics(sim::Simulator, action, ::Nothing) = update_physics(sim, action, _delta_time(sim))

function update_physics(sim::Simulator, action, delta_time)
    sim.wheelVels = action * _robot_speed(sim)
    prev_pos = sim.cur_pos

    # Update the robot's position
    sim.cur_pos, sim.cur_angle = _update_pos(sim.cur_pos,
                                             sim.cur_angle,
                                             _wheel_dist(sim),
                                             sim.wheelVels,
                                             delta_time)
    sim.step_count += 1
    sim.timestamp += delta_time

    sim.last_action = action

    # Compute the robot's speed
    delta_pos = sim.cur_pos .- prev_pos
    sim.speed = norm(delta_pos) / delta_time

    # Update world objects
    objs = _objects(sim)
    for obj in objs
        if !obj.static && obj.kind == "duckiebot"
            obj_i, obj_j = get_grid_coords(_road_tile_size(sim), obj.pos)
            cond(o) = get_grid_coords(_road_tile_size(sim), o.pos) == (obj_i, obj_j) && o != obj
            same_tile_obj = filter(cond, objs)
            step!(obj, delta_time, sim.fixedparams, closest_curve_point, same_tile_obj)
        else
            step!(obj, delta_time)
        end
    end
end

function get_agent_info(sim::Simulator)
    info = Dict()
    pos = sim.cur_pos
    angle = sim.cur_angle
    # Get the position relative to the right lane tangent

    info["action"] = vcat(sim.last_action)
    if _full_transparency(sim)
        #             info['desc'] = """
        #
        # cur_pos, cur_angle ::  simulator frame (non cartesian)
        #
        # egovehicle_pose_cartesian :: cartesian frame
        #
        #     the map goes from (0,0) to (grid_height, grid_width)*sim.road_tile_size
        #
        # """
        try
            lp = get_lane_pos2(sim.fixedparams, pos, angle)
            info["lane_position"] = as_json_dict(lp)
        catch y end

        info["robot_speed"] = sim.speed
        info["proximity_penalty"] = _proximity_penalty2(sim, pos, angle)
        info["cur_pos"] = float.(pos)
        info["cur_angle"] = float(angle)
        info["wheel_velocities"] = [sim.wheelVels...]

        # put in cartesian coordinates
        # (0,0 is bottom left)
        # q = self.cartesian_from_weird(self.cur_pos, self.)
        # info['cur_pos_cartesian'] = [float(p[0]), float(p[1])]
        # info['egovehicle_pose_cartesian'] = {'~SE2Transform': {'p': [float(p[0]), float(p[1])],
        #                                                        'theta': angle}}

        info["timestamp"] = sim.timestamp
        info["tile_coords"] = [get_grid_coords(_road_tile_size(sim), pos)]
        # info['map_data'] = self.map_data
    end
    misc = Dict()
    misc["Simulator"] = info
    return misc
end

#=
function cartesian_from_weird(sim::Simulator, pos, angle)
    gx, gy, gz = pos
    grid_height = self.grid_height
    tile_size = sim.road_tile_size

    # this was before but obviously doesn't work for grid_height = 1
    # cp = [gx, (grid_height - 1) * tile_size - gz]
    cp = [gx, grid_height * tile_size - gz]

    #TODO
    return geometry.SE2_from_translation_angle(cp, angle)
end

function weird_from_cartesian(self, q: np.ndarray)
    #TODO
    cp, angle = geometry.translation_angle_from_SE2(q)

    gx = cp[1]
    gy = 0
    # cp[1] = (grid_height - 1) * tile_size - gz
    grid_height = sim.grid_height
    tile_size = sim.road_tile_size
    # this was before but obviously doesn't work for grid_height = 1
    # gz = (grid_height - 1) * tile_size - cp[1]
    gz = grid_height * tile_size - cp[2]
    return [gx, gy, gz], angle
end
=#
function compute_reward(sim::Simulator, pos, angle, speed)
    # Compute the collision avoidance penalty
    col_penalty = _proximity_penalty2(sim, pos, angle)
    reward = 0f0
    lp = nothing
    # Get the position relative to the right lane tangent
    try
        lp = get_lane_pos2(sim.fixedparams, pos, angle)
    catch y
        if isa(y, NotInLane)
            reward = 40f0 * col_penalty
            return reward
        end
    end
    # Compute the reward
    reward = (
            speed * lp.dot_dir -
            10f0 * abs(lp.dist) +
            40f0 * col_penalty
    )
    return reward
end

function step!(sim::Simulator, action::Vector{Float32})
    action = clamp.(action, -1f0, 1f0)

    for _ in 1:sim.fixedparams.frame_skip
        update_physics(sim, action)
    end

    # Generate the current camera image
    s = render_obs(sim)
    misc = get_agent_info(sim)

    d = _compute_done_reward(sim)
    misc["Simulator"]["msg"] = d.done_why
    return s, action, d.reward, d.done, misc
end

function _compute_done_reward(sim::Simulator)
    # If the agent is not in a valid pose (on drivable tiles)
    if !_valid_pose(sim.fixedparams, sim.cur_pos, sim.cur_angle)
        msg = "Stopping the simulator because we are at an invalid pose."
        #logger.info(msg)
        reward = REWARD_INVALID_POSE
        done_code = "invalid-pose"
        done = true
    # If the maximum time step count is reached
    elseif sim.step_count ≥ _max_steps(sim)
        msg = "Stopping the simulator because we reached max_steps = $(_max_steps(sim))"
        #logger.info(msg)
        done = true
        reward = 0f0
        done_code = "max-steps-reached"
    else
        done = false
        reward = compute_reward(sim, sim.cur_pos, sim.cur_angle, sim.speed)
        msg = ""
        done_code = "in-progress"
    end
    return DoneRewardInfo(done, msg, reward, done_code)
end

function _render_img(fp::FixedSimParams, cur_pos, cur_angle, top_down=true)
    ##
    #Render an image of the environment into a frame buffer
    #Produce a numpy RGB array image as output
    ##

    !fp.graphics && return
    scene = []
    #=
    # Switch to the default context
    # This is necessary on Linux nvidia drivers
    # pyglet.gl._shadow_window.switch_to()
    self.shadow_window.switch_to()

    from pyglet import gl
    # Bind the multisampled frame buffer
    gl.glEnable(gl.GL_MULTISAMPLE)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, multi_fbo)
    gl.glViewport(0, 0, width, height)

    # Clear the color and depth buffers

    c0, c1, c2 = self.horizon_color
    gl.glClearColor(c0, c1, c2, 1.0)
    gl.glClearDepth(1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    # Set the projection matrix
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.gluPerspective(
            self.cam_fov_y,
            width / float(height),
            0.04,
            100.0
    )

    # Set modelview matrix
    # Note: we add a bit of noise to the camera position for data augmentation
    =#
    #logger.info('Pos: %s angle %s' % (self.cur_pos, self.cur_angle))
    pos, angle = cur_pos, cur_angle
    if fp.domain_rand
        pos = pos .+ fp.randomization_settings["camera_noise"]
    end

    x, y, z = pos .+ fp.cam_offset
    dx, dy, dz = get_dir_vec(angle)

    #trans_mat = Matrix{Float32}(I, 4, 4)
    if fp.draw_bbox
        y += 0.8f0
    #    trans_mat = rotate_mat(90f0, 1, 0, 0)
    elseif !top_down
        y += fp.cam_height
    #    trans_mat = rotate_mat(sim.cam_angle[1], (1, 0, 0))
    #    trans_mat = rotate_mat(sim.cam_angle[2], (0, 1, 0)) * trans_mat
    #    trans_mat = rotate_mat(sim.cam_angle[3], (0, 0, 1)) * trans_mat
    #    trans_mat = translation_mat([0f0, 0f0, _perturb(sim, CAMERA_FORWARD_DIST)]) * trans_mat
    end

    cam = nothing
    cam_width, cam_height = fp.camera_width, fp.camera_height
    grid_width, grid_height = _grid(fp).grid_width, _grid(fp).grid_height

    if top_down
        x = (grid_width * _road_tile_size(fp)) / 2f0
        y = 5f0
        z = (grid_height * _road_tile_size(fp)) / 2f0

        eye = Vec3([x], [y], [z])
        target = Vec3([x], [0f0], [z])
        vup = Vec3([0f0], [0f0], [-1f0])
        cam = Camera(eye, target, vup, fp.cam_fov_y, 1f0, cam_width, cam_height)
    else
        eye = Vec3([x], [y], [z])
        target = Vec3([x+dx], [y+dy], [z+dz])
        vup = Vec3([0f0], [1f0], [0f0])
        cam = Camera(eye, target, vup, fp.cam_fov_y, 1f0, cam_width, cam_height)
    end


    # Draw the ground quad
    #gl.glDisable(gl.GL_TEXTURE_2D)
    #gl.glColor3f(*sim.ground_color)
    #gl.glPushMatrix()
    trans_mat = scale_mat([50f0, 1f0, 50f0])
    ground_vlist = transform_mat(_grid(fp).ground_vlist, trans_mat)
    ground_scene = triangulate_faces(ground_vlist, fp.ground_color)
    scene =  vcat(scene, ground_scene)
    # TODO: triangulate this ground vlist and put in scene
    #gl.glPopMatrix()

    # Draw the ground/noise triangles
    #self.tri_vlist.draw(gl.GL_TRIANGLES)

    # Draw the road quads
    #gl.glEnable(gl.GL_TEXTURE_2D)
    #gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    #gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    # For each grid tile
    for j in 1:grid_height
        for i in 1:grid_width
            # Get the tile type and angle
            tile = _get_tile(_grid(fp), i, j)

            isnothing(tile) && continue

            # kind = tile['kind']
            angle = tile["angle"]
            color = tile["color"]
            #texture = tile["texture"]

            pos = [(i-0.5f0), 0f0, (j-0.5f0)] * _road_tile_size(fp)
            #gl.glPushMatrix()
            trans_mat = translation_mat(pos)
            trans_mat = rotate_mat(angle * 90f0) * trans_mat

            # Bind the appropriate texture
            #texture.bind()
            road_vlist = _grid(fp).road_vlist
            road_vlist = transform_mat(road_vlist, trans_mat)
            scene = vcat(scene, triangulate_faces(road_vlist, color))
            #gl.glPopMatrix()
            if tile["drivable"] && fp.draw_curve
                # Find curve with largest dotproduct with heading
                curves = _get_tile(_grid(fp), i, j)["curves"]
                curve_headings = curves[end, :, :] .- curves[1, :, :]
                curve_headings = curve_headings / norm(curve_headings)
                dirVec = get_dir_vec(angle)
                dot_prods = map(i->dot(curve_headings[:, i], dirVec), 1:size(curve_heading, 2))

                # Current ("closest") curve drawn in Red
                pts = curves[:, :, argmax(dot_prods)]
                bezier_draw(pts, 20, true)

                pts = _get_curve(_grid(fp), i, j)
                for (idx, pt) in enumerate(pts)
                    # Don't draw current curve in blue
                    idx == argmax(dot_prods) && continue
                    bezier_draw(pt, 20)
                end
            end
        end
    end

    # For each object
    scene = vcat(scene, map(obj->render(obj, fp.draw_bbox), _objects(fp))...)

    # Draw the agent's own bounding box
    if fp.draw_bbox
        #corners = get_agent_corners(pos, angle)
        #gl.glColor3f(1, 0, 0)
        #gl.glBegin(gl.GL_LINE_LOOP)
        #gl.glVertex3f(corners[0, 0], 0.01, corners[0, 1])
        #gl.glVertex3f(corners[1, 0], 0.01, corners[1, 1])
        #gl.glVertex3f(corners[2, 0], 0.01, corners[2, 1])
        #gl.glVertex3f(corners[3, 0], 0.01, corners[3, 1])
        #gl.glEnd()
    end

    if top_down
        trans_mat = translation_mat(pos...)
        trans_mat = scale_mat(1f0) * trans_mat
        trans_mat = rotate_mat(rad2deg(sim.cur_angle)) * trans_mat
        # glColor3f(*self.color)
        #scene = vcat(scene, render(sim.fixedparams.mesh))
        #gl.glPopMatrix()
    end

    # Resolve the multisampled frame buffer into the final frame buffer
    #gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, multi_fbo)
    #gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, final_fbo)
    #gl.glBlitFramebuffer(
    #        0, 0,
    #        width, height,
    #        0, 0,
    #        width, height,
    #        gl.GL_COLOR_BUFFER_BIT,
    #        gl.GL_LINEAR
    #)

    # Copy the frame buffer contents into a numpy array
    # Note: glReadPixels reads starting from the lower left corner
    #gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, final_fbo)
    #gl.glReadPixels(
    #        0,
    #        0,
    #        width,
    #        height,
    #        gl.GL_RGB,
    #        gl.GL_UNSIGNED_BYTE,
    #        img_array.ctypes.data_as(POINTER(gl.GLubyte))
    #)

    # Unbind the frame buffer
    #gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    # Flip the image because OpenGL maps (0,0) to the lower-left corner
    # Note: this is necessary for gym.wrappers.Monitor to record videos
    # properly, otherwise they are vertically inverted.
    #img_array = np.ascontiguousarray(np.flip(img_array, axis=0))

    return scene, cam
end

function render_obs(sim::Simulator)
    ##
    #Render an observation from the point of view of the agent
    ##

    observation, cam = _render_img(
            sim.fixedparams,
            sim.cur_pos,
            sim.cur_angle,
            #sim.multi_fbo,
            #sim.final_fbo,
            false
    )

    # self.undistort - for UndistortWrapper
    #NOTE: Not distorting as of now
    #if sim.distortion && !sim.undistort
    #    observation = distort(sim.camera_model, observation)
    #end

    # Setup some basic lighting with a far away sun
    #TODO: See this later, use raytracer's light example
    if sim.fixedparams.domain_rand
        light_pos = Vec3(sim.fixedparams.randomization_settings["light_pos"]...)
    else
        light_pos = Vec3([-40f0], [200f0], [100f0])
    end

    light = DistantLight(Vec3([1f0]), 5000f0, Vec3([0f0], [1f0], [0f0]))

    origin, direction = get_primary_rays(cam)

    im = raytrace(origin, direction, observation, light, origin, 2)
    color_r = reshape(im.x, sim.fixedparams.camera_width, sim.fixedparams.camera_height)
    color_g = reshape(im.y, sim.fixedparams.camera_width, sim.fixedparams.camera_height)
    color_b = reshape(im.z, sim.fixedparams.camera_width, sim.fixedparams.camera_height)

    shape = (sim.fixedparams.camera_width, sim.fixedparams.camera_height, 3, 1)
    im_arr = zeroonenorm(reshape(hcat(color_r, color_g, color_b), shape))
    return im_arr
end

function render(sim::Simulator, mode="human", close=false)
    ##
    #Render the environment for human viewing
    ##
    #=
    if close:
        if self.window:
            self.window.close()
        return

    top_down = mode == 'top_down'
    # Render the image
    img = self._render_img(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            self.multi_fbo_human,
            self.final_fbo_human,
            self.img_array_human,
            top_down=top_down
    )

    # self.undistort - for UndistortWrapper
    if self.distortion and not self.undistort and mode != "free_cam":
        img = self.camera_model.distort(img)

    if mode == 'rgb_array':
        return img

    from pyglet import gl, window, image

    if self.window is None:
        config = gl.Config(double_buffer=False)
        self.window = window.Window(
                width=WINDOW_WIDTH,
                height=WINDOW_HEIGHT,
                resizable=False,
                config=config
        )

    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()

    # Bind the default frame buffer
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    # Setup orghogonal projection
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0, 10)

    # Draw the image to the rendering window
    width = img.shape[1]
    height = img.shape[0]
    img = np.ascontiguousarray(np.flip(img, axis=0))
    img_data = image.ImageData(
            width,
            height,
            'RGB',
            img.ctypes.data_as(POINTER(gl.GLubyte)),
            pitch=width * 3,
    )
    img_data.blit(
            0,
            0,
            0,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT
    )

    # Display position/state information
    if mode != "free_cam":
        x, y, z = self.cur_pos
        self.text_label.text = "pos: (%.2f, %.2f, %.2f), angle: %d, steps: %d, speed: %.2f m/s" % (
            x, y, z,
            int(self.cur_angle * 180 / math.pi),
            self.step_count,
            self.speed
        )
        self.text_label.draw()

    # Force execution of queued commands
    gl.glFlush()
=#
end

function _update_pos(pos, angle, wheel_dist, wheelVels, deltaTime)
    ##
    #Update the position of the robot, simulating differential drive
    #
    #returns new_pos, new_angle
    ##

    Vl, Vr = wheelVels
    l = wheel_dist

    # If the wheel velocities are the same, then there is no rotation
    if Vl == Vr
        pos = pos .+ deltaTime * Vl * get_dir_vec(angle)
        return pos, angle
    end

    # Compute the angular rotation velocity about the ICC (center of curvature)
    w = (Vr - Vl) / l

    # Compute the distance to the center of curvature
    r = (l * (Vl + Vr)) / (2f0(Vl - Vr))

    # Compute the rotation angle for this time step
    rotAngle = w * deltaTime

    # Rotate the robot's position around the center of rotation
    r_vec = get_right_vec(angle)
    px, py, pz = pos
    cx = px .+ r * r_vec[1]
    cz = pz .+ r * r_vec[3]
    npx, npz = rotate_point(px, pz, cx, cz, rotAngle)
    pos = [npx, py, npz]

    # Update the robot's direction angle
    angle = angle + rotAngle
    return pos, angle
end


function _actual_center(pos, angle)
    ##
    #Calculate the position of the geometric center of the agent
    #The value of self.cur_pos is the center of rotation.
    ##

    dir_vec = get_dir_vec(angle)
    return pos .+ (CAMERA_FORWARD_DIST - 0.5f0 * ROBOT_LENGTH) .* dir_vec
end


function get_agent_corners(pos, angle)
    agent_corners = agent_boundbox(
            _actual_center(pos, angle),
            ROBOT_WIDTH,
            ROBOT_LENGTH,
            get_dir_vec(angle),
            get_right_vec(angle)
    )
    return agent_corners
end
