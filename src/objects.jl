# coding=utf-8
using Distributions: sample
using .ObjMesh
abstract type AbstractWorldObj end

mutable struct WorldObj <: AbstractWorldObj
    visible::Bool
    color::Vec3
    domain_rand
    angle
    y_rot
    pos
    scale
    min_coords
    max_coords
    kind
    mesh
    optional
    static
    safety_radius
    obj_corners
    obj_norm
end

function WorldObj(obj, domain_rand, safety_radius_mult)
    ##
    #Initializes the object and its properties
    ##
    # XXX this is relied on by things but it is not always set
    # (Static analysis complains)
    visible = true
    # same
    color = Vec3([0f0])
    # maybe have an abstract method is_visible, get_color()

    dict_vals = process_obj_dict(obj, safety_radius_mult)
    pos, scale, min_coords, max_coords, y_rot = dict_vals[6:end]
    dict_vals = dict_vals[1:5]

    angle = deg2rad(y_rot)

    # Find corners and normal vectors assoc world object
    obj_corners = generate_corners(pos, min_coords, max_coords, angle, scale)
    obj_norm = generate_norm(obj_corners)

    WorldObj(visible, color, domain_rand, angle, y_rot, pos,
             scale, min_coords, max_coords, dict_vals...,
             obj_corners, obj_norm)
end

function process_obj_dict(obj, safety_radius_mult)
    kind = obj["kind"]
    mesh = obj["mesh"]
    pos = obj["pos"]
    scale = obj["scale"]
    optional = obj["optional"]
    min_coords = obj["mesh"].min_coords
    max_coords = obj["mesh"].max_coords
    static = obj["static"]
    safety_radius = safety_radius_mult * calculate_safety_radius(mesh, scale)
    y_rot = obj["y_rot"]

    return (kind, mesh, optional, static, safety_radius, pos, scale,
            min_coords, max_coords, y_rot)
end

function render(obj::AbstractWorldObj, draw_bbox)
    ##
    #Renders the object to screen
    ##
    !_visible(obj) && return

    # Draw the bounding box
    if draw_bbox
    #    gl.glColor3f(1, 0, 0)
    #    gl.glBegin(gl.GL_LINE_LOOP)
    #    gl.glVertex3f(self.obj_corners.T[0, 0], 0.01, self.obj_corners.T[1, 0])
    #    gl.glVertex3f(self.obj_corners.T[0, 1], 0.01, self.obj_corners.T[1, 1])
    #    gl.glVertex3f(self.obj_corners.T[0, 2], 0.01, self.obj_corners.T[1, 2])
    #    gl.glVertex3f(self.obj_corners.T[0, 3], 0.01, self.obj_corners.T[1, 3])
    #    gl.glEnd()
    end

    #gl.glTranslatef(*self.pos)
    #gl.glScalef(self.scale, self.scale, self.scale)
    #gl.glRotatef(self.y_rot, 0, 1, 0)
    #gl.glColor3f(*self.color)

    transformation_mat = get_transformation_mat(_pos(obj), _scale(obj), _yrot(obj), (0,1,0))

    tranformed_vlists = transform_mesh(obj.mesh, transformation_mat)
    #let mesh store a matrix. make vec3 out of it colorcoloronly when you render
    # Skipped texture for now
    Δed_faces = triangulate_faces.(tranformed_vlists, obj_mesh.clists)
    Δed_faces = vcat(Δed_faces...)
    #return TriangleMesh(Δ_ed_faces)
end

function triangulate_faces(list_verts::Matrix, color::Vec3)
    v1 = Vec3(list_verts[1, 1:1], -list_verts[1, 2:2], [list_verts[1, 3:3]])
    Δs = []
    for i in 2:size(list_verts, 1)-1
        v2 = Vec3(list_verts[i, 1:1], -list_verts[i, 2:2], list_verts[i, 3:3])
        v3 = Vec3(list_verts[i+1, 1:1], -list_verts[i+1, 2:2], list_verts[i+1, 3:3])
        push!(Δs, Triangle(v1, v2, v3; color=color))
    end
    return Δs
end

function triangulate_faces(list_verts::AbstractArray{T, 3}, list_colors::Vector{Vec3}) where T
    # list_verts: 3D array representing list of faces along 3rd dim.
    # Every face has 3 vertices. Vertices are along 1st dim. X, Y and Z component
    # of vertices, along 2nd dim.

    @assert size(list_verts, 1) == 3
    @assert size(list_verts, 3) ==  size(list_colors, 1)

    num_faces = size(list_verts, 3)
    vt = Vector{Triangle}()
    for i in 1:num_faces
        Δ = triangulate_faces(list_verts[:, :, i], list_colors[i])
        vt = vcat(vt, Δ)
    end
    return vt
end

get_transformation_mat(pos::Vector, scale, θ, rot_axis) =
    rotate_mat(θ) * scale_mat(scale) * translation_mat(pos)

translation_mat(pos...) = translation_mat(pos)

function translation_mat(pos::Vector)
    @assert length(pos) == 3
    mat = Matrix{Float32}(I, 4, 4)
    mat[4, 1:3] .= pos
    return mat
end

scale_mat(pos...) = scale_mat(pos)

function scale_mat(scale)
    mat = Matrix{Float32}(I, 4, 4)
    mat[1:3, 1:3] .*= scale
    return mat
end

function rotate_mat(θ, axis=(0,1,0))
    # axis: one-hot vector, each element corresponds to x, y or z axis
    θ = deg2rad(θ)
    mat = Matrix{Float32}(I, 4, 4)
    axis = argmax(axis)
    if axis == 2
        mat[1:2:3, 1:2:3] .= [cos(θ) -sin(θ); sin(θ) cos(θ)]
    elseif axis == 1
        mat[2:3, 2:3] .= [cos(θ) sin(θ); -sin(θ) cos(θ)]
    else
        mat[1:2, 1:2] .= [cos(θ) sin(θ); -sin(θ) cos(θ)]
    end

    return mat
end

function transform_mat(mat::Matrix{T}, transformation_mat::Matrix{T}) where T
    numVecs, dim = size(mat)
    veclist = ones(eltype(mat), numVecs, dim+1)
    veclist[:, 1:dim] .= mat
    return (veclist * transformation_mat)[:, 1:dim]
end

function tranform_mesh(mesh_mat::AbstractArray{T, 3}, transformation_mat::Matrix{T}) where T
    trans_vlist = similar(mesh_mat)
    num_faces = size(mesh_mat, 3)
    for i in 1:num_faces
        trans_vlist[:, :, i] .= transform_mat(vlist[:, :, i], transformation_mat)
    end

    return trans_vlist
end

transform_mesh(obj_mesh::ObjectMesh, transformation_mat::Matrix) =
[vlist->transform_mesh(vlist, transformation_mat) for vlist in vlists]

# Below are the functions that need to
# be reimplemented for any dynamic object
function check_collision(wobj::WorldObj, agent_corners, agent_norm)
    ##
    #See if the agent collided with this object
    #For static, return false (static collisions checked w
    #numpy in a batch operation)
    ##
    !wobj.static && throw(NotImplementedError)

    return false
end

function proximity(wobj::WorldObj, agent_pos, agent_safety_rad)
    ##
    #See if the agent is too close to this object
    #For static, return 0
    ##
    !wobj.static && throw(NotImplementedError)

    return 0f0
end

function step!(wobj::WorldObj, delta_time)
    ##
    #Use a motion model to move the object in the world
    ##
    !sim.static && throw(NotImplementedError)
end


mutable struct DuckiebotObj <: AbstractWorldObj
    wobj::WorldObj
    follow_dist
    velocity
    max_iterations
    gain
    trim
    radius
    k
    limit
    wheel_dist
    robot_width
    robot_length
end

function DuckiebotObj(obj, domain_rand, safety_radius_mult, wheel_dist,
                      robot_width, robot_length, gain=2.0, trim=0.0,
                      radius=0.0318, k=27.0, limit=1.0)
    wobj = WorldObj(obj, domain_rand, safety_radius_mult)

    if domain_rand
        follow_dist = rand(Uniform(0.3, 0.4))
        velocity = rand(Uniform(0.05, 0.15))
    else
        follow_dist = 0.3
        velocity = 0.1
    end

    max_iterations = 1000

    DuckiebotObj(wobj, follow_dist, velocity, max_iterations, gain, trim,
                 radius, k, limit, wheel_dist, robot_width, robot_length)
end

# FIXME: this does not follow the same signature as WorldOb
function step!(db_obj::DuckiebotObj, delta_time, closest_curve_point, objects)
    ##
    #Take a step, implemented as a PID controller
    ##

    # Find the curve point closest to the agent, and the tangent at that point
    closest_point, closest_tangent = closest_curve_point(db_obj.wobj.pos, db_obj.wobj.angle)

    iterations = 0

    lookup_distance = db_obj.follow_dist
    curve_point = nothing
    while iterations < db_obj.max_iterations
        # Project a point ahead along the curve tangent,
        # then find the closest point to to that
        follow_point = closest_point .+ closest_tangent * lookup_distance
        curve_point, _ = closest_curve_point(follow_point, db_obj.wobj.angle)

        # If we have a valid point on the curve, stop
        isnothing(curve_point) && break

        iterations += 1
        lookup_distance *= 0.5
    end

    # Compute a normalized vector to the curve point
    point_vec = curve_point .- db_obj.wobj.pos
    point_vec ./= norm(point_vec)

    dot = dot(get_right_vec(db_obj, db_obj.wobj.angle), point_vec)
    steering = db_obj.gain * (-dot)

    _update_pos(db_obj, [db_obj.velocity, steering], delta_time)
end

function get_dir_vec(db_obj::DuckiebotObj, angle)
    x = cos(angle)
    z = -sin(angle)
    return [x, 0, z]
end

function get_right_vec(db_obj::DuckiebotObj, angle)
    x = sin(angle)
    z = cos(angle)
    return [x, 0, z]
end

function check_collision(db_obj::DuckiebotObj, agent_corners, agent_norm)
    ##
    #See if the agent collided with this object
    ##
    return intersects_single_obj(
        agent_corners,
        permutedims(db_obj.wobj.obj_corners),
        agent_norm,
        db_obj.wobj.obj_norm
    )
end

function proximity(db_obj::DuckiebotObj, agent_pos, agent_safety_rad)
    ##
    #See if the agent is too close to this object
    #based on a heuristic for the "overlap" between
    #their safety circles
    ##
    d = norm(agent_pos .- pos)
    score = d - agent_safety_rad - self.safety_radius

    return min(0, score)
end

function _update_pos(db_obj::DuckiebotObj, action, deltaTime)
    vel, angle = action

    # assuming same motor constants k for both motors
    k_r = db_obj.k
    k_l = db_obj.k

    # adjusting k by gain and trim
    k_r_inv = (db_obj.gain + db_obj.trim) / k_r
    k_l_inv = (db_obj.gain - db_obj.trim) / k_l

    omega_r = (vel + 0.5f0 * angle * db_obj.wheel_dist) / db_obj.radius
    omega_l = (vel - 0.5f0 * angle * db_obj.wheel_dist) / db_obj.radius

    # conversion from motor rotation rate to duty cycle
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv

    # limiting output to limit, which is 1.0 for the duckiebot
    u_r_limited = max(min(u_r, db_obj.limit), -db_obj.limit)
    u_l_limited = max(min(u_l, db_obj.limit), -db_obj.limit)

    # If the wheel velocities are the same, then there is no rotation
    if u_l_limited == u_r_limited
        db_obj.pos = db_obj.pos .+ deltaTime * u_l_limited * get_dir_vec(db_obj, db_obj.wobj.angle)
        return
    end

    # Compute the angular rotation velocity about the ICC (center of curvature)
    w = (u_r_limited - u_l_limited) / db_obj.wheel_dist

    # Compute the distance to the center of curvature
    r = (db_obj.wheel_dist * (u_l_limited + u_r_limited)) / (2 * (u_l_limited - u_r_limited))

    # Compute the rotation angle for this time step
    rotAngle = w * deltaTime

    # Rotate the robot's position around the center of rotation
    r_vec = get_right_vec(db_obj, db_obj.wobj.angle)
    px, py, pz = db_obj.pos
    cx = px .+ r * r_vec[1:1]
    cz = pz .+ r * r_vec[3:3]
    npx, npz = rotate_point.(px, pz, cx, cz, rotAngle)

    # Update position
    db_obj.pos = vcat(npx, py, npz)

    # Update the robot's direction angle
    db_obj.wobj.angle += rotAngle
    db_obj.wobj.y_rot += rand2deg(rotAngle)

    # Recompute the bounding boxes (BB) for the duckiebot
    db_obj.wobj.obj_corners = agent_boundbox(
        db_obj.pos,
        db_obj.robot_width,
        db_obj.robot_length,
        get_dir_vec(db_obj, db_obj.wobj.angle),
        get_right_vec(db_obj, db_obj.wobj.angle)
    )
end

mutable struct DuckieObj <: AbstractWorldObj
    wobj::WorldObj
    pedestrian_wait_time
    vel
    heading
    start
    center
    pedestrian_active
    wiggle
end

function DuckieObj(obj, domain_rand::Bool, safety_radius_mult, walk_distance)
    wobj = WorldObj(obj, domain_rand, safety_radius_mult)

    walk_distance = walk_distance + 0.25f0

    # Dynamic duckie stuff

    # Randomize velocity and wait time
    if wobj.domain_rand
        pedestrian_wait_time = rand(3:19)
        vel = abs(rand(Normal(0.02f0, 0.005f0)))
    else
        pedestrian_wait_time = 8
        vel = 0.02f0
    end

    # Movement parameters
    heading = heading_vec(wobj.angle)
    start = copy(wobj.pos)
    center = wobj.pos
    pedestrian_active = false

    # Walk wiggle parameter
    wiggle = sample([14, 15, 16], 1)
    wiggle = π ./ wiggle

    time = 0

    DuckieObj(wobj, pedestrian_wait_time, vel, heading,
              start, center, pedestrian_active, wiggle)
end

function check_collision(dobj::DuckieObj, agent_corners, agent_norm)
    ##
    #See if the agent collided with this object
    ##
    return intersects_single_obj(
        agent_corners,
        permutedims(dobj.wobj.obj_corners),
        agent_norm,
        dobj.wobj.obj_norm
    )
end

function proximity(dobj::DuckieObj, agent_pos, agent_safety_rad)
    ##
    #See if the agent is too close to this object
    #based on a heuristic for the "overlap" between
    #their safety circles
    ##
    d = norm(agent_pos .- dobj.center)
    score = d .- agent_safety_rad .- dobj.wobj.safety_radius

    return min(0, score)
end

function step!(dobj::DuckieObj, delta_time)
    ##
    #Use a motion model to move the object in the world
    ##

    dobj.time += delta_time

    # If not walking, no need to do anything
    if !dobj.pedestrian_active
        dobj.pedestrian_wait_time -= delta_time
        if dobj.pedestrian_wait_time ≤ 0
            dobj.pedestrian_active = true
        end
        return
    end

    # Update centers and bounding box
    vel_adjust = dobj.heading * dobj.vel
    dobj.center += vel_adjust
    dobj.obj_corners += vel_adjust[[1, end]]

    distance = norm(dobj.center .- dobj.start)

    distance > dobj.walk_distance && finish_walk(dobj)

    dobj.pos = dobj.center
    angle_delta = dobj.wiggle * sin(48dobj.time)
    dobj.wobj.y_rot = rad2deg(dobj.wobj.angle + angle_delta)
    dobj.wobj.obj_norm = generate_norm(dobj.obj_corners)
end

function finish_walk(dobj::DuckieObj)
    ##
    #After duckie crosses, update relevant attributes
    #(vel, rot, wait time until next walk)
    ##
    dobj.start = copy(dobj.center)
    dobj.angle += π
    dobj.pedestrian_active = false

    if dobj.wobj.domain_rand
        # Assign a random velocity (in opp. direction) and a wait time
        # TODO: Fix this: This will go to 0 over time
        dobj.vel = -sign(dobj.vel) * abs(rand(Normal(0.02f0, 0.005f0)))
        dobj.pedestrian_wait_time = rand(3:19)
    else
        # Just give it the negative of its current velocity
        dobj.vel *= -1
        dobj.pedestrian_wait_time = 8
    end
end

mutable struct TrafficLightObj <: AbstractWorldObj
    wobj::WorldObj
    texs
    time
    freq
    pattern
end

function TrafficLightObj(obj, domain_rand, safety_radius_mult)
    wobj = WorldObj(obj, domain_rand, safety_radius_mult)

    texs = [
        load_texture(get_file_path("src/textures", "trafficlight_card0", "jpg")),
        load_texture(get_file_path("src/textures", "trafficlight_card1", "jpg"))
    ]
    time = 0

    # Frequency and current pattern of the lights
    if wobj.domain_rand
        freq = rand(Normal(4, 7))
        pattern = rand(0:1)
    else
        freq = 5
        pattern = 0
    end

    # Use the selected pattern
    wobj.mesh.textures[1] = texs[pattern]

    TrafficLightObj(wobj, texs, time, freq, pattern)
end

function step!(tl_obj::TrafficLightObj, delta_time)
    ##
    #Changes the light color periodically
    ##

    tl_obj.time += delta_time
    if round(tl_obj.time, 3) % tl_obj.freq == 0  # Swap patterns
        seltl_objf.pattern ^= 1
        tl_obj.wobj.mesh.textures[1] = tl_obj.texs[tl_obj.pattern]
    end
end

function is_green(tl_obj::TrafficLightObj, direction='N')
    if direction == 'N' || direction == 'S'
        if tl_obj.wobj.y_rot == 45 || tl_obj.wobj.y_rot == 135
            return tl_obj.pattern == 0
        elseif tl_obj.wobj.y_rot == 225 || tl_obj.wobj.y_rot == 315
            return tl_obj.pattern == 1
        end
    elseif direction == 'E' || direction == 'W'
        if tl_obj.wobj.y_rot == 45 || tl_obj.wobj.y_rot == 135
            return self.pattern == 1
        elseif tl_obj.wobj.y_rot == 225 || tl_obj.wobj.y_rot == 315
            return self.pattern == 0
        end
    end
    return false
end

_obj_corners(wobj::WorldObj) = wobj.obj_corners
_obj_corners(obj::AbstractWorldObj) = _obj_corners(obj.wobj)

function _set_color!(wobj::WorldObj, color::Vec3)
    wobj.color = color
end

_set_color!(obj::AbstractWorldObj, color::Vec3) = _set_color!(obj.wobj, color)

function _set_visible!(wobj::WorldObj, val::Bool)
    wobj.visible = val
end

_set_visible!(obj::AbstractWorldObj, val::Bool) = _set_color!(obj.wobj, val)
_visible(wobj::WorldObj) = wobj.visible
_visible(obj::AbstractWorldObj) = _visible(obj.wobj)

_optional(wobj::WorldObj) = wobj.optional
_optional(obj::AbstractWorldObj) = _optional(obj.wobj)

_pos(wobj::WorldObj) = wobj.pos
_pos(obj::AbstractWorldObj) = _pos(obj.wobj)

_max_coords(wobj::WorldObj) = wobj.max_coords
_max_coords(obj::AbstractWorldObj) = _max_coords(obj.wobj)

_scale(wobj::WorldObj) = wobj.scale
_scale(obj::AbstractWorldObj) = _scale(obj.wobj)

_y_rot(wobj::WorldObj) = wobj.y_rot
_y_rot(obj::AbstractWorldObj) = _y_rot(obj.wobj)
