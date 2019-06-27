# coding=utf-8
using Distributions: sample
using .ObjMesh
abstract type AbstractWorldObj end

mutable struct WorldObj <: AbstractWorldObj
    visible::Bool
    color::Vec3
    domain_rand::Bool
    angle::Float32
    y_rot::Float32
    pos::Vector{Float32}
    scale::Float32
    min_coords::Vector{Float32}
    max_coords::Vector{Float32}
    kind::String
    mesh::ObjectMesh
    optional::Bool
    static::Bool
    safety_radius::Float32
    obj_corners::Array{Float32, 2}
    obj_norm::Array{Float32, 2}
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

_obj_corners(wobj::WorldObj) = wobj.obj_corners
_obj_corners(obj::AbstractWorldObj) = _obj_corners(obj.wobj)

function _set_color!(wobj::WorldObj, color::Vec3)
    wobj.color = color
end

_set_color!(obj::AbstractWorldObj, color::Vec3) = _set_color!(obj.wobj, color)

function _set_visible!(wobj::WorldObj, val::Bool)
    wobj.visible = val
end

_set_visible!(obj::AbstractWorldObj, val::Bool) = _set_visible!(obj.wobj, val)
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

_mesh(wobj::WorldObj) = wobj.mesh
_mesh(obj::AbstractWorldObj) = _mesh(obj.wobj)

_color(wobj::WorldObj) = wobj.color
_color(obj::AbstractWorldObj) = _color(obj.wobj)

function render(obj::AbstractWorldObj, draw_bbox::Bool)
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

    transformation_mat = get_transformation_mat(_pos(obj), _scale(obj), _y_rot(obj), (0,1,0))
    obj_mesh = _mesh(obj)
    transformed_vlists = transform_mesh(obj_mesh, transformation_mat)
    #let mesh store a matrix. make vec3 out of it only when you render
    Δed_faces = triangulate_faces.(transformed_vlists, obj_mesh.texclists, obj_mesh.clists, obj_mesh.textures)
    Δed_faces = vcat(Δed_faces...)
end

function make_Δ(v1::Vec3, vert2::Vector{Float32}, vert3::Vector{Float32}, mat::Material)
    v2 = Vec3(vert2[1:1], vert2[2:2], vert2[3:3])
    v3 = Vec3(vert3[1:1], vert3[2:2], vert3[3:3])
    Triangle(v1, v2, v3, mat)
end

function make_Δ(v1::Vec3, vert2::Vector{Float32}, vert3::Vector{Float32},
                tc1::Vector{Float32}, tc2::Vector{Float32}, tc3::Vector{Float32},
                texture::NamedTuple)
    v2 = Vec3(vert2[1:1], vert2[2:2], vert2[3:3])
    v3 = Vec3(vert3[1:1], vert3[2:2], vert3[3:3])

    uv_coordinates = [tc1, tc2, tc3]

    mat = Material(;uv_coordinates=uv_coordinates, texture...)
    Triangle(v1, v2, v3, mat)
end

function make_Δ(v1::Vec3, vert2::Vector{Float32}, vert3::Vector{Float32},
                tc1::Vector{Float32}, tc2::Vector{Float32}, tc3::Vector{Float32},
                color::Vec3)
    v2 = Vec3(vert2[1:1], vert2[2:2], vert2[3:3])
    v3 = Vec3(vert3[1:1], vert3[2:2], vert3[3:3])

    uv_coordinates = [tc1, tc2, tc3]

    mat = Material(;uv_coordinates=uv_coordinates, color_diffuse=color)
    Triangle(v1, v2, v3, mat)
end

function triangulate_faces(list_verts::Matrix{Float32}, texc::Matrix{Float32},
                           color::Vec3, texture::NamedTuple)
    v1 = Vec3(list_verts[1, 1:1], list_verts[1, 2:2], list_verts[1, 3:3])
    num_verts = size(list_verts, 1)

    texture_ = (color_diffuse = color,
                texture_diffuse = texture.texture_diffuse)

    Δs = map(i->make_Δ(v1, list_verts[i, :], list_verts[i+1, :], texc[1, :],
                       texc[i, :], texc[i+1, :], texture_), 2:num_verts-1)
    return Δs
end

function triangulate_faces(list_verts::Matrix{Float32}, texc::Matrix{Float32},
                           color::Vec3, texture::Nothing=nothing)
    v1 = Vec3(list_verts[1, 1:1], list_verts[1, 2:2], list_verts[1, 3:3])
    num_verts = size(list_verts, 1)

    Δs = map(i->make_Δ(v1, list_verts[i, :], list_verts[i+1, :], texc[1, :],
                       texc[i, :], texc[i+1, :], color), 2:num_verts-1)
    return Δs
end

function triangulate_faces(list_verts::Matrix{Float32}, texc::Nothing,
                           color::Vec3, texture::Nothing=nothing)
    v1 = Vec3(list_verts[1, 1:1], list_verts[1, 2:2], list_verts[1, 3:3])
    num_verts = size(list_verts, 1)

    mat = Material(;color_diffuse=color)

    Δs = map(i->make_Δ(v1, list_verts[i, :], list_verts[i+1, :], mat), 2:num_verts-1)
    return Δs
end


function triangulate_faces(list_verts::Matrix{Float32}, texc::Nothing,
                           color::Vec3, texture::NamedTuple)
    v1 = Vec3(list_verts[1, 1:1], list_verts[1, 2:2], list_verts[1, 3:3])
    num_verts = size(list_verts, 1)

    texture_ = (color_diffuse = color,
                texture_diffuse = texture.texture_diffuse)
    mat = Material(;texture_...)

    Δs = map(i->make_Δ(v1, list_verts[i, :], list_verts[i+1, :], mat), 2:num_verts-1)
    return Δs
end

function triangulate_faces(list_verts::Array{Float32, 3}, list_texc::Array{Float32, 3},
                           list_colors::Vector{Vec3}, texture::Union{NamedTuple,Nothing})
    # list_verts: 3D array representing list of faces along 3rd dim.
    # Every face has 3 vertices. Vertices are along 1st dim.
    # X, Y and Z component of vertices, along 2nd dim.
    # Similarly for texture coordinates

    @assert size(list_verts, 1) == 3
    @assert size(list_verts, 3) ==  size(list_colors, 1)

    num_faces = size(list_verts, 3)
    vt = Vector{Triangle}()
    for i in 1:num_faces
        Δ = triangulate_faces(list_verts[:, :, i], list_texc[:, :, i],
                              list_colors[i], texture)
        vt = vcat(vt, Δ)
    end
    return vt
end

function triangulate_faces(list_verts::Array{Float32,3}, list_texc::Array{Float32,3},
                           list_colors::Vector{Vec3}, texture::Vec3)
   texture_ = (texture_diffuse = texture,)
   triangulate_faces(list_verts, list_texc, list_colors, texture_)
end

get_transformation_mat(pos::Vector, scale, θ, rot_axis) =
    rotate_mat(θ) * scale_mat(scale) * translation_mat(pos)

translation_mat(pos...) = translation_mat(collect(Float32.(pos)))

function translation_mat(pos::Vector{Float32})
    #@assert length(pos) == 3
    #mat = Matrix{Float32}(I, 3, 3)
    mat = [
        1f0 0f0 0f0 0f0;
        0f0 1f0 0f0 0f0;
        0f0 0f0 1f0 0f0;
        pos[1] pos[2] pos[3] 1f0]

    return mat
end

scale_mat(scale...) = scale_mat(collect(Float32.(scale)))

scale_mat(scale::Float32) = scale_mat(ones(Float32, 3) * scale)

function scale_mat(scale::Vector{Float32})
    # scale is a vector of length 3
    scale_ = vcat(scale, [1f0])
    #mat = Matrix{Float32}(I, 4, 4)
    mat = [
        1f0 0f0 0f0 0f0;
        0f0 1f0 0f0 0f0;
        0f0 0f0 1f0 0f0;
        0f0 0f0 0f0 1f0]

    return mat .* scale_
end

function rotate_mat(θ, axis=(0,1,0))
    # axis: one-hot vector, each element corresponds to x, y or z axis
    θ = deg2rad(θ)
    #mat = Matrix{Float32}(I, 4, 4)
    mat = nothing
    if axis[2] == 1
        mat = [
        cos(θ) 0f0 -sin(θ) 0f0;
        0f0    1f0 0f0     0f0;
        sin(θ) 0f0 cos(θ)  0f0;
        0f0    0f0 0f0     1f0]
    elseif axis[1] == 1
        mat = [
        1f0    0f0    0f0     0f0;
        0f0    cos(θ) sin(θ)  0f0;
        0f0   -sin(θ) cos(θ)  0f0;
        0f0    0f0    0f0     1f0]
    else
        mat = [
         cos(θ)   sin(θ)   0f0  0f0;
        -sin(θ)   cos(θ)   0f0  0f0;
         0f0      0f0      1f0  0f0;
         0f0      0f0      0f0  1f0]
    end

    return mat
end

function transform_mat(mat::Matrix{T}, transformation_mat::Matrix{T}) where T
    numVecs, dim = size(mat)
    mat_ = hcat(mat, ones(eltype(mat), numVecs))
    return (mat_ * transformation_mat)[:, 1:dim]
end

function transform_mesh(vlist::Array{Float32,3}, transformation_mat::Array{Float32,2})
    num_faces = size(vlist, 3)
    trans_vlist = map(i -> transform_mat(vlist[:, :, i], transformation_mat), 1:num_faces)
    return cat(trans_vlist..., dims=3)
end

transform_mesh(obj_mesh::ObjectMesh, transformation_mat::Array{Float32,2}) =
map(vlist->transform_mesh(vlist, transformation_mat), obj_mesh.vlists)

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

function proximity(wobj::WorldObj, agent_pos::Vector{Float32}, agent_safety_rad::Float32)
    ##
    #See if the agent is too close to this object
    #For static, return 0
    ##
    !wobj.static && throw(NotImplementedError)

    return 0f0
end


proximity(abs_wobj::AbstractWorldObj, agent_pos::Vector{Float32},
          agent_safety_rad::Float32) = proximity(abs_wobj.wobj, agent_pos, agent_safety_rad)

function step!(wobj::WorldObj, delta_time::Float32)
    ##
    #Use a motion model to move the object in the world
    ##
    !wobj.static && throw(NotImplementedError)
end

step!(abs_wobj::AbstractWorldObj, delta_time::Float32) = step!(abs_wobj.wobj, delta_time)

mutable struct DuckiebotObj <: AbstractWorldObj
    wobj::WorldObj
    follow_dist::Float32
    velocity::Float32
    max_iterations::Int
    gain::Float32
    trim::Float32
    radius::Float32
    k::Float32
    limit::Float32
    wheel_dist::Float32
    robot_width::Float32
    robot_length::Float32
end

function DuckiebotObj(obj, domain_rand::Bool, safety_radius_mult::Float32,
                      wheel_dist::Float32, robot_width::Float32, robot_length::Float32,
                      gain::Float32=2f0, trim::Float32=0f0, radius::Float32=0.0318f0,
                      k::Float32=27f0, limit::Float32=1f0)
    wobj = WorldObj(obj, domain_rand, safety_radius_mult)
    follow_dist, velocity = nothing, nothing
    if domain_rand
        follow_dist = Float32(rand(Uniform(0.3f0, 0.4f0)))
        velocity = Float32(rand(Uniform(0.05f0, 0.15f0)))
    else
        follow_dist = 0.3f0
        velocity = 0.1f0
    end

    max_iterations = 1000

    DuckiebotObj(wobj, follow_dist, velocity, max_iterations, gain, trim,
                 radius, k, limit, wheel_dist, robot_width, robot_length)
end

function get_dir_vec(db_obj::DuckiebotObj, angle)
    x = cos(angle)
    z = -sin(angle)
    return [x, 0f0, z]
end

function get_right_vec(db_obj::DuckiebotObj, angle)
    x = sin(angle)
    z = cos(angle)
    return [x, 0f0, z]
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

function _update_pos(db_obj::DuckiebotObj, action::Vector{Float32}, deltaTime::Float32)
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
    cx = px + r * r_vec
    cz = pz + r * r_vec
    npx, npz = rotate_point(px, pz, cx, cz, rotAngle)

    # Update position
    db_obj.pos = [npx, py, npz]

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
    pedestrian_wait_time::Int
    vel::Float32
    heading::Vector{Float32}
    start::Vector{Float32}
    center::Vector{Float32}
    pedestrian_active::Bool
    wiggle::Int
end

function DuckieObj(obj, domain_rand::Bool, safety_radius_mult::Float32, walk_distance::Float32)
    wobj = WorldObj(obj, domain_rand, safety_radius_mult)

    walk_distance = walk_distance + 0.25f0

    # Dynamic duckie stuff

    # Randomize velocity and wait time
    if wobj.domain_rand
        pedestrian_wait_time = rand(3:19)
        vel = abs(Float32(rand(Normal(0.02f0, 0.005f0))))
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
    wiggle = Float32(π / wiggle)

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

function proximity(db_obj::Obj, agent_pos::Vector{Float32}, agent_safety_rad::Float32) where Obj <: Union{DuckieObj, DuckiebotObj}
    ##
    #See if the agent is too close to this object
    #based on a heuristic for the "overlap" between
    #their safety circles
    ##
    d = norm(agent_pos .- pos)
    score = d - agent_safety_rad - self.safety_radius

    return min(0f0, score)
end

function step!(dobj::DuckieObj, delta_time::Float32)
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
    dobj.angle += Float32(π)
    dobj.pedestrian_active = false

    if dobj.wobj.domain_rand
        # Assign a random velocity (in opp. direction) and a wait time
        # TODO: Fix this: This will go to 0 over time
        dobj.vel = -sign(dobj.vel) * abs(Float32(rand(Normal(0.02f0, 0.005f0))))
        dobj.pedestrian_wait_time = rand(3:19)
    else
        # Just give it the negative of its current velocity
        dobj.vel *= -1f0
        dobj.pedestrian_wait_time = 8
    end
end

mutable struct TrafficLightObj <: AbstractWorldObj
    wobj::WorldObj
    texs::Vector{Vec3}
    time::Float32
    freq::Float32
    pattern::Int
end

function TrafficLightObj(obj, domain_rand, safety_radius_mult)
    wobj = WorldObj(obj, domain_rand, safety_radius_mult)

    texs = [
        load_texture(get_file_path("src/textures", "trafficlight_card0", "jpg")),
        load_texture(get_file_path("src/textures", "trafficlight_card1", "jpg"))
    ]
    time = 0f0

    # Frequency and current pattern of the lights
    if wobj.domain_rand
        freq = Float32(rand(Normal(4, 7)))
        pattern = rand(1:2)
    else
        freq = 5f0
        pattern = 1
    end

    # Use the selected pattern
    wobj.mesh.textures[1] = texs[pattern]

    TrafficLightObj(wobj, texs, time, freq, pattern)
end

function step!(tl_obj::TrafficLightObj, delta_time::Float32)
    ##
    #Changes the light color periodically
    ##

    tl_obj.time += delta_time
    if round(tl_obj.time, 3) % tl_obj.freq == 0  # Swap patterns
        tl_obj.pattern ⊻= 3
        tl_obj.wobj.mesh.textures[1] = tl_obj.texs[tl_obj.pattern]
    end
end

function is_green(tl_obj::TrafficLightObj, direction='N')
    if direction == 'N' || direction == 'S'
        if tl_obj.wobj.y_rot == 45 || tl_obj.wobj.y_rot == 135
            return tl_obj.pattern == 1
        elseif tl_obj.wobj.y_rot == 225 || tl_obj.wobj.y_rot == 315
            return tl_obj.pattern == 2
        end
    elseif direction == 'E' || direction == 'W'
        if tl_obj.wobj.y_rot == 45 || tl_obj.wobj.y_rot == 135
            return tl_obj.pattern == 2
        elseif tl_obj.wobj.y_rot == 225 || tl_obj.wobj.y_rot == 315
            return tl_obj.pattern == 1
        end
    end
    return false
end
