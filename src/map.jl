struct ObjectData
    objects::Vector{AbstractWorldObj}
    collidable_centers::Vector{Array{Float32, 2}}
    collidable_corners::Vector{Array{Float32, 2}}
    collidable_norms::Vector{Array{Float32, 2}}
    collidable_safety_radii::Vector{Float32}
end

function ObjectData(map_data::Dict, road_tile_size::Float32, domain_rand::Bool, grid::Vector{Union{Missing, Dict{String,Any}}},
                    grid_width::Int, grid_height::Int)
    # Create the objects array
    objects = Vector{AbstractWorldObj}()

    # The corners for every object, regardless if collidable or not
    #object_corners = []

    # Arrays for checking collisions with N static objects
    # (Dynamic objects done separately)
    # (N x 2): Object position used in calculating reward
    collidable_centers = Vector{Vector{Float32}}()

    # (N x 2 x 4): 4 corners - (x, z) - for object's boundbox
    collidable_corners = Vector{Matrix{Float32}}()

    # (N x 2 x 2): two 2D norms for each object (1 per axis of boundbox)
    collidable_norms = Vector{Matrix{Float32}}()

    # (N): Safety radius for object used in calculating reward
    collidable_safety_radii = Vector{Float32}()

    # For each object
    for (obj_idx, desc) in enumerate(get(map_data, "objects", []))
        kind = desc["kind"]

        pos = Float32.(desc["pos"])
        x, z = pos[1:2]
        y = length(pos) == 3 ? pos[3] : 0f0

        rotate = Float32(desc["rotate"])
        optional = get(desc, "optional", false)

        pos = road_tile_size * [x, y, z]

        # Load the mesh
        #TODO
        mesh = ObjMesh.get(kind)

        if "height" in keys(desc)
            scale = Float32(desc["height"]) / mesh.max_coords[2]
        else
            scale = Float32(desc["scale"])
        end

        @assert !("height" ∈ keys(desc) && "scale" ∈ keys(desc)) "cannot specify both height and scale"

        static = get(desc, "static", true)

        obj_desc = Dict([
            "kind"=> kind,
            "mesh"=> mesh,
            "pos"=> pos,
            "scale"=> scale,
            "y_rot"=> rotate,
            "optional"=> optional,
            "static"=> static
        ])

        # obj = nothing

        if static
            if kind == "trafficlight"
                obj = TrafficLightObj(obj_desc, domain_rand, SAFETY_RAD_MULT)
            else
                obj = WorldObj(obj_desc, domain_rand, SAFETY_RAD_MULT)
            end
        else
            if kind == "duckiebot"
                obj = DuckiebotObj(obj_desc, domain_rand, SAFETY_RAD_MULT, WHEEL_DIST,
                                   ROBOT_WIDTH, ROBOT_LENGTH)
            elseif kind == "duckie"
                obj = DuckieObj(obj_desc, domain_rand, SAFETY_RAD_MULT, road_tile_size)
            else
                msg = "I do not know what object this is: $kind"
                error(msg)
            end
        end

        objects = vcat(objects, [obj])

        # Compute collision detection information

        # angle = rotate * (math.pi / 180)

        # Find drivable tiles object could intersect with
        possible_tiles = find_candidate_tiles(_obj_corners(obj), road_tile_size)

        # If the object intersects with a drivable tile

        if static && kind != "trafficlight" && _collidable_object(
                grid, grid_width, grid_height, obj.obj_corners, obj.obj_norm, possible_tiles, road_tile_size)
            collidable_centers = vcat(collidable_centers, pos)
            collidable_corners = cat(collidable_corners, obj.wobj.obj_corners, dims=3)
            collidable_norms = cat(collidable_norms, obj.wobj.obj_norm, dims=3)
            collidable_safety_radii = vcat(collidable_safety_radii, obj.wobj.safety_radius)
        end
    end
    # If there are collidable objects
    if size(collidable_corners, 3) > 0
        # Stack doesn't do anything if there's only one object,
        # So we add an extra dimension to avoid shape errors later
        if ndims(collidable_corners) == 2
            collidable_corners = add_axis(collidable_corners)
            collidable_norms = add_axis(collidable_norms)
        end
    end

    ObjectData(objects, collidable_centers, collidable_corners,
               collidable_norms, collidable_safety_radii)
end

function _get_curve(grid::Vector{Union{Missing,Dict{String,Any}}}, i::Int, j::Int,
                    width::Int, height::Int, road_tile_size::Float32)
    ##
    #    Get the Bezier curve control points for a given tile
    ##
    tile = _get_tile(grid, i, j, width, height)
    @assert !isnothing(tile)

    kind = tile["kind"]
    angle = tile["angle"]

    # Each tile will have a unique set of control points,
    # Corresponding to each of its possible turns

    if startswith(kind, "straight")
        pts = [
                [-0.2f0 0f0 -0.50f0;
                 -0.2f0 0f0 -0.25f0;
                 -0.2f0 0f0  0.25f0;
                 -0.2f0 0f0  0.50f0],

                [0.2f0 0f0  0.50f0;
                 0.2f0 0f0  0.25f0;
                 0.2f0 0f0 -0.25f0;
                 0.2f0 0f0 -0.50f0],
              ] .* road_tile_size

    elseif kind == "curve_left"
        pts = [
                [-0.2f0 0f0 -0.5f0;
                 -0.2f0 0f0  0.0f0;
                  0.0f0 0f0  0.2f0;
                  0.5f0 0f0  0.2f0],

                [0.5f0 0f0 -0.2f0;
                 0.3f0 0f0 -0.2f0;
                 0.2f0 0f0 -0.3f0;
                 0.2f0 0f0 -0.5f0],
              ] .* road_tile_size

    elseif kind == "curve_right"
        pts = [
                [-0.2f0 0f0 -0.5f0;
                 -0.2f0 0f0 -0.2f0;
                 -0.3f0 0f0 -0.2f0;
                 -0.5f0 0f0 -0.2f0],

                [-0.5f0 0f0  0.2f0;
                 -0.3f0 0f0  0.2f0;
                  0.3f0 0f0  0.0f0;
                  0.2f0 0f0 -0.5f0],
              ] .* road_tile_size

    # Hardcoded all curves for 3way intersection
    elseif startswith(kind, "3way")
        pts = [
                [-0.2f0 0f0 -0.50f0;
                 -0.2f0 0f0 -0.25f0;
                 -0.2f0 0f0  0.25f0;
                 -0.2f0 0f0  0.50f0],

                [-0.2f0 0f0 -0.5f0;
                 -0.2f0 0f0  0.0f0;
                  0.0f0 0f0  0.2f0;
                  0.5f0 0f0  0.2f0],

                [0.2f0 0f0  0.50f0;
                 0.2f0 0f0  0.25f0;
                 0.2f0 0f0 -0.25f0;
                 0.2f0 0f0 -0.50f0],

                [0.5f0 0f0 -0.2f0;
                 0.3f0 0f0 -0.2f0;
                 0.2f0 0f0 -0.2f0;
                 0.2f0 0f0 -0.5f0],

                [0.2f0 0f0 0.5f0;
                 0.2f0 0f0 0.2f0;
                 0.3f0 0f0 0.2f0;
                 0.5f0 0f0 0.2f0],

                [ 0.5f0 0f0 -0.2f0;
                  0.3f0 0f0 -0.2f0;
                 -0.2f0 0f0  0.0f0;
                 -0.2f0 0f0  0.5f0],
              ] .* road_tile_size

    # Template for each side of 4way intersection
    elseif startswith(kind, "4way")
        pts = [
                [-0.2f0 0f0 -0.5f0;
                 -0.2f0 0f0  0.0f0;
                  0.0f0 0f0  0.2f0;
                  0.5f0 0f0  0.2f0],

                [-0.2f0 0f0 -0.50f0;
                 -0.2f0 0f0 -0.25f0;
                 -0.2f0 0f0  0.25f0;
                 -0.2f0 0f0  0.50f0],

                [-0.2f0 0f0 -0.5f0;
                 -0.2f0 0f0 -0.2f0;
                 -0.3f0 0f0 -0.2f0;
                 -0.5f0 0f0 -0.2f0],
              ] .* road_tile_size
    else
        @assert false kind
    end

    # Rotate and align each curve with its place in global frame
    if startswith(kind, "4way")
        fourway_pts = []
        # Generate all four sides' curves,
        # with 3-points template above
        for rot in 0:3
            mat = gen_rot_matrix([0f0, 1f0, 0f0], rot * π / 2f0)
            pts_new = map(x -> x * mat, pts)
            add_vec = [(i - 0.5f0) * road_tile_size 0f0 (j - 0.5f0) * road_tile_size;]
            pts_new = map(x-> x .+ add_vec, pts_new)
            push!(fourway_pts, pts_new...)
        end
        return cat(fourway_pts..., dims=3)

    # Hardcoded each curve; just rotate and shift
    elseif startswith(kind, "3way")
        threeway_pts = []
        mat = gen_rot_matrix([0f0, 1f0, 0f0], angle * π / 2f0)
        #NOTE: pts is 3D matrix, find a work around if * does not work
        pts_new = map(x -> x * mat, pts)
        add_vec = [(i - 0.5f0) 0f0 (j - 0.5f0);] * road_tile_size
        pts_new = map(x -> x .+ add_vec, pts_new)
        push!(threeway_pts, pts_new...)

        return cat(threeway_pts..., dims=3)
    else
        mat = gen_rot_matrix([0f0, 1f0, 0f0], angle * π / 2f0)
        pts = map(x -> x * mat, pts)
        add_vec = [(i-0.5f0) 0f0 (j - 0.5f0);] * road_tile_size
        pts = map(x -> x .+ add_vec, pts)
    end

    return cat(pts..., dims=3)
end
#=
struct tile
    coords::Vector{Float32}
    texture
    color
    kind::String
    angle::Float32
    drivable::Bool
    curves
end
=#
struct Grid
    road_tile_size::Float32
    grid_width::Int
    grid_height::Int
    _grid::Vector
    obj_data::ObjectData
    road_vlist::Matrix{Float32}
    road_tlist::Matrix{Float32}
    ground_vlist::Matrix{Float32}
    drivable_tiles::Vector{Dict}
    mesh::ObjectMesh
    start_tile::Union{Nothing,Dict}
end


function _set_tile!(grid::Vector{Union{Missing,Dict{String,Any}}}, i::Int, j::Int,
                    width::Int, height::Int, tile::Dict{String,Any})
    @assert 1 ≤ i ≤ width
    @assert 1 ≤ j ≤ height
    grid[(j-1)*width + i] = tile
end

_get_tile(grid::Grid, i, j) = _get_tile(grid._grid, i, j, grid.grid_width, grid.grid_height)

function _get_tile(grid::Vector{Union{Missing,Dict{String,Any}}}, i::Int, j::Int, width::Int, height::Int)
    ##
    #Returns nothing if the duckiebot is not in a tile.
    ##
    if 1 ≤ i ≤ width && 1 ≤ j ≤ height
        return grid[(j-1)*width + i]
    end

    return nothing
end

function _init_vlists(road_tile_size)
    # Create the vertex list for our road quad
    # Note: the vertices are centered around the origin so we can easily
    # rotate the tiles about their center
    half_size = road_tile_size / 2f0
    verts = [
        -half_size 0f0 -half_size;
         half_size 0f0 -half_size;
         half_size 0f0  half_size;
        -half_size 0f0  half_size;
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

function Grid(map_data::Dict, domain_rand)
    if "tile_size" ∉ keys(map_data)
        msg = "Must now include explicit tile_size in the map data."
        throw(KeyError(msg))
    end

    road_tile_size = Float32(map_data["tile_size"])

    road_vlist, road_tlist, ground_vlist = _init_vlists(road_tile_size)

    tiles = map_data["tiles"]
    @assert length(tiles) > 0
    @assert length(tiles[1]) > 0

    # Create the grid
    grid_height = length(tiles)
    grid_width = length(tiles[1])
    _grid = Vector{Union{Missing, Dict{String,Any}}}(missing, grid_width * grid_height)

    # We keep a separate list of drivable tiles
    drivable_tiles = []

    # For each row in the grid
    for (j, row) ∈ enumerate(tiles)
        msg = "each row of tiles must have the same length"
        if length(row) != grid_width
            error(msg)
        end

        # For each tile in this row
        for (i, tile) in enumerate(row)
            tile = strip(tile)

            tile == "empty" && continue

            if '/' ∈ tile
                kind, orient = split(tile, '/')
                kind = strip(kind, ' ')
                orient = strip(orient, ' ')
                angle = findfirst(isequal(orient), ["S", "E", "N", "W"]) - 1
                drivable = true
            elseif '4' ∈ tile
                kind = "4way"
                angle = 2
                drivable = true
            else
                kind = tile
                angle = 0
                drivable = false
            end

            tile = Dict([
                "coords"=> [i, j],
                "kind"=> kind * "",
                "angle"=> angle,
                "drivable"=> drivable
            ])

            _set_tile!(_grid, i, j, grid_width, grid_height, tile)

            if drivable
                tile["curves"] = _get_curve(_grid, i, j, grid_width,
                                            grid_height, road_tile_size)
                push!(drivable_tiles, tile)
            end
        end
    end
    #TODO
    mesh = ObjMesh.get("duckiebot")
    obj_data = ObjectData(map_data, road_tile_size, domain_rand, _grid, grid_width, grid_height)

    # Get the starting tile from the map, if specified
    start_tile = nothing
    if "start_tile" ∈ keys(map_data)
        coords = map_data["start_tile"]
        coords .+= 1
        start_tile = _get_tile(_grid, coords..., grid_width, grid_height)
    end

    Grid(road_tile_size, grid_width, grid_height, _grid, obj_data, road_vlist,
         road_tlist, ground_vlist, drivable_tiles, mesh, start_tile)
end

_get_curve(grid::Grid, i, j) = _get_curve(grid._grid, i, j, grid.grid_width,
                                          grid.grid_height, grid.road_tile_size)

mutable struct Map
    map_name::String
    map_file_path::String
    map_data::Dict       # The parsed content of the map_file
    _grid::Grid
end

function Map(map_name::String, domain_rand::Bool)
    ##
    #Load the map layout from a YAML file
    ##
    map_file_path = get_file_path("src/maps", map_name, "yaml")
    #logger.debug('loading map file "%s"' % self.map_file_path)

    map_data = nothing
    open(map_file_path, "r") do f
        map_data = YAML.load(f)
    end

    _grid = Grid(map_data, domain_rand)

    Map(map_name, map_file_path, map_data, _grid)
end
