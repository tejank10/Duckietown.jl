struct ObjectData
    objects::Vector
    object_corners
    collidable_centers
    collidable_corners
    collidable_norms
    collidable_safety_radii
end

function ObjectData(map_data::Dict, road_tile_size, domain_rand, grid)
    # Create the objects array
    objects = []

    # The corners for every object, regardless if collidable or not
    object_corners = []

    # Arrays for checking collisions with N static objects
    # (Dynamic objects done separately)
    # (N x 2): Object position used in calculating reward
    collidable_centers = []

    # (N x 2 x 4): 4 corners - (x, z) - for object's boundbox
    collidable_corners = []

    # (N x 2 x 2): two 2D norms for each object (1 per axis of boundbox)
    collidable_norms = []

    # (N): Safety radius for object used in calculating reward
    collidable_safety_radii = []

    # For each object
    for (obj_idx, desc) in enumerate(get(map_data, "objects", []))
        kind = desc["kind"]

        pos = desc["pos"]
        x, z = pos[1:2]
        y = length(pos) == 3 ? pos[3] : 0f0

        rotate = desc["rotate"]
        optional = get(desc, "optional", false)

        pos = road_tile_size * [x, y, z]

        # Load the mesh
        #TODO
        mesh = ObjMesh.get(kind)

        if "height" in keys(desc)
            scale = desc["height"] / mesh.max_coords[2]
        else
            sscale = desc["scale"]
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

        push!(objects, obj)

        # Compute collision detection information

        # angle = rotate * (math.pi / 180)

        # Find drivable tiles object could intersect with
        possible_tiles = find_candidate_tiles(_obj_corners(obj), road_tile_size)

        # If the object intersects with a drivable tile
        if static && kind != "trafficlight" && _collidable_object(
                grid, obj.obj_corners, obj.obj_norm, possible_tiles, road_tile_size)
            push!(collidable_centers, pos)
            push!(collidable_corners, permutedims(obj.obj_corners))
            push!(collidable_norms, obj.obj_norm)
            push!(collidable_safety_radii, obj.safety_radius)
        end
    end
    # If there are collidable objects
    if length(collidable_corners) > 0
        collidable_corners = vcat(collidable_corners...)
        collidable_norms = vcat(collidable_norms...)

        # Stack doesn't do anything if there's only one object,
        # So we add an extra dimension to avoid shape errors later
        if ndims(collidable_corners) == 2
            collidable_corners = add_axis(collidable_corners)
            collidable_norms = add_axis(collidable_norms)
        end
    end

    ObjectData(objects, object_corners, collidable_centers, collidable_corners,
               collidable_norms, collidable_safety_radii)
end


function _get_curve(grid, i, j, road_tile_size)
    ##
    #    Get the Bezier curve control points for a given tile
    ##
    tile = _get_tile(grid, i, j)
    @assert !isa(tile, Nothing)

    kind = tile["kind"]
    angle = tile["angle"]

    # Each tile will have a unique set of control points,
    # Corresponding to each of its possible turns

    if startswith(kind, "straight")
        pts = [
                [-0.2f0 0 -0.50f0;
                 -0.2f0 0 -0.25f0;
                 -0.2f0 0  0.25f0;
                 -0.2f0 0  0.50f0],

                [0.2f0 0  0.50f0;
                 0.2f0 0  0.25f0;
                 0.2f0 0 -0.25f0;
                 0.2f0 0 -0.50f0],
              ] .* road_tile_size

    elseif kind == "curve_left"
        pts = [
                [-0.2f0 0 -0.5f0;
                 -0.2f0 0  0.0f0;
                  0.0f0 0  0.2f0;
                  0.5f0 0  0.2f0],

                [0.5f0 0 -0.2f0;
                 0.3f0 0 -0.2f0;
                 0.2f0 0 -0.3f0;
                 0.2f0 0 -0.5f0],
              ] .* road_tile_size

    elseif kind == "curve_right"
        pts = [
                [-0.2f0 0 -0.5f0;
                 -0.2f0 0 -0.2f0;
                 -0.3f0 0 -0.2f0;
                 -0.5f0 0 -0.2f0],

                [-0.5f0 0  0.2f0;
                 -0.3f0 0  0.2f0;
                  0.3f0 0  0.0f0;
                  0.2f0 0 -0.5f0],
              ] .* road_tile_size

    # Hardcoded all curves for 3way intersection
    elseif startswith(kind, "3way")
        pts = [
                [-0.2f0 0 -0.50f0;
                 -0.2f0 0 -0.25f0;
                 -0.2f0 0  0.25f0;
                 -0.2f0 0  0.50f0],

                [-0.2f0 0 -0.5f0;
                 -0.2f0 0  0.0f0;
                  0.0f0 0  0.2f0;
                  0.5f0 0  0.2f0],

                [0.2f0 0  0.50f0;
                 0.2f0 0  0.25f0;
                 0.2f0 0 -0.25f0;
                 0.2f0 0 -0.50f0],

                [0.5f0 0 -0.2f0;
                 0.3f0 0 -0.2f0;
                 0.2f0 0 -0.2f0;
                 0.2f0 0 -0.5f0],

                [0.2f0 0 0.5f0;
                 0.2f0 0 0.2f0;
                 0.3f0 0 0.2f0;
                 0.5f0 0 0.2f0],

                [ 0.5f0 0 -0.2f0;
                  0.3f0 0 -0.2f0;
                 -0.2f0 0  0.0f0;
                 -0.2f0 0  0.5f0],
              ] .* road_tile_size

    # Template for each side of 4way intersection
    elseif startswith(kind, "4way")
        pts = [
                [-0.2f0 0 -0.5f0;
                 -0.2f0 0  0.0f0;
                  0.0f0 0  0.2f0;
                  0.5f0 0  0.2f0],

                [-0.2f0 0 -0.50f0;
                 -0.2f0 0 -0.25f0;
                 -0.2f0 0  0.25f0;
                 -0.2f0 0  0.50f0],

                [-0.2f0 0 -0.5f0;
                 -0.2f0 0 -0.2f0;
                 -0.3f0 0 -0.2f0;
                 -0.5f0 0 -0.2f0],
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
            mat = gen_rot_matrix([0, 1, 0], rot * π / 2)
            pts_new = map(x -> x * mat, pts)
            add_vec = [(i + 0.5) * road_tile_size 0 (j + 0.5) * road_tile_size;]
            pts_new = map(x-> x .+ add_vec, pts_new)
            push!(fourway_pts, pts_new...)
        end
        return cat(fourway_pts..., dims=3)

    # Hardcoded each curve; just rotate and shift
    elseif startswith(kind, "3way")
        threeway_pts = []
        mat = gen_rot_matrix([0, 1, 0], angle * π / 2)
        #NOTE: pts is 3D matrix, find a work around if * does not work
        pts_new = map(x -> x * mat, pts)
        add_vec = [(i + 0.5) * road_tile_size 0 (j + 0.5) * road_tile_size;]
        pts_new = map(x -> x .+ add_vec, pts_new)
        push!(threeway_pts, pts_new...)

        return cat(threeway_pts..., dims=3)
    else
        mat = gen_rot_matrix([0, 1, 0], angle * π / 2)
        pts = map(x -> x * mat, pts)
        add_vec = [(i + 0.5) * road_tile_size 0 (j + 0.5) * road_tile_size;]
        pts = map(x -> x .+ add_vec, pts)
    end

    return cat(pts...,dims=3)
end

struct Grid
    road_tile_size
    grid_width::Int
    grid_height::Int
    _grid::Matrix
    obj_data::ObjectData
    road_vlist
    road_tlist
    ground_vlist
    drivable_tiles
    mesh
    start_tile
end


function _set_tile!(grid, i, j, tile)
    grid_width, grid_height = size(grid)
    @assert 1 ≤ i ≤ grid_width
    @assert 1 ≤ j ≤ grid_height
    grid[(j-1) * grid_width + i] = tile
end


function _get_tile(grid, i, j)
    ##
    #Returns nothing if the duckiebot is not in a tile.
    ##
    grid_width, grid_height = size(grid)
    if 1 ≤ i ≤ grid_width && 1 ≤ j ≤ grid_height
        return grid[(j-1) * grid_width + i]
    end

    return nothing
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
    _grid = Matrix{Union{Missing, Dict{String,Any}}}(missing, grid_width, grid_height)

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
                angle = findfirst(isequal(orient), ["S", "E", "N", "W"])
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
                "coords"=> (i, j),
                "kind"=> kind * "",
                "angle"=> angle,
                "drivable"=> drivable
            ])

            _set_tile!(_grid, i, j, tile)

            if drivable
                tile["curves"] = _get_curve(_grid, i, j, road_tile_size)
                push!(drivable_tiles, tile)
            end
        end
    end
    #TODO
    mesh = ObjMesh.get("duckiebot")
    obj_data = ObjectData(map_data, road_tile_size, domain_rand, _grid)

    # Get the starting tile from the map, if specified
    start_tile = nothing
    if "start_tile" ∈ keys(map_data)
        coords = map_data["start_tile"]
        start_tile = _get_tile(_grid, coords...)
    end

    Grid(road_tile_size, grid_width, grid_height, _grid, obj_data, road_vlist,
         road_tlist, ground_vlist, drivable_tiles, mesh, start_tile)
end


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
