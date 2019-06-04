using LinearAlgebra
using Statistics

function agent_boundbox(true_pos, width, length, f_vec, r_vec)
    ##
    #Compute bounding box for agent using its dimensions,
    #current position, and angle of rotation
    #Order of points in bounding box:
    #(front)
    #4 - 3
    #|   |
    #1 - 2
    ##

    # halfwidth/length
    hwidth = width / 2
    hlength = length / 2

    # Indexing to make sure we only get the x/z dims
    corners = hcat(
        true_pos .- hwidth .* r_vec .- hlength .* f_vec,
        true_pos .+ hwidth .* r_vec .- hlength .* f_vec,
        true_pos .+ hwidth .* r_vec .+ hlength .* f_vec,
        true_pos .- hwidth .* r_vec .+ hlength .* f_vec
    )[[1, 3], :]
end

function tensor_sat_test(norm::Vector{Matrix{T}}, corners::Vector{Matrix{T}}) where T
    ##
    #Separating Axis Theorem (SAT) extended to >2D.
    #(each input ~ "a list of 2D matrices")
    ##
    dotval = norm .* corners
    mins = minimum.(dotval, dims=2)
    maxs = maximum.(dotval, dims=2)

    return mins, maxs
end

tensor_sat_test(norm::Matrix{T}, corners::Matrix{T}) where T = tensor_sat_test([norm], [corners])
tensor_sat_test(norm::Matrix{T}, corners::Vector{Matrix{T}}) where T = tensor_sat_test([norm], corners)
tensor_sat_test(norm::Vector{Matrix{T}}, corners::Matrix{T}) where T = tensor_sat_test(norm, [corners])

function overlaps(min1, max1, min2, max2)
    ##
    #Helper function to check projection intervals (SAT)
    ##
    return is_between_ordered(min2, min1, max1) || is_between_ordered(min1, min2, max2)
end

function is_between_ordered(val, lowerbound, upperbound)
    ##
    #Helper function to check projection intervals (SAT)
    ##
    return all(lowerbound .≤ val .≤ upperbound)
end

function generate_corners(pos, min_coords, max_coords, θ, scale)
    ##
    #Generates corners given obj pos, extents, scale, and rotation
    ##
    px = pos[1]
    pz = pos[end]
    return permutedims(hcat(
        rotate_point(min_coords[1] * scale + px, min_coords[end] * scale + pz, px, pz, θ),
        rotate_point(max_coords[1] * scale + px, min_coords[end] * scale + pz, px, pz, θ),
        rotate_point(max_coords[1] * scale + px, max_coords[end] * scale + pz, px, pz, θ),
        rotate_point(min_coords[1] * scale + px, max_coords[end] * scale + pz, px, pz, θ)
    ))
end


function tile_corners(pos, width)
    ##
    #Generates the absolute corner coord for a tile, given grid pos and tile width
    ##
    px = pos[1]
    pz = pos[end]

    return [
        px * width - width pz * width - width;
        px * width + width pz * width - width;
        px * width + width pz * width + width;
        px * width - width pz * width + width
    ]
end


function generate_norm(corners)
    ##
    #Generates both (orthogonal, 1 per axis) normal vectors
    #for rectangle given vertices *in a particular order* (see generate_corners)
    ##
    ca = cov(corners, corrected=false)
    vect = eigen(ca).vectors
    return permutedims(vect)
end


function find_candidate_tiles(obj_corners, tile_size)
    ##
    #Finds all of the tiles that a object could intersect with
    #Returns the norms and corners of any of those that are drivable
    ##

    # Find min / max x&y tile coordinates of object
    minx, miny = Int.(floor.(
            minimum(obj_corners, dims=1) / tile_size
    ))

    maxx, maxy = Int.(floor.(
            maximum(obj_corners, dims=1) / tile_size
    ))

    # The max number of tiles we need to check is every possible
    # combination of x and y within the ranges, so enumerate
    xr = collect(minx:maxx)
    yr = collect(miny:maxy)

    possible_tiles = [(x+1, y+1) for x in xr for y in yr]
    return possible_tiles
end

function intersects(duckie::Matrix{T}, objs_stacked, duckie_norm::Matrix{T}, norms_stacked) where T
    ##
    #Helper function for Tensor-based OBB intersection.
    #Variable naming: SAT requires checking of the projection of all normals
    #to all sides, which is where we use tensor_sat_test (gives the mins and maxs)
    #of each projection pair. The variables are named as:
    #{x's norm + projected on + min/max}.
    ##
    duckduck_min, duckduck_max = tensor_sat_test(duckie_norm, permutedims(duckie))
    objduck_min, objduck_max = tensor_sat_test(duckie_norm, objs_stacked)
    duckobj_min, duckobj_max = tensor_sat_test(norms_stacked, permutedims(duckie))
    objobj_min, objobj_max = tensor_sat_test(norms_stacked, objs_stacked)

    # Iterate through each object we are checking against
    for idx in 1:length(objduck_min)
        # If any interval doesn't overlap, immediately know objects don't intersect
        if !overlaps(
                duckduck_min[1], duckduck_max[1], objduck_min[idx][1], objduck_max[idx][1])
            continue
        end
        if !overlaps(
                duckduck_min[2], duckduck_max[2], objduck_min[idx][2], objduck_max[idx][2])
            continue
        end
        if !overlaps(
                duckobj_min[idx][1], duckobj_max[idx][1], objobj_min[idx][1], objobj_max[idx][1])
            continue
        end
        if !overlaps(
                duckobj_min[idx][2], duckobj_max[idx][2], objobj_min[idx][2], objobj_max[idx][2])
            continue
        end
        # All projection intervals overlap, collision with an object
        return true
    end
    return false
end

function intersects_single_obj(duckie, obj, duckie_norm, norm)
    ##
    #Helper function for Single Object OBB intersection.
    #Variable naming: SAT requires checking of the projection of all normals
    #to all sides, which is where we use tensor_sat_test (gives the mins and maxs)
    #of each projection pair. The variables are named as:
    #{x's norm + projected on + min/max}.
    ##
    duckduck_min, duckduck_max = tensor_sat_test(duckie_norm, permutedims(duckie))
    objduck_min, objduck_max = tensor_sat_test(duckie_norm, obj)
    duckobj_min, duckobj_max = tensor_sat_test(norm, permutedims(duckie))
    objobj_min, objobj_max = tensor_sat_test(norm, obj)

    # If any interval doesn't overlap, immediately know objects don't intersect
    if !overlaps(
            duckduck_min[1], duckduck_max[1], objduck_min[1], objduck_max[1])
        return false
    end
    if !overlaps(
            duckduck_min[2], duckduck_max[2], objduck_min[2], objduck_max[2])
        return false
    end
    if !overlaps(
            duckobj_min[1], duckobj_max[1], objobj_min[1], objobj_max[1])
        return false
    end
    if !overlaps(
            duckobj_min[2], duckobj_max[2], objobj_min[2], objobj_max[2])
        return false
    end

    # All projection intervals overlap, collision with an object
    return true
end


function safety_circle_intersection(d, r1, r2)
    #=
    Checks if  two circles with centers separated by d and centered
    at r1 and r2 either intesect or are enveloped (one inside of other)
    =#
    intersect = ((r1 .- r2) .^ 2 .≤ d ^ 2) .& (d ^ 2 .≤ (r1 + r2) .^ 2)

    enveloped = d .< abs(r1 .- r2)

    return any(intersect) || any(enveloped)
end


function safety_circle_overlap(d, r1, r2)
    ##
    #Returns a proxy for area (see issue #24)
    #of two circles with centers separated by d
    #and centered at r1 and r2
    ##
    scores = d .- r1 .- r2
    return sum(scores[findall(s->s < 0, scores)])
end

function calculate_safety_radius(mesh, scale)
    ##
    #Returns a safety radius for an object, and scales
    #it according to the YAML file's scale param for that obj
    ##
    x, _, z = maximum(abs.([mesh.min_coords mesh.max_coords]), dims=2)
    return norm([x, z]) * scale
end

function heading_vec(θ)
    ##
    #Vector pointing in the direction the agent is looking
    ##

    x = cos(θ)
    z = -sin(θ)
    return [x, 0, z]
end
