module ObjMesh

include("graphics.jl")
include("utils.jl")

using RayTracer: Vec3, parse_mtllib!
using .Graphics

cache = Dict()

export ObjectMesh

mutable struct ObjectMesh
    ##
    #Load and render OBJ model files
    ##
    min_coords::Vector{Float32}
    max_coords::Vector{Float32}
    vlists::Vector{Array{Float32, 3}}
    texclists::Vector{Array{Float32, 3}}
    clists::Vector{Vector{Vec3}}
    textures::Vector{Union{Vec3, Nothing}}
end

function ObjectMesh(file_path::String)
    ##
    #Load an OBJ model file
    #
    #Limitations:
    #- only one object/group
    #- only triangle faces
    ##

    # Comments
    # mtllib file_name
    # o object_name
    # v x y z
    # vt u v
    # vn x y z
    # usemtl mtl_name
    # f v0/t0/n0 v1/t1/n1 v2/t2/n2

    #logger.debug('loading mesh "%s"' % os.path.basename(file_path))

    # Attempt to load the materials library
    materials = _load_mtl(file_path)
    mesh_file = open(file_path, "r")

    verts = Vector{Vector{Float32}}()
    texs = Vector{Vector{Float32}}()
    normals = Vector{Vector{Float32}}()
    faces = []

    cur_mtl = ""


    # For each line of the input file
    for line in eachline(mesh_file)
        line = rstrip(line, [' ', '\r', '\n'])

        # Skip comments
        (startswith(line, '#') || line == "") && continue

        tokens = split(line, " ")
        tokens = map(t -> strip(t, ' '), tokens)
        tokens = collect(filter(t -> t != "", tokens))

        prefix = tokens[1]
        tokens = tokens[2:end]

        if prefix == "v"
            vert = map(v->parse(Float32, v), tokens)
            push!(verts, vert)
        end

        if prefix == "vt"
            tc = map(v -> parse(Float32, v), tokens)
            push!(texs, tc)
        end

        if prefix == "vn"
            normal = map(v -> parse(Float32, v), tokens)
            push!(normals, normal)
        end

        if prefix == "usemtl"
            mtl_name = tokens[1]
            if mtl_name in keys(materials)
                cur_mtl = mtl_name
            else
                cur_mtl = ""
            end
        end

        if prefix == "f"
            @assert length(tokens) == 3 "only triangle faces are supported"

            face = Vector{Vector{Int}}()
            for token in tokens
                indices = filter(t -> t != "", split(token, '/'))
                indices = map(i -> parse(Int, i), indices)
                @assert length(indices) == 2 || length(indices) == 3
                push!(face, indices)
            end

            push!(faces, [face, cur_mtl])
        end
    end

    # Sort the faces by material name
    sort!(faces, by=f->f[2])
    # Compute the start and end faces for each chunk in the model
    cur_mtl = nothing
    chunks = []
    for (idx, face) in enumerate(faces)
        face, mtl_name = face
        if mtl_name != cur_mtl
            if length(chunks) > 0
                chunks[end]["end_idx"] = idx-1
            end
            push!(chunks, Dict([
                "mtl"=>materials[mtl_name],
                "start_idx"=>idx,
                "end_idx"=>nothing
            ]))
            cur_mtl = mtl_name
        end
    end

    chunks[end]["end_idx"] = length(faces)
    num_faces = length(faces)
    # logger.debug('num verts=%d' % len(verts))
    # logger.debug('num faces=%d' % num_faces)
    # logger.debug('num chunks=%d' % len(chunks))

    # Create arrays to store the vertex data
    list_verts = zeros(Float32, 3, 3, num_faces)
    list_norms = zeros(Float32, 3, 3, num_faces)
    list_texcs = zeros(Float32, 3, 2, num_faces)
    list_colors = Vector{Vec3}()

    # For each triangle
    for (f_idx, face) in enumerate(faces)
        face, mtl_name = face

        # Get the color for this face
        f_mtl = materials[mtl_name]
        #NOTE: May get some failure/error here because of else condition
        #f_color = !isempty(f_mtl) ? f_mtl.color_diffuse : rgb(1f0)
        f_color = f_mtl.color_diffuse
        # For each tuple of indices
        for (l_idx, indices) in enumerate(face)
            # Note: OBJ uses 1-based indexing
            # and texture coordinates are optional
            if length(indices) == 3
                v_idx, t_idx, n_idx = indices
                vert = verts[v_idx]
                texc = texs[t_idx]
                normal = normals[n_idx]
            else
                v_idx, n_idx = indices
                vert = verts[v_idx]
                normal = normals[n_idx]
                texc = [0f0, 0f0]
            end

            list_verts[l_idx, :, f_idx] .= vert
            list_texcs[l_idx, :, f_idx] .= texc
            list_norms[l_idx, :, f_idx] .= normal
            push!(list_colors, f_color)
        end
    end

    # Re-center the object so that the base is at y=0
    # and the object is centered in x and z
    min_coords = minimum(minimum(list_verts, dims=3), dims=1)
    max_coords = minimum(maximum(list_verts, dims=3), dims=1)

    mean_coords = (min_coords .+ max_coords) / 2f0
    #NOTE: Why is mean coords correct but min coords has error margin of ~0.02?
    #@show mean_coords
    min_y = min_coords[2]
    mean_x = mean_coords[1]
    mean_z = mean_coords[3]
    list_verts[:, 2, :] .= list_verts[:, 2, :] .- min_y
    list_verts[:, 1, :] .= list_verts[:, 1, :] .- mean_x
    list_verts[:, 3, :] .= list_verts[:, 3, :] .- mean_z

    # Recompute the object extents after centering
    min_coords = minimum(minimum(list_verts, dims=3), dims=1)[1, :, 1]
    max_coords = minimum(maximum(list_verts, dims=3), dims=1)[1, :, 1]

    # Vertex list, one per chunk
    vlists = Vector{Array{Float32, 3}}()

    # Texture Coordinates list corresponding to vlist
    texclists = Vector{Array{Float32, 3}}()

    # Color list
    clists = Vector{Vector{Vec3}}()

    # Textures, one per chunk
    textures = Vector{Union{Vec3, Nothing}}()

    # For each chunk
    for chunk in chunks
        start_idx = chunk["start_idx"]
        end_idx = chunk["end_idx"]
        num_faces_chunk = end_idx - start_idx + 1

        # Create a vertex list to be used for rendering

        #TODO
        #vlist = pyglet.graphics.vertex_list(
            #3num_faces_chunk,
            #("v3f", reshape(list_verts[:, :, start_idx:end_idx], :)),
            #("t2f", reshape(list_texcs[:, :, start_idx:end_idx], :)),
            #("n3f", reshape(list_norms[:, :, start_idx:end_idx], :)),
            #("c3f", reshape(list_color[:, :, start_idx:end_idx], :)))

        mtl = chunk["mtl"]

        texture = mtl.texture_diffuse

        push!(vlists, list_verts[:, :, start_idx:end_idx])
        push!(texclists, list_texcs[:, :, start_idx:end_idx])
        push!(clists, list_colors[start_idx:end_idx])
        push!(textures, texture)
    end

    ObjectMesh(min_coords, max_coords, vlists, texclists, clists, textures)
end

function get(mesh_name::String)
    ##
    #Load a mesh or used a cached version
    ##

    # Assemble the absolute path to the mesh file
    file_path = get_file_path("src/meshes", mesh_name, "obj")

    file_path ∈ keys(cache) && (return cache[file_path])

    mesh = ObjectMesh(file_path)
    cache[file_path] = mesh

    return mesh
end


function _load_mtl(model_file::String)
    texture_diffuse = nothing
    model_dir, file_name = splitdir(model_file) .* ""

    # Determine the default texture path for the default material
    tex_name = split(file_name, '.')[1] * ""
    tex_path = get_file_path("src/textures", tex_name, "png")

    if ispath(tex_path)
        texture_diffuse = load_texture(tex_path)
    end

    materials = Dict{String, Union{NamedTuple,Nothing}}([
        # This is default material
        ""=> (color_diffuse = Vec3([0f0], [0f0], [1f0]),
              color_ambient = Vec3(1.0f0),
              color_specular = Vec3(1.0f0),
              specular_exponent = 50.0f0,
              texture_ambient = nothing,
              texture_diffuse = texture_diffuse,
              texture_specular = nothing)
        ])

    mtl_path = model_file[1:end-4] * ".mtl"

    if ispath(mtl_path)
        parse_mtllib!(mtl_path, materials, Float32)
    end

    return materials
end

function render(obj_mesh::ObjectMesh)
    for (idx, vlist) in enumerate(obj_mesh.vlists)
        texture = obj_mesh.textures[idx]

        if !isnothing(texture)
            #gl.glEnable(gl.GL_TEXTURE_2D)
            #gl.glBindTexture(texture.target, texture.id)
        else
            #gl.glDisable(gl.GL_TEXTURE_2D)
        end

        #vlist.draw(gl.GL_TRIANGLES)
    end

    #gl.glDisable(gl.GL_TEXTURE_2D)
end

end #module
