module ObjMesh

include("utils.jl")

cache = Dict()

mutable struct ObjectMesh
    ##
    #Load and render OBJ model files
    ##
    min_coords
    max_coords
    vlists
    textures
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

    verts = []
    texs = []
    normals = []
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

            face = []
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
                chunks[end]["end_idx"] = idx
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

    # Create numpy arrays to store the vertex data
    list_verts = zeros(Float32, 3, 3, num_faces)
    list_norms = zeros(Float32, 3, 3, num_faces)
    list_texcs = zeros(Float32, 3, 2, num_faces)
    list_color = zeros(Float32, 3, 3, num_faces)

    # For each triangle
    for (f_idx, face) in enumerate(faces)
        face, mtl_name = face

        # Get the color for this face
        f_mtl = materials[mtl_name]
        f_color = !isempty(f_mtl) ? f_mtl["Kd"] : [1,1,1]

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
                texc = [0, 0]
            end

            list_verts[l_idx, :, f_idx] .= vert
            list_texcs[l_idx, :, f_idx] .= texc
            list_norms[l_idx, :, f_idx] .= normal
            list_color[l_idx, :, f_idx] .= f_color
        end
    end

    # Re-center the object so that the base is at y=0
    # and the object is centered in x and z
    min_coords = minimum(minimum(list_verts, dims=3)[:, :, 1], dims=1)[1, :]
    max_coords = minimum(maximum(list_verts, dims=3)[:, :, 1], dims=1)[1, :]

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
    min_coords = minimum(minimum(list_verts, dims=3)[:, :, 1], dims=1)[1, :]
    max_coords = minimum(maximum(list_verts, dims=3)[:, :, 1], dims=1)[1, :]
    #@show min_coords, "mc"
    # Vertex lists, one per chunk
    vlists = []

    # Textures, one per chunk
    textures = []

    # For each chunk
    for chunk in chunks
        start_idx = chunk["start_idx"]
        end_idx = chunk["end_idx"]
        num_faces_chunk = end_idx - start_idx

        # Create a vertex list to be used for rendering

        #TODO
        vlist = []#= pyglet.graphics.vertex_list(
            3num_faces_chunk,
            ("v3f", reshape(list_verts[:, :, start_idx:end_idx], :)),
            ("t2f", reshape(list_texcs[:, :, start_idx:end_idx], :)),
            ("n3f", reshape(list_norms[:, :, start_idx:end_idx], :)),
            ("c3f", reshape(list_color[:, :, start_idx:end_idx], :))
        )=#

        mtl = chunk["mtl"]

        texture = "map_Kd" ∈ keys(mtl) ? load_texture(mtl["map_Kd"]) : nothing

        push!(vlists, vlist)
        push!(textures, texture)
    end

    ObjectMesh(min_coords, max_coords, vlists, textures)
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
    model_dir, file_name = splitdir(model_file) .* ""

    # Create a default material for the model
    default_mtl = Dict([
        "Kd"=> [1, 1, 1]
    ])

    # Determine the default texture path for the default material
    tex_name = split(file_name, '.')[1] * ""
    tex_path = get_file_path("textures", tex_name, "png")
    if isdir(tex_path)
        default_mtl["map_Kd"] = tex_path
    end

    materials = Dict([
        ""=> default_mtl
    ])

    mtl_path = split(model_file, '.')[1] * ".mtl"

    !isdir(mtl_path) && (return materials)

    #logger.debug('loading materials from "%s"' % mtl_path)

    mtl_file = open(mtl_path, 'r')

    cur_mtl = nothing

    # For each line of the input file
    for line in mtl_file
        line = rstrip(line, " \r\n")

        # Skip comments
        (startswith(line, '#') || line == "") && continue

        tokens = split(line, ' ')
        tokens = map(t -> strip(t, ' '), tokens)
        tokens = collect(filters(t -> t != "", tokens))

        prefix = tokens[1]
        tokens = tokens[2:end]

        if prefix == "newmtl"
            cur_mtl = Dict()
            materials[tokens[1]] = cur_mtl
        end

        # Diffuse color
        if prefix == "Kd"
            vals = map(v -> float(v), tokens)
            cur_mtl["Kd"] = vals
        end

        # Texture file name
        if prefix == "map_Kd"
            tex_file = tokens[end]
            tex_file = join(model_dir, tex_file)
            cur_mtl["map_Kd"] = tex_file
        end
    end

    close(mtl_file)

    return materials
end


function render(obj_mesh::ObjectMesh) end
#=
def render(self):
    from pyglet import gl
    for idx, vlist in enumerate(self.vlists):
        texture = self.textures[idx]

        if texture:
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glBindTexture(texture.target, texture.id)
        else:
            gl.glDisable(gl.GL_TEXTURE_2D)

        vlist.draw(gl.GL_TRIANGLES)

    gl.glDisable(gl.GL_TEXTURE_2D)
=#
end #module
