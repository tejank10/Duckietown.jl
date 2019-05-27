add_axis(x) = reshape(x, size(x)..., 1)

function get_subdir_path(sub_dir::String)
    # Get the directory this module is located in
    abs_path_module = realpath(@__DIR__)
    module_dir, _ = splitdir(abs_path_module)

    joinpath(module_dir, sub_dir)
end


function get_file_path(sub_dir::String, file_name::String, default_ext::String)
    #=
    Get the absolute path of a resource file, which may be relative to
    the Duckietown.jl module directory, or an absolute path.

    This function is necessary because the simulator may be imported by
    other packages, and we need to be able to load resources no matter
    what the current working directory is.
    =#

    @assert '.' ∉ default_ext
    @assert '/' ∉ default_ext

    # If this is already a real path
    ispath(file_name) && (return file_name)

    subdir_path = get_subdir_path(sub_dir)
    file_path = joinpath(subdir_path, file_name)

    if '.' ∉ file_name
        file_path *= "." * default_ext
    end

    return file_path
end
