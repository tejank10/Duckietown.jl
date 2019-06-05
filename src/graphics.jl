module Graphics
# List of textures available for a given path

using LinearAlgebra

include("utils.jl")

tex_paths = Dict()

# Cache of textures
tex_cache = Dict()

function get(tex_name::String, rng=nothing)
    ##
    #Manage the caching of textures, and texture randomization
    ##
    paths = Base.get(tex_paths, tex_name, [])

    # Get an inventory of the existing texture files
    if length(paths) == 0
        for i in 1:9
            path = get_file_path("src/textures", tex_name * "_$i", "png")
            !ispath(path) && break
            push!(paths, path)
        end
    end

    @assert length(paths) > 0 "failed to load textures for name " * tex_name

    if !isnothing(rng)
        path_idx = rand(rng, 1:length(paths))
        path = paths[path_idx]
    else
        path = paths[1]
    end

    if path ∉ keys(tex_cache)
        #TODO
        tex_cache[path] = []#Texture(load_texture(path))
    end

    return tex_cache[path]
end

function load_texture(color)
    PlainColor(color)
end
#=
function bind(tex::Texture)
        from pyglet import gl
        gl.glBindTexture(self.tex.target, self.tex.id)
end

function load_texture(tex_path)
    from pyglet import gl
    #logger.debug('loading texture "%s"' % os.path.basename(tex_path))
    import pyglet
    img = pyglet.image.load(tex_path)
    tex = img.get_texture()
    gl.glEnable(tex.target)
    gl.glBindTexture(tex.target, tex.id)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGB,
        img.width,
        img.height,
        0,
        gl.GL_RGBA,
        gl.GL_UNSIGNED_BYTE,
        img.get_image_data().get_data('RGBA', img.width * 4)
    )

    return tex

def create_frame_buffers(width, height, num_samples):
    """Create the frame buffer objects"""
    from pyglet import gl

    # Create a frame buffer (rendering target)
    multi_fbo = gl.GLuint(0)
    gl.glGenFramebuffers(1, byref(multi_fbo))
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, multi_fbo)

    # The try block here is because some OpenGL drivers
    # (Intel GPU drivers on macbooks in particular) do not
    # support multisampling on frame buffer objects
    try:
        # Create a multisampled texture to render into
        fbTex = gl.GLuint(0)
        gl.glGenTextures( 1, byref(fbTex))
        gl.glBindTexture(gl.GL_TEXTURE_2D_MULTISAMPLE, fbTex)
        gl.glTexImage2DMultisample(
            gl.GL_TEXTURE_2D_MULTISAMPLE,
            num_samples,
            gl.GL_RGBA32F,
            width,
            height,
            True
        )
        gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER,
                gl.GL_COLOR_ATTACHMENT0,
                gl.GL_TEXTURE_2D_MULTISAMPLE,
            fbTex,load_texture
            0
        )

        # Attach a multisampled depth buffer to the FBO
        depth_rb = gl.GLuint(0)
        gl.glGenRenderbuffers(1, byref(depth_rb))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rb)
        gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, num_samples, gl.GL_DEPTH_COMPONENT, width, height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rb)

    except:
        logger.debug('Falling back to non-multisampled frame buffer')

        # Create a plain texture texture to render into
        fbTex = gl.GLuint(0)
        gl.glGenTextures( 1, byref(fbTex))
        gl.glBindTexture(gl.GL_TEXTURE_2D, fbTex)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None
        )
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            fbTex,
            0
        )

        # Attach depth buffer to FBO
        depth_rb = gl.GLuint(0)
        gl.glGenRenderbuffers(1, byref(depth_rb))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rb)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, width, height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rb)

    # Sanity check
    import pyglet
    if pyglet.options['debug_gl']:
      res = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
      assert res == gl.GL_FRAMEBUFFER_COMPLETE

    # Create the frame buffer used to resolve the final render
    final_fbo = gl.GLuint(0)
    gl.glGenFramebuffers(1, byref(final_fbo))
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, final_fbo)

    # Create the texture used to resolve the final render
    fbTex = gl.GLuint(0)
    gl.glGenTextures(1, byref(fbTex))
    gl.glBindTexture(gl.GL_TEXTURE_2D, fbTex)
    gl.glTexImage2D(
        gl. GL_TEXTURE_2D,
        0,
        gl.GL_RGBA,
        width,
        height,
        0,
        gl. GL_RGBA,
        gl.GL_FLOAT,
        None
    )
    gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
        fbTex,
        0
    )
    import pyglet
    if pyglet.options['debug_gl']:
      res = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
      assert res == gl.GL_FRAMEBUFFER_COMPLETE

    # Enable depth testing
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Unbind the frame buffer
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    return multi_fbo, final_fbo
=#
function rotate_point(px, py, cx, cy, θ)
    ##
    #Rotate a 2D point around a center
    ##

    dx = px - cx
    dy = py - cy

    new_dx = dx * cos(θ) + dy * sin(θ)
    new_dy = dy * cos(θ) - dx * sin(θ)

    return [cx + new_dx, cy + new_dy]
end

function gen_rot_matrix(axis, θ)
    ##
    #Rotation matrix for a counterclockwise rotation around the given axis
    ##

    axis = axis / √dot(axis, axis)
    a = cos(θ/2)
    b, c, d = -axis * sin(θ/2)

    return [
            a^2+b^2-c^2-d^2  2(b*c-a*d)      2(b*d+a*c);
            2(b*c+a*d)       a^2+c^2-b^2-d^2 2(c*d-a*b);
            2(b*d-a*c)       2(c*d+a*b)      a^2+d^2-b^2-c^2]
end


function bezier_point(cps, t)
    ##
    #Cubic Bezier curve interpolation
    #B(t) = (1-t)^3 * P0 + 3t(1-t)^2 * P1 + 3t^2(1-t) * P2 + t^3 * P3
    ##

    p  = ((1-t)^3) * cps[1,:]
    p .+= 3t * ((1-t)^2) * cps[2,:]
    p .+= 3(t^2) * (1-t) * cps[3,:]
    p .+= (t^3) * cps[4,:]

    return p
end

function bezier_tangent(cps, t)
    ##
    #Tangent of a cubic Bezier curve (first order derivative)
    #B'(t) = 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2)
    ##

    p  = 3((1-t)^2) * (cps[2,:] - cps[1,:])
    p += 6(1-t) * t * (cps[3,:] - cps[2,:])
    p += 3(t^2) * (cps[4,:] - cps[3,:])

    p ./= norm(p)

    return p
end

function bezier_closest(cps, p, t_bot=0, t_top=1, n=8)
    mid = (t_bot + t_top) * 0.5

    n == 0 && (return mid)

    p_bot = bezier_point(cps, t_bot)
    p_top = bezier_point(cps, t_top)

    d_bot = norm(p_bot - p)
    d_top = norm(p_top - p)

    d_bot < d_top && return (bezier_closest(cps, p, t_bot, mid, n-1))

    return bezier_closest(cps, p, mid, t_top, n-1)
end

function bezier_draw(cps, n=20, red::Bool=false)
    pts = [bezier_point(cps, i/n) for i in -1:(n-2)]
    #gl.glBegin(gl.GL_LINE_STRIP)

    if red
    #    gl.glColor3f(1, 0, 0)
    else
    #    gl.glColor3f(0, 0, 1)
    end

    for (i, p) in enumerate(pts)
    #    gl.glVertex3f(*p)
    end

    #gl.glEnd()
    #gl.glColor3f(1,1,1)
end


export Texture, gen_rot_matrix, rotate_point,
       bezier_closest, bezier_point, bezier_tangent
end #module
