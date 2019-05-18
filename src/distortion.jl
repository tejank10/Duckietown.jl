#TODO: OpenCV functions
mutable struct Distortion
    camera_matrix::Matrix
    distortion_coefs::Matrix
    rectification_matrix::Matrix
    projection_matrix::Matrix
    mapx
    mapy
    rmapx
    rmapy
end

function Distortion()

    camera_matrix = [
        305.5718893575089 0 303.0797142544728;
        0 308.8338858195428 231.8845403702499;
        0 0 1
    ]

    # distortion parameters - (k1, k2, t1, t2, k3)
    distortion_coefs = [
        -0.2, 0.0305,
        0.0005859930422629722, -0.0006697840226199427, 0
    ]

    distortion_coefs = reshape(distortion_coefs, (1, 5))

    # R - Rectification matrix - stereo cameras only, so identity
    rectification_matrix = eye(3)

    # P - Projection Matrix - specifies the intrinsic (camera) matrix
    #  of the processed (rectified) image
    projection_matrix = [
        220.2460277141687, 0, 301.8668918355899, 0;
        0, 238.6758484095299, 227.0880056118307, 0;
        0, 0, 1, 0
    ]

    # Initialize mappings

    # Used for rectification
    mapx = nothing
    mapy = nothing

    # Used for distortion
    rmapx = nothing
    rmapy = nothing

    Distortion(camera_matrix, distortion_coefs, rectification_matrix,
               projection_matrix, mapx, mapy, rmapx, rmapy)
end

function distort(dist::Distorion, observation)
    ##
    #Distort observation using parameters in constructor
    ##

    if isa(dist.mapx, Nothing)
        # Not initialized - initialize all the transformations we'll need
        dist.mapx = zeros(size(observation))
        dist.mapy = zeros(size(observation))

        H, W, _ = size(observation)

        # Initialize dist.mapx and dist.mapy (updated)
        dist.mapx, dist.mapy = cv2.initUndistortRectifyMap(dist.camera_matrix,
                                                           dist.distortion_coefs, dist.rectification_matrix,
                                                           dist.projection_matrix, (W, H), cv2.CV_32FC1)

        # Invert the transformations for the distortion
        dist.rmapx, dist.rmapy = _invert_map(dist, dist.mapx, dist.mapy)
    end

    return cv2.remap(observation, self.rmapx, self.rmapy, interpolation=cv2.INTER_NEAREST)
end

function _undistort(dist::Distortion, observation)
    ##
    #Undistorts a distorted image using camera parameters
    ##

    # If mapx is None, then distort was never called
    @assert !isa(dist.mapx, Nothing) "You cannot call undistort on a rectified image"

    return cv2.remap(observation, self.mapx, self.mapy, cv2.INTER_NEAREST)
end

function _invert_map(self, mapx, mapy):
    ##
    #Utility function for simulating distortion
    #Source: https://github.com/duckietown/Software/blob/master18/catkin_ws
    #... /src/10-lane-control/ground_projection/include/ground_projection/
    #... ground_projection_geometry.py
    ##

    H, W, _ = size(mapx)
    rmapx = similar(mapx)
    fill!(rmapx, NaN)
    rmapy = similar(mapx)
    fill!(rmapy, NaN)

    for y in 1:H, x in 1:W
        tx = mapx[y, x]
        ty = mapy[y, x]

        tx = Int(round(tx))
        ty = int(round(ty))

        if (1 ≤ tx ≤ W) && (1 ≤ ty ≤ H)
            rmapx[ty, tx] = x
            rmapy[ty, tx] = y
        end
    end
    _fill_holes(dist, rmapx, rmapy)
    return rmapx, rmapy
end

function _fill_holes(dist::Distortion, rmapx, rmapy)
    ##
    #Utility function for simulating distortion
    #Source: https://github.com/duckietown/Software/blob/master18/catkin_ws
    #... /src/10-lane-control/ground_projection/include/ground_projection/
    #... ground_projection_geometry.py
    ##
    H, W, _ = size(rmapx)

    R = 2
    F = R * 2 + 1

    norm(_) = hypot(_[1], _[2])

    deltas0 = [(i - R - 1, j - R - 1) for i in 1:F, j in 1:F]
    deltas0 = [x for x in deltas0 if norm(x) ≤ R]
    sort(deltas0, by=norm)

    holes = Set()

    for i = 1:H, j in 1:W
        if isnan(rmapx[i, j])
            union!(holes, [(i, j)])
        end
    end

    while !isempty(holes)
        nholes = length(holes)
        nholes_filled = 0

        for (i, j) in holes
            # there is nan
            nholes += 1
            for di, dj in deltas0
                u = i + di
                v = j + dj
                if (1 ≤ u ≤ H) && (1 ≤ v ≤ W)
                    if !isnan(rmapx[u, v])
                        rmapx[i, j] = rmapx[u, v]
                        rmapy[i, j] = rmapy[u, v]
                        nholes_filled += 1
                        pop!(holes, (i, j))
                        break
                    end
                end
            end
        end

        nholes_filled == 0 && break
    end
end
