import numpy as np
import math
# Returns a 1D image of the 2D scene as seen from position 'pos' and direction 'dir'
# Uses ray tracing to find the target points.
# Use FOV=90deg for now.
def project( pos, dirs, shapelist, fov=90, resolution=100):
    # calculate max deviation as the lateral distance on a filmstrip placed a unit distance away.
    # RGBA space.
    col_img = np.zeros((resolution,4))
    depth_img = np.zeros((resolution,1))

    max_r = math.tan(((fov/2.0)*math.pi)/180.0);
    f_dir, r_dir = dirs;

    # Shoot rays.
    for k in zip(np.linspace( -max_r, max_r, resolution ), range(0, resolution)):
        direction = (f_dir + k[0] * r_dir)
        direction = direction / np.linalg.norm(direction)
        ray = (pos, direction )

        min_depth = 1000;
        this_color = np.zeros((4,))
        for shape in shapelist:
            hit, color, depth, _ = shape.intersect( ray )
            if hit and depth < min_depth:
                min_depth = depth
                this_color = color

        col_img[k[1]] = this_color
        depth_img[k[1]] = min_depth

    return (col_img, depth_img)