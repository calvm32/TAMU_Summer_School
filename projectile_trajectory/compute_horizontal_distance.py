import numpy as np
from projectile_trajectory.compute_trajectory import compute_trajectory

def compute_horizontal_distance(u_left, s_0, tau, x_0 = 0, y_0 = 0, mu = 0, g = 9.80665):
    """
    Compute the horizontal distance traveled by a projectile.

    Parameters
    ----------
    u_left : scalar
        Launch angle (radians)
    s_0 : scalar
        Initial speed (meters / second)
    tau : scalar
        Time step size (seconds)
    x_0 : scalar
        Initial x position (meters)
    y_0 : scalar
        Initial y position (meters)
    mu : scalar
        Drag coefficient (1 / meters)
    g : scalar
        Accelerationg due to gravity (meters / second**2)

    Returns
    -------
    horizontal_distance_traveled : scalar
        Horizontal distance traveled (meters)

    """

    xs, ys = compute_trajectory(u_left, s_0, tau, x_0, y_0, mu, g) 
    horizontal_distance_traveled = np.sum(xs) 
    return horizontal_distance_traveled