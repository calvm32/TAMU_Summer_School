import numpy as np
from projectile_trajectory.compute_horizontal_distance import compute_horizontal_distance
from projectile_trajectory.find_max import find_max

def find_optimal_angle(s_0, tau, x_0 = 0, y_0 = 0, mu = 0, g = 9.80665):
    """
    Find the optimal launch angle for a projectile.

    Parameters
    ----------
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
        Acceleration due to gravity (meters / second**2)

    Returns
    -------
    optimal_angle : scalar
        Optimal launch angle (radians)
    max_horizontal_distance : scalar
        Maximum horizontal distance traveled (meters)
    """
    f = lambda u_left : compute_horizontal_distance(u_left, s_0, tau, x_0, y_0, mu, g) 

    a = 0 
    b = np.pi/2 
    optimal_angle, max_horizontal_distance = find_max(f,a,b) 
    return optimal_angle, max_horizontal_distance