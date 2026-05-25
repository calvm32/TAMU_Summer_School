import numpy as np
from projectile_trajectory.step_ERK import step_ERK

def compute_trajectory(alpha, s_0, tau, x_0 = 0, y_0 = 0, mu = 0, g = 9.80665):
    """
    Compute the trajectory of a projectile.

    Parameters
    ----------
    alpha : scalar
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
    xs : list
        List of trajectory x coordinates.
    ys : list
        List of trajectory y coordinates.
    """

    # Butcher tableau for forward Euler
    R = 1 
    a = [[0]] 
    b = [1] 
    c = [0] 

    # Initial solution vector u = (x, y, v_x, v_y)
    v_x0 = s_0*np.cos(alpha) 
    v_y0 = s_0*np.sin(alpha) 
    u_0 = np.array([x_0, y_0, v_x0, v_y0]) 

    # Right hand side function
    def F(t, u):
        v_x = u[2] 
        v_y = u[3] 
        s = np.sqrt(v_x**2 + v_y**2)
        return np.array([v_x, v_y, -0.5*mu*s*v_x, -g-0.5*mu*s*v_y ]) 

    # Compute trajectory while y >= 0
    t = 0
    u = u_0
    xs = [x_0]
    ys = [y_0]
    t_new, u_new = step_ERK(F, R, a, b, c, tau, t, u) 
    y_new = u_new[1]
    while y_new >= 0:
        x_new = u_new[0]
        xs.append(x_new)
        ys.append(y_new)
        t = t_new
        u = u_new
        t_new, u_new = step_ERK(F, R, a, b, c, tau, t, u) 
        y_new = u_new[1]

    return xs, ys