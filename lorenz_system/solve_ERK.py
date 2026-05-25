import numpy as np
from projectile_trajectory.step_ERK import step_ERK

def solve_ERK(F, u_0, T, N, R, a, b, c):
    """
    Solve a first order ODE u' = F(t,u) on a time interval [0,T] with initial condition u(0) = u_0 using an explicit Runge-Kutta method.

    Parameters
    ----------
    F : function(scalar, vector) -> vector
        Right hand side function
    u_0 : vector
        Initial condition
    T : scalar
        Final time
    N : integer
        Number of time steps
    R : integer
        Number of Runge-Kutta stages
    a : matrix
        Butcher tableau coefficients a[r][s] for the method, with 0 <= r,s <= R-1 and a[r][s] = 0 for s >= r.
    b : vector
        Butcher tableau coefficients b[r] for the method, with 0 <= r <= R-1.
    c : vector
        Butcher tableau coefficients c[r] for the method, with 0 <= r <= R-1 and c[0] = 0.

    Returns
    -------
    ts : vector
        List of discrete time points [0, tau, 2 * tau, ..., T = N * tau]
    us : matrix
        us[n,:] is the solution vector at time ts[n]
    """

    tau = T/N 
    ts = np.zeros(N+1)
    us = np.zeros((N+1,len(u_0)))
    us[0,:] = u_0 
    u = u_0.copy()
    t = 0
    for n in range(1,N+1):
        t_new, u_new = step_ERK(F, R, a, b, c, tau, t, u) 
        us[n,:] = u_new.copy()
        ts[n] = t_new
        u = u_new.copy()
        t = t_new
    return ts, us