def step_ERK(F, R, a, b, c, tau, t, u):
    """
    Compute one step of an explicit Runge-Kutta scheme defined by a given Butcher tableau.

    Parameters
    ----------
    F : function(scalar, vector) -> vector
        Right hand side function.
    R : integer
        Number of stages.
    a : matrix
        Butcher tableau coefficients a[r][s] for the method, with 0 <= r,s <= R-1 and a[r][s] = 0 for s >= r.
    b : vector
        Butcher tableau coefficients b[r] for the method, with 0 <= r <= R-1.
    c : vector
        Butcher tableau coefficients c[r] for the method, with 0 <= r <= R-1 and c[0] = 0.
    tau : scalar
        Time step size.
    t : scalar
        Current time.
    u : vector
        Solution at time t.

    Returns
    -------
    t_new : scalar
        New time.
    u_new : vector
        Solution at time t_new.
    """

    t_new = t + tau 
    u_new = u.copy()
    K_rs = []
    for r in range(R):
        if r == 0:
            K_r = F(t, u) 
        else:
            t_r = t + tau*c[r] 
            u_r = u.copy()
            for s in range(r):
                u_r += tau*a[r][s]*K_rs[s] 
            K_r = F(t_r, u_r) 
        u_new += tau*b[r]*K_r 
        K_rs.append(K_r.copy())
    return t_new, u_new