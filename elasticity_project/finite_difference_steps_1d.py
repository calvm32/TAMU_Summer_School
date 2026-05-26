import numpy as np

def linear_center_diff_step(c, U, V, n, f, u_left, u_right, v_left, v_right, xs, ts, epsilon = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_points = len(xs)-1

    # solve in all space, for one time
    U_next = np.zeros(total_points+1)
    V_next = np.zeros(total_points+1)

    h = xs[1] - xs[0]
    tau = ts[1] - ts[0]
    for i in range(total_points+1):
        denominator = 2*tau
        forcing = f(ts[n],xs[i])

        if i == 0:
            U_next[i] = U[n-1,i] + denominator*( -(V[n,i+1] - V[n,i])/h )
            V_next[i] = V[n-1,i] + denominator*( -(c**2)*(U[n,i+1] - U[n,i])/h + forcing )
        elif i == total_points:
            U_next[i] = U[n-1,i] + denominator*( -(V[n,i] - V[n,i-1])/h )
            V_next[i] = V[n-1,i] + denominator*( -(c**2)*(U[n,i] - U[n,i-1])/h + forcing )
        else:
            stability_termv = epsilon*(V[n,i+1] - 2*V[n,i] + V[n,i-1])
            stability_termu = epsilon*(U[n,i+1] - 2*U[n,i] + U[n,i-1])
            
            U_next[i] = U[n-1,i] + denominator*( -(V[n,i+1] - V[n,i-1])/(2*h) + stability_termu )
            V_next[i] = V[n-1,i] + denominator*( -(c**2)*(U[n,i+1] - U[n,i-1])/(2*h) + stability_termv + forcing )

        if bc_type == "dirichlet":
            U_next[0] = u_left(ts[n])
            U_next[-1] = u_right(ts[n])      
        elif bc_type == "reflecting":
            U_next[0] = 0
            U_next[-1] = 0   
        elif bc_type == "neumann_right":
            U_next[0] = u_left(ts[n])
            U_next[-1] = h*u_right(ts[n]) + U_next[-2]
        elif bc_type == "neumann_left":
            U_next[0] = h*u_left(ts[n]) + U_next[1]
            U_next[-1] = u_right(ts[n])
        elif bc_type == "neumann":
            U_next[0] = h*u_left(ts[n]) + U_next[1]
            U_next[-1] = h*u_right(ts[n]) + U_next[-2]

    return U_next, V_next

def linear_forward_diff_step(c, U, V, n, f, u_left, u_right, v_left, v_right, xs, ts, epsilon = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_points = len(xs)-1
    
    # solve in all space, for one time
    U_next = np.zeros(total_points+1)
    V_next = np.zeros(total_points+1)

    h = xs[1] - xs[0]
    tau = ts[1] - ts[0]
    for i in range(total_points+1):
        denominator = tau
        forcing = f(ts[n],xs[i])

        if i == 0:
            U_next[i] = U[n,i] + denominator*( -(V[n,i+1] - V[n,i])/h )
            V_next[i] = V[n,i] + denominator*( -(c**2)*(U[n,i+1] - U[n,i])/h + forcing )
        elif i == total_points:
            U_next[i] = U[n,i] + denominator*( -(V[n,i] - V[n,i-1])/h )
            V_next[i] = V[n,i] + denominator*( -(c**2)*(U[n,i] - U[n,i-1])/h + forcing )
        else:
            stability_termv = epsilon*(V[n,i+1] - 2*V[n,i] + V[n,i-1])
            stability_termu = epsilon*(U[n,i+1] - 2*U[n,i] + U[n,i-1])
            
            U_next[i] = U[n,i] + denominator*( -(V[n,i+1] - V[n,i-1])/(2*h) + stability_termu )
            V_next[i] = V[n,i] + denominator*( -(c**2)*(U[n,i+1] - U[n,i-1])/(2*h) + stability_termv + forcing )

        if bc_type == "dirichlet":
            U_next[0] = u_left(ts[n])
            U_next[-1] = u_right(ts[n])
        elif bc_type == "reflecting":
            U_next[0] = 0
            U_next[-1] = 0   
        elif bc_type == "neumann_right":
            U_next[0] = u_left(ts[n])
            U_next[-1] = h*u_right(ts[n]) + U_next[-2]
        elif bc_type == "neumann_left":
            U_next[0] = h*u_left(ts[n]) + U_next[1]
            U_next[-1] = u_right(ts[n])
        elif bc_type == "neumann":
            U_next[0] = h*u_left(ts[n]) + U_next[1]
            U_next[-1] = h*u_right(ts[n]) + U_next[-2]

    return U_next, V_next

# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================

def nonlinear_center_diff_step(c, U, V, n, f, u_left, u_right, v_left, v_right, xs, ts, epsilon = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_points = len(xs)-1
    
    # solve in all space, for one time
    U_next = np.zeros(total_points+1)
    V_next = np.zeros(total_points+1)

    h = xs[1] - xs[0]
    tau = ts[1] - ts[0]
    for i in range(total_points+1):
        denominator = 2*tau
        forcing = f(ts[n],xs[i])

        if i == 0:
            v_x = (V[n,i+1] - V[n,i])/h
            u_x = (U[n,i+1] - U[n,i])/h

            U_next[i] = U[n-1,i] + denominator*( -v_x )
            V_next[i] = V[n-1,i] + denominator*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + forcing )
        elif i == total_points:
            v_x = (V[n,i] - V[n,i-1])/h
            u_x = (U[n,i] - U[n,i-1])/h

            U_next[i] = U[n-1,i] + denominator*( -v_x )
            V_next[i] = V[n-1,i] + denominator*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + forcing )
        else:
            stability_termv = epsilon*(V[n,i+1] - 2*V[n,i] + V[n,i-1])
            stability_termu = epsilon*(U[n,i+1] - 2*U[n,i] + U[n,i-1])

            v_x = (V[n,i+1] - V[n,i-1])/(2*h)
            u_x = (U[n,i+1] - U[n,i-1])/(2*h)
            
            U_next[i] = U[n-1,i] + denominator*( -v_x + stability_termu )
            V_next[i] = V[n-1,i] + denominator*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + stability_termv + forcing )

        if bc_type == "dirichlet":
            U_next[0] = u_left(ts[n])
            U_next[-1] = u_right(ts[n])
        elif bc_type == "reflecting":
            U_next[0] = 0
            U_next[-1] = 0   
        elif bc_type == "neumann_right":
            U_next[0] = u_left(ts[n])
            U_next[-1] = h*u_right(ts[n]) + U_next[-2]
        elif bc_type == "neumann_left":
            U_next[0] = h*u_left(ts[n]) + U_next[1]
            U_next[-1] = u_right(ts[n])
        elif bc_type == "neumann":
            U_next[0] = h*u_left(ts[n]) + U_next[1]
            U_next[-1] = h*u_right(ts[n]) + U_next[-2]
        
        

    return U_next, V_next

def nonlinear_forward_diff_step(c, U, V, n, f, u_left, u_right, v_left, v_right, xs, ts, epsilon = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_points = len(xs)-1
    
    # solve in all space, for one time
    U_next = np.zeros(total_points+1)
    V_next = np.zeros(total_points+1)

    h = xs[1] - xs[0]
    tau = ts[1] - ts[0]
    for i in range(total_points+1):
        denominator = tau
        forcing = f(ts[n],xs[i])

        if i == 0:
            v_x = (V[n,i+1] - V[n,i])/h
            u_x = (U[n,i+1] - U[n,i])/h

            U_next[i] = U[n,i] + denominator*( -v_x )
            V_next[i] = V[n,i] + denominator*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + forcing )
        elif i == total_points:
            v_x = (V[n,i] - V[n,i-1])/h
            u_x = (U[n,i] - U[n,i-1])/h

            U_next[i] = U[n,i] + denominator*( -v_x )
            V_next[i] = V[n,i] + denominator*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + forcing )
        else:
            stability_termv = epsilon*(V[n,i+1] - 2*V[n,i] + V[n,i-1])
            stability_termu = epsilon*(U[n,i+1] - 2*U[n,i] + U[n,i-1])

            v_x = (V[n,i+1] - V[n,i-1])/(2*h)
            u_x = (U[n,i+1] - U[n,i-1])/(2*h)
            
            U_next[i] = U[n,i] + denominator*( -v_x + stability_termu )
            V_next[i] = V[n,i] + denominator*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + stability_termv + forcing )

        if bc_type == "dirichlet":
            U_next[0] = u_left(ts[n])
            U_next[-1] = u_right(ts[n])
        elif bc_type == "reflecting":
            U_next[0] = 0
            U_next[-1] = 0   
        elif bc_type == "neumann_right":
            U_next[0] = u_left(ts[n])
            U_next[-1] = h*u_right(ts[n]) + U_next[-2]
        elif bc_type == "neumann_left":
            U_next[0] = h*u_left(ts[n]) + U_next[1]
            U_next[-1] = u_right(ts[n])
        elif bc_type == "neumann":
            U_next[0] = h*u_left(ts[n]) + U_next[1]
            U_next[-1] = h*u_right(ts[n]) + U_next[-2]

    return U_next, V_next