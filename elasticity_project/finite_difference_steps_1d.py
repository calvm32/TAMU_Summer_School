import numpy as np
from scipy import integrate

def div(U, h, n, i, total_points):
    if i == 0:
        return (U[n,i+1] - U[n,i])/(h)
    elif i == total_points:
        return (U[n,i] - U[n,i-1])/(h)
    else:
        return (U[n,i+1] - U[n,i-1])/(2*h)

def lap(U, h, n, i, total_points):
    if i == 0 or i == total_points:
        return 0
    else:
        return (U[n,i+1] - 2*U[n,i] + U[n,i-1])/(h**2)

def post_processing(U_next, V_next, u_left, u_right, v_left, v_right, bc_type, h, ts, n, total_points):        
    
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

    #V-post processing
    # postprocessing
    V_int = integrate.simpson(V_next, dx=h)

    if V_int != 0:
        V_next -= V_int/total_points

    return U_next, V_next

# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================

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

        stability_termv = epsilon*lap(V,h,n,i,total_points)
        stability_termu = epsilon*lap(U,h,n,i,total_points)
        v_x = div(V,h,n,i,total_points)
        u_x = div(U,h,n,i,total_points)

        U_next[i] = U[n-1,i] + denominator*( -v_x + stability_termu )
        V_next[i] = V[n-1,i] + denominator*( -(c**2)*u_x + forcing + stability_termv )

    U_next, V_next = post_processing(U_next, V_next, u_left, u_right, v_left, v_right, bc_type, h, ts, n, total_points)

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

        stability_termv = epsilon*lap(V,h,n,i,total_points)
        stability_termu = epsilon*lap(U,h,n,i,total_points)
        v_x = div(V,h,n,i,total_points)
        u_x = div(U,h,n,i,total_points)

        U_next[i] = U[n-1,i] + denominator*( -v_x + stability_termu )
        V_next[i] = V[n-1,i] + denominator*( -(c**2)*u_x + forcing + stability_termv )

    U_next, V_next = post_processing(U_next, V_next, u_left, u_right, v_left, v_right, bc_type, h, ts, n, total_points)

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

        stability_termv = epsilon*lap(V,h,n,i,total_points)
        stability_termu = epsilon*lap(U,h,n,i,total_points)
        v_x = div(V,h,n,i,total_points)
        u_x = div(U,h,n,i,total_points)

        U_next[i] = U[n-1,i] + denominator*( -v_x + stability_termu )
        V_next[i] = V[n-1,i] + denominator*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + forcing + stability_termv )

    U_next, V_next = post_processing(U_next, V_next, u_left, u_right, v_left, v_right, bc_type, h, ts, n, total_points)

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

        stability_termv = epsilon*lap(V,h,n,i,total_points)
        stability_termu = epsilon*lap(U,h,n,i,total_points)
        v_x = div(V,h,n,i,total_points)
        u_x = div(U,h,n,i,total_points)

        U_next[i] = U[n-1,i] + denominator*( -v_x + stability_termu )
        V_next[i] = V[n-1,i] + denominator*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + forcing + stability_termv )

    U_next, V_next = post_processing(U_next, V_next, u_left, u_right, v_left, v_right, bc_type, h, ts, n, total_points)

    return U_next, V_next