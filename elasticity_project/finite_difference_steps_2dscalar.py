import numpy as np

def delx(U, h, n, i, j, total_xpoints):
    if i == 0:
        return (U[n,i+1,j] - U[n,i,j])/(h)
    elif i == total_xpoints:
        return (U[n,i,j] - U[n,i-1,j])/(h)
    else:
        return (U[n,i+1,j] - U[n,i-1,j])/(2*h)

def dely(U, h, n, i, j, total_ypoints):
    if j == 0:
        return (U[n,i,j+1] - U[n,i,j])/(h)
    elif j == total_ypoints:
        return (U[n,i,j] - U[n,i,j-1])/(h)
    else:
        return (U[n,i,j+1] - U[n,i,j-1])/(2*h)

def delxx(U, h, n, i, j, total_xpoints):
    if i == 0 or i == total_xpoints:
        return 0
    else:
        return (U[n,i+1,j] -2*U[n,i,j] + U[n,i-1,j])/(h**2)

def delyy(U, h, n, i, j, total_ypoints):
    if j == 0 or j == total_ypoints:
        return 0
    else:
        return (U[n,i,j+1] -2*U[n,i,j] + U[n,i,j-1])/(h**2)

def div(U, hx, hy, n, i, j, total_xpoints, total_ypoints):
    return delx(U,hx,n,i,j,total_xpoints) + dely(U,hy,n,i,j,total_ypoints)

def lap(U, hx, hy, n, i, j, total_xpoints, total_ypoints):
    return delxx(U,hx,n,i,j,total_xpoints) + delyy(U,hy,n,i,j,total_ypoints)

def post_processing(U_next, V_next, u_bcs, v_bcs, bc_type, hx, hy, ts, n):  
    u_left, u_right, u_top, u_bottom = u_bcs[0], u_bcs[1], u_bcs[2], u_bcs[3]
    v_left, v_right, v_top, v_bottom = v_bcs[0], v_bcs[1], v_bcs[2], v_bcs[3]

    if bc_type == "dirichlet":
        U_next[0,:] = u_left(ts[n])
        U_next[-1,:] = u_right(ts[n])    
        U_next[:,0] = u_bottom(ts[n])
        U_next[:,-1] = u_top(ts[n])      
    elif bc_type == "reflecting":
        U_next[0,:] = 0
        U_next[-1,:] = 0   
        U_next[:,0] = 0
        U_next[:,-1] = 0   
    elif bc_type == "neumann_right":
        U_next[0,:] = u_left(ts[n])
        U_next[-1,:] = hx*u_right(ts[n]) + U_next[-2,:]
        U_next[:,0] = u_bottom(ts[n])
        U_next[:,-1] = u_top(ts[n])
    elif bc_type == "neumann_left":
        U_next[0,:] = hx*u_left(ts[n]) + U_next[1,:]
        U_next[-1,:] = u_right(ts[n])
        U_next[:,0] = u_bottom(ts[n])
        U_next[:,-1] = u_top(ts[n])
    elif bc_type == "neumann":
        U_next[0,:] = hx*u_left(ts[n]) + U_next[1,:]
        U_next[-1,:] = hx*u_right(ts[n]) + U_next[-2,:]
        U_next[:,0] = hy*u_bottom(ts[n]) + U_next[:,1]
        U_next[:,-1] = hy*u_top(ts[n]) + U_next[:,-2]

    return U_next, V_next

# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================

def linear_center_diff_step(c, U, V, n, f, u_bcs, v_bcs, xs, ts, stabilization = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_xpoints = len(xs[:,0])-1
    total_ypoints = len(xs[0,:])-1

    # solve in all space, for one time
    U_next = np.zeros_like(xs)
    V_next = np.zeros_like(xs)

    hx = xs[0,1] - xs[0,0]
    hy = xs[1,0] - xs[0,0]
    tau = ts[1] - ts[0]
    for i in range(total_xpoints+1):
        for j in range(total_ypoints+1):
            denominator = 2*tau
            forcing = f(ts[n],xs[i,j])

            stability_termv = stabilization*lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints)
            stability_termu = stabilization*lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints)
            v_x = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints)
            u_x = div(U,hx,hy,n,i,j,total_xpoints,total_ypoints)

            U_next[i,j] = U[n-1,i,j] + denominator*( -v_x + stability_termu )
            V_next[i,j] = V[n-1,i,j] + denominator*( -(c**2)*u_x + forcing + stability_termv )

    U_next, V_next = post_processing(U_next, V_next, u_bcs, v_bcs, bc_type, hx, hy, ts, n)

    return U_next, V_next

def linear_forward_diff_step(c, U, V, n, f, u_bcs, v_bcs, xs, ts, stabilization = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_xpoints = len(xs[:,0])-1
    total_ypoints = len(xs[0,:])-1

    # solve in all space, for one time
    U_next = np.zeros_like(xs)
    V_next = np.zeros_like(xs)

    hx = xs[0,1] - xs[0,0]
    hy = xs[1,0] - xs[0,0]
    tau = ts[1] - ts[0]
    for i in range(total_xpoints+1):
        for j in range(total_ypoints+1):
            denominator = tau
            forcing = f(ts[n],xs[i,j])

            stability_termv = stabilization*lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints)
            stability_termu = stabilization*lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints)
            v_x = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints)
            u_x = div(U,hx,hy,n,i,j,total_xpoints,total_ypoints)

            U_next[i,j] = U[n-1,i,j] + denominator*( -v_x + stability_termu )
            V_next[i,j] = V[n-1,i,j] + denominator*( -(c**2)*u_x + forcing + stability_termv )

    U_next, V_next = post_processing(U_next, V_next, u_bcs, v_bcs, bc_type, hx, hy, ts, n)

    return U_next, V_next

# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================

def nonlinear_center_diff_step(c, U, V, n, f, u_bcs, v_bcs, xs, ts, stabilization = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_xpoints = len(xs[:,0])-1
    total_ypoints = len(xs[0,:])-1

    # solve in all space, for one time
    U_next = np.zeros_like(xs)
    V_next = np.zeros_like(xs)

    hx = xs[0,1] - xs[0,0]
    hy = xs[1,0] - xs[0,0]
    tau = ts[1] - ts[0]
    for i in range(total_xpoints+1):
        for j in range(total_ypoints+1):
            denominator = 2*tau
            forcing = f(ts[n],xs[i,j])

            stability_termv = stabilization*lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints)
            stability_termu = stabilization*lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints)
            v_x = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints)
            u_x = div(U,hx,hy,n,i,j,total_xpoints,total_ypoints)

            U_next[i,j] = U[n-1,i,j] + denominator*( -v_x + stability_termu )
            V_next[i,j] = V[n-1,i,j] + denominator*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + forcing + stability_termv )

    U_next, V_next = post_processing(U_next, V_next, u_bcs, v_bcs, bc_type, hx, hy, ts, n)

    return U_next, V_next

def nonlinear_forward_diff_step(c, U, V, n, f, u_bcs, v_bcs, xs, ts, stabilization = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_xpoints = len(xs[:,0])-1
    total_ypoints = len(xs[0,:])-1

    # solve in all space, for one time
    U_next = np.zeros_like(xs)
    V_next = np.zeros_like(xs)

    hx = xs[0,1] - xs[0,0]
    hy = xs[1,0] - xs[0,0]
    tau = ts[1] - ts[0]
    for i in range(total_xpoints+1):
        for j in range(total_ypoints+1):
            denominator = tau
            forcing = f(ts[n],xs[i,j])

            stability_termv = stabilization*lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints)
            stability_termu = stabilization*lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints)
            v_x = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints)
            u_x = div(U,hx,hy,n,i,j,total_xpoints,total_ypoints)

            U_next[i,j] = U[n-1,i,j] + denominator*( -v_x + stability_termu )
            V_next[i,j] = V[n-1,i,j] + denominator*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + forcing + stability_termv )

    U_next, V_next = post_processing(U_next, V_next, u_bcs, v_bcs, bc_type, hx, hy, ts, n)

    return U_next, V_next