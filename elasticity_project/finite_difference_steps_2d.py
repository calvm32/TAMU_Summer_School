import numpy as np

def kappa(gradu, graduT, mu=0.5, lambd=0, linear=False):
    if linear == True:
        strain_tensor = 0.5*(gradu + graduT)
    else:
        strain_tensor = 0.5*(gradu + graduT + graduT@gradu)

    return 2*mu*strain_tensor + lambd*np.trace(strain_tensor)*np.identity(2)

def delx(U, h, n, i, j, total_xpoints, where="interior", pos1=0, pos2=0):
    if U[0,0,0].shape == (2,):
        if where == "bdy":
            if i == 0:
                return (U[n,i+1,j, pos1] - U[n,i,j, pos1])/(h)
            elif i == total_xpoints:
                return (U[n,i,j, pos1] - U[n,i-1,j, pos1])/(h)
            else:
                return (U[n,i+1,j, pos1] - U[n,i-1,j, pos1])/(2*h)
        
        else:
            return (U[n,i+1,j, pos1] - U[n,i-1,j, pos1])/(2*h)

    elif U[0,0,0].shape == (2,2):
        if where == "bdy":
            if i == 0:
                return (U[n,i+1,j, pos1, pos2] - U[n,i,j, pos1, pos2])/(h)
            elif i == total_xpoints:
                return (U[n,i,j, pos1, pos2] - U[n,i-1,j, pos1, pos2])/(h)
            else:
                return (U[n,i+1,j, pos1, pos2] - U[n,i-1,j, pos1, pos2])/(2*h)
        
        else:
            return (U[n,i+1,j, pos1, pos2] - U[n,i-1,j, pos1, pos2])/(2*h)

def dely(U, h, n, i, j, total_ypoints, where="interior", pos1=0, pos2=0):
    if U[0,0,0].shape == (2,):
        if where == "bdy":
            if j == 0:
                return (U[n,i,j+1, pos1] - U[n,i,j, pos1])/(h)
            elif j == total_ypoints:
                return (U[n,i,j, pos1] - U[n,i,j-1, pos1])/(h)
            else:
                return (U[n,i,j+1, pos1] - U[n,i,j-1, pos1])/(2*h)
        else:
            return (U[n,i,j+1, pos1] - U[n,i,j-1, pos1])/(2*h)

    elif U[0,0,0].shape == (2,2):
        if where == "bdy":
            if j == 0:
                return (U[n,i,j+1, pos1, pos2] - U[n,i,j, pos1, pos2])/(h)
            elif j == total_ypoints:
                return (U[n,i,j, pos1, pos2] - U[n,i,j-1, pos1, pos2])/(h)
            else:
                return (U[n,i,j+1, pos1, pos2] - U[n,i,j-1, pos1, pos2])/(2*h)
        else:
            return (U[n,i,j+1, pos1, pos2] - U[n,i,j-1, pos1, pos2])/(2*h)

def h2delxx(U, h, n, i, j, total_xpoints, where="interior", pos1=0, pos2=0):
    if where == "bdy":
        return 0
    else:
        if U[0,0,0].shape == (2,):
            return (U[n,i+1,j, pos1] -2*U[n,i,j, pos1] + U[n,i-1,j, pos1])
        elif U[0,0,0].shape == (2,2):
            return (U[n,i+1,j, pos1, pos2] -2*U[n,i,j, pos1, pos2] + U[n,i-1,j, pos1, pos2])

def h2delyy(U, h, n, i, j, total_ypoints, where="interior", pos1=0, pos2=0):
    if where == "bdy":
        return 0
    else:
        if U[0,0,0].shape == (2,):
            return (U[n,i,j+1, pos1] -2*U[n,i,j, pos1] + U[n,i,j-1, pos1])
        elif U[0,0,0].shape == (2,2):
            return (U[n,i,j+1, pos1, pos2] -2*U[n,i,j, pos1, pos2] + U[n,i,j-1, pos1, pos2])

def div(U, hx, hy, n, i, j, total_xpoints, total_ypoints, where="interior"):
    if U[0,0,0].shape == (2,): # div of vector
        return delx(U,hx,n,i,j,total_xpoints, where, 0) + dely(U,hy,n,i,j,total_ypoints, where, 1)
    elif U[0,0,0].shape == (2,2): # div of tensor
        return np.array([delx(U,hx,n,i,j,total_xpoints, where, 0, 0) + dely(U, hy,n,i,j,total_ypoints, where, 1, 0), 
                delx(U,hx,n,i,j,total_xpoints, where, 0, 1) + dely(U, hy,n,i,j,total_ypoints, where, 1, 1)])
    else:
        print(f"{U[0,0,0].shape} is invalid dimension for divergence operator")

def grad(U, hx, hy, n, i, j, total_xpoints, total_ypoints, where="interior"):
    if U[0,0,0].shape == (2,):
        return np.array([[delx(U,hx,n,i,j,total_xpoints, where, 0), dely(U,hy,n,i,j,total_ypoints, where, 0)],
                [delx(U,hx,n,i,j,total_xpoints, where, 1), dely(U,hy,n,i,j,total_ypoints, where, 1)]])
    else:
        print(f"{U[0,0,0].shape} is invalid dimension for gradient operator")

def h2lap(U, hx, hy, n, i, j, total_xpoints, total_ypoints, where="interior"):
    if U[0,0,0].shape == (2,):
        return np.array([h2delxx(U,hx,n,i,j,total_xpoints, where, 0) + h2delyy(U,hy,n,i,j,total_ypoints, where, 0),
                h2delxx(U,hx,n,i,j,total_xpoints, where, 1) + h2delyy(U,hy,n,i,j,total_ypoints, where, 1)])
    elif U[0,0,0].shape == (2,2):
        return np.array([[h2delxx(U,hx,n,i,j,total_xpoints, where, 0,0) + h2delyy(U,hy,n,i,j,total_ypoints, where, 0,0),
                h2delxx(U,hx,n,i,j,total_xpoints, where, 0,1) + h2delyy(U,hy,n,i,j,total_ypoints, where, 0,1)],
                [h2delxx(U,hx,n,i,j,total_xpoints, where, 1,0) + h2delyy(U,hy,n,i,j,total_ypoints, where, 1,0),
                h2delxx(U,hx,n,i,j,total_xpoints, where, 1,1) + h2delyy(U,hy,n,i,j,total_ypoints, where, 1,1)]])
    else:
        print(f"{U[0,0,0].shape} is invalid dimension for laplacian operator")

# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================

def post_processing(U_next, V_next, u_bcs, v_bcs, bc_type, hx, hy, ts, n, U_prev=None, V_prev=None, tau=None):  
    u_left, u_right, u_top, u_bottom = u_bcs[0], u_bcs[1], u_bcs[2], u_bcs[3]
    v_left, v_right, v_top, v_bottom = v_bcs[0], v_bcs[1], v_bcs[2], v_bcs[3]

    if "dirichlet" in bc_type[0].keys():
        if "left" in bc_type[0]["dirichlet"]:
            U_next[0,:] = u_left(ts[n])
        if "right" in bc_type[0]["dirichlet"]:
            U_next[-1,:] = u_right(ts[n]) 
        if "bottom" in bc_type[0]["dirichlet"]:
            U_next[:,0] = u_bottom(ts[n])
        if "top" in bc_type[0]["dirichlet"]:
            U_next[:,-1] = u_top(ts[n])      

    if "reflecting" in bc_type[0].keys():
        if "left" in bc_type[0]["reflecting"]:
            U_next[0,:] = np.array([0,0])
        if "right" in bc_type[0]["reflecting"]:
            U_next[-1,:] = np.array([0,0])
        if "bottom" in bc_type[0]["reflecting"]:
            U_next[:,0] = np.array([0,0])
        if "top" in bc_type[0]["reflecting"]:
            U_next[:,-1] = np.array([0,0])

    if "neumann" in bc_type[0].keys():
        if "left" in bc_type[0]["neumann"]:
            U_next[0,:] = hx*u_left(ts[n]) + U_next[1,:]
        if "right" in bc_type[0]["neumann"]:
            U_next[-1,:] = hx*u_right(ts[n]) + U_next[-2,:]
        if "bottom" in bc_type[0]["neumann"]:
            U_next[:,0] = hx*u_bottom(ts[n]) + U_next[:,1]
        if "top" in bc_type[0]["neumann"]:
            U_next[:,-1] = hx*u_top(ts[n]) + U_next[:,-2]

    if "dirichlet" in bc_type[1].keys():
        if "left" in bc_type[1]["dirichlet"]:
            V_next[0,:] = v_left(ts[n])
        if "right" in bc_type[1]["dirichlet"]:
            V_next[-1,:] = v_right(ts[n]) 
        if "bottom" in bc_type[1]["dirichlet"]:
            V_next[:,0] = v_bottom(ts[n])
        if "top" in bc_type[1]["dirichlet"]:
            V_next[:,-1] = v_top(ts[n])      

    if "reflecting" in bc_type[1].keys():
        if "left" in bc_type[1]["reflecting"]:
            V_next[0,:] = np.array([0,0])
        if "right" in bc_type[1]["reflecting"]:
            V_next[-1,:] = np.array([0,0])
        if "bottom" in bc_type[1]["reflecting"]:
            V_next[:,0] = np.array([0,0])
        if "top" in bc_type[1]["reflecting"]:
            V_next[:,-1] = np.array([0,0])

    if "neumann" in bc_type[1].keys():
        if "left" in bc_type[1]["neumann"]:
            V_next[0,:] = hx*v_left(ts[n]) + V_next[1,:]
        if "right" in bc_type[1]["neumann"]:
            V_next[-1,:] = hx*v_right(ts[n]) + V_next[-2,:]
        if "bottom" in bc_type[1]["neumann"]:
            V_next[:,0] = hx*v_bottom(ts[n]) + V_next[:,1]
        if "top" in bc_type[1]["neumann"]:
            V_next[:,-1] = hx*v_top(ts[n]) + V_next[:,-2]

    return U_next, V_next

# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================

def linear_center_diff_step(c, U, V, n, f, u_bcs, v_bcs, xs, ys, ts, epsilon=0, mu=0.5, lambd=0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_xpoints = len(xs)-1
    total_ypoints = len(ys)-1

    # solve in all space, for one time
    U_next = np.zeros_like(U[0])
    V_next = np.zeros_like(V[0])

    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]
    tau = ts[1] - ts[0]
    for i in range(1, total_xpoints):
        for j in range(1, total_ypoints):
            where = "interior"
            
            denominator = 2*tau
            forcing = f(ts[n],xs[i], ys[j],)

            stability_termv = epsilon*h2lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            stability_termu = epsilon*h2lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)

            v_xterm = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            gradu = grad(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            graduT = gradu.T
            u_xterm = kappa(gradu, graduT, mu, lambd, linear=True)
            
            U_next[i,j] = U[n-1,i,j] + denominator*( -v_xterm + stability_termu )
            V_next[i,j] = V[n-1,i,j] + denominator*( -(c**2)*u_xterm + forcing + stability_termv )

    # update top and bottom
    for i in range(total_xpoints+1):
        for j in [0,total_ypoints]:
            where = "bdy"

            denominator = 2*tau
            forcing = f(ts[n],xs[i], ys[j],)

            stability_termv = epsilon*h2lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            stability_termu = epsilon*h2lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)

            v_xterm = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            gradu = grad(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            graduT = gradu.T
            u_xterm = kappa(gradu, graduT, mu, lambd, linear=True)

            U_next[i,j] = U[n-1,i,j] + denominator*( -v_xterm + stability_termu )
            V_next[i,j] = V[n-1,i,j] + denominator*( -(c**2)*u_xterm + forcing + stability_termv )    
            
    # update left and right
    for i in [0, total_xpoints]:
        for j in range(total_ypoints+1):
            where = "bdy"

            denominator = 2*tau
            forcing = f(ts[n],xs[i], ys[j],)

            stability_termv = epsilon*h2lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            stability_termu = epsilon*h2lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)

            v_xterm = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            gradu = grad(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            graduT = gradu.T
            u_xterm = kappa(gradu, graduT, mu, lambd, linear=True)

            U_next[i,j] = U[n-1,i,j] + denominator*( -v_xterm + stability_termu )
            V_next[i,j] = V[n-1,i,j] + denominator*( -(c**2)*u_xterm + forcing + stability_termv )

    U_prev, V_prev = U[n-1], V[n-1]
    U_next, V_next = post_processing(U_next, V_next, u_bcs, v_bcs, bc_type, hx, hy, ts, n, U_prev, V_prev, tau)

    return U_next, V_next

def linear_forward_diff_step(c, U, V, n, f, u_bcs, v_bcs, xs, ys, ts, epsilon=0, mu=0.5, lambd=0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_xpoints = len(xs)-1
    total_ypoints = len(ys)-1

    # solve in all space, for one time
    U_next = np.zeros_like(U[0])
    V_next = np.zeros_like(V[0])

    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]
    tau = ts[1] - ts[0]
    for i in range(1, total_xpoints):
        for j in range(1, total_ypoints):
            where = "interior"

            denominator = tau
            forcing = f(ts[n],xs[i], ys[j],)

            stability_termv = epsilon*h2lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            stability_termu = epsilon*h2lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)

            v_xterm = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            gradu = grad(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            graduT = gradu.T
            u_xterm = kappa(gradu, graduT, mu, lambd, linear=True)

            U_next[i,j] = U[n,i,j] + denominator*( -v_xterm + stability_termu )
            V_next[i,j] = V[n,i,j] + denominator*( -(c**2)*u_xterm + forcing + stability_termv )

    # update top and bottom
    for i in range(total_xpoints+1):
        for j in [0,total_ypoints]:
            where = "bdy"
            
            denominator = tau
            forcing = f(ts[n],xs[i], ys[j],)

            stability_termv = epsilon*h2lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            stability_termu = epsilon*h2lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)

            v_xterm = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            gradu = grad(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            graduT = gradu.T
            u_xterm = kappa(gradu, graduT, mu, lambd, linear=True)

            U_next[i,j] = U[n,i,j] + denominator*( -v_xterm + stability_termu )
            V_next[i,j] = V[n,i,j] + denominator*( -(c**2)*u_xterm + forcing + stability_termv )

    # update left and right
    for i in [0, total_xpoints]:
        for j in range(total_ypoints+1):
            where = "bdy"
            
            denominator = tau
            forcing = f(ts[n],xs[i], ys[j],)

            stability_termv = epsilon*h2lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            stability_termu = epsilon*h2lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)

            v_xterm = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            gradu = grad(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            graduT = gradu.T
            u_xterm = kappa(gradu, graduT, mu, lambd, linear=True)
            
            U_next[i,j] = U[n,i,j] + denominator*( -v_xterm + stability_termu )
            V_next[i,j] = V[n,i,j] + denominator*( -(c**2)*u_xterm + forcing + stability_termv )

    U_prev, V_prev = U[n-1], V[n-1]
    U_next, V_next = post_processing(U_next, V_next, u_bcs, v_bcs, bc_type, hx, hy, ts, n, U_prev, V_prev, tau)

    return U_next, V_next

# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================

def nonlinear_center_diff_step(c, U, V, n, f, u_bcs, v_bcs, xs, ys, ts, epsilon=0, mu=0.5, lambd=0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_xpoints = len(xs)-1
    total_ypoints = len(ys)-1

    # solve in all space, for one time
    U_next = np.zeros_like(U[0])
    V_next = np.zeros_like(V[0])

    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]
    tau = ts[1] - ts[0]
    for i in range(1, total_xpoints):
        for j in range(1, total_ypoints):
            where = "interior"
            
            denominator = 2*tau
            forcing = f(ts[n],xs[i], ys[j],)

            stability_termv = epsilon*h2lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            stability_termu = epsilon*h2lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)

            v_xterm = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            gradu = grad(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            graduT = gradu.T
            u_xterm = (np.identity(2) + gradu)@kappa(gradu, graduT, mu, lambd)

            U_next[i,j] = U[n-1,i,j] + denominator*( -v_xterm + stability_termu )
            V_next[i,j] = V[n-1,i,j] + denominator*( -(c**2)*u_xterm + forcing + stability_termv )

    # update top and bottom
    for i in range(0, total_xpoints+1):
        for j in [0, total_ypoints]:
            where = "bdy"
            
            denominator = 2*tau
            forcing = f(ts[n],xs[i], ys[j],)

            stability_termv = epsilon*h2lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            stability_termu = epsilon*h2lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)

            v_xterm = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            gradu = grad(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            graduT = gradu.T
            u_xterm = (np.identity(2) + gradu)@kappa(gradu, graduT, mu, lambd)

            U_next[i,j] = U[n-1,i,j] + denominator*( -v_xterm + stability_termu )
            V_next[i,j] = V[n-1,i,j] + denominator*( -(c**2)*u_xterm + forcing + stability_termv )

    # update left and right
    for i in [0, total_xpoints]:
        for j in range(total_ypoints+1):
            where = "bdy"
            
            denominator = 2*tau
            forcing = f(ts[n],xs[i], ys[j],)

            stability_termv = epsilon*h2lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            stability_termu = epsilon*h2lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)

            v_xterm = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            gradu = grad(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            graduT = gradu.T
            u_xterm = (np.identity(2) + gradu)@kappa(gradu, graduT, mu, lambd)

            U_next[i,j] = U[n-1,i,j] + denominator*( -v_xterm + stability_termu )
            V_next[i,j] = V[n-1,i,j] + denominator*( -(c**2)*u_xterm + forcing + stability_termv )

    U_prev, V_prev = U[n-1], V[n-1]
    U_next, V_next = post_processing(U_next, V_next, u_bcs, v_bcs, bc_type, hx, hy, ts, n, U_prev, V_prev, tau)

    return U_next, V_next

def nonlinear_forward_diff_step(c, U, V, n, f, u_bcs, v_bcs, xs, ys, ts, epsilon=0, mu=0.5, lambd=0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_xpoints = len(xs)-1
    total_ypoints = len(ys)-1

    # solve in all space, for one time
    U_next = np.zeros_like(U[0])
    V_next = np.zeros_like(V[0])

    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]
    tau = ts[1] - ts[0]
    for i in range(1, total_xpoints):
        for j in range(1, total_ypoints):
            where = "interior"
            
            denominator = tau
            forcing = f(ts[n],xs[i], ys[j],)

            stability_termv = epsilon*h2lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            stability_termu = epsilon*h2lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)

            v_xterm = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            gradu = grad(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            graduT = gradu.T
            u_xterm = (np.identity(2) + gradu)@kappa(gradu, graduT, mu, lambd)

            U_next[i,j] = U[n,i,j] + denominator*( -v_xterm + stability_termu )
            V_next[i,j] = V[n,i,j] + denominator*( -(c**2)*u_xterm + forcing + stability_termv )

    # update top and bottom
    for i in range(total_xpoints+1):
        for j in [0,total_ypoints]:
            where = "bdy"
            
            denominator = tau
            forcing = f(ts[n],xs[i], ys[j],)

            stability_termv = epsilon*h2lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            stability_termu = epsilon*h2lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)

            v_xterm = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            gradu = grad(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            graduT = gradu.T
            u_xterm = (np.identity(2) + gradu)@kappa(gradu, graduT, mu, lambd)

            U_next[i,j] = U[n,i,j] + denominator*( -v_xterm + stability_termu )
            V_next[i,j] = V[n,i,j] + denominator*( -(c**2)*u_xterm + forcing + stability_termv )

    # update left and right
    for i in [0, total_xpoints]:
        for j in range(total_ypoints+1):
            where = "bdy"
            
            denominator = tau
            forcing = f(ts[n],xs[i], ys[j],)

            stability_termv = epsilon*h2lap(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            stability_termu = epsilon*h2lap(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)

            v_xterm = div(V,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            gradu = grad(U,hx,hy,n,i,j,total_xpoints,total_ypoints, where)
            graduT = gradu.T
            u_xterm = (np.identity(2) + gradu)@kappa(gradu, graduT, mu, lambd)

            U_next[i,j] = U[n,i,j] + denominator*( -v_xterm + stability_termu )
            V_next[i,j] = V[n,i,j] + denominator*( -(c**2)*u_xterm + forcing + stability_termv )

    U_prev, V_prev = U[n-1], V[n-1]
    U_next, V_next = post_processing(U_next, V_next, u_bcs, v_bcs, bc_type, hx, hy, ts, n, U_prev, V_prev, tau)

    return U_next, V_next