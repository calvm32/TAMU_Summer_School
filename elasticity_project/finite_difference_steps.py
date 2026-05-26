import numpy as np

def linear_center_diff_step(c, U, V, n, u_left, u_right, v_left, v_right, xs, ts, epsilon = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_points = len(xs)-1

    # solve in all space, for one time
    U_next = np.zeros(total_points+1)
    V_next = np.zeros(total_points+1)

    h = xs[1] - xs[0]
    tau = ts[1] - ts[0]
    for i in range(total_points+1):
        multiplier = 2*tau

        if i == 0:
            U_next[i] = U[n-1,i] + multiplier*( -(V[n,i+1] - V[n,i])/h )
            V_next[i] = V[n-1,i] + multiplier*( -(c**2)*(U[n,i+1] - U[n,i])/h )
        elif i == total_points:
            U_next[i] = U[n-1,i] + multiplier*( -(V[n,i] - V[n,i-1])/h )
            V_next[i] = V[n-1,i] + multiplier*( -(c**2)*(U[n,i] - U[n,i-1])/h )
        else:
            stability_term1 = epsilon*(V[n,i+1] - 2*V[n,i] + V[n,i-1])/(h**2)
            stability_term2 = epsilon*(U[n,i+1] - 2*U[n,i] + U[n,i-1])/(h**2)
            
            U_next[i] = U[n-1,i] + multiplier*( -(V[n,i+1] - V[n,i-1])/(2*h) - stability_term1 )
            V_next[i] = V[n-1,i] + multiplier*( -(c**2)*(U[n,i+1] - U[n,i-1])/(2*h) - stability_term2 )

        if bc_type == "dirichlet":
            U_next[0] = u_left(ts[n])
            U_next[-1] = u_right(ts[n])        
            V_next[0] = v_left(ts[n])
            V_next[-1] = v_right(ts[n])

    #np.savetxt('output.txt', U, fmt='%.2f', delimiter=',')

    return U_next, V_next

def linear_forward_diff_step(c, U, V, n, u_left, u_right, v_left, v_right, xs, ts, epsilon = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_points = len(xs)-1
    
    # solve in all space, for one time
    U_next = np.zeros(total_points+1)
    V_next = np.zeros(total_points+1)

    h = xs[1] - xs[0]
    tau = ts[1] - ts[0]
    for i in range(total_points+1):
        multiplier = tau

        if i == 0:
            U_next[i] = U[n,i] + multiplier*( -(V[n,i+1] - V[n,i])/h )
            V_next[i] = V[n,i] + multiplier*( -(c**2)*(U[n,i+1] - U[n,i])/h )
        elif i == total_points:
            U_next[i] = U[n,i] + multiplier*( -(V[n,i] - V[n,i-1])/h )
            V_next[i] = V[n,i] + multiplier*( -(c**2)*(U[n,i] - U[n,i-1])/h )
        else:
            stability_term1 = epsilon*(V[n,i+1] - 2*V[n,i] + V[n,i-1])/(h**2)
            stability_term2 = epsilon*(U[n,i+1] - 2*U[n,i] + U[n,i-1])/(h**2)
            
            U_next[i] = U[n,i] + multiplier*( -(V[n,i+1] - V[n,i-1])/(2*h) - stability_term1 )
            V_next[i] = V[n,i] + multiplier*( -(c**2)*(U[n,i+1] - U[n,i-1])/(2*h) - stability_term2 )

        if bc_type == "dirichlet":
            U_next[0] = u_left(ts[n])
            U_next[-1] = u_right(ts[n])        
            V_next[0] = v_left(ts[n])
            V_next[-1] = v_right(ts[n])

    np.savetxt('output.txt', U, fmt='%.2f', delimiter=',')

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
        multiplier = 2*tau
        forcing = f(ts[n],xs[i])

        if i == 0:
            v_x = (V[n,i+1] - V[n,i])/h
            u_x = (U[n,i+1] - U[n,i])/h

            U_next[i] = U[n-1,i] + multiplier*( -v_x )
            V_next[i] = V[n-1,i] + multiplier*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + forcing )
        elif i == total_points:
            v_x = (V[n,i] - V[n,i-1])/h
            u_x = (U[n,i] - U[n,i-1])/h

            U_next[i] = U[n-1,i] + multiplier*( -v_x )
            V_next[i] = V[n-1,i] + multiplier*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + forcing )
        else:
            stability_term1 = epsilon*(V[n,i+1] - 2*V[n,i] + V[n,i-1])/(h**2)
            stability_term2 = epsilon*(U[n,i+1] - 2*U[n,i] + U[n,i-1])/(h**2)

            v_x = (V[n,i+1] - V[n,i-1])/(2*h)
            u_x = (U[n,i+1] - U[n,i-1])/(2*h)
            
            U_next[i] = U[n-1,i] + multiplier*( -v_x - stability_term1 )
            V_next[i] = V[n-1,i] + multiplier*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) - stability_term2 + forcing )

        if bc_type == "dirichlet":
            U_next[0] = u_left(ts[n])
            U_next[-1] = u_right(ts[n])        
            V_next[0] = v_left(ts[n])
            V_next[-1] = v_right(ts[n])

    #np.savetxt('output.txt', U, fmt='%.2f', delimiter=',')

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
        multiplier = tau
        forcing = f(ts[n],xs[i])

        if i == 0:
            v_x = (V[n,i+1] - V[n,i])/h
            u_x = (U[n,i+1] - U[n,i])/h

            U_next[i] = U[n,i] + multiplier*( -v_x )
            V_next[i] = V[n,i] + multiplier*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + forcing )
        elif i == total_points:
            v_x = (V[n,i] - V[n,i-1])/h
            u_x = (U[n,i] - U[n,i-1])/h

            U_next[i] = U[n,i] + multiplier*( -v_x )
            V_next[i] = V[n,i] + multiplier*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) + forcing )
        else:
            stability_term1 = epsilon*(V[n,i+1] - 2*V[n,i] + V[n,i-1])/(h**2)
            stability_term2 = epsilon*(U[n,i+1] - 2*U[n,i] + U[n,i-1])/(h**2)

            v_x = (V[n,i+1] - V[n,i-1])/(2*h)
            u_x = (U[n,i+1] - U[n,i-1])/(2*h)
            
            U_next[i] = U[n,i] + multiplier*( -v_x - stability_term1 )
            V_next[i] = V[n,i] + multiplier*( -(c**2)*u_x*(1+u_x)*(1+0.5*u_x) - stability_term2 + forcing )

        if bc_type == "dirichlet":
            U_next[0] = u_left(ts[n])
            U_next[-1] = u_right(ts[n])        
            V_next[0] = v_left(ts[n])
            V_next[-1] = v_right(ts[n])

    return U_next, V_next