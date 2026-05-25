import numpy as np

def linear_center_diff_step(c, U, V, n, xs, ts, epsilon = 0):

    total_times = len(ts)-1
    total_points = len(xs)-1

    # solve in all space, for one time
    U_next = np.zeros(total_points+1)
    V_next = np.zeros(total_points+1)

    h = (xs[-1] - xs[0])/len(xs)
    tau = (ts[-1] - ts[0])/len(ts)
    for i in range(total_points+1):

        if i == 0:
            U_next[i] = U[n-1,i] - (2*tau/h)*(V[n,i+1] - V[n,i])
            V_next[i] = V[n-1,i] - (c**2)*(2*tau/h)*(U[n,i+1] - U[n,i])
        elif i == total_points:
            U_next[i] = U[n-1,i] - (2*tau/h)*(V[n,i] - V[n,i-1])
            V_next[i] = V[n-1,i] - (c**2)*(2*tau/h)*(U[n,i] - U[n,i-1])
        else:
            stability_term1 = 2*epsilon*tau/(h**2)*(V[n,i+1] - 2*V[n,i] + V[n,i-1])
            stability_term2 = 2*epsilon*tau/(h**2)*(U[n,i+1] - 2*U[n,i] + U[n,i-1])
            
            U_next[i] = U[n-1,i] - (tau/h)*(V[n,i+1] - V[n,i-1]) + stability_term1
            V_next[i] = V[n-1,i] - (c**2)*(tau/h)*(U[n,i+1] - U[n,i-1]) + stability_term2

    #np.savetxt('output.txt', U, fmt='%.2f', delimiter=',')

    return U_next, V_next

def linear_forward_diff_step(c, U, V, n, xs, ts, epsilon = 0):

    total_times = len(ts)-1
    total_points = len(xs)-1
    
    # solve in all space, for one time
    U_next = np.zeros(total_points+1)
    V_next = np.zeros(total_points+1)

    h = (xs[-1] - xs[0])/len(xs)
    tau = (ts[-1] - ts[0])/len(ts)
    for i in range(total_points+1):

        if i == 0:
            U_next[i] = U[n,i] - (tau/h)*(V[n,i+1] - V[n,i])
            V_next[i] = V[n,i] - (c**2)*(tau/h)*(U[n,i+1] - U[n,i])
        elif i == total_points:
            U_next[i] = U[n,i] - (tau/h)*(V[n,i] - V[n,i-1])
            V_next[i] = V[n,i] - (c**2)*(tau/h)*(U[n,i] - U[n,i-1])
        else:
            stability_term1 = 2*epsilon*tau/(2*h**2)*(V[n,i+1] - 2*V[n,i] + V[n,i-1])
            stability_term2 = 2*epsilon*tau/(2*h**2)*(U[n,i+1] - 2*U[n,i] + U[n,i-1])
            
            U_next[i] = U[n,i] - (tau/(2*h))*(V[n,i+1] - V[n,i-1]) + stability_term1
            V_next[i] = V[n,i] - (c**2)*(tau/(2*h))*(U[n,i+1] - U[n,i-1]) + stability_term2

    np.savetxt('output.txt', U, fmt='%.2f', delimiter=',')

    return U_next, V_next

# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================

def nonlinear_center_diff_step(c, U, V, n, xs, ts, epsilon = 0):

    total_times = len(ts)-1
    total_points = len(xs)-1
    
    # solve in all space, for one time
    U_next = np.zeros(total_points+1)
    V_next = np.zeros(total_points+1)

    h = (xs[-1] - xs[0])/len(xs)
    tau = (ts[-1] - ts[0])/len(ts)
    for i in range(total_points+1):

        if i == 0:
            dx_v = (V[n,i+1] - V[n,i])/h
            dx_u = (U[n,i+1] - U[n,i])/h
            U_next[i] = U[n-1,i] - (2*tau)*dx_v
            V_next[i] = V[n-1,i] - (c**2)*(2*tau)*dx_u*(1+dx_u)*(1+0.5*dx_u)
        elif i == total_points:
            dx_v = (V[n,i] - V[n,i-1])/h
            dx_u = (U[n,i] - U[n,i-1])/h
            U_next[i] = U[n-1,i] - (2*tau)*dx_v
            V_next[i] = V[n-1,i] - (c**2)*(2*tau)*dx_u*(1+dx_u)*(1+0.5*dx_u)
        else:
            stability_term1 = 2*epsilon*tau/(h**2)*(V[n,i+1] - 2*V[n,i] + V[n,i-1])
            stability_term2 = 2*epsilon*tau/(h**2)*(U[n,i+1] - 2*U[n,i] + U[n,i-1])

            dx_v = (V[n,i+1] - V[n,i-1])/(2*h)
            dx_u = (U[n,i+1] - U[n,i-1])/(2*h)
            
            U_next[i] = U[n-1,i] - (2*tau)*dx_v + stability_term1
            V_next[i] = V[n-1,i] - (c**2)*(2*tau)*dx_u*(1+dx_u)*(1+0.5*dx_u) + stability_term2

    #np.savetxt('output.txt', U, fmt='%.2f', delimiter=',')

    return U_next, V_next

def nonlinear_forward_diff_step(c, U, V, n, xs, ts, epsilon = 0):

    total_times = len(ts)-1
    total_points = len(xs)-1
    
    # solve in all space, for one time
    U_next = np.zeros(total_points+1)
    V_next = np.zeros(total_points+1)

    h = (xs[-1] - xs[0])/len(xs)
    tau = (ts[-1] - ts[0])/len(ts)
    for i in range(total_points+1):

        if i == 0:
            dx_v = (V[n,i+1] - V[n,i])/h
            dx_u = (U[n,i+1] - U[n,i])/h
            U_next[i] = U[n,i] - (tau)*dx_v
            V_next[i] = V[n,i] - (c**2)*(tau)*dx_u
        elif i == total_points:
            dx_v = (V[n,i] - V[n,i-1])/h
            dx_u = (U[n,i] - U[n,i-1])/h
            U_next[i] = U[n,i] - (tau)*dx_v
            V_next[i] = V[n,i] - (c**2)*(tau)*dx_u
        else:
            stability_term1 = 2*epsilon*tau/(2*h**2)*(V[n,i+1] - 2*V[n,i] + V[n,i-1])
            stability_term2 = 2*epsilon*tau/(2*h**2)*(U[n,i+1] - 2*U[n,i] + U[n,i-1])

            dx_v = (V[n,i+1] - V[n,i-1])/(2*h)
            dx_u = (U[n,i+1] - U[n,i-1])/(2*h)
            
            U_next[i] = U[n,i] - (tau)*dx_v + stability_term1
            V_next[i] = V[n,i] - (c**2)*(tau)*dx_u*(1+dx_u)*(1+0.5*dx_u) + stability_term2

    np.savetxt('output.txt', U, fmt='%.2f', delimiter=',')

    return U_next, V_next