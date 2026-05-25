import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

"""
solve { u_t + v_x = 0, v_t + c^2u_x = 0 } on [a,b]
"""

def center_diff_step(U, V, n, xs, ts, epsilon = 0):
    
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

    np.savetxt('output.txt', U, fmt='%.2f', delimiter=',')

    return U_next, V_next

def forward_diff_step(U, V, n, xs, ts, epsilon = 0):
    
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

def solve(c, alpha, beta, u_0, v_0, xs, ts, epsilon = 0):

    total_times = len(ts)-1
    total_points = len(xs)-1

    # solutions in time, space
    U = np.zeros((total_times+1, total_points+1))
    V = np.zeros((total_times+1, total_points+1))

    # initialize
    for i in range(total_points+1):
        U[0, i] = u_0(xs[i])
        V[0, i] = v_0(xs[i])

    U[1,:], V[1,:] = forward_diff_step(U, V, 0, xs, ts, epsilon)

    # time-stepping
    tau = (T-t0)/(total_times+1)
    for n in range(1, total_times):
        U[n+1,:], V[n+1,:] = center_diff_step(U, V, n, xs, ts, epsilon)

    return U, V


if __name__ == "__main__":

    # -------------
    # set constants
    # -------------

    c = 1

    # space discretization
    total_points = 100
    a = -10
    b = 10

    alpha=10
    beta=10

    h = (b - a)/(total_points+1)
    xs = [a + i*h for i in range(total_points + 1)]
    epsilon = 0 #1e-4 #c*h**2 # stability term
    
    # time discretization
    total_times = 100
    t0 = 0
    T = 10

    u_0 = lambda x: np.exp(-x**2)
    v_0 = lambda x: 0

    tau = min((1/c)*h, (T-t0)/total_times)
    ts = []
    time = t0
    while time <= T:
        ts.append(time)
        time += tau

    # -------------------------
    # compute approx. and exact
    # -------------------------

    # approximate solution
    U, V = solve(c, alpha, beta, u_0, v_0, xs, ts, epsilon)

    u_exact = lambda x,t: 0.5*(u_0(x+c*t)+ u_0(x-c*t))

    # --------------------
    # animate the solution
    # --------------------

    # set the initial plot
    fig, ax = plt.subplots()
    U_max = np.max(U)
    U_min = np.min(U)
    abs_U_max = np.max(np.abs(U))
    y_min = U_min - 0.1 * abs_U_max
    y_max = U_max + 0.1 * abs_U_max
    ax.set(
        ylim=[y_min, y_max], xlabel="x", ylabel="u(x,t)", title=f"t = {ts[0]:2f}"
    )
    line = ax.plot(xs, U[0,:])[0]

    # update the plot each frame
    def update(frame):
        line.set_ydata(U[frame,:])
        ax.set(title=f"t = {ts[frame]:2f}")
        return line

    # create the animation
    ani = animation.FuncAnimation(fig=fig, func=update, frames=total_times + 1, interval=30, blit=False)
    plt.show()