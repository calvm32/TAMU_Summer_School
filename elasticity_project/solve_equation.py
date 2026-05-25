import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from elasticity_project.finite_difference_steps import *

"""
solve { u_t + v_x = 0, v_t + c^2u_x = 0 } on [a,b]
"""

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

    U[1,:], V[1,:] = nonlinear_forward_diff_step(c, U, V, 0, xs, ts, epsilon)

    # time-stepping
    tau = (T-t0)/(total_times+1)
    for n in range(1, total_times):
        U[n+1,:], V[n+1,:] = nonlinear_center_diff_step(c, U, V, n, xs, ts, epsilon)

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
    epsilon = 0.001*h**2 # stability term
    
    # time discretization
    total_times = 1000
    t0 = 0
    T = 100

    u_0 = lambda x: np.exp(-x**2)
    v_0 = lambda x: 0

    tau = 0.1 #min((1/c)*h, (T-t0)/total_times)
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
    y_min = 1 # U_min - 0.1 * abs_U_max
    y_max = -1 #U_max + 0.1 * abs_U_max
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