import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from finite_difference_steps import *

"""
solve { u_t + v_x - epsilon(lap(u))= 0, 
        v_t + c^2u_x(1+u_x)(1+0.5u_x) - epsilon(lap(v)) = F 
       } on [a,b]

equivalent form: u_tt - c^2partial_x ( u_x(1+u_x)(1+0.5u_x) ) = F_t
"""

def solve(c, u_left, u_right, v_left, v_right, u_0, v_0, f, xs, ts, epsilon = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_points = len(xs)-1

    # solutions in time, space
    U = np.zeros((total_times+1, total_points+1))
    V = np.zeros((total_times+1, total_points+1))

    # initialize
    for i in range(total_points+1):
        U[0, i] = u_0(xs[i])
        V[0, i] = v_0(xs[i])

    U[1,:], V[1,:] = nonlinear_forward_diff_step(c, U, V, 0, f, u_left, u_right, v_left, v_right, xs, ts, epsilon, bc_type)

    # time-stepping
    tau = ts[1] - ts[0]
    for n in range(1, total_times):
        U[n+1,:], V[n+1,:] = nonlinear_center_diff_step(c, U, V, n, f, u_left, u_right, v_left, v_right, xs, ts, epsilon, bc_type)
        
    return U, V


if __name__ == "__main__":

    # -------------
    # set constants
    # -------------

    c = 0.25

    # space discretization
    total_points = 100
    a = 0
    b = 4

    h = (b - a)/(total_points+1)
    xs = [a + i*h for i in range(total_points + 1)]
    epsilon = 1 #2*h**2 # stability term
    
    # time discretization
    total_times = 100
    t0 = 0
    T = 20

    tau = 0.1*h # min((1/c)*h, (T-t0)/total_times)
    ts = []
    time = t0
    while time <= T:
        ts.append(time)
        time += tau

    # --------------------
    # function definitions
    # --------------------

    factor = 1/(np.pi**2)

    u_0 = lambda x: factor*np.sin(np.pi*x)
    v_0 = lambda x: factor*c*np.sin(np.pi*x)

    u_x  = lambda t,x: factor*np.pi*np.cos(np.pi*(x-c*t))
    v_t  = lambda t,x: -c**2*factor*np.pi*np.cos(np.pi*(x - c*t))
    v_xx = lambda t,x: -factor*c*(np.pi**2)*np.sin(np.pi*(x-c*t))

    f = lambda t,x: v_t(t,x) +c**2*u_x(t,x)*(1+u_x(t,x))*(1+0.5*u_x(t,x)) - epsilon*v_xx

    # -------------------
    # boundary conditions
    # -------------------

    bc_type = "dirichlet"   # available: dirichlet, do_nothing, reflecting , neumann_right

    # values at endpoints for u
    u_left = lambda t: factor*np.sin(np.pi*(a-c*t))
    u_right = lambda t: factor*np.sin(np.pi*(b-c*t))

    # values at endpoints for v
    v_left = lambda t: factor*c*np.sin(np.pi*(a-c*t))
    v_right = lambda t: factor*c*np.sin(np.pi*(b-c*t))

    # -------------------------
    # compute approx. and exact
    # -------------------------

    # approximate solution
    U, V = solve(c, u_left, u_right, v_left, v_right, u_0, v_0, f, xs, ts, epsilon, bc_type)

    u_exact = lambda t,x: factor*np.sin(np.pi*(x-c*t))
    U_exact = np.array([[u_exact(t, x) for x in xs] for t in ts])

    # --------------------
    # animate the solution
    # --------------------

    plot = True

    if plot:
        # set the initial plot
        fig, ax = plt.subplots()
        U_max = np.max(U)
        U_min = np.min(U)
        abs_U_max = np.max(np.abs(U))
        y_min = -1 # U_min - 0.1 * abs_U_max
        y_max = 1 #U_max + 0.1 * abs_U_max
        ax.set(
            ylim=[y_min, y_max], xlabel="x", ylabel="u(x,t)", title=f"t = {ts[0]:2f}"
        )
        line, = ax.plot(xs, U[0,:], label="U (approx)")
        line_exact, = ax.plot(xs, U_exact[0,:], label="U_exact", linestyle="--")

        # update the plot each frame
        def update(frame):
            line.set_ydata(U[frame,:])
            line_exact.set_ydata(U_exact[frame,:])
            ax.set(title=f"t = {ts[frame]:2f}")
            return line, line_exact

        # create the animation
        ani = animation.FuncAnimation(fig=fig, func=update, frames=total_times + 1, interval=30, blit=False)
        plt.legend()
        plt.show()