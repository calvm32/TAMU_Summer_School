import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from finite_difference_steps_2dscalar import *

"""
solve { u_t + v_x - epsilon(lap(u))= 0, 
        v_t + c^2u_x(1+u_x)(1+0.5u_x) - epsilon(lap(v)) = F 
       } on [a,b]

equivalent form: u_tt - c^2partial_x ( u_x(1+u_x)(1+0.5u_x) ) = F_t
"""

def solve_nonlinear(c, u_bcs, v_bcs, u_0, v_0, f, xs, ts, epsilon = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_xpoints = len(xs[:,0])-1
    total_ypoints = len(xs[0,:])-1

    # solutions in time, space
    U = np.zeros((total_times+1, total_xpoints+1, total_ypoints+1))
    V = np.zeros((total_times+1, total_xpoints+1, total_ypoints+1))

    # initialize
    for i in range(total_xpoints+1):
        for j in range(total_ypoints+1):
            U[0, i,j] = u_0(xs[i,j])
            V[0, i,j] = v_0(xs[i,j])

    U[1,:,:], V[1,:,:] = nonlinear_forward_diff_step(c, U, V, 0, f, u_bcs, v_bcs, xs, ts, epsilon, bc_type)

    # time-stepping
    tau = ts[1] - ts[0]
    for n in range(1, total_times):
        if n % 5000 == 0:
            print(f"done w/ {n}/{total_times}")
        U[n+1,:,:], V[n+1,:,:] = nonlinear_center_diff_step(c, U, V, n, f, u_bcs, v_bcs, xs, ts, epsilon, bc_type)
        
    return U, V


def solve_linear(c, u_bcs, v_bcs, u_0, v_0, f, xs, ts, epsilon = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_xpoints = len(xs[:,0])-1
    total_ypoints = len(xs[0,:])-1

    # solutions in time, space
    U = np.zeros((total_times+1, total_xpoints+1, total_ypoints+1))
    V = np.zeros((total_times+1, total_xpoints+1, total_ypoints+1))

    # initialize
    for i in range(total_xpoints+1):
        for j in range(total_ypoints+1):
            U[0, i,j] = u_0(xs[i,j])
            V[0, i,j] = v_0(xs[i,j])

    U[1,:,:], V[1,:,:] = linear_forward_diff_step(c, U, V, 0, f, u_bcs, v_bcs, xs, ts, epsilon, bc_type)

    # time-stepping
    tau = ts[1] - ts[0]
    for n in range(1, total_times):
        if n % 5000 == 0:
            print(f"done w/ {n}/{total_times}")
        U[n+1,:,:], V[n+1,:,:] = linear_center_diff_step(c, U, V, n, f, u_bcs, v_bcs, xs, ts, epsilon, bc_type)
        
    return U, V


if __name__ == "__main__":

    gravity_constant = 0.1 #9.80665
    k_constant = 1

    # -------------
    # set constants
    # -------------

    c = 1.0

    # space discretization
    total_xpoints = 2**2
    total_ypoints = 2**2
    Omega = [[0, 1], [0, 5]]

    hx = (Omega[0][0] - Omega[0][1])/(total_xpoints+1)
    hy = (Omega[1][0] - Omega[1][1])/(total_ypoints+1)

    xs = np.array([(Omega[0][0] + i*hx, Omega[1][0] + j*hy) 
                for i in range(total_xpoints + 1) 
                for j in range(total_ypoints + 1)])   

    epsilon = 16*hx**2 # 1*h**2 # stability term

    # time discretization
    t0 = 0
    T = 5

    CFL = 0.1
    tau = abs(min(CFL*hx, CFL*hy))
    total_times = (T-t0)/tau + 1
    print(f'timestep={tau}')
    ts = []
    time = t0
    while time <= T:
        ts.append(time)
        time += tau

    # --------------------
    # function definitions
    # --------------------

    factor = 1/(np.pi**2)

    u_0 = lambda x: 0
    v_0 = lambda x: 0

    f = lambda t,x: 0 # - gravity_constant*(x**(k_constant+1) - 1/(k_constant + 2))

    # -------------------
    # boundary conditions
    # -------------------

    # available: dirichlet, do_nothing, reflecting, neumann_right, neumann_left, neumann
    bc_type = "dirichlet"   

    # values at endpoints for u (represents either u or u' depending on whether dirichlet or neumann)
    u_left = lambda t: 0
    u_right = lambda t: gravity_constant
    u_top = lambda t: 0
    u_bottom = lambda t: 0

    u_bcs = [u_left, u_right, u_top, u_bottom]

    # values at endpoints for v (represents either v or v' depending on whether dirichlet or neumann)
    v_left = lambda t: 0
    v_right = lambda t: 0
    v_top = lambda t: 0
    v_bottom = lambda t: 0

    v_bcs = [v_left, v_right, v_top, v_bottom]

    # -------------------------
    # compute approx. and exact
    # -------------------------

    # approximate solution
    U_nl, V_nl = solve_nonlinear(c, u_bcs, v_bcs, u_0, v_0, f, xs, ts, epsilon, bc_type)
    U_l, V_l = solve_linear(c, u_bcs, v_bcs, u_0, v_0, f, xs, ts, epsilon, bc_type)

    # --------------------
    # animate the solution
    # --------------------

    plot = True

    if plot:
        total_frames = 250

        # set the initial plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        U_max = np.max(U_nl)
        U_min = np.min(U_nl)
        abs_U_max = np.max(np.abs(U_nl))
        y_min = U_min - 0.1 * abs_U_max
        y_max = U_max + 0.1 * abs_U_max
        ax.set(
            ylim=[y_min, y_max], xlabel="x", ylabel="u(x,t)", title=f"t = {ts[0]:2f}"
        )
        line_nl, = ax.plot(xs, U_nl[0,:,:], label=r"$U_{nonlinear}$")
        line_l, = ax.plot(xs, U_l[0,:,:], label=r"$U_{linear}$")

        # update the plot each frame
        def update(frame):
            time = int(frame*(total_times + 1)/total_frames)
            line_nl.set_ydata(U_nl[time,:,:])
            line_l.set_ydata(U_l[time,:,:])
            ax.set(title=f"t = {ts[time]:2f}")
            return line_nl, line_l

        # create the animation
        ani = animation.FuncAnimation(fig=fig, func=update, frames=total_frames, interval=30, blit=False)
        # writer = animation.FFMpegWriter(fig=fig, func=update, frames=total_frames, interval=30, blit=False, fps=30, metadata=dict(artist='Me'), bitrate=1800)
        # anim.save('my_animation.mp4', writer=writer)

        plt.legend()
        plt.show()
