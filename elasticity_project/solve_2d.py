import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

from finite_difference_steps_2d import *

"""
solve { u_t + v_x - epsilon(lap(u))= 0, 
        v_t + c^2u_x(1+u_x)(1+0.5u_x) - epsilon(lap(v)) = F 
       } on [a,b]

equivalent form: u_tt - c^2partial_x ( u_x(1+u_x)(1+0.5u_x) ) = F_t
"""

def solve_nonlinear(c, u_bcs, v_bcs, u_0, v_0, f, xs, ys, ts, epsilon=0, mu=0.5, lambd=0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_xpoints = len(xs)-1
    total_ypoints = len(ys)-1

    # solutions in time, space
    U = np.zeros((len(ts), len(xs), len(ys), 2))
    V = np.zeros((len(ts), len(xs), len(ys), 2,2))

    # initialize
    for i in range(total_xpoints+1):
        for j in range(total_ypoints+1):
            U[0,i,j] = u_0(xs[i], ys[j])
            V[0,i,j] = v_0(xs[i], ys[j])

    U[1,:,:], V[1,:,:] = nonlinear_forward_diff_step(c, U, V, 0, f, u_bcs, v_bcs, xs, ys, ts, epsilon, mu, lambd, bc_type)

    # time-stepping
    tau = ts[1] - ts[0]
    for n in range(1, total_times):
        if n % 5000 == 0:
            print(f"done w/ {n}/{total_times}")
        U[n+1,:,:], V[n+1,:,:] = nonlinear_center_diff_step(c, U, V, n, f, u_bcs, v_bcs, xs, ys, ts, epsilon, mu, lambd, bc_type)
        
    return U, V


def solve_linear(c, u_bcs, v_bcs, u_0, v_0, f, xs, ys, ts, epsilon=0, mu=0.5, lambd=0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_xpoints = len(xs)-1
    total_ypoints = len(ys)-1

    # solutions in time, space
    U = np.zeros((len(ts), len(xs), len(ys), 2))
    V = np.zeros((len(ts), len(xs), len(ys), 2,2))

    # initialize
    for i in range(total_xpoints+1):
        for j in range(total_ypoints+1):
            U[0,i,j] = u_0(xs[i], ys[j])
            V[0,i,j] = v_0(xs[i], ys[j])

    U[1,:,:], V[1,:,:] = linear_forward_diff_step(c, U, V, 0, f, u_bcs, v_bcs, xs, ys, ts, epsilon, mu, lambd, bc_type)

    # time-stepping
    tau = ts[1] - ts[0]
    for n in range(1, total_times):
        if n % 5000 == 0:
            print(f"done w/ {n}/{total_times}")
        U[n+1,:,:], V[n+1,:,:] = linear_center_diff_step(c, U, V, n, f, u_bcs, v_bcs, xs, ys, ts, epsilon, mu, lambd, bc_type)
        
    return U, V


if __name__ == "__main__":

    # -------------
    # set constants
    # -------------

    gravity_constant = 0.5 #9.80665
    k_constant = 1
    mu=0.5
    lambd=0.5

    c = 1
    CFL = 0.1
    stab_constant = 16

    # space discretization
    total_xpoints = 2**5
    total_ypoints = 2**5
    Omega = [[0, 1], [0, 0.5]]

    # dirichlet left, other sides neumann

    hx = (Omega[0][1] - Omega[0][0])/total_xpoints
    hy = (Omega[1][1] - Omega[1][0])/total_ypoints

    xs = [Omega[0][0] + i*hx for i in range(total_xpoints + 1)]  
    ys = [Omega[1][0] + i*hy for i in range(total_ypoints + 1)]  

    epsilon = stab_constant*hx**2 # 1*h**2 # stability term

    # time discretization
    t0 = 0
    T = 5.0
    total_times = 200

    tau = min(min(CFL*hx, CFL*hy), (T-t0)/(total_times+1))
    total_times = (T-t0)/tau - 1
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

    u_0 = lambda x, y: np.array([0, 0])
    v_0 = lambda x, y: np.array([[0, 0], [0, 0]])

    f = lambda t,x,y: np.array([[0, 0], [0, 0]])

    # -------------------
    # boundary conditions
    # -------------------

    # available: dirichlet, do_nothing, reflecting, neumann
    u_bc_type = {"neumann": ["right"], "dirichlet": ["left"]}
    v_bc_type = {}

    bc_type = [u_bc_type, v_bc_type]

    # values at endpoints for u (represents either u or u' depending on whether dirichlet or neumann)
    u_left = lambda t: np.array([0, 0])
    u_right = lambda t: np.array([0, gravity_constant]) # unused
    u_top = lambda t: np.array([0, 0]) # unused
    u_bottom = lambda t: np.array([0, 0]) # unused

    u_bcs = [u_left, u_right, u_top, u_bottom]

    # values at endpoints for v (represents either v or v' depending on whether dirichlet or neumann)
    v_left = lambda t: np.array([[0, 0], [0, 0]]) # unused
    v_right = lambda t: np.array([[gravity_constant, 0], [0, 0]])
    v_top = lambda t: np.array([[0, 0], [0, 0]]) # unused
    v_bottom = lambda t: np.array([[0, 0], [0, 0]]) # unused

    v_bcs = [v_left, v_right, v_top, v_bottom]

    # -------------------------
    # compute approx. and exact
    # -------------------------

    # approximate solution
    U_nl, V_nl = solve_nonlinear(c, u_bcs, v_bcs, u_0, v_0, f, xs, ys, ts, epsilon, mu, lambd, bc_type)
    U_l, V_l = solve_linear(c, u_bcs, v_bcs, u_0, v_0, f, xs, ys, ts, epsilon, mu, lambd, bc_type)

    # --------------------
    # animate the solution
    # --------------------

    plot = True

    if plot:
        total_frames = 250

        # set the initial plot

        mag_nl = np.linalg.norm(U_nl[...,:], axis=-1)
        mag_l  = np.linalg.norm(U_l[...,:], axis=-1)
        vmin = min(np.min(mag_nl), np.min(mag_l))
        vmax = max(np.max(mag_nl), np.max(mag_l))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        extent = [xs[0], xs[-1], ys[0], ys[-1]]

        ax1.set_title(r"$U_{nonlinear}$")
        ax2.set_title(r"$U_{linear}$")

        for ax in [ax1, ax2]:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect('equal')

        surf_nl = ax1.imshow(mag_nl[0], origin='lower', extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, animated=True)
        surf_l = ax2.imshow(mag_l[0], origin='lower', extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, animated=True)

        cbar = fig.colorbar( surf_nl, ax=[ax1, ax2], shrink=0.9, label=r"$|U|$")
        fig.suptitle(f"t = {ts[0]:2f}", fontsize=14)

        # update the plot each frame
        def update(frame):
            global surf_nl, surf_l
            
            time = int(frame*(total_times + 1)/total_frames)

            surf_nl.set_data(mag_nl[time])
            surf_l.set_data(mag_l[time])

            fig.suptitle(f"t = {ts[time]:2f}", fontsize=14)
            return surf_nl, surf_l

        # create the animation
        ani = animation.FuncAnimation(fig=fig, func=update, frames=total_frames, interval=30, blit=False)
        writer = animation.FFMpegWriter(fps=30, bitrate=1800)
        ani.save("nonlinear_vs_linear_magnitude.mp4", writer=writer, dpi=100)

        plt.legend()
        plt.show()


    # if plot: # 3d plot below
    #     total_frames = 40

    #     # set the initial plot
    #     fig = plt.figure(figsize=(12,6))
    #     gs = GridSpec(1,2, figure=fig) # actual layout
    #     ax1 = fig.add_subplot(gs[0,0], projection='3d')
    #     ax2 = fig.add_subplot(gs[0,1], projection='3d')
    #     X, Y = np.meshgrid(xs, ys, indexing='ij')

    #     U_max = np.max(U_nl)
    #     U_min = np.min(U_nl)
    #     abs_U_max = np.max(np.abs(U_nl))
    #     z_min = -4 #U_min - 0.1 * abs_U_max
    #     z_max = 4 #U_max + 0.1 * abs_U_max
    #     ax1.set(
    #         zlim=[z_min, z_max], xlabel="x", ylabel="y", title=r"$U_{nonlinear}$"
    #     )        
    #     ax2.set(
    #         zlim=[z_min, z_max], xlabel="x", ylabel="y", title=r"$U_{linear}$"
    #     )

    #     Z_nl = np.sqrt(U_nl[0,:,:,0]**2 + U_nl[0,:,:,1]**2)
    #     surf_nl = ax1.plot_surface(X, Y, Z_nl, cmap='viridis')
    #     Z_l = np.sqrt(U_l[0,:,:,0]**2 + U_l[0,:,:,1]**2)
    #     surf_l = ax2.plot_surface(X, Y, Z_l, cmap='viridis')   


    #     fig.suptitle(f"t = {ts[0]:2f}", fontsize=14)

    #     # update the plot each frame
    #     def update(frame):
    #         global surf_nl, surf_l
    #         surf_nl.remove()
    #         surf_l.remove()
            
    #         time = int(frame*(total_times + 1)/total_frames)
            
    #         Z_nl = np.sqrt(U_nl[time,:,:,0]**2 + U_nl[time,:,:,1]**2)
    #         Z_l = np.sqrt(U_l[time,:,:,0]**2 + U_l[time,:,:,1]**2)

    #         surf_nl = ax1.plot_surface(X, Y, Z_nl, cmap='viridis')
    #         surf_l = ax2.plot_surface(X, Y, Z_l, cmap='viridis')   

    #         fig.suptitle(f"t = {ts[time]:2f}", fontsize=14)
    #         return surf_nl, surf_l

    #     # create the animation
    #     ani = animation.FuncAnimation(fig=fig, func=update, frames=total_frames, interval=30, blit=False)
    #     writer = animation.FFMpegWriter(fps=30, bitrate=1800)
    #     ani.save("nonlinear_vs_linear.mp4", writer=writer, dpi=100)

    #     #plt.legend(loc='upper right')
    #     plt.show()
