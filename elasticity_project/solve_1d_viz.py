import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from IPython.display import HTML

from finite_difference_steps_1d import *

"""
solve { u_t + v_x - epsilon(lap(u))= 0, 
        v_t + c^2u_x(1+u_x)(1+0.5u_x) - epsilon(lap(v)) = F 
       } on [a,b]

equivalent form: u_tt - c^2partial_x ( u_x(1+u_x)(1+0.5u_x) ) = F_t
"""

def solve_nonlinear(c, u_left, u_right, v_left, v_right, u_0, v_0, f, xs, ts, epsilon = 0, bc_type="do_nothing"):

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
        if n % 1000 == 0:
            print(f"Nonlinear: done w/ {n}/{total_times}")
        U[n+1,:], V[n+1,:] = nonlinear_center_diff_step(c, U, V, n, f, u_left, u_right, v_left, v_right, xs, ts, epsilon, bc_type)
        
    return U, V


def solve_linear(c, u_left, u_right, v_left, v_right, u_0, v_0, f, xs, ts, epsilon = 0, bc_type="do_nothing"):

    total_times = len(ts)-1
    total_points = len(xs)-1

    # solutions in time, space
    U = np.zeros((total_times+1, total_points+1))
    V = np.zeros((total_times+1, total_points+1))

    # initialize
    for i in range(total_points+1):
        U[0, i] = u_0(xs[i])
        V[0, i] = v_0(xs[i])

    U[1,:], V[1,:] = linear_forward_diff_step(c, U, V, 0, f, u_left, u_right, v_left, v_right, xs, ts, epsilon, bc_type)

    # time-stepping
    tau = ts[1] - ts[0]
    for n in range(1, total_times):
        if n % 1000 == 0:
            print(f"Linear: done w/ {n}/{total_times}")
        U[n+1,:], V[n+1,:] = linear_center_diff_step(c, U, V, n, f, u_left, u_right, v_left, v_right, xs, ts, epsilon, bc_type)
        
    return U, V


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def make_ticks(X_vals, indices, half_len):
    """
    Build a list of horizontal tick-mark segments for a vertical string.

    Each tick is a short horizontal line centred on x=0 at the current
    deformed y-position of each selected material point, so you can
    track where particles are transported to over time.

    Parameters
    ----------
    X_vals  : 1-D array  — deformed y-positions of every node
    indices : array-like — which node indices to mark
    half_len : float     — half the tick length in x-axis units

    Returns
    -------
    list of 2-point segments suitable for LineCollection
    """
    return [[(-half_len, X_vals[i]), (half_len, X_vals[i])] for i in indices]


if __name__ == "__main__":

    # -------------
    # set constants
    # -------------


    gravity_constant = 0.1 #9.80665
    k_constant = 3 #used for non-uniform mass density
    c = 0.5 #wave speed- kinda
    cfl = 0.05 # used to enforce cfl conditon
    stab_constant = 0.01 #strength of diffusion


    # space discretization
    total_points = 2**8
    a = 0
    b = 1

    h = (b - a)/(total_points+1)
    xs = [a + i*h for i in range(total_points + 1)]
    epsilon = stab_constant*(h**2) # 1*h**2 # stability term

    # time discretization
    t0 = 0
    T = 5

    tau = cfl * h / c          # keeps CFL = c·τ/h = cfl regardless of c
    print(f'timestep={tau:.6f},  CFL = {c * tau / h:.3f}')
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

    f = lambda t,x: 0 #- gravity_constant*(x**(k_constant+1) - 1/(k_constant + 2))

    # -------------------
    # boundary conditions
    # -------------------

    # available: dirichlet, do_nothing, reflecting, neumann_right, neumann_left, neumann
    bc_type = "neumann_right"

    # values at endpoints for u (represents either u or u' depending on whether dirichlet or neumann)
    u_left = lambda t: 0
    u_right = lambda t: gravity_constant

    # values at endpoints for v (represents either v or v' depending on whether dirichlet or neumann)
    v_left = lambda t: 0
    v_right = lambda t: 0

    # -------------------------
    # compute approx. and exact
    # -------------------------

    # approximate solutions (nonlinear and linear)
    U_nl, V_nl = solve_nonlinear(c, u_left, u_right, v_left, v_right, u_0, v_0, f, xs, ts, epsilon, bc_type)
    U_l,  V_l  = solve_linear  (c, u_left, u_right, v_left, v_right, u_0, v_0, f, xs, ts, epsilon, bc_type)

    # --- stability check ---
    for name, arr in [("U_nl", U_nl), ("U_l", U_l)]:
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(
                f"{name} contains NaN/Inf — scheme is unstable.\n"
                f"  CFL = {c * tau / h:.3f}  (must be < 1)\n"
                f"  Try reducing c or cfl."
            )

    #----------------------------
    # Subsample to animation frames
    # (avoids keeping every solver time step in memory)
    #----------------------------
    total_frames = 240
    total_solver_steps = len(ts)
    frame_idx = np.round(np.linspace(0, total_solver_steps - 1, total_frames)).astype(int)

    U_nl = U_nl[frame_idx, :]
    U_l  = U_l[ frame_idx, :]
    ts   = [ts[i] for i in frame_idx]
    # V_nl, V_l no longer needed after this point

    #----------------------------
    # Displacement to deformation
    #----------------------------
    string_length = 1
    ref_pos = np.array([i * string_length / total_points for i in range(total_points + 1)])

    X_nl = ref_pos[np.newaxis, :] + U_nl   # shape (total_frames, total_points+1)
    X_l  = ref_pos[np.newaxis, :] + U_l


    # --------------------
    # animate the solution
    # --------------------

    plot = True

    if plot:
        # total_frames already defined above; arrays pre-subsampled to that length

        # flip string so it hangs downward
        X_nl *= -1
        X_l  *= -1

        # U is always ≥ 0 for this BC setup:
        #   - Dirichlet at x=0 pins u=0 (can never go negative there)
        #   - Neumann at x=1 injects only positive strain (+gravity_constant)
        #   - Static equilibrium is u_s(x) = gravity_constant·x ≥ 0
        #   - System oscillates from 0 up to ~2·u_s, so U ∈ [0, 2·gravity_constant]
        # → Use the actual data range instead of a symmetric ±max scale,
        #   and a sequential colourmap so every colour encodes a real value.
        u_min_global = min(np.nanmin(U_nl), np.nanmin(U_l))
        u_max_global = max(np.nanmax(U_nl), np.nanmax(U_l))
        if not (np.isfinite(u_min_global) and np.isfinite(u_max_global)):
            raise RuntimeError("U contains NaN/Inf — check CFL condition.")
        if u_max_global <= u_min_global:
            u_max_global = u_min_global + 1.0
        norm = plt.Normalize(u_min_global, u_max_global)
        cmap = "plasma"

        # axis limits for the string panels
        X_all = np.concatenate([X_nl, X_l])
        X_min, X_max = np.nanmin(X_all), np.nanmax(X_all)
        margin = 0.15 * max(abs(X_max - X_min), 0.1)
        y_min = X_min - margin
        y_max = X_max + margin

        # ------------------------------------------------------------------
        # Figure layout:
        #   [nl_string | U_nl sideplot]   [l_string | U_l sideplot]  [cbar]
        # Four panels via GridSpec; colorbar gets its own axes.
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(14, 8))
        gs  = GridSpec(1, 4, figure=fig,
                       width_ratios=[3, 1.5, 3, 1.5],
                       wspace=0.06, left=0.07, right=0.89)

        ax_nl  = fig.add_subplot(gs[0])
        ax_unl = fig.add_subplot(gs[1], sharey=ax_nl)
        ax_l   = fig.add_subplot(gs[2], sharey=ax_nl)
        ax_ul  = fig.add_subplot(gs[3], sharey=ax_nl)

        # ---- string axes ----
        for ax, label in [(ax_nl, "Nonlinear"), (ax_l, "Linear")]:
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_title(label, fontsize=12)
        ax_nl.set_ylabel("Deformed position", fontsize=9)
        ax_l.tick_params(labelleft=False)

        # ---- U side-plot axes ----
        # y-axis: flipped material coordinate (matches string orientation)
        # x-axis: displacement U
        y_mat = -ref_pos   # 0 at top → -1 at bottom, matching the flipped string
        u_range  = u_max_global - u_min_global
        u_lim_lo = u_min_global - 0.1 * u_range
        u_lim_hi = u_max_global + 0.1 * u_range

        for ax, label in [(ax_unl, r"$U_{nl}(x,t)$"), (ax_ul, r"$U_{l}(x,t)$")]:
            ax.set_xlim(u_lim_lo, u_lim_hi)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(labelleft=False)
            ax.axvline(0, color='gray', lw=0.8, ls='--', zorder=1)
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("displacement", fontsize=8)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3, symmetric=True))

        xs_arr = np.zeros(total_points + 1)  # string lies on x=0

        # ---- initial strings (colored by displacement U) ----
        nl_coll = [colored_line(xs_arr, X_nl[0, :], U_nl[0, :], ax_nl,
                                linewidth=8, cmap=cmap, norm=norm)]
        l_coll  = [colored_line(xs_arr, X_l[0,  :], U_l[0,  :], ax_l,
                                linewidth=8, cmap=cmap, norm=norm)]

        # ---- tick marks at equally-spaced material points ----
        n_ticks      = 20
        tick_half    = 0.25
        tick_indices = np.round(np.linspace(0, total_points, n_ticks)).astype(int)

        nl_ticks = [LineCollection(make_ticks(X_nl[0, :], tick_indices, tick_half),
                                   colors='k', linewidths=2, zorder=4)]
        l_ticks  = [LineCollection(make_ticks(X_l[0,  :], tick_indices, tick_half),
                                   colors='k', linewidths=2, zorder=4)]
        ax_nl.add_collection(nl_ticks[0])
        ax_l.add_collection(l_ticks[0])

        # ---- masses at both ends ----
        nl_top, = ax_nl.plot([0], [X_nl[0,  0]], 's', color='dimgray',
                             markersize=14, zorder=5, label='Fixed end')
        nl_bot, = ax_nl.plot([0], [X_nl[0, -1]], 'o', color='black',
                             markersize=20, zorder=5, label='Hanging mass')
        l_top,  = ax_l.plot( [0], [X_l[0,   0]], 's', color='dimgray',
                             markersize=14, zorder=5)
        l_bot,  = ax_l.plot( [0], [X_l[0,  -1]], 'o', color='black',
                             markersize=20, zorder=5)
        ax_nl.legend(loc='upper left', fontsize=9)

        # ---- initial U profile lines + shaded fill ----
        u_line_nl, = ax_unl.plot(U_nl[0, :], y_mat,
                                 color='steelblue', lw=1.5, zorder=3)
        u_line_l,  = ax_ul.plot( U_l[0,  :], y_mat,
                                 color='tomato',    lw=1.5, zorder=3)
        u_fill_nl  = [ax_unl.fill_betweenx(y_mat, 0, U_nl[0, :],
                                            alpha=0.25, color='steelblue', zorder=2)]
        u_fill_l   = [ax_ul.fill_betweenx(  y_mat, 0, U_l[0,  :],
                                            alpha=0.25, color='tomato',    zorder=2)]

        # ---- shared colorbar ----
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        fig.colorbar(sm, cax=cbar_ax, label="Displacement  U")

        title = fig.suptitle(f"t = {ts[0]:.3f}", fontsize=13)

        def update(frame):
            n = frame  # arrays already subsampled to total_frames rows

            # -- strings --
            nl_coll[0].remove()
            l_coll[0].remove()
            nl_coll[0] = colored_line(xs_arr, X_nl[n, :], U_nl[n, :], ax_nl,
                                      linewidth=8, cmap=cmap, norm=norm)
            l_coll[0]  = colored_line(xs_arr, X_l[n,  :], U_l[n,  :], ax_l,
                                      linewidth=8, cmap=cmap, norm=norm)

            # -- tick marks --
            nl_ticks[0].remove()
            l_ticks[0].remove()
            nl_ticks[0] = LineCollection(make_ticks(X_nl[n, :], tick_indices, tick_half),
                                         colors='k', linewidths=2, zorder=4)
            l_ticks[0]  = LineCollection(make_ticks(X_l[n,  :], tick_indices, tick_half),
                                         colors='k', linewidths=2, zorder=4)
            ax_nl.add_collection(nl_ticks[0])
            ax_l.add_collection(l_ticks[0])

            # -- masses --
            nl_top.set_data([0], [X_nl[n,  0]])
            nl_bot.set_data([0], [X_nl[n, -1]])
            l_top.set_data( [0], [X_l[n,   0]])
            l_bot.set_data( [0], [X_l[n,  -1]])

            # -- U side plots --
            u_line_nl.set_xdata(U_nl[n, :])
            u_line_l.set_xdata( U_l[n,  :])
            u_fill_nl[0].remove()
            u_fill_l[0].remove()
            u_fill_nl[0] = ax_unl.fill_betweenx(y_mat, 0, U_nl[n, :],
                                                 alpha=0.25, color='steelblue', zorder=2)
            u_fill_l[0]  = ax_ul.fill_betweenx(  y_mat, 0, U_l[n,  :],
                                                 alpha=0.25, color='tomato',    zorder=2)

            title.set_text(f"t = {ts[n]:.3f}")
            return (nl_coll[0], l_coll[0],
                    nl_ticks[0], l_ticks[0],
                    nl_top, nl_bot, l_top, l_bot,
                    u_line_nl, u_line_l,
                    u_fill_nl[0], u_fill_l[0])

        ani = animation.FuncAnimation(fig=fig, func=update, frames=total_frames, interval=15, blit=False)

        import imageio_ffmpeg
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = animation.FFMpegWriter(fps=30)
        ani.save('animation.mp4', writer=writer)

        plt.show()
