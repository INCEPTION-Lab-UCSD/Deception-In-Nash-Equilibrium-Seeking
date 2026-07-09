from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns


@dataclass
class AggregativeGameSimulation:
    time: np.ndarray
    actions_1: np.ndarray
    actions_2: np.ndarray
    J_1: np.ndarray
    J_2: np.ndarray
    reaction_curves: np.ndarray
    delta: np.ndarray


def J_1_aggregative(x_1, x_2):
    return np.pow(x_1, 4) + np.pow(x_1, 3) + 2.0 * x_1 * x_2


def J_2_aggregative(x_1, x_2):
    return np.exp(x_2) + np.pow(x_2, 2) + 1.1 * x_1 * x_2


def J_1_perceived(x_1, x_2, delta):
    return J_1_aggregative(x_1, x_2) + delta * x_1**2


# def J_2_perceived(x_1, x_2, delta):
#     return J_2_aggregative(x_1, x_2) + delta * 0.5 * 1.1 * x_2**2


def J_1_grad_1(x_1, x_2, delta):
    return 4.0 * np.pow(x_1, 3) + 3.0 * np.pow(x_1, 2) + 2.0 * x_2 + 2 * delta * x_1


def J_1_grad_2(x_1):
    return 2 * x_1


def J_2_grad_1(x_2):
    return 1.1 * x_2


def J_2_grad_2(x_1, x_2):
    return np.exp(x_2) + 2.0 * x_2 + 1.1 * x_1


def delta_update(x_1, x_2, epsilon, J_2_ref):
    return epsilon * (J_2_aggregative(x_1, x_2) - J_2_ref)


def x_i_deceptive(player_idx, deceived_indices, u_i, omega, a, delta, time_value):

    x_i = u_i + a * np.sin(omega[player_idx] * time_value)

    for idx in deceived_indices:
        x_i += a * delta * np.sin(omega[idx] * time_value)

    return x_i


def x_i_oblivious(player_idx, u_i, omega, a, time_value):
    return u_i + a * np.sin(omega[player_idx] * time_value)


def reaction_curve_1(x):
    return -(4 * np.pow(x[0], 3) + 3 * np.pow(x[0], 2)) / 2.0


def reaction_curve_1_deceptive(x_1, delta):
    return -(4.0 * np.pow(x_1, 3) + 3.0 * np.pow(x_1, 2) + 2.0 * delta * x_1) / 2.0


def simulation_aggregative(
    x0, a, k, omega_1, omega_2, J_2_ref, epsilon, horizon, dt=0.05
):
    time = np.arange(0.0, horizon + dt, dt)
    x = np.asarray(x0, dtype=float)
    actions_1 = np.empty_like(time)
    actions_2 = np.empty_like(time)
    delta = np.empty_like(time)

    reaction_curves = np.empty_like(time)
    omega = np.asarray([omega_1, omega_2])
    J_1 = np.empty_like(time)
    J_2 = np.empty_like(time)
    delta_cur = 0.0
    for idx, time_value in enumerate(time):
        actions_1[idx] = x_i_oblivious(0, x[0], omega, a, time_value)
        actions_2[idx] = x_i_deceptive(1, [0], x[1], omega, a, delta_cur, time_value)
        state = np.asarray([actions_1[idx], actions_2[idx]], dtype=float)
        J_1[idx] = J_1_perceived(actions_1[idx], actions_2[idx], delta_cur)
        J_2[idx] = J_2_aggregative(actions_1[idx], actions_2[idx])
        reaction_curves[idx] = reaction_curve_1(x)
        delta[idx] = delta_cur

        if idx == len(time) - 1:
            continue

        gradient = np.asarray(
            [
                J_1_grad_1(x[0], x[1], delta_cur),
                J_2_grad_2(x[0], x[1]),
            ]
        )

        x = x - dt * k * gradient
        delta_cur = delta_cur + dt * delta_update(x[0], x[1], epsilon, J_2_ref)

    return AggregativeGameSimulation(
        time=time,
        actions_1=actions_1,
        actions_2=actions_2,
        J_1=J_1,
        J_2=J_2,
        reaction_curves=reaction_curves,
        delta=delta,
    )


def simulation_aggregative_nominal(x0, a, k, omega_1, omega_2, horizon, dt=0.05):
    time = np.arange(0.0, horizon + dt, dt)
    x = np.asarray(x0, dtype=float)
    actions_1 = np.empty_like(time)
    actions_2 = np.empty_like(time)
    delta = np.zeros_like(time)
    omega = np.asarray([omega_1, omega_2])
    J_1 = np.empty_like(time)
    J_2 = np.empty_like(time)
    reaction_curves = np.empty_like(time)

    for idx, time_value in enumerate(time):
        actions_1[idx] = x_i_oblivious(0, x[0], omega, a, time_value)
        actions_2[idx] = x_i_oblivious(1, x[1], omega, a, time_value)
        J_1[idx] = J_1_aggregative(actions_1[idx], actions_2[idx])
        J_2[idx] = J_2_aggregative(actions_1[idx], actions_2[idx])
        reaction_curves[idx] = reaction_curve_1(x)

        if idx == len(time) - 1:
            continue

        gradient = np.asarray(
            [
                J_1_grad_1(x[0], x[1], 0.0),
                J_2_grad_2(x[0], x[1]),
            ]
        )
        x = x - dt * k * gradient

    return AggregativeGameSimulation(
        time=time,
        actions_1=actions_1,
        actions_2=actions_2,
        J_1=J_1,
        J_2=J_2,
        reaction_curves=reaction_curves,
        delta=delta,
    )


def reaction_curve_plot(simulation):
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    from scipy.optimize import brentq

    x_1 = np.linspace(-2, 2, 500)

    # Player 2's nominal RC: solve e^x2 + 2*x2 + 1.1*x1 = 0
    def rc_2_nominal(x_1_val, x_2_range=(-10, 10)):
        def eq(x_2):
            return np.exp(x_2) + 2 * x_2 + 1.1 * x_1_val

        try:
            return brentq(eq, *x_2_range)
        except ValueError:
            return np.nan

    # Player 1's deceptive RC family: delta tilts the x_1 coupling.
    def rc_1_deceptive(x_1_val, delta):
        return reaction_curve_1_deceptive(x_1_val, delta)

    delta_values = np.linspace(-3, 6, 20)
    cmap = plt.cm.turbo
    norm = plt.Normalize(vmin=-3, vmax=3)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.set_title("Aggregative Game")
    # Plot deceptive RCs for Player 1 colored by delta
    for delta in delta_values:
        rc = np.array([rc_1_deceptive(x, delta) for x in x_1])
        valid = np.abs(rc) <= 2
        ax.plot(
            x_1[valid],
            rc[valid],
            color=cmap(norm(delta)),
            linewidth=1.8,
        )

    # Plot Player 2's nominal RC in black
    rc_2 = np.array([rc_2_nominal(x_1_val) for x_1_val in x_1])
    valid_rc2 = np.abs(rc_2) <= 2
    ax.plot(
        x_1[valid_rc2],
        rc_2[valid_rc2],
        color="black",
        linewidth=2.2,
        label="RC for player 2",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\delta$", rotation=90)
    cbar.set_ticks([-3, -1.5, 0, 1.5, 3])
    cbar.set_ticklabels(
        [
            r"$\delta=-3$",
            r"$\delta=-1.5$",
            r"$\delta=0$",
            r"$\delta=1.5$",
            r"$\delta=3$",
        ]
    )

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$", rotation=0)
    ax.legend(
        loc="upper right", frameon=True, fancybox=False, edgecolor="0.6", fontsize=11
    )

    return fig, ax


def animate_aggregative_convergence(
    simulation,
    J_2_ref,
    nominal_simulation=None,
    frame_step=20,
    interval=40,
    repeat_delay=1200,
):
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    if frame_step <= 0:
        raise ValueError("frame_step must be a positive integer.")

    time_values = np.asarray(simulation.time, dtype=float)
    if time_values.ndim != 1 or time_values.size == 0:
        raise ValueError("simulation.time must be a non-empty one-dimensional array.")

    if nominal_simulation is None:
        nominal_simulation = simulation_aggregative_nominal(
            x0=np.array([0.0, 0.0], dtype=float),
            a=0.01,
            k=0.03,
            omega_1=470.75,
            omega_2=330.0,
            horizon=float(time_values[-1]),
            dt=float(time_values[1] - time_values[0]) if len(time_values) > 1 else 0.05,
        )

    nominal_time = np.asarray(nominal_simulation.time, dtype=float)
    if nominal_time.shape != time_values.shape:
        raise ValueError("nominal_simulation must match simulation on the time grid.")

    frame_indices = np.arange(0, len(time_values), frame_step, dtype=int)
    if frame_indices[-1] != len(time_values) - 1:
        frame_indices = np.append(frame_indices, len(time_values) - 1)

    sampled_time = time_values[frame_indices]
    j2_nominal = np.asarray(nominal_simulation.J_2, dtype=float)
    j2_deceptive = np.asarray(simulation.J_2, dtype=float)
    actions_1_nominal = np.asarray(nominal_simulation.actions_1, dtype=float)
    actions_2_nominal = np.asarray(nominal_simulation.actions_2, dtype=float)
    actions_1_deceptive = np.asarray(simulation.actions_1, dtype=float)
    actions_2_deceptive = np.asarray(simulation.actions_2, dtype=float)

    y_candidates = np.concatenate(
        [j2_nominal, j2_deceptive, np.asarray([float(J_2_ref)], dtype=float)]
    )
    y_min = float(np.nanmin(y_candidates))
    y_max = float(np.nanmax(y_candidates))
    y_padding = 0.08 * max(1.0, y_max - y_min)

    x_candidates = np.concatenate(
        [
            actions_1_nominal,
            actions_2_nominal,
            actions_1_deceptive,
            actions_2_deceptive,
        ]
    )
    x_min = float(np.nanmin(x_candidates))
    x_max = float(np.nanmax(x_candidates))
    x_padding = 0.12 * max(1.0, x_max - x_min)

    fig, ax_main = plt.subplots(figsize=(8.0, 5.8), constrained_layout=True)
    fig.set_facecolor("white")

    (j2_nominal_line,) = ax_main.plot(
        [],
        [],
        color="tab:blue",
        linestyle=":",
        linewidth=2.4,
        label=r"$J_2$",
    )
    (j2_deceptive_line,) = ax_main.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2.8,
        label="_nolegend_",
    )
    ref_line = ax_main.axhline(
        J_2_ref,
        color="black",
        linestyle="--",
        linewidth=1.9,
        dashes=(3, 3),
        label=r"$J_2^{ref}$",
    )

    ax_main.set_xlim(time_values[0], time_values[-1])
    ax_main.set_ylim(y_min - y_padding, y_max + y_padding)
    ax_main.set_xlabel("Time (s)")
    ax_main.set_ylabel(r"$J_2$")
    ax_main.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        fontsize=11,
        handlelength=1.8,
        handletextpad=0.5,
        borderpad=0.35,
        labelspacing=0.25,
    )

    ax_inset = ax_main.inset_axes([0.34, 0.48, 0.62, 0.47])
    (x1_nominal_line,) = ax_inset.plot(
        [], [], color="tab:blue", linestyle=":", linewidth=2.0
    )
    (x2_nominal_line,) = ax_inset.plot(
        [], [], color="tab:orange", linestyle=":", linewidth=2.0
    )
    (x1_deceptive_line,) = ax_inset.plot([], [], color="tab:blue", linewidth=2.6)
    (x2_deceptive_line,) = ax_inset.plot([], [], color="tab:orange", linewidth=2.6)
    ax_inset.set_xlim(time_values[0], time_values[-1])
    ax_inset.set_ylim(x_min - x_padding, x_max + x_padding)
    ax_inset.set_xticks([0, 50, 100, 150])
    ax_inset.set_yticks([-0.5, 0.0, 0.5])
    ax_inset.tick_params(direction="in")
    ax_inset.text(
        0.80,
        0.55,
        r"$x_1$",
        transform=ax_inset.transAxes,
        fontsize=22,
    )
    ax_inset.text(
        0.80,
        0.35,
        r"$x_2$",
        transform=ax_inset.transAxes,
        fontsize=22,
    )
    ax_inset.annotate(
        "",
        xy=(0.49, 0.66),
        xycoords="axes fraction",
        xytext=(0.66, 0.54),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "0.2"},
    )
    ax_inset.annotate(
        "",
        xy=(0.49, 0.30),
        xycoords="axes fraction",
        xytext=(0.66, 0.43),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "0.2"},
    )

    def update(frame_number):
        idx = frame_indices[frame_number] + 1
        time_slice = time_values[:idx]

        j2_nominal_line.set_data(time_slice, j2_nominal[:idx])
        j2_deceptive_line.set_data(time_slice, j2_deceptive[:idx])
        x1_nominal_line.set_data(time_slice, actions_1_nominal[:idx])
        x2_nominal_line.set_data(time_slice, actions_2_nominal[:idx])
        x1_deceptive_line.set_data(time_slice, actions_1_deceptive[:idx])
        x2_deceptive_line.set_data(time_slice, actions_2_deceptive[:idx])

        return (
            j2_nominal_line,
            j2_deceptive_line,
            ref_line,
            x1_nominal_line,
            x2_nominal_line,
            x1_deceptive_line,
            x2_deceptive_line,
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval,
        repeat=True,
        repeat_delay=repeat_delay,
        blit=False,
    )
    update(0)

    return ani, fig, (ax_main, ax_inset)


def animate_reaction_curves(
    simulation,
    x1_limits=(-4, 4),
    x2_limits=(-4, 4),
    delta_limits=None,
    frame_step=20,
    interval=40,
    repeat_delay=1200,
):
    from scipy.optimize import brentq

    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    x_1 = np.linspace(*x1_limits, 500)

    def rc_2_nominal(x_1_val, x_2_range=(-10, 10)):
        def eq(x_2):
            return np.exp(x_2) + 2 * x_2 + 1.1 * x_1_val

        try:
            return brentq(eq, *x_2_range)
        except ValueError:
            return np.nan

    def rc_1_deceptive(x_1_val, delta):
        return reaction_curve_1_deceptive(x_1_val, delta)

    # Subsample frames
    time_values = np.asarray(simulation.time, dtype=float)
    delta_values = np.asarray(simulation.delta, dtype=float)
    frame_indices = np.arange(0, len(time_values), frame_step, dtype=int)
    if frame_indices[-1] != len(time_values) - 1:
        frame_indices = np.append(frame_indices, len(time_values) - 1)

    sampled_time = time_values[frame_indices]
    sampled_delta = delta_values[frame_indices]

    # Precompute all deceptive RCs for player 1
    rc_curves = []
    for delta in sampled_delta:
        rc = np.array([rc_1_deceptive(x, delta) for x in x_1])
        rc_curves.append(rc)

    cmap = plt.cm.viridis
    if delta_limits is None:
        delta_min = float(np.nanmin(sampled_delta))
        delta_max = float(np.nanmax(sampled_delta))
    else:
        delta_min, delta_max = delta_limits
    norm = plt.Normalize(vmin=delta_min, vmax=delta_max)

    fig, ax_rc = plt.subplots(figsize=(8, 7), constrained_layout=True)

    # --- Reaction curve axis ---
    # Plot static Player 2 RC
    rc_2 = np.array([rc_2_nominal(x_1_val) for x_1_val in x_1])
    valid_rc2 = np.abs(rc_2) <= x2_limits[1]
    ax_rc.plot(
        x_1[valid_rc2],
        rc_2[valid_rc2],
        color="black",
        linewidth=2.2,
        label="RC for player 2",
    )
    fig.suptitle("Aggregative Game Reaction Curve")

    # Persistent accumulated RC lines
    persistent_lines = []
    for _ in range(len(frame_indices)):
        (line,) = ax_rc.plot(
            [], [], linewidth=1.8, alpha=0.2, color="0.75", visible=False
        )
        persistent_lines.append(line)

    # Current highlighted RC
    (current_rc_line,) = ax_rc.plot(
        [], [], linewidth=2.5, zorder=3, label="RC for player 1 (current)"
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_rc, pad=0.02)
    cbar.set_label(r"$\delta$", rotation=90)

    ax_rc.set_xlim(*x1_limits)
    ax_rc.set_ylim(*x2_limits)
    ax_rc.set_xlabel(r"$x_1$")
    ax_rc.set_ylabel(r"$x_2$", rotation=0)
    ax_rc.legend(
        loc="upper right", frameon=True, fancybox=False, edgecolor="0.6", fontsize=10
    )

    # Annotation box for time and delta
    info_text = ax_rc.text(
        0.03,
        0.97,
        "",
        transform=ax_rc.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
        fontsize=10,
    )

    def update(frame_idx):
        delta = float(sampled_delta[frame_idx])
        color = cmap(norm(delta))
        rc = rc_curves[frame_idx]
        valid = np.abs(rc) <= x2_limits[1]

        # Reveal and recolor persistent lines
        for i, line in enumerate(persistent_lines):
            if i <= frame_idx:
                line.set_visible(True)
                line.set_color(cmap(norm(sampled_delta[i])))
                line.set_alpha(0.2 if i < frame_idx else 0.35)
                line.set_linewidth(1.5 if i < frame_idx else 2.0)
                line.set_data(x_1[valid], rc[valid])

        # Current RC
        current_rc_line.set_data(x_1[valid], rc[valid])
        current_rc_line.set_color(color)

        # Info text
        info_text.set_text(
            rf"$t = {sampled_time[frame_idx]:.1f}\,\mathrm{{s}}$"
            + "\n"
            + rf"$\delta = {delta:.4f}$"
        )
        return (*persistent_lines, current_rc_line, info_text)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval,
        repeat=True,
        repeat_delay=repeat_delay,
        blit=False,
    )
    update(0)

    return ani, fig, ax_rc


def run_aggregative_example():
    simulation = simulation_aggregative(
        np.array([0.0, 0.0]),
        a=0.01,
        k=0.03,
        omega_1=470.75,
        omega_2=330,
        epsilon=0.001,
        J_2_ref=-0.1,
        horizon=150.0,
    )

    return animate_reaction_curves(simulation)


def run_aggregative_convergence_example():
    simulation = simulation_aggregative(
        np.array([-0.2, 0.1]),
        a=0.01,
        k=0.03,
        omega_1=470.75,
        omega_2=330.0,
        J_2_ref=-0.1,
        epsilon=0.001,
        horizon=150.0,
        dt=0.05,
    )
    nominal_simulation = simulation_aggregative_nominal(
        np.array([-0.2, 0.1]),
        a=0.01,
        k=0.03,
        omega_1=470.75,
        omega_2=330.0,
        horizon=150.0,
        dt=0.05,
    )
    return animate_aggregative_convergence(
        simulation=simulation,
        nominal_simulation=nominal_simulation,
        J_2_ref=0.61,
    )


if __name__ == "__main__":
    anim = run_aggregative_example()
    _ = anim
    plt.show()
