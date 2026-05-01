from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class DuopolySimulation:
    time: np.ndarray
    actions_1: np.ndarray
    actions_2: np.ndarray
    actions_1_deception: np.ndarray
    actions_2_deception: np.ndarray
    J_1: np.ndarray
    J_2: np.ndarray
    J_1_deception: np.ndarray
    J_2_deception: np.ndarray
    delta: np.ndarray
    x_star: np.ndarray
    x_delta_star: np.ndarray
    delta_star: float


@dataclass
class ReactionCurveAnimation:
    animation: animation.FuncAnimation
    figure: plt.Figure
    axes: plt.Axes


def J_i_duopoly(s_i, x_i, m_i):
    return s_i * (x_i - m_i)


def J_1_duopoly_grad_1(x, p, m_1, S_d):
    return (2.0 * x[0] / p) - (m_1 / p) - (x[1] / p) - S_d


def J_1_duopoly_grad_2(x, p, m_1):
    return -(x[0] / p) + (m_1 / p)


def J_2_duopoly_grad_2(x, p, m_2):
    return -(x[0] / p) + (2.0 * x[1] / p) - (m_2 / p)


def s_2_duopoly(p, x_1, x_2):
    return (x_1 - x_2) / p


def s_1_duopoly(x_1, x_2, p, S_d):
    return S_d - s_2_duopoly(p, x_1, x_2)


def J_1_oblivious_duopoly(x, delta_2, p, m_1, S_d):
    inflated_sales = s_1_duopoly(x[0], x[1], p, S_d) + (delta_2 / (2.0 * p)) * (
        x[0] - m_1
    )
    return inflated_sales * (x[0] - m_1)


def NE_duopoly_1(m_1, m_2, S_d, p):
    return (2.0 * m_1 + m_2 + 2.0 * S_d * p) / 3.0


def NE_duopoly_2(m_1, m_2, S_d, p):
    return (m_1 + 2.0 * m_2 + S_d * p) / 3.0


def DNE_duopoly(delta_2, m_1, m_2, S_d, p):
    denominator = 3.0 - 2.0 * delta_2
    x_1 = ((2.0 - 2.0 * delta_2) * m_1 + m_2 + 2.0 * S_d * p) / denominator
    x_2 = ((1.0 - delta_2) * m_1 + (2.0 - delta_2) * m_2 + S_d * p) / denominator
    return np.array([x_1, x_2], dtype=float)


def x_1_duopoly(u_1, a, omega_1, time_value):
    return u_1 + a * np.sin(omega_1 * time_value)


def x_2_duopoly(u_2, a, omega_1, omega_2, delta_2, time_value):
    return u_2 + a * (
        np.sin(omega_2 * time_value) + delta_2 * np.sin(omega_1 * time_value)
    )


def delta_2_update_duopoly_deception(x, epsilon, J_2_ref, p, m_2):
    s_i = s_2_duopoly(p, x[0], x[1])
    return epsilon * (J_i_duopoly(s_i, x[1], m_2) - J_2_ref)


# calculate the reaction curve of the duopoly problem
def RC_1_nominal(x, m, S_d, p):
    return (m[0] + x[1] + S_d * p) / 2


def RC_1_deceptive(x, m, S_d, p, delta_2):
    return (x[1] + S_d * p + (1 - delta_2) * m[0]) / (2 - delta_2)


def RC_1_nominal_x2(x_1, m, S_d, p):
    return 2.0 * x_1 - m[0] - S_d * p


def RC_1_deceptive_x2(x_1, m, S_d, p, delta_2):
    return (2.0 - delta_2) * x_1 - S_d * p - (1.0 - delta_2) * m[0]


def RC_2_nominal_x2(x_1, m):
    return 0.5 * (x_1 + m[1])


def isoprofit_2(x, m, p, J_2_ref):
    x = np.asarray(x, dtype=float)
    timestamps = len(x)
    actions_upper = np.full(timestamps, np.nan, dtype=float)
    actions_lower = np.full(timestamps, np.nan, dtype=float)

    for i in range(timestamps):
        discriminant = (x[i] - m[1]) ** 2 - 4.0 * p * J_2_ref
        if discriminant >= 0.0:
            sqrt_discriminant = np.sqrt(discriminant)
            actions_upper[i] = 0.5 * (x[i] + m[1] + sqrt_discriminant)
            actions_lower[i] = 0.5 * (x[i] + m[1] - sqrt_discriminant)
    return actions_upper, actions_lower


def simulate_duopoly(
    x0,
    a,
    k,
    omega_1,
    omega_2,
    J_2_ref,
    epsilon,
    S_d,
    p,
    m,
    horizon,
    dt=0.05,
):
    # edit how time is determined
    time = np.arange(0.0, horizon + dt, dt)

    x_nominal = np.array(x0, dtype=float).copy()
    x_deception = np.array(x0, dtype=float).copy()
    delta_2 = 0.0

    actions_1 = np.empty_like(time)
    actions_2 = np.empty_like(time)
    actions_1_deception = np.empty_like(time)
    actions_2_deception = np.empty_like(time)
    J_1 = np.empty_like(time)
    J_2 = np.empty_like(time)
    J_1_deception = np.empty_like(time)
    J_2_deception = np.empty_like(time)
    delta = np.empty_like(time)

    reaction_curves_deception = np.empty_like(time)

    for idx, time_value in enumerate(time):
        actions_1[idx] = x_1_duopoly(x_nominal[0], a, omega_1, time_value)
        actions_2[idx] = x_2_duopoly(x_nominal[1], a, omega_1, omega_2, 0.0, time_value)
        actions_1_deception[idx] = x_1_duopoly(x_deception[0], a, omega_1, time_value)
        actions_2_deception[idx] = x_2_duopoly(
            x_deception[1], a, omega_1, omega_2, delta_2, time_value
        )
        state_deception = np.array(
            [actions_1_deception[idx], actions_2_deception[idx]], dtype=float
        )

        s_1 = s_1_duopoly(actions_1[idx], actions_2[idx], p, S_d)
        s_2 = s_2_duopoly(p, actions_1[idx], actions_2[idx])
        s_1_deception = s_1_duopoly(
            actions_1_deception[idx], actions_2_deception[idx], p, S_d
        )
        s_2_deception = s_2_duopoly(
            p, actions_1_deception[idx], actions_2_deception[idx]
        )

        J_1[idx] = J_i_duopoly(s_1, actions_1[idx], m[0])
        J_2[idx] = J_i_duopoly(s_2, actions_2[idx], m[1])
        J_1_deception[idx] = J_1_oblivious_duopoly(
            state_deception, delta_2, p, m[0], S_d
        )
        J_2_deception[idx] = J_i_duopoly(s_2_deception, actions_2_deception[idx], m[1])
        delta[idx] = delta_2

        reaction_curves_deception[idx] = RC_1_deceptive(x_deception, m, S_d, p, delta_2)

        if idx == len(time) - 1:
            continue

        nominal_gradient = np.array(
            [
                J_1_duopoly_grad_1(x_nominal, p, m[0], S_d),
                J_2_duopoly_grad_2(x_nominal, p, m[1]),
            ]
        )
        deceptive_gradient = np.array(
            [
                J_1_duopoly_grad_1(x_deception, p, m[0], S_d)
                + delta_2 * J_1_duopoly_grad_2(x_deception, p, m[0]),
                J_2_duopoly_grad_2(x_deception, p, m[1]),
            ]
        )

        x_nominal = x_nominal - dt * k * nominal_gradient
        x_deception = x_deception - dt * k * deceptive_gradient
        delta_2 = delta_2 + dt * delta_2_update_duopoly_deception(
            np.array([actions_1_deception[idx], actions_2_deception[idx]]),
            epsilon,
            J_2_ref,
            p,
            m[1],
        )

    x_star = np.array(
        [NE_duopoly_1(m[0], m[1], S_d, p), NE_duopoly_2(m[0], m[1], S_d, p)],
        dtype=float,
    )
    delta_star = float(delta[-1])
    x_delta_star = DNE_duopoly(delta_star, m[0], m[1], S_d, p)

    return DuopolySimulation(
        time=time,
        actions_1=actions_1,
        actions_2=actions_2,
        actions_1_deception=actions_1_deception,
        actions_2_deception=actions_2_deception,
        J_1=J_1,
        J_2=J_2,
        J_1_deception=J_1_deception,
        J_2_deception=J_2_deception,
        delta=delta,
        x_star=x_star,
        x_delta_star=x_delta_star,
        delta_star=delta_star,
    )


def plot_duopoly(simulation, J_2_ref):
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    ax_actions, ax_payoffs, ax_delta = axes

    ax_actions.plot(
        simulation.time,
        simulation.actions_1,
        color="tab:blue",
        linestyle=":",
        linewidth=2.3,
        label=r"$x_1$",
    )
    ax_actions.plot(
        simulation.time,
        simulation.actions_2,
        color="tab:orange",
        linestyle=":",
        linewidth=2.3,
        label=r"$x_2$",
    )
    ax_actions.plot(
        simulation.time,
        simulation.actions_1_deception,
        color="tab:blue",
        linewidth=2.5,
        label=r"$x_1$ with deception",
    )
    ax_actions.plot(
        simulation.time,
        simulation.actions_2_deception,
        color="tab:orange",
        linewidth=2.5,
        label=r"$x_2$ with deception",
    )
    ax_actions.axhline(
        simulation.x_delta_star[0],
        color="black",
        linewidth=1.8,
        alpha=0.9,
        label=r"$x_{\delta,1}$",
    )
    ax_actions.axhline(
        simulation.x_star[1],
        color="black",
        linewidth=1.8,
        linestyle="--",
        alpha=0.9,
        label=r"$x_2^\ast$",
    )
    ax_actions.set_xlim(simulation.time[0], simulation.time[-1])
    ax_actions.set_xlabel("Time (s)")
    ax_actions.set_ylabel("Action")
    ax_actions.legend(loc="center right", frameon=True, fancybox=False, edgecolor="0.6")

    ax_payoffs.plot(
        simulation.time,
        simulation.J_1_deception,
        color="tab:blue",
        linewidth=2.5,
        label=r"$J_1$ with deception",
    )
    ax_payoffs.plot(
        simulation.time,
        simulation.J_2_deception,
        color="tab:orange",
        linewidth=2.5,
        label=r"$J_2$ with deception",
    )
    ax_payoffs.plot(
        simulation.time,
        simulation.J_1,
        color="tab:blue",
        linestyle=":",
        linewidth=2.3,
        label=r"$J_1$",
    )
    ax_payoffs.plot(
        simulation.time,
        simulation.J_2,
        color="tab:orange",
        linestyle=":",
        linewidth=2.3,
        label=r"$J_2$",
    )
    ax_payoffs.axhline(
        J_2_ref,
        color="black",
        linestyle="--",
        linewidth=1.8,
        dashes=(3, 3),
        label=r"$J_2^{ref}$",
    )
    ax_payoffs.set_xlim(simulation.time[0], simulation.time[-1])
    ax_payoffs.set_xlabel("Time (s)")
    ax_payoffs.set_ylabel(r"$J_i$")
    ax_payoffs.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.6")

    ax_delta.plot(simulation.time, simulation.delta, color="0.1", linewidth=2.2)
    ax_delta.axhline(
        simulation.delta_star,
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        dashes=(4, 3),
        label=rf"$\delta^\ast = {simulation.delta_star:.4f}$",
    )
    ax_delta.set_xlim(simulation.time[0], simulation.time[-1])
    ax_delta.set_xlabel("Time (s)")
    ax_delta.set_ylabel(r"$\delta$")
    ax_delta.legend(loc="bottom right", frameon=True, fancybox=False, edgecolor="0.6")

    return fig, axes


def run_duopoly(x, a, k, omega_1, omega_2, J_2_ref, epsilon, S_d, u, p, m, t):
    _ = u
    simulation = simulate_duopoly(
        x0=x,
        a=a,
        k=k,
        omega_1=omega_1,
        omega_2=omega_2,
        J_2_ref=J_2_ref,
        epsilon=epsilon,
        S_d=S_d,
        p=p,
        m=m,
        horizon=t,
    )
    fig, _ = plot_duopoly(simulation, J_2_ref)

    if plt.get_backend().lower() != "agg":
        plt.show()
    return simulation, fig


def plot_RC(simulations):
    fig, ax = plt.subplots()
    for i in range(len(simulations)):
        J_2 = str(simulations[i].J_2_deception)
        ax.plot(
            simulations[i].actions_1,
            simulations[i].actions_2,
            color="grey",
            label=r"$J_2=$" + J_2,
        )


def animate_reaction_curves(
    simulation,
    m,
    S_d,
    p,
    isoprofit_levels=None,
    x1_limits=(25.0, 78.0),
    frame_step=20,
    interval=350,
    repeat_delay=1000,
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

    delta_values = np.asarray(simulation.delta, dtype=float)
    actions_1_deception = np.asarray(simulation.actions_1_deception, dtype=float)
    actions_2_deception = np.asarray(simulation.actions_2_deception, dtype=float)
    time_values = np.asarray(simulation.time, dtype=float)
    if delta_values.ndim != 1 or delta_values.size == 0:
        raise ValueError("simulation.delta must be a non-empty one-dimensional array.")
    if np.any(np.isclose(delta_values, 2.0)):
        raise ValueError(
            "simulation.delta cannot contain 2.0 because the deceptive RC is singular there."
        )
    if frame_step <= 0:
        raise ValueError("frame_step must be a positive integer.")

    x_1 = np.linspace(x1_limits[0], x1_limits[1], 500)
    rotation_point = np.array([m[0], m[0] - S_d * p], dtype=float)
    frame_indices = np.arange(0, len(delta_values), frame_step, dtype=int)
    if frame_indices[-1] != len(delta_values) - 1:
        frame_indices = np.append(frame_indices, len(delta_values) - 1)
    sampled_delta = delta_values[frame_indices]
    sampled_actions_1 = actions_1_deception[frame_indices]
    sampled_actions_2 = actions_2_deception[frame_indices]
    sampled_time = time_values[frame_indices]
    rc_curves = np.array(
        [RC_1_deceptive_x2(x_1, m, S_d, p, delta_2) for delta_2 in sampled_delta]
    )
    isoprofit_levels = [] if isoprofit_levels is None else list(isoprofit_levels)
    isoprofit_curves = [
        (level, *isoprofit_2(x_1, m, p, level)) for level in isoprofit_levels
    ]

    y_min = min(
        np.min(RC_1_nominal_x2(x_1, m, S_d, p)),
        np.min(RC_2_nominal_x2(x_1, m)),
        np.min(rc_curves),
        np.min(actions_2_deception),
    )
    y_max = max(
        np.max(RC_1_nominal_x2(x_1, m, S_d, p)),
        np.max(RC_2_nominal_x2(x_1, m)),
        np.max(rc_curves),
        np.max(actions_2_deception),
    )
    for _, upper_branch, lower_branch in isoprofit_curves:
        if np.any(np.isfinite(upper_branch)):
            y_min = min(y_min, np.nanmin(upper_branch), np.nanmin(lower_branch))
            y_max = max(y_max, np.nanmax(upper_branch), np.nanmax(lower_branch))

    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    cmap = plt.cm.viridis
    norm = plt.Normalize(
        vmin=float(sampled_delta.min()), vmax=float(sampled_delta.max())
    )

    ax.plot(
        x_1,
        RC_2_nominal_x2(x_1, m),
        color="black",
        linewidth=1.6,
        label="RC for player 2",
        zorder=2,
    )
    ax.plot(
        x_1,
        RC_1_nominal_x2(x_1, m, S_d, p),
        color="0.65",
        linewidth=2.0,
        linestyle="--",
        label="Nominal RC for player 1",
        zorder=1,
    )
    ax.set_title("Duopoly Simulation with Deception")
    for level, upper_branch, lower_branch in isoprofit_curves:
        ax.plot(
            x_1,
            upper_branch,
            color="0.55",
            linewidth=1.7,
            zorder=1,
        )
        ax.plot(
            x_1,
            lower_branch,
            color="0.55",
            linewidth=1.7,
            zorder=1,
        )

        valid_upper = np.isfinite(upper_branch)
        valid_lower = np.isfinite(lower_branch)
        if np.any(valid_upper):
            upper_valid_indices = np.where(valid_upper)[0]
            upper_position = {
                750.0: 0.72,
                500.0: 0.52,
                250.0: 0.36,
            }.get(float(level), 0.58)
            upper_idx = upper_valid_indices[
                min(
                    int(upper_position * (len(upper_valid_indices) - 1)),
                    len(upper_valid_indices) - 1,
                )
            ]
            ax.text(
                x_1[upper_idx] + 1.1,
                upper_branch[upper_idx] + 1.2,
                rf"$J_2={level:g}$",
                color="0.25",
            )
    persistent_rc_lines = []
    for curve in rc_curves:
        (line,) = ax.plot(
            x_1,
            curve,
            linewidth=2.0,
            alpha=0.18,
            color="0.75",
            visible=False,
            zorder=2,
        )
        persistent_rc_lines.append(line)

    (current_rc_line,) = ax.plot(
        [],
        [],
        linewidth=2.8,
        zorder=3,
        label="RC for player 1 under deception",
    )
    (trajectory_line,) = ax.plot(
        [],
        [],
        color="red",
        linewidth=1.4,
        label="Deceptive action trajectory",
        zorder=4,
    )
    (action_point,) = ax.plot(
        [],
        [],
        "o",
        color="black",
        markersize=6,
        label="Current deceptive action",
        zorder=5,
    )
    (terminal_point,) = ax.plot(
        simulation.x_delta_star[0],
        simulation.x_delta_star[1],
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.0,
        linestyle="None",
        label="DNE",
        zorder=5,
    )
    (rotation_marker,) = ax.plot(
        rotation_point[0],
        rotation_point[1],
        marker="s",
        markersize=7,
        markerfacecolor="#ff8c8c",
        markeredgecolor="red",
        markeredgewidth=0.8,
        linestyle="None",
        label="Rotation Point",
        zorder=5,
    )

    delta_text = ax.text(
        0.03,
        0.65,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\delta$")

    ax.set_xlim(*x1_limits)
    ax.set_ylim(y_min - 1.5, y_max + 1.5)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        fontsize=11,
        handlelength=1.3,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.3,
    )

    def update(frame_idx):
        delta_2 = float(sampled_delta[frame_idx])
        color = cmap(norm(delta_2))
        current_curve = rc_curves[frame_idx]

        for idx, line in enumerate(persistent_rc_lines):
            if idx <= frame_idx:
                line.set_visible(True)
                line.set_alpha(0.26 if idx < frame_idx else 0.35)
                line.set_color(cmap(norm(sampled_delta[idx])))
                line.set_linewidth(1.7 if idx < frame_idx else 2.2)

        current_rc_line.set_data(x_1, current_curve)
        current_rc_line.set_color(color)
        trajectory_line.set_data(
            sampled_actions_1[: frame_idx + 1],
            sampled_actions_2[: frame_idx + 1],
        )
        action_point.set_data(
            [sampled_actions_1[frame_idx]],
            [sampled_actions_2[frame_idx]],
        )
        delta_text.set_text(
            rf"$t = {sampled_time[frame_idx]:.2f}\,\mathrm{{s}}$"
            + "\n"
            + rf"$\delta = {delta_2:.4f}$"
        )
        return (
            *persistent_rc_lines,
            current_rc_line,
            trajectory_line,
            action_point,
            terminal_point,
            rotation_marker,
            delta_text,
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

    return ReactionCurveAnimation(animation=ani, figure=fig, axes=ax)
