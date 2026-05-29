from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import axes3d


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


@dataclass
class DuopolyPlotAnimation:
    animation: animation.FuncAnimation
    figure: plt.Figure
    axes: np.ndarray


def J_i_duopoly(s_i, x_i, m_i):
    return s_i * (x_i - m_i)


def J_1_duopoly_grad_1(x, p, m_1, S_d):
    return (2.0 * x[0] / p) - (m_1 / p) - (x[1] / p) - S_d


def J_1_duopoly_grad_2(x, p, m_1):
    return -(x[0] / p) + (m_1 / p)


def J_2_duopoly_grad_2(x, p, m_2):
    return -(x[0] / p) + (2.0 * x[1] / p) - (m_2 / p)


def J_2_duopoly_grad_1(x, p, m_2):
    return (x[1] / p) - (m_2 / p)


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


def plotter(simulation, J_2_ref):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    actions, cost, delta = axes

    actions.plot(simulation.time, simulation.actions_1, color="b", label=r"x_1")
    actions.plot(
        simulation.time, simulation.actions_2, color="cornflowerblue", label=r"x_2"
    )
    actions.plot(
        simulation.time,
        simulation.actions_1_deception,
        color="moccasin",
        label=r"x_1 with deception",
    )
    actions.plot(
        simulation.time,
        simulation.actions_2_deception,
        color="wheat",
        label=r"x_2 with deception",
    )
    actions.axhline(
        simulation.x_delta_star, linestyle="--", color="black", label=r"x_{\delta, 1}"
    )
    actions.axhline(simulation.x_star[1], linestyle="--", color="black", label=r"x_2^*")
    actions.set_xlabel("Time (s)")
    actions.set_ylabel("Action")
    actions.legend(loc="best", alignment="center", frameon=True)

    cost.plot(simulation.time, simulation.J_1)


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
    ax_actions.legend(
        loc="center right",
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        fontsize=10,
        handlelength=1.3,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.3,
    )

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
    ax_payoffs.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        fontsize=10,
        handlelength=1.3,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.3,
    )

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
    ax_delta.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        fontsize=10,
        handlelength=1.3,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.3,
    )

    return fig, axes


def animate_duopoly_plot(
    simulation,
    J_2_ref,
    frame_step=10,
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
    frame_indices = np.arange(0, len(time_values), frame_step, dtype=int)
    if frame_indices[-1] != len(time_values) - 1:
        frame_indices = np.append(frame_indices, len(time_values) - 1)

    fig, axes = plot_duopoly(simulation, J_2_ref)
    ax_actions, ax_payoffs, ax_delta = axes

    action_lines = ax_actions.lines[:4]
    payoff_lines = ax_payoffs.lines[:4]
    delta_line = ax_delta.lines[0]

    action_histories = [
        np.asarray(simulation.actions_1, dtype=float),
        np.asarray(simulation.actions_2, dtype=float),
        np.asarray(simulation.actions_1_deception, dtype=float),
        np.asarray(simulation.actions_2_deception, dtype=float),
    ]
    payoff_histories = [
        np.asarray(simulation.J_1_deception, dtype=float),
        np.asarray(simulation.J_2_deception, dtype=float),
        np.asarray(simulation.J_1, dtype=float),
        np.asarray(simulation.J_2, dtype=float),
    ]
    delta_history = np.asarray(simulation.delta, dtype=float)

    for line in [*action_lines, *payoff_lines, delta_line]:
        line.set_data([], [])

    time_box = fig.text(
        0.5,
        0.99,
        "",
        ha="center",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    def update(frame_number):
        idx = frame_indices[frame_number] + 1
        time_slice = time_values[:idx]

        for line, values in zip(action_lines, action_histories):
            line.set_data(time_slice, values[:idx])
        for line, values in zip(payoff_lines, payoff_histories):
            line.set_data(time_slice, values[:idx])
        delta_line.set_data(time_slice, delta_history[:idx])

        time_box.set_text(rf"$t = {time_values[idx - 1]:.2f}\,\mathrm{{s}}$")

        return (
            *action_lines,
            *payoff_lines,
            delta_line,
            time_box,
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

    return DuopolyPlotAnimation(animation=ani, figure=fig, axes=axes)


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
    # Use a clean white theme and larger font scaling for the animation figure.
    sns.set_theme(style="white", context="talk")
    # Override a few Matplotlib defaults so the figure styling matches the rest of the project.
    plt.rcParams.update(
        {
            # Use a serif font family for axis labels and annotations.
            "font.family": "serif",
            # Use Computer Modern math text so LaTeX-like symbols match the paper style.
            "mathtext.fontset": "cm",
            # Keep the top spine visible.
            "axes.spines.top": True,
            # Keep the right spine visible.
            "axes.spines.right": True,
        }
    )

    # Convert the simulated delta history into a NumPy array for indexing during animation.
    delta_values = np.asarray(simulation.delta, dtype=float)
    # Convert the simulated deceptive x_1 trajectory into a NumPy array.
    actions_1_deception = np.asarray(simulation.actions_1_deception, dtype=float)
    # Convert the simulated deceptive x_2 trajectory into a NumPy array.
    actions_2_deception = np.asarray(simulation.actions_2_deception, dtype=float)
    # Convert the simulation time stamps into a NumPy array.
    time_values = np.asarray(simulation.time, dtype=float)
    # Guard against malformed simulation inputs that do not contain a 1D delta history.
    if delta_values.ndim != 1 or delta_values.size == 0:
        raise ValueError("simulation.delta must be a non-empty one-dimensional array.")
    # Guard against the singular delta value where the deceptive reaction curve formula breaks down.
    if np.any(np.isclose(delta_values, 2.0)):
        raise ValueError(
            "simulation.delta cannot contain 2.0 because the deceptive RC is singular there."
        )
    # Guard against invalid animation subsampling settings.
    if frame_step <= 0:
        raise ValueError("frame_step must be a positive integer.")

    # Create a dense x_1 grid on which all reaction curves and level curves will be drawn.
    x_1 = np.linspace(x1_limits[0], x1_limits[1], 500)
    # Compute the fixed rotation point shared by the deceptive player-1 reaction curves.
    rotation_point = np.array([m[0], m[0] - S_d * p], dtype=float)
    # Subsample the simulation indices so the animation does not use every single time step.
    frame_indices = np.arange(0, len(delta_values), frame_step, dtype=int)
    # Always include the final time step so the animation reaches the terminal state.
    if frame_indices[-1] != len(delta_values) - 1:
        frame_indices = np.append(frame_indices, len(delta_values) - 1)
    # Extract the delta values that will actually be shown frame by frame.
    sampled_delta = delta_values[frame_indices]
    # Extract the deceptive x_1 trajectory values that correspond to the shown frames.
    sampled_actions_1 = actions_1_deception[frame_indices]
    # Extract the deceptive x_2 trajectory values that correspond to the shown frames.
    sampled_actions_2 = actions_2_deception[frame_indices]
    # Extract the simulation times that correspond to the shown frames.
    sampled_time = time_values[frame_indices]
    # Precompute the full family of deceptive reaction curves shown over the animation.
    rc_curves = np.array(
        [RC_1_deceptive_x2(x_1, m, S_d, p, delta_2) for delta_2 in sampled_delta]
    )
    # Replace a missing isoprofit specification with an empty list for simpler downstream logic.
    isoprofit_levels = [] if isoprofit_levels is None else list(isoprofit_levels)
    # Precompute the upper and lower player-2 isoprofit branches for each requested level.
    isoprofit_curves = [
        (level, *isoprofit_2(x_1, m, p, level)) for level in isoprofit_levels
    ]

    # Start the y-axis lower bound using the nominal player-1 RC, nominal player-2 RC, deceptive RCs, and trajectory.
    y_min = min(
        np.min(RC_1_nominal_x2(x_1, m, S_d, p)),
        np.min(RC_2_nominal_x2(x_1, m)),
        np.min(rc_curves),
        np.min(actions_2_deception),
    )
    # Start the y-axis upper bound using the same baseline objects.
    y_max = max(
        np.max(RC_1_nominal_x2(x_1, m, S_d, p)),
        np.max(RC_2_nominal_x2(x_1, m)),
        np.max(rc_curves),
        np.max(actions_2_deception),
    )
    # Expand the y-axis bounds to include any finite isoprofit branches that were requested.
    for _, upper_branch, lower_branch in isoprofit_curves:
        # Only update the limits if the upper branch contains at least one real point.
        if np.any(np.isfinite(upper_branch)):
            # Compare the current bounds against the minimum finite values of both branches.
            y_min = min(y_min, np.nanmin(upper_branch), np.nanmin(lower_branch))
            # Compare the current bounds against the maximum finite values of both branches.
            y_max = max(y_max, np.nanmax(upper_branch), np.nanmax(lower_branch))

    # Create the figure and axes for the animated reaction-curve plot.
    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    # Choose a colormap to encode the value of delta across the accumulated deceptive RCs.
    cmap = plt.cm.viridis
    # Normalize the sampled delta range so it maps cleanly into the colormap.
    norm = plt.Normalize(
        vmin=float(sampled_delta.min()), vmax=float(sampled_delta.max())
    )

    # Plot the fixed player-2 nominal reaction curve.
    ax.plot(
        x_1,
        RC_2_nominal_x2(x_1, m),
        color="black",
        linewidth=1.6,
        label="RC for player 2",
        zorder=2,
    )
    # Plot the nominal player-1 reaction curve as a dashed reference line.
    ax.plot(
        x_1,
        RC_1_nominal_x2(x_1, m, S_d, p),
        color="0.65",
        linewidth=2.0,
        linestyle="--",
        label="Nominal RC for player 1",
        zorder=1,
    )
    # Add a static title for the animated duopoly figure.
    ax.set_title("Duopoly Simulation with Deception")
    # Draw each requested player-2 isoprofit pair as a static background overlay.
    for level, upper_branch, lower_branch in isoprofit_curves:
        # Plot the upper branch for the current isoprofit level.
        ax.plot(
            x_1,
            upper_branch,
            color="0.55",
            linewidth=1.7,
            zorder=1,
        )
        # Plot the lower branch for the current isoprofit level.
        ax.plot(
            x_1,
            lower_branch,
            color="0.55",
            linewidth=1.7,
            zorder=1,
        )

        # Identify which points on the upper branch are real and therefore safe to label.
        valid_upper = np.isfinite(upper_branch)
        # Identify which points on the lower branch are real, even though the lower label is no longer shown.
        valid_lower = np.isfinite(lower_branch)
        # Only place a label if there is at least one valid point on the upper branch.
        if np.any(valid_upper):
            # Collect the indices of the finite upper-branch points.
            upper_valid_indices = np.where(valid_upper)[0]
            # Choose a hand-tuned relative position for the label so different J_2 labels spread out visually.
            upper_position = {
                750.0: 0.72,
                500.0: 0.52,
                250.0: 0.36,
            }.get(float(level), 0.58)
            # Convert the desired relative label position into an actual valid array index.
            upper_idx = upper_valid_indices[
                min(
                    int(upper_position * (len(upper_valid_indices) - 1)),
                    len(upper_valid_indices) - 1,
                )
            ]
            # Draw the J_2 label slightly offset from the chosen point on the upper branch.
            ax.text(
                x_1[upper_idx] + 1.1,
                upper_branch[upper_idx] + 1.2,
                rf"$J_2={level:g}$",
                color="0.25",
            )
        # Keep the lower-branch validity computation in place even though no lower-branch label is drawn.
        _ = valid_lower
    # Prepare a list to hold the previously revealed deceptive reaction-curve lines.
    persistent_rc_lines = []
    # Precreate one hidden line per sampled deceptive reaction curve so frames can reveal them cumulatively.
    for curve in rc_curves:
        # Create a hidden line artist for the current precomputed deceptive reaction curve.
        (line,) = ax.plot(
            x_1,
            curve,
            linewidth=2.0,
            alpha=0.18,
            color="0.75",
            visible=False,
            zorder=2,
        )
        # Store the line artist so the update function can reveal and recolor it later.
        persistent_rc_lines.append(line)

    # Create the highlighted current deceptive reaction-curve line that updates each frame.
    (current_rc_line,) = ax.plot(
        [],
        [],
        linewidth=2.8,
        zorder=3,
        label="RC for player 1 under deception",
    )
    # Create the animated deceptive state trajectory line in the x_1-x_2 plane.
    (trajectory_line,) = ax.plot(
        [],
        [],
        color="red",
        linewidth=1.4,
        label="Deceptive action trajectory",
        zorder=4,
    )
    # Create the moving point that marks the current deceptive action on the trajectory.
    (action_point,) = ax.plot(
        [],
        [],
        "o",
        color="black",
        markersize=6,
        label="Current deceptive action",
        zorder=5,
    )
    # Plot the terminal deceptive Nash equilibrium point as a static marker.
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
    # Plot the common rotation point shared by the deceptive player-1 reaction curves.
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

    # Create the annotation box that will display the current simulation time and delta value.
    delta_text = ax.text(
        0.03,
        0.65,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    # Build a dummy scalar mappable so Matplotlib can draw a colorbar for delta.
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Attach an empty array because the scalar mappable is only used for the colorbar.
    sm.set_array([])
    # Add the delta colorbar to the figure.
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    # Label the colorbar with the delta symbol.
    cbar.set_label(r"$\delta$")

    # Apply the requested horizontal plotting window.
    ax.set_xlim(*x1_limits)
    # Apply the vertical plotting window with a little padding above and below.
    ax.set_ylim(y_min - 1.5, y_max + 1.5)
    # Label the horizontal axis with x_1.
    ax.set_xlabel(r"$x_1$")
    # Label the vertical axis with x_2.
    ax.set_ylabel(r"$x_2$")
    # Draw the plot legend with compact spacing so it does not dominate the figure.
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

    # Define the per-frame update routine used by FuncAnimation.
    def update(frame_idx):
        # Read the current sampled delta value for this animation frame.
        delta_2 = float(sampled_delta[frame_idx])
        # Convert the current delta into a display color using the chosen colormap.
        color = cmap(norm(delta_2))
        # Grab the precomputed deceptive reaction curve corresponding to the current frame.
        current_curve = rc_curves[frame_idx]

        # Reveal and recolor all previously visited deceptive reaction-curve lines.
        for idx, line in enumerate(persistent_rc_lines):
            # Only touch lines that correspond to frames already reached by the animation.
            if idx <= frame_idx:
                # Make the line visible once its frame has been reached.
                line.set_visible(True)
                # Fade older curves slightly while keeping the current accumulated endpoint more visible.
                line.set_alpha(0.26 if idx < frame_idx else 0.35)
                # Color the stored curve according to its own delta value.
                line.set_color(cmap(norm(sampled_delta[idx])))
                # Make the most recent accumulated curve a bit thicker than older ones.
                line.set_linewidth(1.7 if idx < frame_idx else 2.2)

        # Update the highlighted current reaction curve.
        current_rc_line.set_data(x_1, current_curve)
        # Color the highlighted current reaction curve using the current delta color.
        current_rc_line.set_color(color)
        # Update the partial deceptive trajectory so it runs from the start up to the current frame.
        trajectory_line.set_data(
            sampled_actions_1[: frame_idx + 1],
            sampled_actions_2[: frame_idx + 1],
        )
        # Move the current-point marker to the current deceptive action.
        action_point.set_data(
            [sampled_actions_1[frame_idx]],
            [sampled_actions_2[frame_idx]],
        )
        # Rewrite the annotation box with the current time and delta.
        delta_text.set_text(
            rf"$t = {sampled_time[frame_idx]:.2f}\,\mathrm{{s}}$"
            + "\n"
            + rf"$\delta = {delta_2:.4f}$"
        )
        # Return the artists that changed so Matplotlib knows what to redraw.
        return (
            *persistent_rc_lines,
            current_rc_line,
            trajectory_line,
            action_point,
            terminal_point,
            rotation_marker,
            delta_text,
        )

    # Create the Matplotlib animation object using the update function defined above.
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval,
        repeat=True,
        repeat_delay=repeat_delay,
        blit=False,
    )
    # Force the first frame to render immediately so the static preview is populated.
    update(0)

    # Package the animation object together with its figure and axes for the caller.
    return ReactionCurveAnimation(animation=ani, figure=fig, axes=ax)


def duopoly_animation_3d(
    simulation,
    m,
    p,
    S_d,
    x1_limits=(30.0, 70.0),
    x2_limits=(30.0, 70.0),
    grid_size=90,
    frame_step=20,
    interval=200,
):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "text.parse_math": True,
        }
    )

    times = np.asarray(simulation.time, dtype=float)
    delta_values = np.asarray(simulation.delta, dtype=float)
    payoffs_1 = np.asarray(simulation.J_1_deception, dtype=float)
    payoffs_2 = np.asarray(simulation.J_2_deception, dtype=float)

    x1 = np.linspace(x1_limits[0], x1_limits[1], grid_size)
    x2 = np.linspace(x2_limits[0], x2_limits[1], grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    S2 = s_2_duopoly(p, X1, X2)
    J2_surface = J_i_duopoly(S2, X2, m[1])

    frame_indices = np.arange(0, len(delta_values), frame_step, dtype=int)
    if frame_indices[-1] != len(delta_values) - 1:
        frame_indices = np.append(frame_indices, len(delta_values) - 1)

    sampled_times = times[frame_indices]
    sampled_delta = delta_values[frame_indices]

    z_min = -1000.0

    z_max = 3000.0

    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    contour_fig, contour_ax = plt.subplots()
    plt.close(contour_fig)

    surfaces = []
    intersection_lines = []

    delta_text = ax.text2D(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    ax.set_xlim(*x1_limits)
    ax.set_ylim(*x2_limits)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$J_i$")
    ax.view_init(elev=28, azim=35)
    ax.legend(
        handles=[
            Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.62),
            Patch(facecolor="tab:orange", edgecolor="tab:orange", alpha=0.58),
        ],
        labels=[r"$\tilde{J}_1$", r"$J_2$"],
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
    )

    def draw_surfaces(delta_2):
        nonlocal surfaces, intersection_lines

        for surface in surfaces:
            surface.remove()
        for line in intersection_lines:
            line.remove()

        J1_surface = J_1_oblivious_duopoly(np.array([X1, X2]), delta_2, p, m[0], S_d)
        valid_j1 = (J1_surface > z_min) & (J1_surface <= z_max)
        valid_j2 = (J2_surface > z_min) & (J2_surface <= z_max)
        J1_surface = np.where(valid_j1, J1_surface, np.nan)
        J2_surface_clipped = np.where(valid_j2, J2_surface, np.nan)
        surface_j1 = ax.plot_surface(
            X1,
            X2,
            J1_surface,
            color="tab:blue",
            alpha=0.62,
            linewidth=0,
            antialiased=True,
            label=r"\~{J}_1",
        )
        surface_j2 = ax.plot_surface(
            X1,
            X2,
            J2_surface_clipped,
            color="tab:orange",
            alpha=0.58,
            linewidth=0,
            antialiased=True,
            label=r"J_2",
        )
        surfaces = [surface_j1, surface_j2]

        contour_ax.clear()
        contour = contour_ax.contour(
            X1, X2, J1_surface - J2_surface_clipped, levels=[0.0]
        )
        intersection_lines = []
        for segment in contour.allsegs[0]:
            if len(segment) < 2:
                continue
            x_segment = segment[:, 0]
            y_segment = segment[:, 1]
            z_segment = J_i_duopoly(
                s_2_duopoly(p, x_segment, y_segment), y_segment, m[1]
            )
            valid = (z_segment > z_min) & (z_segment <= z_max)
            if np.count_nonzero(valid) < 2:
                continue
            (line,) = ax.plot(
                x_segment[valid],
                y_segment[valid],
                z_segment[valid],
                color="red",
                linewidth=2.2,
                zorder=7,
            )
            intersection_lines.append(line)
        ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.6")

        contour.remove()

    def update(frame_idx):
        delta_2 = float(sampled_delta[frame_idx])
        draw_surfaces(delta_2)
        delta_text.set_text(
            rf"$t = {sampled_times[frame_idx]:.2f}\,\mathrm{{s}}$"
            + "\n"
            + rf"$\delta = {delta_2:.4f}$"
        )
        ax.set_title(r"3D payoff surfaces")

        return tuple(surfaces) + tuple(intersection_lines) + (delta_text,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval,
        repeat=True,
        blit=False,
    )

    return ReactionCurveAnimation(animation=ani, figure=fig, axes=ax)
