from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class QuadraticSimulation:
    time: np.ndarray
    nominal_actions: np.ndarray
    deceptive_actions: np.ndarray
    nominal_payoffs: np.ndarray
    deceptive_payoffs: np.ndarray
    delta: np.ndarray
    x_star: np.ndarray
    x_delta_star: np.ndarray
    J_star: np.ndarray
    J_delta_star: np.ndarray
    delta_star: float
    attacker_payoff_idx: int
    victim_idx: int
    attacked_coord_idx: int
    J_ref: float


@dataclass
class QuadraticAnimation:
    animation: animation.FuncAnimation
    figure: plt.Figure
    axes: plt.Axes


def J_i(x, Q, b, p=0.0):
    if not np.allclose(Q, Q.T):
        ValueError("Q must be a symmetric matrix")
    x = np.asarray(x, dtype=float)
    Q = np.asarray(Q, dtype=float)
    b = np.asarray(b, dtype=float)
    return 0.5 * x.T @ Q @ x + b.T @ x + p


def quadratic_player_pseudogradient(Q, x, b, player_idx):
    x = np.asarray(x, dtype=float)
    Q = np.asarray(Q, dtype=float)
    b = np.asarray(b, dtype=float)

    return Q[player_idx] @ x + b[player_idx]


# x^* = -\twiddle{Q}^{-1}b
def solve_quadratic_ne(Q_list, b_list):
    Q_pseudogradient = np.vstack([Q_list[idx][idx] for idx in range(len(Q_list))])
    b_pseudogradient = np.array(
        [b_list[idx][idx] for idx in range(len(Q_list))],
        dtype=float,
    )
    return np.linalg.solve(Q_pseudogradient, -b_pseudogradient)


def solve_quadratic_deceptive_ne(
    Q_list,
    b_list,
    attacked_coord_idx,
    victim_idx,
    delta,
):
    Q_pseudogradient = []
    b_pseudogradient = []
    for player_idx in range(len(Q_list)):
        Q_pseudogradient_row = np.asarray(
            Q_list[player_idx][player_idx], dtype=float
        ).copy()
        b_pseudogradient_element = float(b_list[player_idx][player_idx])
        if player_idx == victim_idx:
            Q_pseudogradient_row = Q_pseudogradient_row + delta * np.asarray(
                Q_list[player_idx][attacked_coord_idx], dtype=float
            )

            b_pseudogradient_element = b_pseudogradient_element + delta * float(
                b_list[player_idx][attacked_coord_idx]
            )
        Q_pseudogradient.append(Q_pseudogradient_row)
        b_pseudogradient.append(b_pseudogradient_element)

    return np.linalg.solve(
        np.vstack(Q_pseudogradient),
        -np.asarray(b_pseudogradient, dtype=float),
    )


def nominal_payoff_plot_indices(time_values, sample_interval=5.0):
    time_values = np.asarray(time_values, dtype=float)
    if len(time_values) <= 2:
        return np.arange(len(time_values), dtype=int)

    time_deltas = np.diff(time_values)
    positive_time_deltas = time_deltas[time_deltas > 0.0]
    if len(positive_time_deltas) == 0:
        return np.arange(len(time_values), dtype=int)

    dt = float(np.median(positive_time_deltas))
    sample_step = max(1, int(round(sample_interval / dt)))
    indices = np.arange(0, len(time_values), sample_step, dtype=int)
    if indices[-1] != len(time_values) - 1:
        indices = np.append(indices, len(time_values) - 1)

    return indices


def prices(omega_list, u, a, attacker_index, victim_index, t, delta):
    prices = []
    num_players = len(omega_list)
    for player_idx in range(num_players):
        if player_idx == attacker_index:
            player_price = u[player_idx] + a * (
                np.sin(omega_list[player_idx] * t)
                + delta * np.sin(omega_list[victim_index] * t)
            )
            prices.append(player_price)

    return np.asarray(prices)


def example_6_parameters():
    q1_raw = np.array(
        [
            [7.0, 3.0, 1.0],
            [2.0, 6.0, -2.0],
            [-3.0, 3.0, 9.0],
        ],
        dtype=float,
    )
    q2_raw = np.array(
        [
            [7.0, -2.0, -3.0],
            [-1.0, 8.0, 1.0],
            [4.0, -3.0, 2.0],
        ],
        dtype=float,
    )
    q3_raw = np.array(
        [
            [-3.0, 2.0, 2.0],
            [-2.0, 2.0, 4.0],
            [3.0, -2.0, 7.0],
        ],
        dtype=float,
    )

    return {
        "Q_list": [
            0.05 * (q1_raw + q1_raw.T),
            0.05 * (q2_raw + q2_raw.T),
            0.025 * (q3_raw + q3_raw.T),
        ],
        "b_list": [
            np.array([2.0, 2.0, -3.0], dtype=float),
            np.array([-1.0, -3.0, 3.0], dtype=float),
            np.array([2.0, 7.0, -3.0], dtype=float),
        ],
        "a": 0.04,
        "k": 0.02,
        "omega": np.array([3172.8, 2044.4, 3057.6], dtype=float),
        "attacker_payoff_idx": 0,
        "victim_idx": 2,
        "attacked_coord_idx": 0,
        "J_ref": 5.0,
        "epsilon": 1.0e-3,
        "horizon": 1500.0,
        "dt": 0.02,
    }


def simulate_quadratic_game(
    Q_list,
    b_list,
    a,
    k,
    omega,
    attacker_payoff_idx,
    victim_idx,
    attacked_coord_idx,
    J_ref,
    epsilon,
    horizon,
    dt=0.5,
    x0=None,
):
    player_count = len(Q_list)
    if x0 is None:
        x0 = np.zeros(player_count, dtype=float)
    time = np.arange(0.0, horizon + dt, dt)

    nominal_state = np.array(x0, dtype=float).copy()
    deceptive_state = np.array(x0, dtype=float).copy()
    delta_value = 0.0

    nominal_actions = np.empty((len(time), player_count), dtype=float)
    deceptive_actions = np.empty((len(time), player_count), dtype=float)
    nominal_payoffs = np.empty((len(time), player_count), dtype=float)
    deceptive_payoffs = np.empty((len(time), player_count), dtype=float)
    delta_history = np.empty(len(time), dtype=float)

    for time_idx, time_value in enumerate(time):
        probe_signal = a * np.sin(omega * time_value)
        nominal_actions[time_idx] = nominal_state + probe_signal
        deceptive_actions[time_idx] = deceptive_state + probe_signal
        deceptive_actions[time_idx, attacked_coord_idx] += (
            a * delta_value * np.sin(omega[victim_idx] * time_value)
        )

        for player_idx in range(player_count):
            nominal_payoffs[time_idx, player_idx] = J_i(
                nominal_actions[time_idx],
                Q_list[player_idx],
                b_list[player_idx],
            )
            deceptive_payoffs[time_idx, player_idx] = J_i(
                deceptive_actions[time_idx],
                Q_list[player_idx],
                b_list[player_idx],
            )

        delta_history[time_idx] = delta_value

        if time_idx == len(time) - 1:
            continue

        nominal_gradient = np.array(
            [
                quadratic_player_pseudogradient(
                    Q_list[player_idx], nominal_state, b_list[player_idx], player_idx
                )
                for player_idx in range(player_count)
            ],
            dtype=float,
        )
        deceptive_gradient = np.array(
            [
                quadratic_player_pseudogradient(
                    Q_list[player_idx],
                    deceptive_state,
                    b_list[player_idx],
                    player_idx,
                )
                for player_idx in range(player_count)
            ],
            dtype=float,
        )
        deceptive_gradient[victim_idx] += delta_value * quadratic_player_pseudogradient(
            Q_list[victim_idx],
            deceptive_state,
            b_list[victim_idx],
            attacked_coord_idx,
        )

        nominal_state = nominal_state - dt * k * nominal_gradient
        deceptive_state = deceptive_state - dt * k * deceptive_gradient
        delta_value = delta_value + dt * epsilon * (
            deceptive_payoffs[time_idx, attacker_payoff_idx] - J_ref
        )

    x_star = solve_quadratic_ne(Q_list, b_list)
    delta_star = float(delta_history[-1])
    x_delta_star = solve_quadratic_deceptive_ne(
        Q_list,
        b_list,
        attacked_coord_idx,
        victim_idx,
        delta_star,
    )
    J_star = np.array(
        [J_i(x_star, Q_list[idx], b_list[idx]) for idx in range(player_count)],
        dtype=float,
    )
    J_delta_star = np.array(
        [J_i(x_delta_star, Q_list[idx], b_list[idx]) for idx in range(player_count)],
        dtype=float,
    )

    return QuadraticSimulation(
        time=time,
        nominal_actions=nominal_actions,
        deceptive_actions=deceptive_actions,
        nominal_payoffs=nominal_payoffs,
        deceptive_payoffs=deceptive_payoffs,
        delta=delta_history,
        x_star=x_star,
        x_delta_star=x_delta_star,
        J_star=J_star,
        J_delta_star=J_delta_star,
        delta_star=delta_star,
        attacker_payoff_idx=attacker_payoff_idx,
        victim_idx=victim_idx,
        attacked_coord_idx=attacked_coord_idx,
        J_ref=J_ref,
    )


def animate_quadratic(simulation, frame_step=10, interval=40, repeat_delay=1000):

    if frame_step <= 0:
        raise ValueError("frame_step must be positive")

    time_values = np.asarray(simulation.time, dtype=float)
    frame_indices = np.arange(0, len(time_values), frame_step, dtype=int)

    if frame_indices[-1] != len(time_values) - 1:
        frame_indices = np.append(frame_indices, len(time_values) - 1)

    fig, ax = plot_quadratic_example_6(simulation)
    nominal_payoffs = np.asarray(simulation.nominal_payoffs, dtype=float)
    deceptive_payoffs = np.asarray(simulation.deceptive_payoffs, dtype=float)

    nominal_lines = [line for line in ax.lines if line.get_gid() == "nominal_payoff"]
    deceptive_lines = [
        line for line in ax.lines if line.get_gid() == "deceptive_payoff"
    ]
    animated_lines = nominal_lines + deceptive_lines

    for line in animated_lines:
        line.set_data([], [])

    time_box = ax.text(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    delta_box = ax.text(
        0.03,
        0.84,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    def update(frame_number):
        idx = frame_indices[frame_number] + 1
        time_slice = time_values[:idx]
        nominal_indices = nominal_payoff_plot_indices(time_slice)

        for player_idx, line in enumerate(nominal_lines):
            line.set_data(
                time_slice[nominal_indices],
                nominal_payoffs[:idx, player_idx][nominal_indices],
            )
        for player_idx, line in enumerate(deceptive_lines):
            line.set_data(time_slice, deceptive_payoffs[:idx, player_idx])

        time_box.set_text(rf"$t = {time_values[idx - 1]:.1f}\,\mathrm{{s}}$")
        delta_box.set_text(rf"$\delta = {simulation.delta[idx - 1]:.4f}$")

        return (*animated_lines, time_box, delta_box)

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

    return QuadraticAnimation(animation=ani, figure=fig, axes=ax)


def plot_quadratic_example_6(simulation):
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.8), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = [r"$J_1$", r"$J_2$", r"$J_3$"]
    nominal_indices = nominal_payoff_plot_indices(simulation.time)

    # loop through player payoffs
    line_nominal = []
    line_deception = []
    line_NE = []
    line_NE_deception = []
    for idx, (color, label) in enumerate(zip(colors, labels)):
        (line_nominal_idx,) = ax.plot(
            simulation.time[nominal_indices],
            simulation.nominal_payoffs[nominal_indices, idx],
            color=color,
            linestyle=(0, (1.2, 2.0)),
            linewidth=5.0,
            dash_capstyle="butt",
            label=label + " nominal",
            zorder=2,
        )
        line_nominal_idx.set_gid("nominal_payoff")

        (line_deception_idx,) = ax.plot(
            simulation.time,
            simulation.deceptive_payoffs[:, idx],
            color=color,
            linewidth=1.6,
            label=label + " with deception",
        )
        line_deception_idx.set_gid("deceptive_payoff")

        line_nominal.append(line_nominal_idx)
        line_deception.append(line_deception_idx)

        line_NE_idx = ax.axhline(
            simulation.J_star[idx],
            color="black",
            linewidth=1.6,
            linestyle="--",
            dashes=(6, 4),
            alpha=0.75,
            zorder=0,
        )
        line_NE_deception_idx = ax.axhline(
            simulation.J_delta_star[idx],
            color="black",
            linewidth=1.6,
            alpha=0.75,
            zorder=0,
        )

        line_NE.append(line_NE_idx)
        line_NE_deception.append(line_NE_deception_idx)

    nominal_handles = []
    deceptive_handles = []
    for color, label in zip(colors, labels):
        nominal_handle = plt.Line2D(
            [0],
            [0],
            color=color,
            linewidth=5.0,
            linestyle=(0, (1.2, 2.0)),
            dash_capstyle="butt",
            label=label + " nominal",
        )
        deceptive_handle = plt.Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.6,
            label=label + " with deception",
        )
        nominal_handles.append(nominal_handle)
        deceptive_handles.append(deceptive_handle)

    trajectory_legend = ax.legend(
        handles=nominal_handles + deceptive_handles,
        loc="lower right",
        ncols=2,
        title="Payoff trajectories",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="0.55",
        fontsize=9,
        title_fontsize=10,
        handlelength=4.5,
        columnspacing=1.1,
        handletextpad=0.55,
        borderpad=0.55,
        labelspacing=0.35,
    )
    ax.add_artist(trajectory_legend)

    equilibrium_handles = [
        plt.Line2D(
            [0],
            [0],
            color="0.2",
            linewidth=1.8,
            linestyle="--",
            dashes=(6, 4),
            label=r"Nominal NE payoff, $J_i(x^*)$",
        ),
        plt.Line2D(
            [0],
            [0],
            color="0.2",
            linewidth=1.8,
            label=r"Deceptive NE payoff, $J_i(x_\delta^*)$",
        ),
    ]
    ax.legend(
        handles=equilibrium_handles,
        loc="upper right",
        title="Equilibrium levels",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="0.55",
        fontsize=9,
        title_fontsize=10,
        handlelength=2.4,
        handletextpad=0.55,
        borderpad=0.55,
        labelspacing=0.35,
    )

    payoff_stack = np.concatenate(
        [
            np.ravel(simulation.nominal_payoffs),
            np.ravel(simulation.deceptive_payoffs),
            np.ravel(simulation.J_star),
            np.ravel(simulation.J_delta_star),
            np.array([simulation.J_ref], dtype=float),
        ]
    )
    y_min = float(np.min(payoff_stack))
    y_max = float(np.max(payoff_stack))
    y_pad = max(1.0, 0.08 * (y_max - y_min))

    ax.set_xlim(simulation.time[0], simulation.time[-1])
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xticks([0.0, 500.0, 1000.0, 1500.0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$J_i$")

    return fig, ax


def run_quadratic_example_6():
    simulation = simulate_quadratic_game(**example_6_parameters())
    plot_quadratic_example_6(simulation)

    return animate_quadratic(simulation, frame_step=400, interval=40)
