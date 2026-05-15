from dataclasses import dataclass

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


def J_i_quadratic(x, Q, b, p=0.0):
    x = np.asarray(x, dtype=float)
    Q = np.asarray(Q, dtype=float)
    b = np.asarray(b, dtype=float)
    return 0.5 * x.T @ Q @ x + b.T @ x + p


def quadratic_partial(x, Q, b, player_idx):
    return float(np.asarray(Q[player_idx], dtype=float) @ x + b[player_idx])


def solve_quadratic_ne(Q_list, b_list):
    system_matrix = np.vstack([Q_list[idx][idx] for idx in range(len(Q_list))])
    offset_vector = np.array(
        [b_list[idx][idx] for idx in range(len(Q_list))],
        dtype=float,
    )
    return np.linalg.solve(system_matrix, -offset_vector)


def solve_quadratic_deceptive_ne(Q_list, b_list, deceiver_idx, victim_idx, delta):
    system_matrix = []
    offset_vector = []
    for player_idx in range(len(Q_list)):
        row = np.asarray(Q_list[player_idx][player_idx], dtype=float).copy()
        offset = float(b_list[player_idx][player_idx])
        if player_idx == victim_idx:
            row = row + delta * np.asarray(Q_list[player_idx][deceiver_idx], dtype=float)
            offset = offset + delta * float(b_list[player_idx][deceiver_idx])
        system_matrix.append(row)
        offset_vector.append(offset)

    return np.linalg.solve(
        np.vstack(system_matrix),
        -np.asarray(offset_vector, dtype=float),
    )


def example_6_parameters():
    return {
        "Q_list": [
            np.array(
                [
                    [0.7, 0.25, -0.1],
                    [0.25, 0.6, 0.05],
                    [-0.1, 0.05, 0.9],
                ],
                dtype=float,
            ),
            np.array(
                [
                    [0.7, -0.15, 0.05],
                    [-0.15, 0.8, -0.1],
                    [0.05, -0.1, 0.2],
                ],
                dtype=float,
            ),
            np.array(
                [
                    [-0.15, 0.0, 0.125],
                    [0.0, 0.1, 0.05],
                    [0.125, 0.05, 0.35],
                ],
                dtype=float,
            ),
        ],
        "b_list": [
            np.array([2.0, 2.0, -3.0], dtype=float),
            np.array([-1.0, -3.0, 3.0], dtype=float),
            np.array([2.0, 7.0, -3.0], dtype=float),
        ],
        "a": 0.04,
        "k": 0.02,
        "omega": np.array([3172.8, 2044.4, 3057.6], dtype=float),
        "deceiver_idx": 0,
        "victim_idx": 2,
        # The scanned caption appears to say J_1^{ref} = -1, but that value does not
        # reproduce the equilibrium levels visible in Fig. 4. A reference near 4.0 does.
        "J_ref": 4.0,
        "epsilon": 2.0e-4,
        "horizon": 1500.0,
        "dt": 0.5,
    }


def simulate_quadratic_game(
    Q_list,
    b_list,
    a,
    k,
    omega,
    deceiver_idx,
    victim_idx,
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

    for idx, time_value in enumerate(time):
        probe_signal = a * np.sin(omega * time_value)
        nominal_actions[idx] = nominal_state + probe_signal
        deceptive_actions[idx] = deceptive_state + probe_signal
        deceptive_actions[idx, deceiver_idx] += (
            a * delta_value * np.sin(omega[victim_idx] * time_value)
        )

        for player_idx in range(player_count):
            nominal_payoffs[idx, player_idx] = J_i_quadratic(
                nominal_actions[idx],
                Q_list[player_idx],
                b_list[player_idx],
            )
            deceptive_payoffs[idx, player_idx] = J_i_quadratic(
                deceptive_actions[idx],
                Q_list[player_idx],
                b_list[player_idx],
            )

        delta_history[idx] = delta_value

        if idx == len(time) - 1:
            continue

        nominal_gradient = np.array(
            [
                quadratic_partial(nominal_state, Q_list[player_idx], b_list[player_idx], player_idx)
                for player_idx in range(player_count)
            ],
            dtype=float,
        )
        deceptive_gradient = np.array(
            [
                quadratic_partial(
                    deceptive_state,
                    Q_list[player_idx],
                    b_list[player_idx],
                    player_idx,
                )
                for player_idx in range(player_count)
            ],
            dtype=float,
        )
        deceptive_gradient[victim_idx] += delta_value * quadratic_partial(
            deceptive_state,
            Q_list[victim_idx],
            b_list[victim_idx],
            deceiver_idx,
        )

        nominal_state = nominal_state - dt * k * nominal_gradient
        deceptive_state = deceptive_state - dt * k * deceptive_gradient
        delta_value = delta_value + dt * epsilon * (
            deceptive_payoffs[idx, deceiver_idx] - J_ref
        )

    x_star = solve_quadratic_ne(Q_list, b_list)
    delta_star = float(delta_history[-1])
    x_delta_star = solve_quadratic_deceptive_ne(
        Q_list,
        b_list,
        deceiver_idx,
        victim_idx,
        delta_star,
    )
    J_star = np.array(
        [J_i_quadratic(x_star, Q_list[idx], b_list[idx]) for idx in range(player_count)],
        dtype=float,
    )
    J_delta_star = np.array(
        [
            J_i_quadratic(x_delta_star, Q_list[idx], b_list[idx])
            for idx in range(player_count)
        ],
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
    )


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
    colors = ["tab:blue", "#d95f0e", "#f0ad00"]
    labels = [r"$J_1$", r"$J_2$", r"$J_3$"]

    for idx, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(
            simulation.time,
            simulation.nominal_payoffs[:, idx],
            color=color,
            linestyle=":",
            linewidth=2.6,
            label=label,
        )
        ax.plot(
            simulation.time,
            simulation.deceptive_payoffs[:, idx],
            color=color,
            linewidth=2.8,
            label=label + " with deception",
        )
        ax.axhline(
            simulation.J_star[idx],
            color="black",
            linewidth=1.6,
            linestyle="--",
            dashes=(6, 4),
            zorder=0,
        )
        ax.axhline(
            simulation.J_delta_star[idx],
            color="black",
            linewidth=1.6,
            zorder=0,
        )

    equilibrium_handles = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.6,
            linestyle="--",
            dashes=(6, 4),
            label=r"$J_i(x^{*})$",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.6,
            label=r"$J_i(x_{\delta})$",
        ),
    ]
    trajectory_handles = [
        plt.Line2D([0], [0], color=colors[0], linestyle=":", linewidth=2.6, label=r"$J_1$"),
        plt.Line2D([0], [0], color=colors[1], linestyle=":", linewidth=2.6, label=r"$J_2$"),
        plt.Line2D([0], [0], color=colors[2], linestyle=":", linewidth=2.6, label=r"$J_3$"),
        plt.Line2D(
            [0],
            [0],
            color=colors[0],
            linewidth=2.8,
            label=r"$J_1$ with deception",
        ),
        plt.Line2D(
            [0],
            [0],
            color=colors[1],
            linewidth=2.8,
            label=r"$J_2$ with deception",
        ),
        plt.Line2D(
            [0],
            [0],
            color=colors[2],
            linewidth=2.8,
            label=r"$J_3$ with deception",
        ),
    ]

    equilibrium_legend = ax.legend(
        handles=equilibrium_handles,
        loc="upper center",
        bbox_to_anchor=(0.63, 0.84),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        fontsize=10,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.3,
    )
    ax.add_artist(equilibrium_legend)
    ax.legend(
        handles=trajectory_handles,
        loc="lower right",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        fontsize=10,
        handlelength=2.2,
        columnspacing=1.0,
        handletextpad=0.45,
        borderpad=0.35,
        labelspacing=0.3,
    )

    ax.set_xlim(simulation.time[0], simulation.time[-1])
    ax.set_ylim(-10.0, 40.0)
    ax.set_xticks([0.0, 500.0, 1000.0, 1500.0])
    ax.set_yticks([-10.0, 0.0, 10.0, 20.0, 30.0, 40.0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$J_i$")

    return fig, ax


def run_quadratic_example_6():
    simulation = simulate_quadratic_game(**example_6_parameters())
    fig, ax = plot_quadratic_example_6(simulation)
    if plt.get_backend().lower() != "agg":
        plt.show()
    return simulation, fig, ax
