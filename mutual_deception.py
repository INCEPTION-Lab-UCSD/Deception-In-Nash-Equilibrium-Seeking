from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import solve_ivp

import duopoly


@dataclass
class MutualDeceptionSimulation:
    time: np.ndarray
    J_1: np.ndarray
    J_2: np.ndarray
    delta_1: np.ndarray
    delta_2: np.ndarray
    J_1_ref: float
    J_2_ref: float


def Q_1_mat(p):
    return 2 * np.array([[-1 / p, 1 / (2 * p)], [1 / (2 * p), 0]])


def Q_2_mat(p):
    return 2 * np.array([[0, 1 / (2 * p)], [1 / (2 * p), -1 / p]])


def b_1_vec(p, m_1, S_d):
    return np.array([(m_1 + S_d * p) / p, -1 * m_1 / p])


def b_2_vec(p, m_2):
    return np.array([-1 * m_2 / p, m_2 / p])


def policy(t, omega_1, omega_2, delta_1, delta_2):
    return np.array(
        [
            np.sin(omega_1 * t) + delta_1 * np.sin(omega_2 * t),
            np.sin(omega_2 * t) + delta_2 * np.sin(omega_1 * t),
        ]
    )


def J_1_quadratic(x, Q_1, b_1):
    return 0.5 * x.T @ Q_1 @ x + b_1.T @ x - 3000


def J_2_quadratic(x, Q_2, b_2):
    return 0.5 * x.T @ Q_2 @ x + b_2.T @ x


def prices(t, u, delta_1, delta_2, omega_1, omega_2, a):
    return u + a * policy(t, omega_1, omega_2, delta_1, delta_2)


def sampled_time_grid(time_horizon, dt):
    if dt <= 0:
        raise ValueError("dt must be positive.")

    time = np.arange(0.0, time_horizon + 0.5 * dt, dt)
    if time[-1] > time_horizon:
        time[-1] = time_horizon
    if time[-1] < time_horizon:
        time = np.append(time, time_horizon)
    return np.unique(time)


def quadratic_payoff_history(states, Q_1, b_1, Q_2, b_2):
    actions = np.asarray(states[:2, :], dtype=float).T
    J_1 = np.array([J_1_quadratic(action, Q_1, b_1) for action in actions])
    J_2 = np.array([J_2_quadratic(action, Q_2, b_2) for action in actions])
    return J_1, J_2


def second_order_initial_state(x0):
    x0 = np.asarray(x0, dtype=float)
    return np.array([x0[0], x0[1], 0.0, 0.0, 0.0, 0.0], dtype=float)


def solve_first_order(
    time_horizon,
    u_0,
    omega_1,
    omega_2,
    a,
    Q_1,
    b_1,
    Q_2,
    b_2,
    k,
    epsilon_1,
    epsilon_2,
    J_1_ref,
    J_2_ref,
    time_eval,
    rtol=1e-6,
    atol=1e-8,
):
    sol = solve_ivp(
        fun=udot,
        t_span=(0, time_horizon),
        y0=u_0,
        rtol=rtol,
        atol=atol,
        t_eval=time_eval,
        max_step=float(np.max(np.diff(time_eval))) if len(time_eval) > 1 else np.inf,
        dense_output=False,
        args=(
            omega_1,
            omega_2,
            a,
            Q_1,
            b_1,
            Q_2,
            b_2,
            k,
            epsilon_1,
            epsilon_2,
            J_1_ref,
            J_2_ref,
        ),
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    return sol


def udot(
    t,
    u,
    omega_1,
    omega_2,
    a,
    Q_1,
    b_1,
    Q_2,
    b_2,
    k,
    epsilon_1,
    epsilon_2,
    J_1_ref,
    J_2_ref,
):
    x = prices(t, u[:2], u[2], u[3], omega_1, omega_2, a)
    J_1 = J_1_quadratic(x, Q_1, b_1)
    J_2 = J_2_quadratic(x, Q_2, b_2)

    return np.array(
        [
            -2.0 * k / a * J_1 * np.sin(omega_1 * t),
            (-2.0 * k / a) * J_2 * np.sin(omega_2 * t),
            epsilon_1 * (J_1 - J_1_ref),
            epsilon_2 * (J_2 - J_2_ref),
        ]
    )


def delta_1_second_order(z, e, G):
    return G[0][1] / G[0][0] * e - (G[0][1] / G[0][0] - 1) * z


def delta_2_second_order(z, e, G):
    return G[1][1] / G[1][0] * e - (G[1][1] / G[1][0] - 1) * z


def solve_second_order(
    time_horizon,
    u_0,
    omega_1,
    omega_2,
    a,
    Q_1,
    b_1,
    Q_2,
    b_2,
    k,
    epsilon_1,
    epsilon_2,
    J_1_ref,
    J_2_ref,
    G,
    rtol=1e-6,
    atol=1e-8,
):
    sol = solve_ivp(
        fun=u_dot_second_order,
        t_span=(0, time_horizon),
        y0=u_0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        args=(
            omega_1,
            omega_2,
            a,
            Q_1,
            b_1,
            Q_2,
            b_2,
            k,
            epsilon_1,
            epsilon_2,
            J_1_ref,
            J_2_ref,
            G,
        ),
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    return sol


# u = [action_1, action_2, z_1, e_1, z_2, e_2]
def u_dot_second_order(
    t,
    u,
    omega_1,
    omega_2,
    a,
    Q_1,
    b_1,
    Q_2,
    b_2,
    k,
    epsilon_1,
    epsilon_2,
    J_1_ref,
    J_2_ref,
    G,
):
    u_1, u_2, z_1, e_1, z_2, e_2 = u
    delta_1 = delta_1_second_order(z_1, e_1, G)
    delta_2 = delta_2_second_order(z_2, e_2, G)

    x = prices(
        t,
        np.array([u_1, u_2]),
        delta_1,
        delta_2,
        omega_1,
        omega_2,
        a,
    )
    J_1 = J_1_quadratic(x, Q_1, b_1)
    J_2 = J_2_quadratic(x, Q_2, b_2)

    return np.array(
        [
            (-2.0 * k / a) * J_1 * np.sin(omega_1 * t),
            (-2.0 * k / a) * J_2 * np.sin(omega_2 * t),
            (1.0 / G[0][0]) * (-z_1 + e_1),
            epsilon_1 * (J_1 - J_1_ref),
            (1.0 / G[1][0]) * (-z_2 + e_2),
            epsilon_2 * (J_2 - J_2_ref),
        ]
    )


def simulate_mutual_deception_duopoly_first_order(
    x0,
    a,
    k,
    omega_1,
    omega_2,
    J_1_ref,
    J_2_ref,
    epsilon,
    epsilon_1,
    epsilon_2,
    S_d,
    p,
    m,
    horizon,
    dt=0.001,
    delta_limit=8.0,
    action_limits=(0.0, 120.0),
):
    time = sampled_time_grid(horizon, dt)
    state = np.array([x0[0], x0[1], 0.0, 0.0], dtype=float)
    Q_1 = Q_1_mat(p)
    Q_2 = Q_2_mat(p)
    b_1 = b_1_vec(p, m[0], S_d)
    b_2 = b_2_vec(p, m[1])

    sol = solve_first_order(
        time_horizon=horizon,
        u_0=state,
        omega_1=omega_1,
        omega_2=omega_2,
        a=a,
        Q_1=Q_1,
        b_1=b_1,
        Q_2=Q_2,
        b_2=b_2,
        k=k,
        epsilon_1=epsilon_1,
        epsilon_2=epsilon_2,
        J_1_ref=J_1_ref,
        J_2_ref=J_2_ref,
        time_eval=time,
    )

    states = sol.y
    J_1, J_2 = quadratic_payoff_history(states, Q_1, b_1, Q_2, b_2)

    _ = epsilon, delta_limit, action_limits
    return MutualDeceptionSimulation(
        sol.t,
        J_1,
        J_2,
        np.array(states[2]),
        np.array(states[3]),
        J_1_ref,
        J_2_ref,
    )


def simulate_mutual_deception_duopoly_second_order(
    x0,
    a,
    k,
    omega_1,
    omega_2,
    J_1_ref,
    J_2_ref,
    epsilon,
    epsilon_1,
    epsilon_2,
    G,
    S_d,
    p,
    m,
    horizon,
    dt=0.05,
    delta_limit=8.0,
    action_limits=(0.0, 120.0),
):
    time = sampled_time_grid(horizon, dt)
    state = second_order_initial_state(x0)
    Q_1 = Q_1_mat(p)
    Q_2 = Q_2_mat(p)
    b_1 = b_1_vec(p, m[0], S_d)
    b_2 = b_2_vec(p, m[1])

    sol = solve_second_order(
        time_horizon=horizon,
        u_0=state,
        omega_1=omega_1,
        omega_2=omega_2,
        a=a,
        Q_1=Q_1,
        b_1=b_1,
        Q_2=Q_2,
        b_2=b_2,
        k=k,
        epsilon_1=epsilon_1,
        epsilon_2=epsilon_2,
        J_1_ref=J_1_ref,
        J_2_ref=J_2_ref,
        G=G,
    )

    states = sol.sol(time)
    J_1, J_2 = quadratic_payoff_history(states, Q_1, b_1, Q_2, b_2)
    delta_1_history = delta_1_second_order(states[2], states[3], G)
    delta_2_history = delta_2_second_order(states[4], states[5], G)

    _ = epsilon, delta_limit, action_limits
    return MutualDeceptionSimulation(
        time,
        J_1,
        J_2,
        delta_1_history,
        delta_2_history,
        J_1_ref,
        J_2_ref,
    )


def simulate_mutual_deception_duopoly(
    x0,
    a,
    k,
    omega_1,
    omega_2,
    J_1_ref,
    J_2_ref,
    epsilon,
    epsilon_1,
    epsilon_2,
    S_d,
    p,
    m,
    first_order_horizon=1000.0,
    second_order_horizon=50.0,
    dt=0.05,
    first_order_dt=0.0001,
    second_order_dt=None,
    G=np.array([[3, 13], [2, 10]]),
    delta_limit=8.0,
    action_limits=(0.0, 120.0),
):
    if second_order_dt is None:
        second_order_dt = dt

    first_order = simulate_mutual_deception_duopoly_first_order(
        x0=x0,
        a=a,
        k=k,
        omega_1=omega_1,
        omega_2=omega_2,
        J_1_ref=J_1_ref,
        J_2_ref=J_2_ref,
        epsilon=epsilon,
        epsilon_1=epsilon_1,
        epsilon_2=epsilon_2,
        S_d=S_d,
        p=p,
        m=m,
        horizon=first_order_horizon,
        dt=first_order_dt,
        delta_limit=delta_limit,
        action_limits=action_limits,
    )
    second_order = simulate_mutual_deception_duopoly_second_order(
        x0=x0,
        a=a,
        k=k,
        omega_1=omega_1,
        omega_2=omega_2,
        J_1_ref=J_1_ref,
        J_2_ref=J_2_ref,
        epsilon=epsilon,
        epsilon_1=epsilon_1,
        epsilon_2=epsilon_2,
        G=G,
        S_d=S_d,
        p=p,
        m=m,
        horizon=second_order_horizon,
        dt=second_order_dt,
        delta_limit=delta_limit,
        action_limits=action_limits,
    )

    return first_order, second_order


def configure_mutual_deception_plot_style():
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def plot_mutual_deception_duopoly(simulation, title=None):
    configure_mutual_deception_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax_profit, ax_delta = axes

    ax_profit.plot(simulation.time, simulation.J_1, linewidth=2.0, label=r"$J_1$")
    ax_profit.plot(simulation.time, simulation.J_2, linewidth=2.0, label=r"$J_2$")
    ax_profit.axhline(
        simulation.J_1_ref, color="black", linewidth=1.5, label=r"$J_1^{ref}$"
    )
    ax_profit.axhline(
        simulation.J_2_ref,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label=r"$J_2^{ref}$",
    )
    ax_profit.set_xlabel("Time (s)")
    ax_profit.set_ylabel("Profit")
    ax_profit.set_xlim(simulation.time[0], simulation.time[-1])
    ax_profit.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        fontsize=10,
    )

    ax_delta.plot(
        simulation.time,
        simulation.delta_1,
        color="black",
        linewidth=2.0,
        label=r"$\delta_1$",
    )
    ax_delta.plot(
        simulation.time,
        simulation.delta_2,
        color="red",
        linewidth=2.0,
        label=r"$\delta_2$",
    )
    ax_delta.set_xlabel("Time (s)")
    ax_delta.set_ylabel(r"$\delta$")
    ax_delta.set_xlim(simulation.time[0], simulation.time[-1])
    ax_delta.legend(loc="best", frameon=False, fontsize=10)

    profit_values = np.concatenate(
        [
            simulation.J_1,
            simulation.J_2,
            np.array([simulation.J_1_ref, simulation.J_2_ref], dtype=float),
        ]
    )
    profit_padding = 0.08 * max(
        1.0, float(np.nanmax(profit_values) - np.nanmin(profit_values))
    )
    ax_profit.set_ylim(
        float(np.nanmin(profit_values) - profit_padding),
        float(np.nanmax(profit_values) + profit_padding),
    )

    delta_values = np.concatenate([simulation.delta_1, simulation.delta_2])
    delta_padding = 0.1 * max(
        1.0, float(np.nanmax(delta_values) - np.nanmin(delta_values))
    )
    ax_delta.set_ylim(
        float(np.nanmin(delta_values) - delta_padding),
        float(np.nanmax(delta_values) + delta_padding),
    )

    if title is not None:
        fig.suptitle(title)

    return fig, np.array([ax_profit, ax_delta], dtype=object)


def plot_mutual_deception_duopoly_first_order(simulation):
    return plot_mutual_deception_duopoly(
        simulation, title="First-order mutual deception"
    )


def plot_mutual_deception_duopoly_second_order(simulation):
    return plot_mutual_deception_duopoly(
        simulation, title="Second-order mutual deception"
    )


def animate_mutual_deception(simulation, frame_step=20, interval=40, repeat_delay=1200):
    if frame_step <= 0:
        raise ValueError("frame_step must be a positive integer.")

    fig, axes = plot_mutual_deception_duopoly(simulation)
    ax_profit, ax_delta = axes
    profit_lines = ax_profit.lines[:2]
    delta_lines = ax_delta.lines[:2]

    time = np.asarray(simulation.time, dtype=float)
    indices = np.arange(0, len(time), frame_step, dtype=int)
    if indices[-1] != len(time) - 1:
        indices = np.append(indices, len(time) - 1)

    for line in [*profit_lines, *delta_lines]:
        line.set_data([], [])

    time_box = fig.text(
        0.5,
        0.99,
        "",
        ha="center",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    def update(frame_idx):
        frame_idx = min(frame_idx, len(indices) - 1)
        idx = indices[frame_idx] + 1
        time_slice = time[:idx]

        profit_lines[0].set_data(time_slice, simulation.J_1[:idx])
        profit_lines[1].set_data(time_slice, simulation.J_2[:idx])
        delta_lines[0].set_data(time_slice, simulation.delta_1[:idx])
        delta_lines[1].set_data(time_slice, simulation.delta_2[:idx])
        time_box.set_text(rf"$t = {time[idx - 1]:.1f}\,\mathrm{{s}}$")

        return (*profit_lines, *delta_lines, time_box)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(indices),
        interval=interval,
        repeat=True,
        repeat_delay=repeat_delay,
        blit=False,
    )
    update(0)

    return duopoly.DuopolyPlotAnimation(animation=ani, figure=fig, axes=axes)


def animate_mutual_deception_duopoly_first_order(
    simulation, frame_step=20, interval=40, repeat_delay=1200
):
    return animate_mutual_deception(
        simulation,
        frame_step=frame_step,
        interval=interval,
        repeat_delay=repeat_delay,
    )


def animate_mutual_deception_duopoly_second_order(
    simulation, frame_step=20, interval=40, repeat_delay=1200
):
    return animate_mutual_deception(
        simulation,
        frame_step=frame_step,
        interval=interval,
        repeat_delay=repeat_delay,
    )


animate_mutual_deception_duopoly = animate_mutual_deception
