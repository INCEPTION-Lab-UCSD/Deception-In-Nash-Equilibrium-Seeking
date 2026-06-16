import math
from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import solve_ivp

import duopoly


@dataclass
class MutualDeceptionDuopolySimulation:
    time_first_order: np.ndarray
    J_1_first_order: np.ndarray
    J_2_first_order: np.ndarray
    delta_1_first_order: np.ndarray
    delta_2_first_order: np.ndarray
    time_second_order: np.ndarray
    J_1_second_order: np.ndarray
    J_2_second_order: np.ndarray
    delta_1_second_order: np.ndarray
    delta_2_second_order: np.ndarray
    J_1_ref: float
    J_2_ref: float


def x_1_duopoly_mutual(u_1, a, omega_1, omega_2, delta_1, m_2, time_value):
    return u_1 + a * (
        np.sin(omega_1 * time_value) + delta_1 * np.sin(omega_2 * time_value)
    )


def x_2_duopoly_mutual(u_2, a, delta_1, delta_2, omega_1, omega_2, time_value):
    return u_2 + a * (
        np.sin(omega_2 * time_value) + delta_2 * np.sin(omega_1 * time_value)
    )


def Q_1(p):
    return 2 * np.array([[-1 / p, 1 / (2 * p)], [1 / (2 * p), 0]])


def Q_2(p):
    return 2 * np.array([[0, 1 / (2 * p)], [1 / (2 * p), -1 / p]])


def pseudogradient_Q(Q_list):
    pseudograd = [Q_list[i][i, :] for i in range(len(Q_list))]
    return np.array(pseudograd)


def pseudogradient_b(b_list):
    pseudograd = [b_list[i][i] for i in range(len(b_list))]
    return pseudograd


def b_1(p, m_1, S_d):
    return np.array([(m_1 + S_d * p) / p, -1 * m_1 / p])


def b_2(p, m_2):
    return np.array([-1 * m_2 / p, m_2 / p])


def A_0(Q_1, Q_2):
    return np.array(Q_1[1, :], Q_2[2, :])


def b_0(b_1, b_2):
    return np.array([b_1[1], b_2[2]])


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


def J_2_quad(r_2, epsilon, r_1, x_0, Q_2, b_2):
    return r_2 * epsilon**2 + r_1 * epsilon + J_2_quadratic(x_0, Q_2, b_2)


def u_dot(omega_1, omega_2, t, k, a, x, Q_1, b_1, S_d, Q_2, b_2):

    return np.array(
        [
            -(2 * k) / a * J_1_quadratic(x, Q_1, b_1, S_d) * np.sin(omega_1 * t),
            -(2 * k) / a * J_2_quadratic(x, Q_2, b_2) * np.sin(omega_2 * t),
        ]
    )


def phi(Q_2):
    return np.array([1, -1 * Q_2[1, 2] / Q_2[2, 2]])


def prices(t, u, delta_1, delta_2, omega_1, omega_2, a):
    return u + a * policy(t, omega_1, omega_2, delta_1, delta_2)


def delta_update(epsilon, epsilon_i, J_i, J_i_ref):
    return epsilon * epsilon_i * (J_i - J_i_ref)


def sol(
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
    rtol=1e-6,
    atol=1e-8,
):
    sol = solve_ivp(
        fun=udot,
        t_span=(0, time_horizon),
        y0=u_0,
        rtol=rtol,
        atol=atol,
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

    print(f" Done. {sol.t.size} adaptive steps taken.")
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


def first_order_u_dot(
    time_value,
    state,
    a,
    k,
    omega_1,
    omega_2,
    Q_1_matrix,
    b_1_vector,
    Q_2_matrix,
    b_2_vector,
    S_d,
    J_1_ref,
    J_2_ref,
    epsilon_1,
    epsilon_2,
):
    price_vector = prices(
        time_value,
        state[:2],
        state[2],
        state[3],
        omega_1,
        omega_2,
        a,
    )
    J_1_value = J_1_quadratic(price_vector, Q_1_matrix, b_1_vector)
    J_2_value = J_2_quadratic(price_vector, Q_2_matrix, b_2_vector)

    if not math.isnan(J_1_value):
        print(J_1_value)
        print(J_2_value)

    return np.array(
        [
            (-2.0 * k / a) * J_1_value * np.sin(omega_1 * time_value),
            (-2.0 * k / a) * J_2_value * np.sin(omega_2 * time_value),
            epsilon_1 * (J_1_value - J_1_ref),
            epsilon_2 * (J_2_value - J_2_ref),
        ],
        dtype=float,
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
    time = np.arange(0.0, horizon + dt, dt)

    # state = [action_1, action_2, delta_1]
    state = np.array([x0[0], x0[1], 0.0, 0.0], dtype=float)
    Q_1_matrix = Q_1(p)
    Q_2_matrix = Q_2(p)
    b_1_vector = b_1(p, m[0], S_d)
    b_2_vector = b_2(p, m[1])

    J_1 = np.empty_like(time)
    J_2 = np.empty_like(time)
    delta_1_history = np.empty_like(time)
    delta_2_history = np.empty_like(time)

    for idx, time_value in enumerate(time):
        J_1[idx] = J_1_quadratic(state[:2], Q_1_matrix, b_1_vector)
        J_2[idx] = J_2_quadratic(state[:2], Q_2_matrix, b_2_vector)
        delta_1_history[idx] = state[2]
        delta_2_history[idx] = state[3]

        if idx == len(time) - 1:
            continue

        u_dot_args = (
            a,
            k,
            omega_1,
            omega_2,
            Q_1_matrix,
            b_1_vector,
            Q_2_matrix,
            b_2_vector,
            S_d,
            J_1_ref,
            J_2_ref,
            epsilon_1,
            epsilon_2,
        )
        k_1 = first_order_u_dot(time_value, state, *u_dot_args)
        k_2 = first_order_u_dot(
            time_value + 0.5 * dt, state + 0.5 * dt * k_1, *u_dot_args
        )
        k_3 = first_order_u_dot(
            time_value + 0.5 * dt, state + 0.5 * dt * k_2, *u_dot_args
        )
        k_4 = first_order_u_dot(time_value + dt, state + dt * k_3, *u_dot_args)
        state = state + (dt / 6.0) * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)

    _ = epsilon, delta_limit, action_limits

    return {
        "time": time,
        "J_1": J_1,
        "J_2": J_2,
        "delta_1": delta_1_history,
        "delta_2": delta_2_history,
    }


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
    compensator_gain_1,
    compensator_gain_2,
    S_d,
    p,
    m,
    horizon,
    dt=0.05,
    delta_limit=8.0,
    action_limits=(0.0, 120.0),
):
    time = np.arange(0.0, horizon + dt, dt)
    state = np.array(x0, dtype=float).copy()
    delta_1 = 0.0
    delta_2 = 0.0
    compensator_1 = 0.0
    compensator_2 = 0.0

    J_1 = np.empty_like(time)
    J_2 = np.empty_like(time)
    delta_1_history = np.empty_like(time)
    delta_2_history = np.empty_like(time)

    for idx, time_value in enumerate(time):
        action_1 = x_1_duopoly_mutual(
            state[0], a, omega_1, omega_2, delta_1, m[1], time_value
        )
        action_2 = duopoly.x_2_duopoly(
            state[1], a, omega_1, omega_2, delta_2, time_value
        )
        action_1 = float(np.clip(action_1, *action_limits))
        action_2 = float(np.clip(action_2, *action_limits))

        s_1 = duopoly.s_1_duopoly(action_1, action_2, p, S_d)
        s_2 = duopoly.s_2_duopoly(p, action_1, action_2)
        J_1[idx] = duopoly.J_i_duopoly(s_1, action_1, m[0])
        J_2[idx] = duopoly.J_i_duopoly(s_2, action_2, m[1])
        delta_1_history[idx] = delta_1
        delta_2_history[idx] = delta_2

        if idx == len(time) - 1:
            continue

        gradient = np.array(
            [
                duopoly.J_1_duopoly_grad_1(state, p, m[0], S_d)
                + delta_2 * duopoly.J_1_duopoly_grad_2(state, p, m[0]),
                duopoly.J_2_duopoly_grad_2(state, p, m[1])
                + delta_1 * duopoly.J_2_duopoly_grad_1(state, p, m[1]),
            ],
            dtype=float,
        )
        state = np.clip(state - dt * k * gradient, *action_limits)

        delta_1_error = J_1[idx] - J_1_ref
        delta_2_error = J_2[idx] - J_2_ref
        compensator_1 = compensator_1 + dt * (
            epsilon_1 * delta_1_error - compensator_gain_1 * compensator_1
        )
        compensator_2 = compensator_2 + dt * (
            epsilon_2 * delta_2_error - compensator_gain_2 * compensator_2
        )
        delta_1 = float(delta_1 + dt * compensator_1)
        delta_2 = float(delta_2 + dt * compensator_2)

    return {
        "time": time,
        "J_1": J_1,
        "J_2": J_2,
        "delta_1": delta_1_history,
        "delta_2": delta_2_history,
    }


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
    compensator_gain_1=0.45,
    compensator_gain_2=0.65,
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
        compensator_gain_1=compensator_gain_1,
        compensator_gain_2=compensator_gain_2,
        S_d=S_d,
        p=p,
        m=m,
        horizon=second_order_horizon,
        dt=second_order_dt,
        delta_limit=delta_limit,
        action_limits=action_limits,
    )

    return MutualDeceptionDuopolySimulation(
        time_first_order=first_order["time"],
        J_1_first_order=first_order["J_1"],
        J_2_first_order=first_order["J_2"],
        delta_1_first_order=first_order["delta_1"],
        delta_2_first_order=first_order["delta_2"],
        time_second_order=second_order["time"],
        J_1_second_order=second_order["J_1"],
        J_2_second_order=second_order["J_2"],
        delta_1_second_order=second_order["delta_1"],
        delta_2_second_order=second_order["delta_2"],
        J_1_ref=float(J_1_ref),
        J_2_ref=float(J_2_ref),
    )


def plot_mutual_deception_duopoly(simulation):
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    fig = plt.figure(figsize=(15.8, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.15, 1.15, 1.25],
        height_ratios=[1.0, 1.0],
    )
    ax_profit_first = fig.add_subplot(grid[:, 0])
    ax_profit_second = fig.add_subplot(grid[:, 1])
    ax_delta_first = fig.add_subplot(grid[0, 2])
    ax_delta_second = fig.add_subplot(grid[1, 2])

    ax_profit_first.plot(
        simulation.time_first_order,
        simulation.J_1_first_order,
        color="tab:blue",
        linewidth=2.0,
        label=r"$J_1$",
    )
    ax_profit_first.plot(
        simulation.time_first_order,
        simulation.J_2_first_order,
        color="tab:orange",
        linewidth=2.0,
        label=r"$J_2$",
    )
    ax_profit_first.axhline(
        simulation.J_1_ref,
        color="black",
        linewidth=1.7,
        label=r"$J_1^{ref}$",
    )
    ax_profit_first.axhline(
        simulation.J_2_ref,
        color="black",
        linewidth=1.7,
        linestyle="--",
        dashes=(3, 3),
        label=r"$J_2^{ref}$",
    )
    ax_profit_first.set_xlabel("Time (s)")
    ax_profit_first.set_ylabel("Profit")
    ax_profit_first.set_xlim(
        simulation.time_first_order[0], simulation.time_first_order[-1]
    )
    ax_profit_first.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        fontsize=10,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.3,
    )

    ax_profit_second.plot(
        simulation.time_second_order,
        simulation.J_1_second_order,
        color="tab:blue",
        linewidth=2.0,
        label=r"$J_1$",
    )
    ax_profit_second.plot(
        simulation.time_second_order,
        simulation.J_2_second_order,
        color="tab:orange",
        linewidth=2.0,
        label=r"$J_2$",
    )
    ax_profit_second.axhline(
        simulation.J_1_ref,
        color="black",
        linewidth=1.7,
        label=r"$J_1^{ref}$",
    )
    ax_profit_second.axhline(
        simulation.J_2_ref,
        color="black",
        linewidth=1.7,
        linestyle="--",
        dashes=(3, 3),
        label=r"$J_2^{ref}$",
    )
    ax_profit_second.set_xlabel("Time (s)")
    ax_profit_second.set_ylabel("Profit")
    ax_profit_second.set_xlim(
        simulation.time_second_order[0], simulation.time_second_order[-1]
    )
    ax_profit_second.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        fontsize=10,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.3,
    )

    ax_delta_first.plot(
        simulation.time_first_order,
        simulation.delta_1_first_order,
        color="black",
        linewidth=1.9,
        label=r"$\delta_1$",
    )
    ax_delta_first.plot(
        simulation.time_first_order,
        simulation.delta_2_first_order,
        color="red",
        linewidth=1.8,
        label=r"$\delta_2$",
    )
    ax_delta_first.set_xlim(
        simulation.time_first_order[0], simulation.time_first_order[-1]
    )
    ax_delta_first.legend(
        loc="upper right",
        frameon=False,
        fontsize=10,
        handlelength=1.6,
        handletextpad=0.35,
    )

    ax_delta_second.plot(
        simulation.time_second_order,
        simulation.delta_1_second_order,
        color="black",
        linewidth=1.9,
        label=r"$\delta_1$",
    )
    ax_delta_second.plot(
        simulation.time_second_order,
        simulation.delta_2_second_order,
        color="red",
        linewidth=1.8,
        label=r"$\delta_2$ with compensator",
    )
    ax_delta_second.set_xlabel("Time (s)")
    ax_delta_second.set_xlim(
        simulation.time_second_order[0], simulation.time_second_order[-1]
    )
    ax_delta_second.legend(
        loc="upper right",
        frameon=False,
        fontsize=10,
        handlelength=1.6,
        handletextpad=0.35,
    )

    profit_values = np.concatenate(
        [
            simulation.J_1_first_order,
            simulation.J_2_first_order,
            simulation.J_1_second_order,
            simulation.J_2_second_order,
            np.array([simulation.J_1_ref, simulation.J_2_ref], dtype=float),
        ]
    )
    profit_padding = 0.08 * max(
        1.0, float(np.nanmax(profit_values) - np.nanmin(profit_values))
    )
    profit_limits = (
        float(np.nanmin(profit_values) - profit_padding),
        float(np.nanmax(profit_values) + profit_padding),
    )
    ax_profit_first.set_ylim(*profit_limits)
    ax_profit_second.set_ylim(*profit_limits)

    delta_first_values = np.concatenate(
        [simulation.delta_1_first_order, simulation.delta_2_first_order]
    )
    delta_second_values = np.concatenate(
        [simulation.delta_1_second_order, simulation.delta_2_second_order]
    )
    delta_first_padding = 0.1 * max(
        1.0, float(np.nanmax(delta_first_values) - np.nanmin(delta_first_values))
    )
    delta_second_padding = 0.1 * max(
        1.0, float(np.nanmax(delta_second_values) - np.nanmin(delta_second_values))
    )
    ax_delta_first.set_ylim(
        float(np.nanmin(delta_first_values) - delta_first_padding),
        float(np.nanmax(delta_first_values) + delta_first_padding),
    )
    ax_delta_second.set_ylim(
        float(np.nanmin(delta_second_values) - delta_second_padding),
        float(np.nanmax(delta_second_values) + delta_second_padding),
    )

    return fig, np.array(
        [ax_profit_first, ax_profit_second, ax_delta_first, ax_delta_second],
        dtype=object,
    )


def animate_mutual_deception_duopoly(
    simulation,
    frame_step_first_order=20,
    frame_step_second_order=4,
    interval=40,
    repeat_delay=1200,
):
    if frame_step_first_order <= 0 or frame_step_second_order <= 0:
        raise ValueError("frame steps must be positive integers.")

    fig, axes = plot_mutual_deception_duopoly(simulation)
    ax_profit_first, ax_profit_second, ax_delta_first, ax_delta_second = axes

    lines_profit_first = ax_profit_first.lines[:2]
    lines_profit_second = ax_profit_second.lines[:2]
    lines_delta_first = ax_delta_first.lines[:2]
    lines_delta_second = ax_delta_second.lines[:2]

    time_first = np.asarray(simulation.time_first_order, dtype=float)
    time_second = np.asarray(simulation.time_second_order, dtype=float)
    first_indices = np.arange(0, len(time_first), frame_step_first_order, dtype=int)
    second_indices = np.arange(0, len(time_second), frame_step_second_order, dtype=int)
    if first_indices[-1] != len(time_first) - 1:
        first_indices = np.append(first_indices, len(time_first) - 1)
    if second_indices[-1] != len(time_second) - 1:
        second_indices = np.append(second_indices, len(time_second) - 1)

    for line in [
        *lines_profit_first,
        *lines_profit_second,
        *lines_delta_first,
        *lines_delta_second,
    ]:
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
        first_frame_idx = min(frame_idx, len(first_indices) - 1)
        second_frame_idx = min(frame_idx, len(second_indices) - 1)
        first_idx = first_indices[first_frame_idx] + 1
        second_idx = second_indices[second_frame_idx] + 1

        first_slice = time_first[:first_idx]
        second_slice = time_second[:second_idx]

        lines_profit_first[0].set_data(
            first_slice, simulation.J_1_first_order[:first_idx]
        )
        lines_profit_first[1].set_data(
            first_slice, simulation.J_2_first_order[:first_idx]
        )
        lines_delta_first[0].set_data(
            first_slice, simulation.delta_1_first_order[:first_idx]
        )
        lines_delta_first[1].set_data(
            first_slice, simulation.delta_2_first_order[:first_idx]
        )

        lines_profit_second[0].set_data(
            second_slice, simulation.J_1_second_order[:second_idx]
        )
        lines_profit_second[1].set_data(
            second_slice, simulation.J_2_second_order[:second_idx]
        )
        lines_delta_second[0].set_data(
            second_slice, simulation.delta_1_second_order[:second_idx]
        )
        lines_delta_second[1].set_data(
            second_slice, simulation.delta_2_second_order[:second_idx]
        )

        time_box.set_text(
            rf"$t_{{1st}} = {time_first[first_idx - 1]:.1f}\,\mathrm{{s}}$"
            + "    "
            + rf"$t_{{2nd}} = {time_second[second_idx - 1]:.1f}\,\mathrm{{s}}$"
        )

        return (
            *lines_profit_first,
            *lines_profit_second,
            *lines_delta_first,
            *lines_delta_second,
            time_box,
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max(len(first_indices), len(second_indices)),
        interval=interval,
        repeat=True,
        repeat_delay=repeat_delay,
        blit=False,
    )
    update(0)

    return duopoly.DuopolyPlotAnimation(animation=ani, figure=fig, axes=axes)
