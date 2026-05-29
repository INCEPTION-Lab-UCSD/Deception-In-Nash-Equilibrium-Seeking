from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def delta_update(epsilon, epsilon_i, J_i, J_i_ref):
    return epsilon * epsilon_i * (J_i - J_i_ref)


def J_2_deceived(x_1, x_2, m_2, p, delta_1):
    x_1_bar = x_1 - 0.5 * delta_1 * m_2
    x_2_bar = (1.0 - 0.5 * delta_1) * x_2
    sigma_2 = (delta_1 * m_2**2.0) / (2.0 * p)

    return -(1 / p) * (x_1_bar - x_2_bar) * (x_2 - m_2) + sigma_2


def J_1_deceived(x_1, x_2, m_1, p, delta_2, S_d):
    x_1_bar = x_1 - 0.5 * delta_2 * m_1
    x_2_bar = (1.0 - 0.5 * delta_2) * x_2
    s_2_bar = (x_1_bar - x_2_bar) / p
    s_1_bar = S_d - s_2_bar
    sigma_1 = (delta_2 * m_1**2.0) / (2.0 * p)
    return s_1_bar * (x_1 - m_1) + sigma_1


# def J_1_deceived(x_1, x_2, m_1, p, delta_2, S_d):
#     x_1_bar = x_1 - 0.5 * delta_2 * m_1
#     x_2_bar = (1.0 - 0.5 * delta_2) * x_2
#     sigma_1 = (delta_2 * m_1**2.0) / (2.0 * p)
#     return (1.0 / p) * (x_1_bar - x_2_bar) * (
#         x_1 - m_1
#     ) + sigma_1  # note sign: J̃₁ uses s̃₁


def J_1_grad_1(x_1, x_2, m_1, delta_2, p, S_d):
    x_1_bar = x_1 - 0.5 * delta_2 * m_1
    x_2_bar = (1.0 - 0.5 * delta_2) * x_2
    return S_d - (x_1_bar - x_2_bar) / p - (x_1 - m_1) / p


# def J_1_grad_1(x_1, x_2, m_1, delta_2, p):
#     return (1.0 / p) * (2.0 * (1.0 - 0.5 * delta_2) * x_1 - m_1 - x_2)


def J_1_grad_2(x_1, m_1, delta_2, p):
    return -1.0 / p * (1.0 - delta_2 / 2.0) * (x_1 - m_1)


def J_2_grad_1(x_2, m_2, p):
    return (-x_2 + m_2) / p


def J_2_grad_2(x_1, x_2, m_2, delta_1, p):
    x_1_bar = x_1 - 0.5 * delta_1 * m_2
    x_2_bar = (1.0 - 0.5 * delta_1) * x_2
    return (1.0 / p) * (1.0 - 0.5 * delta_1) * (x_2 - m_2) - (x_1_bar - x_2_bar) / p


# def J_2_grad_2(x_1, x_2, m_2, delta_1, p):
#     return (1.0 / p) * (2.0 * (1.0 - 0.5 * delta_1) * x_2 - m_2 - x_1)


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
    dt=0.05,
    delta_limit=8.0,
    action_limits=(0.0, 120.0),
):
    time = np.arange(0.0, horizon + dt, dt)
    state = np.array(x0, dtype=float).copy()
    delta_1 = 0.0
    delta_2 = 0.0

    J_1 = np.empty_like(time)
    J_2 = np.empty_like(time)
    delta_1_history = np.empty_like(time)
    delta_2_history = np.empty_like(time)

    for idx, time_value in enumerate(time):
        action_1 = x_1_duopoly_mutual(
            state[0], a, omega_1, omega_2, delta_1, m[1], time_value
        )
        action_2 = x_2_duopoly_mutual(
            state[1], a, delta_1, delta_2, omega_1, omega_2, time_value
        )
        action_1 = float(np.clip(action_1, *action_limits))
        action_2 = float(np.clip(action_2, *action_limits))
        s_1 = duopoly.s_1_duopoly(action_1, action_2, p, S_d)
        s_2 = duopoly.s_2_duopoly(p, action_1, action_2)
        J_1[idx] = duopoly.J_i_duopoly(s_1, action_1, m[0])
        J_2[idx] = duopoly.J_i_duopoly(s_2, action_2, m[1])

        # J_1[idx] = J_1_deceived(action_1, action_2, m[0], p, delta_2, S_d)
        # J_2[idx] = J_2_deceived(action_1, action_2, m[1], p, delta_1)
        delta_1_history[idx] = delta_1
        delta_2_history[idx] = delta_2

        if idx == len(time) - 1:
            continue

        gradient = np.asarray(
            [
                J_1_grad_1(state[0], state[1], m[0], delta_2, p, S_d),
                J_2_grad_2(state[0], state[1], m[1], delta_1, p),
            ]
        )

        # gradient = np.asarray(
        #     [
        #         J_1_grad_1(state[0], state[1], m[0], delta_2, p)
        #         + delta_2 * J_1_grad_2(state[0], m[0], delta_2, p),
        #         J_2_grad_2(state[0], state[1], m[1], delta_1, p)
        #         + delta_1 * J_2_grad_1(state[1], m[1], p),
        #     ],
        #     dtype=float,
        # )
        state = np.clip(state - dt * k * gradient, *action_limits)

        delta_1 = np.clip(
            delta_1 + dt * delta_update(epsilon, epsilon_1, J_1[idx], J_1_ref),
            -delta_limit,
            delta_limit,
        )
        delta_2 = np.clip(
            delta_2 + dt * delta_update(epsilon, epsilon_2, J_2[idx], J_2_ref),
            -delta_limit,
            delta_limit,
        )

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
        delta_1 = float(
            np.clip(delta_1 + dt * compensator_1, -delta_limit, delta_limit)
        )
        delta_2 = float(
            np.clip(delta_2 + dt * compensator_2, -delta_limit, delta_limit)
        )

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
    compensator_gain_1=0.45,
    compensator_gain_2=0.65,
    delta_limit=8.0,
    action_limits=(0.0, 120.0),
):
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
        dt=dt,
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
        dt=dt,
        delta_limit=delta_limit,
        action_limits=action_limits,
    )
    print(second_order["delta_1"])
    print(second_order["delta_2"])

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
