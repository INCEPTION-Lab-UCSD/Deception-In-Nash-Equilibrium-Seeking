from dataclasses import dataclass

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
    inflated_sales = s_1_duopoly(x[0], x[1], p, S_d) + (
        delta_2 / (2.0 * p)
    ) * (x[0] - m_1)
    return inflated_sales * (x[0] - m_1)


def NE_duopoly_1(m_1, m_2, S_d, p):
    return (2.0 * m_1 + m_2 + 2.0 * S_d * p) / 3.0


def NE_duopoly_2(m_1, m_2, S_d, p):
    return (m_1 + 2.0 * m_2 + S_d * p) / 3.0


def deceptive_ne_duopoly(delta_2, m_1, m_2, S_d, p):
    denominator = 3.0 - 2.0 * delta_2
    x_1 = ((2.0 - 2.0 * delta_2) * m_1 + m_2 + 2.0 * S_d * p) / denominator
    x_2 = ((1.0 - delta_2) * m_1 + (2.0 - delta_2) * m_2 + S_d * p) / denominator
    return np.array([x_1, x_2], dtype=float)


def x_1_duopoly(u_1, a, omega_1, time_value):
    return u_1 + a * np.sin(omega_1 * time_value)


def x_2_duopoly(u_2, a, omega_1, omega_2, delta_2, time_value):
    return u_2 + a * (np.sin(omega_2 * time_value) + delta_2 * np.sin(omega_1 * time_value))


def delta_2_update_duopoly_deception(x, epsilon, J_2_ref, p, m_2):
    s_i = s_2_duopoly(p, x[0], x[1])
    return epsilon * (J_i_duopoly(s_i, x[1], m_2) - J_2_ref)


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
        s_2_deception = s_2_duopoly(p, actions_1_deception[idx], actions_2_deception[idx])

        J_1[idx] = J_i_duopoly(s_1, actions_1[idx], m[0])
        J_2[idx] = J_i_duopoly(s_2, actions_2[idx], m[1])
        J_1_deception[idx] = J_1_oblivious_duopoly(state_deception, delta_2, p, m[0], S_d)
        J_2_deception[idx] = J_i_duopoly(s_2_deception, actions_2_deception[idx], m[1])
        delta[idx] = delta_2

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
    x_delta_star = deceptive_ne_duopoly(delta_star, m[0], m[1], S_d, p)

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
        simulation.time, simulation.actions_1, color="tab:blue", linestyle=":", linewidth=2.3, label=r"$x_1$"
    )
    ax_actions.plot(
        simulation.time, simulation.actions_2, color="tab:orange", linestyle=":", linewidth=2.3, label=r"$x_2$"
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
        simulation.x_delta_star[0], color="black", linewidth=1.8, alpha=0.9, label=r"$x_{\delta,1}$"
    )
    ax_actions.axhline(simulation.x_star[1], color="black", linewidth=1.8, linestyle="--", alpha=0.9, label=r"$x_2^\ast$")
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
        simulation.time, simulation.J_1, color="tab:blue", linestyle=":", linewidth=2.3, label=r"$J_1$"
    )
    ax_payoffs.plot(
        simulation.time, simulation.J_2, color="tab:orange", linestyle=":", linewidth=2.3, label=r"$J_2$"
    )
    ax_payoffs.axhline(
        J_2_ref, color="black", linestyle="--", linewidth=1.8, dashes=(3, 3), label=r"$J_2^{ref}$"
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
    ax_delta.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.6")

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
