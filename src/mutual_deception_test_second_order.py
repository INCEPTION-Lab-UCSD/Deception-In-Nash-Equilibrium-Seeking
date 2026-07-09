import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ── Parameters ────────────────────────────────────────────────────────────────
h = 0.0001
S = 100
p = 0.2
m1 = 30
m2 = 30

# ── Matrices ──────────────────────────────────────────────────────────────────
Q1 = 2 * np.array([[-1 / p, 1 / (2 * p)], [1 / (2 * p), 0]])

b1 = np.array([(m1 + S * p) / p, -m1 / p])

Q2 = 2 * np.array([[0, 1 / (2 * p)], [1 / (2 * p), -1 / p]])

b2 = np.array([-m2 / p, m2 / p])

A0 = np.array([Q1[0, :], Q2[1, :]])

b0 = np.array([b1[0], b2[1]])

x0 = -np.linalg.solve(A0, b0)

# ── Scalar constants ───────────────────────────────────────────────────────────
c1 = -(Q1[0, 0] + Q2[1, 1]) / Q1[0, 1]
c2 = np.linalg.det(A0) / (Q2[0, 1] * Q1[1, 1] - Q2[1, 1] * Q1[0, 1])

w = np.array([1, -Q2[0, 1] / Q2[1, 1]])

q1 = -(b1[1] + Q1[1, :] @ x0)
q2 = Q1[1, :] @ w
q3 = Q1[0, :] @ w
r2 = 0.5 * w @ Q2 @ w
r1 = (b2 + Q2 @ x0) @ w

# ── Excitation / algorithm hyperparameters ────────────────────────────────────
a = 0.05
k = -0.03
w1 = 11877.75
w2 = 12436.5
epsilon1 = -0.001
epsilon2 = -0.0005
J1ref = 1200.0
J2ref = 1800.0

# Lead-lag filter gains (player 1: g1_1, g1_2; player 2: g2_1, g2_2)
g1_1 = 3.0
g1_2 = 13.0
g2_1 = 2.0
g2_2 = 10.0


# ── Objective functions ───────────────────────────────────────────────────────
def J1(x):
    return 0.5 * x @ Q1 @ x + b1 @ x - 3000.0


def J2(x):
    return 0.5 * x @ Q2 @ x + b2 @ x


def J2quad(eps):
    return r2 * eps**2 + r1 * eps + J2(x0)


# ── Price perturbation ────────────────────────────────────────────────────────
def prices(t, u, d1, d2):
    return u + a * np.array(
        [
            np.sin(w1 * t) + d1 * np.sin(w2 * t),
            np.sin(w2 * t) + d2 * np.sin(w1 * t),
        ]
    )


# ── Lead-lag filters: delta_i is an algebraic function of (z_i, e_i) ──────────
def delta1(z, e):
    return (g1_2 / g1_1) * e - (g1_2 / g1_1 - 1.0) * z


def delta2(z, e):
    return (g2_2 / g2_1) * e - (g2_2 / g2_1 - 1.0) * z


# ── Second-order state derivative (6-component) ───────────────────────────────
def udot(time, u):
    """
    State: u = [u1, u2, z1, e1, z2, e2]
      u1, u2   – price/action estimates
      z1, e1   – lead-lag filter states for player 1's deception delta1
      z2, e2   – lead-lag filter states for player 2's deception delta2

    delta1 = delta1(z1, e1),  delta2 = delta2(z2, e2)  (algebraic, not states)
    """
    u1, u2, z1, e1, z2, e2 = u

    d1 = delta1(z1, e1)
    d2 = delta2(z2, e2)

    x = prices(time, np.array([u1, u2]), d1, d2)
    J1v = J1(x)
    J2v = J2(x)

    return np.array(
        [
            (-2.0 * k / a) * J1v * np.sin(w1 * time),  # u1_dot
            (-2.0 * k / a) * J2v * np.sin(w2 * time),  # u2_dot
            (1.0 / g1_1) * (-z1 + e1),  # z1_dot
            epsilon1 * (J1v - J1ref),  # e1_dot
            (1.0 / g2_1) * (-z2 + e2),  # z2_dot
            epsilon2 * (J2v - J2ref),  # e2_dot
        ]
    )


# ── Initial condition ─────────────────────────────────────────────────────────
u0 = np.array([50.0, 100.0 / 3.0, 0.0, 0.0, 0.0, 0.0])


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive RK45 via scipy.integrate.solve_ivp
# ══════════════════════════════════════════════════════════════════════════════
def run_adaptive_rk45(horizon=50.0, rtol=1e-6, atol=1e-8):
    print(
        f"Running adaptive RK45 (second-order / lead-lag) over t = [0, {horizon}] ..."
    )
    sol = solve_ivp(
        fun=udot,
        t_span=(0.0, horizon),
        y0=u0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    print(f"  Done. {sol.t.size} adaptive steps taken.")
    return sol


# ── Run ────────────────────────────────────────────────────────────────────────
horizon = 50.0
sol = run_adaptive_rk45(horizon=horizon)

t_plot = np.linspace(0.0, horizon, 200_000)
u_plot = sol.sol(t_plot)  # shape (6, len(t_plot))

J1_plot = np.array([J1(u_plot[:2, i]) for i in range(len(t_plot))])
J2_plot = np.array([J2(u_plot[:2, i]) for i in range(len(t_plot))])
d1_plot = delta1(u_plot[2], u_plot[3])
d2_plot = delta2(u_plot[4], u_plot[5])

print(f"\nx0   = {x0}")
print(f"u(T) = {sol.sol(horizon)}")
print(f"delta1(T) = {delta1(*sol.sol(horizon)[2:4])}")
print(f"delta2(T) = {delta2(*sol.sol(horizon)[4:6])}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle(
    "Mutual Deception Duopoly – Second-Order (Lead-Lag) Dynamics, RK45", fontsize=13
)

axes[0, 0].plot(t_plot, J1_plot, linewidth=0.4)
axes[0, 0].axhline(J1ref, color="r", linestyle="--", label=f"J1ref = {J1ref}")
axes[0, 0].set_xlabel("time")
axes[0, 0].set_ylabel("J1")
axes[0, 0].set_title("Player 1 payoff")
axes[0, 0].legend()

axes[0, 1].plot(t_plot, J2_plot, linewidth=0.4)
axes[0, 1].axhline(J2ref, color="r", linestyle="--", label=f"J2ref = {J2ref}")
axes[0, 1].set_xlabel("time")
axes[0, 1].set_ylabel("J2")
axes[0, 1].set_title("Player 2 payoff")
axes[0, 1].legend()

axes[0, 2].plot(t_plot, u_plot[0], linewidth=0.4, label="u1")
axes[0, 2].plot(t_plot, u_plot[1], linewidth=0.4, label="u2")
axes[0, 2].set_xlabel("time")
axes[0, 2].set_ylabel("price estimate")
axes[0, 2].set_title("Price averages u1, u2")
axes[0, 2].legend()

axes[1, 0].plot(t_plot, u_plot[2], linewidth=0.4, label="z1")
axes[1, 0].plot(t_plot, u_plot[3], linewidth=0.4, label="e1")
axes[1, 0].set_xlabel("time")
axes[1, 0].set_ylabel("filter state")
axes[1, 0].set_title("Player 1 lead-lag states z1, e1")
axes[1, 0].legend()

axes[1, 1].plot(t_plot, u_plot[4], linewidth=0.4, label="z2")
axes[1, 1].plot(t_plot, u_plot[5], linewidth=0.4, label="e2")
axes[1, 1].set_xlabel("time")
axes[1, 1].set_ylabel("filter state")
axes[1, 1].set_title("Player 2 lead-lag states z2, e2")
axes[1, 1].legend()

axes[1, 2].plot(t_plot, d1_plot, linewidth=0.4, label="δ1")
axes[1, 2].plot(t_plot, d2_plot, linewidth=0.4, label="δ2")
axes[1, 2].set_xlabel("time")
axes[1, 2].set_ylabel("deception parameter")
axes[1, 2].set_title("Effective deception δ1(t), δ2(t)")
axes[1, 2].legend()

plt.tight_layout()

plt.show()
