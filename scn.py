"""
Spiking LQG via Spike Coding Networks (SCN): Closed-form scaffold
-------------------------------------------------------------------
Implements the voltage dynamics and weight blocks from the paper:
- Fast term:   Ω_f = - D^T D                               (instantaneous reset on spikes)
- Slow term:   Ω_s = D_x^T (A + λ I) D_x
- Kalman:      Ω_k = - D_x^T K_f C D_x,   F_k = D_x^T K_f
- Control:     Ω_c = - D_x^T B K_c D_x,   Ω_z = D_x^T B K_c D_z
- Readouts:    x̂ = D_x r, ẑ = D_z r,    u = D_u r with D_u = -K_c (D_x - D_z)
- Target repr: D_z^T (ż + λ z)

A simple spring–mass–damper (SMD) demo is provided at the bottom.
"""

from dataclasses import dataclass
import numpy as np
from numpy.random import default_rng
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

def make_random_decoders(K: int, N: int, rng, col_norm: float = 1.0):
    """D in R^{K x N}. Columns ~ N(0, I) then normalized to 'col_norm'."""
    D = rng.normal(size=(K, N))
    norms = np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    D = D / norms * col_norm
    return D

def continuous_lqe(A, C, Sigma_d, Sigma_n):
    """
    Continuous-time Kalman gain (LQE): K_f = P_e C^T Σ_n^{-1},
    where P_e solves: A P + P A^T - P C^T Σ_n^{-1} C P + Σ_d = 0
    """
    # CARE on estimator in "dual" form: A^T, C^T, etc.
    # SciPy solves: A^T P + P A - P C^T R^{-1} C P + Q = 0  -> with Q=Σ_d, R=Σ_n
    P = solve_continuous_are(A.T, C.T, Sigma_d, Sigma_n)
    Kf = (P @ C.T) @ np.linalg.inv(Sigma_n)
    return Kf, P

def continuous_lqr(A, B, Q, R):
    """
    Continuous-time LQR: u = -K_c x, K_c = R^{-1} B^T P_c,
    where P_c solves: A^T P + P A - P B R^{-1} B^T P + Q = 0
    """
    P = solve_continuous_are(A, B, Q, R)
    Kc = np.linalg.inv(R) @ (B.T @ P)
    return Kc, P


# -----------------------------
# Data classes for clarity
# -----------------------------

@dataclass
class Plant:
    A: np.ndarray     # (K x K)
    B: np.ndarray     # (K x U)
    C: np.ndarray     # (M x K)
    Sigma_d: np.ndarray  # (K x K) process noise cov (continuous)
    Sigma_n: np.ndarray  # (M x M) measurement noise cov (continuous)

@dataclass
class LQG:
    Kf: np.ndarray    # (K x M)
    Kc: np.ndarray    # (U x K)

@dataclass
class SCNWeights:
    # N neurons, K state dim, M output dim, U control dim
    Omega_f_x: np.ndarray  # (N x N)
    Omega_f_z: np.ndarray  # (N x N)
    Omega_s:   np.ndarray  # (N x N)
    Omega_k:   np.ndarray  # (N x N)
    F_k:       np.ndarray  # (N x M)
    Omega_c:   np.ndarray  # (N x N)
    Omega_z:   np.ndarray  # (N x N)
    D_x:       np.ndarray  # (K x N)
    D_z:       np.ndarray  # (K x N)
    D_u:       np.ndarray  # (U x N)
    T:         np.ndarray  # (N,) thresholds


# -----------------------------
# Closed-form construction
# -----------------------------

def build_scn_weights(plant: Plant, lqg: LQG, N: int, lam: float, rng, col_norm=1.0) -> SCNWeights:
    """
    Build all weight blocks as in the paper, for a shared population that
    jointly represents x̂ and ẑ with separate decoders D_x and D_z.
    """
    A, B, C = plant.A, plant.B, plant.C
    Kf, Kc = lqg.Kf, lqg.Kc
    K, U = A.shape[0], B.shape[1]
    M = C.shape[0]

    # Decoders (K x N)
    D_x = make_random_decoders(K, N, rng, col_norm=col_norm)
    D_z = make_random_decoders(K, N, rng, col_norm=col_norm)

    # Fast lateral terms (instantaneous resets on spikes)
    Omega_f_x = - D_x.T @ D_x                      # (N x N)
    Omega_f_z = - D_z.T @ D_z                      # (N x N)

    # Slow term implements (A + λ I) on the x̂ readout
    Omega_s = D_x.T @ (A + lam * np.eye(K)) @ D_x  # (N x N)

    # Kalman innovation injection: D_x^T Kf (y - C x̂)
    F_k = D_x.T @ Kf                               # (N x M)
    Omega_k = - D_x.T @ Kf @ C @ D_x               # (N x N)

    # Control effect after substituting u = -Kc (x̂ - ẑ)
    # Split into Ω_c r + Ω_z r = D_x^T B (-Kc D_x + Kc D_z) r
    Omega_c = - D_x.T @ B @ Kc @ D_x               # (N x N)
    Omega_z =   D_x.T @ B @ Kc @ D_z               # (N x N)

    # Control readout (U x N)
    D_u = - Kc @ (D_x - D_z)

    # Spike thresholds: T_i = 0.5 (||D_x[:,i]||^2 + ||D_z[:,i]||^2)
    Tx = np.sum(D_x * D_x, axis=0)
    Tz = np.sum(D_z * D_z, axis=0)
    T  = 0.5 * (Tx + Tz)

    return SCNWeights(
        Omega_f_x=Omega_f_x, Omega_f_z=Omega_f_z, Omega_s=Omega_s,
        Omega_k=Omega_k, F_k=F_k, Omega_c=Omega_c, Omega_z=Omega_z,
        D_x=D_x, D_z=D_z, D_u=D_u, T=T
    )


# -----------------------------
# Simulation loop
# -----------------------------

@dataclass
class SimConfig:
    dt: float = 5e-4
    T: float = 4.0
    lam: float = 10.0
    max_spikes_per_step: int = 5_000
    silence_at: float | None = None  # time to silence a fraction of neurons
    silence_frac: float = 0.0        # 0.3 means silence 30% of neurons


def step_target_profile(t, steps=((0.5, 0.2), (2.0, -0.15), (3.0, 0.0))):
    """
    Piecewise-constant desired POSITION for SMD (units arbitrary).
    'steps' is a tuple of (t_switch, level).
    """
    level = 0.0
    for ts, val in steps:
        if t >= ts:
            level = val
    return level


def run_scn_lqg_smd_demo(N=200, seed=1, cfg: SimConfig = SimConfig()):
    rng = default_rng(seed)

    # --- Define the plant: spring–mass–damper in state [x, v]^T
    m, k, c = 1.0, 20.0, 2.0
    A = np.array([[0, 1],
                  [-k/m, -c/m]])
    B = np.array([[0.0],
                  [1.0/m]])
    C = np.array([[1.0, 0.0]])  # observe position only
    K = A.shape[0]; U = B.shape[1]; M = C.shape[0]

    # Noise covariances (continuous-time interpretation)
    Sigma_d = np.diag([1e-3, 1e-2])   # process noise
    Sigma_n = np.diag([1e-3])         # measurement noise

    plant = Plant(A=A, B=B, C=C, Sigma_d=Sigma_d, Sigma_n=Sigma_n)

    # LQG gains
    Q = np.diag([50.0, 1.0])   # track position (x) strongly, mild penalty on velocity
    R = np.diag([0.1])         # control effort penalty
    Kc, Pc = continuous_lqr(A, B, Q, R)
    Kf, Pe = continuous_lqe(A, C, Sigma_d, Sigma_n)
    lqg = LQG(Kf=Kf, Kc=Kc)

    # Build SCN weights
    W = build_scn_weights(plant, lqg, N=N, lam=cfg.lam, rng=rng, col_norm=1.0)

    # Precompute
    dt, Ttot = cfg.dt, cfg.T
    steps = int(Ttot / dt)
    tgrid = np.arange(steps) * dt

    # State variables
    x = np.zeros(K)                 # plant state
    v = np.zeros(N)                 # neuron voltages
    r = np.zeros(N)                 # filtered spikes
    s = np.zeros(N)                 # spike rates (N/dt) within a time step
    active = np.ones(N, dtype=bool) # neuron activity mask (for silencing)

    # Logging
    X   = np.zeros((steps, K))      # true state
    Xh  = np.zeros((steps, K))      # estimated x̂ = D_x r
    U   = np.zeros((steps, U))      # control
    Y   = np.zeros((steps, M))      # measurement
    Z   = np.zeros((steps, K))      # target z
    SPT = []                        # spike raster list of (time_idx, neuron_idx)

    # For target derivative
    z_prev = np.zeros(K)

    # For silencing
    silenced = False
    silence_step = None
    if cfg.silence_at is not None:
        silence_step = int(cfg.silence_at / dt)

    # Combined fast reset matrix for shared population
    Omega_f_total = W.Omega_f_x + W.Omega_f_z

    for t_idx in range(steps):
        t = tgrid[t_idx]

        # -- Target z(t) in state space: track desired position, zero velocity
        z_pos = step_target_profile(t)
        z = np.array([z_pos, 0.0])
        z_dot = (z - z_prev) / dt   # derivative (zero except at step changes)
        z_prev = z.copy()

        # -- Readouts
        x_hat = W.D_x @ r
        y_hat = C @ x_hat

        # -- Control readout u = D_u r
        u = (W.D_u @ r).reshape(-1)

        # -- Plant evolution (Euler + noise)
        # Continuous-time noise ~ N(0, Σ_d * dt)
        w = rng.multivariate_normal(mean=np.zeros(K), cov=plant.Sigma_d * dt)
        x = x + dt * (plant.A @ x + plant.B @ u) + w
        # Measurement with noise (discrete sample)
        n = rng.multivariate_normal(mean=np.zeros(M), cov=plant.Sigma_n)
        y = (plant.C @ x) + n

        # -- Voltage dynamics (continuous part)
        # NOTE: We DO NOT include Ω_f * s here because we apply instantaneous resets
        # every time a spike occurs (fast term handled explicitly below).
        dv = (-cfg.lam * v
              + W.Omega_s @ r
              + W.Omega_c @ r
              + W.Omega_z @ r
              + W.Omega_k @ r
              + W.F_k @ y
              + (W.D_z.T @ (z_dot + cfg.lam * z))
              )

        v = v + dt * dv

        # -- Greedy one-at-a-time spiking with instantaneous fast resets
        s[:] = 0.0
        spikes_used = 0
        # Effective "distance to threshold":
        delta = v - W.T

        while True:
            # pick neuron with largest (v_i - T_i)
            i = np.argmax(delta)
            if delta[i] <= 0.0 or spikes_used >= cfg.max_spikes_per_step:
                break
            if not active[i]:
                # If silenced, prevent spiking by setting it below threshold and continue
                delta[i] = -np.inf
                continue

            # Emit a spike at neuron i
            s[i] += 1.0 / dt   # so that ∫ s dt = number of spikes
            SPT.append((t_idx, i))

            # Fast instantaneous reset: v ← v + Ω_f_total * e_i  (add i-th column)
            v += Omega_f_total[:, i]

            # Update distance to threshold efficiently (only columns change)
            delta = v - W.T
            spikes_used += 1

        # -- Filtered spikes r:  ṙ = -λ r + s
        r = r + dt * (-cfg.lam * r + s)

        # -- Optional neuron silencing at a given time
        if (not silenced) and (silence_step is not None) and (t_idx >= silence_step):
            silenced = True
            num_silence = int(cfg.silence_frac * N)
            kill_idx = rng.choice(np.where(active)[0], size=num_silence, replace=False)
            active[kill_idx] = False
            # Clamp their voltages and filtered spikes
            v[kill_idx] = 0.0
            r[kill_idx] = 0.0

        # -- Log
        X[t_idx]  = x
        Xh[t_idx] = W.D_x @ r
        U[t_idx]  = u
        Y[t_idx]  = y
        Z[t_idx]  = z

    # ------------- Plots -------------
    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    axs[0].plot(tgrid, X[:, 0], label='x (true position)')
    axs[0].plot(tgrid, Xh[:, 0], label='x̂ (SCN)')
    axs[0].plot(tgrid, Z[:, 0], label='z (target)', linestyle='--')
    axs[0].set_ylabel('Position')
    axs[0].legend(loc='best')
    axs[0].grid(True)

    axs[1].plot(tgrid, X[:, 1], label='v (true)')
    axs[1].plot(tgrid, Xh[:, 1], label='v̂ (SCN)')
    axs[1].set_ylabel('Velocity')
    axs[1].legend(loc='best')
    axs[1].grid(True)

    axs[2].plot(tgrid, U[:, 0], label='u')
    axs[2].set_ylabel('Control')
    axs[2].legend(loc='best')
    axs[2].grid(True)

    # Spike raster
    axs[3].set_ylabel('Neuron idx')
    axs[3].set_xlabel('Time (s)')
    if len(SPT) > 0:
        tt, ii = zip(*SPT)
        axs[3].scatter(np.array(tt) * dt, np.array(ii), s=1.0)
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

    # Return logs for further analysis if needed
    return {
        "t": tgrid, "X": X, "Xh": Xh, "U": U, "Y": Y, "Z": Z,
        "spikes": np.array(SPT, dtype=int),
        "weights": W
    }


if __name__ == "__main__":
    # Example run:
    cfg = SimConfig(
        dt=5e-4, T=4.0, lam=10.0,
        silence_at=2.0, silence_frac=0.3  # silence 30% neurons at t=2s
    )
    logs = run_scn_lqg_smd_demo(N=250, seed=7, cfg=cfg)
