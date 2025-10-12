import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)

# True plant (double integrator)
A_true = np.array([[1.0, 1.0],
                   [0.0, 1.0]])
B_true = np.array([[0.0],
                   [1.0]])
C_true = np.eye(2)          # LDS1: full observation
V = np.diag([0.01, 0.01])   # process noise
W = np.diag([0.04, 0.25])   # obs noise

# LQR cost
Q = np.array([[1.0, 0.0],
              [0.0, 0.0]])
R = 1.0

def col(v): return np.array(v, float).reshape(-1,1)

# ----- Utilities: DARE (for LQR), spectral radius -----
def dare_iter(A, B, Q, R, iters=3000, tol=1e-9):
    P = Q.copy()
    for _ in range(iters):
        BtPB = B.T @ P @ B
        K = np.linalg.solve(R + BtPB, B.T @ P @ A)
        P_new = A.T @ P @ A - A.T @ P @ B @ K + Q
        if np.linalg.norm(P_new - P, ord='fro') < tol:
            return P_new
        P = P_new
    return P

def lqr_gain(A, B, Q, R):
    P = dare_iter(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

def rho(M):  # spectral radius
    return float(np.max(np.abs(np.linalg.eigvals(M))))

# ======================================================
# Minimal local-learner: learn only a=A01 and b=B10; C fixed to I; L fixed
# ======================================================
class MinimalLocalDI:
    def __init__(self, tau=1):
        self.tau = 1                # use τ=1 as in the paper’s simple predictor
        self.a = 0.2                # A[0,1]
        self.b = 0.2                # B[1,0]
        self.C = np.eye(2)          # FIXED for LDS1
        self.L = np.diag([0.2, 0.4])# simple fixed observer gain (works well here)

        # learning rates (normalized LMS-style)
        self.alpha_a = 5e-4
        self.alpha_b = 5e-4

        # controller (we’ll replace with LQR after ID)
        self.K = np.array([[0.0, 0.0]])

    @property
    def A(self):
        return np.array([[1.0, self.a],
                         [0.0, 1.0]])

    @property
    def B(self):
        return np.array([[0.0],
                         [self.b]])

    def open_loop_id(self, episodes=180, T=60, sigma_u=0.5):
        """ID with random u_t, local rules using innovation e_t = y_t - xhat_t."""
        mse = []
        for ep in range(episodes):
            x = col([-1.0, 0.0]) + 0.1*np.random.randn(2,1)
            xhat = x.copy()
            xh=[xhat.copy()]; uh=[np.zeros((1,1))]; eh=[np.zeros((2,1))]
            ms = 0.0
            for t in range(T):
                y = C_true @ x + col(np.random.multivariate_normal(np.zeros(2), W))
                e = y - xhat                                   # innovation

                # predictor
                u = col(np.random.randn(1) * sigma_u)
                xhat_next = self.A @ xhat + self.B @ u + self.L @ e

                # local parameter updates (delayed presynaptic terms, τ=1)
                if t >= 1:
                    x_tau = xh[-1]; u_tau = uh[-1]; e_tau = eh[-1]
                    Le = self.L @ e

                    nx = float(1e-6 + x_tau[1,0]**2)          # normalize
                    nu = float(1e-6 + u_tau[0,0]**2)

                    # ∆a ∝ (Le_pos) * x_vel(τ)
                    self.a += self.alpha_a * (Le[0,0] * x_tau[1,0]) / nx
                    # ∆b ∝ (Le_vel) * u(τ)
                    self.b += self.alpha_b * (Le[1,0] * u_tau[0,0]) / nu

                    # keep params in plausible range
                    self.a = float(np.clip(self.a, 0.5, 1.5))
                    self.b = float(np.clip(self.b, 0.1, 2.0))

                # true dynamics
                v = col(np.random.multivariate_normal(np.zeros(2), V))
                x = A_true @ x + B_true @ u + v

                ms += float((e.T @ e))
                xhat = xhat_next
                xh.append(xhat.copy()); uh.append(u.copy()); eh.append(e.copy())
            mse.append(ms / T)
        return np.array(mse)

    def set_lqr_controller(self):
        """Stable K from LQR on (A_hat, B_hat)."""
        K = lqr_gain(self.A, self.B, Q, R)
        # mild backoff if needed
        if rho(self.A - self.B @ K) >= 1.0:
            K *= 0.95 / rho(self.A - self.B @ K)
        self.K = K

    def evaluate(self, trials=40, T=40, use_estimator=True):
        """Deterministic controller eval with or without estimator in the loop."""
        total = 0.0
        cost = []
        for _ in range(trials):
            x = col([-1.0, 0.0]) + 0.1*np.random.randn(2,1)
            xhat = x.copy()
            c = 0.0
            for t in range(T):
                if use_estimator:
                    y = C_true @ x + col(np.random.multivariate_normal(np.zeros(2), W))
                    e = y - xhat
                    u = - self.K @ xhat
                    xhat = self.A @ xhat + self.B @ u + self.L @ e
                else:
                    u = - self.K @ x

                v = col(np.random.multivariate_normal(np.zeros(2), V))
                x = A_true @ x + B_true @ u + v
                c += float((x.T @ Q @ x + R * (u.T @ u)))
            total += c
            cost.append(c/T)
        return total / trials, cost


# ---------------- Run: ID -> LQR -> Evaluate ----------------
agent = MinimalLocalDI(tau=1)

mse = agent.open_loop_id(episodes=180, T=60, sigma_u=0.5)
A_hat, B_hat = agent.A.copy(), agent.B.copy()

agent.set_lqr_controller()
cost_det_noKF, cost_det_noKF_list = agent.evaluate(trials=1000, T=40, use_estimator=False)  # ideal state for control
cost_det_withKF, cost_det_withKF_list = agent.evaluate(trials=1000, T=40, use_estimator=True) # using xhat in the loop

print("=== Identification ===")
print("MSE last-10 avg:", float(np.mean(mse[-10:])))
print("A_hat:\n", A_hat, "\nB_hat:\n", B_hat)

print("\n=== Control (LQR on (A_hat,B_hat)) ===")
print("K:\n", agent.K, "  rho(A-BK):", rho(agent.A - agent.B @ agent.K))
print("Avg cost (no estimator in loop):", float(cost_det_noKF))
print("Avg cost (with estimator in loop):", float(cost_det_withKF))

plt.figure()
plt.plot(mse); plt.xlabel("Episode"); plt.ylabel("Innovation MSE"); plt.title("ID MSE"); plt.tight_layout(); plt.show()
plt.plot(cost_det_noKF_list); plt.xlabel("Episode"); plt.ylabel("Controller Cost"); plt.title("Controller Cost"); plt.tight_layout(); plt.show()
plt.plot(cost_det_withKF_list); plt.xlabel("Episode"); plt.ylabel("Controller Cost"); plt.title("Controller Cost"); plt.tight_layout(); plt.show()
