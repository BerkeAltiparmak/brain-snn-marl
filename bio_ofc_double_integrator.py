
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)

A_true = np.array([[1.0, 1.0],[0.0, 1.0]])
B_true = np.array([[0.0],[1.0]])
C_true = np.eye(2)
V = np.diag([0.01, 0.01])
W = np.diag([0.04, 0.25])
Q = np.array([[1.0, 0.0],[0.0, 0.0]])
R = 1.0

def col(v): return np.array(v, dtype=float).reshape(-1,1)
def clip_matrix(M, limit=1.0): np.clip(M, -limit, limit, out=M)

class BioOFCDI2:
    def __init__(self, tau=1):
        self.tau = tau  # for now not used
        self.n,self.m,self.p = 2,1,2
        self.A = np.eye(2)
        self.B = np.array([[0.0],[0.1]])
        self.C = np.eye(2)
        self.L = 0.05*np.random.randn(2,2)
        self.K = 0.01*np.random.randn(1,2)
        self.alpha_A = 1e-3; self.alpha_B = 1e-3; self.alpha_C = 1e-3; self.alpha_L = 5e-3
        self.lr_K = 1e-4; self.momentum = 0.9; self.sigma = 0.05
        self.Z = np.zeros((1,2)); self.G = np.zeros((1,2))
    def open_loop_id(self, episodes=60, T=20):
        mse=[]
        for ep in range(episodes):
            x = col([-1.0,0.0]); xhat = x.copy()
            xhat_hist=[xhat.copy()]; u_hist=[np.zeros((1,1))]; e_hist=[np.zeros((2,1))]
            ms=0.0
            for t in range(T):
                y = C_true @ x + col(np.random.multivariate_normal(np.zeros(2), W))
                e = y - self.C @ xhat
                u = col(np.random.randn(1)*0.1)
                xhat_next = self.A@xhat + self.B@u + self.L@e
                if t>=1:
                    x_tau=xhat_hist[-1]; u_tau=u_hist[-1]; e_tau=e_hist[-1]; Le=self.L@e
                    self.A += self.alpha_A*(Le@x_tau.T)
                    self.B += self.alpha_B*(Le@u_tau.T)
                    self.L += self.alpha_L*(Le@e_tau.T)
                    self.C += self.alpha_C*(e@xhat_next.T)
                    #clip_matrix(self.A,2.0); clip_matrix(self.B,2.0); clip_matrix(self.C,2.0); clip_matrix(self.L,2.0)
                v = col(np.random.multivariate_normal(np.zeros(2), V))
                x = A_true@x + B_true@u + v
                ms += float((e.T@e).squeeze())
                xhat=xhat_next; xhat_hist.append(xhat.copy()); u_hist.append(u.copy()); e_hist.append(e.copy())
            mse.append(ms/T)
        return np.array(mse), self.A, self.B, self.C
    def controller_learning(self, episodes=120, T=20):
        costs=[]; self.Z[:]=0; self.G[:]=0
        for ep in range(episodes):
            x = col([-1.0,0.0]); xhat=x.copy(); csum=0.0
            for t in range(T):
                y = C_true @ x + col(np.random.multivariate_normal(np.zeros(2), W))
                e = y - self.C @ xhat
                xi = col(np.random.randn(1)*self.sigma)
                u = - self.K @ xhat - xi
                xhat_next = self.A@xhat + self.B@u + self.L@e
                v = col(np.random.multivariate_normal(np.zeros(2), V))
                x = A_true@x + B_true@u + v
                c = float((x.T@Q@x + R*(u.T@u)).squeeze()); csum += c
                self.Z += xi @ xhat.T
                self.G = self.momentum*self.G + c*self.Z
                self.K -= self.lr_K*self.G
                clip_matrix(self.K, 1.0)
                xhat = xhat_next
            costs.append(csum / T)
        return np.array(costs)
    def evaluate(self, trials=10, T=20):
        tot=0.0
        for _ in range(trials):
            x = col([-1.0,0.0]); xhat=x.copy(); c=0.0
            for t in range(T):
                y = C_true @ x + col(np.random.multivariate_normal(np.zeros(2), W))
                e = y - self.C @ xhat
                u = - self.K @ xhat
                xhat = self.A@xhat + self.B@u + self.L@e
                v = col(np.random.multivariate_normal(np.zeros(2), V))
                x = A_true@x + B_true@u + v
                c += float((x.T@Q@x + R*(u.T@u)).squeeze())
            tot += c / T
        return tot/trials
