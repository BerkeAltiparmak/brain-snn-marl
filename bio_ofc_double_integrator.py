
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