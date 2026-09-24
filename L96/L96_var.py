#import
import sys, os
sys.path.insert(0, os.path.abspath('..'))   # the git/ folder, where utility.py lives
from utilityl96 import *  
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import qr
from tqdm import tqdm
import time as tmtrack

def gaspari_cohn(d, L):
    """
    d: array of distances (>=0)
    L: localization radius (in grid points)
    returns same shape array of taper values in [0,1]
    """
    r = np.abs(d) / L
    rho = np.zeros_like(r)

    # 0 <= r <= 1
    m1 = (r <= 1.0)
    rm = r[m1]
    rho[m1] = 1 - (5.0/3.0)*rm**2 + (5.0/8.0)*rm**3 + 0.5*rm**4 - 0.25*rm**5

    # 1 < r <= 2
    m2 = (r > 1.0) & (r <= 2.0)
    rm = r[m2]
    rho[m2] = (4 - 5*rm + (5.0/3.0)*rm**2 + (5.0/8.0)*rm**3
               - 0.5*rm**4 + (1.0/12.0)*rm**5)

    # r > 2 -> already 0
    return rho

def build_l96_localization_matrix(n=40, L=4.0):
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    d = np.minimum(d, n - d)  # periodic distance
    return gaspari_cohn(d, L)

# --- Lorenz 96 model ---
def lorenz96(x, F):
    return l96_scale*(np.roll(x, -1, axis=1) - np.roll(x, 2, axis=1))*np.roll(x, 1, axis=1) - x + F
def rk4_step(X, F, dt):
    k1 = lorenz96(X, F)
    k2 = lorenz96(X + dt * k1 / 2, F)
    k3 = lorenz96(X + dt * k2 / 2, F)
    k4 = lorenz96(X + dt * k3, F)
    return X + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
# --- Parameters ---
l96_scale = np.tile([600,200,0.5,0.15], 2500)
L_96_F = np.tile([2,10,200,500], 2500)
N = np.size(l96_scale)
F = L_96_F
dt = 0.00001

traj_rk4 = np.load("L96_truth.npy")
L96_truth = traj_rk4.copy()

obs_mask= [i*4 for i in range(int(np.size(l96_scale)/4))]
number_ens = 1000
time = 1002
assimilation_gap = 10

L96_obs = L96_truth[:time] + np.random.normal(size=(time,N)) * 1
L96_ens = np.zeros((time,number_ens,N))
L96_ens[0] = L96_truth[0] + np.random.normal(size=(number_ens,N))
H = lambda x: x

t1 = tmtrack.time()
loc_matrix = build_l96_localization_matrix(n=N,L=4)
for t in range(time-1):
    np.save("VarL96_t10_out.npy", L96_ens.mean(axis = 1))
    L96_ens[t+1] = rk4_step(L96_ens[t], F, dt)
    if (t+1) % assimilation_gap == 0:
        print('current time is', t)
        obs = torch.tensor(L96_obs[t+1])
        # replace with __all__ = ['EnVar3D', 'CVT3dEnVar', 'CVT3dEnVarPerturbedObs', 'EnVar3DPerturbedObs']
        envar = CVT3dEnVarPerturbedObs(H=H, y_obs=obs, B = torch.cov(torch.tensor(L96_ens[t+1].T)) * torch.tensor(loc_matrix) ,R=torch.eye(N)*1, \
                        Xb= torch.tensor(L96_ens[t+1]),obs_idx=obs_mask, learning_rate=0.0001, \
                            max_iterations=10000, convergence_threshold=1e-3, device = "cuda:7")
    # Run assimilation
        x_analysis,idx_nan,output = envar.run() #,x_path
        L96_ens[t+1] = x_analysis.cpu().numpy()
        print("Analysis state:", x_analysis.mean(axis = 0))
        if idx_nan:
            print("stop at time", t)
            break
t2 = tmtrack.time()
print(f"Elapsed time: {t2 - t1:.3f} seconds")
np.save("VarL96_t10_out.npy", L96_ens.mean(axis = 1))