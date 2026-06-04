#import
import torch
import numpy as np
from ensf_inc import EnSF
import matplotlib.pyplot as plt
import math
from scipy.linalg import qr
from tqdm import tqdm
import time as tmtrack

def rk4_step_safe(X, F, dt, n_sub):
    """
    RK4 with fallback: if the coarse step overflows or produces NaNs,
    redo using n_sub substeps of size dt/n_sub.
    """
    # --- First try: single big step ---
    with np.errstate(over='raise', invalid='raise'):
        try:
            X_new = rk4_step(X, F, dt)
            if not np.all(np.isfinite(X_new)):
                raise FloatingPointError("Non-finite values in RK4 step.")
            return X_new
        except FloatingPointError:
            # fall through to fine-stepping
            pass

    # --- Fallback: fine steps with smaller dt ---
    new_dt = dt / n_sub
    X_fine = X.copy()

    with np.errstate(over='raise', invalid='raise'):
        print("fine RK4 Applied")
        for _ in range(n_sub):
            X_fine = rk4_step(X_fine, F, new_dt)
            if not np.all(np.isfinite(X_fine)):
                raise RuntimeError(
                    f"RK4 still produced non-finite values even with dt={new_dt}."
                )
    return X_fine

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
Use_this_GPU = 'cuda:0'
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
assimilation_gap = 5
cov_control = 2
loc_r = 4

L96_obs = L96_truth[:time,obs_mask] + np.random.normal(size=(time,len(obs_mask))) * 1
L96_ens = np.zeros((time,number_ens,N))
ens_inf = np.tile(1.00*np.array((1, 1, 1, 1)), 2500)
L96_ens[0] = L96_truth[0] + np.random.normal(size=(number_ens,N)) * ens_inf

EnSF_inc = EnSF(n_dim = N, ensemble_size = number_ens ,eps_alpha=0.01, eps_beta= 0.00, device= Use_this_GPU ,obs_sigma = 0.03, euler_steps = 1000)
t1 = tmtrack.time()
loc_matrix = build_l96_localization_matrix(n=N,L=loc_r)
for t in range(time):
    np.save(f"EnSF_L96_1000ensembles_t{assimilation_gap}.npy", L96_ens.mean(axis = 1))
    L96_ens[t+1] = rk4_step_safe(L96_ens[t], F, dt, n_sub=100) #rk4_step(L96_ens[t], F, dt)
    if (t+1) % assimilation_gap == 0:
        print('current time is', t)
        temp_cov = np.cov(L96_ens[t+1].T)*loc_matrix
        cov_scale = np.diag(temp_cov)
        temp_scale = np.abs(temp_cov).max()/cov_control
        xens = EnSF_inc.state_inc(obs = L96_obs[t+1], x0=np.zeros((number_ens,N)),x_prior=L96_ens[t+1],\
                                  ens_cov= temp_cov/temp_scale,sparse_idx=obs_mask) #
        L96_ens[t+1] = xens.cpu().numpy()
t2 = tmtrack.time()
print(f"Elapsed time: {t2 - t1:.3f} seconds")
np.save(f"EnSF_L96_1000ensembles_t{assimilation_gap}.npy", L96_ens.mean(axis = 1))