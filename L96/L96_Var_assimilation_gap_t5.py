#import
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import qr
from tqdm import tqdm
import time as tmtrack

# 3D Var
class EnVar3D:
    def __init__(
        self,
        H,
        y_obs,
        B,
        R,
        Xb,
        obs_idx,
        learning_rate=1e-2,
        max_iterations=100,
        convergence_threshold=1e-5,
        device="cuda"
    ):
        self.device = device
        self.obs_idx = obs_idx
        self.H = H  # Observation operator (callable)
        self.y_obs = y_obs.to(device)
        self.B_inv = torch.linalg.inv(B).to(device)
        self.B = B.to(device)
        # 1) Symmetrize (important after localization & float32)
        self.B = 0.5 * (self.B + self.B.T)
        # 2) Eigen-decompose and clamp small / negative eigenvalues
        evals, Q = torch.linalg.eigh(self.B)  # evals sorted ascending
        # Choose a floor for eigenvalues (e.g. 1e-3 of max or based on diag)
        lam_max = evals.max()
        eps = 1e-3 * lam_max  # you can try 1e-4, 1e-2 etc.
        evals_clamped = torch.clamp(evals, min=eps)
        # Rebuild a strictly SPD matrix
        self.B = (Q * evals_clamped) @ Q.T
        # Optional: symmetrize again just to be clean
        self.B = 0.5 * (self.B + self.B.T)
        self.L = torch.linalg.cholesky(self.B)  # B = L L^T
        self.R_inv = torch.linalg.inv(R).to(device)
        self.x_prior = Xb.to(device)  # Background ensemble (N, state_dim)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.ensemble_size, self.state_dim = self.x_prior.shape
        self.x_inc_init = torch.zeros_like(self.x_prior).to(device)

    def run(self):
        sparse_index = torch.zeros(self.x_prior.size(dim = 1),dtype=torch.bool,device=self.device)
        sparse_index[self.obs_idx] = True
        x_inc = torch.zeros_like(self.x_prior, requires_grad=True).to(self.device)
        prev_cost = float('inf')
        idx_nan = False
        output = torch.zeros(self.max_iterations+1,self.state_dim).to(self.device)
        output[0,:] = (self.x_prior + x_inc).mean(dim = 0)
        for iteration in range(self.max_iterations):
            if x_inc.grad is not None:
                x_inc.grad.zero_()
            # Apply observation operator
            Hx_inc = self.H(x_inc - self.x_inc_init)
            obs_inc = (self.y_obs - self.H(self.x_prior))
            # Innovation
            innovation = Hx_inc - obs_inc
            #print(innovation)
            # Cost function. self.R_inv = self.R_inv.unsqueeze(0).unsqueeze(1)
            R_inv = self.R_inv.clone()
            #print(sparse_index)
            R_inv[:,~sparse_index] = 0
            
            cost_obs = 0.5 * torch.einsum("ij,jk,ik->i", innovation.float(), R_inv.float(), innovation.float())  # Efficient quadratic form
            # cost_bkg = 0.5 * torch.einsum("ij,jk,ik->i", x_inc, self.B_inv, x_inc)            # Efficient quadratic form
            #y = torch.cholesky_solve(x_inc.unsqueeze(-1), self.L).squeeze(-1)  # shape (N, state_dim)
            rhs = x_inc.T.contiguous()                 # (d, N)
            y  = torch.cholesky_solve(rhs, self.L).T   # (N, d)

            cost_bkg = 0.5 * torch.einsum("ij,ij->i", x_inc, y)
            cost = cost_obs + cost_bkg
            #print(self.B_inv)

            grad_x_inc = torch.autograd.grad(outputs=cost,
                                 inputs=x_inc,
                                 grad_outputs=torch.ones_like(cost),  # Identity for element-wise gradient
                                 create_graph=True)[0]


            with torch.no_grad():
                x_inc -= self.learning_rate * grad_x_inc
            cost_value = cost.mean().item()
            output[iteration+1] = (self.x_prior + x_inc).mean(dim = 0)
            if (iteration < 1) or(iteration%10000 == 0):
                print(f"Iteration {iteration+1}: Cost = {cost_value:.6f}")
            #print(self.x_prior + x_inc)
            
            if cost_value>10000000000000:
                print("Cost is NaN")
                idx_nan = True
                break

            if abs(prev_cost - cost_value) < self.convergence_threshold:
                print("Convergence reached.")
                break

            prev_cost = cost_value

        # Final analysis state 
        x_analysis = self.x_prior + x_inc
        return x_analysis.detach() , idx_nan,  output[:iteration+2].detach()
    
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
assimilation_gap = 5

L96_obs = L96_truth[:time] + np.random.normal(size=(time,N)) * 1
L96_ens = np.zeros((time,number_ens,N))
L96_ens[0] = L96_truth[0] + np.random.normal(size=(number_ens,N))
H = lambda x: x

t1 = tmtrack.time()
loc_matrix = build_l96_localization_matrix(n=N,L=4)
for t in range(time-1):
    np.save("VarL96_t5_out.npy", L96_ens.mean(axis = 1))
    L96_ens[t+1] = rk4_step(L96_ens[t], F, dt)
    if (t+1) % assimilation_gap == 0:
        print('current time is', t)
        obs = torch.tensor(L96_obs[t+1])
        envar = EnVar3D(H=H, y_obs=obs, B = torch.cov(torch.tensor(L96_ens[t+1].T)) * torch.tensor(loc_matrix) ,R=torch.eye(N)*0.5, \
                        Xb= torch.tensor(L96_ens[t+1]),obs_idx=obs_mask, learning_rate=0.0001, \
                            max_iterations=50000, convergence_threshold=1e-3, device = "cuda")
    # Run assimilation
        x_analysis,idx_nan,output = envar.run() #,x_path
        L96_ens[t+1] = x_analysis.cpu().numpy()
        print("Analysis state:", x_analysis.mean(axis = 0))
        if idx_nan:
            print("stop at time", t)
            break
t2 = tmtrack.time()
print(f"Elapsed time: {t2 - t1:.3f} seconds")
np.save("VarL96_t5_out.npy", L96_ens.mean(axis = 1))