"""
EnVar3D                 3D-EnVar, gradient descent directly on the state increment.
CVT3dEnVar              3D-EnVar with the classical control-variable transform x_inc = B^(1/2) v.
CVT3dEnVarPerturbedObs  CVT3dEnVar where each member assimilates its own perturbed observation.
EnVar3DPerturbedObs     EnVar3D where each member assimilates its own perturbed observation.
"""
import torch

__all__ = ['EnVar3D', 'CVT3dEnVar', 'CVT3dEnVarPerturbedObs', 'EnVar3DPerturbedObs']

use_this = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        device=use_this
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
        self.B_lam_max = evals_clamped.max().detach()
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
            # Solve B y = x_inc for all members at once (N right-hand sides); the batched
            # x_inc.unsqueeze(-1) form broadcasts L to (N, d, d) and runs out of memory for large d.
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
            if (iteration < 1000) or(iteration%10000 == 0):
                print(f"Iteration {iteration+1}: Cost = {cost_value:.6f}")
            #print(self.x_prior + x_inc)

            if cost_value>10000000000: #10000000
                print("Cost is NaN")
                idx_nan = True
                break

            if abs(prev_cost - cost_value) < self.convergence_threshold:
                print("Convergence reached.")
                break

            prev_cost = cost_value

        # Final analysis state
        x_analysis = self.x_prior + x_inc
        print("iteration stopped at step: ",iteration)
        return x_analysis.detach()

class CVT3dEnVar(EnVar3D):
    """3D-EnVar using the classical control-variable transform."""

    def run(self):
        sparse_index = torch.zeros(self.x_prior.size(dim = 1),dtype=torch.bool,device=self.device)
        sparse_index[self.obs_idx] = True
        # Cap the step by the CVT Hessian bound: lambda_max(I + L'H'R^-1 H L) <= 1 + lambda_max(B) * max(R^-1 on observed dims)
        R_inv_obs = torch.diagonal(self.R_inv)[self.obs_idx]
        curvature_bound = 1.0 + self.B_lam_max * R_inv_obs.max()
        self.effective_learning_rate = min(self.learning_rate, 0.9 / curvature_bound.item())
        print(f"Effective learning rate: {self.effective_learning_rate:.6e}")
        control_var = torch.zeros_like(self.x_prior, requires_grad=True)
        prev_cost = float('inf')
        idx_nan = False
        x_inc = control_var @ self.L.T  # x_inc = L v in row-batch form
        for iteration in range(self.max_iterations):
            if control_var.grad is not None:
                control_var.grad.zero_()
            # Classical control-variable transform: x_inc = B^(1/2) v
            x_inc = control_var @ self.L.T
            # Apply observation operator
            Hx_inc = self.H(x_inc - self.x_inc_init)
            obs_inc = (self.y_obs - self.H(self.x_prior))
            # Innovation
            innovation = Hx_inc - obs_inc
            # Cost function. self.R_inv = self.R_inv.unsqueeze(0).unsqueeze(1)
            R_inv = self.R_inv.clone()
            R_inv[:,~sparse_index] = 0

            cost_obs = 0.5 * torch.einsum("ij,jk,ik->i", innovation.float(), R_inv.float(), innovation.float())
            cost_bkg = 0.5 * torch.einsum("ij,ij->i", control_var, control_var)
            cost = cost_obs + cost_bkg

            grad_control_var = torch.autograd.grad(outputs=cost,
                                 inputs=control_var,
                                 grad_outputs=torch.ones_like(cost),
                                 create_graph=True)[0]

            with torch.no_grad():
                control_var -= self.effective_learning_rate * grad_control_var
            x_inc = control_var @ self.L.T
            cost_value = cost.mean().item()

            if (iteration < 1000) or(iteration%10000 == 0):
                print(f"Iteration {iteration+1}: Cost = {cost_value:.6f}")

            if cost_value>10000000000: #10000000
                print("Cost is NaN")
                idx_nan = True
                break

            if abs(prev_cost - cost_value) < self.convergence_threshold:
                print("Convergence reached.")
                break

            prev_cost = cost_value

        # Final analysis state
        x_inc = control_var @ self.L.T
        x_analysis = self.x_prior + x_inc
        print("iteration stopped at step: ",iteration)
        return x_analysis.detach()


class CVT3dEnVarPerturbedObs(CVT3dEnVar):
    """CVT3dEnVar with per-member perturbed observations (EDA / stochastic-EnKF style).

    Each member i assimilates its own y_i = y + eps_i with eps_i ~ N(0, R), drawn once
    before the optimization. The converged analysis ensemble then has covariance
    (I-KH) B (I-KH)' + K R K' = (B^-1 + H'R^-1 H)^-1, the exact Bayesian posterior
    covariance, instead of the under-dispersed (I-KH) B (I-KH)' produced when every
    member assimilates the same y. Drop-in replacement for CVT3dEnVar: identical
    constructor and identical run() return values.
    """

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
        device=use_this
    ):
        super().__init__(H, y_obs, B, R, Xb, obs_idx,
                         learning_rate=learning_rate,
                         max_iterations=max_iterations,
                         convergence_threshold=convergence_threshold,
                         device=device)
        self.R = R.to(device)
        self.L_R = torch.linalg.cholesky(self.R)  # R = L_R L_R^T

    def run(self):
        sparse_index = torch.zeros(self.x_prior.size(dim = 1),dtype=torch.bool,device=self.device)
        sparse_index[self.obs_idx] = True
        # One fixed observation perturbation per member: eps_i ~ N(0, R), row form eps = z @ L_R^T.
        # Unobserved dims receive perturbation too; with a diagonal R
        # they carry zero weight through the masked R_inv.
        obs_pert = torch.randn(self.ensemble_size, self.state_dim,
                               device=self.device, dtype=self.L_R.dtype) @ self.L_R.T
        # Center the perturbations so the analysis-ensemble mean is unaffected by sampling noise.
        obs_pert = obs_pert - obs_pert.mean(dim=0, keepdim=True)
        # Cap the step by the CVT Hessian bound: lambda_max(I + L'H'R^-1 H L) <= 1 + lambda_max(B) * max(R^-1 on observed dims)
        R_inv_obs = torch.diagonal(self.R_inv)[self.obs_idx]
        curvature_bound = 1.0 + self.B_lam_max * R_inv_obs.max()
        self.effective_learning_rate = min(self.learning_rate, 0.9 / curvature_bound.item())
        print(f"Effective learning rate: {self.effective_learning_rate:.6e}")
        control_var = torch.zeros_like(self.x_prior, requires_grad=True)
        prev_cost = float('inf')
        idx_nan = False
        x_inc = control_var @ self.L.T  # x_inc = L v in row-batch form
        for iteration in range(self.max_iterations):
            if control_var.grad is not None:
                control_var.grad.zero_()
            # Classical control-variable transform: x_inc = B^(1/2) v
            x_inc = control_var @ self.L.T
            # Apply observation operator
            Hx_inc = self.H(x_inc - self.x_inc_init)
            # Perturbed innovation: d_i = (y + eps_i) - H(xb_i)
            obs_inc = (self.y_obs + obs_pert - self.H(self.x_prior))
            # Innovation
            innovation = Hx_inc - obs_inc
            # Cost function. self.R_inv = self.R_inv.unsqueeze(0).unsqueeze(1)
            R_inv = self.R_inv.clone()
            R_inv[:,~sparse_index] = 0

            cost_obs = 0.5 * torch.einsum("ij,jk,ik->i", innovation.float(), R_inv.float(), innovation.float())
            cost_bkg = 0.5 * torch.einsum("ij,ij->i", control_var, control_var)
            cost = cost_obs + cost_bkg

            grad_control_var = torch.autograd.grad(outputs=cost,
                                 inputs=control_var,
                                 grad_outputs=torch.ones_like(cost),
                                 create_graph=True)[0]

            with torch.no_grad():
                control_var -= self.effective_learning_rate * grad_control_var
            x_inc = control_var @ self.L.T
            cost_value = cost.mean().item()
            if (iteration < 1000) or(iteration%10000 == 0):
                print(f"Iteration {iteration+1}: Cost = {cost_value:.6f}")

            if cost_value>10000000000: #10000000
                print("Cost is NaN")
                idx_nan = True
                break

            if abs(prev_cost - cost_value) < self.convergence_threshold:
                print("Convergence reached.")
                break

            prev_cost = cost_value

        # Final analysis state
        x_inc = control_var @ self.L.T
        x_analysis = self.x_prior + x_inc
        print("iteration stopped at step: ",iteration)
        return x_analysis.detach()


class EnVar3DPerturbedObs(EnVar3D):
    """EnVar3D with per-member perturbed observations (EDA / stochastic-EnKF style).

    Each member i assimilates its own y_i = y + eps_i with eps_i ~ N(0, R), drawn once
    before the optimization. The converged analysis ensemble then has covariance
    (I-KH) B (I-KH)' + K R K' = (B^-1 + H'R^-1 H)^-1, the exact Bayesian posterior
    covariance, instead of the under-dispersed (I-KH) B (I-KH)' produced when every
    member assimilates the same y. Drop-in replacement for EnVar3D: identical
    constructor and identical run() return values.
    """

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
        device=use_this
    ):
        super().__init__(H, y_obs, B, R, Xb, obs_idx,
                         learning_rate=learning_rate,
                         max_iterations=max_iterations,
                         convergence_threshold=convergence_threshold,
                         device=device)
        self.R = R.to(device)
        self.L_R = torch.linalg.cholesky(self.R)  # R = L_R L_R^T

    def run(self):
        sparse_index = torch.zeros(self.x_prior.size(dim = 1),dtype=torch.bool,device=self.device)
        sparse_index[self.obs_idx] = True
        # One fixed observation perturbation per member: eps_i ~ N(0, R), row form eps = z @ L_R^T.
        # Unobserved dims receive perturbation too; with a diagonal R
        # they carry zero weight through the masked R_inv.
        obs_pert = torch.randn(self.ensemble_size, self.state_dim,
                               device=self.device, dtype=self.L_R.dtype) @ self.L_R.T
        # Center the perturbations so the analysis-ensemble mean is unaffected by sampling noise.
        obs_pert = obs_pert - obs_pert.mean(dim=0, keepdim=True)
        x_inc = torch.zeros_like(self.x_prior, requires_grad=True).to(self.device)
        prev_cost = float('inf')
        idx_nan = False
        for iteration in range(self.max_iterations):
            if x_inc.grad is not None:
                x_inc.grad.zero_()
            # Apply observation operator
            Hx_inc = self.H(x_inc - self.x_inc_init)
            # Perturbed innovation: d_i = (y + eps_i) - H(xb_i)
            obs_inc = (self.y_obs + obs_pert - self.H(self.x_prior))
            # Innovation
            innovation = Hx_inc - obs_inc
            # Cost function. self.R_inv = self.R_inv.unsqueeze(0).unsqueeze(1)
            R_inv = self.R_inv.clone()
            R_inv[:,~sparse_index] = 0

            cost_obs = 0.5 * torch.einsum("ij,jk,ik->i", innovation.float(), R_inv.float(), innovation.float())  # Efficient quadratic form
            # cost_bkg = 0.5 * torch.einsum("ij,jk,ik->i", x_inc, self.B_inv, x_inc)            # Efficient quadratic form
            # Solve B y = x_inc for all members at once (N right-hand sides); the batched
            # x_inc.unsqueeze(-1) form broadcasts L to (N, d, d) and runs out of memory for large d.
            rhs = x_inc.T.contiguous()                 # (d, N)
            y  = torch.cholesky_solve(rhs, self.L).T   # (N, d)
            cost_bkg = 0.5 * torch.einsum("ij,ij->i", x_inc, y)
            cost = cost_obs + cost_bkg

            grad_x_inc = torch.autograd.grad(outputs=cost,
                                 inputs=x_inc,
                                 grad_outputs=torch.ones_like(cost),  # Identity for element-wise gradient
                                 create_graph=True)[0]

            with torch.no_grad():
                x_inc -= self.learning_rate * grad_x_inc
            cost_value = cost.mean().item()
            if (iteration < 1000) or(iteration%10000 == 0):
                print(f"Iteration {iteration+1}: Cost = {cost_value:.6f}")

            if cost_value>10000000000: #10000000
                print("Cost is NaN")
                idx_nan = True
                break

            if abs(prev_cost - cost_value) < self.convergence_threshold:
                print("Convergence reached.")
                break

            prev_cost = cost_value

        # Final analysis state
        x_analysis = self.x_prior + x_inc
        print("iteration stopped at step: ",iteration)
        return x_analysis.detach()
