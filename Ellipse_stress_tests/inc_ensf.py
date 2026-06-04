import torch
import numpy as np
import time
import math


class EnSF:
    def __init__(self, n_dim, ensemble_size,eps_alpha,eps_beta, device,obs_sigma,euler_steps):
    ####################################################################
    # EnSF setup
    # define the diffusion process eps_alpha
    # ensemble size ensemble_size = 250
        self.n_dim = n_dim
        self.ensemble_size = ensemble_size
        self.eps_alpha = eps_alpha
        self.eps_beta = eps_beta
        self.device = device
        self.obs_sigma = obs_sigma
        self.euler_steps = euler_steps

    # computation setting
        #torch.set_default_dtype(torch.float16) # half precision
        #device = 'cuda' 'cpu'

# compact version
    def cond_alpha(self,t):
        # conditional information
        # alpha_t(0) = 1
        # alpha_t(1) = esp_alpha \approx 0
        return 1 - (1-self.eps_alpha)*t

    def cond_sigma_sq(self,t):
    # conditional sigma^2
    # sigma2_t(0) = 0
    # sigma2_t(1) = 1
    # sigma(t) = t
        return t*(1-self.eps_beta) + self.eps_beta

# drift function of forward SDE
    def f(self,t):
        # f=d_(log_alpha)/dt
        alpha_t = self.cond_alpha(t)
        f_t = -(1-self.eps_alpha) / alpha_t
        return f_t


    def g_sq(self,t):
        # g = d(sigma_t^2)/dt -2f sigma_t^2
        d_sigma_sq_dt = 1-self.eps_beta
        g2 = d_sigma_sq_dt - 2*self.f(t)*self.cond_sigma_sq(t)
        return g2

    def g(self,t):
        return np.sqrt(self.g_sq(t))


# generate sample with reverse SDE
    def reverse_SDE(self, x0, time_steps):
        # x_T: sample from standard Gaussian
        # x_0: target distribution to sample from
        ensemble_size = self.ensemble_size
        n_dim = self.n_dim
        device = self.device


        # Generate the time mesh
        dt = 1.0/time_steps

        # Initialization
        xt = torch.randn(ensemble_size,n_dim, device=device)
        path_all = torch.randn(ensemble_size,n_dim,time_steps, device=device)
        t = 1.0

        # define storage
        # forward Euler sampling
        for i in range(time_steps):
            # prior score evaluation
            alpha_t = self.cond_alpha(t)#alpha_fun(t)
            sigma2_t = self.cond_sigma_sq(t)#sigma2_fun(t)
            # Evaluate the diffusion term
            diffuse = self.g(t) #diffuse_fun(t)
            # Update
            xt += - dt*( self.f(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/sigma2_t) - diffuse**2 * 0 ) \
                    + np.sqrt(dt)*diffuse*torch.randn_like(xt)
            path_all[:,:,i]=xt.squeeze(-1)
      
            # update time
            t = t - dt

      
        return path_all

    def score(self, x0, time_steps):
        # x_T: sample from standard Gaussian
        # x_0: target distribution to sample from
        ensemble_size = self.ensemble_size
        n_dim = self.n_dim
        device = self.device


        # Generate the time mesh
        dt = 1.0/time_steps

        # Initialization
        score = torch.randn(ensemble_size,time_steps, device=device)
        xt = torch.randn(ensemble_size,n_dim, device=device)
        t = 1.0

        # define storage
        # forward Euler sampling
        for i in range(time_steps):
            # prior score evaluation
            alpha_t = self.cond_alpha(t)#alpha_fun(t)
            sigma2_t = self.cond_sigma_sq(t)#sigma2_fun(t)
            # Evaluate the diffusion term
            diffuse = self.g(t) #diffuse_fun(t)
            # Update
            score[:,i] =   (xt - alpha_t*x0).squeeze(-1)/sigma2_t
            # update time
            t = t - dt
            # if i == 950:
            #     break

      
        return score

    def state_update_twostep_step2(self,x_input,scale_input,enscov):
        torch.set_default_dtype(torch.float32)
        euler_steps = self.euler_steps
        
        # filtering ensemble
        x_state = torch.tensor(x_input,device=self.device)
        enscov = torch.tensor(enscov,device=self.device)
        #scale state
        x_state = x_state / scale_input
        torch.cuda.empty_cache()

        # generate posterior sample
        x_state = self.reverse_SDE_twostep_step2(x0=x_state,time_steps=euler_steps,ens_cov= enscov) 
        x_state = x_state * scale_input
        return  x_state 

    def reverse_SDE_twostep_step2(self, x0, time_steps, looptime, ens_cov):
        # x_T: sample from standard Gaussian
        # x_0: target distribution to sample from
        ensemble_size = self.ensemble_size
        n_dim = self.n_dim
        device = self.device
        xt = torch.randn(ensemble_size,n_dim, device=device)
        #drift_fun=f, diffuse_fun=g, alpha_fun=cond_alpha, sigma2_fun=cond_sigma_sq,  score_likelihood=None, 
        ens_sigma = ens_cov.to(xt.dtype)
        path_all = torch.randn(ensemble_size,n_dim,time_steps, device=device)
        xobs = x0[:,0].clone()
        
        for loop in range(looptime):
            #increment
            # Generate the time mesh
            dt = 1.0/time_steps
            # Initialization
            xt = torch.randn(ensemble_size,n_dim, device=device)
            t = 1.0
            # reverse SDE sampling process
            for i in range(time_steps):
                # prior score evaluation
                alpha_t = self.cond_alpha(t)#alpha_fun(t)
                sigma2_t = self.cond_sigma_sq(t)#sigma2_fun(t)
                # Evaluate the diffusion term
                diffuse = self.g(t) #diffuse_fun(t)
                # Update.  alpha_t*mu
                xt += - dt*( self.f(t)*xt + diffuse**2 * (torch.linalg.solve((alpha_t**2 *ens_sigma+sigma2_t*torch.eye(ens_sigma.size(0),device=self.device)), xt - alpha_t*x0, left=False)) ) \
                    + np.sqrt(dt)*diffuse*torch.randn_like(xt)
                path_all[:,:,i]=xt.squeeze(-1)
                # update time
                t = t - dt
            x0 = xt.clone()
            x0[:,0] = xobs.clone()
            x0[:,1] = (x0[:,1]-x0[:,1].mean())/x0[:,1].std()*2 + x0[:,1].mean()
        return path_all

    def forward_SDE(self, x0, time_steps):
            # x_T: sample from standard Gaussian
            # x_0: target distribution to sample from
            ensemble_size = self.ensemble_size
            n_dim = self.n_dim
            device = self.device


            # Generate the time mesh
            dt = 1.0/time_steps

            # Initialization
            path_all = torch.randn(ensemble_size,n_dim,time_steps, device=device)
            t = 0.0

            # define storage
            # forward Euler sampling
            for i in range(time_steps):
                # prior score evaluation
                alpha_t = self.cond_alpha(t)#alpha_fun(t)
                sigma2_t = self.cond_sigma_sq(t)#sigma2_fun(t)
                # Evaluate the diffusion term
                diffuse = self.g(t) #diffuse_fun(t)
                # Update
                x0 += dt* self.f(t)*x0 + np.sqrt(dt)*diffuse*torch.randn_like(x0)
                path_all[:,:,i]=x0.squeeze(-1)
        
                # update time
                t = t + dt

        
            return path_all
    
    def socre_incremental(self, xt, x0,x_prior, t,obs,obs_sigma,sparse_idx):
        # obs: (d)
        # xt: (ensemble, d)
        ensemble_size = self.ensemble_size
        n_dim = self.n_dim
        device = self.device        
        score_x = torch.zeros(ensemble_size,n_dim, device=device)
        obs_inc = obs - x_prior
        Hof_inc = xt - x0     
        score_x = (-(  Hof_inc - obs_inc)/(obs_sigma)**2 ).type_as(score_x)
        score_x[:,~sparse_idx] = 0 
        tau = self.g_tau(t)
        #print(score_x.sum(dim = 0))
        return tau*score_x, -score_x*(Hof_inc - obs_inc)

    def RSDE_incremental(self, obs,x0,x_prior, time_steps,ens_cov,sparse_idx,cov_scale,tau_scale):
        sparse_index = torch.zeros(x_prior.size(dim = 1),dtype=torch.bool,device=self.device)
        sparse_index[sparse_idx] = True
        # x_T: sample from standard Gaussian
        # x_0: target distribution to sample from
        ensemble_size = self.ensemble_size
        n_dim = self.n_dim
        device = self.device
        #drift_fun=f, diffuse_fun=g, alpha_fun=cond_alpha, sigma2_fun=cond_sigma_sq,  score_likelihood=None, 
        # reverse SDE sampling process
        # Generate the time mesh
        dt = 1.0/time_steps
        # Initialization
        xt = torch.randn(ensemble_size,n_dim, device=device)
        t = 1.0
        path_all = torch.randn(ensemble_size,n_dim,time_steps, device=device)
        total_score_save = torch.zeros(time_steps,2, device=device)   
        cond_save = torch.zeros(time_steps, device=device)    
        # forward Euler sampling
        for i in range(time_steps):
            # prior score evaluation
            alpha_t = self.cond_alpha(t)#alpha_fun(t)
            sigma2_t = self.cond_sigma_sq(t)#sigma2_fun(t)
            # Evaluate the diffusion term
            diffuse = self.g(t) #diffuse_fun(t)
            # Update. 
            tau_score,like_score = self.socre_incremental(xt, x0,x_prior, t,obs,self.obs_sigma,sparse_index)
            temp_cov = alpha_t**2 *ens_cov + sigma2_t * torch.eye(ens_cov.size(0),device=self.device)
            prior_score = ( torch.linalg.solve(temp_cov, xt - alpha_t*x0, left=False)) 
            total_score = prior_score - tau_score * tau_scale
            # v = xt - x0
            # total_score_save[i,1] = like_score.sum(dim=1).mean()
            # L = torch.linalg.cholesky(ens_cov*cov_scale)         
            # y = torch.cholesky_solve(v.T, L).T 
            # total_score_save[i,0] = (v * y).sum(dim=1).mean()
            xt += - dt*( self.f(t)*xt + diffuse**2 *  total_score) \
                                    + np.sqrt(dt)*diffuse*torch.randn_like(xt)
            path_all[:,:,i]=xt.squeeze(-1) + x_prior
            cond_save[i] = torch.linalg.cond(alpha_t**2 *ens_cov*cov_scale + sigma2_t * torch.eye(ens_cov.size(0),device=self.device))
            # update time
            t = t - dt
        return path_all, total_score_save,cond_save
 
    def RSDE_inc(self, obs,x0,x_prior, time_steps,ens_cov,sparse_idx,cov_scale,tau_scale):
        sparse_index = torch.zeros(x_prior.size(dim = 1),dtype=torch.bool,device=self.device)
        sparse_index[sparse_idx] = True
        # x_T: sample from standard Gaussian
        # x_0: target distribution to sample from
        ensemble_size = self.ensemble_size
        n_dim = self.n_dim
        device = self.device
        #drift_fun=f, diffuse_fun=g, alpha_fun=cond_alpha, sigma2_fun=cond_sigma_sq,  score_likelihood=None, 
        # reverse SDE sampling process
        # Generate the time mesh
        dt = 1.0/time_steps
        # Initialization
        xt = torch.randn(ensemble_size,n_dim, device=device)
        t = 1.0
        path_all = torch.randn(ensemble_size,n_dim,time_steps, device=device)
        total_score_save = torch.zeros(time_steps,2, device=device)   
        cond_save = torch.zeros(time_steps, device=device)    
        # forward Euler sampling
        for i in range(time_steps):
            # prior score evaluation
            alpha_t = self.cond_alpha(t)#alpha_fun(t)
            sigma2_t = self.cond_sigma_sq(t)#sigma2_fun(t)
            # Evaluate the diffusion term
            diffuse = self.g(t) #diffuse_fun(t)
            # Update. 
            tau_score,like_score = self.socre_incremental(xt, x0,x_prior, t,obs,self.obs_sigma,sparse_index)
            temp_cov = alpha_t**2 *ens_cov + sigma2_t * torch.eye(ens_cov.size(0),device=self.device)
            prior_score = ( torch.linalg.solve(temp_cov, xt - alpha_t*x0, left=False)) 
            total_score = prior_score - tau_score * tau_scale
            # v = xt - x0
            # total_score_save[i,1] = like_score.sum(dim=1).mean()
            # L = torch.linalg.cholesky(ens_cov*cov_scale)         
            # y = torch.cholesky_solve(v.T, L).T 
            # total_score_save[i,0] = (v * y).sum(dim=1).mean()
            xt += - dt*( self.f(t)*xt + diffuse**2 *  total_score) \
                                    + np.sqrt(dt)*diffuse*torch.randn_like(xt)
            path_all[:,:,i]=xt.squeeze(-1) + x_prior
            cond_save[i] = torch.linalg.cond(alpha_t**2 *ens_cov*cov_scale + sigma2_t * torch.eye(ens_cov.size(0),device=self.device))
            # update time
            t = t - dt
        path_all[:,sparse_index,i] =  (path_all[:,sparse_index,i] - path_all[:,sparse_index,i].mean(dim=0))*math.sqrt(tau_scale) + path_all[:,sparse_index,i].mean(dim=0)
        return path_all, total_score_save,cond_save
 
                
    # damping function(tau(0) = 1;  tau(1) = 0;)
    def g_tau(self, t):
        return 1-t
