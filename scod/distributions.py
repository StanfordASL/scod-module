import torch
from torch.distributions.transforms import AbsTransform
from torch.functional import broadcast_shapes
import torch.nn as nn
import numpy as np
from torch import distributions
from torch.distributions.multivariate_normal import _batch_mahalanobis

"""
this file implements different output distributional families, with their loss functions and Fisher matrices

specifically, for each family, if F(theta) = LL^T
we implement left multiplication by L^T by the function
apply_sqrt_F
"""

class Bernoulli(distributions.Bernoulli):
    def __init__(self, probs=None, logits=None, validate_args=None):
        self.use_logits = False
        if probs is None:
            self.use_logits = True
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)
    
    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T, 
        return L^T vec, blocking gradients through L

        if self._param is probs, then 
            F = 1/(p(1-p))
        if self._param is logits, then
            F = p(1-p), where p = sigmoid(logit)
        """
        p = self.probs.detach()
        L = torch.sqrt( p*(1-p) ) + 1e-10 # for stability
        
        if self.use_logits:
            return L * vec
        else:
            return vec / L

class MultivariateNormal(distributions.multivariate_normal.MultivariateNormal):
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        super().__init__(loc, covariance_matrix=covariance_matrix, precision_matrix=precision_matrix, scale_tril=scale_tril, validate_args=validate_args)

    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T, 
        return L^T vec, blocking gradients through L

        Here, we assume only loc varies, and do not consider cov as a backpropable parameter
        
        F = Sigma^{-1}

        Here, we use code from torch.distributions.multivariate_normal: _batched_mahalanobis()
        but skip the square and sum operation. All this reshaping handles different batch dimensions
        for vec and self.unbroadcasted_scale_tril, and is most likely overkill for this application
        """
        n = vec.size(-1)
        batch_shape = vec.shape[:-1]

        bL = self._unbroadcasted_scale_tril
        
        # Assume that bL.shape = (i, 1, n, n)
        # vec.shape = (..., i, j, n)
        # resehape vec to be (..., 1, j, i, 1, n) to apply batched tri.solve
        bx_batch_dims = len(batch_shape)
        bL_batch_dims = bL.dim() - 2
        outer_batch_dims = bx_batch_dims - bL_batch_dims
        old_batch_dims = outer_batch_dims + bL_batch_dims
        new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
        # Reshape vec with the shape (..., 1, i, j, 1, n)
        vec_new_shape = vec.shape[:outer_batch_dims]
        for (sL, sx) in zip(bL.shape[:-2], vec.shape[outer_batch_dims:-1]):
            vec_new_shape += (sx // sL, sL)
        vec_new_shape += (n,)
        vec = vec.reshape(vec_new_shape)
        # Permute vec to make it have shape (..., 1, j, i, 1, n)
        permute_dims = (list(range(outer_batch_dims)) +
                        list(range(outer_batch_dims, new_batch_dims, 2)) +
                        list(range(outer_batch_dims + 1, new_batch_dims, 2)) +
                        [new_batch_dims])
        vec = vec.permute(permute_dims)

        flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
        flat_x = vec.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
        flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
        M_swap = torch.triangular_solve(flat_x_swap, flat_L, upper=False)[0] # shape = b x n x c
        M = M_swap.transpose(-2,-1) # (b x c x n)
        M = M.transpose(-2,-3) # (c x b x n)

        permuted_M = M.reshape(vec.shape[:-1] + [n]) # shape = (..., 1, j, i, 1, n)
        permute_inv_dims = list(range(outer_batch_dims))
        for i in range(bL_batch_dims):
            permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
        reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1, n)

        return reshaped_M.reshape(vec.shape)

class Normal(distributions.normal.Normal):
    def __init__(self, loc, scale, validate_args=None):
        super().__init__(loc, scale, validate_args=validate_args)

    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T, 
        return L^T vec, blocking gradients through L

        Here, we assume only loc varies, and do not consider cov as a backpropable parameter
        
        F = Sigma^{-1}

        """
        return vec / self.stddev.detach()

class Categorical(distributions.categorical.Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None):
        self.use_logits = False
        if probs is None:
            self.use_logits = True
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T, 
        return L^T vec, blocking gradients through L

        if self._param is probs, then 
            F = (diag(p^{-1}))
        if self._param is logits, then
            F = (diag(p) - pp^T) = LL^T
        """
        p = self.probs.detach()
        if self.use_logits:
            vec_bar = torch.sum(p*vec, dim=-1, keepdim=True)
            return torch.sqrt(p)*(vec - vec_bar)
        else:
            return vec / (torch.sqrt(p) + 1e-8)

# class DistFam(nn.Module):
#     def loss(self, thetas, targets):
#         """
#         thetas (..., thetadim)
#         targets (..., ydim)
#         """
#         return self._loss(thetas, targets)
    
#     def metric(self, outputs, targets):
#         """
#         return a user facing metric based on
#         outputs = self.output(theta)
        
#         returns {'name':value}
#         """
#         return {}
        
#     def apply_sqrt_F(self, theta):
#         """
#         F(theta) = LL^T

#         returns y = L^T x
#         """
#         raise NotImplementedError
        
#     def output(self, theta):
#         """
#         turns theta into user-facing output
#         """
#         return theta
    
#     def uncertainty(self, outputs):
#         """
#         outputs: [..., outputdim]
#         returns unc [...], uncertainty score per output
#         """
#         return 1. + 0.*outputs[...,0]
        
#     def merge_ensemble(self, thetas):
#         return NotImplementedError

# class GaussianFixedDiagVar(DistFam):
#     """
#     P(theta) = N(theta, diag(sigma))
    
#     Here, F(theta) = diag(sigma)^-1
#     normalized F = diag(sigma)^(-1) / sum(sigma^-1)
#     """
#     def __init__(self, sigma_diag=np.array([1.]), min_val=-5, max_val=5):
#         super().__init__()
#         self.sigma_diag = nn.Parameter(torch.from_numpy(sigma_diag).float(), requires_grad=False)
#         self.min_val=min_val
#         self.max_val=max_val

#     def dist(self, thetas):
#         C = torch.diag_embed(torch.ones_like(thetas)*self.sigma_diag)
#         return torch.distributions.MultivariateNormal(loc = thetas, covariance_matrix=C )
        
#     def loss(self, thetas, targets):
#         err = targets - thetas
#         return 0.5 * torch.sum( err**2 / self.sigma_diag, dim=-1 ) +0.5*err.shape[-1]*np.log(2*np.pi) + 0.5*torch.sum(torch.log(self.sigma_diag))

#     @torch.no_grad()
#     def metric(self, outputs, targets):
#         err = outputs - targets
#         return 'Mahalanobis Error', torch.sum(err**2 / self.sigma_diag, dim=-1)
        
#     def apply_sqrt_F(self, theta, exact=True):
#         return theta / torch.sqrt( self.sigma_diag )
    
#     def apply_sqrt_G(self, theta, y):
#         """
#         returns S^T theta, where 
#         $ \nabla^2_\theta log p (y \mid \theta) := S S^T
#         """
#         return theta / torch.sqrt( self.sigma_diag )

#     def uncertainty(self, outputs):
#         return torch.sum(self.sigma_diag) + 0*outputs[...,0]
    
#     def merge_ensemble(self, thetas):
#         mu = torch.mean(thetas, dim=0)
#         diag_var = torch.mean(thetas**2, dim=0) - mu**2
#         unc = torch.sqrt(torch.sum(diag_var, dim=-1))
#         return mu, unc
    
# class Bernoulli(DistFam):
#     """
#     theta \in \R^1
#     P(theta) = Bern( theta )
    
#     Here, F(theta) = 1/(p(1-p))
#     normalized F(theta) = 1
#     """
#     def __init__(self):
#         super().__init__()
#         self._loss = nn.BCELoss()
    
#     def dist(self, thetas):
#         return torch.distributions.Bernoulli(probs = thetas)
    
    
#     @torch.no_grad()
#     def metric(self, outputs, targets):
#         accuracy = (targets*(outputs > 0.5) + (1-targets)*(outputs < 0.5))[...,0]
#         return 'Accuracy', accuracy
        
#     def apply_sqrt_F(self, theta):
#         t = theta.detach()
#         L = torch.sqrt( t*(1-t) ) + 1e-10 # for stability
#         return theta / L
    
#     def merge_ensemble(self, thetas):
#         mu = torch.mean(thetas, dim=0)
#         unc = (- mu*torch.log(mu) - (1-mu)*torch.log(1-mu))[...,0]
#         return mu, unc
    
# class BernoulliLogit(DistFam):
#     """
#     theta \in \R^1
#     P(theta) = Bern( 1/(1 + exp(-theta)) )
    
#     Here, F(theta) = p(1-p)
#     Normalize F(theta) = 1
#     """
#     def __init__(self):
#         super().__init__()
#         self._loss = nn.BCEWithLogitsLoss(reduction='none')
        
#     def dist(self, thetas):
#         return torch.distributions.Bernoulli(logits = thetas)
        
#     @torch.no_grad()
#     def metric(self, outputs, targets):
#         accuracy = (targets*(outputs > 0.5) + (1-targets)*(outputs < 0.5))[...,0]
#         return 'accuracy', accuracy
# #         return '-log(p(y))', -torch.log(outputs[...,0])
        
#     def apply_sqrt_F(self, theta, exact=True):
#         t = theta.detach()
        
#         p = torch.sigmoid(t)
#         L = torch.sqrt(p*(1-p))
#         return L*theta
    
#     def output(self, theta):
#         return torch.sigmoid(theta)
    
#     def uncertainty(self, outputs):
#         return (- outputs*torch.log(outputs) - (1-outputs)*torch.log(1-outputs))[...,0]
    
#     def merge_ensemble(self, thetas):
#         ps = torch.sigmoid(thetas)
#         mu = torch.mean(ps, dim=0)
#         unc = (- mu*torch.log(mu) - (1-mu)*torch.log(1-mu))[...,0]
#         if torch.isnan(unc):
#             unc = torch.zeros_like(mu)[...,0]
#         return mu, unc
    
# class Categorical(DistFam):
#     """
#     theta \in \R^k, 1^T theta = 1, theta > 0
#     P(theta) = Categorical( p = theta )
    
#     Here, F(theta) = (diag(p^{-1})) = LL^T
#     L = diag(p^{-1/2})
#     normalized F = diag(p^{-1}) / sum(p^{-1})
#     """
#     def __init__(self):
#         super().__init__()
#         self.nll_loss = nn.NLLLoss(reduction='none')
    
#     def dist(self, thetas):
#         return torch.distributions.Categorical(probs = thetas)
        
#     def loss(self, thetas, targets):
#         return self.nll_loss(torch.log(thetas), targets)
    
#     @torch.no_grad()
#     def metric(self, outputs, targets):
# #         pred_label = torch.argmax(outputs, dim=-1)
# #         accuracy = 1.*(targets == pred_label)
# #         return 'Accuracy', accuracy
#         prob_y = torch.gather(outputs,-1,targets[:,None])[...,0]
#         return '-log(p(y))', -torch.log(prob_y)
        
#     def apply_sqrt_F(self, theta):
#         t = theta.detach()
#         L_diag = torch.sqrt(t) + 1e-7 # for stability
#         return theta / L_diag 

#     def uncertainty(self, outputs):
#         return -torch.sum(outputs*torch.log(outputs), dim=-1)
    
#     def merge_ensemble(self, thetas):
#         mu = torch.mean(thetas, dim=0)
#         unc = -torch.sum(mu*torch.log(mu), dim=-1)
#         return mu, unc
    
# class CategoricalLogit(DistFam):
#     """
#     theta \in \R^k
#     P(theta) = Categorical( p = SoftMax(theta) )
    
#     Here, F(theta) = (diag(p) - pp^T) = LL^T
#     L = (I - p1^T) diag(p^{1/2})
#     L^T = diag(p^{1/2}) (I - 1p^T)
#     """
#     def __init__(self):
#         super().__init__()
#         self._loss = nn.CrossEntropyLoss(reduction='none')
        
#     def dist(self, thetas):
#         return torch.distributions.Categorical(logits = thetas)
    
#     @torch.no_grad()
#     def metric(self, outputs, targets):
# #         pred_label = torch.argmax(outputs, dim=-1)
# #         accuracy = 1.*(targets == pred_label)
# #         return 'Accuracy', accuracy
#         prob_y = torch.gather(outputs,-1,targets[:,None])[...,0]
#         return '-log(p(y))', -torch.log(prob_y)
        
#     def apply_sqrt_F(self, theta, exact=True):
#         t = theta.detach()
        
#         # exact computation
#         if exact:
#             p = torch.softmax(t, dim=-1)
#             theta_bar = torch.sum(p*theta, dim=-1)[...,None]
#             result = torch.sqrt(p)*(theta - theta_bar)
        
#         # or, just sample a couple outputs from p(y) and then compute gradients
#         else:
#             logp = torch.log_softmax(theta, dim=-1)
#             vals, idx = logp.topk(min(5,logp.shape[-1]))
#             result = -torch.exp(vals).detach()*torch.gather(logp,-1,idx)
#         return result

#     def apply_sqrt_G(self, theta, y):
#         """
#         returns S^T theta, where 
#         $ \nabla^2_\theta log p (y \mid \theta) := S S^T
#         """
        
#         return theta / torch.sqrt( self.sigma_diag )
    
#     def output(self, theta):
#         return torch.softmax(theta, dim=-1)
    
#     def uncertainty(self, outputs):
#         return -torch.sum(outputs*torch.log(outputs), dim=-1)
    
#     def merge_ensemble(self, thetas):
#         ps = torch.softmax(thetas, dim=-1)
#         mu = torch.mean(ps, dim=0)
#         unc = -torch.sum(mu*torch.log(mu), dim=-1)
#         return mu, unc