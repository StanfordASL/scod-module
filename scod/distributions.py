from numpy.core.fromnumeric import var
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
we implement left multiplication by L^T using the function

apply_sqrt_F

we also implement:

marginalize(var): which returns the distribution if the parameters used to construct the original 
    distribution were normally distributed with diagonal variance given by var

merge_batch(): returns the distribution approximating the mixture distribution constructed by 
    summing across the first batch dimension. useful if you construct this distribution with a batch
    of parameters corresponding to MC samples, and you want a single distribution approximation of the mixture.

metric(label): which returns a more human friendly measure of error between the distribution and the label
    for Normal distributions, this is the MSE, while for discrete distributions, this yields the 0-1 error

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

    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with mean and a diagonal variance as given

        inputs:
            diag_var: variance of parameter (1,)
        """
        if self.use_logits:
            kappa = 1. / torch.sqrt(1. + np.pi / 8 * diag_var)
            p = torch.sigmoid(kappa * self.logits)
        else:
            p = self.probs # gaussian posterior in probability space is not useful
        return Bernoulli(probs=p)

    def merge_batch(self):
        p_mean = self.probs.mean(dim=0)
        return Bernoulli(probs=p_mean)

    def metric(self, y):
        """
        classification error (1- accuracy)
        """
        return ((self.probs >= 0.5) != y).float()

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

        permuted_M = M.reshape(vec.shape[:-1] + (n,)) # shape = (..., 1, j, i, 1, n)
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

    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with a diagonal variance as given

        inputs:
            diag_var: variance of parameter (d,)
        """
        stdev = torch.sqrt(self.variance + diag_var)
        return Normal(loc=self.mean, scale=stdev)

    def merge_batch(self):
        diag_var = torch.mean(self.mean**2, dim=0) - self.mean.mean(dim=0)**2 + self.variance.mean(dim=0)
        return Normal(loc=self.mean.mean(dim=0), scale=torch.sqrt(diag_var))

    def metric(self, y):
        return torch.mean( torch.sum( (y - self.mean)**2, dim=-1) )

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

    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with a diagonal variance as given

        inputs:
            diag_var: variance of parameter (d,)
        """
        if self.use_logits:
            # @TODO: allow selecting this via an argument
            # probit approx
            kappa = 1. / torch.sqrt(1. + np.pi / 8 * diag_var)
            scaled_logits = kappa*self.logits
            dist = Categorical(logits=scaled_logits)

            # laplace bridge
            # d = diag_var.shape[-1]
            # sum_exp = torch.sum(torch.exp(-self.logits), dim=-1, keepdim=True)
            # alpha = 1. / diag_var * (1 - 2./d + torch.exp(self.logits)/(d**2) * sum_exp)
            # dist = distributions.Dirichlet(alpha)
            # return distributions.Categorical(probs=torch.nan_to_num(dist.mean, nan=1.0))
        else:
            p = self.probs # gaussian posterior in probability space is not useful
            return Categorical(probs=p)
        return dist

    def merge_batch(self):
        p_mean = self.probs.mean(dim=0)
        return Categorical(probs=p_mean)

    def metric(self, y):
        """
        classification error (1- accuracy)
        """
        return (torch.argmax(self.probs, dim=-1) != y).float()