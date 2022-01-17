import torch
import numpy as np

def gaussianKLDiv(Sig1, Sig2, err=None):
    """
    computes D_KL( N(mu1, Sig1) || N(mu2, Sig2) ) where err = mu1 - mu2 (or mu2 - mu1)
    """
    Sig2_inv = torch.linalg.inv(Sig2 + 1e-5*torch.eye(Sig2.shape[0]))
    Sig_ratio = Sig2_inv @ Sig1

    loss = torch.trace(Sig_ratio) # tr Sig2^{-1} Sig1
    loss -= Sig2.shape[0] # dimension of multivariate Guassian
    loss -= torch.slogdet(Sig_ratio).logabsdet # log det ( Sig2^{-1} Sig1 )

    if err is not None:
        loss += torch.dot(err, Sig2_inv @ err) # (mu1 - mu2)^T Sig2^{-1} (mu1 - mu2)

    return loss

def gaussianWassersteinDist(Sig1, Sig2, err=None):
    """
    Computes W_2( N(mu1, Sig1),  N(mu2, Sig2) ) where err = mu1 - mu2
    """
    Sig2eigdecomp = torch.linalg.eigh(Sig2)

    Sig2half =  Sig2eigdecomp.eigenvectors @ ( torch.sqrt( torch.clamp_min(torch.abs(Sig2eigdecomp.eigenvalues), min=0) )[None, :] * Sig2eigdecomp.eigenvectors ).t() 
    # print((Sig2half @ Sig2half - Sig2).abs().sum())
    # Sig2half = torch.linalg.cholesky(Sig2 + 1e-5*torch.eye(Sig2.shape[0]))
    loss = torch.trace(Sig1)
    loss += torch.trace(Sig2)
    eigs = torch.clamp_min(torch.abs( torch.linalg.eigvals(Sig2half @ (Sig1) @ Sig2half) ) , min = 0)
    loss -= 2*torch.sum(torch.sqrt(eigs))
    if err is not None:
        loss += (err**2).sum()

    return loss