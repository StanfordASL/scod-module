from scod import distributions
import torch
from torch import nn
from copy import deepcopy
from torch.cuda.amp.autocast_mode import autocast

from tqdm import tqdm
from torch.autograd import grad
from .sketching.sketched_pca import alg_registry

import numpy as np

class Projector(nn.Module):
    def __init__(self, N, r, device = torch.device('cpu')):
        super().__init__()
        self.N = N
        self.r = r
        
        self.device = device
        
        self.eigs = nn.Parameter(torch.zeros(self.r, device=self.device), requires_grad=False)
        self.basis = nn.Parameter(torch.zeros(self.N, self.r, device=self.device), requires_grad=False)
        
    @torch.no_grad()
    def process_basis(self, eigs, basis):
        self.eigs.data = eigs.to(self.device)
        self.basis.data = basis.to(self.device)

    def ortho_proj(self, L, n_eigs):
        """
        we have U = basis,
        computes ||(I-UU^T)L||^2_F
        """
        basis = self.basis[:,-n_eigs:]
        proj_L = basis.t() @ L 
        proj_L = basis @ proj_L
        return torch.norm(L - proj_L) 
    
    def posterior_pred(self, L, n_eigs, eps):
        """
        we have U = basis,
        computes ||(I-UU^T)L||^2_F
        """
        basis = self.basis[:,-n_eigs:]
        eigs = torch.clamp( self.eigs[-n_eigs:], min=0.)

        scaling = torch.sqrt( eigs / ( eigs + 1./(eps) ) )
        proj_L = scaling[:,None] * (basis.t() @ L)

        return torch.sqrt( torch.sum(L**2) - torch.sum(proj_L**2) )
    
    @torch.no_grad()
    def compute_distance(self, L, proj_type, n_eigs=None, eps=1.):
        if n_eigs is None:
            n_eigs = self.r
            
        L.to(self.device)
        if proj_type == 'ortho':
            return self.ortho_proj(L, n_eigs)
        elif proj_type == 'posterior_pred':
            return self.posterior_pred(L, n_eigs, eps)
        else:
            raise ValueError(proj_type +" is not an understood projection type.")

    def compute_diag_var(self, J, P=None, n_eigs=None, eps=1):
        """
        given J, (d, N)
        
        if P is not given, assumes P is the identity matrix

        returns diagonal variance of P J Sig J^T P^T, where
            Sig = ( 1/eps I + M U D U^T )^{-1}
                = eps I - eps U (1/(Meps) D^{-1} + I)^{-1} U^T
        """
        basis = self.basis[:,-n_eigs:]
        eigs = torch.clamp( self.eigs[-n_eigs:], min=0.)
        

        JJT = J @ J.T # torch.sum(J**2, dim=-1)

        scaling = eigs / (eigs + 1./ (eps))
        neg_term = torch.sqrt(scaling[None,:])*(J @ basis)
        neg_term = neg_term @ neg_term.T
        
        if P is not None:
            chol_sig = torch.linalg.cholesky(JJT - neg_term)
            sig = torch.sum( (P @ chol_sig)**2, dim=-1)  #( P[:,None,:] @ ( (JJT - neg_term) @ P.T) )[:,0]
            return eps*sig
        
        else: 
            return torch.diagonal(eps*(JJT - neg_term)) # shape (d,d)

base_config = {
    'num_samples': None, # sketch size T (T)
    'num_eigs': 10, # low rank estimate to recover (k)
    'n_fisher_samples': None, # None -> use full Fisher; k -> project using random projection; 0 -> use y values in dataset
    'sketch_type': 'gaussian', # sketch type 
    'unc_type': 'marginalized' # "marginalized" returns p(y|x) = \int p(y|z)p(z|x) dz, "KL" returns E[ KL(p( y|z) || p(y|zbar)) ]
}


class SCOD(nn.Module):
    """
    Wraps a trained model with functionality for adding epistemic uncertainty estimation.
    """
    def __init__(self, model, dist_constructor, args={}, parameters=None):
        """
        model: base DNN to equip with an uncertainty metric
        dist_fam: distributions.DistFam object representing how to interpret output of model
        args: configuration variables - defaults are in base_config
        """
        super().__init__()
        
        self.config = deepcopy(base_config)
        self.config.update(args)
        
        self.model = model
        self.dist_constructor = dist_constructor

        # extract device from model
        self.device = next(model.parameters()).device
        
        # extract parameters to consider in sketch - keep all that yield valid gradients
        if parameters is None:
            self.trainable_params = list(filter(lambda x: x.requires_grad, self.model.parameters()))
        else:
            self.trainable_params = list(parameters)
        self.n_params = int(sum(p.numel() for p in self.trainable_params))
                
        self.num_samples = self.config['num_samples']
        self.num_eigs = self.config['num_eigs']
        
        if self.num_samples is None:
            self.num_samples = 6*self.num_eigs + 4
            
        self.configured = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        
        self.sketch_class = alg_registry[self.config['sketch_type']] 
        print(self.sketch_class)
        
        self.projector = Projector(
            N=self.n_params,
            r=2*max(self.num_eigs + 2, (self.num_samples-1)//3),
            device=self.device
        )

        self.log_eps = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=True) # prior variance
        self.scaling_factor = nn.Parameter(torch.ones(1, device=self.device), requires_grad=False) # final scaling factor to rescale uncertainty output
        self.hyperparameters = [self.log_eps]

        self.output_dim = None

    @property
    def eps(self):
        return torch.exp(self.log_eps)
        
    def process_dataset(self, dataset):
        """
        summarizes information about training data by logging gradient directions
        seen during training, and then using gram schmidt of these to form
        an orthonormal basis. directions not seen during training are 
        taken to be irrelevant to data, and used for detecting generalization
        
        dataset - torch dataset of (input, target) pairs
        """
        # loop through data, one sample at a time
        print("computing basis") 
            
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=1, 
                                                 shuffle=True,
                                                 num_workers=0,
                                                 pin_memory=True)
        
        sketch = self.sketch_class(N=self.n_params, 
                                   r=self.num_eigs,
                                   T=self.num_samples,
                                   device=self.device)


        n_data = len(dataloader)
        for i, (inputs,labels) in tqdm(enumerate(dataloader), total=n_data):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast():
                thetas = self.model(inputs) # get params of output dist
            
                T = self.config["n_fisher_samples"]
                if T is not None and T == 0:
                    # hack for FCN training on VOC (label = 255 means ignore the pixel)
                    thetas_flat = thetas.reshape(1,-1,thetas.shape[-1])
                    labels_flat = labels.reshape(1,-1)
                    valid_label_idx = torch.nonzero(torch.logical_and(labels  >= 0, labels < thetas.shape[-1]), as_tuple=True)[1]
                    valid_thetas = thetas_flat[:,valid_label_idx,:]
                    valid_labels = labels_flat[:,valid_label_idx]
                    thetas = valid_thetas
                    labels = valid_labels

                    dist = self.dist_constructor(thetas)
                    Lt_th = -dist.log_prob(labels)[None] # - log p(y | x)
                else:
                    dist = self.dist_constructor(thetas)
                    Lt_th = dist.apply_sqrt_F(thetas).mean(dim=0) # L^\T theta
                
                # flatten
                Lt_th = Lt_th.view(-1)
                if T is not None and 0 < T < Lt_th.shape[-1]:
                    # apply random proj
                    P = torch.randn(T, Lt_th.shape[-1], device=self.device) / np.sqrt(T)
                    Lt_th = Lt_th @ P.T

            
            Lt_J = self._get_weight_jacobian(Lt_th) # L^\T J, J = dth / dw
            sketch.low_rank_update(Lt_J.t()) # add 1/M J^T L L^T J to the sketch
        
        del Lt_J

        self.output_dim = thetas.shape[-1]
        
        eigs, eigvs = sketch.eigs()
        del sketch

        self.projector.process_basis(eigs, eigvs)
            
        self.configured.data = torch.ones(1, dtype=torch.bool)

    
    def _get_weight_jacobian(self, vec):
        """
        returns d x nparam matrix, with each row being d(vec[i])/d(weights)
        """
        assert len(vec.shape) == 1
        grad_vecs = []
        for j in range(vec.shape[0]):
            grads = grad(vec[j], self.trainable_params, retain_graph=True, only_inputs=True, allow_unused=True)
            g = torch.cat([gr.contiguous().view(-1) for gr in grads]).detach()
            grad_vecs.append(g)
            
        return torch.stack(grad_vecs)
            
    def _get_grad_vec(self):
        """
        returns gradient of NN parameters flattened into a vector
        assumes backward() has been called so each parameters grad attribute
        has been updated
        """
        return torch.cat([p.grad.contiguous().view(-1) 
                             for p in self.trainable_params]
                        )
    
    def forward(self, inputs, n_eigs=None, proj_type="posterior_pred", T=None):
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input
                            
        returns 
            mu = model(inputs) -- shape (N, 1)
            unc = hessian based uncertainty estimates shape (N)
        """
        if not self.configured:
            print("Must call process_dataset first before using model for predictions.")
            raise NotImplementedError
            
        if n_eigs is None:
            n_eigs = self.num_eigs

        if T is not None and T == 0:
            thetas = self.model(inputs)
            dists = self.dist_constructor(thetas)
            unc = dists.entropy()
            return thetas, unc
            
        N = inputs.shape[0]
        
        thetas = self.model(inputs)
        proj_thetas = thetas
        P = None
        if T is not None:
            D = thetas.reshape(N,-1).shape[-1]
            P = torch.randn(D, T)
            P,_ = torch.linalg.qr(P, 'reduced')
            P = P.to(self.device)
            proj_thetas = thetas.reshape(N,-1) @ P

        unc = []
        dists = []

        for j in range(N):
            J = self._get_weight_jacobian(proj_thetas[j,...].view(-1)).detach()
            dist = self.dist_constructor(thetas[j,...])
            f_var = self.projector.compute_diag_var(J, P=P, n_eigs=n_eigs, eps=self.eps)
            f_var = f_var.reshape(thetas[j,...].shape)
            output_dist = dist.marginalize(f_var)
            dists.append(output_dist)
            unc.append( (output_dist.entropy() / self.scaling_factor ).sum())

        unc = torch.stack(unc)
        return dists, unc

    def calibrate(self, val_dataset, percentile=0.99, calibrate_eps=False):
        """
        evalutes the uncalibrated score on the val_dataset, 
        and then selects a
        scaling factor such that percentile % of the dataset are below 1
        """
        scores = []

        dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=10)

        for i, (inputs,labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs, uncs = self.forward(inputs)

            scores.append(uncs)
        
        scores = torch.cat(scores)
        value = torch.quantile(scores, percentile)

        self.scaling_factor.data *= value

