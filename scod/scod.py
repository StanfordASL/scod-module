import torch
from torch import nn
from copy import deepcopy

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
    
    def posterior_pred(self, L, n_eigs, Meps):
        """
        we have U = basis,
        computes ||(I-UU^T)L||^2_F
        """
        basis = self.basis[:,-n_eigs:]
        eigs = torch.clamp( self.eigs[-n_eigs:], min=0.)

        scaling = torch.sqrt( eigs / ( eigs + 1./(Meps) ) )
        proj_L = scaling[:,None] * (basis.t() @ L)

        return torch.sqrt( torch.sum(L**2) - torch.sum(proj_L**2) )
    
    @torch.no_grad()
    def compute_distance(self, L, proj_type, n_eigs=None, Meps=5000.):
        if n_eigs is None:
            n_eigs = self.r
            
        L.to(self.device)
        if proj_type == 'ortho':
            return self.ortho_proj(L, n_eigs)
        elif proj_type == 'posterior_pred':
            return self.posterior_pred(L, n_eigs, Meps)
        else:
            raise ValueError(proj_type +" is not an understood projection type.")

base_config = {
    'num_samples': None, # sketch size T (T)
    'num_eigs': 10, # low rank estimate to recover (k)
    'weighted': False, # weight samples in sketch by loss
    'sketch_type': 'gaussian', # sketch type 
    'curvature_type': 'fisher', # 'fisher' or 'sampled_ggn'
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
        
        self.projector = Projector(
            N=self.n_params,
            r=2*max(self.num_eigs + 2, (self.num_samples-1)//3),
            device=self.device
        )
        
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
                                                 shuffle=True)
        
        sketch = self.sketch_class(N=self.n_params, 
                                   M=len(dataloader),
                                   r=self.num_eigs,
                                   T=self.num_samples,
                                   device=self.device)
        
        n_data = len(dataloader)
        for i, (inputs,labels) in tqdm(enumerate(dataloader), total=n_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            thetas = self.model(inputs) # get params of output dist

            dist = self.dist_constructor(thetas)
            weight = 1.
            
            Lt_th = dist.apply_sqrt_F(thetas).mean(dim=0) # L^\T theta
            Lt_J = self._get_weight_jacobian(Lt_th) # L^\T J, J = dth / dw
            sketch.low_rank_update(Lt_J.t(),weight) # add 1/M J^T L L^T J to the sketch
        
        del Lt_J
        
        eigs, eigvs = sketch.eigs()
        del sketch
        
        self.projector.process_basis(eigs, eigvs)
            
        self.configured.data = torch.ones(1, dtype=torch.bool)

        self.scaling_factor = torch.ones(1, device=self.device)
    
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
    
    def forward(self, inputs, n_eigs=None, proj_type="posterior_pred", Meps=5000):
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
            
        N = inputs.shape[0]
        
        thetas = self.model(inputs)
        dists = self.dist_constructor(thetas)
        unc = torch.zeros(N)
        
        # batch apply sqrt(I_th) to output
        Lt_th = dists.apply_sqrt_F(thetas)

        # compute uncertainty by backpropping back into each sample
        for j in range(N):
            # dist = self.dist_constructor(thetas[j,:])
            # Lt_th = dist.apply_sqrt_F(thetas[j,:])
            Lt_J = self._get_weight_jacobian(Lt_th[j,:])    
            unc[j] = self.projector.compute_distance(Lt_J.t(), proj_type, n_eigs=n_eigs, Meps=Meps) / self.scaling_factor

        return thetas, unc

    def calibrate(self, val_dataset, percentile=0.99):
        """
        evalutes the uncalibrated score on the val_dataset, and then selects a
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

        self.scaling_factor *= value

