from typing import Union, Optional, Callable, Tuple, List
from scod.sketching.sketched_pca import SRFT_SinglePassPCA
import torch
from torch import nn
from copy import deepcopy
from torch.cuda.amp.autocast_mode import autocast

from tqdm.autonotebook import tqdm
from torch.autograd import grad
from .sketching.sketched_pca import alg_registry
from .sketching.utils import random_subslice
from .distributions import ExtendedDistribution

import numpy as np


class Projector(nn.Module):
    def __init__(self, N : int, 
                 r : int, 
                 device : torch.DeviceObjType = torch.device('cpu')) -> None:
        """
        Class which handles computations comparing a test-time jacobian to
        the top eigenvalues and eigenvectors of the Gauss Newton matrix
        Inputs: N, r, device
            expects eigenvectors of dim N x r
            eigenvalues of dim r
        """
        super().__init__()
        self.N = N
        self.r = r
        
        self.device = device
        
        self.eigs = nn.Parameter(1e-8*torch.ones(self.r, device=self.device), requires_grad=False)
        self.basis = nn.Parameter(torch.randn(self.N, self.r, device=self.device), requires_grad=False)
        
    @torch.no_grad()
    def process_basis(self, eigs : torch.Tensor,
                      basis : torch.Tensor) -> None:
        n_eigs = eigs.shape[0]
        self.eigs.data[-n_eigs:] = eigs.to(self.device)
        self.basis.data[:,-n_eigs:] = basis.to(self.device)

    def ortho_proj(self, L : torch.Tensor, 
                   n_eigs : torch.Tensor) -> torch.Tensor:
        """
        we have U = basis,
        computes ||(I-UU^T)L||^2_F
        """
        basis = self.basis[:,-n_eigs:]
        proj_L = basis.t() @ L 
        proj_L = basis @ proj_L
        return torch.norm(L - proj_L) 
    
    def posterior_pred(self, L : torch.Tensor, 
                       n_eigs : torch.Tensor, 
                       eps : torch.Tensor) -> torch.Tensor:
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
    def compute_distance(self, L : torch.Tensor, 
                         proj_type : torch.Tensor, 
                         n_eigs : Optional[int] = None, 
                         eps : Union[torch.Tensor , float] =1.) -> torch.Tensor:
        if n_eigs is None:
            n_eigs = self.r
            
        L.to(self.device)
        if proj_type == 'ortho':
            return self.ortho_proj(L, n_eigs)
        elif proj_type == 'posterior_pred':
            return self.posterior_pred(L, n_eigs, eps)
        else:
            raise ValueError(proj_type +" is not an understood projection type.")

    def compute_diag_var(self, J : torch.Tensor, 
                               P : Optional[torch.Tensor] = None, 
                               n_eigs : Optional[int] = None, 
                               eps : Union[torch.Tensor, float] = 1.) -> torch.Tensor:
        """
        given J, (d, N)
        
        if P is not given, assumes P is the identity matrix

        returns diagonal variance of P J Sig J^T P^T, where
            Sig = ( 1/eps I + M U D U^T )^{-1}
                = eps I - eps U (1/(eps) D^{-1} + I)^{-1} U^T
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
    'sketch_type': 'gaussian', # sketch type 
    'offline_proj_dim': None, # whether to subsample rows during offline computation
    'online_proj_dim': None, # whether to project output down before taking gradients at test time
    'online_proj_type': 'gaussian' # how to do output projection
}


class SCOD(nn.Module):
    """
    Wraps a trained model with functionality for adding epistemic uncertainty estimation.
    """
    def __init__(self, model : nn.Module, 
                       dist_constructor : Callable[[torch.Tensor], ExtendedDistribution], 
                       args : dict = {}, 
                       parameters : Optional[nn.ParameterList] = None) -> None:
        """
        model: base DNN to equip with an uncertainty metric
        dist_constructor: a function mapping network output to a Distribution object, defining a distribution
            over the output space. The labels in a dataset should lie in the support of this distribution.
        args: configuration variables - defaults are in base_config
            'num_samples': default=None, otherwise int > 0, sketch dimension T (T)
            'num_eigs': default=10, otherwise int > 0 low rank estimate to recover (k)
            'sketch_type': default='srft', sketch type either 'gaussian' or 'srft' for
                linear sketching techniques, or 'ipca' for an incremental PCA approach (much slower)
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
        print("Weight space dimension: %1.3e"%self.n_params)
                
        self.num_samples = self.config['num_samples']
        self.num_eigs = self.config['num_eigs']
        
        if self.num_samples is None:
            self.num_samples = 6*self.num_eigs + 4
            
        self.configured = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)

        # --------------------------------
        # options for offline computation
        self.offline_projection_dim = T = self.config['offline_proj_dim']

        # Empirical Fisher:
        # Whether we should just use empirical fisher, i.e. outer products 
        # of gradients of the negative log prob
        self.use_empirical_fisher = (T is not None and T == 0) 
        
        # Random proj:
        # Rather than compute the whole Fisher, instead, randomly subsample rows of the jacobian
        # This subsampling isn't performed if T is equal or larger than the output dimension
        self.use_random_proj = (T is not None and T > 0)
        
        # Approximation alg:
        # Determines how to aggregate per-datapoint information into final low-rank GGN approx
        self.sketch_class = alg_registry[self.config['sketch_type']] 
        
        # --------------------------------
        # options for online computation
        self.online_projection_dim = self.config['online_proj_dim']
        self.online_projection_type = self.config['online_proj_type']
 

        self.projector = Projector(
            N=self.n_params,
            r=2*max(self.num_eigs + 2, (self.num_samples-1)//3),
            device=self.device
        )

        self.log_eps = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=True) # prior variance
        self.scaling_factor = nn.Parameter(torch.ones(1, device=self.device), requires_grad=False) # final scaling factor to rescale uncertainty output
        self.hyperparameters = [self.log_eps]

    @property
    def eps(self) -> torch.Tensor:
        """
        returns eps, the scale on the prior covariance
        """
        return torch.exp(self.log_eps)
        
    def process_dataset(self, dataset : torch.utils.data.Dataset ) -> None:
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
                                                 num_workers=4,
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
                z = self.model(inputs) 
                dist = self.dist_constructor(z)

                if self.use_empirical_fisher:
                    # contribution of this datapoint is
                    # C = J_l^T J_l, where J_l = d(-log p(y | x))/dw
                    pre_jac_factor = -dist.validated_log_prob(labels) # shape [1]
                else:
                    # contribution of this datapoint is
                    # C = J_f^T L L^T J
                    Lt_z = dist.apply_sqrt_F(z).mean(dim=0) # L^\T theta
                    # flatten
                    pre_jac_factor = Lt_z.view(-1) # shape [prod(event_shape)]

                if self.use_random_proj:
                    pre_jac_factor = random_subslice(pre_jac_factor, dim=0, k=self.offline_projection_dim, scale=True)

            sqrt_C_T = self._get_weight_jacobian(pre_jac_factor) # shape ([T x N])

            sketch.low_rank_update(sqrt_C_T.t()) # add C = sqrt_C sqrt_C^T to the sketch
        
        del sqrt_C_T # free memory @TODO: sketch could take output tensors to populate directly
        eigs, eigvs = sketch.eigs()
        del sketch
        self.projector.process_basis(eigs, eigvs)
            
        self.configured.data = torch.ones(1, dtype=torch.bool)

    def optimize_output_projection(self, dataset : torch.utils.data.Dataset ) -> None:
        """
        Loops through dataset to determine best projection matrix to use for output compression
        """
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=1, 
                                                 shuffle=True,
                                                 num_workers=4,
                                                 pin_memory=True)
        
        sketch = None
        
        n_data = len(dataloader)
        for i, (inputs,labels) in tqdm(enumerate(dataloader), total=n_data):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast():
                z = self.model(inputs).view(-1) # flatten (assumes batch size = 1)
                
                output_shape = z.shape[0]
                if self.use_random_proj:
                    z, idx = random_subslice(z, dim=0, k=self.offline_projection_dim, scale=True, return_idx=True)

            J = self._get_weight_jacobian(z) # shape ([T x N])
            diag_var = (J * J).sum(-1) # shape (T)
            if self.use_random_proj:
                full_diag_var = torch.zeros(output_shape, device=z.device)
                full_diag_var[idx] = diag_var
                diag_var = full_diag_var # ( output_shape )
            
            if sketch is None:
                sketch = SRFT_SinglePassPCA(N=output_shape, 
                                        r=self.online_projection_dim,
                                        device=self.device)
            sketch.low_rank_update(diag_var[:,None]) # add this datapoint to the sketch

        self.online_proj_basis = sketch.eigs()[1] # (output_shape x self.online_projection_dim) 

    
    def _get_weight_jacobian(self, vec : torch.Tensor) -> torch.Tensor:
        """
        returns k x nparam matrix, with each row being d(vec[i])/d(weights) for i = 1, ..., k
        """
        assert len(vec.shape) == 1
        grad_vecs = []
        for j in range(vec.shape[0]):
            grads = grad(vec[j], self.trainable_params, retain_graph=True, only_inputs=True, allow_unused=True)
            g = torch.cat([gr.contiguous().view(-1) for gr in grads]).detach()
            grad_vecs.append(g)
            
        return torch.stack(grad_vecs)

    def output_projection(self, output_size, T):
        if self.online_projection_type == 'PCA':
            return self.online_proj_basis[:,-T:].detach()
        elif self.online_projection_type == 'blocked':
            P = torch.eq(torch.floor( torch.arange(output_size, device=self.device)[:,None] / (output_size // T) ), torch.arange(T, device=self.device)[None,:]).float()
            P /= torch.norm(P, dim=0, keepdim=True)
            return P
        else:
            P = torch.randn(output_size, T, device=self.device)
            P,_ = torch.linalg.qr(P, 'reduced')
            return P
            
    def forward(self, inputs : torch.Tensor, 
                      n_eigs : Optional[int] = None, 
                      T : Optional[int] = None) -> Tuple[ List[torch.distributions.Distribution], torch.Tensor ] :
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input
                            
        returns 
            mu = model(inputs) list of N distribution objects
            unc = hessian based uncertainty estimates shape (N), torch.Tensor
        """
        if not self.configured:
            print("Must call process_dataset first before using model for predictions.")
            raise NotImplementedError

        if T is None:
            T = self.online_projection_dim

        # skip computation if projection dim is 0
        if T is not None and T == 0:
            z = self.model(inputs)
            dists = [self.dist_constructor(z[j,...]) for j in range(z.shape[0])]
            unc = torch.stack([dist.entropy().sum() for dist in dists])
            return dists, unc
            
        if n_eigs is None:
            n_eigs = self.num_eigs
            
        N = inputs.shape[0] # batch size
        
        z = self.model(inputs) # batch of outputs
        flat_z = z.view(N, -1) # batch of flattened outputs
        flat_z_shape = flat_z.shape[-1] # flattened output size

        # by default there is no projection matrix and 
        # proj_flat_z = flat_z
        P = None # projection matrix
        proj_flat_z = flat_z

        if T is not None and T < flat_z.shape[-1]:
            # if projection dim is provided and less than original dim
            # @TODO: generate P through a function call depending on
            # self.online_proj_type
            P = self.output_projection(flat_z_shape, T)
            proj_flat_z = flat_z @ P
        unc = []
        dists = []

        for j in range(N):
            J_proj_flat_z = self._get_weight_jacobian(proj_flat_z[j,:]).detach()
            z_flat_var = self.projector.compute_diag_var(J_proj_flat_z, P=P, n_eigs=n_eigs, eps=self.eps)
            z_var = z_flat_var.view(z[j,...].shape)

            dist = self.dist_constructor(z[j,...])
            output_dist = dist.marginalize(z_var)

            dists.append( output_dist )
            unc.append( (output_dist.entropy() / self.scaling_factor ).sum())

        unc = torch.stack(unc)
        return dists, unc

    def optimize_nll(self, 
                     dataset : torch.utils.data.Dataset,
                     num_epochs : int = 2,
                     batch_size : int = 20):
        """
        tunes prior variance scale (eps) via SGD to minimize 
        validation nll on a given dataset
        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=4, pin_memory=True)
        
        dataset_size = len(dataset)
        optimizer = torch.optim.Adam(self.hyperparameters, lr=1.)

        with tqdm(total=num_epochs, position=0) as pbar:
            pbar2 = tqdm(total=dataset_size, position=1)
            for epoch in range(num_epochs):
                pbar2.refresh()
                pbar2.reset(total=dataset_size)
                for inputs, labels in dataloader:                
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    dists, _ = self.forward(inputs)
                    loss = 0
                    for dist, label in zip(dists, labels):
                        loss += -dist.validated_log_prob(label).mean()

                    
                    loss /= len(dists)
                    
                    loss.backward()
                    optimizer.step()
                    
                    pbar2.set_postfix(batch_loss=loss.item(), eps=self.eps.item())
                    pbar2.update(inputs.shape[0])
            
                pbar.set_postfix(eps=self.eps.item())
                pbar.update(1)

    def optimize_entropy_separation(self,
                                    val_dataset : torch.utils.data.Dataset,
                                    ood_dataset : torch.utils.data.Dataset,
                                    num_epochs = 1,
                                    batch_size = 20):
        """
        optimizes self.logeps via SGD, to maximize:
            E_(x \sim ood)[unc(x)] - E_(x \sim val)[unc(x)]
        """
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2, pin_memory=True)
        ood_dataloader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=2, pin_memory=True)
        
        dataset_size = len(val_dataset)
        optimizer = torch.optim.Adam(self.hyperparameters, lr=0.1)

        with tqdm(total=num_epochs, position=0) as pbar:
            pbar2 = tqdm(total=dataset_size, position=1)
            for epoch in range(num_epochs):
                pbar2.refresh()
                pbar2.reset(total=dataset_size)
                for (val_inputs, _), (ood_inputs, _) in zip(val_dataloader, ood_dataloader):                
                    val_inputs = val_inputs.to(self.device)
                    ood_inputs = ood_inputs.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    _, val_unc = self.forward(val_inputs)
                    _, ood_unc = self.forward(ood_inputs)
                    loss = val_unc.mean() - ood_unc.mean()
                    
                    loss.backward()
                    optimizer.step()
                    
                    pbar2.set_postfix(batch_loss=loss.item())
                    pbar2.update(val_inputs.shape[0])
            
                pbar.set_postfix(eps=self.eps.item())
                pbar.update(1)


    def calibrate(self, val_dataset, percentile=0.99) -> None:
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