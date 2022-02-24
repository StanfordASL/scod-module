"""
SCOD: Sketching Curvature for OOD Detection
"""
from typing import Optional, Callable, Tuple, List
from copy import deepcopy, copy

import torch
from torch import nn
from torch.autograd import grad
from torch.cuda.amp.autocast_mode import autocast

from tqdm.autonotebook import tqdm

from .sketching.sketched_pca import alg_registry
from .sketching.utils import random_subslice
from .distributions import ExtendedDistribution
from .utils import gaussian_kl_div, gaussian_wasserstein_dist


base_config = {
    "num_samples": None,  # sketch size T (T)
    "num_eigs": 10,  # low rank estimate to recover (k)
    "prior_type": "scalar",  # options are 'scalar' (isotropic prior), 'per_parameter', 'per_weight'
    "sketch_type": "gaussian",  # sketch type
    "offline_proj_dim": None,  # whether to subsample rows during offline computation
    "online_proj_dim": None,  # whether to project output down before taking gradients at test time
    "online_proj_type": "gaussian",  # how to do output projection
}


class SCOD(nn.Module):
    """
    Wraps a trained model with functionality for adding epistemic uncertainty estimation.
    """

    def __init__(
        self,
        model: nn.Module,
        dist_constructor: Callable[[torch.Tensor], ExtendedDistribution],
        args: Optional[dict] = None,
        parameters: Optional[nn.ParameterList] = None,
    ) -> None:
        """Initializes SCOD module as a wrapper around an existing, pre-trained DNN.

        Args:
            model (nn.Module): Pre-trained DNN
            dist_constructor (Callable[[torch.Tensor], ExtendedDistribution]):
                Function mapping DNN output to a ExtendedDistribution object.
            args (dict, optional):
                Configuration parameters for SCOD. If None, uses default settings. Defaults to None.
            parameters (nn.ParameterList, optional):
                The parameters of model to consider when computing approximate Bayesian posterior.
                If None, uses all model.parameters() for which requires_grad is True.  Defaults to None.
        """
        super().__init__()

        self.config = deepcopy(base_config)
        if args is not None:
            self.config.update(args)

        self.model = model
        self.dist_constructor = dist_constructor

        # extract device from model
        self.device = next(model.parameters()).device

        # extract parameters to consider in sketch - keep all that yield valid gradients
        if parameters is None:
            self.trainable_params = list(
                filter(lambda x: x.requires_grad, self.model.parameters())
            )
        else:
            self.trainable_params = list(parameters)
        self.n_weights_per_param = list(p.numel() for p in self.trainable_params)
        self.n_weights = int(sum(self.n_weights_per_param))
        print("Weight space dimension: %1.3e" % self.n_weights)

        self.num_samples = self.config["num_samples"]
        self.num_eigs = self.config["num_eigs"]

        if self.num_samples is None:
            self.num_samples = 6 * self.num_eigs + 4

        self.configured = nn.Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )

        # ---------------------------
        # options for prior representation and computation
        self.prior_type = self.config["prior_type"]
        self.log_prior_scale = nn.Parameter(self._init_log_prior(), requires_grad=True)

        # --------------------------------
        # options for offline computation
        self.offline_projection_dim = T = self.config["offline_proj_dim"]

        # Empirical Fisher:
        # Whether we should just use empirical fisher, i.e. outer products
        # of gradients of the negative log prob
        self.use_empirical_fisher = T is not None and T == 0

        # Random proj:
        # Rather than compute the whole Fisher, instead, randomly subsample rows of the jacobian
        # This subsampling isn't performed if T is equal or larger than the output dimension
        self.use_random_proj = T is not None and T > 0

        # Approximation alg:
        # Determines how to aggregate per-datapoint information into final low-rank GGN approx
        self.sketch_class = alg_registry[self.config["sketch_type"]]

        # --------------------------------
        # options for online computation
        self.online_projection_dim = self.config["online_proj_dim"]
        self.online_projection_type = self.config["online_proj_type"]

        # --------------------------------
        # Parameters to be saved as part of self.state_dict() for easy reloading:
        # Low-rank approx of Gauss Newton, filled in by self.process_dataset
        self.GGN_eigs = nn.Parameter(
            torch.zeros(self.num_eigs, device=self.device), requires_grad=False
        )

        self.GGN_basis = nn.Parameter(
            torch.zeros(self.n_weights, self.num_eigs, device=self.device),
            requires_grad=False,
        )

        # stores what prior scales were used in GGN computation
        self.GGN_sqrt_prior = nn.Parameter(copy(self.sqrt_prior), requires_grad=False)
        self.GGN_is_aligned = True

        self.hyperparameters = [self.log_prior_scale]

    def _init_log_prior(self) -> torch.Tensor:
        """Returns intial value of log_prior parameter, depending on self.prior_type

        Raises:
            ValueError: If self.prior_type is not an understood value.

        Returns:
            torch.Tensor: torch.zeros() of the correct shape
        """
        if self.prior_type == "per_parameter":
            n_prior_params = len(self.trainable_params)
        elif self.prior_type == "per_weight":
            n_prior_params = self.n_weights
        elif self.prior_type == "scalar":
            n_prior_params = 1
        else:
            raise ValueError(
                "prior_type must be one of (scalar, per_parameter, per_weight)"
            )
        return torch.zeros(n_prior_params, device=self.device)

    def _broadcast_to_n_weights(self, v: torch.Tensor) -> torch.Tensor:
        """Broadcasts a vector to be the length of self.n_weights
        The vector must be one of the following lengths:
          - 1 -> broadcasts as usual
          - len(self.trainable_parameters) -> each element is repeated
                for the number of weights in each parameter
          - n_weights -> returned as is

        Args:
            v (torch.Tensor): vector to be expanded

        Raises:
            ValueError: v is not of a compatible length

        Returns:
            torch.Tensor: expanded view into v
        """
        assert len(v.shape) == 1
        k = v.shape[0]
        if k == 1:
            return v.expand(self.n_weights)

        if k == len(self.trainable_params):
            return torch.cat(
                [vi.expand(n) for vi, n in zip(v, self.n_weights_per_param)]
            )

        if k == self.n_weights:
            return v

        raise ValueError(
            "Input vector is not compatible with self.n_weights or self.trainable_params"
        )

    @property
    def sqrt_prior(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Learned elements of diagonal part of Sigma_0^{1/2}
        """
        return torch.exp(0.5 * self.log_prior_scale)

    def _get_weight_jacobian(
        self, vec: torch.Tensor, scaled_by_prior: bool = True
    ) -> torch.Tensor:
        """Returns a k x n_param matrix, with each row being d(vec[i])/d(weights) for i = 1, .., k
        (Currently just loops over dimensionality of vec, as batch jacobians aren't supported in pytorch)

        if scaled_by_prior is True, then scales jacobian by self.sqrt_prior before returning

        Args:
            vec (torch.Tensor): function output to differentiate via backprop
            scaled_by_prior (bool, optional): whether to scale jacobian by prior. Defaults to True.

        Returns:
            torch.Tensor: J = df/dweights
        """
        assert len(vec.shape) == 1
        grad_vecs = []
        for j in range(vec.shape[0]):
            grads = grad(
                vec[j],
                self.trainable_params,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True,
            )
            g = torch.cat([gr.contiguous().view(-1) for gr in grads]).detach()
            if scaled_by_prior:
                g = g * self._broadcast_to_n_weights(self.sqrt_prior)
            grad_vecs.append(g)

        return torch.stack(grad_vecs)

    def _predictive_var(
        self, J: torch.Tensor, n_eigs: Optional[int] = None, prior_only: bool = False
    ):
        r"""
        Returns the predictive covariance matrix
            Sigma_z = J ( I + G )^{-1} J.T  (shape: d x d)

        where J are scaled Jacobian matrices and G is the approximated Gauss Newton matrix
        assumes J = [d x n_weights]

        n_eigs dictates how many eigenvalues of G to use, by default equal to self.num_eigs

        prior_only: if True, ignores G, returning the prior predictive variance

        ---

        Interally, G is represented as U diag(eigs) U.T
        Thus, by the Woodbury identity, (I + G)^{-1} is equal to
            I - U diag( eigs / (eigs + 1) ) U.T

        Assumes J is the jacobian already scaled by the sqrt prior covariance.

        Returns diag( J ( I - U D U^T ) J^T )
        where D = diag( eigs / (eigs + 1) )

        and U diag(eigs) U^T ~= G, the Gauss Newton matrix computed using gradients
        already scaled by the sqrt prior covariance

        that is, we let J = (df/dw) \Sigma_0^{1/2}
        and G = sum( J_i^T H_i J_i )
        """
        JJT = J @ J.T
        if not self.configured or prior_only:
            return JJT

        if n_eigs is None:
            n_eigs = self.num_eigs

        # pylint: disable=invalid-unary-operand-type
        basis = self.GGN_basis[:, -n_eigs:]
        eigs = torch.clamp(self.GGN_eigs[-n_eigs:], min=0.0)

        if self.GGN_is_aligned:
            scaling = eigs / (eigs + 1.0)
            neg_term = torch.sqrt(scaling[None, :]) * (J @ basis)
            neg_term = neg_term @ neg_term.T
        else:
            # G = rescaling*U D (rescaling*U).T
            # the rescaling breaks the orthogonality of the basis, need to do matrix inversion
            rescaling = self._broadcast_to_n_weights(
                self.sqrt_prior / self.GGN_sqrt_prior
            )
            basis = rescaling[:, None] * basis
            inv_term = torch.linalg.inv(torch.diag_embed(1.0 / eigs) + basis.T @ basis)
            scaled_jac = J @ basis
            neg_term = scaled_jac @ inv_term @ scaled_jac.T

        return JJT - neg_term

    def process_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        dataloader_kwargs: Optional[dict] = None,
    ) -> None:
        """
        summarizes information about training data by logging gradient directions
        seen during training, and then using gram schmidt of these to form
        an orthonormal basis. directions not seen during training are
        taken to be irrelevant to data, and used for detecting generalization

        dataset - torch dataset of (input, target) pairs
        """
        # loop through data, one sample at a time
        print("computing basis")
        if dataloader_kwargs is None:
            dataloader_kwargs = {
                "num_workers": 4,
            }
            if self.device.type != "cpu":
                dataloader_kwargs["pin_memory"] = True
        dataloader_kwargs.update({"batch_size": 1})

        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

        sketch = self.sketch_class(
            N=self.n_weights, r=self.num_eigs, T=self.num_samples, device=self.device
        )

        n_data = len(dataloader)
        for i, (inputs, labels) in tqdm(enumerate(dataloader), total=n_data):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast():
                z = self.model(inputs)
                dist = self.dist_constructor(z)

                if self.use_empirical_fisher:
                    # contribution of this datapoint is
                    # C = J_l^T J_l, where J_l = d(-log p(y | x))/dw
                    pre_jac_factor = -dist.validated_log_prob(labels)  # shape [1]
                else:
                    # contribution of this datapoint is
                    # C = J_f^T L L^T J
                    Lt_z = dist.apply_sqrt_F(z).mean(dim=0)  # L^\T theta
                    # flatten
                    pre_jac_factor = Lt_z.view(-1)  # shape [prod(event_shape)]

                if self.use_random_proj:
                    pre_jac_factor = random_subslice(
                        pre_jac_factor, dim=0, k=self.offline_projection_dim, scale=True
                    )

            sqrt_C_T = self._get_weight_jacobian(
                pre_jac_factor, scaled_by_prior=True
            )  # shape ([T x N])

            sketch.low_rank_update(
                sqrt_C_T.t()
            )  # add C = sqrt_C sqrt_C^T to the sketch

        del sqrt_C_T  # free memory @TODO: sketch could take output tensors to populate directly
        eigs, eigvs = sketch.eigs()
        del sketch
        self.GGN_eigs.data = eigs.to(self.device)
        self.GGN_basis.data = eigvs.to(self.device)
        self.GGN_sqrt_prior.data = copy(self.sqrt_prior)
        self.GGN_is_aligned = True

        self.configured.data = torch.ones(1, dtype=torch.bool)

    def kernel_matrix(self, inputs, prior_only=True, n_eigs=None):
        """
        returns a kernel (gram) matrix for the set of inputs

        returns a matrix of size: [KD x KD] where inputs is [X, d]
        """
        N = inputs.shape[0]  # batch size

        z = self.model(inputs)  # batch of outputs
        flat_z = z.view(N, -1)  # batch of flattened outputs
        jacs = []
        for j in range(N):
            J = self._get_weight_jacobian(flat_z[j, :], scaled_by_prior=True)
            jacs.append(J)

        jacs = torch.cat(jacs, dim=0)  # (KD x N)
        return self._predictive_var(
            jacs, n_eigs=n_eigs, prior_only=prior_only
        )  # (KD x KD)

    def output_projection(self, output_size, T):
        if self.online_projection_type == "PCA":
            return self.online_proj_basis[:, -T:].detach()
        elif self.online_projection_type == "blocked":
            P = torch.eq(
                torch.floor(
                    torch.arange(output_size, device=self.device)[:, None]
                    / (output_size // T)
                ),
                torch.arange(T, device=self.device)[None, :],
            ).float()
            P /= torch.norm(P, dim=0, keepdim=True)
            return P
        else:
            P = torch.randn(output_size, T, device=self.device)
            P, _ = torch.linalg.qr(P, "reduced")
            return P

    def forward(
        self,
        inputs: torch.Tensor,
        use_prior: bool = False,
        n_eigs: Optional[int] = None,
        T: Optional[int] = None,
    ) -> Tuple[List[torch.distributions.Distribution], torch.Tensor]:
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input

        returns
            mu = model(inputs) list of N distribution objects
            unc = hessian based uncertainty estimates shape (N), torch.Tensor
        """
        if T is None:
            T = self.online_projection_dim

        # skip computation if projection dim is 0
        if T is not None and T == 0:
            z = self.model(inputs)
            dists = [self.dist_constructor(z[j, ...]) for j in range(z.shape[0])]
            unc = torch.stack([dist.entropy().sum() for dist in dists])
            return dists, unc

        if n_eigs is None:
            n_eigs = self.num_eigs

        N = inputs.shape[0]  # batch size

        z = self.model(inputs)  # batch of outputs
        flat_z = z.view(N, -1)  # batch of flattened outputs
        flat_z_shape = flat_z.shape[-1]  # flattened output size

        # by default there is no projection matrix and proj_flat_z = flat_z
        P = None  # projection matrix
        proj_flat_z = flat_z
        if T is not None and T < flat_z.shape[-1]:
            # if projection dim is provided and less than original dim
            P = self.output_projection(flat_z_shape, T)
            proj_flat_z = flat_z @ P

        unc = []
        dists = []
        for j in range(N):
            J_proj_flat_z = self._get_weight_jacobian(
                proj_flat_z[j, :], scaled_by_prior=True
            )
            proj_flat_z_var = self._predictive_var(
                J_proj_flat_z, n_eigs=n_eigs, prior_only=use_prior
            )
            if P is not None:
                chol_sig = torch.linalg.cholesky(proj_flat_z_var)
                flat_z_diag_var = torch.sum(
                    (P @ chol_sig) ** 2, dim=-1
                )  # ( P[:,None,:] @ ( (JJT - neg_term) @ P.T) )[:,0]
            else:
                flat_z_diag_var = torch.diagonal(proj_flat_z_var)

            z_var = flat_z_diag_var.view(z[j, ...].shape)
            dist = self.dist_constructor(z[j, ...])
            output_dist = dist.marginalize(z_var)

            dists.append(output_dist)
            unc.append((output_dist.entropy()).sum())

        unc = torch.stack(unc)
        return dists, unc

    def optimize_prior_scale_by_nll(
        self,
        dataset: torch.utils.data.Dataset,
        dataloader_kwargs: Optional[dict] = None,
        num_epochs: int = 2,
    ):
        """
        tunes prior variance scale (eps) via SGD to minimize
        validation nll on a given dataset
        """
        self.GGN_is_aligned = False
        if dataloader_kwargs is None:
            dataloader_kwargs = {
                "num_workers": 4,
                "batch_size": 20,
                "shuffle": True,
            }
            if self.device.type != "cpu":
                dataloader_kwargs["pin_memory"] = True

        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

        dataset_size = len(dataset)
        optimizer = torch.optim.Adam(self.hyperparameters, lr=1e-1)

        losses = []

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
                        loss += -dist.log_prob(
                            label
                        ).mean()  # -dist.validated_log_prob(label).mean()

                    loss /= len(dists)

                    loss.backward()
                    optimizer.step()

                    pbar2.set_postfix(
                        batch_loss=loss.item(), eps=self.sqrt_prior.mean().item()
                    )
                    pbar2.update(inputs.shape[0])

                    losses.append(loss.item())

                pbar.set_postfix(eps=self.sqrt_prior.mean().item())
                pbar.update(1)

        return losses

    def optimize_prior_scale_by_GP_kernel(
        self,
        dataset: torch.utils.data.Dataset,
        GP_kernel: Callable[[torch.Tensor], torch.Tensor],
        GP_mu: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dataloader_kwargs: Optional[dict] = None,
        num_epochs: int = 20,
        grad_accumulation_steps: int = 5,
        dist_loss: str = "wass",
    ):
        """
        tunes prior variance scale (eps) via SGD to minimize
        difference between prior and a given GP
        GP_kernel should take in a batch of data and produce a gram matrix
        """
        self.GGN_is_aligned = False
        if dataloader_kwargs is None:
            dataloader_kwargs = {
                "num_workers": 4,
                "batch_size": 20,
                "shuffle": True,
            }
            if self.device.type != "cpu":
                dataloader_kwargs["pin_memory"] = True

        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

        dataset_size = len(dataset)
        optimizer = torch.optim.Adam(self.hyperparameters, lr=1e-1)

        losses = []
        min_eigs = []

        grad_counter = 0
        with tqdm(total=num_epochs, position=0) as pbar:
            pbar2 = tqdm(total=dataset_size, position=1)
            for epoch in range(num_epochs):
                pbar2.refresh()
                pbar2.reset(total=dataset_size)
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    K_dnn = self.kernel_matrix(inputs, prior_only=True)
                    min_eig = torch.min(torch.abs(torch.linalg.eigvals(K_dnn)))

                    K_gp = GP_kernel(inputs).to(self.device, non_blocking=True)
                    err = None
                    if GP_mu is not None:
                        mu_dnn = self.model(inputs)
                        mu_gp = GP_mu(inputs)
                        err = (mu_dnn - mu_gp).view(-1)

                    if dist_loss == "fwd_kl":
                        loss = gaussian_kl_div(K_gp, K_dnn, err)
                    elif dist_loss == "rev_kl":
                        loss = gaussian_kl_div(K_dnn, K_gp, err)
                    else:
                        loss = gaussian_wasserstein_dist(K_gp, K_dnn, err)

                    loss = loss / grad_accumulation_steps

                    loss.backward()

                    grad_counter += 1

                    if grad_counter % grad_accumulation_steps == 0:
                        # clip gradients
                        nn.utils.clip_grad_norm_(self.hyperparameters, 5.0)
                        # after grad_accumulation steps have passed, then take the gradient step
                        optimizer.step()
                        # zero the parameter gradients
                        optimizer.zero_grad()

                    pbar2.set_postfix(
                        batch_loss=loss.item(), eps=self.sqrt_prior.mean().item()
                    )
                    pbar2.update(inputs.shape[0])

                    losses.append(loss.item())
                    min_eigs.append(min_eig.item())

                pbar.set_postfix(eps=self.sqrt_prior.mean().item())
                pbar.update(1)

        return losses, min_eigs
