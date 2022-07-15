# SCOD: Sketching Curvature for Out-of-Distribution Detection

This repository implements SCOD, a mechanism to take any pre-trained neural network and a set of training data to construct an efficient characterization of epistemic uncertainty. The methods implemented here builds on work detailed in our paper
> Sharma, Apoorva, Navid Azizan, and Marco Pavone. "Sketching Curvature for Efficient Out-of-Distribution Detection for Deep Neural Networks." [arXiv preprint arXiv:2102.12567](https://arxiv.org/abs/2102.12567) (2021). 


This repo implements SCOD as a wrapper to an existing pytorch module. For example, if `model` is a pytorch model, then we can construct a SCOD wrapper by calling
```
from scod import SCOD

# model is a torch.nn.Module that has already been trained

scod_model = SCOD(model)
```

Offline, SCOD processes a set of training data. SCOD assumes a probabilistic treatment of the model output, i.e. if `z = model(x)`, then z is the parameter which defines a distribution over the targets `y`. This map from `z` to a probability distribution is represented in the framework through a `scod.distributions.DistributionLayer` object, which acts like a torch network layer, but outputs a `torch.distributions.Distribution` object rather than a tensor. SCOD provides DistributionLayers representing common choices for regression and classification tasks, e.g. interpreting `z` as logits specifying a categorical distribution over the target classes `y`.

```
# assuming a classification problem
from scod.distributions import CategoricalLogitLayer

# dataset is a torch.utils.data.Dataset object containing the training data

# assuming classification problem
dist_layer = CategoricalLogitLayer()
scod_model.process_dataset(dataset, dist_layer)
```

After processing the training data, SCOD can make predictions on test data. Instead of outputing a point estimate of `z`, SCOD instead outputs a Gaussian predictive distribution over `z`, specified by a mean and diagonal variance.

```
# x_test is a batch of test points, shape (N, d_x)

z_mean, z_var = scod_model(x_test)
# z_mean has shape (N, d_z), z_var has shape (N, d_z)
```

To use these predictions in the context of OOD detection, we can create a `scod.OodDetector` object, which uses the Gaussian predictions on `z` to estimate the overall entropy of the marginal distribution on the targets `y`, a scalar quantity which can be thresholded to separate in-distribution from out-of-distribution data.

```
from scod import OodDetector

ood_detector = OodDetector(scod_model, dist_layer)

ood_signal = ood_detector(x_test)
# ood_signal has shape (N,)
```

The IPython notebooks in the `demos/` folder provide an interactive demonstration of SCOD.