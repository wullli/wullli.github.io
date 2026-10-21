---
title: Pyro Fundamentals on the Example of a Bayesian Neural Network
date: 2026-10-13
summary: Implementing the Bayesian neural network from the previous part in Pyro and training it with SVI.
tags: stats, svi, probabilities, pyro
draft: true
---

## The Diabetes dataset



Below follows the code for the BNN model from the previous part in Pyro.


We start with the imports and some constants. We have 5 neurons in the hidden layer, two inputs and one output. 
Furthermore, we set the prior for the BNN parameters to a standard normal.

```python fold-imports
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from pyro.distributions import constraints
from sklearn.metrics import f1_score
from pyro.infer import Predictive

DIM = 5
PRIOR = dist.Normal(0.0, 1.0)
IN_DIM = 2
OUT_DIM = 1
```


Next, we define what happens during a forward pass of the network:


```python fold-imports
def forward(x: torch.Tensor, 
            weights_hidden: torch.Tensor, 
            weights_out: torch.Tensor, 
            bias_h: torch.Tensor,
            bias_o: torch.Tensor):
    h = torch.matmul(x, weights_hidden) + bias_h
    h = torch.relu(h)
    y = torch.matmul(h, weights_out) + bias_o
    return y
```


Then we define the generative model. This is where Pyro comes into play for the first time.

```python fold-imports
def model(x: torch.Tensor, y: torch.Tensor | None = None):
    weights_hidden = pyro.sample("w_h", PRIOR.expand([IN_DIM, DIM]).to_event(2))
    weights_out = pyro.sample("w_o", PRIOR.expand([DIM, OUT_DIM]).to_event(2))
    bias_h = pyro.sample("b_h", PRIOR.expand([DIM]).to_event(1))
    bias_o = pyro.sample("b_o", PRIOR.expand([1]).to_event(1))

    logits = forward(x, weights_hidden, weights_out, bias_h, bias_o).flatten()
    with pyro.plate("data", x.shape[0]):
        pyro.deterministic("p", torch.sigmoid(logits))
        return pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)
```

What initially might be confusing is the `pyro.sample` here. The `expand` function here is necessary,
since the prior is univariate and would return just one dimension when sampling. However, 
we want our weights to be a matrix, where each element is drawn independently from this prior.
The `expand` just allows us to add batch dimensions for the sampling. The `to_event(2)` denotes, that we
will treat the sampled weight matrix as a single random variable subsequently (joint), with the 
argument indicating how many of the dimensions should be considered event instead of batch dimensions.

Subsequently the logits are calculated using the sampled parameters and a output is sampled using
a Bernoulli distribution with the logits as its parameters. If an observation is given, the outputs 
are not actually sampled and the `pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)`
will simply return the observation `y`. However, in the latter case, the statement will have the effect of marking this node
in the underlying bayesian network Pyro generates as observed and calculating how likely the observation is given
our current distribution (Bernoulli of the logits). This score is saved in the background and it corresponds to the likelihood of the ELBO
as previously introduced ($$$$).


$$
\text{ELBO} = \underbrace{\mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log p(\mathbf{x} \mid \boldsymbol{\Theta})\big]}_{\text{reconstruction / data fit}} - \underbrace{D_\text{KL}\big(q(\boldsymbol{\Theta}) \parallel p(\boldsymbol{\Theta})\big)}_{\text{regularization}}
$$



```python fold-imports
def guide(x: torch.Tensor, y: torch.Tensor | None = None):
    weights_hidden_loc = pyro.param("w_h_loc", torch.randn(IN_DIM, DIM))
    weights_hidden_scale = pyro.param("w_h_scale", torch.full((IN_DIM, DIM), 0.1), constraint=constraints.positive)

    weights_out_loc = pyro.param("w_o_loc", torch.randn(DIM, OUT_DIM))
    weights_out_scale = pyro.param("w_o_scale", torch.full((DIM, OUT_DIM), 0.1), constraint=constraints.positive)

    bias_h_loc = pyro.param("b_h_loc", torch.full((DIM,), 0.0))
    bias_h_scale = pyro.param("b_h_scale", torch.full((DIM,), 0.05), constraint=constraints.positive)

    bias_o_loc = pyro.param("b_o_loc", torch.full((1,), 0.0))
    bias_o_scale = pyro.param("b_o_scale", torch.full((1,), 0.05), constraint=constraints.positive)

    pyro.sample("w_h", dist.Normal(weights_hidden_loc, weights_hidden_scale).to_event(2))
    pyro.sample("w_o", dist.Normal(weights_out_loc, weights_out_scale).to_event(2))
    pyro.sample("b_h", dist.Normal(bias_h_loc, bias_h_scale).to_event(1))
    pyro.sample("b_o", dist.Normal(bias_o_loc, bias_o_scale).to_event(1))
```


![Left: the learned q for the diabetes BNN, with glucose and pedigree as inputs. Right: one sample from q, which is an ordinary network](images/bnn_q_sample_diabetes.svg)

Each sample from $q$ gives a slightly different network and therefore a slightly different prediction. The spread of these predictions is the uncertainty of our model.

#### Pyro Model
```python fold-imports
{% include "bnn.py" %}
```


1. Split the joint

$$
\log p(\boldsymbol{\Theta}, \mathbf{x}) = \log p(\mathbf{x} \mid \boldsymbol{\Theta}) + \log p(\boldsymbol{\Theta})
$$

2. Substitute into the ELBO (expectation is linear)

$$
\text{ELBO} = \mathbb{E}{q}\big[\log p(\mathbf{x} \mid \boldsymbol{\Theta})\big] + \mathbb{E}{q}\big[\log p(\boldsymbol{\Theta})\big] -
\mathbb{E}_{q}\big[\log q(\boldsymbol{\Theta})\big]
$$

3. Combine the last two terms

$$
\mathbb{E}{q}\big[\log p(\boldsymbol{\Theta}) - \log q(\boldsymbol{\Theta})\big] = -,\mathbb{E}{q}\left[\log
\frac{q(\boldsymbol{\Theta})}{p(\boldsymbol{\Theta})}\right] = -,D_\text{KL}\big(q(\boldsymbol{\Theta}) \parallel p(\boldsymbol{\Theta})\big)
$$

Result

$$
\text{ELBO} = \underbrace{\mathbb{E}{\boldsymbol{\Theta} \sim q}\big[\log p(\mathbf{x} \mid \boldsymbol{\Theta})\big]}{\text{reconstruction / data
fit}} - \underbrace{D_\text{KL}\big(q(\boldsymbol{\Theta}) \parallel p(\boldsymbol{\Theta})\big)}_{\text{regularization}}
$$
