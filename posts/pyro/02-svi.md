---
title: Variational Inference
date: 2026-10-13
summary: This article is intended as my way to learn pyro, though it may be useful for someone else.
tags: stats, svi, probabilities, pyro
draft: true
---

## Integrating is sometimes hard
In order to understand SVI and ELBO we first need to understand why these tools are actually needed.
Coming from a deep learning background I feel like an intuitive way to show this is using a simple Bayesian neural network 
with weights $\boldsymbol{\Theta}$ and data $\mathbf{X}$.
We use uppercase letters for random variables and lowercase letters for their concrete values:
the weights are a random vector $\boldsymbol{\Theta} = (\Theta_1, \dots, \Theta_n)$, and $\boldsymbol{\theta} = (\theta_1, \dots, \theta_n)$ is one particular setting of them.
Likewise, $\mathbf{X}$ is the data as a random variable and $\mathbf{x}$ is the dataset we actually observed.
When we write densities such as $p(\boldsymbol{\theta} \mid \mathbf{x})$, this is short for the density of $\boldsymbol{\Theta}$ given $\mathbf{X} = \mathbf{x}$, evaluated at $\boldsymbol{\theta}$.

![Example Bayesian neural network with 2 inputs, 5 hidden units and 15 weights, each with the same prior N(0, 1)](images/bnn_network.svg)

A neural nework becomes Bayesian, if the weights are distributions instead of point estimates, such that
we get a distribution over networks that we could sample from or such that we could model uncertainty better.
We can write down the posterior (our belief about the weights after observing the data) easily using the Bayes rule:

$$
p(\boldsymbol{\theta} \mid \mathbf{x}) = \frac{p(\boldsymbol{\theta})\, p(\mathbf{x} \mid \boldsymbol{\theta})}{p(\mathbf{x})}
$$

However, we cannot directly calculate this posterior, since it would require us to integrate over all possible weights to
calculate $p(\mathbf{x})$:

$$
p(\mathbf{x}) = \int_{\mathbb{R}^n} p(\boldsymbol{\theta})\; p(\mathbf{x} \mid \boldsymbol{\theta})\; d\boldsymbol{\theta}
$$

Why this is hard: If there is no closed-form solution for an integral (like here), we have to evaluate the function we want to integrate at certain points and then 
add the slices we get together to approximate the interval. However, if we want to evaluate with the same precision in higher dimensional functions, we need exponentially
many evaluations of the function. So if we evaluate $m = 100$ points for $n = 10$ weights, we get $m^n = 10^{20}$ evaluations, which is not feasible. If 
we assume our hardware can perform $10^{11}$ evaluations per second, it would take us around 32 years for all evaluations.

![Grid points needed to integrate over 1, 2 and 3 weights at the same spacing](images/curse_of_dimensionality.svg)

## Estimating Posteriors
Now, there is two common ways to model the posterior when we cannot calculate the required probabilities by integration or closed-form.
The first is sampling, where we estimate the posterior by sampling parameters for our posterior model (e.g., importance sampling, rejection sampling, MCMC, ...).
The second is posing the problem as an optimization and using optimizers to find a good approximation for the posterior (variational inference).
We will not go into depth on the difference of these approaches, but generally sampling becomes very expensive with big datasets, since
sampled parameters require calculation of the likelihood using the complete dataset. However, there is a trade-off: sampling (MCMC) is asymptotically exact, which
is not the case for SVI, but we can estimate gradients using a subset of the data easily, making it more scalable to large datasets.


Since we cannot integrate, we need a model of the posterior that we can optimize using some objective.
Note that our unobserved (latent) variables are the weights $\boldsymbol{\Theta}$.
Therefore, we approximate the joint posterior $p(\boldsymbol{\theta} \mid \mathbf{x})$ over all weights.
Our estimate for the posterior we will call $q(\boldsymbol{\theta})$ from here on out. Unlike the prior, we model each weight $\Theta_i$ with its own normal distribution (its own mean and variance),
all independent of each other:

$$
q(\boldsymbol{\theta}) = \prod_{i=1}^{n} \mathcal{N}(\theta_i;\, \mu_i, \sigma_i^2)
$$

$\mathcal{N}(\theta_i;\, \mu_i, \sigma_i^2)$ just states that we evaluate the normal density at value $\theta_i$ with mean $\mu_i$ and variance $\sigma_i^2$ (the parameters of the normal distribution).
The parameters $\mu_i$ and $\sigma_i$ are what we will optimize.

Note the difference: $\boldsymbol{\theta}$ is a concrete set of weights, i.e. one ordinary network. $q(\boldsymbol{\theta})$ is a distribution over such sets, and $\mu_i, \sigma_i$ are the parameters of that distribution.
A normal network learns $\boldsymbol{\theta}$ directly; a Bayesian network learns $\mu_i$ and $\sigma_i$. Every time we sample from $q$, we get one concrete network.

![Left: the learned q, with its own normal distribution on every edge. Right: one sample from q, which is an ordinary network](images/bnn_q_sample.svg)

Each edge has its own independent distribution (this is called the mean-field assumption). The true posterior also contains correlations between weights, e.g. if $\theta_1$ is large, $\theta_6$ might need to be small to produce the same output. Our $q$ ignores these correlations, which is one of the reasons it is only an approximation.
In the end, we approximate the posterior as follows:

$$
p(\boldsymbol{\theta} \mid \mathbf{x}) \approx q(\boldsymbol{\theta})
$$

More intuitively, we search for the best approximation in the space of our possible posteriors according to some distance as 
visualized in the figure below.

![The variational family Q contains every q we can express; VI moves from an initial guess to the member q* closest to the true posterior, 
  which lies outside Q](images/variational_family.svg)

## Evidence Lower Bound (ELBO)

Now to compare our two distributions, we can use the KL-divergence, which is a distance measure between two probability distributions.
In our example, it is defined as:

$$
D_\text{KL}\big(q \parallel p\big) = \mathbb{E}_{\boldsymbol{\Theta} \sim q} \left[ \log \frac{q(\boldsymbol{\Theta})}{p(\boldsymbol{\Theta} \mid \mathbf{x})} \right]
$$

Here the expectation is over the random weights $\boldsymbol{\Theta}$ drawn from $q$, which is why they are written in uppercase inside it.
However, we cannot use this as an objective, since it would require us to calculate $p(\boldsymbol{\theta} \mid \mathbf{x})$, which is exactly what we could not compute since we cannot compute $p(\mathbf{x})$:
 
$$
p(\boldsymbol{\theta} \mid \mathbf{x}) = \frac{p(\boldsymbol{\theta}, \mathbf{x})}{\int_{\mathbb{R}^n} p(\boldsymbol{\theta}, \mathbf{x})\; d\boldsymbol{\theta}}
$$

Through rearanging the Kullback-Leibler (KL) divergence (not shown for simplicity), it can be written as follows:

$$
D_\text{KL}\big(q \parallel p\big) = \mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log q(\boldsymbol{\Theta})\big] - \mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log p(\boldsymbol{\Theta}, \mathbf{x})\big] + \log p(\mathbf{x})
$$

This solving for $\log p(\mathbf{x})$ we get the following:

$$
\log p(\mathbf{x}) = \underbrace{\mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log p(\boldsymbol{\Theta}, \mathbf{x})\big] - \mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log q(\boldsymbol{\Theta})\big]}_{\text{ELBO}} + D_\text{KL}\big(q \parallel p\big)
$$

What this means: $\log p(\mathbf{x})$ does not depend on our approximation of the posterior, so its constant with regard to our optimization. 
Since it is a sum of the ELBO plus the KL divergence, if the ELBO increases, the KL divergence needs to decrease such that we get the same $\log p(\mathbf{x})$.
Therefore we can simply maximize the ELBO in order to minimize the KL-divergence.



### ELBO interpretation

Now, the above version of the ELBO is a bit hard to interpret. However, it turns out that it can be rearranged into a more intuitive form by a few steps.
With the chain rule, we can split the joint probability of $\Theta$ and $x$ and due to the log product rule, this becomes a sum:

$$
\tag{1}
\text{ELBO} = \mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log p(\mathbf{x} \mid \boldsymbol{\Theta}) + \log p(\boldsymbol{\Theta})\big] -
\mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log q(\boldsymbol{\Theta})\big]
$$

Since the sum of expecations is equal to the expecation of the sum, we can split the expecations:

$$
\tag{2}
\text{ELBO} = \mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log p(\mathbf{x} \mid \boldsymbol{\Theta})\big] + \mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log p(\boldsymbol{\Theta})\big] -
\mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log q(\boldsymbol{\Theta})\big]
$$

Now we have a difference between $p(\Theta)$ and $q(\Theta)$ in log space. This means we can apply the quotient rule to get a ratio of these.
Looking back we can see that the expecation of a ratio of two distributions is exactly a KL-divergence, but
the distribution we average over needs to be in the numerator. We can achieve this by negating and the quotient rule
since $\log\frac{p}{q} = \log p - \log q = -\log q + \log p = - (\log q - \log p) = -\log\frac{q}{p}$.

$$
\tag{3}
\mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log p(\boldsymbol{\Theta}) - \log q(\boldsymbol{\Theta})\big] = - \mathbb{E}_{\boldsymbol{\Theta} \sim q}\left[\log
\frac{q(\boldsymbol{\Theta})}{p(\boldsymbol{\Theta})}\right] = - D_\text{KL}\big(q(\boldsymbol{\Theta}) \parallel p(\boldsymbol{\Theta})\big)
$$

Now the ELBO becomes a difference of the probability of the data given the parameters and the KL-divergence of
our approximation of the posterior $q(\Theta)$ and the prior on the parameters $p(\Theta)$. 

$$
\tag{4}
\text{ELBO} = \underbrace{\mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log p(\mathbf{x} \mid \boldsymbol{\Theta})\big]}_{\text{reconstruction}} 
- \underbrace{D_\text{KL}\big(q(\boldsymbol{\Theta}) \parallel p(\boldsymbol{\Theta})\big)}_{\text{regularization}}
$$

This is generally interpreted as 
a objective with two components: reconstruction and regularization.

* **Reconstruction**: Rises with the likelihood of the data given the parameters. So the likelier our data given our model parameters, the higher this value.
* **Regularization**: Penalizes if our posterior approximation strays to far from our prior distribution. So this is a regularization.


## The Mean-Field Assumption
Remember that we try to model the true posterior using a simpler drop-in $q$. In principle, there could be covariance between our $\theta_i$ in the true posterior. 
However, the expectations in the ELBO require us to integrate over possible $\Theta$, which results in the same infeasible integral as before.
With the mean-field assumption, we just say we can approximate the true posterior sufficiently by assuming that all our parameters $\theta_i$
are independently distributed. This means the joint distribution becomes factorizable, since independent probabilities can be multiplied to 
get the joint probability:

$$
q(\theta_1, \theta_2, \dots, \theta_n) = q(\theta_1) \times q(\theta_2) \times \dots \times q(\theta_n)
$$

Due to this, the expectation can be taken for each variable separately:

$$
\mathbb{E}_{\boldsymbol{\Theta} \sim q}\big[\log q(\boldsymbol{\Theta})\big] = \mathbb{E}_{\boldsymbol{\Theta} \sim q}\Big[\sum_{i=1}^{n} \log q(\Theta_i)\Big] = \sum_{i=1}^{n} \mathbb{E}_{\Theta_i \sim q}\big[\log q(\Theta_i)\big]
$$

Since the ELBO requires the expectation of the log, the product becomes a sum. An expectation of a sum is equal to the sum of expecations.
So we can simply calculate the expectation for each weight separately, which we know how to, since 
each weight is normally distributed.

$$\mathbb{E}_{\boldsymbol{\Theta} \sim q}[\log p(\Theta, \mathbf{x})] = \int q(\theta),\log p(\theta, \mathbf{x}),d\theta$$

In the next part we will go over the Pyro fundamentals using this BNN as an example.
