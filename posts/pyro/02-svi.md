---
title: Variational Inference
date: 2026-10-13
summary: Why exact Bayesian inference is intractable, and how variational inference and the evidence lower bound (ELBO) turn it into an optimization problem.
tags: variational inference, elbo, bayesian inference, bayesian neural networks
draft: true
---

## Integrating is sometimes hard
In order to understand stochastic variational inference (SVI) and the evidence lower bound (ELBO) we first need to understand why these tools are actually needed.
Coming from a deep learning background I feel like an intuitive way to show this is using a simple Bayesian neural network (BNN) 
with weights $\boldsymbol{\Theta}$ and data $\mathbf{X}$.
We use uppercase letters for random variables and lowercase letters for their concrete values:
the weights are a random vector $\boldsymbol{\Theta} = (\Theta_1, \dots, \Theta_n)$, and $\boldsymbol{\theta} = (\theta_1, \dots, \theta_n)$ is one particular setting of them.
Likewise, $\mathbf{X}$ is the data as a random variable and $\mathbf{x}$ is the dataset we actually observed.
When we write densities such as $\cpost{p(\boldsymbol{\theta} \mid \mathbf{x})}$, this is short for the density of $\boldsymbol{\Theta}$ given $\mathbf{X} = \mathbf{x}$, evaluated at $\boldsymbol{\theta}$.

![Example Bayesian neural network with 2 inputs, 5 hidden units and 15 weights, each with the same prior N(0, 1); the 6 biases are not shown](images/bnn_network.svg)

A neural network becomes Bayesian if the weights are distributions instead of point estimates, such that
we get a distribution over networks that we could sample from or such that we could model uncertainty better.
We can write down the posterior (our belief about the weights after observing the data) easily using Bayes' rule:

$$
\cpost{p(\boldsymbol{\theta} \mid \mathbf{x})} = \frac{\cprior{p(\boldsymbol{\theta})}\, \clik{p(\mathbf{x} \mid \boldsymbol{\theta})}}{\cev{p(\mathbf{x})}}
$$

However, we cannot directly calculate this posterior, since it would require us to integrate over all possible weights to
calculate $\cev{p(\mathbf{x})}$:

$$
\cev{p(\mathbf{x})} = \int_{\mathbb{R}^n} p(\boldsymbol{\theta}, \mathbf{x})\; d\boldsymbol{\theta} = \int_{\mathbb{R}^n} \cprior{p(\boldsymbol{\theta})}\; \clik{p(\mathbf{x} \mid \boldsymbol{\theta})}\; d\boldsymbol{\theta}
$$


Why this is hard: If there is no closed-form solution for an integral (like here), we have to evaluate the function we want to integrate at certain points and then 
add the slices we get together to approximate the integral. However, if we want to evaluate with the same precision in higher dimensional functions, we need exponentially
many evaluations of the function. So if we evaluate $m = 100$ points per weight for the $n = 21$ weights of our network (15 connections plus 6 biases), we get $m^n = 100^{21} = 10^{42}$ evaluations, which is not feasible. If 
we assume our hardware can perform $10^{11}$ evaluations per second, it would take us around $3 \times 10^{23}$ years, more than $10^{13}$ times the age of the universe.

![Grid points needed to integrate over 1, 2 and 3 weights at the same spacing](images/curse_of_dimensionality.svg)

## Estimating Posteriors
Now, there are two common ways to model the posterior when we cannot calculate the required probabilities by integration or closed-form.
The first is sampling, where we estimate the posterior by sampling parameters for our posterior model (e.g., importance sampling, rejection sampling, Markov chain Monte Carlo (MCMC), ...).
The second is posing the problem as an optimization and using optimizers to find a good approximation for the posterior (variational inference, VI).
We will not go into depth on the difference of these approaches, but generally sampling becomes very expensive with big datasets, since
sampled parameters require calculation of the likelihood using the complete dataset. However, there is a trade-off: sampling (MCMC) is asymptotically exact, which
is not the case for SVI, but we can estimate gradients using a subset of the data easily, making it more scalable to large datasets.


Since we cannot integrate, we need a model of the posterior that we can optimize using some objective.
Note that our unobserved (latent) variables are the weights $\boldsymbol{\Theta}$.
Therefore, we approximate the joint posterior $\cpost{p(\boldsymbol{\theta} \mid \mathbf{x})}$ over all weights.
Our estimate for the posterior we will call $\cq{q(\boldsymbol{\theta})}$ from here on out. Unlike the prior, we model each weight $\Theta_i$ with its own normal distribution (its own mean and variance),
all independent of each other:

$$
\cq{q(\boldsymbol{\theta})} = \prod_{i=1}^{n} \mathcal{N}(\theta_i;\, \cpar{\mu_i}, \cpar{\sigma_i^2})
$$

$\mathcal{N}(\theta_i;\, \cpar{\mu_i}, \cpar{\sigma_i^2})$ just states that we evaluate the normal density at value $\theta_i$ with mean $\cpar{\mu_i}$ and variance $\cpar{\sigma_i^2}$ (the parameters of the normal distribution).
The parameters $\cpar{\mu_i}$ and $\cpar{\sigma_i}$ are what we will optimize.

Note the difference: $\boldsymbol{\theta}$ is a concrete set of weights, i.e. one ordinary network. $\cq{q(\boldsymbol{\theta})}$ is a distribution over such sets, and $\cpar{\mu_i}, \cpar{\sigma_i}$ are the parameters of that distribution.
A normal network learns $\boldsymbol{\theta}$ directly, while a Bayesian network learns $\cpar{\mu_i}$ and $\cpar{\sigma_i}$. Every time we sample from $\cq{q}$, we get one concrete network.

![Left: the learned q, with its own normal distribution on every edge. Right: one sample from q, which is an ordinary network](images/bnn_q_sample.svg)

Each edge has its own independent distribution (this is called the mean-field assumption). The true posterior also contains correlations between weights, e.g. if $\theta_1$ is large, $\theta_6$ might need to be small to produce the same output. Our $\cq{q}$ ignores these correlations, which is one of the reasons it is only an approximation.
In the end, we approximate the posterior as follows:

$$
\cpost{p(\boldsymbol{\theta} \mid \mathbf{x})} \approx \cq{q(\boldsymbol{\theta})}
$$

More intuitively, we search for the best approximation in the space of our possible posteriors to some measure of difference as 
visualized in the figure below.

![The variational family Q contains every q we can express; VI moves from an initial guess to the member q* closest to the true posterior, 
  which lies outside Q](images/variational_family.svg)

## Evidence Lower Bound (ELBO)

Now to compare our two distributions, we can use the Kullback-Leibler (KL) divergence, which measures how different two probability distributions are.
In our example, it is defined as:

$$
D_\text{KL}\big(\cq{q} \parallel \cpost{p}\big) = \mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}} \left[ \log \frac{\cq{q(\boldsymbol{\Theta})}}{\cpost{p(\boldsymbol{\Theta} \mid \mathbf{x})}} \right]
$$

Here the expectation is over the random weights $\boldsymbol{\Theta}$ drawn from $\cq{q}$, which is why they are written in uppercase inside it.
However, we cannot use this as an objective, since it would require us to calculate $\cpost{p(\boldsymbol{\theta} \mid \mathbf{x})}$, which is exactly what we could not compute since we cannot compute $\cev{p(\mathbf{x})}$:
 
$$
\cpost{p(\boldsymbol{\theta} \mid \mathbf{x})} = \frac{p(\boldsymbol{\theta}, \mathbf{x})}{\cev{\int_{\mathbb{R}^n} p(\boldsymbol{\theta}, \mathbf{x})\; d\boldsymbol{\theta}}}
$$

Through rearranging the KL divergence (not shown for simplicity), it can be written as follows:

$$
D_\text{KL}\big(\cq{q} \parallel \cpost{p}\big) = \mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \cq{q(\boldsymbol{\Theta})}\big] - \mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log p(\boldsymbol{\Theta}, \mathbf{x})\big] + \cev{\log p(\mathbf{x})}
$$

Thus, solving for $\cev{\log p(\mathbf{x})}$ we get the following:

$$
\cev{\log p(\mathbf{x})} = \underbrace{\mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log p(\boldsymbol{\Theta}, \mathbf{x})\big] - \mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \cq{q(\boldsymbol{\Theta})}\big]}_{\text{ELBO}} + D_\text{KL}\big(\cq{q} \parallel \cpost{p}\big)
$$

What this means: $\cev{\log p(\mathbf{x})}$ does not depend on our approximation of the posterior, so it's constant with regard to our optimization. 
Since it is a sum of the ELBO plus the KL divergence, if the ELBO increases, the KL divergence needs to decrease such that we get the same $\cev{\log p(\mathbf{x})}$.
Therefore we can simply maximize the ELBO in order to minimize the KL divergence.

Additionally, we know that the KL divergence cannot be negative, which means that $\cev{\log p(\mathbf{x})}$ is at least
the ELBO (when $D_\text{KL} = 0$), thus the name evidence *lower bound*. It describes a lower bound on the log-evidence $\cev{\log p(\mathbf{x})}$.
Taking $e^{\text{ELBO}}$  would give us the minimum probability density of the evidence.

### ELBO interpretation

Now, the above version of the ELBO is a bit hard to interpret. However, it turns out that it can be rearranged into a more intuitive form by a few steps.
With the chain rule, we can split the joint probability of $\boldsymbol{\Theta}$ and $\mathbf{x}$ and due to the log product rule, this becomes a sum:

$$
\tag{1}
\text{ELBO} = \mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \clik{p(\mathbf{x} \mid \boldsymbol{\Theta})} + \log \cprior{p(\boldsymbol{\Theta})}\big] -
\mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \cq{q(\boldsymbol{\Theta})}\big]
$$

Since the sum of expectations is equal to the expectation of the sum, we can split the expectations:

$$
\tag{2}
\text{ELBO} = \mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \clik{p(\mathbf{x} \mid \boldsymbol{\Theta})}\big] + \mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \cprior{p(\boldsymbol{\Theta})}\big] -
\mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \cq{q(\boldsymbol{\Theta})}\big]
$$

Now we have a difference between $\cprior{p(\boldsymbol{\Theta})}$ and $\cq{q(\boldsymbol{\Theta})}$ in log space. This means we can apply the log quotient rule to get a ratio of these.
Looking back we can see that the expectation of a log-ratio of two distributions is exactly a KL divergence, but
the distribution we average over needs to be in the numerator. We can achieve this by negating and applying the log quotient rule
since $\log\frac{\cprior{p}}{\cq{q}} = \log \cprior{p} - \log \cq{q} = -\log \cq{q} + \log \cprior{p} = - (\log \cq{q} - \log \cprior{p}) = -\log\frac{\cq{q}}{\cprior{p}}$.

$$
\tag{3}
\mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \cprior{p(\boldsymbol{\Theta})} - \log \cq{q(\boldsymbol{\Theta})}\big] = - \mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\left[\log
\frac{\cq{q(\boldsymbol{\Theta})}}{\cprior{p(\boldsymbol{\Theta})}}\right] = - D_\text{KL}\big(\cq{q(\boldsymbol{\Theta})} \parallel \cprior{p(\boldsymbol{\Theta})}\big)
$$

Now the ELBO becomes a difference of the expected log-likelihood of the data given the parameters and the KL divergence of
our approximation of the posterior $\cq{q(\boldsymbol{\Theta})}$ and the prior on the parameters $\cprior{p(\boldsymbol{\Theta})}$. 

$$
\tag{4}
\text{ELBO} = \underbrace{\mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \clik{p(\mathbf{x} \mid \boldsymbol{\Theta})}\big]}_{\clik{\text{reconstruction}}} 
- \underbrace{D_\text{KL}\big(\cq{q(\boldsymbol{\Theta})} \parallel \cprior{p(\boldsymbol{\Theta})}\big)}_{\cprior{\text{regularization}}}
$$

This is generally interpreted as 
an objective with two components: reconstruction and regularization.

* **Reconstruction**: Rises with the likelihood of the data given the parameters. So the likelier our data given our model parameters, the higher this value.
* **Regularization**: Penalizes if our posterior approximation strays too far from our prior distribution.


## The Mean-Field Assumption
We try to model the true posterior using a simpler drop-in $\cq{q}$. In principle, there could be covariance between our $\theta_i$ in the true posterior. 
However, that would mean that the number of parameters of our model goes from $2n$ parameters (one $\cpar{\mu}$ and $\cpar{\sigma}$ per weight) to $n + n(n+1)/2$ parameters (one $\cpar{\mu}$ per weight and one symmetric $n \times n$ covariance matrix).
With the mean-field assumption, we just say we can approximate the true posterior sufficiently by assuming that all our parameters $\theta_i$
are independently distributed. This means the joint distribution becomes factorizable, since independent probabilities can be multiplied to 
get the joint probability:

$$
\cq{q(\theta_1, \theta_2, \dots, \theta_n)} = q(\theta_1) \times q(\theta_2) \times \dots \times q(\theta_n)
$$

Due to this, the expectation can be taken for each variable separately:

$$
\mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \cq{q(\boldsymbol{\Theta})}\big] = \mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\Big[\sum_{i=1}^{n} \log \cq{q(\Theta_i)}\Big] = \sum_{i=1}^{n} \mathbb{E}_{\Theta_i \sim \cq{q}}\big[\log \cq{q(\Theta_i)}\big]
$$

Since the ELBO requires the expectation of the log, the product becomes a sum. An expectation of a sum is equal to the sum of expectations.
So we can simply calculate the expectation for each weight separately, which we know how to do, since 
each weight is normally distributed. 
While we could calculate $\mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \cq{q(\boldsymbol{\Theta})}\big]$ in closed-form for a multivariate normal on the weights, 
it becomes more computationally and memory expensive (e.g., $n \times n$ covariance matrix), which is why the mean-field assumption is often used in practice.

## The Reparameterization Trick
There is still a problem when optimizing the ELBO with the current formulation. That is, in order to perform gradient ascent, we need to
calculate the gradient of the objective (ELBO) with respect to our parameters. For the regularising part of the ELBO ($D_\text{KL}\big(\cq{q(\boldsymbol{\Theta})} \parallel \cprior{p(\boldsymbol{\Theta})}\big)$), this is easy, since we can calculate 
the expectations in closed-form. However, in a model like a neural network, the expected log-likelihood $\mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \clik{p(\mathbf{x} \mid \boldsymbol{\Theta})}\big]$ does not have a known closed-form solution.
In principle, we could approximate the integral using a grid like described earlier, but then we have the curse of dimensionality again that makes this infeasible.
Also, it would require us to evaluate the network at each grid point, which becomes extremely expensive for large networks.
So the problematic expectation is this one:

$$
\nabla_{\cpar{\mu},\cpar{\sigma}}\, \mathbb{E}_{\boldsymbol{\Theta} \sim \cq{q}}\big[\log \clik{p(\mathbf{x} \mid \boldsymbol{\Theta})}\big]
$$

We've established that we cannot calculate the expectation in closed-form. An easy way to calculate it is simply sampling weights $\boldsymbol{\theta}$ from $\cq{q}$, evaluating $\log \clik{p(\mathbf{x} \mid \boldsymbol{\theta})}$ for each sample and
then taking the average. However, sampling is a problem, since we cannot differentiate through it, so we cannot calculate the gradient anymore.
Think of a discrete distribution over classes *apple* and *pear*. While we might have a probability for each of the classes,
when sampling we simply pick one at random according to these probabilities. This means the outcome as a function of the probabilities is a step function, which has a derivative of 0 almost everywhere and is undefined at the jumps.

Now there is a neat trick that people use to circumvent this: A standard normal $\mathcal{N}(0, 1)$ scaled by $\cpar{\sigma_i}$ and shifted by $\cpar{\mu_i}$
will be distributed $\mathcal{N}(\cpar{\mu_i}, \cpar{\sigma_i^2})$. Leveraging this property is called the reparameterization trick. 
So the below sampling step is not differentiable:

$$
\theta_i \sim \mathcal{N}(\cpar{\mu_i}, \cpar{\sigma_i^2})
$$

Instead, we can draw a sample $\epsilon$ from a standard normal and scale and shift it using our $\cpar{\sigma_i}$ and $\cpar{\mu_i}$:

$$
\theta_i = \cpar{\mu_i} + \cpar{\sigma_i} \epsilon
$$

Since we differentiate with respect to our parameters, epsilon is considered fixed and we don't need to differentiate through any sampling.



