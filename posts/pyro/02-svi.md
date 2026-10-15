---
title: Stochastic Variational Inference and the Evidence Lower Bound
date: 2026-10-13
summary: This article is intended as my way to learn pyro, though it may be useful for someone else.
tags: stats, svi, probabilities, pyro
draft: true
---

## Integrating is sometimes hard
In order to understand SVI and ELBO we first need to understand why these tools are actually needed.
Consider the previous fictional example, we can calculate the posterior easily using the Bayes rule:

$$
P(\text{Nathaniel} \mid \text{match}) = \frac{P(\text{Nathaniel})\, P(\text{match} \mid \text{Nathaniel})}{P(\text{match})}
$$

However, let's assume the probability of having a match when Nathaniel is really Williams father depends on the fraction
of DNA you actually share with Thomas Bayes, and this fraction depends on the number of generations between you. 
Let's say the fractions are denoted as $a_i$ with $i$ indicating the generations away from Thomas Bayes. 
Again, this is fictional and only an example that is not realistic.
We can write the fractions as a vector:

$$\mathbf{a} = (a_1, \dots, a_n)$$

In our family tree, $n = 4$ generations separate you from Thomas Bayes (you → William → Nathaniel → Joshua → Thomas).

Further, lets say the $a_i$ are beta distributed with $\text{Beta}(\alpha, \alpha)$, an the distribution does not change through time.
To calculate $P(\text{match} \mid \text{Nathaniel})$ we now need to integrate over all possible $a_i$. So the fraction of shared DNA would be $s = \prod_{i=1}^{n}a_i$.
However, to model the probability of a match given a shared part, we need another function $g(\cdot)$. Using this we can say how fast the probability of
a detected match grows with the fraction of shared DNA. A simple choice could be an exponential saturation function $g(s) = 1 - e^{-k \cdot s}$ as shown below with a parameter $k$ that
controls how fast we approach a probability of 1.
With $\alpha = 10$ and $k \approx 6$, the average match probability stays at $P(\text{match} \mid \text{Nathaniel}) \approx 0.3$, as in the previous article.

![Match probability as a function of shared DNA fraction](images/match_probability.svg)


If we draw the fractions from a Beta distribution, the probability of an $\mathbf{a}$ vector would be:

$$
p(\mathbf{a}) = \prod_{i=1}^{n} \text{Beta}(a_i;\,\alpha,\alpha)
$$

The probability of a match given Nathaniel is the father and passed on DNA fractions is then:

$$
P(\text{match} \mid \text{Nathaniel}, \mathbf{a}) = g(s)
$$

To marginalize out $\mathbf{a}$ and get to $P(\text{match} \mid \text{Nathaniel})$ we need to calculate the following integral:

$$
P(\text{match} \mid \text{Nathaniel}) = \int_{[0, 1]^n} p(\mathbf{a})\; P(\text{match} \mid \text{Nathaniel}, \mathbf{a})\; d\mathbf{a}
$$

Why this is hard: If there is no closed-form solution for an integral, we have to evaluate the function we want to integrate at certain points and then 
add the slices we get together to approximate the interval. However, if we want to evaluate with the same precision in higher dimensional functions, we need exponentially
many evaluations of the function. So if we evaluate $m = 100$ points for $n = 10$ generations, we get $m^n = 10^{20}$ evaluations, which is not feasible. If 
we assume our hardware can perform $10^{11}$ evaluations per second, it would take us around 32 years for all evaluations.

![Grid points needed to integrate over 1, 2 and 3 generations at the same spacing](images/curse_of_dimensionality.svg)

## Stochastic Variational Inference

Now, there is two common ways to model the posterior when we cannot calculate the required probabilities by integration or closed-form.
The first is sampling, where we estimate the posterior by sampling parameters for our posterior model (e.g., importance sampling, rejection sampling, MCMC, ...).
The second is posing the problem as an optimization and using optimizers to find a good approximation for the posterior (variational inference).
We will not go into depth on the difference of these approaches, but generally sampling becomes very expensive with big datasets, since
sampled parameters require calculation of the likelihood using the complete dataset. However, there is a trade-off: sampling (MCMC) is asymptotically exact, which
is not the case for SVI, but we can estimate gradients using a subset of the data easily, making it more scalable to large datasets.


## Evidence Lower Bound
Since we cannot integrate, we need a model of the posterior that we can optimize using some objective.
Note that we now have two unobserved (latent) variables: whether Nathaniel is the father and the DNA fractions $\mathbf{a}$.
Therefore, we approximate the joint posterior $P(\text{Nathaniel}, \mathbf{a} \mid \text{match})$ instead of only $P(\text{Nathaniel} \mid \text{match})$.
Our estimate for the posterior we will call $q(\text{Nathaniel}, \mathbf{a})$ from here on out. Let's say we model Nathaniel with a Bernoulli distribution and the fractions $a_i$ with a single Beta distribution shared across generations,
all independent of each other:

$$
q(\text{Nathaniel}, \mathbf{a}) = \text{Bernoulli}(\text{Nathaniel}; w) \prod_{i=1}^{n} \text{Beta}(a_i;\, u, v)
$$

$\text{Bernoulli}(\text{Nathaniel}; w)$ just states that we evaluate the Bernoulli density at value Nathaniel with the Bernoulli probability $w$ (the parameter of the Bernoulli distribution).
Likewise, $\text{Beta}(a_i;\, u, v)$ evaluates the Beta density at $a_i$ with parameters $u$ and $v$. The parameters $w$, $u$ and $v$ are what we will optimize.
In the end, we are mostly interested in Nathaniel, so we approximate the posterior as follows:

$$
P(\text{Nathaniel}{=}1 \mid \text{match}) \approx w
$$

Now to compare our two distributions, we can use the KL-divergence, which is a distance measure between two probability distributions.
In our example, it is defined as:

$$
D_\text{KL}\big(q \parallel P\big) = \mathbb{E}_q \left[ \log \frac{q(\text{Nathaniel}, \mathbf{a})}{P(\text{Nathaniel}, \mathbf{a} \mid \text{match})} \right]
$$

However, we cannot use this as an objective, since it would require us to calculate $P(\text{Nathaniel}, \mathbf{a} \mid \text{match})$, which is exactly what we want to approximate:

$$
P(\text{Nathaniel}, \mathbf{a} \mid \text{match}) = \frac{P(\text{Nathaniel}, \mathbf{a}, \text{match})}{\sum_{\text{Nathaniel}} \int_{[0, 1]^n} P(\text{Nathaniel}, \mathbf{a}, \text{match})\; d\mathbf{a}}
$$

Through rearanging the Kullback-Leibler (KL) divergence (not shown for simplicity), it can be written as follows:

$$
D_\text{KL}\big(q \parallel P\big) = \mathbb{E}_q\big[\log q(\text{Nathaniel}, \mathbf{a})\big] - \mathbb{E}_q\big[\log P(\text{Nathaniel}, \mathbf{a}, \text{match})\big] + \log P(\text{match})
$$

This solving for $\log P(\text{match})$ we get the following:

$$
\log P(\text{match}) = \underbrace{\mathbb{E}_q\big[\log P(\text{Nathaniel}, \mathbf{a}, \text{match})\big] - \mathbb{E}_q\big[\log q(\text{Nathaniel}, \mathbf{a})\big]}_{\text{ELBO}} + D_\text{KL}\big(q \parallel P\big)
$$

What this means: $\log P(\text{match})$ does not depend on our approximation of the posterior, so its constant with regard to our optimization. 
Since it is a sum of the ELBO plus the KL divergence, if the ELBO increases, the KL divergence needs to decrease such that we get the same $\log P(\text{match})$.
Therefore we can simply maximize the ELBO in order to minimize the KL-divergence.


$$
P(\text{Nathaniel} \mid \text{match}) = \frac{P(\text{match} \mid \text{Nathaniel}), P(\text{Nathaniel})}{P(\text{match})}
$$

P(match | Nathaniel) is the integral from line 52. To get it, you average over a because you don't know it:

$$
P(\text{match} \mid \text{Nathaniel}) = \int p(\mathbf{a}), P(\text{match} \mid \text{Nathaniel}, \mathbf{a}), d\mathbf{a}
$$

Plugging in the two cases

- Nathaniel=0: P(match | 0, a) = 0.01 for every a, so the integral is just 0.01. Nothing hard here.
- Nathaniel=1: P(match | 1, a) = g(s), which depends on a. Write I = ∫ p(a) g(s) da. This is the n-dimensional integral that's hard to compute.

Then:

$$
P(\text{Nathaniel}{=}1 \mid \text{match}) = \frac{P(1), I}{P(1), I + P(0)\cdot 0.01}
$$

