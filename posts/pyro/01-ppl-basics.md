---
title: What is Probabilistic Programming?
date: 2026-10-13
summary: This article is intended as my way to learn Pyro, though it may be useful for someone else.
tags: pyro, probabilistic programming, bayesian inference
draft: true
---

## Why a language for probabilities?

Initially the term "probabilistic programming" sounds vague, at least that was my experience.
Why do we need a programming language for probabilities, can we not just do this in standard 
languages already? It only started to make sense when I thought 
about it as a declarative language (think Prolog), that lets you specify probabilistic processes and models
and allows you to perform inference and optimization automatically. 

In essence, it just gives you the 
declarative tools for distributions, sampling procedures, evidence etc. and standard tools for optimization, 
such that you can run arbitrary probabilistic models. In Prolog, we would define facts and rules: a deterministic model,
while in Pyro we define a probabilistic model.
Likewise, when running a query in Prolog, the engine performs backtracking automatically (thus inferring the solution). The equivalent in Pyro is the inference via, e.g., stochastic variational inference (SVI). 

This article is my way of learning the fundamentals of Pyro,
by documenting my understanding along the process and coming up with examples for myself.

## Example: Are you related to Thomas Bayes?

Consider the following example (fictional): You want to infer whether you share a common ancestor with Thomas Bayes. However, you are unsure if your father (William Bayes) is the son of Thomas Bayes' brother Nathaniel, or descended from another Bayes family. Still, you have estimates for how likely the relationship is: Let’s assume Nathaniel was one of only two men named Bayes of fathering age in the area where William grew up, the other belonging to an unrelated Bayes family. Refer to Figure (a) below for an overview.

![Fictional family tree linking you to Thomas Bayes](images/bayes_family_tree.svg)

William named his eldest son Nathaniel, which hints at a family connection and shifts the odds from 1:1 to 1.5:1. Furthermore, a DNA test shows that you share a stretch of DNA with a known descendant of Joshua Bayes. This results in the probabilistic model depicted in (b), with the following numbers (for illustration, not realistic):

- **Prior:** $\cprior{P(\text{Nathaniel is William's father})} = 0.6$
- **Match if related:** $\clik{P(\text{DNA match} \mid \text{related})} = 0.3$
- **Match if not related:** $\clik{P(\text{DNA match} \mid \text{not related})} = 0.01$

## Attempt 1: Prolog

You might define the Prolog program as follows to get an answer to your question. Prolog would then backtrack to find you the solution that satisfies all constraints.
We can start by declaring the two possible outcomes (worlds): Nathaniel is William's father or he is a stranger.

```prolog
world(nathaniel_bayes).
world(stranger).
```

We define the predicates that hold in our example, that is we know that Thomas and Nathaniel are siblings in any world (_), that William is your father in any world
and that there are different possible fathers for William (worlds `W`).

```prolog
parent(_, joshua_bayes, thomas_bayes).
parent(_, joshua_bayes, nathaniel_bayes).
parent(_, william, you).
parent(W, W, william).
```

We can also define what an ancestor is. Here we say it's either a parent directly or recursively in the same world.

```prolog
ancestor(W, A, X) :- parent(W, A, X).
ancestor(W, A, X) :- parent(W, P, X), ancestor(W, A, P).
```

Being related means that two people are linked by an ancestor in the same world.

```prolog
related(W, X, Y) :- ancestor(W, A, X), ancestor(W, A, Y).
```

We also define that we observed a DNA match.

```prolog
observed(dna_match).
```

Now we can check which world is valid. Initially we need to define that `W` is actually a world.
Then we define a rule to decide what valid means.
Here `->` is indicating an if-then-else. So if we observe a dna_match, world `W` is valid
if in this world you are related to Thomas Bayes. In general, `related` succeeds once for every common ancestor it finds, i.e., possibly several times
per world; the `once` statement stops after the first proof, so each valid world is returned only once.

```prolog
valid(W) :- 
    world(W), 
    (observed(dna_match) -> once(related(W, you, thomas_bayes)) ; true ).
```

However, as you may have noticed, in plain Prolog, we cannot infer how probable each scenario is exactly, since we can only work with facts. 
We simply drop the probabilities and treat the DNA match as a hard fact, therefore Prolog will respond with `W = nathaniel_bayes` for the query `valid(W)`.

## Attempt 2: Pyro

In Pyro, we instead define a probabilistic model, a guide, and an inference procedure.

### The generative model

Here we simply model the generative process by sampling that Nathaniel is William's father with 60% probability.
We then say the probability of a match depends on whether Nathaniel is the father (`torch.where`), using the sampled fact to select this probability.
Finally, we provide our observation that there was in fact a DNA match observed using `pyro.sample` with `obs=...`.
Calling `model()` will result in one sample from the model.

```python fold-imports
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

def model():
    nathaniel = pyro.sample("nathaniel_is_father", dist.Bernoulli(0.6))
    p_match = torch.where(nathaniel.bool(), torch.tensor(0.3), torch.tensor(0.01))
    pyro.sample("dna_match", dist.Bernoulli(p_match), obs=torch.tensor(1.0))
```

### The guide

The `guide` is our current estimate of the posterior, i.e.,

$$\cpost{P(\text{Nathaniel is William's father} \mid \text{DNA match})}$$

It samples the same latent variable as the model (`nathaniel_is_father`), but with a learnable probability `q` instead of the prior.

```python
def guide():
    q = pyro.param("q", torch.tensor(0.5), constraint=constraints.unit_interval)
    pyro.sample("nathaniel_is_father", dist.Bernoulli(q))
```

### Inference with SVI

Using SVI, we iteratively update the guide to maximize the evidence lower bound (ELBO), which moves it closer to the true posterior.
Note that here we explicitly define how we infer our posterior as opposed to implicitly backtracking in Prolog.
In this example we could calculate the solution analytically, but this is not possible in some complex models.

```python
pyro.clear_param_store()
svi = SVI(model, guide, Adam({"lr": 0.01}), Trace_ELBO(max_plate_nesting=0))
for step in range(20_000):
    loss = svi.step()

print(f"P(Nathaniel is father | DNA match) = {pyro.param('q').item():.3f}")
```

#### Output
```bash
P(Nathaniel is father | DNA match) = 0.985
```

Although it would be possible to get the exact solution with Pyro, the example shows standard SVI for simplicity.
Here, we iteratively step towards better posteriors (q) using gradient descent with an Adam optimizer. Due to noisy gradients,
our estimate for the posterior will still be an approximation and not exact.

## Checking against the exact answer

For reference, we can calculate the exact solution by applying Bayes' rule.
First, we calculate the marginal probability of a match (the evidence) by summing over the hypotheses $h \in \{\text{Nathaniel}, \text{stranger}\}$ for William's father.

$$
\cev{P(\text{match})} = \sum_h \cprior{P(h)}\, \clik{P(\text{match} \mid h)} = \cprior{0.6} \cdot \clik{0.3} + \cprior{0.4} \cdot \clik{0.01} = 0.184
$$

Then we get the posterior by plugging everything into Bayes' rule.

$$
\cpost{P(\text{Nathaniel} \mid \text{match})} = \frac{\cprior{P(\text{Nathaniel})}\, \clik{P(\text{match} \mid \text{Nathaniel})}}{\cev{P(\text{match})}} = \frac{0.18}{\cev{0.184}} \approx 0.978.
$$

We can see that our approximation using SVI is in the ballpark, but not terribly accurate.
Since our guide (a Bernoulli with free parameter `q`) can represent the exact posterior, the gap is caused by noisy gradients.
There would be ways to mitigate this, but we won't consider them here for simplicity.

## Conclusion

We can think of Pyro as a declarative language, similar to Prolog, 
that allows us to reason about uncertainty and gives us tools for inference.
So far we treated SVI and the ELBO as a black box that moves the guide closer to the posterior.
In the next part we look at why we need them in the first place and what the ELBO actually optimizes.
