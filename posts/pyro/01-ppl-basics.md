---
title: A brief introduction to Probabilistic Programming in Pyro 
date: 2026-10-13
summary: This article is intended as my way to learn pyro, though it may be useful for someone else.
tags: stats, svi, probabilities, pyro
draft: true
---


Initially the term "probabilistic programming" sounds vague, at least that was my experience.
This article was intended as my way to learn the fundamentals of Pyro
by documenting my understanding along the process and coming up with examples for myself.
Why do we need a programming language for probabilities, can we not just do this in standard 
languages already? It only started to make sense when I thought 
about it as a declarative language (think Prolog), that lets you specify probabilistic processes and models
and allows you to perform inference and optimization automatically. 

In essence, it just gives you the 
declarative tools for distributions, sampling procedures, evidence etc. and standard tools for optimization, 
such that you can run arbitrary probabilistic models. In Prolog, we would define facts and rules, in Pyro we define a probabilistic process. 
Likewise, when running a query in Prolog, the engine performs backtracking automatically (thus inferring the solution). The equivalent in Pyro is the inference via e.g., stochastic variational inference (SVI). 


Consider the following example (fictional): You want to infer whether you share a common ancestor with Thomas Bayes. However, you are unsure if your father (William Bayes) is descended from one of Thomas Bayes' brothers, or from another Bayes family. Still, you have estimates for how likely the relationship is: Let’s assume Nathaniel was one of only two men named Bayes of fathering age in the area where William grew up, the other belonging to an unrelated Bayes family. Refer to Figure (a) for an overview.

William naming his eldest son Nathaniel shifts the odds from 1:1 to 1.5:1, giving a 0.6 probability that Nathaniel was William's father. Furthermore, a DNA test shows that you share a stretch of DNA with a known descendant of Joshua Bayes, which happens with probability 0.3 if you are related to him and 0.01 if you are not (for illustration, not realistic). This results in the probabilistic model depicted in (b).

![Fictional family tree linking you to Thomas Bayes](images/bayes_family_tree.svg)

You might define the prolog program as follows to get an answer to your question. Prolog would then backtrack to find you the solution that satisfies all constraints.


#### Prolog Model
```prolog
world(nathaniel_bayes).
world(stranger).

parent(_, joshua_bayes, thomas_bayes).
parent(_, joshua_bayes, nathaniel_bayes).
parent(W, W, william).
parent(_, william, you).

ancestor(W, A, X) :- parent(W, A, X).
ancestor(W, A, X) :- parent(W, P, X), ancestor(W, A, P).

related(W, X, Y) :- ancestor(W, A, X), ancestor(W, A, Y).

observed(dna_match).

consistent(W) :- world(W), ( observed(dna_match) -> once(related(W, you, thomas_bayes)) ; true ).
```

However, as you noticed, in plain prolog, we cannot infer how probable each scenario is exactly, since we can only work with facts. 
We simply drop the probabilities and treat the DNA match as a hard fact, therefore prolog will respond with `W = nathaniel_bayes` for the query `consistent(W)`.


In Pyro, we would instead define the probabilistic model below. Here we simply model the generative process by sampling that Nathaniel is William's father with 60% probability.
We then say the probability of a match depends on whether Nathaniel is the father (`torch.where`), using the sampled fact to select this probability.
We then provide our observation that there was in fact a DNA-match observed using `pyro.sample`.
Calling `model()` will result in one sample from the model.
The `guide` here is our current estimate of the posterior, i.e.,

$$P(\text{Nathaniel is William's father} \mid \text{DNA match})$$

Using SVI, we iteratively update the guide to maximize the evidence lower bound (ELBO), which moves it closer to the true posterior. In this example we could calculate the solution
analytically, but this is not possible in some complex models. Note that here we explicitly define how we infer
our posterior as opposed to implicitly backtracking in Prolog.

#### Pyro Model
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

def guide():
    q = pyro.param("q", torch.tensor(0.5), constraint=constraints.unit_interval)
    pyro.sample("nathaniel_is_father", dist.Bernoulli(q))

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

Although it would be possible to get the exact solution with Pyro, the example shows standard SVI for clarity.
Here, we iteratively step towards better posteriors (q) using gradient descent with an Adam optimizer. Due to noisy gradients,
our estimate for the posterior will still be an approximation and not exact. However, for reference, 
we can calculate the exact solution by applying Bayes' rule as follows.

First, we calculate the marginal probability of a match (the evidence) by summing over the hypotheses $h \in \{\text{Nathaniel}, \text{stranger}\}$ for William's father.

$$
P(\text{match}) = \sum_h P(h)\, P(\text{match} \mid h) = 0.6 \cdot 0.3 + 0.4 \cdot 0.01 = 0.184
$$

Then we get the posterior by plugging everything into Bayes' rule.

$$
P(\text{Nathaniel} \mid \text{match}) = \frac{P(\text{Nathaniel})\, P(\text{match} \mid \text{Nathaniel})}{P(\text{match})} = \frac{0.18}{0.184} \approx 0.978.
$$


We can see that our approximation using SVI is in the ballpark, but not terribly accurate. There would be ways to mitigate this, 
but we wont consider them here for simplicity. In summary, we can think of Pyro as a declarative language, similar to Prolog, 
that allows us to reason about uncertainty and gives us tools for inference.


