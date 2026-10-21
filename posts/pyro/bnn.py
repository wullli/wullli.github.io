
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

def forward(x: torch.Tensor, 
            weights_hidden: torch.Tensor, 
            weights_out: torch.Tensor, 
            bias_h: torch.Tensor,
            bias_o: torch.Tensor):
    h = torch.matmul(x, weights_hidden) + bias_h
    h = torch.relu(h)
    y = torch.matmul(h, weights_out) + bias_o
    return y


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


def model(x: torch.Tensor, y: torch.Tensor | None = None):
    weights_hidden = pyro.sample("w_h", PRIOR.expand([IN_DIM, DIM]).to_event(2))
    weights_out = pyro.sample("w_o", PRIOR.expand([DIM, OUT_DIM]).to_event(2))
    bias_h = pyro.sample("b_h", PRIOR.expand([DIM]).to_event(1))
    bias_o = pyro.sample("b_o", PRIOR.expand([1]).to_event(1))

    logits = forward(x, weights_hidden, weights_out, bias_h, bias_o).flatten()
    with pyro.plate("data", x.shape[0]):
        pyro.deterministic("p", torch.sigmoid(logits))
        return pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)


def fit(x_train, y_train):
    x_train_tensor = torch.as_tensor(x_train).float()

    y_train_tensor = torch.as_tensor(y_train).float()

    svi = SVI(model, guide, Adam({"lr": 0.001}), Trace_ELBO())
    for step in range(10_000):
        loss = svi.step(x_train_tensor, y_train_tensor)

        if step % 1000 == 0:
            print(f"Loss in step {step}: {loss}")


def score(y_pred_p: torch.Tensor, y_test_tensor):
    y_pred = y_pred_p > 0.5
    return f1_score(y_test_tensor, y_pred.detach().numpy(), average='macro')


def evaluate(x_test, y_test):
    x_test_tensor = torch.as_tensor(x_test).float()
    y_test_tensor = torch.as_tensor(y_test).float().detach().numpy()

    predictive = Predictive(model, guide=guide, num_samples=500)(x_test_tensor, None)


    y_pred_scores = np.array([score(y_pred_p, y_test_tensor) for y_pred_p in predictive["p"]])

    ci_lower = np.percentile(y_pred_scores, 2.5)
    ci_upper = np.percentile(y_pred_scores, 97.5)

    mean_score = score(predictive["p"].mean(dim=0), y_test_tensor)
    return mean_score, (ci_lower, ci_upper) 


def main():
    pyro.set_rng_seed(0)
    x, y = fetch_openml("diabetes", version=1, return_X_y=True, as_frame=False)

    data: list[np.ndarray]
    data = train_test_split(x, y, random_state=0, test_size=0.3)
    x_train, x_test, y_train, y_test = data
    x_train, x_test = x_train[:, [1, 6]], x_test[:, [1, 6]]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    y_train = (y_train == "tested_positive").astype(int)
    y_test = (y_test == "tested_positive").astype(int)

    pyro.clear_param_store()
    guide(torch.as_tensor(x_train_scaled).float())  # creates w_h_loc, w_h_scale, w_o_

    f1, ci = evaluate(x_test_scaled, y_test)
    print(f"F1 score before: {f1}, CI: [{ci[0]}, {ci[1]}]")

    fit(x_train_scaled, y_train)

    f1, ci = evaluate(x_test_scaled, y_test)
    print(f"F1 score after: {f1}, CI: [{ci[0]}, {ci[1]}]")


if __name__ == "__main__":
    main()
