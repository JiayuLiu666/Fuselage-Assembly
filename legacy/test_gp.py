import torch
from botorch.models import FixedNoiseGP

X = torch.rand(10, 2)
Y = torch.rand(10, 1)
Yvar = torch.full((10, 1), 1e-4)

model = FixedNoiseGP(X, Y, Yvar)
model.eval()

# Case 1: just call model
try:
    _ = model(X[:1])
    model.get_fantasy_model(X[:1], Y[:1], noise=torch.tensor([1e-4]))
    print("model(X) worked!")
except Exception as e:
    print(f"model(X) failed: {e}")

model2 = FixedNoiseGP(X, Y, Yvar)
model2.eval()

# Case 2: call posterior
try:
    _ = model2.posterior(X[:1])
    model2.get_fantasy_model(X[:1], Y[:1], noise=torch.tensor([1e-4]))
    print("model2.posterior(X) worked!")
except Exception as e:
    print(f"model2.posterior(X) failed: {e}")
