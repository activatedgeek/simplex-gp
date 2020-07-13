import torch
import gpytorch
from bilateral_kernel import BilateralKernel
from utils import UCIDataset

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(BilateralKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

device = "cpu"
datasets = ["3droad", "elevators"]
train_datasets = UCIDataset.create(*datasets, mode="train", device=device, dtype=torch.float32)
test_datasets = UCIDataset.create(*datasets, mode="test", device=device, dtype=torch.float32)

for train_dataset, test_dataset in zip(train_datasets, test_datasets):
    train_x, train_y = train_dataset.x, train_dataset.y
    test_x, test_y = test_dataset.x, test_dataset.y
    print(f"Train: N={train_x.size(0)}  D={train_x.size(1)}")

    x_mean = train_x.mean(0)
    x_std = train_x.std(0) + 1e-6

    y_mean = train_y.mean(0)
    y_std = train_y.std(0) + 1e-6

    train_x = (train_x - x_mean) / x_std
    train_y = (train_y - y_mean) / y_std

    test_x = (test_x - x_mean) / x_std
    test_y = (test_y - y_mean) / y_std

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    model = model.to(device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 100
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_y = likelihood(model(test_x)).mean
        print(f"RMSE: {(pred_y - test_y).pow(2).mean(0).sqrt()}")
