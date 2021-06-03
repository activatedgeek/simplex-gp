import os
import torch
import gpytorch as gp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from gpytorch_lattice_kernel import RBFLattice


class SimplexGPModel(gp.models.ExactGP):
  def __init__(self, train_x, train_y, order=1, min_noise=1e-4):
    likelihood = gp.likelihoods.GaussianLikelihood(
                  noise_constraint=gp.constraints.GreaterThan(min_noise))
    super().__init__(train_x, train_y, likelihood)
    self.mean_module = gp.means.ConstantMean()
    self.base_covar_module = RBFLattice(order=order)
    self.covar_module = gp.kernels.ScaleKernel(self.base_covar_module)

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gp.distributions.MultivariateNormal(mean_x, covar_x)


class ExactModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, min_noise=1e-4):
      # assert train_x.is_contiguous(), 'Need contiguous x for KeOps'

      likelihood = gp.likelihoods.GaussianLikelihood(
                    noise_constraint=gp.constraints.GreaterThan(min_noise))
      super().__init__(train_x, train_y, likelihood)
      self.mean_module = gp.means.ConstantMean()
      if torch.cuda.is_available():
        self.base_covar_module = gp.kernels.keops.RBFKernel()
      else:
        self.base_covar_module = gp.kernels.RBFKernel()
      self.covar_module = gp.kernels.ScaleKernel(self.base_covar_module)

    def forward(self, x):
      # assert x.is_contiguous(), 'Need contiguous x for KeOps'

      mean_x = self.mean_module(x)
      covar_x = self.covar_module(x)
      return gp.distributions.MultivariateNormal(mean_x, covar_x)


def train(x, y, model, mll, optim, lanc_iter=100, pre_size=100):
  model.train()

  optim.zero_grad()

  with gp.settings.cg_tolerance(1e-4), \
       gp.settings.max_preconditioner_size(pre_size), \
       gp.settings.max_root_decomposition_size(lanc_iter):
    output = model(x)
    loss = -mll(output, y)

    loss.backward()

    optim.step()

  return {
    'train/mll': -loss.detach().item(),
  }


def train_model(model_cls, device, x, y):
  model = model_cls(x, y).to(device)
  mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

  train_dict = None  
  for _ in tqdm(range(100)):
    train_dict = train(x, y, model, mll, optimizer)
  return train_dict


def main():
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  print(f'Using device {device}')

  df = pd.read_csv(f'{os.path.dirname(__file__)}/../notebooks/snelson.csv')
  train_x = torch.from_numpy(df.x.values[:, np.newaxis]).float().to(device).contiguous()
  train_y = torch.from_numpy(df.y.values).float().to(device).contiguous()
  print(train_x.shape, train_y.shape)

  sgp_mll = train_model(SimplexGPModel, device, train_x, train_y)['train/mll']
  keops_mll = train_model(ExactModel, device, train_x, train_y)['train/mll']
  
  delta = np.abs(sgp_mll - keops_mll)

  print(f'\nSimplex-GP MLL: {sgp_mll:.6f}\nKeOps MLL: {keops_mll:.6f}\nDelta: {delta:.6f}')
  
  ## Make sure this always gives a pre-defined result.
  assert np.abs(sgp_mll - keops_mll) < 0.1


if __name__ == '__main__':
  main()
