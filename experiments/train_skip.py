import os
import torch
from torch.utils.tensorboard import SummaryWriter
import gpytorch as gp
from tqdm.auto import tqdm
import wandb
from pathlib import Path

from utils import set_seeds, standardize, UCIDataset


class SKIPGPModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, grid_size=100):
        likelihood = gp.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ProductStructureKernel(
            gp.kernels.ScaleKernel(
                gp.kernels.GridInterpolationKernel(gp.kernels.RBFKernel(), grid_size=grid_size, num_dims=1)
            ), num_dims=train_x.size(-1)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


def train(x, y, model, mll, optim):
  model.train()

  optim.zero_grad()

  output = model(x)

  loss = -mll(output, y)

  loss.backward()

  optim.step()

  return {
    'train/ll': -loss.detach().item()
  }


def test(x, y, model, mll):
  model.eval()

  with torch.no_grad():
    preds = model(x)

  pred_y = model.likelihood(model(x))
  rmse = (pred_y.mean - y).pow(2).mean(0).sqrt()

  return {
    'test/rmse': rmse.item()
  }


def main(dataset: str = None, data_dir: str = None, lr: float = 0.1,
         lanc_iter: int = 30, epochs: int = 10, precon_size: int = 10,
         log_int: int = 1, seed: int = None):
    if data_dir is None and os.environ.get('DATADIR') is not None:
        data_dir = Path(os.path.join(os.environ.get('DATADIR'), 'uci'))

    assert dataset is not None, f'Select a dataset from "{data_dir}"'

    set_seeds(seed)

    wandb.init(tensorboard=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = SummaryWriter(log_dir=wandb.run.dir)

    train_dataset = UCIDataset.create(dataset, uci_data_dir=data_dir,
                                      mode="train", device=device)
    test_dataset = UCIDataset.create(dataset, uci_data_dir=data_dir,
                                     mode="test", device=device)

    train_x, train_y = train_dataset.x, train_dataset.y
    test_x, test_y = test_dataset.x, test_dataset.y
    train_x, train_y, test_x, test_y = standardize(train_x, train_y, test_x, test_y)

    print(f'"{dataset}": D = {train_x.size(-1)}, Train N = {train_x.size(0)}, Test N = {test_x.size(0)}')

    model = SKIPGPModel(train_x, train_y).to(device)
    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with gp.settings.use_toeplitz(True):
      for i in tqdm(range(epochs)):

          with gp.settings.use_toeplitz(False), \
               gp.settings.max_root_decomposition_size(lanc_iter):
            train_dict = train(train_x, train_y, model, mll, optimizer)

            for k, v in train_dict.items():
              logger.add_scalar(k, v, global_step=i + 1)

          with gp.settings.use_toeplitz(False), \
               gp.settings.max_preconditioner_size(precon_size), \
               gp.settings.max_root_decomposition_size(lanc_iter), \
               gp.settings.fast_pred_var():
            test_dict = test(test_x, test_y, model, mll)

            for k, v in test_dict.items():
              logger.add_scalar(k, v, global_step=i + 1)


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  import fire
  fire.Fire(main)
