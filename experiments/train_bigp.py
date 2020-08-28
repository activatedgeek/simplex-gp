import os
import torch
from torch.utils.tensorboard import SummaryWriter
import gpytorch as gp
from tqdm.auto import tqdm
import wandb
from pathlib import Path

from bi_gp.bilateral_kernel import BilateralKernel
from utils import set_seeds, standardize, UCIDataset


class BilateralGPModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gp.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(BilateralKernel(ard_num_dims=train_x.size(-1)))

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


def main(dataset=None, data_dir=None,
         epochs=100, lr=0.01,
         log_int=1, seed=None):
    if data_dir is None and os.environ.get('DATADIR') is not None:
        data_dir = Path(os.path.join(os.environ.get('DATADIR'), 'uci'))

    assert dataset is not None, f'Select a dataset from "{data_dir}"'

    set_seeds(seed)

    wandb.init(tensorboard=True)

    ## Disable GPU for now.
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    logger = SummaryWriter(log_dir=wandb.run.dir)

    train_dataset = UCIDataset.create(dataset, uci_data_dir=data_dir,
                                      mode="train", device=device)
    test_dataset = UCIDataset.create(dataset, uci_data_dir=data_dir,
                                     mode="test", device=device)

    train_x, train_y = train_dataset.x, train_dataset.y
    test_x, test_y = test_dataset.x, test_dataset.y
    train_x, train_y, test_x, test_y = standardize(train_x, train_y, test_x, test_y)

    print(f'"{dataset}": D = {train_x.size(-1)}, Train N = {train_x.size(0)}, Test N = {test_x.size(0)}')

    model = BilateralGPModel(train_x, train_y).to(device)
    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in tqdm(range(epochs)):
        train_dict = train(train_x, train_y, model, mll, optimizer)
        for k, v in train_dict.items():
            logger.add_scalar(k, v, global_step=i + 1)
        
        if (i % log_int) == 0:
          test_dict = test(test_x, test_y, model, mll)
          for k, v in test_dict.items():
              logger.add_scalar(k, v, global_step=i + 1)

    test_dict = test(test_x, test_y, model, mll)
    for k, v in test_dict.items():
        logger.add_scalar(k, v, global_step=i + 1)

if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  import fire
  fire.Fire(main)
