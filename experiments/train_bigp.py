import os
import torch
from torch.utils.tensorboard import SummaryWriter
import gpytorch as gp
from tqdm.auto import tqdm
import wandb
from pathlib import Path
from timeit import default_timer as timer

from bi_gp.bilateral_kernel import BilateralKernel
from utils import set_seeds, log_scalar_dict, standardize, UCIDataset


class BilateralGPModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gp.likelihoods.GaussianLikelihood(
                      noise_constraint=gp.constraints.GreaterThan(1e-2))
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(BilateralKernel(ard_num_dims=train_x.size(-1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


def train(x, y, model, mll, optim, lanc_iter=100):
  t_start = timer()

  model.train()

  optim.zero_grad()

  with gp.settings.max_root_decomposition_size(lanc_iter), \
       gp.settings.cg_tolerance(1.0):
    output = model(x)

    loss = -mll(output, y)

    loss.backward()

  optim.step()

  t_end = timer()

  return {
    'train/ll': -loss.detach().item(),
    'train/ts': t_end - t_start
  }


def test(x, y, model, mll, lanc_iter=100, pre_size=100):
  t_start = timer()

  model.eval()

  with torch.no_grad(), \
       gp.settings.eval_cg_tolerance(1e-2), \
       gp.settings.max_preconditioner_size(pre_size), \
       gp.settings.max_root_decomposition_size(lanc_iter), \
       gp.settings.fast_pred_var():
      preds = model(x)

      pred_y = model.likelihood(model(x))
      rmse = (pred_y.mean - y).pow(2).mean(0).sqrt()

  t_end = timer()

  return {
    'test/rmse': rmse.item(),
    'test/ts': t_end - t_start
  }


def main(dataset: str = None, data_dir: str = None,
         epochs: int = 100, lr: int = 0.01, lanc_iter: int = 100, pre_size: int = 100,
         log_int: int = 1, seed: int = None):
    if data_dir is None and os.environ.get('DATADIR') is not None:
        data_dir = Path(os.path.join(os.environ.get('DATADIR'), 'uci'))

    assert dataset is not None, f'Select a dataset from "{data_dir}"'

    set_seeds(seed)

    ## Disable GPU for now.
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    train_dataset = UCIDataset.create(dataset, uci_data_dir=data_dir,
                                      mode="train", device=device)
    test_dataset = UCIDataset.create(dataset, uci_data_dir=data_dir,
                                     mode="test", device=device)

    train_x, train_y = train_dataset.x, train_dataset.y
    test_x, test_y = test_dataset.x, test_dataset.y
    train_x, train_y, test_x, test_y = standardize(train_x, train_y, test_x, test_y)

    print(f'"{dataset}": D = {train_x.size(-1)}, Train N = {train_x.size(0)}, Test N = {test_x.size(0)}')

    wandb.init(tensorboard=True, config={
      'dataset': dataset,
      'lr': lr,
      'lanc_iter': lanc_iter,
      'pre_size': pre_size,
      'D': train_x.size(-1),
      'N_train': train_x.size(0),
      'N_test': test_x.size(0),
    })
    logger = SummaryWriter(log_dir=wandb.run.dir)

    model = BilateralGPModel(train_x, train_y).to(device)
    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in tqdm(range(epochs)):
      train_dict = train(train_x, train_y, model, mll, optimizer,
                         lanc_iter=lanc_iter)
      log_scalar_dict(train_dict, logger, global_step=i + 1)
      
      if (i % log_int) == 0:
        test_dict = test(test_x, test_y, model, mll,
                         pre_size=pre_size, lanc_iter=lanc_iter)
        log_scalar_dict(test_dict, logger, global_step=i + 1)

    test_dict = test(test_x, test_y, model, mll,
                      pre_size=pre_size, lanc_iter=lanc_iter)
    log_scalar_dict(test_dict, logger, global_step=i + 1)


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  import fire
  fire.Fire(main)
