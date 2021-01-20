import os
import torch
import numpy as np
import gpytorch as gp
from tqdm.auto import tqdm
import wandb
from pathlib import Path
from timeit import default_timer as timer

from utils import set_seeds, prepare_dataset, EarlyStopper


class SKIPGPModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, grid_size=100, nu=None, min_noise=1e-4):
        likelihood = gp.likelihoods.GaussianLikelihood(
                      noise_constraint=gp.constraints.GreaterThan(min_noise))
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.base_covar_module = gp.kernels.MaternKernel(nu=nu) \
          if nu is not None else gp.kernels.RBFKernel()
        self.covar_module = gp.kernels.ProductStructureKernel(
          gp.kernels.ScaleKernel(
            gp.kernels.GridInterpolationKernel(self.base_covar_module, grid_size=grid_size, num_dims=1)
          ), num_dims=train_x.size(-1)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


def train(x, y, model, mll, optim, lanc_iter=100):
  model.train()

  optim.zero_grad()

  with gp.settings.max_root_decomposition_size(lanc_iter), \
       gp.settings.use_toeplitz(False), \
       gp.settings.cg_tolerance(1.0):
    t_start = timer()
    
    output = model(x)
    loss = -mll(output, y)

    loss_ts = timer() - t_start

    t_start = timer()

    loss.backward()
    optim.step()

    bw_ts = timer() - t_start

  return {
    'train/mll': -loss.detach().item(),
    'train/loss_ts': loss_ts,
    'train/bw_ts': bw_ts,
    'train/total_ts': loss_ts + bw_ts
  }


def test(x, y, model, mll, lanc_iter=100, pre_size=100, label='test'):
  model.eval()

  with torch.no_grad(), \
       gp.settings.eval_cg_tolerance(1e-2), \
       gp.settings.max_preconditioner_size(pre_size), \
       gp.settings.max_root_decomposition_size(lanc_iter), \
       gp.settings.fast_pred_var(), \
       gp.settings.use_toeplitz(False):
      t_start = timer()
    
      # pred_y = model.likelihood(model(x))
      pred_y = model(x)
      pred_ts = timer() - t_start

      rmse = (pred_y.mean - y).pow(2).mean(0).sqrt()
      mae = (pred_y.mean - y).abs().mean(0)
      nll = - torch.distributions.Normal(pred_y.mean,
        pred_y.variance.add(model.likelihood.noise).sqrt()).log_prob(y).mean()

  torch.cuda.empty_cache()

  return {
    f'{label}/rmse': rmse.item(),
    f'{label}/mae': mae.item(),
    f'{label}/pred_ts': pred_ts,
    f'{label}/nll': nll.item()
  }


def main(dataset: str = None, data_dir: str = None, log_int: int = 1, seed: int = None, device: int = 0,
         epochs: int = 1000, lr: int = 0.1, p_epochs: int = 200, lanc_iter: int = 100, pre_size: int = 100,
         grid_size: int = None, total_grid_size: int = 100, nu: float = None, min_noise: float = 1e-4):
    wandb.init(config={
      'method': 'SKIP',
      'dataset': dataset,
      'lr': lr,
      'lanc_iter': lanc_iter,
      'pre_size': pre_size,
      'nu': nu
    })
    
    set_seeds(seed)
    device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"

    data_iter = prepare_dataset(dataset, uci_data_dir=data_dir, device=device)
    _, train_x, train_y = next(data_iter)
    _, val_x, val_y = next(data_iter)
    _, test_x, test_y = next(data_iter)

    if grid_size is None:
      grid_size = total_grid_size // train_x.size(-1)

    print(f'"{dataset}": grid_size={grid_size}; D = {train_x.size(-1)}, Train N = {train_x.size(0)}, Val N = {val_x.size(0)} Test N = {test_x.size(0)}')

    wandb.config.update({
      'grid_size': grid_size,
      'D': train_x.size(-1),
      'N_train': train_x.size(0),
      'N_test': test_x.size(0),
      'N_val': val_x.size(0)
    })

    model = SKIPGPModel(train_x, train_y, grid_size=grid_size, nu=nu, min_noise=min_noise).to(device)
    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopper(patience=p_epochs)

    for i in tqdm(range(epochs)):
      with gp.settings.use_toeplitz(True):
        train_dict = train(train_x, train_y, model, mll, optimizer,
                           lanc_iter=lanc_iter)
      wandb.log(train_dict, step=i + 1)
      
      if (i % log_int) == 0:
        val_dict = test(val_x, val_y, model, mll,
                        pre_size=pre_size, lanc_iter=lanc_iter,
                        label='val')

        test_dict = test(test_x, test_y, model, mll,
                         pre_size=pre_size, lanc_iter=lanc_iter)

        stopper(-val_dict['val/rmse'], dict(
          state_dict=model.state_dict(),
          summary={
            'test/best_rmse': test_dict['test/rmse'],
            'test/best_nll': test_dict['test/nll'],
            'val/best_step': i + 1
          }
        ))
        wandb.log(val_dict, step=i + 1)
        wandb.log(test_dict, step=i + 1)
        for k, v in stopper.info().get('summary').items():
          wandb.run.summary[k] = v
        torch.save(stopper.info().get('state_dict'), Path(wandb.run.dir) / 'model.pt')

        if stopper.is_done():
          break

    wandb.run.summary['val/best_step'] = stopper.info().get('step')
    wandb.run.summary['test/best_rmse'] = stopper.info().get('rmse')
    torch.save(stopper.info().get('state_dict'), Path(wandb.run.dir) / 'model.pt')


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  import fire
  fire.Fire(main)
