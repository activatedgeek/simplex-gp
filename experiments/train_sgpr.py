import os
import torch
import gpytorch as gp
from tqdm.auto import tqdm
import wandb
from pathlib import Path
from timeit import default_timer as timer

from utils import set_seeds, prepare_dataset


class SGPRModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, n_inducing=500):
        likelihood = gp.likelihoods.GaussianLikelihood(
                      noise_constraint=gp.constraints.GreaterThan(1e-4))
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.base_covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel(ard_num_dims=train_x.size(-1)))
        self.covar_module = gp.kernels.InducingPointKernel(self.base_covar_module,
          inducing_points=train_x[torch.randperm(train_x.size(0))[:n_inducing]], likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


def train(x, y, model, mll, optim, lanc_iter=100):
  model.train()

  optim.zero_grad()

  with gp.settings.max_root_decomposition_size(lanc_iter), \
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
    'train/ll': -loss.detach().item(),
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
       gp.settings.fast_pred_var():
      t_start = timer()
    
      # pred_y = model.likelihood(model(x))
      pred_y = model(x)
      pred_ts = timer() - t_start

      rmse = (pred_y.mean - y).pow(2).mean(0).sqrt()
      mae = (pred_y.mean - y).abs().mean(0)

  return {
    f'{label}/rmse': rmse.item(),
    f'{label}/mae': mae.item(),
    f'{label}/pred_ts': pred_ts
  }


def main(dataset: str = None, data_dir: str = None, log_int: int = 1, seed: int = None, device: int = 0,
         epochs: int = 100, lr: int = 1e-3, lanc_iter: int = 100, pre_size: int = 10,
         n_inducing: int = 500):
    set_seeds(seed)
    device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"

    data_iter = prepare_dataset(dataset, uci_data_dir=data_dir, device=device)
    _, train_x, train_y = next(data_iter)
    _, val_x, val_y = next(data_iter)
    _, test_x, test_y = next(data_iter)

    print(f'"{dataset}": D = {train_x.size(-1)}, Train N = {train_x.size(0)}, Val N = {val_x.size(0)} Test N = {test_x.size(0)}')

    wandb.init(config={
      'method': 'SGPR',
      'dataset': dataset,
      'lr': lr,
      'lanc_iter': lanc_iter,
      'pre_size': pre_size,
      'n_inducing': n_inducing,
      'D': train_x.size(-1),
      'N_train': train_x.size(0),
      'N_test': test_x.size(0),
      'N_val': val_x.size(0),
    })

    model = SGPRModel(train_x, train_y, n_inducing=n_inducing).to(device)
    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in tqdm(range(epochs)):
      train_dict = train(train_x, train_y, model, mll, optimizer,
                         lanc_iter=lanc_iter)
      wandb.log(train_dict, step=i + 1)
      
      if (i % log_int) == 0:
        val_dict = test(val_x, val_y, model, mll,
                        pre_size=pre_size, lanc_iter=lanc_iter,
                        label='val')
        wandb.log(val_dict, step=i + 1)

        test_dict = test(test_x, test_y, model, mll,
                         pre_size=pre_size, lanc_iter=lanc_iter)
        wandb.log(test_dict, step=i + 1)

        torch.save(model.state_dict(), Path(wandb.run.dir) / 'model.pt')

    if (i % log_int) != 0:
      test_dict = test(test_x, test_y, model, mll,
                        pre_size=pre_size, lanc_iter=lanc_iter)
      wandb.log(test_dict, step=i + 1)

    torch.save(model.state_dict(), Path(wandb.run.dir) / 'model.pt')


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  import fire
  fire.Fire(main)