import os
import torch
import wandb
from timeit import default_timer as timer
from gpytorch.kernels import RBFKernel, MaternKernel

from bi_gp.bilateral_kernel import MaternLattice, RBFLattice
from utils import set_seeds, prepare_dataset


def rel_err(x,y):
    return ((x-y)**2).mean().sqrt()/((x**2).mean().sqrt()+(y**2).mean().sqrt())

def main(dataset: str = None, data_dir: str = None, seed: int = None, device: int = 0,
         nu: float = None, order: int = 1, ell: float = 1.0, n_data: int = None):
    
    kern = 'rbf' if nu is None else 'mat'
    
    wandb.init(config={
      'kernel': kern,
      'dataset': dataset,
      'nu': nu,
      'order': order,
      'lengthscale': ell,
      'n_data': n_data
    })

    set_seeds(seed)
    device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"

    data_iter = prepare_dataset(dataset, uci_data_dir=data_dir, device=device, train_val_split=1.0)
    _, X, y = next(data_iter)
    if n_data is not None:
      perm = torch.randperm(n_data)
      X, y = X[perm], y[perm]

    if kern == "rbf":
      K_gt = RBFKernel().to(device)
      K_lattice = RBFLattice(order=order).to(device)
    elif kern == "mat":
      K_gt = MaternKernel(nu=nu).to(device)
      K_lattice = MaternLattice(nu=nu, order=order).to(device)
    else:
      raise NotImplementedError

    K_gt.lengthscale = K_lattice.lengthscale = ell

    if X.is_cuda:
      start = torch.cuda.Event(enable_timing=True)
      start.record()
    else:
      start = timer()

    mvm_gt = K_gt(X, X) @ y

    if X.is_cuda:
      end = torch.cuda.Event(enable_timing=True)
      end.record()
      torch.cuda.current_stream().synchronize()
    else:
      end = timer()

    ## To build cache
    # mvm_lattice = K_lattice(X, X) @ y
    # torch.cuda.current_stream().synchronize()

    if X.is_cuda:
      start2 = torch.cuda.Event(enable_timing=True)
      start2.record()
    else:
      start2 = timer()

    mvm_lattice = K_lattice(X, X) @ y

    if X.is_cuda:
      end2 = torch.cuda.Event(enable_timing=True)
      end2.record()
      torch.cuda.current_stream().synchronize()
    else:
      end2 = timer()

    if X.is_cuda:
      t = start.elapsed_time(end) / 1000
      t2 = start2.elapsed_time(end2) / 1000
    else:
      t = end - start
      t2 = end2 - start2

    err = rel_err(mvm_gt,mvm_lattice/(mvm_lattice/mvm_gt).mean())

    wandb.log({ 'ts/ref': t, 'ts/lattice': t2, 'ts/rel': t2 / t, 'err/rel_err': err })


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  import fire
  fire.Fire(main)
