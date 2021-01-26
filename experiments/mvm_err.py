import os
import torch
import wandb
# from timeit import default_timer as timer
from gpytorch.kernels import RBFKernel, MaternKernel

from bi_gp.bilateral_kernel import MaternLattice, RBFLattice
from utils import set_seeds, prepare_dataset


def rel_err(x,y):
    return ((x-y)**2).mean().sqrt()/((x**2).mean().sqrt()+(y**2).mean().sqrt())

def main(dataset: str = None, data_dir: str = None, seed: int = None, device: int = 0,
         kern: str = None, nu: float = None, order: int = 1, ell: float = 1.0, n: int = None):
    wandb.init(config={
      'kernel': kern,
      'dataset': dataset,
      'nu': nu,
      'order': order,
      'lengthscale': ell,
    })

    assert kern in ['rbf', 'mat'], "Invalid kernel"

    set_seeds(seed)
    device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"

    data_iter = prepare_dataset(dataset, uci_data_dir=data_dir, device=device, train_val_split=1.0)
    _, X, y = next(data_iter)
    if n is not None:
      perm = torch.randperm(n)
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

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # start = timer()

    mvm_gt = K_gt(X, X) @ y

    # end = timer()
    end.record()
    torch.cuda.current_stream().synchronize()

    ## To build cache
    mvm_lattice = K_lattice(X, X) @ y
    torch.cuda.current_stream().synchronize()

    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    # start2 = timer()

    mvm_lattice = K_lattice(X, X) @ y

    # end2 = timer()
    end2.record()
    torch.cuda.current_stream().synchronize()

    # wandb.log({ 'ts/ref': end - start })
    # wandb.log({ 'ts/lattice': end2 - start2 })

    wandb.log({ 'ts/ref': start.elapsed_time(end) / 1000 })
    wandb.log({ 'ts/lattice': start2.elapsed_time(end2) / 1000 })

    err = rel_err(mvm_gt,mvm_lattice/(mvm_lattice/mvm_gt).mean())

    wandb.log({ 'err/rel_err': err })


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  import fire
  fire.Fire(main)
