import os
import torch
import wandb
from timeit import default_timer as timer
from gpytorch.kernels.keops import RBFKernel, MaternKernel

from bi_gp.bilateral_kernel import MaternLattice, RBFLattice
from utils import set_seeds, prepare_dataset


def rel_err(x,y):
  return ((x-y)**2).mean().sqrt()/((x**2).mean().sqrt()+(y**2).mean().sqrt())


def cos_err(x,y):
  return (x * y).sum() / (x.norm(p=2) * y.norm(p=2))


def compute(K, X, y, n_iter=5):
  if X.is_cuda:
    start = torch.cuda.Event(enable_timing=True)
    start.record()
  else:
    start = timer()
  
  for _ in range(n_iter):
    mvm = K(X, X) @ y
  
  if X.is_cuda:
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize()
  else:
    end = timer()

  if X.is_cuda:
    t = start.elapsed_time(end) / n_iter / 1000
  else:
    t = (end - start) / n_iter

  return mvm, t

def main(dataset: str = None, data_dir: str = None, seed: int = None, device: int = 0,
         nu: float = None, order: int = 1, ell: float = 1.0, n_data: int = None,
         n_iter: int = 5):
    
    kern = 'rbf' if nu is None else 'mat'
    
    wandb.init(config={
      'kernel': kern,
      'dataset': dataset,
      'nu': nu,
      'order': order,
      'lengthscale': ell,
    })

    set_seeds(seed)
    device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"

    data_iter = prepare_dataset(dataset, uci_data_dir=data_dir, device=device, train_val_split=1.0)
    _, X, y = next(data_iter)
    if n_data is not None:
      perm = torch.randperm(n_data)
      X, y = X[perm], y[perm]

    X = (X - X.mean(dim=0, keepdim=True)) / X.std(dim=0, keepdim=True)
    y = (y - y.mean(dim=0, keepdim=True)) / y.std(dim=0, keepdim=True)

    # X, y = torch.randn(int(1e7), 20).to(device), torch.rand(int(1e7), 1).to(device)

    wandb.config.update({ 'n': X.shape[0], 'd': X.shape[1] })

    if kern == "rbf":
      K_gt = RBFKernel().to(device)
      K_lattice = RBFLattice(order=order).to(device)
    elif kern == "mat":
      K_gt = MaternKernel(nu=nu).to(device)
      K_lattice = MaternLattice(nu=nu, order=order).to(device)
    else:
      raise NotImplementedError

    if ell is not None:
      K_gt.lengthscale = K_lattice.lengthscale = ell

    print(f'Lattice (n = {X.shape[0]}, d = {X.shape[1]})..')
    compute(K_lattice, X, y, n_iter=n_iter)  ## warm up cache
    mvm_lattice, t2 = compute(K_lattice, X, y, n_iter=n_iter)
    print('..Done.')
    torch.cuda.empty_cache()

    print(f'Exact (n = {X.shape[0]}, d = {X.shape[1]})..')
    compute(K_gt, X, y, n_iter=n_iter)  ## warm up cache
    mvm_gt, t = compute(K_gt, X, y, n_iter=n_iter)
    print('..Done.')
    torch.cuda.empty_cache()

    err = rel_err(mvm_gt,mvm_lattice/(mvm_lattice/mvm_gt).mean())
    c_err = cos_err(mvm_gt, mvm_lattice)

    wandb.log({ 'ts/ref': t, 'ts/lattice': t2, 'ts/rel': (t2 - t) / t })
    wandb.log({ 'err/rel_err': err, 'err/cos_err': c_err })


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  import fire
  fire.Fire(main)
