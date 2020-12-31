import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from timeit import default_timer as timer


def test_cpu(root, src, ref, cdebug=False):
  cpu_lattice = load(name=f'cpu_lattice{"_debug" if cdebug else ""}',
                     verbose=cdebug,
                     extra_cflags=['-DDEBUG'] if cdebug else None,
                     sources=[(root / '..' / 'lattice.cpp')])
  
  start = timer()

  res = cpu_lattice.filter(src, ref, 1)

  ts = timer() - start
  return res, ts

def test_gpu(root, src, ref, cdebug=False):
  device = 'cuda:0' if torch.cuda.is_available() else None

  assert device is not None

  src = src.to(device)
  ref = ref.to(device)

  gpu_lattice = load(name=f'gpu_lattice{"_debug" if cdebug else ""}',
                     verbose=cdebug,
                     extra_cflags=['-DDEBUG'] if cdebug else None,
                     extra_cuda_cflags=['-DDEBUG'] if cdebug else None,
                     sources=[
                       (root / 'permutohedral_cuda.cpp'),
                       (root / 'permutohedral_cuda_kernel.cu')
                     ])

  start = timer()

  res = gpu_lattice.filter(src, ref, 1)
  
  ts = timer() - start
  return res, ts

def main(n=1000, d=10, cdebug=True):
  root = Path(os.path.dirname(__file__))

  with torch.no_grad():
    ref = torch.rand(n, d).float()
    src = ref.norm(dim=-1, keepdim=True)

  print(f'N: {ref.size(0)}, pD: {ref.size(1)}')

  res_cpu, ts_cpu = test_cpu(root, src, ref, cdebug=cdebug)

  print('-------------------------------')

  res_gpu, ts_gpu = test_gpu(root, src, ref, cdebug=cdebug)
  res_gpu = res_gpu.cpu()

  print('-------------------------------')

  try:
    assert torch.allclose(res_cpu, res_gpu), 'CPU/GPU mismatch!'
    print('Matched!')
  except AssertionError as aexc:
    rel_err = (res_cpu - res_gpu).norm(p=2) / res_cpu.norm(p=2)
    abs_rel_err = (res_cpu - res_gpu).norm(p=1) / res_cpu.norm(p=1)

    delta_mu = (res_cpu - res_gpu).abs().mean()
    delta_std = (res_cpu - res_gpu).abs().std()

    print(f'Rel. Err.: {rel_err}')
    print(f'Abs. Rel. Err.: {abs_rel_err}')
    print(f'Pointwise Abs. Err.: {delta_mu} +/- {delta_std}')
    
    print('CPU Output:')
    print(res_cpu)
    print('GPU Output:')
    print(res_gpu)

    ## Relative errors may still be small enough to be ok.
    print(aexc)

  print(f'Speedup: ~{ts_cpu / ts_gpu:.2f}x')


if __name__ == "__main__":
  from fire import Fire
  Fire(main)