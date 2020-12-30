import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from timeit import default_timer as timer


def test_cpu(root, src, ref, cdebug=False):
  cpu_lattice = load(name="cpu_lattice",
                     verbose=cdebug,
                     extra_cflags=['-DDEBUG'] if cdebug else None,
                     sources=[(root / '..' / 'lattice.cpp')])
  
  start = timer()

  res = cpu_lattice.filter(src, ref, 1)

  print(f'CPU finished in: {(timer() - start):.6f}s')

  return res


def test_gpu(root, src, ref, cdebug=False):
  device = 'cuda:0' if torch.cuda.is_available() else None

  assert device is not None

  src = src.to(device)
  ref = ref.to(device)

  gpu_lattice = load(name="gpu_lattice",
                     verbose=cdebug,
                     extra_cflags=['-DDEBUG'] if cdebug else None,
                     extra_cuda_cflags=['-DDEBUG'] if cdebug else None,
                     sources=[
                       (root / 'permutohedral_cuda.cpp'),
                       (root / 'permutohedral_cuda_kernel.cu')
                     ])

  start = timer()

  res = gpu_lattice.filter(src, ref, 1)
  
  print(f'GPU finished in: {(timer() - start):.6f}s')

  return res


def main(n=1000, d=10, cdebug=True):
  root = Path(os.path.dirname(__file__))

  with torch.no_grad():
    ref = torch.rand(n, d).float()
    src = ref.norm(dim=-1, keepdim=True)

  print(f'N: {ref.size(0)}, pD: {ref.size(1)}')

  res_cpu = test_cpu(root, src, ref, cdebug=cdebug)

  print('-------------------------------')

  res_gpu = test_gpu(root, src, ref, cdebug=cdebug).cpu()

  print('-------------------------------')

  ## May still be small enough to be ok.
  rel_err = (res_cpu - res_gpu).norm(p=2) / res_cpu.norm(p=2)
  abs_rel_err = (res_cpu - res_gpu).norm(p=1) / res_cpu.norm(p=1)
  print(f'Rel. Err.: {rel_err}')
  print(f'Abs. Rel. Err.: {abs_rel_err}')
  
  # print('CPU:')
  # print(res_cpu)
  # print('GPU:')
  # print(res_gpu)

if __name__ == "__main__":
  from fire import Fire
  Fire(main)