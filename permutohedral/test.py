import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from timeit import default_timer as timer


def test_cpu(root, src, ref):
  cpu_lattice = load(name="lattice",
                     verbose=True,
                     sources=[(root / 'bi_gp' / 'lattice.cpp')])
  
  start = timer()

  cpu_lattice.filter(src, ref, 1)

  print(f'CPU finished in: {(timer() - start):.6f}s')


def test_gpu(root, src, ref):
  device = 'cuda:0' if torch.cuda.is_available() else None

  assert device is not None

  src = src.to(device)
  ref = ref.to(device)

  gpu_lattice = load(name="gpu_lattice",
                     verbose=True,
                     sources=[
                       (root / 'permutohedral' / 'permutohedral_cuda.cpp'),
                       (root / 'permutohedral' / 'permutohedral_cuda_kernel.cu')
                     ])

  start = timer()

  result = gpu_lattice.filter(src, ref)

  print(f'GPU finished in: {(timer() - start):.6f}s')

if __name__ == "__main__":
  root = Path(os.path.dirname(__file__)) / '..'

  with torch.no_grad():
    ref = torch.arange(0., 5., 1.).unsqueeze(-1).float()
    # ref = torch.rand(1000, 30).float()
    src = (ref**2).cos()

  print(f'N: {ref.size(0)}, pD: {ref.size(1)}')
  # test_cpu(root, src, ref)
  test_gpu(root, src, ref)