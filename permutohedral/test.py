import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load


def test_cpu(root, src, ref):
  cpu_lattice = load(name="lattice",
                     sources=[(root / 'bi_gp' / 'lattice.cpp')])
  cpu_lattice.filter(src, ref, 1)


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
  gpu_lattice.filter(src, ref)

if __name__ == "__main__":
  root = Path(os.path.dirname(__file__)) / '..'

  with torch.no_grad():
    # ref = torch.randn(100000, 20).float()
    ref = torch.arange(0., 5., 1.).unsqueeze(-1).float()
    src = (ref**2).cos()

  print(f'N: {ref.size(0)}, pD: {ref.size(1)}')
  test_cpu(root, src, ref)
  test_gpu(root, src, ref)