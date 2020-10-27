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
                     sources=[
                       (root / 'permutohedral' / 'permutohedral_cuda.cpp'),
                       (root / 'permutohedral' / 'permutohedral_cuda_kernel.cu')
                     ])
  gpu_lattice.filter(src, ref)

if __name__ == "__main__":
  root = Path(os.path.dirname(__file__)) / '..'

  with torch.no_grad():
    # ref = torch.randn(1000000, 100).float()
    ref = torch.arange(0.0, 5., 0.5).unsqueeze(-1).float()
    src = (ref**2).cos()

  # test_cpu(root, src, ref)
  test_gpu(root, src, ref)