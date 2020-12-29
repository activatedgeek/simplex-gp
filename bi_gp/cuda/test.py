import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from timeit import default_timer as timer


def test_cpu(root, src, ref):
  cpu_lattice = load(name="cpu_lattice",
                     verbose=True,
                     sources=[(root / '..' / 'lattice.cpp')])
  
  start = timer()

  res = cpu_lattice.filter(src, ref, 1)

  print(f'CPU finished in: {(timer() - start):.6f}s')

  return res


def test_gpu(root, src, ref):
  device = 'cuda:0' if torch.cuda.is_available() else None

  assert device is not None

  src = src.to(device)
  ref = ref.to(device)

  gpu_lattice = load(name="gpu_lattice",
                     verbose=True,
                     sources=[
                       (root / 'permutohedral_cuda.cpp'),
                       (root / 'permutohedral_cuda_kernel.cu')
                     ])

  start = timer()

  res = gpu_lattice.filter(src, ref, 1)
  
  print(f'GPU finished in: {(timer() - start):.6f}s')

  return res

if __name__ == "__main__":
  root = Path(os.path.dirname(__file__))

  with torch.no_grad():
    ref = torch.arange(0., 5., 1.).unsqueeze(-1).float()
    # ref = torch.rand(1000, 30).float()
    src = (ref**2).cos()

  print(f'N: {ref.size(0)}, pD: {ref.size(1)}')

  res_cpu = test_cpu(root, src, ref)

  print('-------------------------------')

  res_gpu = test_gpu(root, src, ref).cpu()

  try:
    assert torch.allclose(res_cpu, res_gpu), 'CPU/GPU mismatch.'
  except AssertionError:
    print(f'Error norm: {(res_cpu - res_gpu).norm(p=2).item():.6f} (this may be small enough to be ok!)')
    raise

  print('Matched!')