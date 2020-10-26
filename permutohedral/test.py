import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load


if __name__ == "__main__":
  device = 'cuda:0' if torch.cuda.is_available() else None

  root = Path(os.path.dirname(__file__)) / '..'

  # cpu_lattice = load(name="lattice",
  #                    sources=[(root / 'bi_gp' / 'lattice.cpp')])

  ref = torch.arange(0.0, 100000. + 1e-3, 0.5).unsqueeze(-1).to(device)
  src = (ref**2).cos().to(device)

  gpu_lattice = load(name="gpu_lattice",
                     sources=[
                       (root / 'permutohedral' / 'permutohedral_cuda.cpp'),
                       (root / 'permutohedral' / 'permutohedral_cuda_kernel.cu')
                     ])
  
  # print(torch.allclose(ref + src, gpu_lattice.filter(src, ref)))
  print(gpu_lattice.filter(src, ref))
