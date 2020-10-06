import os
import torch		
from torch.utils.cpp_extension import load		

if __name__ == "__main__":
  lattice = load(name="lattice",
                 sources=[f"{os.path.dirname(__file__)}/lattice.cpp"])		

  N = int(1e4)
  ref = torch.rand(N, 6)
  src = torch.rand(N, 3)
  src[:,-1] = torch.ones(N)

  lattice.filter(src, ref, 1)
