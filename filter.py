import torch
from torch.utils.cpp_extension import load
lattice = load(name="lattice",sources=["lattice.cpp"])
latticefilter = lattice.filter