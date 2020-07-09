import torch
#import lattice
from torch.utils.cpp_extension import load
lattice = load(name="lattice",sources=["lattice.cpp"])
ref = torch.rand(100000,3)
src = torch.rand(100000,6)
#out = torch.zeros(1000000,16)
#print(lattice.testfunc(torch.ones((2,3))))
src[:,-1] = torch.ones(100000)
print(ref)
print(src)
print(lattice.filter(src,ref))
# Problem with undefined symbols is in the gcc version whihc is too new, need 4.9.2