
import torch
from gpytorch.kernels import Kernel
from gpytorch.lazy import LazyTensor
import torch
import torch.nn as nn
import time
import numpy as np
import math
import gpytorch

from bi_gp.gaussian_matrix import LatticeFilter


class LazyBilateral(LazyTensor):
    def __init__(self,x):
        super().__init__(x)
        self.x = x
    def _matmul(self,V):
        return LatticeFilter.apply(V,self.x)
    def _size(self):
        return torch.Size((self.x.shape[-2],self.x.shape[-2]))
    def _transpose_nonbatch(self):
        return self
    def diag(self):
        return torch.ones_like(self.x[...,0])

class RectangularLazyBilateral(LazyTensor):
    def __init__(self,xin,xout):
        super().__init__(xin, xout)
        self.xin = xin
        self.xout = xout
    def _matmul(self,V):
        n = V.shape[-2]
        assert n==self.xin.shape[-2], f"mismatched shapes? {V.shape,self.xin.shape}"
        x_large = torch.cat([self.xin,self.xout],dim=-2)
        V_large = torch.zeros(*V.shape[:-2],x_large.shape[-2],V.shape[-1],device=V.device,dtype=V.dtype)
        V_large[...,:n,:] += V
        return LatticeFilter(V_large,x_large)[...,:n]
    def _size(self):
        return torch.Size((*self.xout.shape[:-1],self.xin.shape[-2]))
    def _transpose_nonbatch(self):
        return RectangularLazyBilateral(self.xout,self.xin)

class BilateralKernel(Kernel):
    has_lengthscale = True
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, diag=False,**kwargs):#what are the extra kwargs for??
        if diag==True:
            return torch.ones_like(x1[...,0])
        # TODO(sanyam): verify this change.
        if x1.shape == x2.shape and x1.eq(x2).all():
            return LazyBilateral(x1.div(self.lengthscale))
        else:
            return RectangularLazyBilateral(x1.div(self.lengthscale),x2.div(self.lengthscale))
        
