
import torch
from gpytorch.kernels import Kernel
from gpytorch.lazy import LazyTensor
import torch
import torch.nn as nn
import time
import numpy as np
import math
import gpytorch
from functools import partial
from bi_gp.gaussian_matrix import LatticeFilter,LatticeFilterGeneral


class LazyBilateral(LazyTensor):
    def __init__(self,x):
        super().__init__(x)
        self.x = x
        self._use_gpu=False
    def to(self,device): #TODO: accept more general .to arguments
        if device in ('cpu',):
            self._use_gpu=False
        if device in ('gpu',):
            self._use_gpu=True
        return super().to(device)

    def _matmul(self,V):
        return LatticeFilter.apply(V,self.x,gpu=self._use_gpu)
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
        assert n==self.xout.shape[-2], f"mismatched shapes? {V.shape,self.xout.shape}"
        x_large = torch.cat([self.xout,self.xin],dim=-2)
        V_large = torch.zeros(*V.shape[:-2],x_large.shape[-2],V.shape[-1],device=V.device,dtype=V.dtype)
        V_large[...,:n,:] += V
        return LatticeFilter.apply(V_large,x_large)[...,n:,:]
    def _size(self):
        return torch.Size((*self.xin.shape[:-1],self.xout.shape[-2]))
    def _transpose_nonbatch(self):
        return RectangularLazyBilateral(self.xout,self.xin)

class BilateralKernel(Kernel):
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):#what are the extra kwargs for??
        if diag == True:
            assert False
            return torch.ones_like(x1[..., 0])

        if x1.shape == x2.shape and x1.eq(x2).all():
            return LazyBilateral(x1.div(self.lengthscale))
        else:
            return RectangularLazyBilateral(x1.div(self.lengthscale),x2.div(self.lengthscale))

class SquareLazyLattice(LazyTensor):
    def __init__(self,x,dkernel):
        super().__init__(x)
        self.x = x
        self._use_gpu=False
        self.dkernel=dkernel

    def to(self,device): #TODO: accept more general .to arguments
        if device in ('cpu',):
            self._use_gpu=False
        if device in ('gpu',):
            self._use_gpu=True
        return super().to(device)

    def _matmul(self,V):
        return LatticeFilterGeneral.apply(V,self.x,self.dkernel,gpu=self._use_gpu)
    def _size(self):
        return torch.Size((self.x.shape[-2],self.x.shape[-2]))
    def _transpose_nonbatch(self):
        return self
    def diag(self):
        return torch.ones_like(self.x[...,0])

class RectangularLazyLattice(LazyTensor):
    def __init__(self,xin,xout,dkernel):
        super().__init__(xin, xout)
        self.xin = xin
        self.xout = xout
        self.dkernel=dkernel
        self._use_gpu=False
    def to(self,device): #TODO: accept more general .to arguments
        if device in ('cpu',):
            self._use_gpu=False
        if device in ('gpu',):
            self._use_gpu=True
        return super().to(device)
    def _matmul(self,V):
        n = V.shape[-2]
        assert n==self.xout.shape[-2], f"mismatched shapes? {V.shape,self.xout.shape}"
        x_large = torch.cat([self.xout,self.xin],dim=-2)
        V_large = torch.zeros(*V.shape[:-2],x_large.shape[-2],V.shape[-1],device=V.device,dtype=V.dtype)
        V_large[...,:n,:] += V
        return LatticeFilterGeneral.apply(V_large,x_large,self.dkernel,gpu=self._use_gpu)[...,n:,:]
    def _size(self):
        return torch.Size((*self.xin.shape[:-1],self.xout.shape[-2]))
    def _transpose_nonbatch(self):
        return RectangularLazyLattice(self.xout,self.xin,self.dkernel)

class DiscretizedKernelFN(nn.Module):
    def __init__(self,kernel_fn,order):
        super().__init__()
        self.kernel_fn = kernel_fn
        self._forward_coeffs = None
        self._deriv_coeffs = None
        self.order = order
    def get_coeffs(self):
        if self._forward_coeffs is None:
            self._forward_coeffs = get_coeffs(self.kernel_fn)
        return self._forward_coeffs
    def get_deriv_coeffs(self):
        if self._deriv_coeffs is None:
            self._deriv_coeffs = get_coeffs(lambda x: torch.autograd.grad(self.kernel_fn(x),x)[0].item()/x)
        return self._deriv_coeffs

class LatticeAccelerated(Kernel):
    has_lengthscale=True
    def __init__(self,kernel_fn,order=2):
        """ Wraps a stationary kernel with the permutohedral lattice acceleartion.
            Given kernel defined as a function of d=|x1-x2| which is differentiable,
            returns a Gpytorch kernel."""
        super().__init__()
        self.dkernel_fn = DiscretizedKernelFN(kernel_fn,order)    

    def forward(self, x1, x2, diag=False, **params):#what are the extra kwargs for??
        if diag == True:
            assert False
            return torch.ones_like(x1[..., 0])

        if x1.shape == x2.shape and x1.eq(x2).all():
            return SquareLazyLattice(x1.div(self.lengthscale),self.dkernel_fn)
        else:
            return RectangularLazyLattice(x1.div(self.lengthscale),x2.div(self.lengthscale),self.dkernel_fn)

def rbf(d):
    return torch.exp(-d**2)

def matern(d,nu=.5):
    exp_component = torch.exp(-np.sqrt(nu * 2) * d)
    if nu == 0.5:
        constant_component = 1
    elif nu == 1.5:
        constant_component = (np.sqrt(3) * d).add(1)
    elif nu == 2.5:
        constant_component = (np.sqrt(5) * d).add(1).add(5.0 / 3.0 * d ** 2)
    else:
        raise NotImplementedError
    return constant_component * exp_component

def RBFLattice():
    return LatticeAccelerated(rbf,order=2)

def MaternLattice(nu):
    return LatticeAccelerated(partial(matern,nu=nu),order=2)