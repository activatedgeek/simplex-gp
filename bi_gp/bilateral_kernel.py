
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
from bi_gp.gaussian_matrix import LatticeFilterGeneral
from bi_gp.discretized_coefficients import get_coeffs

# class LazyBilateral(LazyTensor):
#     def __init__(self,x):
#         super().__init__(x)
#         self.x = x

#     def _matmul(self,V):
#         return LatticeFilter.apply(V,self.x)
#     def _size(self):
#         return torch.Size((self.x.shape[-2],self.x.shape[-2]))
#     def _transpose_nonbatch(self):
#         return self
#     def diag(self):
#         return torch.ones_like(self.x[...,0])

# class RectangularLazyBilateral(LazyTensor):
#     def __init__(self,xin,xout):
#         super().__init__(xin, xout)
#         self.xin = xin
#         self.xout = xout
#     def _matmul(self,V):
#         n = V.shape[-2]
#         assert n==self.xout.shape[-2], f"mismatched shapes? {V.shape,self.xout.shape}"
#         x_large = torch.cat([self.xout,self.xin],dim=-2)
#         V_large = torch.zeros(*V.shape[:-2],x_large.shape[-2],V.shape[-1],device=V.device,dtype=V.dtype)
#         V_large[...,:n,:] += V
#         return LatticeFilter.apply(V_large,x_large)[...,n:,:]
#     def _size(self):
#         return torch.Size((*self.xin.shape[:-1],self.xout.shape[-2]))
#     def _transpose_nonbatch(self):
#         return RectangularLazyBilateral(self.xout,self.xin)

# class BilateralKernel(Kernel):
#     has_lengthscale = True

#     def forward(self, x1, x2, diag=False, **params):#what are the extra kwargs for??
#         if diag == True:
#             assert False
#             return torch.ones_like(x1[..., 0])

#         if x1.shape == x2.shape and x1.eq(x2).all():
#             return LazyBilateral(x1.div(self.lengthscale))
#         else:
#             return RectangularLazyBilateral(x1.div(self.lengthscale),x2.div(self.lengthscale))

class SquareLazyLattice(LazyTensor):
    def __init__(self,x,dkernel=None):
        super().__init__(x,dkernel=dkernel)
        self.x = x
        self.dkernel=dkernel

    def _matmul(self,V):
        return LatticeFilterGeneral.apply(V,self.x,self.dkernel)
    def _size(self):
        return torch.Size((self.x.shape[-2],self.x.shape[-2]))
    def _transpose_nonbatch(self):
        return self
    def diag(self):
        return torch.ones_like(self.x[...,0])

class RectangularLazyLattice(LazyTensor):
    def __init__(self,xin,xout,dkernel=None):
        super().__init__(xin, xout,dkernel=dkernel)
        self.xin = xin
        self.xout = xout
        self.dkernel=dkernel
        self._use_gpu=False

    def _matmul(self,V):
        n = V.shape[-2]
        assert n==self.xout.shape[-2], f"mismatched shapes? {V.shape,self.xout.shape}"
        x_large = torch.cat([self.xout,self.xin],dim=-2)
        V_large = torch.zeros(*V.shape[:-2],x_large.shape[-2],V.shape[-1],device=V.device,dtype=V.dtype)
        V_large[...,:n,:] += V
        return LatticeFilterGeneral.apply(V_large,x_large,self.dkernel)[...,n:,:]
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
        self._forward_coeffs = get_coeffs(lambda d: self.kernel_fn(d**2),self.order)
        print(f"Discretized kernel coeffs: {self._forward_coeffs}")
        def gradkern(x):
            with torch.autograd.enable_grad():
                z = x**2+torch.zeros_like(x,requires_grad=True)
                g = torch.autograd.grad(self.kernel_fn(z).sum(),z)[0]
            return g
        self._deriv_coeffs = get_coeffs(gradkern,self.order)
        print(f"Discretized kernel deriv coeffs: {self._deriv_coeffs}")
    def get_coeffs(self):
        return self._forward_coeffs
    def get_deriv_coeffs(self):
        return self._deriv_coeffs




class LatticeAccelerated(Kernel):
    has_lengthscale=True
    def __init__(self,kernel_fn,*args,order=2,**kwargs):
        """ Wraps a stationary kernel with the permutohedral lattice acceleartion.
            Given kernel defined as a function of d=|x1-x2| which is differentiable,
            returns a Gpytorch kernel."""
        super().__init__(*args,**kwargs)
        self.dkernel_fn = DiscretizedKernelFN(kernel_fn,order)    

    def forward(self, x1, x2, diag=False, **params):#what are the extra kwargs for??
        if diag == True:
            assert False
            return torch.ones_like(x1[..., 0])

        if x1.shape == x2.shape and x1.eq(x2).all():
            return SquareLazyLattice(x1.div(self.lengthscale),self.dkernel_fn)
        else:
            return RectangularLazyLattice(x1.div(self.lengthscale),x2.div(self.lengthscale),self.dkernel_fn)

def rbf(d2):
    return torch.exp(-d2)

# from torch.autograd import Function
# class Matern(Function):
#     @staticmethod
#     def forward(ctx,d2,nu=.5):
#         d  =d2.abs().sqrt()#(d2.abs()+1e-3).sqrt()
#         exp_component = torch.exp(-np.sqrt(nu * 2) * d)
#         if nu == 0.5:
#             constant_component = 1
#         elif nu == 1.5:
#             constant_component = (np.sqrt(3) * d).add(1)
#         elif nu == 2.5:
#             constant_component = (np.sqrt(5) * d).add(1).add(5.0 / 3.0 * d ** 2)
#         else:
#             raise NotImplementedError
#         if any(ctx.needs_input_grad):
#             ctx.nu=nu
#             ctx.save_for_backward(exp_component)
#         return constant_component * exp_component
#     @staticmethod
#     def backward(ctx,grad_output):
#         if ctx.needs_input_grad[1]: raise NotImplementedError # Gradients wrt to nu are not currently supported
#         exp = ctx.saved_tensors
#         if ctx.nu == 0.5:
#             g = 
#         elif ctx.nu == 1.5:
#             constant_component = (np.sqrt(3) * d).add(1)
#         elif ctx.nu == 2.5:
#             constant_component = (np.sqrt(5) * d).add(1).add(5.0 / 3.0 * d ** 2)
#         else:
#             raise NotImplementedError
#         return grad_source, grad_reference,None

def matern(d2,nu=.5):
    d  =d2.abs().sqrt()#(d2.abs()+1e-3).sqrt()
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

def RBFLattice(*args,**kwargs):
    return LatticeAccelerated(rbf,*args,order=2,**kwargs)

def BilateralKernel(*args,**kwargs):
    return RBFLattice(*args,**kwargs)

def MaternLattice(*args,nu=.5,**kwargs,):
    return LatticeAccelerated(partial(matern,nu=nu),*args,order=3,**kwargs)