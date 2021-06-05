
import os
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load
from gpytorch.kernels import Kernel
from gpytorch.lazy import LazyTensor
import logging


def get_coeffs(kernel_fn,order):
    """ Given a raw stationary and isotropic kernel function (R -> R),
        computes the optimal discrete filter coefficients to a fixed order
        by matching the coverage in spatial domain (from the highest and lowest samples)
        with the coverage in the frequency domain (from the nyquist frequency)."""
    N = 10**4
    x = np.linspace(-30,30,N)
    fn_values = kernel_fn(torch.from_numpy(x).float()).cpu().data.numpy()
    w = 2*np.pi*np.fft.fftfreq(N,60/N)
    fft_values = np.absolute(np.fft.fft(fn_values)/(2*np.pi*np.sqrt(N)))
    
    obj_fn = partial(coverage_diff,order=order,x=x,w=w,fn_values=fn_values,fft_values=fft_values)
    s = binary_search(0,(.1,9),obj_fn,1e-4) # Search for zeros of objective function (up to 1e-4 precision)
    vals = kernel_fn(s*torch.arange(-order,order+1).float())
    return vals/vals[order]

def coverage_diff(spacing,order,x,w,fn_values,fft_values):
    """ Given sample spacing and filter order, compute the difference in coverage over
        spatial and frequency domains. """
    k = 2*order+1
    a = spacing*k/2
    nyquist_w = np.pi/spacing
    spatial_coverage = fn_values[(-a<=x)&(x<=a)].sum()/fn_values.sum() #(dx's cancel)
    spectral_coverage = fft_values[(-nyquist_w<=w)&(w<=nyquist_w)].sum()/fft_values.sum() #(dw's cancel)
    logging.info(f"cov: x {spatial_coverage:.2f} w {spectral_coverage:.2f}")
    return spatial_coverage-spectral_coverage

def binary_search(target,bounds,fn,eps=1e-2):
    """ Perform binary search to find the input corresponding to the target output
        of a given monotonic function fn up to the eps precision. Requires initial bounds
        (lower,upper) on the values of x."""
    lb,ub = bounds
    i = 0
    while ub-lb>eps:
        guess = (ub+lb)/2
        y = fn(guess)
        if y<target:
            lb = guess
        elif y>=target:
            ub = guess
        i+=1
        if i>500: assert False
    return (ub+lb)/2


class LatticeFilterGeneral(Function):
    method = None

    @staticmethod
    def lazy_compile(is_cuda):
        if is_cuda:
            LatticeFilterGeneral.method = load(name="gpu_lattice", verbose=True,
                sources=[
                    os.path.join(os.path.dirname(__file__), 'cuda', 'permutohedral_cuda.cpp'),
                    os.path.join(os.path.dirname(__file__), 'cuda', 'permutohedral_cuda_kernel.cu')
                ]).filter
        else:
            LatticeFilterGeneral.method = load(name="cpu_lattice", verbose=True,
                sources=[
                    os.path.join(os.path.dirname(__file__), 'cpp', 'lattice.cpp')
                ]).filter
        
    @staticmethod
    def forward(ctx, source, reference,kernel_fn):
        if LatticeFilterGeneral.method is None:
            LatticeFilterGeneral.lazy_compile(source.is_cuda)

        #W = torch.exp(-((reference[None,:,:] - reference[:,None,:])**2).sum(-1)).double()
        #ctx.W = W
        # Typical runtime of O(nd^2 + n*L), Worst case O(nd^2 + n*L*d)
        assert source.shape[0] == reference.shape[0], \
            "Incompatible shapes {}, and {}".format(source.shape,reference.shape)
        coeffs = kernel_fn.get_coeffs().to(source.device)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(source,reference) # TODO: add batch compatibility
            # ctx.gpu = source.is_cuda
            ctx.kernel_fn= kernel_fn
            ctx.coeffs = coeffs
            ctx.deriv_coeffs = ctx.kernel_fn.get_deriv_coeffs().to(source.device)
        # filtermethod = gpulattice if source.is_cuda else cpulattice
        filtermethod = LatticeFilterGeneral.method
        filtered_output = filtermethod(source,reference.contiguous(),coeffs)
        return filtered_output
    @staticmethod
    def backward(ctx,grad_output):
        assert LatticeFilterGeneral.method is not None
        # Typical runtime of O(nd^2 + 2L*n*d), Worst case  O(nd^2 + 2L*n*d^2)
        # Does not support second order autograd at the moment
        # filtermethod = gpulattice if ctx.gpu else cpulattice
        filtermethod = LatticeFilterGeneral.method
        with torch.no_grad():
            src, ref = ctx.saved_tensors
            g = grad_output
            n,L = src.shape[-2:]
            d = ref.shape[-1]
            grad_source = grad_reference = None
            if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
                grad_source = filtermethod(g,ref,ctx.coeffs)#ctx.W@g#latticefilter(g,ref) # Matrix is symmetric
            if ctx.needs_input_grad[1]: # try torch.no_grad ()
                gf = grad_and_ref = grad_output[...,None]*ref[...,None,:] # n x L x d
                grad_and_ref_flat = grad_and_ref.view(grad_and_ref.shape[:-2]+(L*d,))
                sf = src_and_ref = src[...,None]*ref[...,None,:] # n x L x d
                src_and_ref_flat = src_and_ref.view(src_and_ref.shape[:-2]+(L*d,))
                #n x (L+Ld+L+Ld):   n x L       n x Ld     n x L   n x Ld 
                all_ = torch.cat([g,grad_and_ref_flat,src,src_and_ref_flat],dim=-1)
                filtered_all = filtermethod(all_,ref.contiguous(),ctx.deriv_coeffs)#ctx.W@all_#torch.randn_like(all_)#
                [wg,wgf,ws,wsf] = torch.split(filtered_all,[L,L*d,L,L*d],dim=-1)
                # has shape n x d 
                grad_reference = -2*(sf*wg[...,None] - src[...,None]*wgf.view(-1,L,d) + gf*ws[...,None] - g[...,None]*wsf.view(-1,L,d)).sum(-2) # sum over L dimension
                if ctx.needs_input_grad[0]: grad_source = wg
        return grad_source, grad_reference,None


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
            # assert False
            return torch.ones_like(x1[..., 0])

        if x1.shape == x2.shape and x1.eq(x2).all():
            return SquareLazyLattice(x1.div(self.lengthscale),self.dkernel_fn)
        else:
            return RectangularLazyLattice(x1.div(self.lengthscale),x2.div(self.lengthscale),self.dkernel_fn)

def rbf(d2):
    return torch.exp(-d2)

#TODO: Deal with Matern derivative at 0
from torch.autograd import Function
class Matern(Function):
    @staticmethod
    def forward(ctx,d2,nu):
        d  =d2.abs().sqrt()#(d2.abs()+1e-3).sqrt()
        exp_component = torch.exp(-np.sqrt(nu * 2) * d)
        if nu == 1.5:
            polynomial = (np.sqrt(3) * d).add(1)
        elif nu == 2.5:
            polynomial = (np.sqrt(5) * d).add(1).add(5.0 / 3.0 * d ** 2)
        else:
            raise NotImplementedError
        if any(ctx.needs_input_grad):
            ctx.nu=nu
            ctx.save_for_backward(d,exp_component)
        return polynomial * exp_component
    @staticmethod
    def backward(ctx,grad_output):
        if ctx.needs_input_grad[1]: raise NotImplementedError # Gradients wrt to nu are not currently supported
        d,exp = ctx.saved_tensors
        if ctx.nu == 1.5:
            polynomial = -(3/2)
        elif ctx.nu == 2.5:
            polynomial = -(5/6)*(1+d*np.sqrt(5))
        else:
            raise NotImplementedError
        return grad_output*polynomial*exp,None

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

def RBFLattice(*args,order=2,**kwargs):
    return LatticeAccelerated(rbf,*args,order=order,**kwargs)

def BilateralKernel(*args,**kwargs):
    return RBFLattice(*args,**kwargs)

def MaternLattice(*args,nu=1.5,order=3,**kwargs,):
    return LatticeAccelerated(lambda d2: Matern.apply(d2,nu),*args,order=order,**kwargs)