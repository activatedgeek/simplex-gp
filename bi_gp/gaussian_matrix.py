import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import math
import gpytorch

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
                    os.path.join(os.path.dirname(__file__), 'lattice.cpp')
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


if __name__=="__main__":
    from torch.autograd import gradcheck
    import numpy as np
    from PIL import Image
    def read_img(filename):
        img = Image.open(filename).convert('RGB')
        img = np.array(img).astype(float)/255
        return img
    sigma_p = .01
    sigma_c = .125
    img = read_img('./lattice/lite/images/input.bmp')[::64,::64]
    h,w,c = img.shape
    position = np.mgrid[:h,:w].transpose((1,2,0))/np.sqrt(h**2+w**2)
    reference = np.zeros((h,w,5))
    reference[...,:3] = img/sigma_c
    reference[...,3:] = position/sigma_p
    #reference = position/sigma_p
    homo_src = np.ones((h,w,3+1))
    homo_src[...,:c] = img
    ref_arr = torch.tensor(reference.reshape((h*w,-1)).astype(np.float64),requires_grad=True)
    src_arr = torch.tensor(homo_src.reshape((h*w,-1)).astype(np.float64),requires_grad=False)
    #ref_arr.requires_grad=True#False
    #src_arr.requires_grad=True#True
    ref_arr = torch.rand(80,3,dtype=torch.double,requires_grad=True)
    src_arr = torch.rand(80,2,dtype=torch.double,requires_grad=False) # Because of single precision
    #print("AAAA")
    #test = gradcheck(LatticeFilter.apply,(src,ref),eps=1e-3,rtol=5e-2,atol=1e-2)
    test = gradcheck(LatticeFilterGeneral.apply,(src_arr,ref_arr),eps=1e-5,rtol=5e-4,atol=1e-5)
    print(test) # Gradients are perhaps wrong still (need to implement double precision method)
    
