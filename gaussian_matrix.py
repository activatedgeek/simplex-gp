import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
from torch.autograd import Function
import torch.nn.functional as F
import time
import numpy as np
import concurrent.futures
import torch.multiprocessing as mp
import math
import gpytorch
#import multiprocessing as mp
lattice = load(name="lattice",sources=["lattice.cpp"])
latticefilter = lattice.filter




class LatticeGaussian(nn.Module):
    def __init__(self,ref):
        super().__init__()
        self.ref = ref

    def __matmul__(self,U):
        return self(U)
    # def __rmatmul__(self,U):
    #     return self(U)

    def forward(self,U):
        return LatticeFilter.apply(U,self.ref)# - U #usually substracts U: 0 along the diagonal.



class BatchedAdjacency(nn.Module):
    def __init__(self,num_threads=8):
        super().__init__()
        self.num_threads = num_threads
    def forward(self, src_imgs,guide_imgs):
        bs,L,h,w = src_imgs.shape
        bs,d,h,w = guide_imgs.shape
        flat_srcs = src_imgs.view(bs,L,-1).permute(0,2,1)
        flat_refs = guide_imgs.view(bs,d,-1).permute(0,2,1)
        filtered_imgs = BatchedLatticeFilter.apply(flat_srcs,flat_refs,\
                                self.num_threads).permute(0,2,1).reshape(src_imgs.shape)
        return filtered_imgs - src_imgs
    # def forward(self,Us,refs):
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_threads) as executor:
    #         filtered_mb = torch.stack(list(executor.map(lattice_filter_img,Us,refs)))
    #     return filtered_mb - Us

# def lattice_filter_img(src,ref):
#         c,h,w = src.shape
#         k,h,w = ref.shape
#         flat_src = src.view(c,-1).t()
#         flat_ref = ref.view(k,-1).t()
#         return LatticeFilter.apply(flat_src,flat_ref).t().reshape(src.shape)

# def batched_filter(flat_srcs,flat_refs,num_threads):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
#         filtered_mb = torch.stack(list(executor.map(latticefilter,flat_srcs,flat_refs)))
#     return filtered_mb

def batched_filter(flat_srcs,flat_refs,num_threads):
    print(f"numthreads: {num_threads}")
    process_pool = mp.Pool(processes=num_threads)
    filtered_srcs = process_pool.starmap(latticefilter,list(zip(flat_srcs,flat_refs)))
    process_pool.close()
    process_pool.join()
    filtered_mb = torch.stack(filtered_srcs)
    return filtered_mb

class BatchedLatticeFilter(Function):
    @staticmethod
    def forward(ctx,flat_srcs,flat_refs,num_threads):
        assert flat_srcs.shape[:2] == flat_refs.shape[:2], \
            "Incompatible shapes {}, and {}".format(flat_srcs.shape,flat_refs.shape)
        ctx.save_for_backward(flat_srcs,flat_refs)
        ctx.num_threads = num_threads
        filtered_mb = batched_filter(flat_srcs,flat_refs,num_threads)
        return filtered_mb
    @staticmethod
    def backward(ctx,grad_output):
        with torch.no_grad():
            srcs, refs = ctx.saved_tensors
            num_threads = ctx.num_threads
            g = grad_output
            n,L = srcs.shape[-2:]
            d = refs.shape[-1]
            bs = srcs.shape[0]
            grad_source = grad_reference = None
            if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
                grad_source = batched_filter(g,refs,num_threads) # Matrix is symmetric
            if ctx.needs_input_grad[1]: # try torch.no_grad ()
                s = []
                s.append(time.time())
                gf = grad_and_ref = grad_output[...,None]*refs[...,None,:] # bs x n x L x d
                grad_and_ref_flat = grad_and_ref.view(grad_and_ref.shape[:-2]+(L*d,))
                sf = src_and_ref = srcs[...,None]*refs[...,None,:] # bs x n x L x d
                src_and_ref_flat = src_and_ref.view(src_and_ref.shape[:-2]+(L*d,))
                s.append(time.time())
                #bs x n x (L+Ld+L+Ld):   bs x n x L       bs x n x Ld     bs x n x L   bs x n x Ld 
                all_ = torch.cat([g,grad_and_ref_flat,srcs,src_and_ref_flat],dim=-1)
                s.append(time.time())
                filtered_all = batched_filter(all_,refs,num_threads)#ctx.W@all_#torch.randn_like(all_)#
                s.append(time.time())
                [wg,wgf,ws,wsf] = torch.split(filtered_all,[L,L*d,L,L*d],dim=-1)
                s.append(time.time())
                # has shape bs x n x d 
                grad_reference = -2*(sf*wg[...,None] - srcs[...,None]*wgf.view(bs,n,L,d) + gf*ws[...,None] - g[...,None]*wsf.view(bs,n,L,d)).sum(-2) # sum over L dimension
                if ctx.needs_input_grad[0]: grad_source = wg
                s.append(time.time())
                s = np.array(s)
            #print(f"{s[1:]-s[:-1]}")
        return grad_source, grad_reference, None # num_threads needs no grad

class LatticeFilter(Function):
    @staticmethod
    def forward(ctx, source, reference):
        #W = torch.exp(-((reference[None,:,:] - reference[:,None,:])**2).sum(-1)).double()
        #ctx.W = W
        # Typical runtime of O(nd^2 + n*L), Worst case O(nd^2 + n*L*d)
        assert source.shape[0] == reference.shape[0], \
            "Incompatible shapes {}, and {}".format(source.shape,reference.shape)
        ctx.save_for_backward(source,reference) # TODO: add batch compatibility
        s0 = time.time()
        filtered_output = latticefilter(source,reference)
        return filtered_output
    @staticmethod
    def backward(ctx,grad_output):
        # Typical runtime of O(nd^2 + 2L*n*d), Worst case  O(nd^2 + 2L*n*d^2)
        # Does not support second order autograd at the moment
        with torch.no_grad():
            src, ref = ctx.saved_tensors
            g = grad_output
            n,L = src.shape[-2:]
            d = ref.shape[-1]
            grad_source = grad_reference = None
            if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
                grad_source = latticefilter(g,ref)#ctx.W@g#latticefilter(g,ref) # Matrix is symmetric
            if ctx.needs_input_grad[1]: # try torch.no_grad ()
                s = []
                s.append(time.time())
                gf = grad_and_ref = grad_output[...,None]*ref[...,None,:] # n x L x d
                grad_and_ref_flat = grad_and_ref.view(grad_and_ref.shape[:-2]+(L*d,))
                sf = src_and_ref = src[...,None]*ref[...,None,:] # n x L x d
                src_and_ref_flat = src_and_ref.view(src_and_ref.shape[:-2]+(L*d,))
                s.append(time.time())
                #n x (L+Ld+L+Ld):   n x L       n x Ld     n x L   n x Ld 
                all_ = torch.cat([g,grad_and_ref_flat,src,src_and_ref_flat],dim=-1)
                s.append(time.time())
                filtered_all = latticefilter(all_,ref)#ctx.W@all_#torch.randn_like(all_)#
                s.append(time.time())
                [wg,wgf,ws,wsf] = torch.split(filtered_all,[L,L*d,L,L*d],dim=-1)
                s.append(time.time())
                # has shape n x d 
                grad_reference = -2*(sf*wg[...,None] - src[...,None]*wgf.view(-1,L,d) + gf*ws[...,None] - g[...,None]*wsf.view(-1,L,d)).sum(-2) # sum over L dimension
                if ctx.needs_input_grad[0]: grad_source = wg
                s.append(time.time())
                s = np.array(s)
            #print(f"{s[1:]-s[:-1]}")
        return grad_source, grad_reference
        



# class LSHGaussian(nn.Module):
#     def __init__(self,ref):
#         super().__init__()
#         self.ref = ref

#     def __matmul__(self,U):
#         return self(U)
#     # def __rmatmul__(self,U):
#     #     return self(U)

#     def forward(self,U):
#         return crf.lsh.filter(U,self.ref,5,5,30) - U

#     def backward(self,*args,**kwargs):
#         raise NotImplementedError


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
    test = gradcheck(LatticeFilter.apply,(src_arr,ref_arr),eps=1e-5,rtol=5e-4,atol=1e-5)
    print(test) # Gradients are perhaps wrong still (need to implement double precision method)
    