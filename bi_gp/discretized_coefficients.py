import numpy as np
import torch
from functools import partial
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
    print(vals)
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