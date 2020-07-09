#import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import setuptools
# import torch.utils.cpp_extension
# Doesn't work properly :(
# ext_modules = [setuptools.Extension(
#     name='lattice',
#     sources = ['lattice.cpp'],
#     include_dirs= ['/usr/local/lib/python3.6/site-packages/torch/lib/include', '/usr/local/lib/python3.6/site-packages/torch/lib/include/TH', '/usr/local/lib/python3.6/site-packages/torch/lib/include/THC','/usr/local/include/python3.6m/pybind11/','/usr/local/include/python3.6m/'],
#     extra_compile_args=['-std=c++11'],
#     language='c++')]

setup(
    name='lattice',
    ext_modules=[CppExtension('lattice', ['lattice.cpp'])],
    cmdclass={'build_ext':BuildExtension})

# setuptools.Extension(
#     name='lattice',
#     sources=['lattice.cpp'],
#     include_dirs=torch.utils.cpp_extension.include_paths(),
#     language='c++'
# )