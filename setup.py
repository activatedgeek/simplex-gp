import os
from setuptools import setup, find_packages

VERSION = os.environ.get('GITHUB_REF', '0.0.dev0')
if VERSION.startswith('refs/tags'):
  VERSION = VERSION.split('/')[-1]

with open('README.md') as f:
  README = f.read()

setup(
  name='gpytorch-lattice-kernel',
  description='Lattice kernel for scalable Gaussian processes in GPyTorch',
  long_description=README,
  long_description_content_type='text/markdown',
  version=VERSION,
  url='https://github.com/activatedgeek/simplex-gp',
  license='Apache License 2.0',
  classifiers=[
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
  ],
  packages=find_packages(exclude=[
    'configs',
    'configs.*',
    'experiments',
    'experiments.*',
    'notebooks',
    'notebooks.*',
    'tests',
    'tests.*',
  ]),
  include_package_data=True,
  python_requires='>=3.6, <4',
  install_requires=['numpy', 'torch', 'gpytorch'],
  extras_require={})
