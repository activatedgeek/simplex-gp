# Simplex-GPs

<!-- ![](https://img.shields.io/badge/arXiv-yyyy.xxxx-red) -->
![](https://img.shields.io/badge/ICML-2021-brightgreen)

This repository hosts the code for [_SKIing on Simplices: Kernel Interpolation on the Permutohedral Lattice for Scalable Gaussian Processes_](#) (Simplex-GPs) by 
[Sanyam Kapoor](https://im.perhapsbay.es), [Marc Finzi](https://mfinzi.github.io),
[Ke Alexander Wang](https://keawang.github.io), 
[Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

## Usage

The lattice kernels are packaged as GPyTorch modules, and can be used as a 
fast approximation to either the [`RBFKernel`](https://docs.gpytorch.ai/en/stable/kernels.html#rbfkernel)
or the [`MaternKernel`](https://docs.gpytorch.ai/en/stable/kernels.html#maternkernel).
The corresponding replacement modules are `RBFLattice` and `MaternLattice`.

A complete usable GPyTorch model that uses the `RBFLattice` kernel looks like this:

```diff
import gpytorch as gp
from gpytorch_lattice_kernel import RBFLattice

class SimplexGPModel(gp.models.ExactGP):
  def __init__(self, train_x, train_y, min_noise=1e-4,
               order=1):
    likelihood = gp.likelihoods.GaussianLikelihood(
                  noise_constraint=gp.constraints.GreaterThan(min_noise))
    super().__init__(train_x, train_y, likelihood)

    self.mean_module = gp.means.ConstantMean()
    self.covar_module = gp.kernels.ScaleKernel(
-      gp.kernels.RBFKernel(ard_num_dims=train_x.size(-1))
+      RBFLattice(ard_num_dims=train_x.size(-1), order=order)
    )

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gp.distributions.MultivariateNormal(mean_x, covar_x)
```

See [train_simplexgp.py](./experiments/train_simplexgp.py) for full usage.

The [GPyTorch Regression Tutorial](https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html)
provides a simpler example on toy data, where this kernel can be used as a 
drop-in replacement.

## Install

To use the kernel in your code, install the package as:

```shell
pip install gpytorch-lattice-kernel
```

**NOTE**: The kernel is compiled lazily from source using [CMake](https://cmake.org). 
If the compilation fails, you may need to install a more recent version.

### Local Setup

For a local development setup, create the `conda` environment

```shell
$ conda env create -f environment.yml
```

Remember to add the root of the project to PYTHONPATH if not already.

```shell
$ export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

## Results

The proposed kernel can be used with GPyTorch as usual. An example script to
reproduce results is,

```shell
python experiments/train_simplexgp.py --dataset=elevators --data-dir=<path/to/uci/data>
```

We use [Fire](https://google.github.io/python-fire/guide/) to handle CLI arguments.
All arguments of the `main` function are therefore valid arguments to the CLI.

All figures in the paper can be reproduced via [notebooks](./notebooks).

**NOTE**: The UCI dataset `mat` files are available [here](https://cims.nyu.edu/~andrewgw/pattern/).
