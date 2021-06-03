# Simplex-GPs

![](https://img.shields.io/badge/arXiv-2021.xxxx-red)
![](https://img.shields.io/badge/ICML-2021-brightgreen)

This repository hosts the code for [_SKIing on Simplices: Kernel Interpolation on the Permutohedral Lattice for Scalable Gaussian Processes_](#) (Simplex-GPs) by 
[Sanyam Kapoor](https://im.perhapsbay.es), [Marc Finzi](https://mfinzi.github.io),
[Ke Alexander Wang](https://keawang.github.io), 
[Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

## The Idea

Building upon the approximation proposed by 
[Structured Kernel Interpolation](http://proceedings.mlr.press/v37/wilson15.pdf) (SKI),
Simplex-GPs approximate the computation of the kernel matrices by tiling the 
space using a [sparse permutohedral lattice](http://graphics.stanford.edu/papers/permutohedral/), 
instead of a rectangular grid.

![](https://i.imgur.com/rLJOe5g.png)

The matrix-vector product implied by the kernel operations in SKI are now
approximated via the three stages visualized above --- 
_splat_ (projection onto the permutohedral lattice),
_blur_ (applying the blur operation as a matrix-vector product), and
_slice_ (re-projecting back into the original space).

This alleviates the curse of dimensionality associated with SKI operations,
allowing them to scale beyond ~5 dimensions, and provides competitive advantages
in terms of runtime and memory costs, at little expense of downstream performance.
See our manuscript for complete details.

## Usage

The lattice kernels are packaged as GPyTorch modules, and can be used as a 
fast approximation to either the [`RBFKernel`](https://docs.gpytorch.ai/en/stable/kernels.html#rbfkernel)
or the [`MaternKernel`](https://docs.gpytorch.ai/en/stable/kernels.html#maternkernel).
The corresponding replacement modules are `RBFLattice` and `MaternLattice`.

`RBFLattice` kernel is simple to use by changing a single line of code:
```diff
import gpytorch as gp
from gpytorch_lattice_kernel import RBFLattice

class SimplexGPModel(gp.models.ExactGP):
  def __init__(self, train_x, train_y):
    likelihood = gp.likelihoods.GaussianLikelihood()
    super().__init__(train_x, train_y, likelihood)

    self.mean_module = gp.means.ConstantMean()
    self.covar_module = gp.kernels.ScaleKernel(
-      gp.kernels.RBFKernel(ard_num_dims=train_x.size(-1))
+      RBFLattice(ard_num_dims=train_x.size(-1), order=1)
    )

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gp.distributions.MultivariateNormal(mean_x, covar_x)
```

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
Additionally, `ninja` is required for compilation. One way to install is:

```shell
conda install -c conda-forge cmake ninja
```

### Local Setup

For a local development setup, create the `conda` environment

```shell
$ conda env create -f environment.yml
```

Remember to add the root of the project to PYTHONPATH if not already.

```shell
$ export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

### Test

To verify the code is working as expected, a simple [test file](./tests/train_snelson.py) 
is provided, that tests for the training marginal likelihood achieved by 
Simplex-GPs and Exact-GPs. Run as:

```shell
python tests/train_snelson.py
```

The [Snelson 1-D toy dataset](http://www.gatsby.ucl.ac.uk/~snelson/) is used.
A copy is available in [snelson.csv](./notebooks/snelson.csv).

## Results

The proposed kernel can be used with GPyTorch as usual. An example script to
reproduce results is,

```shell
python experiments/train_simplexgp.py --dataset=elevators --data-dir=<path/to/uci/data/mat/files>
```

We use [Fire](https://google.github.io/python-fire/guide/) to handle CLI arguments.
All arguments of the `main` function are therefore valid arguments to the CLI.

All figures in the paper can be reproduced via [notebooks](./notebooks).

**NOTE**: The UCI dataset `mat` files are available [here](https://cims.nyu.edu/~andrewgw/pattern/).
