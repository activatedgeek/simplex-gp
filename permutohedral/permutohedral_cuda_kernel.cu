#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 1024
#define PTAccessor2D torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits>
#define Tensor2PTAccessor2D(x) x.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()

using at::Tensor;
using std::vector;

/**
 * NOTE: This block explains the comment codes used throughout.
 *  DYNALLOC: Dynamic malloc could be potentially disastrous for speed.
 *            Need better pattern -- shared thread memory, or pre-allocation.
 **/

template <typename scalar_t>
__global__ void splat_kernel(
    PTAccessor2D ref,
    const scalar_t* scaleFactor
  ) {
  const size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= ref.size(0)) {
    return;
  }

  auto pos = ref[n];
  const char16_t pd = ref.size(1);

  /** WARN: DYNALLOC. **/
  scalar_t* elevated = new scalar_t[pd + 1];
  scalar_t* y = new scalar_t[pd + 1];
  scalar_t* bary = new scalar_t[pd + 2];
  char16_t* rank = new char16_t[pd + 1];

  elevated[pd] = - pd * pos[pd - 1] * scaleFactor[pd - 1];
  for (char16_t i = pd - 1; i > 0; i--) {
    elevated[i] = elevated[i + 1] - i * pos[i - 1] * scaleFactor[i - 1] +
                  (i + 2) * pos[i] * scaleFactor[i];
  }
  elevated[0] = elevated[1] + 2.0 * pos[0] * scaleFactor[0];

  size_t h = 0;
  for (char16_t i = 0; i <= pd; ++i) {
    y[i] = round(elevated[i] / (pd + 1)) * (pd + 1);
    rank[i] = 0;
    bary[i] = 0.0;

    h += y[i];
  }
  h /= (pd + 1);
  bary[pd + 1] = 0.0;

  for (char16_t i = 0; i < pd; i++) {
    for (char16_t j = i + 1; j <= pd; j++) {
      if (elevated[i] - y[i] < elevated[j] - y[j]) {
        rank[i]++;
      } else {
        rank[j]++;
      }
    }
  }

  if (h > 0) {
    for (char16_t i = 0; i <= pd; i++) {
      if (rank[i] >= pd + 1 - h){
          y[i] -= pd + 1;
          rank[i] += h - (pd + 1);
      }
      else {
        rank[i] += h;
      }
    }
  } else if (h < 0) {
    for (char16_t i = 0; i <= pd; i++) {
      if (rank[i] < -h) {
        y[i] += pd + 1;
        rank[i] += (pd + 1) + h;
      } else {
        rank[i] += h;
      }
    }
  }

  for (char16_t i = 0; i <= pd; i++) {
    bary[pd - rank[i]] += (elevated[i] - y[i]) / (pd + 1);
    bary[pd + 1 - rank[i]] -= (elevated[i] - y[i]) / (pd + 1);
  }
  bary[0] += 1.0f + bary[pd + 1];

  /** TODO: lock-free add to a hashtable + bookkeeping. **/

  delete [] elevated;
  delete [] rank;
}

template <typename scalar_t>
class PermutohedralLatticeGPU {
private:
  char16_t pd, vd;
  size_t N;
  scalar_t* scaleFactor;
public:
  PermutohedralLatticeGPU(char16_t pd_, char16_t vd_, size_t N_): 
    pd(pd_), vd(vd_), N(N_) {
    
    /** TODO: Adjust this scale factor for larger kernel stencils. **/
    scalar_t invStdDev = (pd + 1) * sqrt(2.0f / 3);

    cudaMallocManaged(&scaleFactor, pd * sizeof(scalar_t));
    for (char16_t i = 0; i < pd; ++i) {
      scaleFactor[i] = invStdDev / ((scalar_t) sqrt((i + 1) * (i + 2)));
    }
  }

  ~PermutohedralLatticeGPU() {
    cudaFree(scaleFactor);
  }

  Tensor filter(Tensor src, Tensor ref) {
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    std::cout << pd << " " << vd << " " << N << std::endl;

    splat_kernel<scalar_t><<<blocks, threads>>>(
      Tensor2PTAccessor2D(ref),
      scaleFactor
    );

    /** TODO: fixme once computations completed **/
    return torch::ones_like(src);
  }
};

Tensor permutohedral_cuda_filter(Tensor src, Tensor ref) {
  Tensor out;

  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "permutohedral_lattice", ([&]{
    PermutohedralLatticeGPU<scalar_t> lattice(ref.size(-1), src.size(-1),
                                              src.size(0));
    out = lattice.filter(src, ref);
  }));

  return out;
}
