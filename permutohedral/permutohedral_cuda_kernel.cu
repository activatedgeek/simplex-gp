#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

#define BLOCK_SIZE 256
#define PTAccessor2D(T) torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits>
#define Ten2PTAccessor2D(T, x) x.packed_accessor32<T,2,torch::RestrictPtrTraits>()
#define TenOpt(x) torch::dtype(x.dtype()).device(x.device().type(), x.device().index())
#define TenOptType(T, x) torch::dtype(T).device(x.device().type(), x.device().index())

using at::Tensor;

/**
 * NOTE: This block explains the comment codes used throughout.
 *  DYNALLOC: Dynamic malloc could be potentially disastrous for speed.
 *            Better patterns, contiguous shared thread memory, or pre-allocation.
 **/

template <typename scalar_t>
__global__ void splat_kernel(
    PTAccessor2D(scalar_t) ref,
    PTAccessor2D(scalar_t) E,
    PTAccessor2D(int16_t) Y,
    PTAccessor2D(int16_t) R,
    PTAccessor2D(scalar_t) B,
    const scalar_t* scaleFactor,
    const int16_t* canonical,
    int64_t* counter
  ) {
  const size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= ref.size(0)) {
    return;
  }

  const uint16_t pd = ref.size(1);
  auto pos = ref[n];
  auto elevated = E[n];
  auto y = Y[n];
  auto rank = R[n];
  auto bary = B[n];

  elevated[pd] = - pd * pos[pd - 1] * scaleFactor[pd - 1];
  for (uint16_t i = pd - 1; i > 0; i--) {
    elevated[i] = elevated[i + 1] - i * pos[i - 1] * scaleFactor[i - 1] +
                  (i + 2) * pos[i] * scaleFactor[i];
  }
  elevated[0] = elevated[1] + 2.0 * pos[0] * scaleFactor[0];

  int16_t h = 0;
  for (uint16_t i = 0; i <= pd; ++i) {
    y[i] = static_cast<int16_t>(round(elevated[i] / (pd + 1))) * (pd + 1);
    h += y[i];

    rank[i] = 0;
    bary[i] = 0.0;
  }
  h /= (pd + 1);

  bary[pd + 1] = 0.0;

  for (uint16_t i = 0; i < pd; ++i) {
    for (uint16_t j = i + 1; j <= pd; ++j) {
      if (elevated[i] - y[i] < elevated[j] - y[j]) {
        rank[i]++;
      } else {
        rank[j]++;
      }
    }
  }

  if (h > 0) {
    for (uint16_t i = 0; i <= pd; ++i) {
      if (rank[i] >= pd + 1 - h) {
          y[i] -= pd + 1;
          rank[i] += h - (pd + 1);
      }
      else {
        rank[i] += h;
      }
    }
  } else if (h < 0) {
    for (uint16_t i = 0; i <= pd; ++i) {
      if (rank[i] < -h) {
        y[i] += pd + 1;
        rank[i] += h + (pd + 1);
      } else {
        rank[i] += h;
      }
    }
  }

  for (uint16_t i = 0; i <= pd; ++i) {
    bary[pd - rank[i]] += (elevated[i] - y[i]) / (pd + 1);
    bary[pd + 1 - rank[i]] -= (elevated[i] - y[i]) / (pd + 1);
  }
  bary[0] += 1.0 + bary[pd + 1];

  /** TODO: lock-free add to a hashtable + bookkeeping. **/
  gpuAtomicAdd(counter, static_cast<int64_t>(1));
}

template <typename scalar_t>
class PermutohedralLatticeGPU {
private:
  uint16_t pd, vd;
  size_t N;
  scalar_t* scaleFactor;
  int16_t* canonical;
  int64_t* counter;
public:
  PermutohedralLatticeGPU(uint16_t pd_, uint16_t vd_, size_t N_): 
    pd(pd_), vd(vd_), N(N_) {
    
    /** TODO: Adjust this scale factor for larger kernel stencils. **/
    scalar_t invStdDev = (pd + 1) * sqrt(2.0f / 3);

    /** TODO: error checks after every malloc **/
    cudaMallocManaged(&scaleFactor, pd * sizeof(scalar_t));
    for (uint16_t i = 0; i < pd; ++i) {
      scaleFactor[i] = invStdDev / ((scalar_t) sqrt((i + 1) * (i + 2)));
    }

    cudaMallocManaged(&canonical, (pd + 1) * (pd + 1) * sizeof(int16_t));
    for (uint16_t i = 0; i <= pd; ++i) {
      for (uint16_t j = 0; j <= pd - i; ++j) {
        canonical[i * (pd + 1) + j] = i;
      }
      for (uint16_t j = pd - i + 1; j <= pd; ++j) {
        canonical[i * (pd + 1) + j] = i - (pd + 1);
      }
    }

    cudaMallocManaged(&counter, sizeof(int64_t));
    *counter = static_cast<int64_t>(0);
  }

  ~PermutohedralLatticeGPU() {
    cudaFree(scaleFactor);
    cudaFree(canonical);
    cudaFree(counter);
  }

  Tensor filter(Tensor src, Tensor ref) {
    // initialize helper tensors.
    _matE = torch::zeros({static_cast<int64_t>(N), static_cast<int64_t>(pd + 1)},
                         TenOpt(ref));
    _matY = torch::zeros({static_cast<int64_t>(N), static_cast<int64_t>(pd + 1)},
                         TenOptType(torch::kI16, ref));
    _matR = torch::zeros({static_cast<int64_t>(N), static_cast<int64_t>(pd + 1)},
                         TenOptType(torch::kI16, ref));
    _matB = torch::zeros({static_cast<int64_t>(N), static_cast<int64_t>(pd + 2)},
                         TenOpt(ref));

    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    splat_kernel<scalar_t><<<blocks, threads>>>(
      Ten2PTAccessor2D(scalar_t,ref),
      Ten2PTAccessor2D(scalar_t,_matE),
      Ten2PTAccessor2D(int16_t,_matY),
      Ten2PTAccessor2D(int16_t,_matR),
      Ten2PTAccessor2D(scalar_t,_matB),
      scaleFactor,
      canonical,
      counter
    );
    cudaDeviceSynchronize();

    std::cout << *counter << std::endl;

    /** TODO: fixme once computations completed **/
    return torch::ones_like(src);
  }
private:
  // Matrices for lattice operations.
  Tensor _matE, _matY, _matR, _matB;
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
