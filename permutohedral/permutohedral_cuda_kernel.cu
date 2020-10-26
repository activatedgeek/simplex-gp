#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 1024
#define PTAccessor2D torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits>
#define Tensor2PTAccessor2D(x) x.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()

using at::Tensor;
using std::vector;

template <typename scalar_t>
__device__ __forceinline__ scalar_t* embed_vec(PTAccessor2D ref, const int n) {
  const size_t pd = ref.size(-1);
  scalar_t* elevated = new scalar_t[pd + 1];

  /** TODO: project onto Hd. **/

  return elevated;
}

template <typename scalar_t>
__global__ void splat_kernel(PTAccessor2D ref) {
  const size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= ref.size(0)) {
    return;
  }
  
  auto elevated = embed_vec<scalar_t>(ref, n);

  /** TODO: find enclosing simplex. **/

  /** TODO: find barycentric coordinates. **/

  /** TODO: lock-free add to a hashtable + bookkeeping. **/
}

template <typename scalar_t>
class PermutohedralLatticeGPU {
private:
  size_t pd, vd, N;
public:
  PermutohedralLatticeGPU(size_t pd_, size_t vd_, size_t N_): 
    pd(pd_), vd(vd_), N(N_) {
    /** TODO: init scale factors + std. deviation. **/
  }

  Tensor filter(Tensor src, Tensor ref) {
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    splat_kernel<scalar_t><<<blocks, threads>>>(
      Tensor2PTAccessor2D(ref)
    );

    /** TODO: fixme once computations completed **/
    return torch::ones_like(src);
  }
};

Tensor permutohedral_cuda_filter(Tensor src, Tensor ref) {
  Tensor out;

  AT_DISPATCH_FLOATING_TYPES(src.type(), "permutohedral_lattice", ([&]{
    PermutohedralLatticeGPU<scalar_t> lattice(ref.size(-1), src.size(-1),
                                              src.size(0));
    out = lattice.filter(src, ref);
  }));

  return out;
}
