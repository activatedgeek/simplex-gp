#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using at::Tensor;

// CUDA kernel forward declarations
Tensor permutohedral_cuda_filter(Tensor src, Tensor ref, const Tensor coeffs);

Tensor filter(Tensor src, Tensor ref, const Tensor coeffs) {
  CHECK_INPUT(src);
  CHECK_INPUT(ref);
  CHECK_INPUT(coeffs);

  return permutohedral_cuda_filter(src, ref, coeffs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("filter", &filter, "Permutohedral Lattice Filter (CUDA)");
}