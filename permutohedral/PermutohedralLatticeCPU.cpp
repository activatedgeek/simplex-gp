#include <torch/extension.h>
#include "PermutohedralLatticeCPU.h"

typedef float float_type;

at::Tensor filter(at::Tensor src, at::Tensor ref) {
  int vd = src.size(-1);
  int pd = ref.size(-1);
  int n = src.size(0);

  auto src_acc = src.accessor<float_type, 2>();
  auto ref_acc = ref.accessor<float_type, 2>();

  float_type *src_arr = new float_type[n * vd];
  float_type *ref_arr = new float_type[n * pd];
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t d = 0; d < vd; ++d) {
      src_arr[i * vd + d] = src_acc[i][d];
    }
    for (int64_t d = 0; d < pd; ++d) {
      ref_arr[i * pd + d] = ref_acc[i][d];
    }
  }

  auto lattice = PermutohedralLatticeCPU<float_type>(pd, vd, n);
  
  float_type *out_arr = new float_type[n * vd];
  lattice.filter(out_arr, src_arr, ref_arr);

  delete [] src_arr;
  delete [] ref_arr;
  // delete [] out_arr;

  return torch::from_blob(out_arr, {n, vd}, torch::dtype(torch::kFloat32));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("filter", &filter, "Permutohedral Lattice filter");
}