#include <torch/torch.h>
#include <ATen/ATen.h>
//#include "PermutohedralLatticeCPU.h"
#include "permutohedral.h"

at::Tensor filter(at::Tensor src, at::Tensor ref) {
    //at::Tensor a = at::zeros_like(z);
    at::Tensor out = PermutohedralLattice::filter(src,ref);
    return out;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("filter", &filter, "lattice filter");
}