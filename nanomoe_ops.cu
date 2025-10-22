#include "indices.h"
#include <torch/extension.h>

namespace nanomoe {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indices_variable", &megablocks::indices, "variable-size indices construction for sparse matrix.");
}

}  // namespace nanomoe
