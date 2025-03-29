#include <torch/library.h>
#include <torch/extension.h>

#include "torch_npu/csrc/aten/common/from_blob.h"

namespace {
    torch::Tensor weak_ref_tensor(torch::Tensor& tensor)
    {
        // Ensure tensor is on NPU
        if (tensor.is_privateuseone()) {
          throw std::runtime_error("Tensor must be on NPU device");
        }

        // Get the raw data pointer
        void* data_ptr = tensor.data_ptr();

        // Get tensor sizes and strides
        std::vector<int64_t> sizes = tensor.sizes().vec();
        std::vector<int64_t> strides = tensor.strides().vec();

        // Get tensor options (dtype, device)
        auto options = tensor.options();

        // Create a new tensor from the raw data pointer
        auto new_tensor = at_npu::native::from_blob(data_ptr, sizes, strides, options);

        return new_tensor;
    }
}

// TORCH_LIBRARY(_C, m) {
//     m.def("weak_ref_tensor", &weak_ref_tensor);
// }

TORCH_LIBRARY_IMPL(_C, PrivateUse1, m) {
    m.impl("weak_ref_tensor", &weak_ref_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weak_ref_tensor", &weak_ref_tensor, "return a weak ref for npu tensor");
}
