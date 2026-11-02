// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TORCH_KERNEL_HELPER_H
#define TORCH_KERNEL_HELPER_H

#include <ATen/ATen.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_kernel {

#define DEVICE_TYPE c10::DeviceType::PrivateUse1

class TorchNpuHelper
{
public:
    inline static at::Tensor CopyTensorHostToDevice(const at::Tensor &cpu_tensor)
    {
        at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
        int deviceIndex = 0;
        c10_npu::GetDevice(&deviceIndex);
        return cpuPinMemTensor.to(c10::Device(DEVICE_TYPE, deviceIndex), cpuPinMemTensor.scalar_type(), true, true);
    }

    inline static at::Tensor CopyScalarToDevice(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type)
    {
        return CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
    }

    inline static void *ConvertType(const at::Tensor &at_tensor)
    {
        return const_cast<void *>(at_tensor.data_ptr());
    }

    template <typename T>
    inline static T ConvertType(T value)
    {
        return value;
    }

    template <typename... Ts>
    inline static constexpr auto ConvertTypes(Ts &...args)
    {
        return std::make_tuple(ConvertType(args)...);
    }
};
}  // namespace ascend_kernel

/**
 * @brief Launch real kernel function on NPU
 *
 * @param kernel_name      [in] name of kernel
 * @param blockdim         [in] dim size of block
 */
#define EXEC_KERNEL_CMD(kernel_name, blockdim, ...)                                            \
    do {                                                                                       \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                        \
        auto converted_params = ascend_kernel::TorchNpuHelper::ConvertTypes(__VA_ARGS__); \
        auto acl_call = [acl_stream, blockdim, converted_params]() -> int {                    \
            std::apply(                                                                        \
                [&](auto &&...params) {                                                        \
                    ACLRT_LAUNCH_KERNEL(kernel_name)                                           \
                    (blockdim, acl_stream, params...);                                         \
                },                                                                             \
                converted_params);                                                             \
            return 0;                                                                          \
        };                                                                                     \
        at_npu::native::OpCommand::RunOpApi(#kernel_name, acl_call);                           \
    } while (false)

#endif  // TORCH_KERNEL_HELPER_H
