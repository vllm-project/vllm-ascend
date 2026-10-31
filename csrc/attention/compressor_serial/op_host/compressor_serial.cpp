// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>
#include <ATen/ATen.h>
#include <initializer_list>
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" void compressor_serial_cast_launch(uint32_t, void*, void*, void*, uint64_t, int);
extern "C" void compressor_serial_vector_launch(uint32_t, void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*, void*, void*, int, int, int);

namespace vllm_ascend {
void compressor_serial_cast(const at::Tensor& x, at::Tensor& out, int64_t blocks)
{
    TORCH_CHECK(x.device().is_privateuseone() && out.device() == x.device(), "NPU device mismatch");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "BF16/FP16 required");
    TORCH_CHECK(out.scalar_type() == at::kFloat && x.sizes() == out.sizes(), "invalid FP32 output");
    TORCH_CHECK(blocks > 0 && blocks <= 72, "invalid block count");
    if (!x.numel()) return;
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    compressor_serial_cast_launch(blocks, stream, x.data_ptr(), out.data_ptr(), x.numel(),
                                 x.scalar_type() == at::kBFloat16 ? 0 : 1);
}

void compressor_serial_vector(const at::Tensor& projection, at::Tensor& state, const at::Tensor& ape,
    const at::Tensor& gamma, const at::Tensor& sin, const at::Tensor& cos, const at::Tensor& table,
    const at::Tensor& cu, const at::Tensor& used, const at::Tensor& start, at::Tensor& pooled,
    at::Tensor& output, const at::Tensor& params, int64_t stage, int64_t blocks, int64_t threads)
{
    TORCH_CHECK(stage == 0 || stage == 1, "stage must be pool or finalize/update");
    TORCH_CHECK(blocks > 0 && blocks <= 72, "invalid block count");
    TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "invalid thread count");
    for (const auto* tensor : std::initializer_list<const at::Tensor*>{&projection, &state, &ape, &gamma, &sin, &cos, &pooled}) {
        TORCH_CHECK(tensor->device() == projection.device() && tensor->scalar_type() == at::kFloat,
                    "FP32 tensors on one NPU required");
    }
    for (const auto* tensor : std::initializer_list<const at::Tensor*>{&ape, &gamma, &sin, &cos, &pooled, &output}) {
        TORCH_CHECK(tensor->is_contiguous(), "contiguous vector buffers required");
    }
    TORCH_CHECK(pooled.sizes() == output.sizes(), "pooled/output shape mismatch");
    TORCH_CHECK(gamma.dim() == 1 && (gamma.numel() == 128 || gamma.numel() == 512), "invalid head dimension");
    TORCH_CHECK(state.dim() == 3 && state.stride(2) == 1 && state.stride(1) == state.size(2), "invalid cache strides");
    TORCH_CHECK(projection.device().is_privateuseone() && projection.is_contiguous(), "invalid projection");
    TORCH_CHECK(params.scalar_type() == at::kInt && params.numel() == 152 && params.is_contiguous(), "invalid metadata ABI");
    TORCH_CHECK(params.device() == projection.device() && output.device() == projection.device(), "device mismatch");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16 || output.scalar_type() == at::kHalf, "invalid output dtype");
    for (const auto* tensor : {&table, &cu, &used, &start}) {
        TORCH_CHECK(tensor->device() == projection.device() && tensor->scalar_type() == at::kInt && tensor->is_contiguous(),
                    "contiguous int32 metadata on one NPU required");
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    compressor_serial_vector_launch(blocks, stream, projection.data_ptr(), state.data_ptr(), ape.data_ptr(),
        gamma.data_ptr(), sin.data_ptr(), cos.data_ptr(), table.data_ptr(), cu.data_ptr(), used.data_ptr(),
        start.data_ptr(), pooled.data_ptr(), output.data_ptr(), params.data_ptr(), stage,
        output.scalar_type() == at::kBFloat16 ? 0 : 1, threads);
}
} // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("compressor_serial_cast(Tensor x, Tensor(a!) out, int blocks) -> ()");
    m.def("compressor_serial_vector(Tensor projection, Tensor(a!) state, Tensor ape, Tensor gamma, "
          "Tensor sin, Tensor cos, Tensor table, Tensor cu, Tensor used, Tensor start, Tensor(b!) pooled, "
          "Tensor(c!) output, Tensor params, int stage, int blocks, int threads) -> ()");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("compressor_serial_cast", &vllm_ascend::compressor_serial_cast);
    m.impl("compressor_serial_vector", &vllm_ascend::compressor_serial_vector);
}

// Both helpers mutate supplied buffers and have no tensor return values.
TORCH_LIBRARY_IMPL(_C_ascend, Meta, m) {
    m.impl("compressor_serial_cast", [](const at::Tensor&, at::Tensor&, int64_t) {});
    m.impl("compressor_serial_vector", [](const at::Tensor&, at::Tensor&, const at::Tensor&,
        const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
        const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, const at::Tensor&,
        int64_t, int64_t, int64_t) {});
}
