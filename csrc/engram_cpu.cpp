// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {
constexpr int64_t kGroupSize = 32;
constexpr int64_t kParallelMinRows = 256;
constexpr int64_t kParallelGrain = 128;

#if defined(__aarch64__)
uint16x4_t to_bfloat16(float32x4_t values) {
    const auto bits = vreinterpretq_u32_f32(values);
    const auto bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1)));
    const auto rounded = vshrn_n_u32(vaddq_u32(bits, bias), 16);
    // Match c10::BFloat16's canonical NaN and round-to-nearest-even.
    return vbsl_u16(vmovn_u32(vceqq_f32(values, values)), rounded, vdup_n_u16(0x7fc0));
}
#endif

void engram_int8_lookup_cpu(const at::Tensor& weight, const at::Tensor& scale,
                          const at::Tensor& ids, at::Tensor& output) {
    for (const auto& tensor : {weight, scale, ids, output}) {
        TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous(),
                    "Engram CPU lookup requires contiguous CPU tensors");
    }
    TORCH_CHECK(weight.scalar_type() == at::kChar && scale.scalar_type() == at::kFloat &&
                ids.scalar_type() == at::kLong && output.scalar_type() == at::kBFloat16,
                "Engram CPU lookup expects INT8 weight, FP32 scale, INT64 IDs and BF16 output");
    TORCH_CHECK(weight.dim() == 2 && scale.dim() == 2 && ids.dim() == 1 && output.dim() == 2,
                "Invalid Engram CPU lookup dimensions");
    const auto rows = weight.size(0);
    const auto width = weight.size(1);
    const auto groups = width / kGroupSize;
    TORCH_CHECK(width > 0 && width % kGroupSize == 0 && scale.size(0) == rows &&
                scale.size(1) == groups && output.size(0) == ids.numel() && output.size(1) == width,
                "Invalid Engram CPU lookup shapes");
    const auto* codes = weight.const_data_ptr<int8_t>();
    const auto* scales = scale.const_data_ptr<float>();
    const auto* indices = ids.const_data_ptr<int64_t>();
    auto* out = output.mutable_data_ptr<at::BFloat16>();
    // Tiny decode batches do not amortize the thread-pool handoff.
    const auto grain = ids.numel() < kParallelMinRows ? ids.numel() : kParallelGrain;
    at::parallel_for(0, ids.numel(), grain, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const auto row = indices[i];
            TORCH_CHECK_INDEX(row >= 0 && row < rows, "Engram CPU lookup ID outside shard");
            for (int64_t g = 0; g < groups; ++g) {
                const float factor = scales[row * groups + g];
#if defined(__aarch64__)
                for (int64_t j = 0; j < kGroupSize; j += 8) {
                    const auto column = g * kGroupSize + j;
                    const auto values = vmovl_s8(vld1_s8(codes + row * width + column));
                    const auto low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(values)));
                    const auto high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(values)));
                    auto* destination = reinterpret_cast<uint16_t*>(out + i * width + column);
                    vst1_u16(destination, to_bfloat16(vmulq_n_f32(low, factor)));
                    vst1_u16(destination + 4, to_bfloat16(vmulq_n_f32(high, factor)));
                }
#else
                for (int64_t j = 0; j < kGroupSize; ++j) {
                    const auto column = g * kGroupSize + j;
                    out[i * width + column] = at::BFloat16(float(codes[row * width + column]) * factor);
                }
#endif
            }
        }
    });
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("engram_int8_lookup_cpu(Tensor weight, Tensor scale, Tensor ids, Tensor(a!) output) -> ()");
}

TORCH_LIBRARY_IMPL(_C_ascend, CPU, m) {
    m.impl("engram_int8_lookup_cpu", &engram_int8_lookup_cpu);
}
