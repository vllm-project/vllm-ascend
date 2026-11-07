// SPDX-License-Identifier: Apache-2.0
// Direct-kernel registration follows the existing BGMV OpCommand path.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
namespace vllm_ascend {
void li_quant_cache_impl(void* stream, void* x, void* slots, void* keyCache,
                         void* scaleCache, void* key, void* scale,
                         int64_t capacity, int64_t bf16, int64_t slots64);

namespace {
void check_li_quant_cache_metadata(
    const at::Tensor& x, const at::Tensor& slots,
    at::Tensor keyCache, at::Tensor scaleCache)
{
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 6 && x.size(1) == 128,
                "li_quant_cache requires x[6,128]");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
                "li_quant_cache requires BF16 or FP16 input");
    TORCH_CHECK(slots.dim() == 1 && slots.numel() == 6 &&
                (slots.scalar_type() == at::kLong || slots.scalar_type() == at::kInt),
                "li_quant_cache requires INT32/INT64 slots[6]");
    TORCH_CHECK(keyCache.dim() == 2 && keyCache.size(0) > 0 &&
                keyCache.size(1) == 128 && keyCache.scalar_type() == at::kChar,
                "li_quant_cache requires INT8 key cache[capacity,128]");
    TORCH_CHECK(scaleCache.dim() == 2 && scaleCache.size(0) == keyCache.size(0) &&
                scaleCache.size(1) == 1 && scaleCache.scalar_type() == at::kHalf,
                "li_quant_cache requires FP16 scale cache[capacity,1]");
    for (const auto& tensor : {x, slots, keyCache, scaleCache}) {
        TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous(),
                    "li_quant_cache requires contiguous tensors on the same device");
    }
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> li_quant_cache(
    const at::Tensor& x, const at::Tensor& slots,
    at::Tensor keyCache, at::Tensor scaleCache)
{
    check_li_quant_cache_metadata(x, slots, keyCache, scaleCache);
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "li_quant_cache requires NPU tensors");
    c10_npu::OptionalNPUGuard guard(x.device());
    auto key = at::empty({6, 128}, x.options().dtype(at::kChar));
    auto scale = at::empty({6, 1}, x.options().dtype(at::kHalf));
    void* xPtr = x.data_ptr();
    void* slotsPtr = slots.data_ptr();
    void* keyCachePtr = keyCache.data_ptr();
    void* scaleCachePtr = scaleCache.data_ptr();
    void* keyPtr = key.data_ptr();
    void* scalePtr = scale.data_ptr();
    int64_t capacity = keyCache.size(0);
    int64_t bf16 = x.scalar_type() == at::kBFloat16;
    int64_t slots64 = slots.scalar_type() == at::kLong;
    auto stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand command;
    command.Name("li_quant_cache");
    command.SetCustomHandler([=]() -> int {
        li_quant_cache_impl(stream, xPtr, slotsPtr, keyCachePtr, scaleCachePtr,
                            keyPtr, scalePtr, capacity, bf16, slots64);
        return 0;
    });
    command.Run();
    return {key, scale};
}

std::tuple<at::Tensor, at::Tensor> li_quant_cache_meta(
    const at::Tensor& x, const at::Tensor& slots,
    at::Tensor keyCache, at::Tensor scaleCache)
{
    check_li_quant_cache_metadata(x, slots, keyCache, scaleCache);
    return {at::empty_like(x, x.options().dtype(at::kChar)),
            at::empty({x.size(0), 1}, x.options().dtype(at::kHalf))};
}
}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, library)
{
    library.def("li_quant_cache(Tensor x, Tensor slots, Tensor(a!) key_cache, "
                "Tensor(b!) scale_cache) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, library)
{
    library.impl("li_quant_cache", TORCH_FN(vllm_ascend::li_quant_cache));
}
TORCH_LIBRARY_IMPL(_C_ascend, Meta, library)
{
    library.impl("li_quant_cache", TORCH_FN(vllm_ascend::li_quant_cache_meta));
}
#endif
