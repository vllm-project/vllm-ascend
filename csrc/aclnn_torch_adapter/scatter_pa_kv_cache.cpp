/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */
#include <array>
#include <string>
#include <third_party/op-plugin/op_plugin/utils/op_api_common_base.h>

namespace vllm_ascend {
void scatter_pa_kv_cache_cached(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& slot_mapping,
    c10::string_view cache_mode,
    c10::string_view scatter_mode)
{
    std::string cache_mode_str(cache_mode);
    std::string scatter_mode_str(scatter_mode);
    char* cache_mode_ptr = cache_mode_str.data();
    char* scatter_mode_ptr = scatter_mode_str.data();
    c10::optional<at::Tensor> optional_tensor = c10::nullopt;
    const std::array<int64_t, 2> strides_size = {1, 1};
    const std::array<int64_t, 2> offsets_size = {0, 0};
    at::IntArrayRef strides(strides_size);
    at::IntArrayRef offsets(offsets_size);

    // Use the installed PTA extension dispatcher; retain strided cache views.
    EXEC_NPU_CMD_EXT(aclnnScatterPaKvCache,
                     key, key_cache, slot_mapping, value, value_cache,
                     optional_tensor, optional_tensor, optional_tensor,
                     cache_mode_ptr, scatter_mode_ptr, strides, offsets);
}
} // namespace vllm_ascend
