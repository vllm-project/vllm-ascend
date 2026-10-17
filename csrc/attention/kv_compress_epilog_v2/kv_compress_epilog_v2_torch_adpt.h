/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef VLLM_ASCEND_KV_COMPRESS_EPILOG_V2_TORCH_ADPT_H
#define VLLM_ASCEND_KV_COMPRESS_EPILOG_V2_TORCH_ADPT_H
#include <cctype>
#include <string>
#include <torch/library.h>

namespace vllm_ascend {
namespace kv_compress_v2 {
using namespace at_npu::native;

namespace {
// 与算子侧 IR/tiling 一致的维度与量化约束常量。
constexpr int64_t KCEV2_CACHE_RANK_2D = 2;
constexpr int64_t KCEV2_CACHE_RANK_4D = 4;
constexpr int64_t KCEV2_CACHE_UNIT_DIM_INDEX = 2;
constexpr int64_t KCEV2_CACHE_LAST_DIM_INDEX = 3;
constexpr int64_t KCEV2_QUANT_GROUP_16 = 16;
constexpr int64_t KCEV2_QUANT_GROUP_32 = 32;

std::string NormalizeQuantMode(const std::string &quantMode) {
  size_t begin = 0;
  size_t end = quantMode.size();
  while (begin < end &&
         std::isspace(static_cast<unsigned char>(quantMode[begin]))) {
    ++begin;
  }
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(quantMode[end - 1]))) {
    --end;
  }

  std::string mode;
  mode.reserve(end - begin);
  for (size_t i = begin; i < end; ++i) {
    mode.push_back(static_cast<char>(
        std::tolower(static_cast<unsigned char>(quantMode[i]))));
  }
  if (mode == "mxfp8_bf16" || mode == "mxfp4_bf16") {
    return mode;
  }
  TORCH_CHECK(false,
              "quant_mode should be one of [mxfp8_bf16, mxfp4_bf16], but got '",
              quantMode, "'");
}

void ValidateKvCompressEpilogV2Inputs(at::Tensor &cache, const at::Tensor &x,
                                      const at::Tensor &slotMapping,
                                      int64_t quantGroupSize,
                                      const std::string &quantMode,
                                      double xScale) {
    TORCH_CHECK(cache.dim() == KCEV2_CACHE_RANK_2D || cache.dim() == KCEV2_CACHE_RANK_4D,
                "cache must be 2D or 4D, but got rank ", cache.dim());
    if (cache.dim() == KCEV2_CACHE_RANK_4D) {
        TORCH_CHECK(cache.size(KCEV2_CACHE_UNIT_DIM_INDEX) == 1,
                    "4D cache dim 2 must be 1, but got ", cache.size(KCEV2_CACHE_UNIT_DIM_INDEX));
        TORCH_CHECK(cache.stride(KCEV2_CACHE_LAST_DIM_INDEX) == 1,
                    "4D cache last dimension must be contiguous, but stride(3)=",
                    cache.stride(KCEV2_CACHE_LAST_DIM_INDEX));
    }
    TORCH_CHECK(x.dim() == KCEV2_CACHE_RANK_2D, "x must be 2D, but got rank ", x.dim());
    TORCH_CHECK(slotMapping.dim() == 1, "slot_mapping must be 1D, but got rank ",
                slotMapping.dim());
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0, "x dimensions must be positive");
    TORCH_CHECK(slotMapping.size(0) == x.size(0),
                "slot_mapping length must equal x dim 0, got ",
                slotMapping.size(0), " and ", x.size(0));
    TORCH_CHECK(x.scalar_type() == at::kBFloat16,
                "x dtype must be bfloat16, got ", x.scalar_type());
    TORCH_CHECK(slotMapping.scalar_type() == at::kInt ||
                    slotMapping.scalar_type() == at::kLong,
                "slot_mapping dtype must be int32 or int64, got ",
                slotMapping.scalar_type());
    TORCH_CHECK(x.is_contiguous() && slotMapping.is_contiguous(),
                "x and slot_mapping must be contiguous");
    if (cache.dim() == KCEV2_CACHE_RANK_2D) {
        TORCH_CHECK(cache.is_contiguous(),
                    "2D cache must remain contiguous for the existing layout1 path");
    }
    const bool validModeAndGroup =
        (quantMode == "mxfp8_bf16" && quantGroupSize == KCEV2_QUANT_GROUP_32) ||
        (quantMode == "mxfp4_bf16" &&
         (quantGroupSize == KCEV2_QUANT_GROUP_16 || quantGroupSize == KCEV2_QUANT_GROUP_32));
    TORCH_CHECK(validModeAndGroup,
        "invalid quant_mode and quant_group_size combination: "
        "supported combinations are (mxfp8_bf16, 32), "
        "(mxfp4_bf16, 32), and (mxfp4_bf16, 16), but got ",
        quantMode, " and group size ", quantGroupSize);
    TORCH_CHECK(quantGroupSize != 0 && x.size(1) % quantGroupSize == 0,
        "x last dimension must be divisible by quant_group_size, got d=",
        x.size(1), " and quant_group_size=", quantGroupSize);
  TORCH_CHECK(x.size(1) <= 8192, "x last dimension must not exceed 8192, got ",
              x.size(1));
  TORCH_CHECK(xScale == 1.0, "x_scale is reserved and must be 1.0, got ",
              xScale);

  const bool fp8Cache = cache.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                        cache.scalar_type() == at::ScalarType::Float8_e5m2;
  TORCH_CHECK((quantMode == "mxfp8_bf16" && fp8Cache) ||
                  (quantMode == "mxfp4_bf16" && cache.scalar_type() == at::kByte),
              "cache dtype does not match quant_mode: mxfp8_bf16 requires "
              "float8_e4m3fn/float8_e5m2 and mxfp4_bf16 requires uint8");
}
} // namespace

void KvCompressEpilogV2Npu(at::Tensor &cache, const at::Tensor &x,
                           const at::Tensor &slotMapping,
                           int64_t quantGroupSize, std::string quantMode,
                           bool roundScale, double xScale) {
  const std::string normalizedQuantMode = NormalizeQuantMode(quantMode);
  ValidateKvCompressEpilogV2Inputs(cache, x, slotMapping, quantGroupSize,
                                   normalizedQuantMode, xScale);
  const int64_t quantModeInt = normalizedQuantMode == "mxfp8_bf16" ? 2 : 4;
  const int64_t blockStride = cache.stride(0);
  EXEC_NPU_CMD(aclnnKvCompressEpilogV2, cache, x, slotMapping,
                  quantGroupSize, quantModeInt, roundScale, xScale, blockStride);
}

} // namespace kv_compress_v2
} // namespace vllm_ascend


#endif
