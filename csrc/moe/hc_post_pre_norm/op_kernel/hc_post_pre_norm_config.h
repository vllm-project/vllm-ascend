/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef VLLM_ASCEND_HC_POST_PRE_NORM_CONFIG_H
#define VLLM_ASCEND_HC_POST_PRE_NORM_CONFIG_H

#include <cstddef>
#include <cstdint>

namespace HcPostPreNorm {
constexpr int64_t kHcMult = 4;
constexpr int64_t kHiddenSize = 5120;

constexpr std::size_t kFp32OutputIndex = 5;
constexpr std::size_t kReturnFp32AttrIndex = 5;
constexpr uint64_t kWithFp32TilingKey = 2;
constexpr uint64_t kWithoutFp32TilingKey = 3;
}  // namespace HcPostPreNorm

#endif
