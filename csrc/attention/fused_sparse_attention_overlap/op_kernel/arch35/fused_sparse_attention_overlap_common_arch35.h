/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_sparse_attention_overlap_common_arch35.h
 * \brief
 */
#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_COMMON_ARCH35_H
#define FUSED_SPARSE_ATTENTION_OVERLAP_COMMON_ARCH35_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"

constexpr uint64_t BLOCK_BYTE = 32;
constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;

constexpr uint32_t L0AB_SHARED_SIZE_64K = 65536; // 65536 represents 64 * 1024
constexpr uint32_t L0C_SHARED_SIZE_256K = 262144; // 262144 represents 256 * 1024

constexpr uint32_t BUFFER_SIZE_16K = 16384; // 16384 represents 16 * 1024
constexpr uint32_t BUFFER_SIZE_32K = 32768; // 32768 represents 32 * 1024
constexpr uint32_t BUFFER_SIZE_128K = 131072; // 131072 represents 128 * 1024

constexpr uint32_t CV_RATIO = 2;
constexpr uint64_t SYNC_MODE = 4;

static constexpr uint32_t FSA_ARCH35_SYNC_MODE0 = 0;

enum class FusedSparseAttentionOverlapLayoutArch35 {
    BSND = 0,
    TND = 1,
    PA_BSND = 2,
};

enum class FusedSparseAttentionOverlapTemplateModeArch35 {
    C_TEMPLATE_MODE = 0,
    V_TEMPLATE_MODE = 1
};

namespace BaseApi {
__aicore__ constexpr uint64_t Align2Func(uint64_t data)
{
    return (data + 1UL) >> 1UL << 1UL; // Align up to 2 by adding 1 and shifting by 1
}

__aicore__ constexpr uint64_t Align8Func(uint64_t data)
{
    return (data + 7UL) >> 3UL << 3UL; // Align up to 8 by adding 7 and shifting by 3
}

__aicore__ constexpr uint64_t Align16Func(uint64_t data)
{
    return (data + 15UL) >> 4UL << 4UL; // Align up to 16 by adding 15 and shifting by 4
}

__aicore__ constexpr uint64_t Align64Func(uint64_t data)
{
    return (data + 63UL) >> 6UL << 6UL; // Align up to 64 by adding 63 and shifting by 6
}
}

#define TEMPLATE_INTF \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool isFd, bool isPa, \
    FusedSparseAttentionOverlapLayoutArch35 LAYOUT_T, FusedSparseAttentionOverlapLayoutArch35 KV_LAYOUT_T, \
    FusedSparseAttentionOverlapTemplateModeArch35 TEMPLATE_MODE, bool IS_SPLIT_G>

#define TEMPLATE_INTF_ARGS \
    Q_T, KV_T, T, OUTPUT_T, isFd, isPa, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, IS_SPLIT_G

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(Q_T) \
    X(KV_T) \
    X(T) \
    X(OUTPUT_T) \

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(isFd, bool, false) \
    X(isPa, bool, true) \
    X(LAYOUT_T, FusedSparseAttentionOverlapLayoutArch35, FusedSparseAttentionOverlapLayoutArch35::BSND) \
    X(KV_LAYOUT_T, FusedSparseAttentionOverlapLayoutArch35, FusedSparseAttentionOverlapLayoutArch35::PA_BSND) \
    X(TEMPLATE_MODE, FusedSparseAttentionOverlapTemplateModeArch35, \
        FusedSparseAttentionOverlapTemplateModeArch35::V_TEMPLATE_MODE) \
    X(IS_SPLIT_G, bool, false)


/* 1. Generate template parameters with default values. */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type name = default_val,

#define TEMPLATES_DEF \
template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 2. Generate template parameters without default values. */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT \
template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. Generate arguments with default values. */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

#endif // FUSED_SPARSE_ATTENTION_OVERLAP_COMMON_ARCH35_H
