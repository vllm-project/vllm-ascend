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
 * \file kv_quant_sparse_attn_sharedkv_common_arch35.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_ATTN_AHSREDKV_COMMON_ARCH35_H
#define KV_QUANT_SPARSE_ATTN_AHSREDKV_COMMON_ARCH35_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "../kv_quant_sparse_attn_sharedkv_common.h"

constexpr uint64_t BLOCK_BYTE = 32;
constexpr uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;

constexpr uint32_t L0AB_SHARED_SIZE_64K = 65536; // 65536表示64*1024
constexpr uint32_t L0C_SHARED_SIZE_256K = 262144; // 262144表示256 * 1024

constexpr uint32_t BUFFER_SIZE_16K = 16384; // 16384表示16 * 1024
constexpr uint32_t BUFFER_SIZE_32K = 32768; // 32768表示32 * 1024
constexpr uint32_t BUFFER_SIZE_128K = 131072; // 131072表示128 * 1024

constexpr uint32_t CV_RATIO = 2;
constexpr uint64_t SYNC_MODE = 4;

namespace BaseApi {
__aicore__ constexpr uint64_t Align2Func(uint64_t data) {
    return (data + 1UL) >> 1UL << 1UL; // 向上2对齐, +1移位2
}

__aicore__ constexpr uint64_t Align8Func(uint64_t data) {
    return (data + 7UL) >> 3UL << 3UL; // 向上8对齐, +7移位3
}

__aicore__ constexpr uint64_t Align16Func(uint64_t data) {
    return (data + 15UL) >> 4UL << 4UL; // 向上16对齐, +15移位4
}

__aicore__ constexpr uint64_t Align64Func(uint64_t data) {
    return (data + 63UL) >> 6UL << 6UL; // 向上64对齐, +63移位6
}
}

#define TEMPLATE_INTF \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool isFd, bool isPa, SAS_LAYOUT LAYOUT_T, \
    SAS_LAYOUT KV_LAYOUT_T, SASTemplateMode TEMPLATE_MODE, bool IS_SPLIT_G>

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
    X(LAYOUT_T, SAS_LAYOUT, SAS_LAYOUT::BSND) \
    X(KV_LAYOUT_T, SAS_LAYOUT, SAS_LAYOUT::PA_ND) \
    X(TEMPLATE_MODE, SASTemplateMode, SASTemplateMode::SCFA_TEMPLATE_MODE) \
    X(IS_SPLIT_G, bool, false) \


/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type name = default_val,

#define TEMPLATES_DEF \
template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 2. 生成不带带默认值的模版Template */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT \
template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. 生成有默认值的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

#endif
