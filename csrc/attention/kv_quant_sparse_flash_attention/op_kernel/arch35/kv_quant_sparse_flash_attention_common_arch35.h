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
 * \file kv_quant_sparse_flash_attention_common_arch35.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_COMMON_ARCH35_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_COMMON_ARCH35_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"

#if __has_include("../../sparse_flash_attention/arch35/common/util_regbase.h")
#include "../../sparse_flash_attention/arch35/common/util_regbase.h"
#else
#include "../../../sparse_flash_attention/op_kernel/arch35/common/util_regbase.h"
#endif

#if __has_include("../../common/op_kernel/buffer.h")
#include "../../common/op_kernel/buffer.h"
#else
#include "../../common/buffer.h"
#endif
#if __has_include("../../common/op_kernel/buffer_manager.h")
#include "../../common/op_kernel/buffer_manager.h"
#else
#include "../../common/buffer_manager.h"
#endif
#if __has_include("../../common/op_kernel/buffers_policy.h")
#include "../../common/op_kernel/buffers_policy.h"
#else
#include "../../common/buffers_policy.h"
#endif

constexpr uint64_t BLOCK_BYTE = 32;
constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;

constexpr uint32_t BUFFER_SIZE_16K = 16384; // 16384表示16 * 1024
constexpr uint32_t BUFFER_SIZE_32K = 32768; // 32768表示32 * 1024
constexpr uint32_t BUFFER_SIZE_128K = 131072; // 131072表示128 * 1024

constexpr uint32_t L0AB_SHARED_SIZE_64K = 65536; // 65536表示64*1024
constexpr uint32_t L0C_SHARED_SIZE_256K = 262144; // 262144表示256 * 1024

constexpr uint32_t CV_RATIO = 2;
constexpr uint64_t SYNC_MODE = 4;

static constexpr uint32_t QSFA_SYNC_MODE0 = 0;

enum class QSFA_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BSND = 2,
};

enum class QSFATemplateMode {
    SWA_TEMPLATE_MODE = 0,
    CFA_TEMPLATE_MODE = 1,
    SCFA_TEMPLATE_MODE = 2
};

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
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool isFd, bool isPa, QSFA_LAYOUT LAYOUT_T, \
    QSFA_LAYOUT KV_LAYOUT_T, QSFATemplateMode TEMPLATE_MODE, bool IS_SPLIT_G>

#define TEMPLATE_INTF_ARGS \
    Q_T, KV_T, T, OUTPUT_T, isFd, isPa, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, IS_SPLIT_G

#define QSFA_CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(Q_T) \
    X(KV_T) \
    X(T) \
    X(OUTPUT_T) \

#define QSFA_CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(isFd, bool, false) \
    X(isPa, bool, true) \
    X(LAYOUT_T, QSFA_LAYOUT, QSFA_LAYOUT::BSND) \
    X(KV_LAYOUT_T, QSFA_LAYOUT, QSFA_LAYOUT::PA_BSND) \
    X(TEMPLATE_MODE, QSFATemplateMode, QSFATemplateMode::SCFA_TEMPLATE_MODE) \
    X(IS_SPLIT_G, bool, false)


/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type name = default_val,

#define TEMPLATES_DEF \
template <QSFA_CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
    QSFA_CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 2. 生成不带默认值的模版Template */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT \
template <QSFA_CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
    QSFA_CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. 生成有默认值的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    QSFA_CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    QSFA_CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

#endif //KV_QUANT_SPARSE_FLASH_ATTENTION_COMMON_ARCH35_H
