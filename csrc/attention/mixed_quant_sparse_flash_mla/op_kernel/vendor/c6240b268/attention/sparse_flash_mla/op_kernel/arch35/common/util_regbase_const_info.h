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
 * \file util_regbase_const_info.h
 * \brief sparse_flash_mla / mixed_quant_sparse_flash_mla / quant_sparse_flash_mla 三个算子共用的
 *        regbaseutil::ConstInfo 字段宏定义。各算子在各自的 util_regbase.h 中保留 struct ConstInfo
 *        声明，并按需组合以下宏，从而兼容模板化（HIGH_PERF）与非模板化两种形式。
 */

#ifndef SMLA_COMMON_UTIL_REGBASE_CONST_INFO_H
#define SMLA_COMMON_UTIL_REGBASE_CONST_INFO_H

/* 三个算子共有的字段（不含 topk/lse 相关） */
#define SMLA_CONST_INFO_COMMON_FIELDS \
    /* 8 字节对齐字段 */ \
    int64_t dSize;       /* query d 512 */ \
    int64_t dSizeV;      /* key d 512 */ \
    int64_t dSizeVInput; /* key input d */ \
    int64_t gSize;       /* g轴的大小 */ \
    int64_t n2Size; \
    int64_t s1Size;    /* s1总大小 */ \
    int64_t s2Size;    /* s2总大小 */ \
    int64_t cmpS2Size; /* s2总大小 */ \
    /* 轴的乘积 */ \
    int64_t s1Dv; \
    int64_t gS1Dv; \
    int64_t n2GS1Dv; \
    int64_t s2Dv; \
    int64_t n2S2Dv; \
    int64_t s1S2; \
    int64_t gS1; \
    int64_t gDv; \
    int64_t n2Dv; \
    int64_t n2G; \
    int64_t n2GDv; \
    int64_t s2BaseN2Dv; \
    int64_t s1BaseN2GDv; \
    /* matmul跳读参数 */ \
    int64_t mm1Ka; \
    /* dq 或者attentionOut的Stride */ \
    int64_t attentionOutStride; \
    /* 4 字节对齐字段 */ \
    uint32_t bSize; \
    uint32_t needInit; \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    uint32_t aivIdx; \
    uint32_t oriSparseBlockCount; \
    uint32_t cmpSparseBlockCount; \
    uint32_t alignedOriSparseBlockCount; \
    uint32_t alignedCmpSparseBlockCount; \
    uint32_t actualSeqLenSize; /* 用户输入的actualseq的长度 */ \
    /* service mm1 mm2 pageAttention */ \
    uint32_t oriBlockSize; \
    uint32_t cmpBlockSize; \
    uint32_t oriMaxBlockNumPerBatch; \
    uint32_t cmpMaxBlockNumPerBatch; \
    int32_t oriWinLeft; \
    int32_t oriWinRight; \
    uint32_t sparseBlockSize; \
    uint32_t cmpRatio; \
    float softmaxScale; \
    uint32_t oriMaskMode; \
    uint32_t cmpMaskMode; \
    /* 1 字节字段 */ \
    uint8_t subBlockIdx

/* nope/rope 分拆维度，mixed 使用（sparse/quant 无） */
#define SMLA_CONST_INFO_NOPE_ROPE_FIELDS \
    int64_t dSizeNope; /* key nope d 448 */ \
    int64_t dSizeRope  /* key rope d 64 */

/* sparse_flash_mla 独有字段 */
#define SMLA_CONST_INFO_SPARSE_ONLY_FIELDS \
    int64_t mm1Kb; \
    bool isActualLenDimsOriKVNull /* 判断是否有actualseq_kv */

/* KV 张量 stride，三个算子共用 */
#define SMLA_CONST_INFO_KV_STRIDE_FIELDS \
    uint32_t oriKvStride; \
    uint32_t cmpKvStride

/* topk length 相关字段，三个算子共有 */
#define SMLA_CONST_INFO_TOPK_FIELDS \
    bool hasOriTopkLength; \
    bool hasCmpTopkLength

/* lse 相关字段，三个算子共用 */
#define SMLA_CONST_INFO_LSE_FIELDS bool isSoftmaxLseEnable

#endif // SMLA_COMMON_UTIL_REGBASE_CONST_INFO_H
