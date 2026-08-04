/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_tiling_datay.h
 * \brief
 */

#ifndef MLA_PROLOG_TILING_DATA_H
#define MLA_PROLOG_TILING_DATA_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
// 1. 基础参数结构体（对应 MlaPrologBaseParams 宏定义）
struct MlaPrologBaseParams {
    uint32_t batchSize;     // batch size（批大小）
    uint32_t stepBatchSize; // batch size per step 32（每步批大小）
    uint32_t mSubSize;
    uint32_t mSubCoreNum;
    uint32_t stepNumHeadDequant; // head size per step when dequant before mmqn（反量化前每步头大小）
    uint32_t tokenSize;          // token size = batchSize * seq1Size（token总数：批大小×序列1长度）
    uint32_t seq1Size;           // seq1（序列1长度，通常为query序列长度）
    uint32_t seq2Size;           // seq2（序列2长度，通常为key/value序列长度）
    uint32_t headSizeX;          // head size of Input Hidden 7168（输入隐藏层的头维度）
    uint32_t headSizeCq;         // head size of Latent Query 1536（潜在Query的头维度）
    uint32_t headSizeCkv;        // head size of Latent KeyValue 512（潜在KeyValue的头维度）
    uint32_t headSizeQc;         // head size of Query = dimHeadSizeQc * numHeadSize = 128 * 32（Query总头维度）
    uint32_t headSizeQr;         // head size of Query Rope = dimHeadRope * numHeadSize = 64 * 32（带RoPE的Query头维度）
    uint32_t headSizeKr;         // head size of Key Rope 64（带RoPE的Key头维度）
    uint32_t numHeadSize;        // number of head 32（头数量）
    uint32_t numHeadKvSize;      // number of headkv（KeyValue的头数量）
    uint32_t dimHeadSizeQc;      // dim size per query head 128（单个Query头的维度）
    uint32_t dimHeadRope;        // dim size per rope head 64（单个带RoPE头的维度）
    uint32_t blockNum;           // pa block num（PA格式的块数量）
    uint32_t blockSize;          // pa block size 128（PA格式的块大小）
    uint32_t mm1BlockNum;        // 24  Cq（矩阵乘1的块数量，对应Cq计算）
    uint32_t mm2BlockNum;        // 9   Ckv（矩阵乘2的块数量，对应Ckv计算）
    uint32_t mm3BlockNum;        // 24  QcQr（矩阵乘3的块数量，对应QcQr计算）
    uint32_t mm4BlockNum;        // 24  Qn（矩阵乘4的块数量，对应Qn计算）
    uint32_t vectorBlockNum;     // 32（向量计算的块数量）
    uint32_t mm1SingleCoreN;     // single headSizeCq（单核心矩阵乘1的N维度大小，对应单个Cq头维度）
    uint32_t mm2SingleCoreN;     // single headSizeCkv+headSizeKr（单核心矩阵乘2的N维度大小，Ckv+Kr头维度之和）
    uint32_t mm3SingleCoreN;     // single headSizeQc+headSizeQr（单核心矩阵乘3的N维度大小，Qc+Qr头维度之和）
    uint32_t mm4SingleCoreBatch; // single numHeadSize（单核心矩阵乘4的批大小，对应单个头数量）
    uint32_t dtileSize;
    uint32_t kvQuantMode;
    uint32_t tileSize;
    uint32_t ckvkrRepoMode;
    uint32_t quantScaleRepoMode;
    uint32_t queryNormFlag;
    float reciprocalCq;  // 1 / headSizeCq（headSizeCq的倒数，用于快速计算）
    float epsilonCq;     // Cq计算的epsilon（数值稳定性参数）
    float reciprocalCkv; // 1 / headSizeCkv（headSizeCkv的倒数，用于快速计算）
    float epsilonCkv;    // Ckv计算的epsilon（数值稳定性参数）
    float kNopeClipAlpha;
    float qcQrScale;            // query 的尺度矫正因子
    float kcScale;              // kv 的尺度矫正因子
    uint16_t isQcQrScaleEnable; // query 的尺度矫正因子是否生效（默认是1.0的时候不生效）
    uint16_t isKcScaleEnable;   // kv 的尺度矫正因子是否生效（默认是1.0的时候不生效）
};

// 2. 完整分块数据结构体（对应 MlaPrologTilingData 宏定义，嵌套基础参数）
struct MlaPrologTilingData {
    MlaPrologBaseParams baseParams; // 嵌套基础参数结构体（包含维度、头信息等核心配置）
};
} // namespace optiling

#endif // MLA_PROLOG_TILING_DATA_H