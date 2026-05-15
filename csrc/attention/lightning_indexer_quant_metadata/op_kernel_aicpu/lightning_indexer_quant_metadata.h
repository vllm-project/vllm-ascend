/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file lightning_indexer_quant_metadata.h
 * \brief
 */

#ifndef LIGHTNING_INDEXER_QUANT_METADATA_H
#define LIGHTNING_INDEXER_QUANT_METADATA_H

#include <cstdint>

namespace optiling {
const uint32_t AIC_CORE_NUM = 36;
const uint32_t AIV_CORE_NUM = 36 * 2;
const uint32_t MAX_LD_NUM = AIC_CORE_NUM;
constexpr uint32_t LIQ_META_SIZE = 1024;
using LIQ_METADATA_T = int32_t;

namespace detail {
    // 分核功能模块输出：LD信息，包含需要归约的数据索引及其分核信息
    struct LDResult {
        // 1、归约任务的索引信息
        uint32_t ldNum = 0U;                        // 归约任务数量
        uint32_t ldBN2Idx[MAX_LD_NUM];                // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
        uint32_t ldMIdx[MAX_LD_NUM];                  // 每个归约任务的GS1索引，脚标为归约任务的序号
        uint32_t ldS2SplitNum[MAX_LD_NUM];            // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
        
        // 2、FD负载均衡阶段，归约任务的分核（vec）信息
        uint32_t ldUsedVecNum = 0U;                 // 归约过程使用的vector数量
        uint32_t ldBalanceMBaseSize = 0U;           //  规约过程中，m方向基本块大小
        uint32_t ldBalanceMSplitNum[MAX_LD_NUM];      // 每个归约任务m轴切分份数，脚标为归约任务的序号
        uint32_t ldBalanceMTailSize[MAX_LD_NUM];      // 每个归约任务m轴切分的最后一份的大小，脚标为归约任务的序号
        uint32_t ldBalanceEndIdx1[AIV_CORE_NUM];    // FD负载均衡阶段，每个vector的一级索引，脚标为vector ID，值为归约任务的ID
        uint32_t ldBalanceEndIdx2[AIV_CORE_NUM];    // FD负载均衡阶段，每个vector的二级索引，脚标为vector ID，值为归约任务的m轴切分ID
    };

    struct LiqMetaData { // __attribute__((aligned(8))) 
        uint32_t usedCoreNum = 0U;                  // 使用的核数量
        uint32_t mBaseSize = 0U;                    
        uint32_t s2BaseSize = 0U;
        uint32_t bN2End[AIC_CORE_NUM];                  // 每个核处理数据的BN2结束点
        uint32_t mEnd[AIC_CORE_NUM];                    // 每个核处理数据的M结束点
        uint32_t s2End[AIC_CORE_NUM];                   // 每个核处理数据的S2结束点
        uint32_t headLdDataIdx[AIC_CORE_NUM];           // 每个core处理的第1个归约任务的数据应存放的workspace位置
        struct LDResult ldRes;                      // LD信息
    };
};
static_assert(LIQ_META_SIZE * sizeof(LIQ_METADATA_T) >= sizeof(detail::LiqMetaData));
};

#endif