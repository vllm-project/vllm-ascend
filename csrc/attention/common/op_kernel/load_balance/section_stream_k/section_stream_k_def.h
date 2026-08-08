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
 * \file section_stream_k_def.h
 * \brief SectionStreamK结构体定义
 */

#ifndef SECTION_STREAM_K_DEF_H
#define SECTION_STREAM_K_DEF_H

#include <vector>
#include "../base_info.h"
#include "../load_balance_common.h"

namespace load_balance {

/******************************** RETURN CODE ********************************/
#define SECTION_STREAM_K_SUCCESS                        0
#define SECTION_STREAM_K_ERROR_INVALID_PARAM            1
#define SECTION_STREAM_K_ERROR_AIV_LESS_THAN_AIC        2
/****************************************************************************/

struct SectionStreamKParam : GeneralBalanceParam {
    uint32_t l2Byte { 128 * 1024U * 1024U };              // L2 size in Byte, 0 means infinite large
    int64_t v0Cost { 0 };
};

struct SectionStreamKFaResult {
    uint32_t usedCoreNum { 0U };                            // FA中使用的AIC数量
    std::vector<uint32_t> bN2End {};                        // 每个核处理数据的BN2结束点
    std::vector<uint32_t> gS1End {};                        // 每个核处理数据的GS1结束点
    std::vector<uint32_t> s2End {};                         // 每个核处理数据的S2结束点
    std::vector<uint32_t> firstFdDataWorkspaceIdx {};       // FD块在workspace中的位置

    explicit SectionStreamKFaResult(uint32_t aicNum)
        : bN2End(aicNum),
          gS1End(aicNum),
          s2End(aicNum),
          firstFdDataWorkspaceIdx(aicNum) {}
};

struct SectionStreamKFdResult {
    uint32_t usedVecNum { 0U };                 // 归约过程中使用的AIV数量
    // 1、归约任务的索引信息
    std::vector<uint32_t> bN2Idx {};            // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
    std::vector<uint32_t> gS1Idx {};            // 每个归约任务的GS1索引，脚标为归约任务的序号
    std::vector<uint32_t> workspaceIdx {};      // 每个归约任务在workspace中的存放位置
    std::vector<uint32_t> s2SplitNum {};        // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
    // 2、FD kernel阶段，归约任务的分核信息
    std::vector<uint32_t> taskIdx {};           // 每个AIV处理的归约任务的对应ID
    std::vector<uint32_t> mStart {};            // 每个AIV处理的归约任务的M轴相对起点
    std::vector<uint32_t> mLen {};              // 每个AIV处理的归约任务的M轴行数

    explicit SectionStreamKFdResult(uint32_t aicNum, uint32_t aivNum)
        : bN2Idx(aicNum),
          gS1Idx(aicNum),
          workspaceIdx(aicNum),
          s2SplitNum(aicNum),
          taskIdx(aivNum),
          mStart(aivNum),
          mLen(aivNum) {}
};

struct SectionStreamKResult {
    uint32_t sectionNum { 0U };
    std::vector<SectionStreamKFaResult> sectionFaResult {};
    std::vector<SectionStreamKFdResult> sectionFdResult {};
};

}

#endif