/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DISPATCH_GMM_COMBINE_DECODE_BASE_H
#define DISPATCH_GMM_COMBINE_DECODE_BASE_H

#include "../common/moe_distribute_base.h"

#define TemplateMC2TypeClass typename ExpandXType, typename W1ScaleType, typename W2ScaleType, typename WType, typename ExpandIdxType, bool IsNeedReduceScatter, uint32_t EXEC_FLAG
#define TemplateMC2TypeFunc ExpandXType, W1ScaleType, W2ScaleType, WType, ExpandIdxType, IsNeedReduceScatter, EXEC_FLAG
#define TemplateDispatchTypeClass                                                                          \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist, \
        bool IsNeedAllgater, uint32_t EXEC_FLAG
#define TemplateDispatchTypeFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist, IsNeedAllgater, EXEC_FLAG

constexpr uint32_t STATE_OFFSET = 512;
constexpr uint64_t WIN_STATE_OFFSET = 512 * 1024;
constexpr uint64_t STATE_WIN_OFFSET = 900 * 1024;
constexpr uint64_t GROUP_TOKEN_NUM_OFFSET = 932 * 1024;
constexpr uint64_t SOFT_SYNC_OFFSET = 964 * 1024;
constexpr uint32_t SELF_STATE_OFFSET = 256 * 1024;
constexpr uint32_t SUM_TMP_TENSOR_SIZE = 1024;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint32_t TOKEN_EXTRA_SPACE = 512;
constexpr uint32_t INT32_COUNT_PER_BLOCK = 8;
constexpr uint32_t SOFT_SYNC_SPACE_SIZE = 512;
constexpr int64_t LOOP_TMP_SIZE = 4096;
constexpr int32_t SUB_AIV_NUM = 2;
constexpr int32_t ODD_EVEN_BASE = 2;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t GATHER_SECOND_NUM = 2;
constexpr uint32_t MAX_QUANT_ROW_ONCE = 8;
constexpr uint32_t QUANT_SPACE_FACTOR = 176 * 1024 / 11;  // up to 176KB for quant
#ifndef OPT_RANK_OFFSET
#define OPT_RANK_OFFSET 512
#endif

#define CEIL_UP(x) ((x + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN)
#define CEIL(x, y) (((x) + (y - 1)) / (y))
#define UB_BLOCK_SIZE (32)
#define GET_WIND_STATE_ADDR_BY_RANK_ID(rankId)                                                                    \
    (((epRankId == rankId)                                                                                        \
          ? ((GM_ADDR)(winContext_->localWindowsExp))                                                             \
          : ((GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsExp))) + \
     dataState * WIN_STATE_OFFSET)
#define GET_WIND_ADDR_BY_RANK_ID(rankId)                                                                         \
    (((epRankId == rankId)                                                                                       \
          ? ((GM_ADDR)(winContext_->localWindowsIn))                                                             \
          : ((GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsIn))) + \
     winDataSizeOffset + rankId * OPT_RANK_OFFSET)
#define TOKEN_FLAG_1 (0x55555555)
#define TOKEN_FLAG_2 (0x33333333)
#define V_TO_C_FLAG_1 (0x03030303)
#define V_TO_C_FLAG_2 (0x05050505)
#define CV_FLAG_INDEX 0
#define GROUP_ID_INDEX 1
#define PRE_COUNT_INDEX 2
#define SELF_COUNT_INDEX 3
#define TOTAL_COUNT_INDEX 4
#define GROUP_TOKEN_COUNT 3  // equal to SELF_COUNT_INDEX
#define GROUP_INFO_SIZE 32

__aicore__ inline static void EncreaseSyncFlag(__gm__ uint8_t *flagAddr, uint8_t idx)
{
    // flag++, like set flag
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(flagAddr + idx * SOFT_SYNC_SPACE_SIZE);
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        global);
    __asm__ __volatile__("");
    uint8_t value = global.GetValue(0);
    global.SetValue(0, value + 1);
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        global);
    __asm__ __volatile__("");
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline static void CheckSyncFlag(__gm__ uint8_t *flagAddr, uint8_t idx, uint32_t target)
{
    //  check flag, like wait flag
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(flagAddr + idx * SOFT_SYNC_SPACE_SIZE);
    while (true) {
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(global);
        __asm__ __volatile__("");
        uint8_t value = global.GetValue(0);
        if (value >= target) {
            __asm__ __volatile__("");
            AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(global);
            __asm__ __volatile__("");
            break;
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline static void CalQuantRow(const uint32_t column, uint32_t &row)
{
    row = QUANT_SPACE_FACTOR / column;
    row = row < MAX_QUANT_ROW_ONCE ? row : MAX_QUANT_ROW_ONCE;
}


#endif  // DISPATCH_GMM_COMBINE_DECODE_BASE_H
