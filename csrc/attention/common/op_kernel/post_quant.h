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
 * \file post_quant.h
 * \brief
 */
#ifndef POST_QUANT_H
#define POST_QUANT_H
#include "memory_copy.h"
#include "vector_common.h"

struct PostQuantInfo_V2 {
    uint32_t gSize;
    uint32_t dSize;
    uint32_t s1Size; // actualS1
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t gS1DealSize;
    uint32_t colCount;
};

template <UbFormat UB_FORMAT>
class PostQuant {
public:
    template <typename PARAM_T, GmFormat GM_FORMAT>
    __aicore__ inline void InitPerChannel(FaGmTensor<PARAM_T, GM_FORMAT> &srcTensor, __gm__ uint8_t *quantParam,
        uint32_t n2Size, uint32_t gSize, uint32_t dSize)
    {
        GlobalTensor<PARAM_T> quantParamGm;
        quantParamGm.SetGlobalBuffer((__gm__ PARAM_T*)quantParam);
        srcTensor.gmTensor = quantParamGm;
        srcTensor.offsetCalculator.Init(n2Size, gSize, dSize);
    }

    __aicore__ inline void InitPerTensor(float &value, __gm__ uint8_t *quantParam, bool isQuant2Bf16)
    {
        if (isQuant2Bf16) {
            GlobalTensor<bfloat16_t> quantParamGm;
            quantParamGm.SetGlobalBuffer((__gm__ bfloat16_t*)quantParam);
            value = ToFloat(quantParamGm.GetValue(0));
        } else {
            GlobalTensor<float> quantParamGm;
            quantParamGm.SetGlobalBuffer((__gm__ float*)quantParam);
            value = quantParamGm.GetValue(0);
        }
    }

    template <typename PARAM_T, GmFormat GM_FORMAT>
    __aicore__ inline void CopyParamsGmToUb(LocalTensor<PARAM_T> &dstUb, FaGmTensor<PARAM_T, GM_FORMAT> &srcTensor,
                                            PostQuantInfo_V2 &postQuantInfo)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;

        if constexpr (UB_FORMAT == UbFormat::S1G) {
            uint32_t s1IdxStart = postQuantInfo.gS1Idx / offsetCalculator.GetDimG();
            uint32_t gIdxStart = postQuantInfo.gS1Idx % offsetCalculator.GetDimG();
            uint32_t s1IdxEnd = (postQuantInfo.gS1Idx + postQuantInfo.gS1DealSize) / offsetCalculator.GetDimG();
            uint32_t gIdxEnd = (postQuantInfo.gS1Idx + postQuantInfo.gS1DealSize) % offsetCalculator.GetDimG();

            if (s1IdxEnd - s1IdxStart > 1) {
                // 存在完整中间段, 拷贝完整G
                uint64_t offset = offsetCalculator.GetOffset(postQuantInfo.n2Idx, 0, 0);
                uint32_t blockCount = offsetCalculator.GetDimG();
                CopySingleMatrixNDToND<PARAM_T>(dstUb, srcTensor.gmTensor[offset], offsetCalculator.GetDimG(),
                    offsetCalculator.GetDimD(), offsetCalculator.GetStrideG(), postQuantInfo.colCount);
            } else {
                // 处理第一段S1
                uint32_t headSize = 0;
                if (s1IdxStart == s1IdxEnd) {
                    headSize = gIdxEnd - gIdxStart;
                } else {
                    headSize = offsetCalculator.GetDimG() - gIdxStart;
                }
                uint64_t offset = offsetCalculator.GetOffset(postQuantInfo.n2Idx, gIdxStart, 0);
                CopySingleMatrixNDToND<PARAM_T>(dstUb, srcTensor.gmTensor[offset],
                    headSize, offsetCalculator.GetDimD(), offsetCalculator.GetStrideG(), postQuantInfo.colCount);

                // 处理第二段S1
                if ((s1IdxEnd - s1IdxStart == 1) && (gIdxEnd > 0)) {
                    offset = offsetCalculator.GetOffset(postQuantInfo.n2Idx, 0, 0);
                    uint32_t ubOffset = headSize * postQuantInfo.colCount;

                    CopySingleMatrixNDToND<PARAM_T>(dstUb[ubOffset], srcTensor.gmTensor[offset],
                        gIdxEnd, offsetCalculator.GetDimD(), offsetCalculator.GetStrideG(), postQuantInfo.colCount);
                }
            }
        } else {
            uint32_t gIdxStart = postQuantInfo.gS1Idx / postQuantInfo.s1Size;
            uint32_t s1IdxStart = postQuantInfo.gS1Idx % postQuantInfo.s1Size;

            uint64_t offset = offsetCalculator.GetOffset(postQuantInfo.n2Idx, gIdxStart, 0);
            // postQuantInfo.gS1DealSize + s1IdxStart是将第一个G的S1部分补齐后的总GS1行数
            CopySingleMatrixNDToND<PARAM_T>(dstUb, srcTensor.gmTensor[offset],
                ((postQuantInfo.gS1DealSize + s1IdxStart) + (postQuantInfo.s1Size - 1)) / postQuantInfo.s1Size,
                offsetCalculator.GetDimD(), offsetCalculator.GetStrideG(), postQuantInfo.colCount);
        }
    }

    // src0Ub为需要量化的数据, src1为存放scale/offset参数的UB
    __aicore__ inline void AddOffset(LocalTensor<float> &dstUb, LocalTensor<float> &src0Ub, LocalTensor<float> &src1Ub, PostQuantInfo_V2 &postQuantInfo)
    {
        if constexpr (UB_FORMAT == UbFormat::S1G) {
            uint32_t s1IdxStart = postQuantInfo.gS1Idx / postQuantInfo.gSize;
            uint32_t gIdxStart = postQuantInfo.gS1Idx % postQuantInfo.gSize;
            uint32_t s1IdxEnd = (postQuantInfo.gS1Idx + postQuantInfo.gS1DealSize) / postQuantInfo.gSize;
            uint32_t gIdxEnd = (postQuantInfo.gS1Idx + postQuantInfo.gS1DealSize) % postQuantInfo.gSize;

            if (s1IdxEnd - s1IdxStart > 1) {
                // offsetUb为完整G
                uint32_t headSize = postQuantInfo.gSize - gIdxStart;

                uint32_t src0Offset = 0;
                uint32_t src1Offset = gIdxStart * postQuantInfo.colCount;
                uint32_t dealSize = headSize * postQuantInfo.colCount;
                Add(dstUb[src0Offset], src0Ub[src0Offset], src1Ub[src1Offset], dealSize);

                for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
                    src0Offset += dealSize;
                    dealSize = postQuantInfo.gSize * postQuantInfo.colCount;
                    Add(dstUb[src0Offset], src0Ub[src0Offset], src1Ub, dealSize);
                }

                if (gIdxEnd > 0) {
                    src0Offset += dealSize;
                    dealSize = gIdxEnd * postQuantInfo.colCount;
                    Add(dstUb[src0Offset], src0Ub[src0Offset], src1Ub, dealSize);
                }
            } else {
                uint32_t dealSize = postQuantInfo.gS1DealSize * postQuantInfo.colCount;
                Add(dstUb, src0Ub, src1Ub, dealSize);
            }
        } else {
            uint32_t gIdxStart = postQuantInfo.gS1Idx / postQuantInfo.s1Size;
            uint32_t s1IdxStart = postQuantInfo.gS1Idx % postQuantInfo.s1Size;
            uint32_t gIdxEnd = (postQuantInfo.gS1Idx + postQuantInfo.gS1DealSize) / postQuantInfo.s1Size;
            uint32_t s1IdxEnd = (postQuantInfo.gS1Idx + postQuantInfo.gS1DealSize) % postQuantInfo.s1Size;

            // 处理第一个S
            uint32_t headS1 = 0;
            if (gIdxStart == gIdxEnd) {
                headS1 = s1IdxEnd - s1IdxStart;
            } else {
                headS1 = postQuantInfo.s1Size - s1IdxStart;
            }
            uint32_t src0Offset = 0;
            uint32_t src1Offset = 0; // src1中存放[gIdxStart, gIdxEnd)的数据
            fa_base_vector::VecAddMatForBigRowCount(dstUb[src0Offset], src1Ub[src1Offset], src0Ub[src0Offset],
                headS1, postQuantInfo.colCount, postQuantInfo.dSize);

            if (gIdxEnd - gIdxStart >= 1) {
                // 处理中间块
                src0Offset += headS1 * postQuantInfo.colCount;
                src1Offset += postQuantInfo.colCount;
                for (uint32_t i = gIdxStart + 1; i < gIdxEnd; i++) {
                    fa_base_vector::VecAddMatForBigRowCount(dstUb[src0Offset], src1Ub[src1Offset], src0Ub[src0Offset],
                        postQuantInfo.s1Size, postQuantInfo.colCount, postQuantInfo.dSize);

                    src0Offset += postQuantInfo.s1Size * postQuantInfo.colCount;
                    src1Offset += postQuantInfo.colCount;
                }

                // 处理尾块
                if (s1IdxEnd > 0) {
                    fa_base_vector::VecAddMatForBigRowCount(dstUb[src0Offset], src1Ub[src1Offset], src0Ub[src0Offset],
                        s1IdxEnd, postQuantInfo.colCount, postQuantInfo.dSize);
                }
            }
        }
    }

    // src0Ub为需要量化的数据, src1为存放scale/offset参数的UB
    __aicore__ inline void MulScale(LocalTensor<float> &dstUb, LocalTensor<float> &src0Ub, LocalTensor<float> &src1Ub, PostQuantInfo_V2 &postQuantInfo)
    {
        if constexpr (UB_FORMAT == UbFormat::S1G) {
            uint32_t s1IdxStart = postQuantInfo.gS1Idx / postQuantInfo.gSize;
            uint32_t gIdxStart = postQuantInfo.gS1Idx % postQuantInfo.gSize;
            uint32_t s1IdxEnd = (postQuantInfo.gS1Idx + postQuantInfo.gS1DealSize) / postQuantInfo.gSize;
            uint32_t gIdxEnd = (postQuantInfo.gS1Idx + postQuantInfo.gS1DealSize) % postQuantInfo.gSize;

            if (s1IdxEnd - s1IdxStart > 1) {
                // offsetUb为完整G
                uint32_t headSize = postQuantInfo.gSize - gIdxStart;

                uint32_t src0Offset = 0;
                uint32_t src1Offset = gIdxStart * postQuantInfo.colCount;
                uint32_t dealSize = headSize * postQuantInfo.colCount;
                Mul(dstUb[src0Offset], src0Ub[src0Offset], src1Ub[src1Offset], dealSize);

                for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
                    src0Offset += dealSize;
                    dealSize = postQuantInfo.gSize * postQuantInfo.colCount;
                    Mul(dstUb[src0Offset], src0Ub[src0Offset], src1Ub, dealSize);
                }

                if (gIdxEnd > 0) {
                    src0Offset += dealSize;
                    dealSize = gIdxEnd * postQuantInfo.colCount;
                    Mul(dstUb[src0Offset], src0Ub[src0Offset], src1Ub, dealSize);
                }
            } else {
                uint32_t dealSize = postQuantInfo.gS1DealSize * postQuantInfo.colCount;
                Mul(dstUb, src0Ub, src1Ub, dealSize);
            }
        } else {
            uint32_t gIdxStart = postQuantInfo.gS1Idx / postQuantInfo.s1Size;
            uint32_t s1IdxStart = postQuantInfo.gS1Idx % postQuantInfo.s1Size;
            uint32_t gIdxEnd = (postQuantInfo.gS1Idx + postQuantInfo.gS1DealSize) / postQuantInfo.s1Size;
            uint32_t s1IdxEnd = (postQuantInfo.gS1Idx + postQuantInfo.gS1DealSize) % postQuantInfo.s1Size;

            // 处理第一个S
            uint32_t headS1 = 0;
            if (gIdxStart == gIdxEnd) {
                headS1 = s1IdxEnd - s1IdxStart;
            } else {
                headS1 = postQuantInfo.s1Size - s1IdxStart;
            }
            uint32_t src0Offset = 0;
            uint32_t src1Offset = 0; // src1中存放[gIdxStart, gIdxEnd)的数据
            fa_base_vector::VecMulMatForBigRowCount(dstUb[src0Offset], src1Ub[src1Offset], src0Ub[src0Offset],
                headS1, postQuantInfo.colCount, postQuantInfo.dSize);

            if (gIdxEnd - gIdxStart >= 1) {
                // 处理中间块
                src0Offset += headS1 * postQuantInfo.colCount;
                src1Offset += postQuantInfo.colCount;
                for (uint32_t i = gIdxStart + 1; i < gIdxEnd; i++) {
                    fa_base_vector::VecMulMatForBigRowCount(dstUb[src0Offset], src1Ub[src1Offset], src0Ub[src0Offset],
                        postQuantInfo.s1Size, postQuantInfo.colCount, postQuantInfo.dSize);

                    src0Offset += postQuantInfo.s1Size * postQuantInfo.colCount;
                    src1Offset += postQuantInfo.colCount;
                }

                // 处理尾块
                if (s1IdxEnd > 0) {
                    fa_base_vector::VecMulMatForBigRowCount(dstUb[src0Offset], src1Ub[src1Offset], src0Ub[src0Offset],
                        s1IdxEnd, postQuantInfo.colCount, postQuantInfo.dSize);
                }
            }
        }
    }
};

#endif