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
 * \file FixpipeOut.h
 * \brief
 */
#ifndef FIXPIPEOUT_H
#define FIXPIPEOUT_H

constexpr FixpipeConfig PFA_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true}; // ROW_MAJOR: 使能NZ2ND，输出数据格式为ND格式; true: 用于用户指定目的地址的位置是否是UB
constexpr FixpipeConfig PFA_CFG_ROW_MAJOR_GM = {CO2Layout::ROW_MAJOR, false}; // ROW_MAJOR: 使能NZ2ND，输出数据格式为ND格式; true: 用于用户指定目的地址的位置是否是UB

struct fixpipeOutParams {
    uint32_t fixpOutMSize;
    uint32_t fixpOutNSize;
};

template<typename mmOutputType, typename computeType, typename l0cType>
__aicore__ inline void FixpipeMmCopyOutToUB(LocalTensor<mmOutputType>& mmResUb, LocalTensor<l0cType>& L0CTensor, const fixpipeOutParams& fixpOutParam)
{
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> L0C2UbFixpParams; // L0C->UB
    L0C2UbFixpParams.nSize = (fixpOutParam.fixpOutNSize + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐
    L0C2UbFixpParams.mSize = (fixpOutParam.fixpOutMSize + 1) >> 1 << 1; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    L0C2UbFixpParams.srcStride = ((L0C2UbFixpParams.mSize + 15) >> 4) << 4; // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T) //源NZ矩阵中相邻Z排布的起始地址偏移
    L0C2UbFixpParams.dstStride = (L0C2UbFixpParams.nSize + 15) >> 4 << 4; // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除
    L0C2UbFixpParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    L0C2UbFixpParams.params.ndNum = 1;
    L0C2UbFixpParams.params.srcNdStride = 0;
    L0C2UbFixpParams.params.dstNdStride = 0;
    Fixpipe<mmOutputType, computeType, PFA_CFG_ROW_MAJOR_UB>(mmResUb, L0CTensor, L0C2UbFixpParams); // 将matmul结果从L0C搬运到UB
}

template<typename mmOutputType, typename computeType, typename l0cType>
__aicore__ inline void FixpipeMmCopyOutToGm(GlobalTensor<mmOutputType>& mmResGm,LocalTensor<l0cType>& L0CTensor, const fixpipeOutParams& fixpOutParam)
{
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> L0C2GmFixpParams; // L0C->Gm
    L0C2GmFixpParams.nSize = (fixpOutParam.fixpOutNSize + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐；分档计算且vector1中通过mask筛选出实际有效值
    L0C2GmFixpParams.mSize = (fixpOutParam.fixpOutMSize + 1) >> 1 << 1; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小；同mmadParams.m
    L0C2GmFixpParams.srcStride = ((L0C2GmFixpParams.mSize + 15) >> 4) << 4; // L0C上bmm1结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔）
    L0C2GmFixpParams.dstStride = (L0C2GmFixpParams.nSize + 15) >> 4 << 4; // mmResGm上两行之间的间隔
    L0C2GmFixpParams.dualDstCtl = 1;
    L0C2GmFixpParams.params.ndNum = 1;
    L0C2GmFixpParams.params.srcNdStride = 0;
    L0C2GmFixpParams.params.dstNdStride = 0;
    Fixpipe<mmOutputType, computeType, PFA_CFG_ROW_MAJOR_GM>(mmResGm, L0CTensor, L0C2GmFixpParams); // 将matmul结果从L0C搬运到Gm
}
#endif