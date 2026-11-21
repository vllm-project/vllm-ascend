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
 * \file gmmsq_weight_quant_resplit_controller.h
 * \brief GMMSQ MxA8W4 Controller for multi-core scheduling and group traversal
 */
#ifndef GMMSQ_WEIGHT_QUANT_RESPLIT_CONTROLLER_H
#define GMMSQ_WEIGHT_QUANT_RESPLIT_CONTROLLER_H

#include "kernel_cube_intf.h"
#include "kernel_operator_list_tensor_intf.h"
#include "gmmsq_math_common.h"
#include "basic_block_config.h"
#include "basic_block_vf_mx.h"
#include "weight_quant_tool.h"
#include "gmmsq_weight_quant_vcv_basic_block.h"
#include "grouped_matmul_swiglu_quant_v2_weight_quant_tiling_data.h"

using AscendC::BLOCK_CUBE;
using AscendC::GlobalTensor;
using AscendC::IsSameType;
using AscendC::ListTensorDesc;
using GMMSQArch35Tiling::GMMSQWeightQuantTilingData;
using namespace WeightQuantBatchMatmulV2::Arch35;

namespace GMMSQWeightQuant {
#define GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM \
    template <typename xType, typename wType, typename weightScaleType, typename xScaleType, typename yType, \
              typename yScaleType, const WqmmConfig& wqmmConfig, const VecAntiQuantConfig& vecConfig, \
              bool isSingleMultiSingle, typename actType>

// 类声明处带默认实参（默认 SwiGLU）；成员类外定义只能使用不含默认值的版本
#define GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM_DEFAULT \
    template <typename xType, typename wType, typename weightScaleType, typename xScaleType, typename yType, \
              typename yScaleType, const WqmmConfig& wqmmConfig, const VecAntiQuantConfig& vecConfig, \
              bool isSingleMultiSingle, typename actType = GmmVecActivationSwiGLU>

#define GMMSQ_WQ_RESPLIT_CONTROLLER_CLASS \
    GMMSQWeightQuantResplitController<xType, wType, weightScaleType, xScaleType, yType, yScaleType, wqmmConfig, \
                                      vecConfig, isSingleMultiSingle, actType>

GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM_DEFAULT
class GMMSQWeightQuantResplitController {
public:
    __aicore__ inline GMMSQWeightQuantResplitController(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR xScale,
                                GM_ADDR y, GM_ADDR yScale, const GMMSQWeightQuantTilingData* __restrict baseTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitOffsetParam(BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM]);
    __aicore__ inline void SplitNByMultiCore(BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM],
                                             BasicBlockControlParam& ctrlParam, uint64_t basicBlockCount,
                                             uint64_t basicBlockSize);
    __aicore__ inline uint64_t GetSplitValueFromGroupList(uint64_t groupIdx);
    __aicore__ inline void UpdateGmAddr(uint64_t groupIdx, uint64_t mSize, uint64_t kSize, uint64_t nSize);
    __aicore__ inline uint64_t GetSwitchedProcessId(const BasicBlockControlParam& ctrlParam);

    template <typename T>
    __aicore__ inline __gm__ T* GetTensorAddr(uint64_t groupIdx, GM_ADDR tensorPtr)
    {
        ListTensorDesc listTensorDesc(reinterpret_cast<__gm__ void*>(tensorPtr));
        return listTensorDesc.GetDataPtr<T>(groupIdx);
    }

    const GMMSQWeightQuantTilingData* tiling_;

    __gm__ xType* xGm_;
    __gm__ wType* weightGm_;
    __gm__ weightScaleType* weightScaleGm_;
    __gm__ yType* yGm_;
    __gm__ xScaleType* xScaleGm_;
    __gm__ yScaleType* yScaleGm_;
    // 输出基址（清零尾部未覆盖行时使用，yGm_/yScaleGm_ 会随 group 遍历推进）
    __gm__ yType* yBase_;
    __gm__ yScaleType* yScaleBase_;
    GM_ADDR weightTensorPtr_;
    GM_ADDR weightScaleTensorPtr_;
    GlobalTensor<int64_t> groupListGm_;
    GMMSQWeightQuantVcvBasicBlock<xType, wType, weightScaleType, xScaleType, yType, yScaleType, wqmmConfig, vecConfig,
                                  actType>
        basicBlock_;

    uint64_t preOffset_ = 0;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_512 = 512;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256 = 256;
    static constexpr uint64_t M_L1 = 256;
    static constexpr uint64_t KB_L1_SIZE = 256;
    static constexpr uint64_t N_L1 = 256;
    static constexpr uint64_t L1K_DYNAMIC_CONFIG_M_THRESHOLD = MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256;
    static constexpr uint64_t A_L1_BUFFER_KB_UNITS = 128;
};

GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_RESPLIT_CONTROLLER_CLASS::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
                                                               GM_ADDR groupList, GM_ADDR xScale, GM_ADDR y,
                                                               GM_ADDR yScale,
                                                               const GMMSQWeightQuantTilingData* __restrict baseTiling)
{
    tiling_ = baseTiling;

    xGm_ = reinterpret_cast<__gm__ xType*>(x);
    weightTensorPtr_ = weight;
    weightScaleTensorPtr_ = weightScale;
    weightGm_ = GetTensorAddr<wType>(0, weightTensorPtr_);
    weightScaleGm_ = GetTensorAddr<weightScaleType>(0, weightScaleTensorPtr_);
    xScaleGm_ = reinterpret_cast<__gm__ xScaleType*>(xScale);
    yGm_ = reinterpret_cast<__gm__ yType*>(y);
    yScaleGm_ = reinterpret_cast<__gm__ yScaleType*>(yScale);
    yBase_ = yGm_;
    yScaleBase_ = yScaleGm_;
    groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupList));
    basicBlock_.Init();
    basicBlock_.SetActParams(tiling_->beta, tiling_->invBeta, tiling_->linearBeta, tiling_->invLinearBeta);
}

GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_RESPLIT_CONTROLLER_CLASS::Process()
{
    uint32_t cubeBlockIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        cubeBlockIdx = cubeBlockIdx >> 1;
    }

    BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM];
    InitOffsetParam(offsetParam);

    // cacheline 128 B对齐，b4为 256 个元素
    bool isCacheLineUnaligned = offsetParam[0].kSize % 128 != 0;
    if constexpr (IsSameType<wType, int4b_t>::value || IsSameType<wType, fp4x2_e2m1_t>::value ||
                  IsSameType<wType, fp4x2_e1m2_t>::value) {
        isCacheLineUnaligned = offsetParam[0].kSize % 256 != 0;
    }

    BasicBlockControlParam ctrlParam;
    ctrlParam.processId = 0;
    uint64_t coveredRows = 0;
    for (uint32_t groupIdx = 0, startBasicBlockId = 0; groupIdx < tiling_->groupNum; ++groupIdx) {
        // 根据groupListType切分M轴
        ctrlParam.mSize = GetSplitValueFromGroupList(groupIdx);
        // group_list 在 device 上，host 侧 tiling 校验不到内容。若调用方传入非单调 cumsum
        // （offset - preOffset_ 在 uint64 下溢出成天文数字）或 counts 求和 > M，mSize 会超过剩余
        // 行数，进而越界写 y。这里按剩余额度夹紧：非法输入退化为少算，而不是越界写。
        // 合法输入下 coveredRows + mSize <= tiling_->mSize 恒成立，此夹紧不生效。
        // tiling_->mSize == 0 表示 host 未提供总 M（SwiGLU 通路不填该字段），此时整体跳过覆盖保护，
        // 否则会把 mSize 夹成 0。共版内核（swiglu_quant / situ_quant）靠这一约定区分。
        if (tiling_->mSize != 0) {
            const uint64_t remainingRows = coveredRows < tiling_->mSize ? tiling_->mSize - coveredRows : 0;
            if (ctrlParam.mSize > remainingRows) {
                ctrlParam.mSize = remainingRows;
            }
        }
        coveredRows += ctrlParam.mSize;
        if (ctrlParam.mSize > 0 && offsetParam[ctrlParam.processId].nSize > 0) {
            uint64_t mBlkNum = CeilDivide(ctrlParam.mSize, M_L1);
            ctrlParam.mL1Size = CeilDivide(ctrlParam.mSize, mBlkNum);
            basicBlock_.UpdateGlobalAddr(xGm_, weightGm_, weightScaleGm_, xScaleGm_,
                                         ctrlParam.mL1Size < ctrlParam.mSize || isCacheLineUnaligned);
            ctrlParam.curBasicBlockId =
                cubeBlockIdx >= startBasicBlockId ? cubeBlockIdx : cubeBlockIdx + tiling_->coreNum;
            ctrlParam.basicBlockLimit = startBasicBlockId;
            for (ctrlParam.mOffset = 0; ctrlParam.mOffset < ctrlParam.mSize; ctrlParam.mOffset += ctrlParam.mL1Size) {
                ctrlParam.nOffset = 0;
                // SwiGLU splits N axis into swish/gate halves, so each core processes N_L1/2 per half
                SplitNByMultiCore(offsetParam, ctrlParam, CeilDivide(offsetParam[0].nSize / 2, N_L1 / 2), N_L1 / 2);
                ctrlParam.basicBlockLimit += CeilDivide(offsetParam[0].nSize / 2, N_L1 / 2);
            }
            startBasicBlockId = ctrlParam.basicBlockLimit % tiling_->coreNum;
        }
        UpdateGmAddr(groupIdx, ctrlParam.mSize, offsetParam[0].kSize, offsetParam[0].nSize);
    }

    basicBlock_.End(offsetParam[GetSwitchedProcessId(ctrlParam)]);

    // group_list 未覆盖的尾部行（丢 token 场景）不属于任何专家，显式写 0，
    // 避免下游读到未初始化数据。由 AIV0 单独完成，避免多核重复搬运。
    // tiling_->mSize == 0（SwiGLU 通路的「未提供」哨兵）时该条件恒不成立，天然跳过。
    if (coveredRows < tiling_->mSize && AscendC::GetBlockIdx() == 0) {
        uint64_t nOut = tiling_->nSize / 2;
        uint64_t nScalePerRow = CeilDivide(CeilAlign(nOut, static_cast<uint64_t>(64)),
                                           static_cast<uint64_t>(tiling_->groupSize));
        basicBlock_.ZeroTailOutput(yBase_, yScaleBase_, coveredRows, tiling_->mSize - coveredRows, nOut, nScalePerRow);
    }
}

GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_RESPLIT_CONTROLLER_CLASS::InitOffsetParam(
    BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM])
{
    offsetParam[0].kbL1Size = KB_L1_SIZE;
    offsetParam[0].kaL1Size = offsetParam[0].kbL1Size;
    offsetParam[0].kSize = tiling_->kSize;
    offsetParam[0].nSize = tiling_->nSize;
    offsetParam[0].kAlign = CeilAlign(tiling_->kSize, static_cast<uint64_t>(BLOCK_CUBE));
    offsetParam[0].nAlign = CeilAlign(tiling_->nSize, static_cast<uint64_t>(BLOCK_CUBE));

    offsetParam[1].kbL1Size = KB_L1_SIZE;
    offsetParam[1].kaL1Size = offsetParam[0].kbL1Size;
    offsetParam[1].kSize = offsetParam[0].kSize;
    offsetParam[1].nSize = offsetParam[0].nSize;
    offsetParam[1].kAlign = offsetParam[0].kAlign;
    offsetParam[1].nAlign = offsetParam[0].nAlign;
}

GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_RESPLIT_CONTROLLER_CLASS::SplitNByMultiCore(
    BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM], BasicBlockControlParam& ctrlParam,
    uint64_t basicBlockCount, uint64_t basicBlockSize)
{
    for (; ctrlParam.curBasicBlockId < ctrlParam.basicBlockLimit + basicBlockCount;
         ctrlParam.curBasicBlockId += tiling_->coreNum) {
        offsetParam[ctrlParam.processId].mSize = ctrlParam.mSize;
        offsetParam[ctrlParam.processId].mOffset = ctrlParam.mOffset;
        offsetParam[ctrlParam.processId].mL1Size = ctrlParam.mOffset + ctrlParam.mL1Size > ctrlParam.mSize ?
                                                       ctrlParam.mSize - ctrlParam.mOffset :
                                                       ctrlParam.mL1Size;
        offsetParam[ctrlParam.processId].nOffset =
            ctrlParam.nOffset +
            ((ctrlParam.curBasicBlockId - ctrlParam.basicBlockLimit) % basicBlockCount) * basicBlockSize;

        // GMM输出前N/2为激活值，后N/2为gate值，因此weight和weightScale分两次、各载入nL1Size/2
        // halfNRemaining: nOffset为前N/2的N方向偏移 尾块大小基于前N/2计算得到
        uint64_t halfNRemaining = offsetParam[ctrlParam.processId].nSize / 2 - offsetParam[ctrlParam.processId].nOffset;
        // 根据一半的尾块大小计算出nL1完整大小
        offsetParam[ctrlParam.processId].nL1Size =
            halfNRemaining >= basicBlockSize ? basicBlockSize * 2 : halfNRemaining * 2;
        offsetParam[ctrlParam.processId].yGmAddr = reinterpret_cast<GM_ADDR>(yGm_);
        offsetParam[ctrlParam.processId].yScaleGmAddr = reinterpret_cast<GM_ADDR>(yScaleGm_);

        offsetParam[ctrlParam.processId].kbL1Size =
            (offsetParam[ctrlParam.processId].mL1Size <= L1K_DYNAMIC_CONFIG_M_THRESHOLD &&
             offsetParam[ctrlParam.processId].nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
                MX_A8W4_L1_K_CONFIG_512 :
                MX_A8W4_L1_K_CONFIG_256;

        uint64_t mL1Align = CeilAlign(offsetParam[ctrlParam.processId].mL1Size, BLOCK_CUBE);
        uint64_t kaDepth = CeilDivide(offsetParam[ctrlParam.processId].nL1Size, mL1Align * 2);
        uint64_t aL1Size = A_L1_BUFFER_KB_UNITS * GetKBUnit<xType>();
        uint64_t maxKaDepth = aL1Size / (mL1Align * offsetParam[ctrlParam.processId].kbL1Size);
        if (kaDepth > maxKaDepth) {
            kaDepth = maxKaDepth;
        }
        offsetParam[ctrlParam.processId].kaL1Size = kaDepth * offsetParam[ctrlParam.processId].kbL1Size;

        basicBlock_.ComputeBasicBlock(offsetParam[ctrlParam.processId], offsetParam[GetSwitchedProcessId(ctrlParam)]);
        ctrlParam.processId = GetSwitchedProcessId(ctrlParam);
    }
    ctrlParam.nOffset += basicBlockSize * basicBlockCount;
}

GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_RESPLIT_CONTROLLER_CLASS::UpdateGmAddr(uint64_t groupIdx, uint64_t mSize,
                                                                       uint64_t kSize, uint64_t nSize)
{
    xGm_ += mSize * kSize;
    uint64_t kAlignSize = CeilAlign(kSize, K_ALIGNMENT64);
    if constexpr (isSingleMultiSingle) {
        uint64_t nextGroupIdx = groupIdx + 1;
        if (nextGroupIdx < tiling_->groupNum) {
            weightGm_ = GetTensorAddr<wType>(nextGroupIdx, weightTensorPtr_);
            weightScaleGm_ = GetTensorAddr<weightScaleType>(nextGroupIdx, weightScaleTensorPtr_);
        }
    } else {
        // weight 为fp4, 大小除2
        weightGm_ += (nSize * kSize) >> 1;
        weightScaleGm_ += nSize * CeilDivide(kAlignSize, static_cast<uint64_t>(tiling_->groupSize));
    }
    xScaleGm_ += mSize * CeilDivide(kAlignSize, static_cast<uint64_t>(tiling_->groupSize));
    // 输出为 N / 2
    yGm_ += mSize * (nSize / 2);
    yScaleGm_ +=
        mSize * CeilDivide(CeilAlign(nSize / 2, static_cast<uint64_t>(64)), static_cast<uint64_t>(tiling_->groupSize));
}

GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline uint64_t GMMSQ_WQ_RESPLIT_CONTROLLER_CLASS::GetSplitValueFromGroupList(uint64_t groupIdx)
{
    uint64_t splitValue = 0;
    if (tiling_->groupListType == 0) {
        uint64_t offset = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else {
        splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
    }
    return splitValue;
}

GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline uint64_t GMMSQ_WQ_RESPLIT_CONTROLLER_CLASS::GetSwitchedProcessId(
    const BasicBlockControlParam& ctrlParam)
{
    return 1 - ctrlParam.processId;
}

} // namespace GMMSQWeightQuant

#endif // GMMSQ_WEIGHT_QUANT_RESPLIT_CONTROLLER_H
