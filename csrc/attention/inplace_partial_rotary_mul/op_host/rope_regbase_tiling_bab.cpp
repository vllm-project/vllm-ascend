/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file rope_regbase_tiling_bab.cc
 * \brief
 */

#include "inplace_partial_rotary_mul_tiling.h"

namespace optiling {
constexpr uint64_t ROPE_BAB_TILING_PRIORITY = 20000;
constexpr uint32_t MIN_UB_LOAD_D_NUM = 4; // x, y或in, cos输入开doubleBuffer
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr int64_t MIN_COPY_BLOCK_COUNT = 4095;
constexpr size_t WORK_SPACE_SIZE = static_cast<size_t>(16) * 1024 * 1024;
constexpr int64_t TILING_KEY_BAB = 20020;

ge::graphStatus RopeRegBaseTilingClassBAB::DoOpTiling()
{
    ubSize_ = aicoreParams_.ubSize;
    coreNum_ = aicoreParams_.blockDim;
    ge::graphStatus status = SplitUb();
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context_->GetNodeName(), "SplitUb Failed.");
        return ge::GRAPH_FAILED;
    }
    status = SplitCore();
    if (status != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context_->GetNodeName(), "SplitCore Failed.");
        return ge::GRAPH_FAILED;
    }
    if (blockNumB_ * blockNumS_ > coreNum_) {
        OPS_LOG_E(context_->GetNodeName(), "split coreNum [%ld] large than coreNum[%ld]", blockNumB_ * blockNumS_,
                coreNum_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClassBAB::SplitCore()
{
    // 尝试先对B分核，再尝试优先对S分核，比较二者切分之后的总核数
    auto blockFactorB1 = CeilDiv(b_, coreNum_);
    auto blockNumB1 = CeilDiv(b_, blockFactorB1);
    if (blockNumB1 == 0) {
        OPS_LOG_I("RopeRegBaseTilingClassBAB SplitCore error, blockNumB1 == 0");
        return ge::GRAPH_FAILED;
    }
    auto blockNumS1 = std::min(coreNum_ / blockNumB1, s_);
    auto blockFactorS1 = CeilDiv(s_, blockNumS1);
    blockNumS1 = CeilDiv(s_, blockFactorS1);
    auto usedCoreNum1 = blockNumB1 * blockNumS1;

    auto blockFactorS2 = CeilDiv(s_, coreNum_);
    auto blockNumS2 = CeilDiv(s_, blockFactorS2);
    if (blockNumS2 == 0) {
        OPS_LOG_I("RopeRegBaseTilingClassBAB SplitCore error, blockNumS2 == 0");
        return ge::GRAPH_FAILED;
    }
    auto blockNumB2 = std::min(coreNum_ / blockNumS2, b_);
    auto blockFactorB2 = CeilDiv(b_, blockNumB2);
    blockNumB2 = CeilDiv(b_, blockFactorB2);
    auto usedCoreNum2 = blockNumB2 * blockNumS2;

    // ubFactorS 很小的时候，选择核数多的，ubFactorS大于blockFactorS时， 综合考虑分核和UB切分
    auto ubFactorS1 = std::min(ubFactorS_, blockFactorS1);
    auto ubFactorS2 = std::min(ubFactorS_, blockFactorS2);
    if (usedCoreNum1 * ubFactorS1 >= usedCoreNum2 * ubFactorS2) {
        blockNumB_ = blockNumB1;
        blockFactorB_ = blockFactorB1;
        blockNumS_ = blockNumS1;
        blockFactorS_ = blockFactorS1;
        usedCoreNum_ = usedCoreNum1;
        ubFactorS_ = ubFactorS1;
    } else {
        blockNumB_ = blockNumB2;
        blockFactorB_ = blockFactorB2;
        blockNumS_ = blockNumS2;
        blockFactorS_ = blockFactorS2;
        usedCoreNum_ = usedCoreNum2;
        ubFactorS_ = ubFactorS2;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClassBAB::SplitUb()
{
    uint32_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t dAlign = CeilAlign(sliceLength_ * typeSize / dSplitCoef_, blockSize_) * dSplitCoef_;
    // interleave模式需要补pad, D对齐
    int64_t canLoadDNum = FloorDiv(ubSize_, dAlign);
    if (canLoadDNum < MIN_UB_LOAD_D_NUM + MIN_UB_LOAD_D_NUM) {
        OPS_LOG_E(context_->GetNodeName(), "ubSize_ can't load 10 d_, d_ = %ld.", d_);
        return ge::GRAPH_FAILED;
    }
    canLoadDNum = canLoadDNum / MIN_UB_LOAD_D_NUM;
    int64_t ubLoopNum = CeilDiv(n_, (canLoadDNum - 1)); // 这里减去sin, cos 开doubleBuffer
    ubFactorN_ = std::min(CeilDiv(n_, ubLoopNum), MIN_COPY_BLOCK_COUNT / dSplitCoef_);
    ubLoopNumN_ = CeilDiv(n_, ubFactorN_);
    if (ubFactorN_ == 0) {
        OPS_LOG_I("RopeRegBaseTilingClassBAB SplitUb error, ubFactorN_ == 0");
        return ge::GRAPH_FAILED;
    }
    ubTailFactorN_ = (n_ % ubFactorN_ == 0) ? ubFactorN_ : n_ % ubFactorN_;
    int64_t ubFactorS = FloorDiv(canLoadDNum, n_ + 1);
    ubFactorS_ = (ubFactorS == 0) ? 1 : ubFactorS;
    return ge::GRAPH_SUCCESS;
}

void RopeRegBaseTilingClassBAB::PrintTilingData()
{
    OPS_LOG_I(context_->GetNodeName(),
            "RopeRegBaseTilingClassBAB tilingData: useCoreNum is %ld,"
            "B is %ld, CosB is %ld, S is %ld, D is %ld, N is %ld, blockNumB %ld,"
            "blockFactorB_ is %ld, blockNumS %ld, blockFactorS is %ld, ubLoopNumS is %ld,"
            "ubFactorS is %ld, ubTailFactorS %ld, ubLoopNumB is %ld, ubFactorB is %ld,"
            "ubTailFactorB is %ld, ubLoopNumN is %ld, ubFactorN is %ld, ubTailFactorN is %ld,"
            "rotaryMode is %ld, tilingKey is %ld, sliceStart is %ld, sliceEnd is %ld, sliceLength is %ld",
            usedCoreNum_, tilingData_.get_B(), tilingData_.get_CosB(), tilingData_.get_S(), tilingData_.get_D(),
            tilingData_.get_N(), tilingData_.get_blockNumB(), tilingData_.get_blockFactorB(),
            tilingData_.get_blockNumS(), tilingData_.get_blockFactorS(), tilingData_.get_ubLoopNumS(),
            tilingData_.get_ubFactorS(), tilingData_.get_ubTailFactorS(), tilingData_.get_ubLoopNumB(),
            tilingData_.get_ubFactorB(), tilingData_.get_ubTailFactorB(), tilingData_.get_ubLoopNumN(),
            tilingData_.get_ubFactorN(), tilingData_.get_ubTailFactorN(), tilingData_.get_rotaryMode(), tilingKey_, tilingData_.get_sliceStart(), tilingData_.get_sliceEnd(), tilingData_.get_sliceLength());
    return;
}

ge::graphStatus RopeRegBaseTilingClassBAB::PostTiling()
{
    tilingData_.set_B(b_);
    tilingData_.set_CosB(0);
    tilingData_.set_S(s_);
    tilingData_.set_D(d_);
    tilingData_.set_N(n_);
    tilingData_.set_blockNumB(blockNumB_);
    tilingData_.set_blockFactorB(blockFactorB_);
    tilingData_.set_blockNumS(blockNumS_);
    tilingData_.set_blockFactorS(blockFactorS_);
    tilingData_.set_ubLoopNumS(ubLoopNumS_);
    tilingData_.set_ubFactorS(ubFactorS_);
    tilingData_.set_ubTailFactorS(ubTailFactorS_);
    tilingData_.set_ubLoopNumB(ubLoopNumB_);
    tilingData_.set_ubFactorB(ubFactorB_);
    tilingData_.set_ubTailFactorB(ubTailFactorB_);
    tilingData_.set_ubLoopNumN(ubLoopNumN_);
    tilingData_.set_ubFactorN(ubFactorN_);
    tilingData_.set_ubTailFactorN(ubTailFactorN_);
    tilingData_.set_rotaryMode(static_cast<int64_t>(rotaryMode_));
    tilingData_.set_sliceStart(static_cast<int64_t>(sliceStart_));
    tilingData_.set_sliceEnd(static_cast<int64_t>(sliceEnd_));
    tilingData_.set_sliceLength(static_cast<int64_t>(sliceLength_));
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(usedCoreNum_);
    context_->SetTilingKey(tilingKey_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t RopeRegBaseTilingClassBAB::GetTilingKey() const
{
    return TILING_KEY_BAB;
}

// REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMul, RopeRegBaseTilingClassBAB, ROPE_BAB_TILING_PRIORITY);
} // namespace optiling
