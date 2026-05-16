/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_INIT_OUTPUTS_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_INIT_OUTPUTS_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/matrix_coord.hpp"

namespace NpuArch::Epilogue::Block {

template <
    class AttnOutType_,
    class LseOutType_,
    LseMode LSE_MODE_>
class BlockEpilogue<
    EpilogueAtlasA2InitOutWhenZero<LSE_MODE_>,
    AttnOutType_,
    LseOutType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2InitOutWhenZero<LSE_MODE_>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementAttnOut = typename AttnOutType_::Element;
    using ElementLseOut = typename LseOutType_::Element;

    using LayoutAttnOut = typename AttnOutType_::Layout;
    using LayoutLseOut = typename LseOutType_::Layout;

    static constexpr LseMode LSE_MODE = DispatchPolicy::LSE_MODE;
    static constexpr float ATTN_OUT_INI = 0;
    static constexpr float LSE_OUT_INI = 3e+99;
    static constexpr uint32_t HALF_ELEM_NUM_PER_BLK = 16;
    static constexpr uint32_t FLOAT_ELEM_NUM_PER_BLK = 8;
    static constexpr uint32_t HALF_ELEM_NUM_PER_RPT = 128;
    static constexpr uint32_t FLOAT_ELEM_NUM_PER_RPT = 64;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        // Allocate UB space
        constexpr uint32_t ATTN_OUT_INIT_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LSE_OUT_INIT_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;

        attnOutUbTensor = resource.ubBuf.template GetBufferByByte<ElementAttnOut>(ATTN_OUT_INIT_UB_TENSOR_OFFSET);
        lseOutUbTensor = resource.ubBuf.template GetBufferByByte<ElementLseOut>(LSE_OUT_INIT_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementAttnOut> gOutput,
        AscendC::GlobalTensor<ElementLseOut> gLse,
        const LayoutAttnOut &layoutOutput,
        const LayoutLseOut &layoutLse,
        uint32_t qSThisSubBlock, uint32_t qNThisSubBlock)
    {
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t qHeads = layoutLse.shape(1);
        uint32_t embedV = oHiddenSize / qHeads;
        uint32_t embedRoundV = NpuArch::Detail::Alignment::RoundUp(embedV, HALF_ELEM_NUM_PER_BLK);
        AscendC::PipeBarrier<PIPE_ALL>();
        // init attnOut with 0
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::Duplicate(attnOutUbTensor, static_cast<ElementAttnOut>(ATTN_OUT_INI), embedRoundV * qSThisSubBlock);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
        for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
            AscendC::DataCopyPad(
                gOutput[qNIdx * embedV],
                attnOutUbTensor,
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, embedV * sizeof(ElementAttnOut),
                    0, (oHiddenSize - embedV) * sizeof(ElementAttnOut), 0));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        if constexpr (LSE_MODE_ == LseMode::OUT_ONLY) {
            // init lseOut with inf
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::Duplicate(lseOutUbTensor, LSE_OUT_INI, qSThisSubBlock * FLOAT_ELEM_NUM_PER_BLK);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
            for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                AscendC::DataCopyPad(
                    gLse[qNIdx],
                    lseOutUbTensor,
                    AscendC::DataCopyExtParams(
                        qSThisSubBlock, sizeof(ElementLseOut),
                        0, (qHeads - 1) * sizeof(ElementLseOut), 0));
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementAttnOut> gOutput,
        AscendC::GlobalTensor<ElementLseOut> gLse,
        const LayoutAttnOut &layoutOutput,
        const LayoutLseOut &layoutLse,
        uint32_t qSBlockSize, uint32_t qNBlockSize)
    {
        uint32_t rowNum = qSBlockSize * qNBlockSize;
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t qHeads = layoutLse.shape(1);
        uint32_t embedV = oHiddenSize / qHeads;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ? 1
            : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock =
            (qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualSubBlock = (subBlockIdx == 1U) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetSubBlock = subBlockIdx * rowSplitSubBlock;
        uint32_t outRowOffsetSubBlock = (qNBlockSize == 1U) ? rowOffsetSubBlock : 0;
        uint32_t outColOffsetSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock * embedV;
        uint32_t qSThisSubBlock = (qNBlockSize == 1U) ? rowActualSubBlock : qSBlockSize;
        int64_t outOffsetSubBlock =
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetSubBlock, outColOffsetSubBlock));
        auto gOutputSubBlock = gOutput[outOffsetSubBlock];
        auto layoutOutputSubBlock = layoutOutput;

        uint32_t outLseRowOffsetSubBlock = (qNBlockSize == 1U) ?
            rowOffsetSubBlock : 0;
        uint32_t outLseColOffsetSubBlock = (qNBlockSize == 1U) ?
            0 : subBlockIdx * qNSplitSubBlock;
        int64_t lseOffsetSubBlock =
            layoutLse.GetOffset(MatrixCoord(outLseRowOffsetSubBlock, outLseColOffsetSubBlock));
        auto gLseThisSubBlock = gLse[lseOffsetSubBlock];
        auto layoutLseThisSubBlock = layoutLse;

        if (rowActualSubBlock > 0U) {
            SubCoreCompute(
                gOutputSubBlock, gLseThisSubBlock,
                layoutOutputSubBlock, layoutLseThisSubBlock,
                qSThisSubBlock, qNThisSubBlock);
        }
    }
private:
    AscendC::LocalTensor<ElementAttnOut> attnOutUbTensor;
    AscendC::LocalTensor<ElementLseOut> lseOutUbTensor;
};
}
#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_INIT_OUTPUTS_HPP