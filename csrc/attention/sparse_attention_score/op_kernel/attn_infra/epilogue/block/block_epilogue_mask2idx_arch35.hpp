/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_MASK2IDX_ARCH35
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_MASK2IDX_ARCH35

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/matrix_coord.hpp"

namespace NpuArch::Epilogue::Block {

template <
    class ElementSparseMask_,
    class ElementSparseIdx_,
    class ElementSparseCount_
>
class BlockEpilogue <
    EpilogueBsaMask2Idx,
    ElementSparseMask_,
    ElementSparseIdx_,
    ElementSparseCount_>
{
public:
    using DispatchPolicy = EpilogueBsaMask2Idx;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSparseMask = ElementSparseMask_;
    using ElementSparseIdx = ElementSparseIdx_;
    using ElementSparseCount = ElementSparseCount_;

    static constexpr uint32_t IO_STAGES = DispatchPolicy::IO_STAGES;
    static constexpr uint32_t PRE_ROW_TILE = 128;
    static constexpr uint32_t PRE_COL_TILE = 64;
    static constexpr uint32_t PRE_ELEM_NUM_PER_LOOP = PRE_ROW_TILE * PRE_COL_TILE;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        constexpr uint32_t MASK_PAT_IN_UINT8 = 0;
        constexpr uint32_t MASK_PAT_IN_FP16 = 2 * PRE_ELEM_NUM_PER_LOOP;
        constexpr uint32_t MASK_PAT_IN_FP32 = 4 * PRE_ELEM_NUM_PER_LOOP; // 2(db)+2
        constexpr uint32_t MASK_PAT_IN_BIT = 8 * PRE_ELEM_NUM_PER_LOOP; // 1+2+4
        constexpr uint32_t MASK_IDX = 9 * PRE_ELEM_NUM_PER_LOOP; // 1+2+4+1
        constexpr uint32_t RSVD_SPARSE_IDX = 13 * PRE_ELEM_NUM_PER_LOOP; // 1+2+4+1+4
        constexpr uint32_t RSVD_SPARSE_COUNT = 21 * PRE_ELEM_NUM_PER_LOOP; // 1+2+4+1+4+8(db)

        for (uint32_t i = 0; i < IO_STAGES; i++) {
            maskPatUb8[i] = resource.ubBuf.template GetBufferByByte<ElementSparseMask>(
                MASK_PAT_IN_UINT8 + i * PRE_ELEM_NUM_PER_LOOP * sizeof(ElementSparseMask));
            sparseIdxUb[i] = resource.ubBuf.template GetBufferByByte<ElementSparseIdx>(
                RSVD_SPARSE_IDX + i * PRE_ELEM_NUM_PER_LOOP * sizeof(ElementSparseIdx));
            sparseCountUb[i] = resource.ubBuf.template GetBufferByByte<ElementSparseCount>(
                RSVD_SPARSE_COUNT + i * PRE_ROW_TILE * sizeof(ElementSparseCount));
        }
        maskPatUb16 = resource.ubBuf.template GetBufferByByte<half>(MASK_PAT_IN_FP16);
        maskPatUb32 = resource.ubBuf.template GetBufferByByte<float>(MASK_PAT_IN_FP32);
        maskPatInBitUb8 = resource.ubBuf.template GetBufferByByte<uint8_t>(MASK_PAT_IN_BIT);
        maskPatInBitUb32 = resource.ubBuf.template GetBufferByByte<uint32_t>(MASK_PAT_IN_BIT);
        maskIdxUb = resource.ubBuf.template GetBufferByByte<ElementSparseIdx>(MASK_IDX);
    }

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementSparseMask> gSparseMask,
        AscendC::GlobalTensor<ElementSparseIdx> gSparseIdx,
        AscendC::GlobalTensor<ElementSparseCount> gSparseCount,
        uint32_t totalRowNumBlockMask,
        uint32_t yBlockNumAligned,
        uint32_t avgRowPerSubCore,
        uint32_t preActiveSubCoreNum)
    {
        uint32_t subCoreIdx = AscendC::GetBlockIdx();
        uint32_t curSubCoreRowOffset = subCoreIdx * avgRowPerSubCore;
        uint32_t actDealtRow = (subCoreIdx == preActiveSubCoreNum - 1) ?
            (totalRowNumBlockMask - curSubCoreRowOffset) : avgRowPerSubCore;
        
        if (subCoreIdx < preActiveSubCoreNum) {
            uint32_t rowLoop = CeilDiv(actDealtRow, PRE_ROW_TILE);
            uint32_t colLoop = CeilDiv(yBlockNumAligned, PRE_COL_TILE);
            uint32_t IdxPingPongFlag = 0;
            uint32_t CountPingPongFlag = 0;
            for (uint32_t i = 0; i < rowLoop; i++) {
                uint32_t curLoopRowOffset = i * PRE_ROW_TILE;
                uint32_t globalRowOffset = curSubCoreRowOffset + curLoopRowOffset;
                uint32_t actDealtRowCurLoop = (i == rowLoop - 1) ? (actDealtRow - curLoopRowOffset) : PRE_ROW_TILE;

                uint64_t rsvdCountPerRow[PRE_ROW_TILE] = {0};
                uint64_t rsvdCountPerRowCurColLoop[PRE_ROW_TILE] = {0};
                for (uint32_t j = 0; j < colLoop; j++) {
                    uint32_t curLoopColOffset = j * PRE_COL_TILE;
                    uint32_t actDealtColCurLoop =
                        (j == colLoop - 1) ? (yBlockNumAligned - curLoopColOffset) : PRE_COL_TILE;
                    uint32_t actDealtColCurLoop32 = CeilDiv(actDealtColCurLoop, 32) * 32;
                    AscendC::CreateVecIndex(
                        maskIdxUb, static_cast<int32_t>(curLoopColOffset), actDealtColCurLoop);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(IdxPingPongFlag);
                    AscendC::DataCopyPad(
                        maskPatUb8[IdxPingPongFlag],
                        gSparseMask[globalRowOffset * yBlockNumAligned + curLoopColOffset],
                        AscendC::DataCopyExtParams(
                            actDealtRowCurLoop,
                            actDealtColCurLoop * sizeof(ElementSparseMask),
                            (yBlockNumAligned - actDealtColCurLoop) * sizeof(ElementSparseMask),
                            0, 0),
                        AscendC::DataCopyPadExtParams<ElementSparseMask>(
                            true, 0, (actDealtColCurLoop32 - actDealtColCurLoop), 0));
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                    AscendC::Cast(
                        maskPatUb16, maskPatUb8[IdxPingPongFlag],
                        AscendC::RoundMode::CAST_NONE, actDealtRowCurLoop * actDealtColCurLoop32);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(IdxPingPongFlag);

                    AscendC::Cast(
                        maskPatUb32, maskPatUb16,
                        AscendC::RoundMode::CAST_NONE, actDealtRowCurLoop * actDealtColCurLoop32);
                    AscendC::PipeBarrier<PIPE_V>();
                    for (uint32_t k = 0; k < actDealtRowCurLoop; k++) {
                        AscendC::CompareScalar(
                            maskPatInBitUb8[k * PRE_COL_TILE],
                            maskPatUb32[k * actDealtColCurLoop32],
                            static_cast<float>(1.0), AscendC::CMPMODE::GE, actDealtColCurLoop);
                        AscendC::PipeBarrier<PIPE_V>();
                        if (k == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(IdxPingPongFlag);
                        }
                        AscendC::GatherMask(
                            sparseIdxUb[IdxPingPongFlag][k * PRE_COL_TILE],
                            maskIdxUb,
                            maskPatInBitUb32[k * PRE_COL_TILE / 4],
                            true, actDealtColCurLoop, {1, 1, 0, 0}, rsvdCountPerRowCurColLoop[k]);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);


                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                        if (k == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                        }
                        AscendC::DataCopyPad(
                            gSparseIdx[
                                globalRowOffset * yBlockNumAligned + k * yBlockNumAligned +
                                rsvdCountPerRow[k]],
                            sparseIdxUb[IdxPingPongFlag][k * PRE_COL_TILE],
                            AscendC::DataCopyExtParams(
                                1, rsvdCountPerRowCurColLoop[k] * sizeof(ElementSparseIdx), 0, 0, 0));
                        if (k == actDealtRowCurLoop - 1) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(IdxPingPongFlag);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);

                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
                        rsvdCountPerRow[k] += rsvdCountPerRowCurColLoop[k];
                        if (k == actDealtRowCurLoop - 1) {
                            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                        }
                    }
                    IdxPingPongFlag = 1 - IdxPingPongFlag;
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(CountPingPongFlag);
                for (uint32_t k = 0; k < actDealtRowCurLoop; k++) {
                    sparseCountUb[CountPingPongFlag].SetValue(k, static_cast<int32_t>(rsvdCountPerRow[k]));
                }
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(CountPingPongFlag + 2);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(CountPingPongFlag + 2);
                AscendC::DataCopyPad(
                    gSparseCount[globalRowOffset],
                    sparseCountUb[CountPingPongFlag],
                    AscendC::DataCopyExtParams(1, actDealtRowCurLoop * sizeof(ElementSparseCount), 0, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(CountPingPongFlag);

                CountPingPongFlag = 1 - CountPingPongFlag;
            }
        }
    }
private:
    AscendC::LocalTensor<uint8_t> maskPatUb8[IO_STAGES];
    AscendC::LocalTensor<int32_t> sparseCountUb[IO_STAGES];
    AscendC::LocalTensor<uint8_t> maskPatInBitUb8;
    AscendC::LocalTensor<uint32_t> maskPatInBitUb32;
    
    AscendC::LocalTensor<int32_t> maskIdxUb;
    AscendC::LocalTensor<int32_t> sparseIdxUb[IO_STAGES];
    AscendC::LocalTensor<half> maskPatUb16;
    AscendC::LocalTensor<float> maskPatUb32;
};

}  // namespace NpuArch::Epilogue::Block
#endif