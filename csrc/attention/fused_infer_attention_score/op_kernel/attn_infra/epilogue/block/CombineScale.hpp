/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_COMBINE_SCALE_HPP
#define EPILOGUE_BLOCK_COMBINE_SCALE_HPP

#include <limits>
#include "../../../attn_infra/fused_base_defs.hpp"
#include "../../../attn_infra/arch/fused_resource.hpp"
#include "adv_api/pad/broadcast.h"
#include "adv_api/reduce/reduce.h"

namespace NpuArch::Epilogue::Block {

template <
    class OutputType_,
    class LseType_>
class CombineScale {
public:
    using ElementOutput = typename OutputType_::Element;
    using ElementLse = typename LseType_::Element;
    using ArchTag = Arch::AtlasA2;

    static constexpr uint32_t STAGE2_UB_UINT8_BLOCK_SIZE = 6144; // 24 * 64 * 4
    static constexpr uint32_t UB_UINT8_LINE_SIZE = 32768; // 1 * 64 * 128 * 4

    __aicore__ inline
    CombineScale() {}

    __aicore__ inline
    ~CombineScale() {}

    __aicore__ inline
    void init(Arch::Resource<ArchTag> &resource) {
        constexpr uint32_t LL_UB_OFFSET = 0;
        constexpr uint32_t LM_UB_OFFSET = 1 * STAGE2_UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t BROADCAST_OFFSET = 2 * STAGE2_UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t TL_UB_OFFSET = 3 * STAGE2_UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t RS_UB_OFFSET = 4 * STAGE2_UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t TS_UB_OFFSET = 5 * STAGE2_UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GL_UB_OFFSET = 6 * STAGE2_UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t BROADCASTO_OFFSET = 7 * STAGE2_UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GO_UB_OFFSET = 7 * STAGE2_UB_UINT8_BLOCK_SIZE + 1 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t GO16_UB_OFFSET = 7 * STAGE2_UB_UINT8_BLOCK_SIZE + 2 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t TEMP_REDUCE_OFFSET = 7 * STAGE2_UB_UINT8_BLOCK_SIZE + 3 * UB_UINT8_LINE_SIZE;

        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_OFFSET);
        broadCastTensor = resource.ubBuf.template GetBufferByByte<float>(BROADCAST_OFFSET);
        tlUbTensor = resource.ubBuf.template GetBufferByByte<float>(TL_UB_OFFSET);
        rsUbTensor = resource.ubBuf.template GetBufferByByte<float>(RS_UB_OFFSET);
        tsUbTensor = resource.ubBuf.template GetBufferByByte<float>(TS_UB_OFFSET);
        broadCastScaleTensor = resource.ubBuf.template GetBufferByByte<float>(BROADCAST_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_OFFSET);
        broadCastOTensor = resource.ubBuf.template GetBufferByByte<float>(BROADCASTO_OFFSET);
        toUbTensor = resource.ubBuf.template GetBufferByByte<float>(BROADCASTO_OFFSET);
        goUbTensor = resource.ubBuf.template GetBufferByByte<float>(GO_UB_OFFSET);
        loFloatUbTensor = resource.ubBuf.template GetBufferByByte<float>(GO16_UB_OFFSET);
        go16UbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_OFFSET);
        tempReduceMax = resource.ubBuf.template GetBufferByByte<uint8_t>(TEMP_REDUCE_OFFSET);
        tempReduceSum = resource.ubBuf.template GetBufferByByte<uint8_t>(TEMP_REDUCE_OFFSET);
    }

    __aicore__ inline void operator()(
        uint32_t qHeads,
        uint32_t kvSplitCoreNum,
        uint32_t headSizeV,
        __gm__ splitNode *splitInfo,
        AscendC::GlobalTensor<ElementLse> lGmTensor,
        AscendC::GlobalTensor<ElementLse> oCoreTmpGmTensor,
        AscendC::GlobalTensor<ElementOutput> oGmTensor,
        AscendC::GlobalTensor<int64_t> gActualQseqlen,
        bool inputLayoutTND = true,
        bool outputLse = false,
        AscendC::GlobalTensor<ElementLse> oLseGmTensor = AscendC::GlobalTensor<ElementLse>()
    ) {
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        int64_t vectorsubBlockID = AscendC::GetSubBlockIdx();
        int64_t subBlockNum = AscendC::GetBlockNum() * 2;
        int64_t subBlockID = AscendC::GetBlockIdx();

        for (uint32_t process = subBlockID; process < kvSplitCoreNum * 2; process += subBlockNum) {
            uint32_t batchIdx = splitInfo->batchIdx[process/2];
            uint32_t headStartIndx = splitInfo->headStartIdx[process/2];
            uint32_t headEndIndx = splitInfo->headEndIdx[process/2];
            uint32_t qStartIndx = splitInfo->qStartIdx[process/2];
            uint32_t qEndIndx = splitInfo->qEndIdx[process/2];


            uint32_t q_len = (qEndIndx - qStartIndx);
            uint32_t n_len = (headEndIndx - headStartIndx);

            uint32_t sum = q_len * n_len;
            uint32_t sum_former = q_len == 1 ? sum / 2 : (q_len / 2) * n_len;

            uint32_t addrLOffset = vectorsubBlockID == 0 ? splitInfo->lseTaskOffset[process/2] : splitInfo->lseTaskOffset[process/2] + sum_former;
            uint32_t addrOOffset = vectorsubBlockID == 0 ? splitInfo->oTaskOffset[process/2] : splitInfo->oTaskOffset[process/2] + sum_former * headSizeV;

            uint32_t prevQSeqlenSum = 0;
            if (inputLayoutTND) {
                prevQSeqlenSum = (batchIdx == 0) ?
                    0 : static_cast<uint32_t>(gActualQseqlen.GetValue(batchIdx - 1));
            }
            uint32_t baseGmOffset = prevQSeqlenSum * qHeads * headSizeV + qStartIndx * qHeads * headSizeV + headStartIndx * headSizeV;
            uint32_t gmOScalar = 0;
            if (q_len == 1) {
                gmOScalar = vectorsubBlockID == 0 ? baseGmOffset
                                                : baseGmOffset + sum_former * headSizeV;
            } else {
                uint32_t q_half = q_len / 2;
                gmOScalar = vectorsubBlockID == 0 ? baseGmOffset
                                                : baseGmOffset + q_half * qHeads * headSizeV;
            }

            uint32_t splitNum = splitInfo->splitNum[process/2];

            uint32_t splitNumAlign = (splitNum + 7) / 8 * 8;  // 32b align
            uint32_t lseBlock = vectorsubBlockID == 0 ? sum_former : sum - sum_former;
            if (lseBlock == 0) {
                continue;
            }
            uint32_t lseBlockAlign = (lseBlock + 7) / 8 * 8;  // 32b align
            int32_t count = splitNum * lseBlockAlign;
            int32_t lnCount = 1 * lseBlockAlign;
            // Initialize LSE UB space
            int32_t calcLen = splitNumAlign * lseBlockAlign;
            int32_t oCount = lseBlock * headSizeV;
            int32_t lseCount = lseBlockAlign * headSizeV;
            int32_t oCount_vector = sum * headSizeV;

            uint32_t headSizeVPad = (headSizeV + 15) / 16 * 16;

            AscendC::Duplicate(llUbTensor, std::numeric_limits<float>::lowest(), calcLen);
            AscendC::Duplicate(tlUbTensor, 0.0f, calcLen);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

            // Copy LSE from GM to UB
            uint32_t srcStride = vectorsubBlockID == 0 ? sum - sum_former : sum_former;
            AscendC::DataCopyPad(llUbTensor, lGmTensor[addrLOffset],
                                AscendC::DataCopyExtParams(splitNum, lseBlock * sizeof(float), srcStride * sizeof(float), 0, 0),
                                AscendC::DataCopyPadExtParams<float>(false, 0, lseBlockAlign - lseBlock, 0));

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            // ReduceMax
            uint32_t reduceMaxShape[] = { splitNumAlign, lseBlockAlign };
            AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, false>(lmUbTensor, llUbTensor, tempReduceMax, reduceMaxShape, true);
            AscendC::PipeBarrier<PIPE_V>();

            // Broadcast Max
            uint32_t dstShapeBroadcast[] = { splitNum, lseBlockAlign };
            uint32_t srcShapeBroadcast[] = { 1, lseBlockAlign };
            AscendC::BroadCast<float, 2, 0>(broadCastTensor, lmUbTensor, dstShapeBroadcast, srcShapeBroadcast, tempReduceSum);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Sub(tlUbTensor, llUbTensor, broadCastTensor, count);
            AscendC::PipeBarrier<PIPE_V>();

            // expf
            AscendC::Exp(tlUbTensor, tlUbTensor, count);
            AscendC::PipeBarrier<PIPE_V>();

            // ReduceSum
            uint32_t reduceSumShape[] = { splitNumAlign, lseBlockAlign };
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(rsUbTensor, tlUbTensor, tempReduceSum, reduceSumShape, true);
            AscendC::PipeBarrier<PIPE_V>();

            // Ln
            AscendC::Ln(rsUbTensor, rsUbTensor, lnCount);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Add(tsUbTensor, rsUbTensor, lmUbTensor, lnCount);
            AscendC::PipeBarrier<PIPE_V>();

            if (outputLse) {
                uint32_t baseGmOffsetLse =
                    prevQSeqlenSum * qHeads + qStartIndx * qHeads + headStartIndx;
                uint32_t gmLseScalar = 0;
                if (q_len == 1) {
                    gmLseScalar = (vectorsubBlockID == 0) ? baseGmOffsetLse
                                                          : baseGmOffsetLse + sum_former;
                } else {
                    uint32_t q_half = q_len / 2;
                    gmLseScalar = (vectorsubBlockID == 0) ? baseGmOffsetLse
                                                          : baseGmOffsetLse + q_half * qHeads;
                }

                if (q_len == 1) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);

                    AscendC::DataCopyPad(
                        oLseGmTensor[gmLseScalar], tsUbTensor,
                        AscendC::DataCopyExtParams(1, lseBlock * sizeof(float), 0, 0, 0));
                } else {
                    AscendC::Brcb(loFloatUbTensor.ReinterpretCast<uint32_t>(),
                                  tsUbTensor.ReinterpretCast<uint32_t>(),
                                  (lseBlockAlign + FLOAT_PER_BLOCK - 1) / FLOAT_PER_BLOCK,
                                  AscendC::BrcbRepeatParams(1, 8));
                    AscendC::PipeBarrier<PIPE_V>();

                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);

                    uint32_t qTile = lseBlock / n_len;
                    for (uint32_t qi = 0; qi < qTile; ++qi) {
                        AscendC::DataCopyPad(
                            oLseGmTensor[gmLseScalar + qi * qHeads],
                            loFloatUbTensor[qi * n_len * FLOAT_PER_BLOCK],
                            AscendC::DataCopyExtParams(n_len, sizeof(float), 0, 0, 0));
                    }
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
            }

            // Broadcast scale
            AscendC::BroadCast<float, 2, 0>(broadCastScaleTensor, tsUbTensor, dstShapeBroadcast, srcShapeBroadcast, tempReduceSum);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Sub(glUbTensor, llUbTensor, broadCastScaleTensor, count);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Exp(glUbTensor, glUbTensor, count);
            AscendC::PipeBarrier<PIPE_V>();

            uint32_t tokenTile = UB_UINT8_LINE_SIZE / (headSizeVPad * sizeof(float));

            if (q_len > 1 && tokenTile >= n_len && n_len % 8 == 0) {
                tokenTile = (tokenTile / n_len) * n_len;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            for (uint32_t tileStart = 0; tileStart < lseBlock; tileStart += tokenTile) {
                uint32_t curTile = (tileStart + tokenTile <= lseBlock) ?
                                    tokenTile : (lseBlock - tileStart);
                uint32_t curTileAlign = (curTile + 7) / 8 * 8;
                uint32_t oTileCount = curTile * headSizeV;
                uint32_t oTileCountPad = curTile * headSizeVPad;

                uint32_t headSizeV8 = (headSizeV + 7) / 8 * 8;
                uint32_t rightPadFloat = headSizeV8 - headSizeV;
                uint32_t dstStrideReadFloat = (headSizeVPad * sizeof(float) - headSizeV8 * sizeof(float)) / 32;

                uint32_t blockLenWrite = headSizeV * sizeof(ElementOutput);
                uint32_t blockLenWriteAligned = (blockLenWrite + 31) / 32 * 32;
                uint32_t srcStrideWrite = (headSizeVPad * sizeof(ElementOutput) - blockLenWriteAligned) / 32;                

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                for (uint32_t nIdx = 0; nIdx < splitNum; nIdx++) {

                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
 
                    AscendC::DataCopyPad(loFloatUbTensor,
                        oCoreTmpGmTensor[addrOOffset + tileStart * headSizeV + nIdx * oCount_vector],
                        AscendC::DataCopyExtParams(curTile, headSizeV * sizeof(float), 0, dstStrideReadFloat, 0),
                        AscendC::DataCopyPadExtParams<float>(true, 0, rightPadFloat, 0));

                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                    uint32_t dstShapeO[2] = { curTileAlign, headSizeVPad };
                    uint32_t srcShapeO[2] = { curTileAlign, 1 };
                    AscendC::BroadCast<float, 2, 1>(broadCastOTensor,
                        glUbTensor[nIdx * lseBlockAlign + tileStart],
                        dstShapeO, srcShapeO, tempReduceSum);
                    AscendC::PipeBarrier<PIPE_V>();

                    AscendC::Mul(toUbTensor, loFloatUbTensor, broadCastOTensor, oTileCountPad);
                    AscendC::PipeBarrier<PIPE_V>();

                    if (nIdx == 0) {
                        AscendC::Adds(goUbTensor, toUbTensor, 0.0f, oTileCountPad);
                        AscendC::PipeBarrier<PIPE_V>();
                    } else {
                        AscendC::Add(goUbTensor, toUbTensor, goUbTensor, oTileCountPad);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                }
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

                // Cast and move out
                if (std::is_same<ElementOutput, bfloat16_t>::value) {
                    AscendC::Cast(go16UbTensor, goUbTensor, AscendC::RoundMode::CAST_RINT, oTileCountPad);
                } else {
                    AscendC::Cast(go16UbTensor, goUbTensor, AscendC::RoundMode::CAST_NONE, oTileCountPad);
                }
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);

                if (q_len == 1) {
                    AscendC::DataCopyPad(oGmTensor[gmOScalar + tileStart * headSizeV], go16UbTensor,
                        AscendC::DataCopyExtParams(curTile, blockLenWrite, srcStrideWrite, 0, 0));
                } else if (tileStart % n_len == 0 && curTile % n_len == 0) {
                    uint32_t qTile = curTile / n_len;
                    uint32_t gmTileOffset = (tileStart / n_len) * qHeads * headSizeV;
                    for (uint32_t qi = 0; qi < qTile; qi++) {
                        uint32_t ubQiOffset = qi * n_len * headSizeVPad;
                        uint32_t gmQiOffset = gmOScalar + gmTileOffset + qi * qHeads * headSizeV;
                        AscendC::DataCopyPad(oGmTensor[gmQiOffset], go16UbTensor[ubQiOffset],
                            AscendC::DataCopyExtParams(n_len, blockLenWrite, srcStrideWrite, 0, 0));
                    }
                } else {
                    uint32_t remaining = curTile;
                    uint32_t ubRow = 0;
                    uint32_t flatIdx = tileStart;
                    while (remaining > 0) {
                        uint32_t qIdx = flatIdx / n_len;
                        uint32_t nStart = flatIdx % n_len;
                        uint32_t nCount = (n_len - nStart < remaining) ?
                                           (n_len - nStart) : remaining;
                        uint32_t gmOff = qIdx * qHeads * headSizeV + nStart * headSizeV;
                        uint32_t ubOffset = ubRow * headSizeVPad;
                        AscendC::DataCopyPad(oGmTensor[gmOScalar + gmOff],
                            go16UbTensor[ubOffset],
                            AscendC::DataCopyExtParams(nCount, blockLenWrite, srcStrideWrite, 0, 0));
                        ubRow += nCount;
                        flatIdx += nCount;
                        remaining -= nCount;
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

private:
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> tlUbTensor;
    AscendC::LocalTensor<float> rsUbTensor;
    AscendC::LocalTensor<float> tsUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> toUbTensor;
    AscendC::LocalTensor<float> goUbTensor;
    AscendC::LocalTensor<ElementOutput> go16UbTensor;
    AscendC::LocalTensor<float> loFloatUbTensor;

    AscendC::LocalTensor<uint8_t> tempReduceMax;
    AscendC::LocalTensor<uint8_t> tempReduceSum;
    AscendC::LocalTensor<float> broadCastTensor;
    AscendC::LocalTensor<float> broadCastScaleTensor;
    AscendC::LocalTensor<float> broadCastOTensor;
};
}

#endif // EPILOGUE_BLOCK_COMBINE_SCALE_HPP