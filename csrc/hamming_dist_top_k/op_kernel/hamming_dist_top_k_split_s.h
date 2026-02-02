/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file hamming_dist_top_k_split_s.h
 * \brief
 */
#ifndef HAMMING_DIST_TOP_K_SPLIT_S_H
#define HAMMING_DIST_TOP_K_SPLIT_S_H

#define KEEP_TAIL_TYPE_ONE_BLOCK 0
#define KEEP_TAIL_TYPE_TWO_BLOCKS 1
#define KEEP_TAIL_TYPE_TWO_TILES 2

#include <vector>
#include "hamming_dist_top_k_base.h"

namespace AscendC {

constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t MAX_SELECT_AND_CAST_COUNT = 254;
constexpr uint32_t RESET_NUM = 0U;
constexpr uint32_t COMPRESS_RATE = 8;
constexpr uint32_t VECTOR_CUBE_RATIO = 2;
constexpr uint32_t INT4B_TYPE_SIZE_DIV_RATE = 2;
constexpr uint64_t CAST_MASK = 128;
constexpr uint32_t MAX_BATCH_SIZE = 150;
constexpr uint32_t MAX_CHUNK_SIZE = 16;
constexpr uint32_t MIN_CHUNK_SIZE = 1;
constexpr uint32_t CHUNK_TOPK_MIN_SEQ_LEN = 32;

class HammingDistTopKSplitSKernel {
public:
    __aicore__ inline HammingDistTopKSplitSKernel() {}
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR keyCompressed, GM_ADDR keyCompressedRope, GM_ADDR k,
                                GM_ADDR seqLen, GM_ADDR chunkSize, GM_ADDR keyBlockTable, GM_ADDR mask,
                                GM_ADDR indices, GM_ADDR workSpace,
                                const HammingDistTopKTilingData &tilingData, TPipe *que)
    {
        const TCubeTiling &tiling = tilingData.matmulTiling;
        const TCubeTiling &tilingRope = tilingData.matmulTilingRope;
        const TopkTiling &topkTiling = tilingData.topkTiling;
        const HammingDistTopKTilingParams &tilingParam = tilingData.params;
        pipe_ = que;
        tilingData_ = tilingData;
        InitTilingParams(tiling, tilingRope, topkTiling, tilingParam);
        InitParams();
        InitGlobalBuffers(query, keyCompressed, keyCompressedRope, k, seqLen, chunkSize, keyBlockTable, mask, indices, workSpace);
        mm_.SetSubBlockIdx(0);
        mm_.Init(&tiling, pipe_);
        if (param_.supportKeyRope) {
            mmRope_.SetSubBlockIdx(0);
            mmRope_.Init(&tilingRope, pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        ComputeChunkTopKEffectLen();

        ComputeBatchSeqLenTileInfo();

        if ASCEND_IS_AIV {
            // InitLocalBuffersForUnpack();
            InitLocalBuffersForUnpackQ();
            UnpackQueryCompressed();
            pipe_->Reset();
            InitLocalBuffersForUnpackK();
            UnpackKeyCompressed(false);
            if (param_.supportKeyRope) {                
                UnpackKeyCompressed(true);
            }
            VectorNotifyCube<PIPE_MTE3>(SYNC_AIV_ONLY_ALL_FLAG, SYNC_AIV_AIC_FLAG);
            pipe_->Reset();
            InitLocalBuffersForTopK();
            // 初始化内部排序需要空间
            if (!param_.supportOffload) {
                InitLocalBuffersForTopKSort();
            }
            InitLocalBuffersForTableBlock();
        }

        if ASCEND_IS_AIC {
            CubeWaitVector(SYNC_AIV_AIC_FLAG);
        }

        if ASCEND_IS_AIV {
            blockIdx_ = blockIdx_ / VECTOR_CUBE_RATIO;
        }

        ComputeMMTopKBatchNGroup();

        if ASCEND_IS_AIV {
            SyncAivOnly<PIPE_MTE3>(SYNC_AIV_ONLY_ALL_FLAG);
            MergeTopKInGroup();
        }
    }

    __aicore__ inline void ProcessL2cache()
    {
    }

protected:
    __aicore__ inline void UnpackQueryCompressed()
    {
        if ASCEND_IS_AIC {
            return;
        }
        LocalTensor<half> constTensor = constBuf_.template Get<half>(), selectTensor = selectBuf_. template Get<half>();
        LocalTensor<half> qReduceSumTensor = qReduceSumBuf_. template Get<half>();
        LocalTensor<half> qReduceSumLastRowTensor = qReduceSumLastRowBuf_. template Get<half>();
        Duplicate<half>(constTensor, 1, param_.dimension);
        uint8_t repeatTimes = (param_.dimension + MAX_FP16_PROCESS_NUM - 1)/ MAX_FP16_PROCESS_NUM;   // 每块 128
        uint64_t qMask = MAX_FP16_PROCESS_NUM;
        uint32_t compressedDimension = param_.dimension / COMPRESS_RATE;
        uint32_t vCoreNum = 2 * uint32_t(param_.usedCoreNum);
        uint32_t pad_dim = param_.rope_dimension / 2;
        uint32_t valid_dim = param_.nope_dimension + pad_dim;
        for (uint32_t curBatch = blockIdx_; curBatch < param_.batchN; curBatch += vCoreNum) {
            uint64_t queryGmOffset = curBatch * param_.headGroupNum * compressedDimension;
            LocalTensor<uint8_t> queryCompressed = queryCompressedInQueue_.AllocTensor<uint8_t>();
            DataCopyExtParams queryCopyInParams{1, param_.headGroupNum * compressedDimension, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> queryCopyInPadParams{false, 0, 0, 0};
            DataCopyPad(queryCompressed, queryGm_[queryGmOffset], queryCopyInParams, queryCopyInPadParams);

            queryCompressedInQueue_.EnQue(queryCompressed);
            queryCompressed = queryCompressedInQueue_.DeQue<uint8_t>();
            PipeBarrier<PIPE_V>();
            uint8_t selectRepeatTimes = param_.headGroupNum * param_.dimension / MAX_FP16_PROCESS_NUM;  // param_.dimension时，param_.headGroupNum需是偶数
            SelectCustom<half>(selectTensor, queryCompressed, constTensor, selectRepeatTimes);
            PipeBarrier<PIPE_V>();
            if (param_.supportKeyRope) {
                for (uint32_t i = 0; i < param_.headGroupNum; i++) {
                    // DumpTensor(selectTensor[i * param_.dimension], 139, 672);
                    Duplicate<half>(selectTensor[i * param_.dimension + valid_dim], 0, pad_dim);
                    // DumpTensor(selectTensor[i * param_.dimension], 141, 672);
                }
                PipeBarrier<PIPE_V>();
            }

            queryCompressedInQueue_.FreeTensor(queryCompressed);

            LocalTensor<half> qHashTensor = selectTensor;
            if (param_.headGroupNum > 0) {
                static constexpr AscendC::CumSumConfig cumSumConfig{false, false, true};
                const AscendC::CumSumInfo cumSumInfo{param_.headGroupNum, param_.dimension};
                AscendC::CumSum<half, cumSumConfig>(qReduceSumTensor, qReduceSumLastRowTensor, selectTensor, cumSumInfo);
                PipeBarrier<PIPE_V>();
                // DumpTensor(qReduceSumTensor, 152, 2048);
                if (param_.headGroupNum > 8) {
                    /* 将取值缩放到[-8,8]范围内 */
                    uint32_t div = matmul::CeilDiv(param_.headGroupNum, 8);
                    half reciprocalDiv = static_cast<half>((float)1.0 / div);
                    AscendC::Muls(qReduceSumLastRowTensor, qReduceSumTensor[(param_.headGroupNum - 1) * param_.dimension],
                        reciprocalDiv, qMask, repeatTimes, {1, 1, 8, 8});
                    PipeBarrier<PIPE_V>();
                }
                qHashTensor = qReduceSumLastRowTensor;
            }

            LocalTensor<int4b_t> queryUnpacked = queryUnpackedOutQueue_.AllocTensor<int4b_t>();
            Cast<int4b_t, half>(queryUnpacked, qHashTensor, RoundMode::CAST_CEIL, qMask, repeatTimes, {1, 1, 2, 8});
            PipeBarrier<PIPE_V>();
            queryUnpackedOutQueue_.EnQue(queryUnpacked);
            queryUnpacked = queryUnpackedOutQueue_.DeQue<int4b_t>();
            uint64_t unpackQGmOffset = queryGmOffset * COMPRESS_RATE / param_.headGroupNum;
            DataCopyExtParams copyQOutParams{1, static_cast<uint32_t>(param_.dimension / 2), 0, 0, 0}; /* 2: 1 / size of int4b_t */
            DataCopyPad(qUnpackGm_[unpackQGmOffset], queryUnpacked, copyQOutParams);
            queryUnpackedOutQueue_.FreeTensor(queryUnpacked);

            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
    }

    __aicore__ inline void DataCopyInForKeyCompressed(
        bool useCopyPad, uint32_t copySeqLen, uint32_t compressedDimension, LocalTensor<uint8_t> &keyCompressedLocal, uint64_t keyGmOffset, GlobalTensor<uint8_t>& keyCompressedGm)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (useCopyPad) {
            // copySeqLen <= TileN1，compressedDimension值固定，block数据不会超过uint16的边界值
            DataCopyExtParams copyInParams{1, static_cast<uint32_t>(copySeqLen * compressedDimension), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> copyInPadParams{false, 0, 0, 0};
            DataCopyPad(keyCompressedLocal, keyCompressedGm[keyGmOffset], copyInParams, copyInPadParams);
        } else {
            DataCopy(keyCompressedLocal, keyCompressedGm[keyGmOffset], copySeqLen * compressedDimension);
        }
    }

    __aicore__ inline void DataCopyOutForKeyUnpacked(uint32_t copySeqLen, LocalTensor<int4b_t> &keyUnpackedLocal, uint64_t keyGmOffset)
    {
        if ASCEND_IS_AIC {
            return;
        }
        // copySeqLen <= TileN1，param_.dimension值固定，block数据不会超过uint16的边界值
        DataCopyParams copyParams{1, static_cast<uint16_t>(copySeqLen * param_.dimension / 2 / BLOCK_CUBE), 0, 0}; // 2: 1/2, size of int4b_t
        DataCopy(unpackGm_[keyGmOffset], keyUnpackedLocal, copyParams); // output to outQueue1 with DB_ON
    }

    __aicore__ inline void UnpackKeyCompressed(bool isKeyRope)
    {
       if ASCEND_IS_AIC {
            return;
        }
        uint32_t dimension = isKeyRope ? param_.rope_dimension : param_.nope_dimension;;
        GlobalTensor<int4b_t> unpackGm = isKeyRope ? kRopeUnpackGm_ : unpackGm_;
        GlobalTensor<uint8_t> keyCompressedGm = isKeyRope ? keyCompressedRopeGm_ : keyCompressedGm_;
        LocalTensor<half> constTensor = constBuf_.template Get<half>(), selectTensor = selectBuf_.template Get<half>();
        Duplicate<half>(constTensor, 1, dimension);
        uint32_t curBatchIdx = 0, curHeadIdx = 0, seqLens = 0, kScalar = 0, effectLen = 0, allSLoops = 0, sloops = 0, minSloops = 0, maxSloops = 0, tailSloops = 0, tailSeqLens = 0, preSloops = 0, blockIdx = 0, selectAndCastCount = 0, selectAndCastOffset = 0, compressedDimension = dimension / COMPRESS_RATE, vCoreNum = 2 * uint32_t(param_.usedCoreNum);
        uint64_t keyUnpackInGmOffset = 0;
        bool hasTailSeqLens = false;
        for (uint32_t i = 0; i < param_.batchN; i++) {
            curBatchIdx = i / param_.head;
            // 当前batch不计算Key直接跳过
            if (supportMask_) {
                bool batchMask = maskGm_.GetValue(curBatchIdx);
                //YF_LOG("curBatchIdx = %d, batchMask = %d\n", curBatchIdx, batchMask);
                if (!batchMask) {
                    continue;
                }
            }
            curHeadIdx = i % param_.head;
            seqLens = uint32_t(seqLenGm_.GetValue(curBatchIdx));
            kScalar = uint32_t(kGm_.GetValue(curBatchIdx));
            effectLen = effectLenArr_[curBatchIdx];
            if (seqLens == 0 || kScalar == 0 || effectLen == 0) {
                continue; // seqLen=0或者kScalar==0或者chunkTopK effectLen==0直接跳过
            }
            allSLoops = DivCeil(seqLens, param_.tileN1), tailSeqLens = seqLens % param_.tileN1, hasTailSeqLens = tailSeqLens != 0, tailSloops = allSLoops % vCoreNum, minSloops = allSLoops / vCoreNum, maxSloops = tailSloops == 0 ? minSloops : minSloops + 1;
            if (blockIdx_ < tailSloops) {
                sloops = maxSloops, preSloops = blockIdx_ * sloops;
            } else {
                sloops = minSloops, preSloops = tailSloops == 0 ? blockIdx_ * minSloops : tailSloops * maxSloops + (blockIdx_ - tailSloops) * minSloops;
            }
            bool isTailSloopOwner = preSloops + sloops == allSLoops, isTailSeqLenNotAlignedForUint8 = isTailSloopOwner && (tailSeqLens % 2 != 0);
            for (uint32_t j = 0; j < sloops; j++) {
                bool isLastLoop = j == sloops - 1, useDataCopyPadForKeyCompressed = isLastLoop && isTailSeqLenNotAlignedForUint8;
                uint32_t copySeqLen = (isTailSloopOwner && isLastLoop && hasTailSeqLens) ? tailSeqLens : param_.tileN1, selectAndCastLoops = copySeqLen / MAX_SELECT_AND_CAST_COUNT, selectAndCastTail = copySeqLen % MAX_SELECT_AND_CAST_COUNT;
                if (isContinuousBatch_) {
                    blockIdx = keyBlockTableGm_.GetValue(curBatchIdx * param_.blockCount + preSloops + j), keyUnpackInGmOffset = (blockIdx * param_.head + curHeadIdx) * param_.tileN1 * compressedDimension;
                } else {
                    keyUnpackInGmOffset = (i * param_.maxSeqLen + (preSloops + j) * param_.tileN1) * compressedDimension;
                }
                LocalTensor<uint8_t> keyCompressedLocal = keyCompressedInBuf_.AllocTensor<uint8_t>();
                LocalTensor<int4b_t> keyUnpackedLocal = keyUnpackedOutBuf_.AllocTensor<int4b_t>();
                DataCopyInForKeyCompressed(useDataCopyPadForKeyCompressed, copySeqLen, compressedDimension, keyCompressedLocal, keyUnpackInGmOffset, keyCompressedGm);
                keyCompressedInBuf_.EnQue<uint8_t>(keyCompressedLocal);
                keyCompressedLocal = keyCompressedInBuf_.DeQue<uint8_t>();
                uint64_t keyUnpackOutGmOffset = (i * param_.maxSeqLen + (preSloops + j) * param_.tileN1) * dimension;
                // support key rope
                if (dimension > 128 || isKeyRope) {
                    uint32_t tileN1RepeatTimes = (copySeqLen * dimension + MAX_FP16_PROCESS_NUM - 1) / MAX_FP16_PROCESS_NUM;
                    selectAndCastLoops = (tileN1RepeatTimes + MAX_SELECT_AND_CAST_COUNT - 1) / MAX_SELECT_AND_CAST_COUNT;
                    for (uint32_t k = 0; k < selectAndCastLoops; k++) {
                        uint32_t localStart = k * MAX_SELECT_AND_CAST_COUNT;
                        uint32_t remainRepeats = tileN1RepeatTimes - localStart;
                        uint8_t repeatTimes = remainRepeats > MAX_SELECT_AND_CAST_COUNT ? MAX_SELECT_AND_CAST_COUNT : remainRepeats;  // 需<=255
                        uint32_t elemOffset = localStart * MAX_FP16_PROCESS_NUM;
                        uint32_t maskOffset = elemOffset / COMPRESS_RATE;
                        // YF_LOG("localStart=%d, remainRepeats=%d, repeatTimes=%d, elemOffset=%d, maskOffset=%d\n", localStart, remainRepeats, repeatTimes, elemOffset, maskOffset);
                        if (repeatTimes > 0) {
                            SelectCustom<half>(selectTensor, keyCompressedLocal[maskOffset], constTensor, repeatTimes);
                            Cast<int4b_t, half>(keyUnpackedLocal[elemOffset], selectTensor, RoundMode::CAST_CEIL, CAST_MASK, repeatTimes, {1, 1, 2, 8});
                        }
                    }
                    keyUnpackedOutBuf_.EnQue(keyUnpackedLocal);
                    keyUnpackedLocal = keyUnpackedOutBuf_.DeQue<int4b_t>();
                    DataCopyParams copyParams{1, static_cast<uint16_t>(tileN1RepeatTimes * MAX_FP16_PROCESS_NUM / 2 / BLOCK_CUBE), 0, 0};  // 2: 1/2, size of int4b_t
                    DataCopy(unpackGm[keyUnpackOutGmOffset], keyUnpackedLocal, copyParams);  // output to outQueue1 with DB_ON
                    keyCompressedInBuf_.FreeTensor(keyCompressedLocal);
                    keyUnpackedOutBuf_.FreeTensor(keyUnpackedLocal);
                } else {
                    bool isSelectAndCastHasTail = selectAndCastTail != 0;
                    selectAndCastLoops = isSelectAndCastHasTail ? selectAndCastLoops + 1 : selectAndCastLoops;
                    for (uint32_t k = 0; k < selectAndCastLoops; k++) {
                        selectAndCastCount = (isSelectAndCastHasTail && k == selectAndCastLoops - 1) ? selectAndCastTail : MAX_SELECT_AND_CAST_COUNT, selectAndCastOffset = k * MAX_SELECT_AND_CAST_COUNT * dimension;
                        SelectCustom<half>(selectTensor[selectAndCastOffset], keyCompressedLocal[selectAndCastOffset / COMPRESS_RATE], constTensor, static_cast<uint8_t>(selectAndCastCount));
                        Cast<int4b_t, half>(keyUnpackedLocal[selectAndCastOffset], selectTensor[selectAndCastOffset], RoundMode::CAST_CEIL, CAST_MASK, static_cast<uint8_t>(selectAndCastCount), {1, 1, 2, 8});
                    }
                    keyCompressedInBuf_.FreeTensor(keyCompressedLocal);
                    keyUnpackedOutBuf_.EnQue<int4b_t>(keyUnpackedLocal);
                    keyUnpackedLocal = keyUnpackedOutBuf_.DeQue<int4b_t>();
                    DataCopyOutForKeyUnpacked(copySeqLen, keyUnpackedLocal, keyUnpackOutGmOffset);
                    keyUnpackedOutBuf_.FreeTensor(keyUnpackedLocal);
                }
            }
        }
    }

    __aicore__ inline void ComputeMM(uint32_t batchIdx, uint32_t curHeadIdx, uint32_t curCoreDealSize, uint32_t tileIdx)
    {
        if ASCEND_IS_AIV {
            return;
        }

        mm_.SetOrgShape(param_.M, curCoreDealSize, param_.ka);     // 原始计算形状=(M, N, Ka/Kb)=(1, 8192, 512)
        mm_.SetSingleShape(param_.M, curCoreDealSize, param_.ka);  // 单核计算形状=(M, N, Ka/Kb)=(1, 8192, 512)
        float tmp = 1;
        uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
        mm_.SetQuantScalar(ans);
        mmOffsetA_ = batchIdx * 1 * param_.head * (param_.ka + param_.rope_ka) + curHeadIdx * (param_.ka + param_.rope_ka);
        mmOffsetB_ = (batchIdx * param_.maxSeqLen * param_.head + curHeadIdx * param_.maxSeqLen + tileIdx * tileSeqLenSize_) * param_.kb;
        mmOffsetC_ =  batchIdx * param_.maxSeqLen * param_.head  + curHeadIdx * param_.maxSeqLen + tileIdx * tileSeqLenSize_;
        mm_.SetTensorA(qUnpackGm_[mmOffsetA_], AMatmulType::isTrans);
        mm_.SetTensorB(unpackGm_[mmOffsetB_], BMatmulType::isTrans);
        mm_.IterateAll(matmulGm_[mmOffsetC_]);     // d

        if (param_.supportKeyRope) {
            SetFlag<HardEvent::FIX_MTE2>(eventIDFIX_MTE2);
            WaitFlag<HardEvent::FIX_MTE2>(eventIDFIX_MTE2);
            // DumpTensor(matmulGm_[mmOffsetC_], 377, curCoreDealSize);

            mmRope_.SetOrgShape(param_.M, curCoreDealSize, param_.rope_ka);     // 原始计算形状=(M, N, Ka/Kb)=(1, 8192, 64)
            mmRope_.SetSingleShape(param_.M, curCoreDealSize, param_.rope_ka);  // 单核计算形状=(M, N, Ka/Kb)=(1, 8192, 64)
            mmRope_.SetQuantScalar(ans);
            mmOffsetARope_ = mmOffsetA_ + param_.ka;
            mmOffsetBRope_ = (batchIdx * param_.maxSeqLen * param_.head + curHeadIdx * param_.maxSeqLen + tileIdx * tileSeqLenSize_) * param_.rope_kb;
            mmRope_.SetTensorA(qUnpackGm_[mmOffsetARope_], AMatmulType::isTrans);
            mmRope_.SetTensorB(kRopeUnpackGm_[mmOffsetBRope_], BMatmulType::isTrans);
            mmRope_.IterateAll(matmulGm_[mmOffsetC_], 1);  // c
            // DumpTensor(matmulGm_[mmOffsetC_], 387, curCoreDealSize);
        }
    }

    __aicore__ inline void ComputeBatchSeqLenTileInfo()
    {
        uint32_t batchSeqLenTileNum = 0;
        for (uint32_t batchIdx = 0; batchIdx < param_.batch; batchIdx++){
            uint32_t curSeqLen = uint32_t(seqLenGm_.GetValue(batchIdx));
            uint32_t curSeqLenTileNum = DivCeil(curSeqLen, tileSeqLenSize_);
            batchSeqLenTileNum += curSeqLenTileNum;
            batchSeqTileN[batchIdx] = batchSeqLenTileNum;
        }
    }

    __aicore__ inline void ComputeChunkTopKEffectLen()
    {
        uint32_t lastLen = 0, effectLen = 0, maxChunkSize = MIN_CHUNK_SIZE;
        for (uint32_t bIdx = 0; bIdx < param_.batch; bIdx++) {
            uint32_t seqLen = uint32_t(seqLenGm_.GetValue(bIdx));
            uint32_t chunkSize = isChunkTopK_ ? uint32_t(chunkSizeGm_.GetValue(bIdx)) : MIN_CHUNK_SIZE;
            maxChunkSize = chunkSize > maxChunkSize ? chunkSize : maxChunkSize;
            minChunkSize_ = chunkSize < minChunkSize_ ? chunkSize : minChunkSize_;
            if(chunkSize == 1 || chunkSize == 8 || chunkSize == 16)
            {
                // seqLen <= 32时即effectLen=0，跳过hammingDistTopK
                if (seqLen <= CHUNK_TOPK_MIN_SEQ_LEN) {
                    lastLen = seqLen;
                    effectLen = 0;
                } else {
                    lastLen = (chunkSize != 0) ? (seqLen % chunkSize + MAX_CHUNK_SIZE > seqLen ? seqLen : seqLen % chunkSize + MAX_CHUNK_SIZE) : 0; // min
                    effectLen = seqLen - lastLen > 0 ? seqLen - lastLen : 0; // max
                }
                effectLenArr_[bIdx] = effectLen;
            } 
            else if (chunkSize == 64)
            {
                effectLenArr_[bIdx] = ((seqLen + 63)/64)*64;
            }
            else if (chunkSize == 128)
            {
                effectLenArr_[bIdx] = ((seqLen + 127)/128)*128;
            }
            else
            {
                effectLenArr_[bIdx] = 0;
            }
            
        }
        uint32_t maxSeqLenLastLen = param_.maxSeqLen % maxChunkSize + MAX_CHUNK_SIZE > param_.maxSeqLen ? 
                                    param_.maxSeqLen : param_.maxSeqLen % maxChunkSize + MAX_CHUNK_SIZE;
        maxEffectLen_ = param_.maxSeqLen - maxSeqLenLastLen > 0 ? param_.maxSeqLen - maxSeqLenLastLen : 0;
    }

    __aicore__ inline void ComputeMMTopKBatchNGroup()
    {
        /*
        batchN核分组示例

        示例1
        batch : 2
        headN = 8 = 1 * 8
        coreNum = 20
        SeqLen 86400 11296 ; totalSeqTileN 6 + 1
        coreNumInGroup=3 coreId:0  (0-2 16384 batch 0) 8 (3-4 16384 batch 0) 16 (5 4480 batch 0) (6 11296 batch 1)
        coreNumInGroup=3 coreId:1  (0-2 16384 batch 0)9 (3-4 16384 batch 0) 17 (5 4480 batch 0) (6 11296 batch 1)
        coreNumInGroup=3 coreId:2  (0-2 16384 batch 0)10 (3-4 16384 batch 0) 18 (5 4480 batch 0) (6 11296 batch 1)
        coreNumInGroup=3 coreId:3  (0-2 16384 batch 0)11 (3-4 16384 batch 0) 19 (5 4480 batch 0) (6 11296 batch 1)
        coreNumInGroup=2 coreId:4 (0-3 16384 batch 0) 12 (4-5 16384 4480 batch 0) (6 11296 batch 1)
        coreNumInGroup=2 coreId:5 (0-3 16384 batch 0) 13  (4-5 16384 4480 batch 0) (6 11296 batch 1)
        coreNumInGroup=2 coreId:6 (0-3 16384 batch 0) 14  (4-5 16384 4480 batch 0) (6 11296 batch 1)
        coreNumInGroup=2 coreId:7 (0-3 16384 batch 0) 15  (4-5 16384 4480 batch 0) (6 11296 batch 1)

        示例2
        batchN = 240 = 30 * 8
        coreNum = 20
        batchIdx = 0
        SeqLen 16384
        coreNumInGroup=3 coreId:0  (0-9 16384 batch 0 -9) 8 (10-19 16384 batch 10-19)16 (20-29 16384 batch 10-19)
        coreNumInGroup=3 coreId:1  (0-9 16384 batch 0-9)9 (10-19 16384 batch 10-19) 17 (20-29 16384 batch 10-19) 
        coreNumInGroup=3 coreId:2  (0-9 16384 batch 0-9)10 (10-19 16384 batch 10-19) 18 (20-29 16384 batch 10-19)
        coreNumInGroup=3 coreId:3  (0-9 16384 batch 0-9)11 (10-19 16384 batch 10-19) 19 (20-29 16384 batch 10-19)
        coreNumInGroup=2 coreId:4 (0-14 16384 batch 0-14) 12 (15-29 16384 batch 15-29) 
        coreNumInGroup=2 coreId:5 (0-14 16384 batch 0-14) 13 (15-29 16384 batch 15-29) 
        coreNumInGroup=2 coreId:6 (0-14 16384 batch 0-14) 14 (15-29 16384 batch 15-29) 
        coreNumInGroup=2 coreId:7 (0-14 16384 batch 0-14) 15 (15-29 16384 batch 15-29) 
        */
        uint32_t cCoreNumInGroup = param_.usedCoreNum / param_.head; //param_.head is num_kv_heads
        cCoreNumInGroup = (blockIdx_ % param_.head < param_.usedCoreNum % param_.head) ? cCoreNumInGroup + 1: cCoreNumInGroup;
        uint32_t cCoreIdxInGroup = blockIdx_ / param_.head;
        uint32_t curBatchIdx = 0;
        uint32_t curHeadIdx = blockIdx_ % param_.head;


        // 按照两个vector 两倍的tileN2大小切分
        uint32_t totalSeqTileN = batchSeqTileN[param_.batch - 1];
        uint32_t minTileNumInCore = totalSeqTileN / cCoreNumInGroup;
        uint32_t maxTileNumInCore = totalSeqTileN % cCoreNumInGroup == 0 ? minTileNumInCore : minTileNumInCore + 1;
        uint32_t maxTileNumCoreIdxInGroup = totalSeqTileN % cCoreNumInGroup == 0 ? 0 : totalSeqTileN % cCoreNumInGroup - 1;
        uint32_t tileNumInCore = cCoreIdxInGroup < totalSeqTileN % cCoreNumInGroup ? maxTileNumInCore : minTileNumInCore;

        uint32_t minTileNumStartTileIdx = (maxTileNumCoreIdxInGroup + 1) * maxTileNumInCore + (cCoreIdxInGroup - maxTileNumCoreIdxInGroup - 1) * minTileNumInCore;
        uint32_t startTileIdx = cCoreIdxInGroup <= maxTileNumCoreIdxInGroup ? tileNumInCore * cCoreIdxInGroup : minTileNumStartTileIdx;
        uint32_t endTileIdx = startTileIdx + tileNumInCore - 1;

        
        // CV流水外循环
        for (uint32_t tileIdx = startTileIdx; tileIdx <= endTileIdx; tileIdx++) {
            for (uint32_t i = 0; i < param_.batch; i++) {
                if (tileIdx < batchSeqTileN[i]) {
                    curBatchIdx = i;
                    break;
                }
            }
            // 是否跳过
            if (supportMask_) {
                bool batchMask = maskGm_.GetValue(curBatchIdx);
                if (!batchMask) {
                    //YF_LOG("curBatchIdx = %d, batchMask = %d\n", curBatchIdx, batchMask);
                    continue;
                }
            }
            uint32_t curSeqLen = uint32_t(seqLenGm_.GetValue(curBatchIdx));
            uint32_t curKScalar = uint32_t(kGm_.GetValue(curBatchIdx));
            uint32_t curEffectLen = effectLenArr_[curBatchIdx];
            if (curSeqLen == 0 || curKScalar == 0 || curEffectLen == 0) {
                continue; // seqLen=0或者kScalar==0或者chunkTopK effectLen==0直接跳过
            }
            uint32_t tileSTailSize = curSeqLen % tileSeqLenSize_;
            uint32_t realTileIdx = curBatchIdx == 0 ? tileIdx : tileIdx - batchSeqTileN[curBatchIdx - 1];
            uint32_t curCoreDealSize = ((realTileIdx + 1) * tileSeqLenSize_ <= curSeqLen) ? tileSeqLenSize_ : tileSTailSize;

            if ASCEND_IS_AIC {
                ComputeMM(curBatchIdx, curHeadIdx, curCoreDealSize, realTileIdx);   
                CubeNotifyVector<PIPE_FIX>(SYNC_AIC_ONLY_ALL_FLAG, SYNC_AIC_AIV_FLAG2 + innerSplitGMPingPongFlag_);
            }
            if ASCEND_IS_AIV {
                VectorWaitCube(SYNC_AIC_AIV_FLAG2 + innerSplitGMPingPongFlag_);
                ComputeTopK(curCoreDealSize, curBatchIdx, curHeadIdx, realTileIdx, false);
            }
            innerSplitGMPingPongFlag_ &= 1;
            innerSplitGMPingPongFlag_ ^= 1;
        }
    }

    __aicore__ inline void ComputeTopK(uint32_t curCoreDealSize, uint32_t curBatchIdx, uint32_t curHeadIdx, uint32_t tileIdx, bool isApproxiTopK)
    {
        if ASCEND_IS_AIC {
            return;
        }
        
        uint32_t curVectorDealSize;
        uint32_t vectorTileSSize = tileSeqLenSize_ / VECTOR_CUBE_RATIO; // 8092/2=4096
        uint32_t curSeqLen = uint32_t(seqLenGm_.GetValue(curBatchIdx)); // 27761
        uint32_t curKScalar = uint32_t(kGm_.GetValue(curBatchIdx)); //125
        
        if (subBlockIdx_ == 0) {
            curVectorDealSize = curCoreDealSize < vectorTileSSize ? curCoreDealSize : vectorTileSSize;
        } else if (subBlockIdx_ == 1 && curCoreDealSize > vectorTileSSize) {
            curVectorDealSize = curCoreDealSize - vectorTileSSize;
        } else {
            return; // S切分尾块无法分到第二个vector时，直接返回
        }


        uint32_t curEffectLen = effectLenArr_[curBatchIdx];
        uint32_t beforeCurVectorDealSize = tileIdx * tileSeqLenSize_ + subBlockIdx_ * vectorTileSSize;
        if (beforeCurVectorDealSize >= curEffectLen) {
            return; // chunkTopK如果当前tileIdx下，超出effectLen大小的直接跳过
        }
        uint32_t curChunkSize = isChunkTopK_ ? uint32_t(chunkSizeGm_.GetValue(curBatchIdx)) : MIN_CHUNK_SIZE; // chunkTopK与常规TopK兼容
        // chunkTopK、常规TopK，当前有效处理长度

        uint32_t lastSubBlockSeqLenPadded = 0; // for chunk_size = 16
        uint32_t lastSubBlockSeqLenTruncated = 0; // for chunk_size = 128
        uint32_t curVectorTopKDealSize  = 0;

        if(curChunkSize == 1 || curChunkSize == 8 || curChunkSize == 16)
        {
            lastSubBlockSeqLenTruncated = (curEffectLen - 1) % vectorTileSSize + 1;
            //uint32_t curVectorTopKDealSize = beforeCurVectorDealSize + curVectorDealSize > curEffectLen ? curVectorTopKDealSize / 
            curVectorTopKDealSize = beforeCurVectorDealSize + curVectorDealSize > curEffectLen ? lastSubBlockSeqLenTruncated / curChunkSize : curVectorDealSize / curChunkSize;
        }
        else if (curChunkSize == 64 || curChunkSize == 128)
        {
            lastSubBlockSeqLenPadded =  (curEffectLen - 1) % vectorTileSSize + 1;
            curVectorTopKDealSize =  DivCeil(curVectorDealSize, curChunkSize);
        }
        else
        {
            return;
        }

        curKScalar = isApproxiTopK ? DivCeil(curVectorTopKDealSize * curKScalar, isChunkTopK_ ? curEffectLen / curChunkSize : curSeqLen) : curKScalar;
        uint32_t startSeqIdx = curBatchIdx * param_.head * param_.maxSeqLen + curHeadIdx * param_.maxSeqLen + tileIdx * tileSeqLenSize_ + subBlockIdx_ * vectorTileSSize;
        uint32_t startIdxInCurWholeSeq = tileIdx * tileSeqLenSize_ / curChunkSize + subBlockIdx_ * vectorTileSSize / curChunkSize;
        uint32_t topKResultInterval = isApproxiTopK ? param_.maxK : DivCeil(maxEffectLen_, param_.tileN2) * param_.maxK;
        uint32_t topkResultOffset = curBatchIdx * param_.head * topKResultInterval + curHeadIdx * topKResultInterval + tileIdx * VECTOR_CUBE_RATIO * curKScalar + subBlockIdx_ * curKScalar;
        

        
        uint32_t chunkPerBlock = param_.tileN1 / curChunkSize; //128/16=8
        uint32_t lastSubBlockChunkNum =  0;
        if(curChunkSize == 1 || curChunkSize == 8 || curChunkSize == 16)
        {
            lastSubBlockChunkNum = lastSubBlockSeqLenTruncated / curChunkSize; 
        }
        else
        {
            lastSubBlockChunkNum = lastSubBlockSeqLenPadded / curChunkSize; 
        }

        uint32_t curSeqLenTileNum = DivCeil(curSeqLen, tileSeqLenSize_);
        
        uint32_t lastTileSeqLen = (curEffectLen - 1) % tileSeqLenSize_ + 1;
        //uint32_t skipTailChunkNum = DivCeil(SKIP_TAIL_TOKEN_NUM, curChunkSize);
        //uint32_t skipHeadChunkNum = DivCeil(SKIP_HEAD_TOKEN_NUM, curChunkSize);
        uint32_t skipTailChunkNum = param_.recent;
        uint32_t skipHeadChunkNum = param_.sink;
        
        bool flagLastTileHasTwoSubBlocks = (lastTileSeqLen > vectorTileSSize);       
        int keepTailType = KEEP_TAIL_TYPE_ONE_BLOCK;
        uint32_t lastSubBlockTailChunkNum = min(lastSubBlockChunkNum, skipTailChunkNum);  //# of chunks that are kept in the last sub block (vector)
        // YF_LOG("curEffectLen=%d, curSeqLen=%d, lastTileSeqLen=%d, tileSeqLenSize_=%d, lastSubBlockSeqLenPadded=%d, flagLastTileHasTwoSubBlocks=%d, lastSubBlockChunkNum=%d, lastSubBlockTailChunkNum=%d\n", curEffectLen,curSeqLen, lastTileSeqLen, tileSeqLenSize_, lastSubBlockSeqLenPadded, flagLastTileHasTwoSubBlocks, lastSubBlockChunkNum, lastSubBlockTailChunkNum); 
        uint32_t secondLastSubBlockTailChunkNum = 0; //# of chunks that are kept in the second last sub block (vector)
        if(lastTileSeqLen < skipTailChunkNum*curChunkSize)  //SKIP_TAIL_BLOCK_NUM assumes block_size=128
        {
            keepTailType = KEEP_TAIL_TYPE_TWO_TILES;
            secondLastSubBlockTailChunkNum = skipTailChunkNum - lastSubBlockTailChunkNum;
        }
        else if(flagLastTileHasTwoSubBlocks && (lastTileSeqLen % vectorTileSSize < skipTailChunkNum*curChunkSize) )
        {
            keepTailType = KEEP_TAIL_TYPE_TWO_BLOCKS;
            secondLastSubBlockTailChunkNum = skipTailChunkNum - lastSubBlockTailChunkNum;
        }
        else
        {
            keepTailType = KEEP_TAIL_TYPE_ONE_BLOCK;
            secondLastSubBlockTailChunkNum = 0;
        }
        // YF_LOG("keepTailType=%d, lastSubBlockChunkNum=%d, secondLastSubBlockTailChunkNum=%d\n", keepTailType, lastSubBlockChunkNum, secondLastSubBlockTailChunkNum);


        /* we do not use Case 1 such that we can easily add head and tail top-k chunks
        // Case1 若curVectorTopKDealSize < curKScalar_, 无需做topk。
        if (curVectorTopKDealSize <= curKScalar) {      // TODO
            DataMoveWithoutTopK(curEffectLen, curVectorTopKDealSize, curChunkSize, startSeqIdx, startIdxInCurWholeSeq, topkResultOffset, isApproxiTopK);
            return; 
        }
        */

        // Case2 一次性搬运curVectorTopKDealSize大小, topK耗尽模式
        LocalTensor<half> topKValueInTensor = topKValueInQueue_.AllocTensor<half>();

        // YF_LOG("curBatchIdx = %d curHeadIdx = %d curVectorTopKDealSize = %d\n", curBatchIdx, curHeadIdx, curVectorTopKDealSize);
        // 2.1 chunkTopK 需要对chunk内先求max值再做topK
        if (curChunkSize > MIN_CHUNK_SIZE) {
            LocalTensor<half> chunkReduceMaxValueInTensor = chunkReduceMaxValueInQueue_.AllocTensor<half>();
            ReduceMaxCustom(matmulGm_[startSeqIdx], chunkReduceMaxValueInTensor, topKValueInTensor, curVectorTopKDealSize, curChunkSize);
            chunkReduceMaxValueInQueue_.EnQue(chunkReduceMaxValueInTensor);
            chunkReduceMaxValueInTensor = chunkReduceMaxValueInQueue_.DeQue<half>();
            chunkReduceMaxValueInQueue_.FreeTensor(chunkReduceMaxValueInTensor);
        } else {
            // 2.2 常规TopK 直接做topK curVectorTopKDealSize为分核后的处理长度，不会引入转换异常
            DataCopyExtParams copyInParams{1, static_cast<uint32_t>(curVectorTopKDealSize * sizeof(half)), 0, 0, 0};
            DataCopyPadExtParams<half> copyInPadParams{false, 0, 0, 0};
            DataCopyPad(topKValueInTensor, matmulGm_[startSeqIdx], copyInParams, copyInPadParams);
        }

      

        if( (tileIdx == 0) && (subBlockIdx_ == 0) )
        {
            Duplicate(topKValueInTensor, static_cast<half>(MAX_HALF_VALUE), min(skipHeadChunkNum, curVectorTopKDealSize));
        }
        
        switch(keepTailType)
        {
            case KEEP_TAIL_TYPE_ONE_BLOCK:
                if(tileIdx == curSeqLenTileNum-1)
                {
                    
                    // YF_LOG("skipTailChunkNum=%d, skipHeadChunkNum=%d, keepTailType=%d, lastSubBlockTailChunkNum=%d, secondLastSubBlockTailChunkNum=%d,curVectorTopKDealSize=%d, flagLastTileHasTwoSubBlocks=%d\n", skipTailChunkNum, skipHeadChunkNum, keepTailType, lastSubBlockTailChunkNum, secondLastSubBlockTailChunkNum, curVectorTopKDealSize, flagLastTileHasTwoSubBlocks);

                    if(flagLastTileHasTwoSubBlocks)
                    {
                        if(subBlockIdx_ == 1)
                        {
                            //copy last 
                            //YF_LOG("ldeng 501\n")
                            // DumpTensor(topKValueInTensor, 502, topKValueInTensor.GetSize());
                            FillMaxValueFromTail(topKValueInTensor, curVectorTopKDealSize, lastSubBlockTailChunkNum,curChunkSize);
                            // DumpTensor(topKValueInTensor, 504, topKValueInTensor.GetSize());
                        }
                    }
                    else
                    {
                        if(subBlockIdx_ == 0)
                        {
                            //copy last
                            // YF_LOG("ldeng 512, curVectorTopKDealSize=%d, lastSubBlockTailChunkNum=%d\n", curVectorTopKDealSize, lastSubBlockTailChunkNum)
                            // DumpTensor(topKValueInTensor, 513, topKValueInTensor.GetSize());
                            FillMaxValueFromTail(topKValueInTensor, curVectorTopKDealSize, lastSubBlockTailChunkNum,curChunkSize);
                            // DumpTensor(topKValueInTensor, 515, topKValueInTensor.GetSize());
                        }
                    }
                }
                break;
            case KEEP_TAIL_TYPE_TWO_BLOCKS:
                if(tileIdx == curSeqLenTileNum-1)
                {
                    if(subBlockIdx_ == 1)
                    {
                        //copy last
                        FillMaxValueFromTail(topKValueInTensor, curVectorTopKDealSize, lastSubBlockTailChunkNum,curChunkSize);
                    }
                    else
                    {
                        //copy second last
                        FillMaxValueFromTail(topKValueInTensor, curVectorTopKDealSize, secondLastSubBlockTailChunkNum,curChunkSize);
                    }
                }
                break;
            case KEEP_TAIL_TYPE_TWO_TILES:
                if(tileIdx == curSeqLenTileNum-1)
                {
                    if(subBlockIdx_ == 0)
                    {
                        //copy last
                        FillMaxValueFromTail(topKValueInTensor, curVectorTopKDealSize, lastSubBlockTailChunkNum,curChunkSize);                    
                    }

                }
                else if(tileIdx == curSeqLenTileNum-2)
                {
                    if(subBlockIdx_ == 1)
                    {
                        //copy second last
                        FillMaxValueFromTail(topKValueInTensor, curVectorTopKDealSize, secondLastSubBlockTailChunkNum,curChunkSize);                        
                    }
                }
                break;
        }

        // DumpTensor(topKValueInTensor, 666, topKValueInTensor.GetSize());
        topKValueInQueue_.EnQue(topKValueInTensor);
        topKValueInTensor = topKValueInQueue_.DeQue<half>();
        // 2.3 chunkTopK, 常规TopK耗尽模式
        TopKInExhaustionMode(curVectorTopKDealSize, curKScalar, topKValueInTensor, topkResultOffset, startIdxInCurWholeSeq, isApproxiTopK);
        topKValueInQueue_.FreeTensor(topKValueInTensor);
    }

    __aicore__ inline void DataMoveWithoutTopK(uint32_t curEffectLen, uint32_t curVectorTopKDealSize, uint32_t curChunkSize, uint32_t startSeqIdx, uint32_t startIdxInCurWholeSeq, uint32_t topkResultOffset, bool isApproxiTopK)
    {
        if ASCEND_IS_AIC {
            return;
        }
        LocalTensor<half> topKValueTensor = topKValueInnerOutQueue_.AllocTensor<half>();
        LocalTensor<int32_t> topKIdexTensor = topKIndexInnerOutQueue_.AllocTensor<int32_t>();
        // chunkTopK先做chunk max 生成index后搬出,常规TopK直接搬入UB, 生成idex后再搬出
        if (curChunkSize > MIN_CHUNK_SIZE) {
            LocalTensor<half> chunkReduceMaxValueInTensor = chunkReduceMaxValueInQueue_.AllocTensor<half>();
            ReduceMaxCustom(matmulGm_[startSeqIdx], chunkReduceMaxValueInTensor, topKValueTensor, curVectorTopKDealSize, curChunkSize);
            chunkReduceMaxValueInQueue_.EnQue(chunkReduceMaxValueInTensor);
            chunkReduceMaxValueInTensor = chunkReduceMaxValueInQueue_.DeQue<half>();
            chunkReduceMaxValueInQueue_.FreeTensor(chunkReduceMaxValueInTensor);
        } else {
            // curVectorTopKDealSize为分核后的处理长度，不会引入转换异常
            DataCopyExtParams copyInParams{1, static_cast<uint32_t>(curVectorTopKDealSize * sizeof(half)), 0, 0, 0};
            DataCopyPadExtParams<half> copyInPadParams{false, 0, 0, 0};
            DataCopyPad(topKValueTensor, matmulGm_[startSeqIdx], copyInParams, copyInPadParams);
        }
        ArithProgression(topKIdexTensor, static_cast<int32_t>(startIdxInCurWholeSeq), 1, static_cast<int32_t>(curVectorTopKDealSize));
        topKValueInnerOutQueue_.EnQue(topKValueTensor);
        topKIndexInnerOutQueue_.EnQue(topKIdexTensor);
        topKValueTensor = topKValueInnerOutQueue_.DeQue<half>();
        topKIdexTensor = topKIndexInnerOutQueue_.DeQue<int32_t>();
        DataCopyFromUBToGM(isApproxiTopK, curVectorTopKDealSize, topkResultOffset, topKValueTensor, topKIdexTensor);
        topKValueInnerOutQueue_.FreeTensor(topKValueTensor);
        topKIndexInnerOutQueue_.FreeTensor(topKIdexTensor);
    }

    __aicore__ inline void DataCopyFromUBToGM(bool isApproxiTopK, uint32_t copyLen, uint32_t topkResultOffset, const LocalTensor<half> &topKValueTensor, const LocalTensor<int32_t> &topKIdexTensor)
    {
        if ASCEND_IS_AIC {
            return;
        }
        // copyLen的最大值为TileN2=4k，不会导致转换越界
        if (isApproxiTopK) {
            DataCopyExtParams copyTopKIdxToOutParams{1, static_cast<uint32_t>(copyLen * sizeof(uint32_t)), 0, 0, 0};
            DataCopyPad(indicesGm_[topkResultOffset], topKIdexTensor, copyTopKIdxToOutParams);
        } else {
            DataCopyExtParams copyTopKValueOutParams{1, static_cast<uint32_t>(copyLen * sizeof(half)), 0, 0, 0};
            DataCopyPad(topkValueGm_[topkResultOffset], topKValueTensor, copyTopKValueOutParams);
            DataCopyExtParams copyTopKIdxToWorkSpaceParams{1, static_cast<uint32_t>(copyLen * sizeof(uint32_t)), 0, 0, 0};
            DataCopyPad(topkIdxGm_[topkResultOffset], topKIdexTensor, copyTopKIdxToWorkSpaceParams);
        }
    }

    __aicore__ inline void TopKInExhaustionMode(uint32_t curVectorTopKDealSize, uint32_t curKScalar, const LocalTensor<half> &topKValueInTensor, uint32_t topkResultOffset, uint32_t startIdxInCurWholeSeq, bool isApproxiTopK)
    {
        /* 重新计算使用的InnerSize，保证在Copy指令中，两个LocalTensor的起点位置都是32Bytes对齐的 */
        uint32_t effectiveInnerSize = curKScalar + (param_.topKInnerSize - curKScalar) / BLOCK_CUBE * BLOCK_CUBE;
        uint32_t curDealSeqLen = 0, curExtendSeqLen = 0, topKLoop = 0, ubStartIdxInTileSeq = 0;
        topKLoop = curVectorTopKDealSize < effectiveInnerSize ? 1 : DivCeil(curVectorTopKDealSize - effectiveInnerSize, effectiveInnerSize - curKScalar) + 1;
        LocalTensor<half> topKValueInnerOutTensor;
        LocalTensor<int32_t> topKIdexInnerOutTensor;
        for(uint32_t loopIdx = 0; loopIdx < topKLoop; loopIdx++) {
            LocalTensor<half> topKValueInnerInTensor = topKValueInnerInQueue_.AllocTensor<half>();
            LocalTensor<int32_t> topKIdexInnerInTensor = topKIndexInnerInQueue_.AllocTensor<int32_t>();
            if (loopIdx == 0) {
                // 第一次topk处理, 搬运长度为min(curVectorTopKDealSize, effectiveInnerSize)
                curDealSeqLen = curVectorTopKDealSize < effectiveInnerSize ? curVectorTopKDealSize: effectiveInnerSize; // 32长度对齐
                /* 由于做topk时作为inner的curDealSeqLen可能小于tiling中设置的inner，因此需要对无效数据部分进行最小值填充 */ 
                FillMinValue(curDealSeqLen, topKValueInnerInTensor);
                DataCopyInUB(curDealSeqLen, topKValueInnerInTensor, topKValueInTensor[ubStartIdxInTileSeq], MAX_FP16_PROCESS_NUM);
                // startIdxInCurWholeSeq和curDealSeqLen为分核后的单核序号及处理长度，不会引入转换异常
                ArithProgression(topKIdexInnerInTensor, static_cast<int32_t>(startIdxInCurWholeSeq), 1, static_cast<int32_t>(curDealSeqLen));
                ubStartIdxInTileSeq += effectiveInnerSize;
                startIdxInCurWholeSeq += effectiveInnerSize;
            } else {
                // 非第一次topk处理, 搬运长度为curKScalar, 补充长度为curExtendSeqLen
                topKValueInnerOutTensor = topKValueInnerOutQueue_.DeQue<half>();
                topKIdexInnerOutTensor = topKIndexInnerOutQueue_.DeQue<int32_t>();
                // 补充
                uint32_t maxExtendSeqLen = effectiveInnerSize - curKScalar;
                curExtendSeqLen = (curVectorTopKDealSize - ubStartIdxInTileSeq) >= maxExtendSeqLen ? maxExtendSeqLen : curVectorTopKDealSize - ubStartIdxInTileSeq;
                curDealSeqLen = curKScalar + curExtendSeqLen;
                /* 由于做topk时作为inner的curDealSeqLen可能小于tiling中设置的inner，因此需要对无效数据部分进行最小值填充 */ 
                FillMinValue(curDealSeqLen, topKValueInnerInTensor);
                DataCopyInUB(curDealSeqLen, topKValueInnerInTensor, topKValueInTensor[ubStartIdxInTileSeq - curKScalar], MAX_FP16_PROCESS_NUM);
                // startIdxInCurWholeSeq和curDealSeqLen为分核后的单核序号及处理长度，不会引入转换异常
                ArithProgression(topKIdexInnerInTensor, static_cast<int32_t>(startIdxInCurWholeSeq - curKScalar), 1, static_cast<int32_t>(curExtendSeqLen + curKScalar));
                // 获取新数据
                DataCopyInUB(curKScalar, topKValueInnerInTensor, topKValueInnerOutTensor, MAX_FP16_PROCESS_NUM);
                DataCopyInUB(curKScalar, topKIdexInnerInTensor, topKIdexInnerOutTensor, MAX_INT32_PROCESS_NUM);
                ubStartIdxInTileSeq += maxExtendSeqLen;
                startIdxInCurWholeSeq += maxExtendSeqLen;
                topKValueInnerOutQueue_.FreeTensor(topKValueInnerOutTensor);
                topKIndexInnerOutQueue_.FreeTensor(topKIdexInnerOutTensor);
            }
            topKValueInnerInQueue_.EnQue(topKValueInnerInTensor);
            topKIndexInnerInQueue_.EnQue(topKIdexInnerInTensor);
            topKValueInnerInTensor = topKValueInnerInQueue_.DeQue<half>();
            topKIdexInnerInTensor = topKIndexInnerInQueue_.DeQue<int32_t>();
            topKValueInnerOutTensor = topKValueInnerOutQueue_.AllocTensor<half>();
            topKIdexInnerOutTensor = topKIndexInnerOutQueue_.AllocTensor<int32_t>();
            TopKCustom(topKValueInnerOutTensor, topKIdexInnerOutTensor, topKValueInnerInTensor, topKIdexInnerInTensor, static_cast<int32_t>(curKScalar), tilingData_, curDealSeqLen);
            topKValueInnerInQueue_.FreeTensor(topKValueInnerInTensor);
            topKIndexInnerInQueue_.FreeTensor(topKIdexInnerInTensor);
            topKValueInnerOutQueue_.EnQue(topKValueInnerOutTensor);
            topKIndexInnerOutQueue_.EnQue(topKIdexInnerOutTensor);
        }
        topKValueInnerOutTensor = topKValueInnerOutQueue_.DeQue<half>();
        topKIdexInnerOutTensor = topKIndexInnerOutQueue_.DeQue<int32_t>();
        DataCopyFromUBToGM(isApproxiTopK, curKScalar, topkResultOffset, topKValueInnerOutTensor, topKIdexInnerOutTensor);
        topKValueInnerOutQueue_.FreeTensor(topKValueInnerOutTensor);
        topKIndexInnerOutQueue_.FreeTensor(topKIdexInnerOutTensor);
    }

    __aicore__ inline void FillMinValue(uint32_t copyLen, LocalTensor<half>& topKValueInTensor)
    {
        if ASCEND_IS_AIC {
            return;
        }
        uint32_t copyLenFloorAligned = copyLen / BLOCK_CUBE * BLOCK_CUBE;
        if (copyLenFloorAligned < param_.topKInnerSize) {
            Duplicate(topKValueInTensor[copyLenFloorAligned], static_cast<half>(MIN_HALF_VALUE), param_.topKInnerSize - copyLenFloorAligned);
        }
    }

    template <typename T>
    __aicore__ inline void DataCopyInUB(uint32_t curDealSeqLen, const LocalTensor<T> &topKToTensor, const LocalTensor<T> &topKFromTensor, uint32_t maxProcessNum)
    {
        if ASCEND_IS_AIC {
            return;
        }
        uint64_t mask;
        uint8_t repeatTimes;
        mask = curDealSeqLen > maxProcessNum ? maxProcessNum : curDealSeqLen;
        repeatTimes = curDealSeqLen / maxProcessNum;
        if (repeatTimes > 0) {
            Copy(topKToTensor, topKFromTensor, mask, repeatTimes, {1, 1, 8, 8});
        }
        if (curDealSeqLen % maxProcessNum != 0) {
            mask = curDealSeqLen % maxProcessNum;
            Copy(topKToTensor[repeatTimes * maxProcessNum], topKFromTensor[repeatTimes * maxProcessNum], mask, 1, {1, 1, 8, 8});
        }
    }

    __aicore__ inline void MergeTopKInGroup()
    {
        if ASCEND_IS_AIC {
            return;
        }
        LocalTensor<half> topKValueOutTensor;
        LocalTensor<int32_t> topKIndexOutTensor;
        uint32_t vectorTotalNum = param_.usedCoreNum * VECTOR_CUBE_RATIO;
        uint32_t iterTime = DivCeil(param_.batchN, vectorTotalNum), iterTail = param_.batchN % vectorTotalNum;
        for(uint32_t iterIdx = 0; iterIdx < iterTime; iterIdx++) {
            uint32_t curIterDealBatchN = (iterIdx + 1) * vectorTotalNum <= param_.batchN ? vectorTotalNum : iterTail;
            if (blockIdx_ * VECTOR_CUBE_RATIO + subBlockIdx_ >= curIterDealBatchN) {
                return;
            }
            uint32_t curVectorDealBatchNIdx = iterIdx * vectorTotalNum + blockIdx_ * VECTOR_CUBE_RATIO + subBlockIdx_;
            uint32_t curBatchIdx = curVectorDealBatchNIdx / param_.head;

            // 当前batch是否需要skip
            if (supportMask_) {
                bool batchMask = maskGm_.GetValue(curBatchIdx);
                //YF_LOG("curBatchIdx = %d, batchMask = %d\n", curBatchIdx, batchMask);
                if (!batchMask) {
                    // todo block table直接赋值给输出的Indices
                    SetBlockTableForIndices(curBatchIdx, curVectorDealBatchNIdx * param_.maxK);
                    //YF_LOG("curBatchIdx = %d SetBlockTableForIndices\n", curBatchIdx);
                    continue;
                }
            }

            uint32_t curKScalar = uint32_t(kGm_.GetValue(curBatchIdx)), curEffectLen = effectLenArr_[curBatchIdx], curSeqLen = uint32_t(seqLenGm_.GetValue(curBatchIdx));
            uint32_t curChunkSize = isChunkTopK_ ? uint32_t(chunkSizeGm_.GetValue(curBatchIdx)) : MIN_CHUNK_SIZE;
            if (curSeqLen == 0 || curKScalar == 0 || curEffectLen == 0) {
                continue; // seqLen=0或者kScalar==0或者chunkTopK effectLen==0直接跳过
            }
            uint32_t preTopkLenLast = ((curEffectLen / curChunkSize) % (param_.tileN2 / curChunkSize)) < curKScalar ? 
                                        (curEffectLen / curChunkSize) % (param_.tileN2 / curChunkSize) : curKScalar;
            uint32_t preTopKLen = curEffectLen / param_.tileN2 * curKScalar + preTopkLenLast;
            uint64_t mergeTopKOutGmOffset = curVectorDealBatchNIdx * param_.maxK;
            uint64_t topKResultGmOffset = curVectorDealBatchNIdx * DivCeil(maxEffectLen_, param_.tileN2) * param_.maxK;
            if (preTopKLen <= curKScalar) {
                LocalTensor<int32_t> topKInIndexTensor = topKIndexInnerOutQueue_.AllocTensor<int32_t>();
                DataCopyExtParams copyIndexParams{1, static_cast<uint32_t>(curKScalar * sizeof(int32_t)), 0, 0, 0};
                DataCopyPadExtParams<int32_t> copyIndexPadParams{false, 0, 0, 0};
                DataCopyPad(topKInIndexTensor, topkIdxGm_[topKResultGmOffset], copyIndexParams, copyIndexPadParams); // curKScalar的最大值为2048，不会引入转换越界
                PipeBarrier<PIPE_V>();
                if (!param_.supportOffload && (curChunkSize == 64 || curChunkSize == 128)) {
                    ReMappingBlockTableIndices(topKInIndexTensor, curBatchIdx, curKScalar, mergeTopKOutGmOffset);
                } else {
                    topKIndexInnerOutQueue_.EnQue(topKInIndexTensor);
                    topKInIndexTensor = topKIndexInnerOutQueue_.DeQue<int32_t>();
                    DataCopyExtParams copyTopKIdxParams{1, static_cast<uint32_t>(curKScalar * sizeof(int32_t)), 0, 0, 0}; // curKScalar的最大值为2048，不会引入转换越界
                    DataCopyPad(indicesGm_[mergeTopKOutGmOffset], topKInIndexTensor, copyTopKIdxParams);
                }
                topKIndexInnerOutQueue_.FreeTensor(topKInIndexTensor);
                continue;
            }
            uint32_t blockTile = preTopKLen > param_.topKInnerSize ? param_.topKInnerSize : preTopKLen;
            uint32_t topKBlockNum = DivCeil(preTopKLen - blockTile, blockTile - curKScalar) + 1;
            uint32_t blockTail = topKBlockNum > 1 ? (preTopKLen - ((topKBlockNum - 1) * (blockTile - curKScalar))): preTopKLen;
            MergeTopKInExhaustionMode(topKBlockNum, topKResultGmOffset, blockTail, blockTile, curKScalar, topKValueOutTensor, topKIndexOutTensor);
            topKValueOutTensor = topKValueInnerOutQueue_.DeQue<half>();
            topKIndexOutTensor = topKIndexInnerOutQueue_.DeQue<int32_t>();
            if (!param_.supportOffload && (curChunkSize == 64 || curChunkSize == 128)) {
                ReMappingBlockTableIndices(topKIndexOutTensor, curBatchIdx, curKScalar, mergeTopKOutGmOffset);
            } else {
                DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(curKScalar * sizeof(int32_t)), 0, 0, 0}; // {blockNum, blockLen, srcStride, dstStride, rsv(no concern)}
                DataCopyPad(indicesGm_[mergeTopKOutGmOffset], topKIndexOutTensor, copyOutParams); // curKScalar的最大值为2048，不会引入转换越界
            }
            
            topKValueInnerOutQueue_.FreeTensor(topKValueOutTensor);
            topKIndexInnerOutQueue_.FreeTensor(topKIndexOutTensor);
        }
    }

    __aicore__ inline void ReMappingBlockTableIndices(LocalTensor<int32_t>& topKIndexOutTensor, uint32_t curBatchIdx, uint32_t curKScalar, uint64_t outGmOffset) {
        if ASCEND_IS_AIC { return; }

        // topK小于32，不使用内部API排序
        if (curKScalar < 32) {
            useInnerSort = false;
        }

        if (useInnerSort) {
            CustomSort(topKIndexOutTensor, curKScalar);
            SelectBlockTableFromTopK(curBatchIdx, curKScalar, outGmOffset);
            return;
        }

        // 将 TopK 的“chunk索引”映射成 block_id，并写回 GM(indices)
        WriteBlockTableFromTopK(curBatchIdx, topKIndexOutTensor, curKScalar, outGmOffset);
    }

    __aicore__ inline void CustomSort(LocalTensor<int32_t>& topKIndexOutTensor, uint32_t len) {
        if ASCEND_IS_AIC { return; }
        if (len <= 1) { return; }

        LocalTensor<float> valueLocal = topKIndexSortInQueue_.AllocTensor<float>();
        Cast<float, int32_t>(valueLocal, topKIndexOutTensor, RoundMode::CAST_CEIL, static_cast<uint32_t>(len));//, static_cast<uint8_t>((len + 63) / 64), {1, 1, 8, 8});
        topKIndexSortInQueue_.EnQue(valueLocal);
        LocalTensor<uint32_t> indexLocal = topKIndexInnerInQueue_.AllocTensor<uint32_t>();
        topKIndexInnerInQueue_.EnQue(indexLocal);

        uint32_t repeatTimes = (len + 31) / 32;
        valueLocal = topKIndexSortInQueue_.DeQue<float>();
        indexLocal = topKIndexInnerInQueue_.DeQue<uint32_t>();
        LocalTensor<float> sortedLocal = topKIndexSortCalcQueue_.AllocTensor<float>();
        LocalTensor<float> concatTmpLocal = topKIndexSortTmpQueue_.AllocTensor<float>();
        LocalTensor<float> concatLocal;

        Concat(concatLocal, valueLocal, concatTmpLocal, repeatTimes);
        Sort<float, true>(sortedLocal, concatLocal, indexLocal, concatTmpLocal, repeatTimes);
        Extract(valueLocal, indexLocal, sortedLocal, repeatTimes);

        topKIndexSortTmpQueue_.FreeTensor(concatTmpLocal);
        topKIndexInnerInQueue_.FreeTensor(indexLocal);
        topKIndexSortCalcQueue_.FreeTensor(sortedLocal);

        topKIndexSortInQueue_.EnQue(valueLocal);
    }

    __aicore__ inline void SelectBlockTableFromTopK(uint32_t curBatchIdx, uint32_t curKScalar, uint64_t outGmOffset)
    {
        if ASCEND_IS_AIC {
            return;
        }

        // 降序
        LocalTensor<float> sortedTopKIndexTensor = topKIndexSortInQueue_.DeQue<float>();
        LocalTensor<int32_t> blockIdUb = topKIndexInnerInQueue_.AllocTensor<int32_t>();

        // 映射block table
        // 直接用标量方式在UB里读写
        __ubuf__ const float *in_ptr = reinterpret_cast<__ubuf__ const float *>(sortedTopKIndexTensor.GetPhyAddr());
        __ubuf__ int32_t *out_ptr = reinterpret_cast<__ubuf__ int32_t *>(blockIdUb.GetPhyAddr());

        LocalTensor<int32_t> tableBlockTensor = tableBlockBuf_.template Get<int32_t>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(param_.blockCount * sizeof(int32_t)), 0, 0};
        DataCopy(tableBlockTensor, keyBlockTableGm_[curBatchIdx * param_.blockCount], copyParams);

        for (uint32_t i = 0; i < curKScalar; ++i) {
            const int32_t idx = static_cast<int32_t>(in_ptr[i]);
            out_ptr[curKScalar - 1 - i] = isContinuousBatch_
                             ? tableBlockTensor.GetValue(static_cast<uint32_t>(idx))
                             : (idx + 1);  // 无block_table时，约定block_id = idx + 1（1-based）
        }

        // DumpTensor(blockIdUb, 859, 32);

        DataCopyExtParams cpOut{1, static_cast<uint32_t>(curKScalar * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(indicesGm_[outGmOffset], blockIdUb, cpOut);

        topKIndexInnerInQueue_.FreeTensor(blockIdUb);
        topKIndexSortInQueue_.FreeTensor(sortedTopKIndexTensor);
    }

    __aicore__ inline void SetBlockTableForIndices(uint32_t curBatchIdx, uint64_t outGmOffset) {
        // 找到当前batch对应的tableblock
        LocalTensor<int32_t> tableBlockTensor = tableBlockBuf_.template Get<int32_t>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(param_.blockCount * sizeof(int32_t)), 0, 0};
        DataCopy(tableBlockTensor, keyBlockTableGm_[curBatchIdx * param_.blockCount], copyParams);
        //DumpTensor(tableBlockTensor, 921, 64);
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);

        // 直接复制
        uint32_t copyLen = param_.blockCount < param_.maxK ? param_.blockCount : param_.maxK;
        DataCopyExtParams cpOut{1, static_cast<uint32_t>(copyLen * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(indicesGm_[outGmOffset], tableBlockTensor, cpOut);
    }

    __aicore__ inline void MergeTopKInExhaustionMode(uint32_t topKBlockNum, uint64_t topKResultGmOffset, uint32_t blockTail, uint32_t blockTile, uint32_t curKScalar, LocalTensor<half>& topKValueOutTensor, LocalTensor<int32_t>& topKIndexOutTensor)
    { 
        if ASCEND_IS_AIC {
            return;
        }
        for (uint32_t loop = 0; loop < topKBlockNum; loop++) {
            LocalTensor<half> topKInValueTensor = topKValueInnerInQueue_.AllocTensor<half>();
            LocalTensor<int32_t> topKInIndexTensor = topKIndexInnerInQueue_.AllocTensor<int32_t>();
            uint32_t copyLen = loop == topKBlockNum - 1 ? blockTail : blockTile;
            uint64_t curLoopTopKResultGmOffset = loop == 0 ? topKResultGmOffset : topKResultGmOffset + loop * (blockTile - curKScalar);
            GenerateTopKValueTensor(copyLen, blockTile, curLoopTopKResultGmOffset, curKScalar, topKInValueTensor);
            GenerateTopKIndexTensor(copyLen, blockTile, curLoopTopKResultGmOffset, curKScalar, topKInIndexTensor);
            if (loop > 0) {
                topKValueOutTensor = topKValueInnerOutQueue_.DeQue<half>();
                topKIndexOutTensor = topKIndexInnerOutQueue_.DeQue<int32_t>();
                topKInValueTensor = topKValueInnerInQueue_.DeQue<half>();
                topKInIndexTensor = topKIndexInnerInQueue_.DeQue<int32_t>();
                uint64_t valueMask = curKScalar > MAX_FP16_PROCESS_NUM ? MAX_FP16_PROCESS_NUM : curKScalar;
                uint8_t valueRepeatTimes = curKScalar / MAX_FP16_PROCESS_NUM;
                uint64_t indexMask = curKScalar > MAX_INT32_PROCESS_NUM ? MAX_INT32_PROCESS_NUM : curKScalar;
                uint8_t indexRepeatTimes = curKScalar / MAX_INT32_PROCESS_NUM;
                if (valueRepeatTimes > 0) {
                    Copy(topKInValueTensor, topKValueOutTensor, valueMask, valueRepeatTimes, {1, 1, 8, 8});
                }
                if (indexRepeatTimes > 0) {
                    Copy(topKInIndexTensor, topKIndexOutTensor, indexMask, indexRepeatTimes, {1, 1, 8, 8});
                }
                if (curKScalar % MAX_FP16_PROCESS_NUM != 0) {
                    Copy(topKInValueTensor[valueRepeatTimes * MAX_FP16_PROCESS_NUM], topKValueOutTensor[valueRepeatTimes * MAX_FP16_PROCESS_NUM], curKScalar % MAX_FP16_PROCESS_NUM, 1, {1, 1, 8, 8});
                }
                if (curKScalar % MAX_INT32_PROCESS_NUM != 0) {
                    Copy(topKInIndexTensor[indexRepeatTimes * MAX_INT32_PROCESS_NUM], topKIndexOutTensor[indexRepeatTimes * MAX_INT32_PROCESS_NUM], curKScalar % MAX_INT32_PROCESS_NUM, 1, {1, 1, 8, 8});
                }
                PipeBarrier<PIPE_V>();
                topKValueInnerInQueue_.EnQue(topKInValueTensor);
                topKIndexInnerInQueue_.EnQue(topKInIndexTensor);
                topKValueInnerOutQueue_.FreeTensor(topKValueOutTensor);
                topKIndexInnerOutQueue_.FreeTensor(topKIndexOutTensor);
            }
            topKInValueTensor = topKValueInnerInQueue_.DeQue<half>();
            topKInIndexTensor = topKIndexInnerInQueue_.DeQue<int32_t>();
            topKValueOutTensor = topKValueInnerOutQueue_.AllocTensor<half>();
            topKIndexOutTensor = topKIndexInnerOutQueue_.AllocTensor<int32_t>();
            TopKCustom(topKValueOutTensor, topKIndexOutTensor, topKInValueTensor, topKInIndexTensor, curKScalar, tilingData_, copyLen);
            topKValueInnerInQueue_.FreeTensor(topKInValueTensor);
            topKIndexInnerInQueue_.FreeTensor(topKInIndexTensor);
            topKValueInnerOutQueue_.EnQue(topKValueOutTensor);
            topKIndexInnerOutQueue_.EnQue(topKIndexOutTensor);
        }
    }

    __aicore__ inline void GenerateTopKValueTensor(uint32_t copyLen, uint32_t blockTile, uint64_t topkGmOffset, uint32_t curK, LocalTensor<half>& topKInValueTensor) 
    {
        if ASCEND_IS_AIC {
            return;
        }
        uint32_t copyLenAligned = copyLen / BLOCK_CUBE * BLOCK_CUBE; /* floor aligned for datacopy */
        if (copyLenAligned < param_.topKInnerSize) {
            Duplicate(topKInValueTensor[copyLenAligned], static_cast<half>(MIN_HALF_VALUE), param_.topKInnerSize - copyLenAligned);
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        uint32_t copyLenCeilAligned = DivCeil(copyLen * sizeof(half), BLOCK_CUBE) * BLOCK_CUBE / sizeof(half);
        // copyLen的最大值为TileN2=4k，不会引入转换越界
        DataCopyExtParams copyInValueParams{1, static_cast<uint32_t>(copyLen * sizeof(half)), 0, 0, 0};
        DataCopyPadExtParams<half> copyInValuePadParams{true, 0, static_cast<uint8_t>(copyLenCeilAligned - copyLen), static_cast<half>(MIN_HALF_VALUE)};
        DataCopyPad(topKInValueTensor, topkValueGm_[topkGmOffset], copyInValueParams, copyInValuePadParams);
        topKValueInnerInQueue_.EnQue(topKInValueTensor);
    }

    __aicore__ inline void GenerateTopKIndexTensor(uint32_t copyLen, uint32_t blockTile, uint64_t topkGmOffset, uint32_t curK, LocalTensor<int32_t>& topKInIndexTensor)
    {
        if ASCEND_IS_AIC {
            return;
        }
        // copyLen的最大值为TileN2=4k，不会引入转换越界
        DataCopyExtParams copyIndexParams{1, static_cast<uint32_t>(copyLen * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> copyIndexPadParams{false, 0, 0, 0};
        DataCopyPad(topKInIndexTensor, topkIdxGm_[topkGmOffset], copyIndexParams, copyIndexPadParams);
        topKIndexInnerInQueue_.EnQue(topKInIndexTensor);  
    }

    template <pipe_t pipe=PIPE_S>
    __aicore__ inline void SyncAivOnly(uint16_t eventId) {
        CrossCoreSetFlag<SYNC_MODE0, pipe>(eventId);
        CrossCoreWaitFlag(eventId);
    }

    template <pipe_t pipe=PIPE_S>
    __aicore__ inline void VectorNotifyCube(uint16_t aivOnlyEventId, uint16_t aiv2AicEventId) {
        SyncAivOnly<pipe>(aivOnlyEventId);
        CrossCoreSetFlag<SYNC_MODE2, pipe>(aiv2AicEventId);
    }

    __aicore__ inline void CubeWaitVector(uint16_t aiv2AicEventId) {
        CrossCoreWaitFlag(aiv2AicEventId);
    }

    template <pipe_t pipe=PIPE_S>
    __aicore__ inline void CubeNotifyVector(uint16_t aicOnlyEventId, uint16_t aic2AivEventId) {
        CrossCoreSetFlag<SYNC_MODE2, pipe>(aic2AivEventId);
    }

    __aicore__ inline void VectorWaitCube(uint16_t aic2AivEventId) {
        CrossCoreWaitFlag(aic2AivEventId);
    }

protected:
    uint64_t innerSplitLoopTimes_ = 0;
    bool innerSplitGMPingPongFlag_ = 0;
    uint32_t topKBlockNum = 1;
    uint32_t batchSeqTileN[MAX_BATCH_SIZE] = {0};
    uint32_t effectLenArr_[MAX_BATCH_SIZE] = {0};
    bool isChunkTopK_ = true;
    uint32_t maxEffectLen_ = 0;
    uint32_t minChunkSize_ = MAX_CHUNK_SIZE;
    uint32_t tileSeqLenSize_ = 0;
    bool isContinuousBatch_ = true;
    bool supportMask_ = true;

    static constexpr uint32_t BLOCK_CUBE = 32;
    static constexpr uint64_t SYNC_MODE0 = 0;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_AIV_ONLY_ALL_FLAG = 0;
    static constexpr uint64_t SYNC_AIC_ONLY_ALL_FLAG = 1;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 2;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 4;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG2 = 6;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG2 = 8;
    int32_t eventIDFIX_MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_MTE2));

    GlobalTensor<uint8_t> queryGm_;
    GlobalTensor<uint8_t> keyCompressedGm_;
    GlobalTensor<uint8_t> keyCompressedRopeGm_;
    GlobalTensor<int32_t> kGm_;
    GlobalTensor<int32_t> seqLenGm_;
    GlobalTensor<int32_t> chunkSizeGm_;
    GlobalTensor<int32_t> keyBlockTableGm_;
    GlobalTensor<bool> maskGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int4b_t> unpackGm_;
    GlobalTensor<int4b_t> kRopeUnpackGm_;
    GlobalTensor<int4b_t> qUnpackGm_;
    GlobalTensor<half> matmulGm_;
    GlobalTensor<half> topkValueGm_;
    GlobalTensor<int32_t> topkIdxGm_;

    TPipe *pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> keyCompressedInBuf_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> keyUnpackedOutBuf_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER_NUM> chunkReduceMaxValueInQueue_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER_NUM> topKValueInQueue_;
    TQue<TPosition::VECIN, 1> topKValueInnerInQueue_;
    TQue<TPosition::VECIN, 1> topKIndexInnerInQueue_;
    TQue<TPosition::VECOUT, 1> topKValueInnerOutQueue_;
    TQue<TPosition::VECOUT, 1> topKIndexInnerOutQueue_;
    TBuf<TPosition::VECCALC> constBuf_;
    TBuf<TPosition::VECCALC> selectBuf_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER_NUM> queryCompressedInQueue_;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER_NUM> queryUnpackedOutQueue_;
    TBuf<TPosition::VECCALC> qReduceSumLastRowBuf_;
    TBuf<TPosition::VECCALC> qReduceSumBuf_;

    TQue<TPosition::VECOUT, 1> topKIndexSortInQueue_;
    TQue<TPosition::VECIN, 1> topKIndexSortTmpQueue_;
    TQue<TPosition::VECIN, 1> topKIndexSortCalcQueue_;
    TBuf<TPosition::VECIN> tableBlockBuf_;

    int32_t layerIdScalar_;
    TilingParam param_;
    TopkTiling topkTiling_;
    HammingDistTopKTilingData tilingData_;

    using AMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, true>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    // notice: the TPos of ctype must be ub given by mm api when iterate<false>,
    // but actually we can move data to gm then to ub.
    using CMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, half>;
    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, MM_CFG_NO_PRELOAD> mm_;
    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, MM_CFG_NO_PRELOAD> mmRope_;

    uint32_t blockIdx_;
    uint32_t subBlockIdx_;
    uint64_t mmOffsetA_;
    uint64_t mmOffsetB_;
    uint64_t mmOffsetARope_;
    uint64_t mmOffsetBRope_;
    uint64_t mmOffsetC_;

    uint64_t index_;
    // 是否使用算子API进行排序
    bool useInnerSort = true;
 
    __aicore__ inline void InitParams()
    {
        blockIdx_ = GetBlockIdx();
        subBlockIdx_ = GetSubBlockIdx();
        mmOffsetA_ = RESET_NUM;
        mmOffsetB_ = RESET_NUM;
        mmOffsetARope_ = RESET_NUM;
        mmOffsetBRope_ = RESET_NUM;
        mmOffsetC_ = RESET_NUM;
        index_ = RESET_NUM;
    }

    __aicore__ inline void InitTilingParams(const TCubeTiling &tiling, const TCubeTiling &tilingRope, const TopkTiling &topkTiling, const HammingDistTopKTilingParams &tilingParam)
    {
        // tiling data for select
        param_.batch = tilingParam.batch;
        param_.maxK = tilingParam.maxK;
        param_.batchN = tilingParam.batchN;
        param_.tileN1 = tilingParam.tileN1;
        param_.tileN2 = tilingParam.tileN2;
        param_.dimension = tilingParam.dimension;
        param_.nope_dimension = tilingParam.nope_dimension;
        param_.rope_dimension = tilingParam.rope_dimension;
        param_.head = tilingParam.head;
        param_.layerSize = tilingParam.layerSize;
        param_.layerSizeRope = tilingParam.layerSizeRope;
        param_.qHead = tilingParam.qHead;
        param_.headGroupNum = tilingParam.headGroupNum;

        // support key rope
        param_.supportKeyRope = tilingParam.supportKeyRope > 0;

        // tiling data for matmul
        param_.usedCoreNum =  tilingParam.usedCoreNum;
        param_.M = tiling.M;
        param_.N = tiling.N;
        param_.ka = tiling.Ka;  // dimension
        param_.kb = tiling.Kb;  // dimension
        if (param_.supportKeyRope) {
            param_.rope_ka = tilingRope.Ka;  // dimension
            param_.rope_kb = tilingRope.Kb;  // dimension
        }
        // YF_LOG("param_.M=%d, param_.N=%d, param_.ka=%d, param_.kb=%d, param_.rope_ka=%d, param_.rope_kb=%d\n", param_.M, param_.N, param_.ka, param_.kb, param_.rope_ka, param_.rope_kb);
        param_.matmulResultSize = tilingParam.matmulResultSize;  // ceil(seq_len, chunk_size) * chunk_size = 8320

        // tiling data for topk
        param_.maxSeqLen = tilingParam.maxSeqLen;
        param_.sink = tilingParam.sink;
        param_.recent = tilingParam.recent;
        param_.blockCount = tilingParam.blockCount;
        param_.topKInnerSize = tilingParam.topKInnerSize;
        param_.topKValueSize = tilingParam.topKValueSize;
        param_.topKIdexSize = tilingParam.topKIdexSize;
        topkTiling_ = topkTiling;
        tileSeqLenSize_ = VECTOR_CUBE_RATIO * param_.tileN2;

        // support offload
        param_.supportOffload = tilingParam.supportOffload > 0;
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR query, GM_ADDR keyCompressed, GM_ADDR keyCompressedRope, GM_ADDR k, GM_ADDR seqLen,
        GM_ADDR chunkSize, GM_ADDR keyBlockTable, GM_ADDR mask, GM_ADDR indices, GM_ADDR workSpace)
    {
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(query));
        keyCompressedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(keyCompressed));
        keyCompressedRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(keyCompressedRope));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(k));
        seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLen));
        chunkSizeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunkSize));
        keyBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(keyBlockTable));
        maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(mask));
        supportMask_ = maskGm_.GetPhyAddr() != nullptr;
        isChunkTopK_ = chunkSizeGm_.GetPhyAddr() != nullptr;
        isContinuousBatch_ = keyBlockTableGm_.GetPhyAddr() != nullptr;
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));

        uint32_t unpackResultWorkspaceOffset = 0;
        uint32_t kNopeUnpackResultWorkspaceOffset = unpackResultWorkspaceOffset + param_.layerSize * 8 /2;
        uint32_t matmulResultWorkspaceOffset = kNopeUnpackResultWorkspaceOffset + param_.layerSizeRope * 8 /2;
        uint32_t topkValueResultWorkspaceOffset = matmulResultWorkspaceOffset + param_.matmulResultSize * 2;
        uint32_t topkIdxResultWorkspaceOffset = topkValueResultWorkspaceOffset + param_.topKValueSize * 2;
        uint32_t queryUnpackResultWorkspaceOffset = topkIdxResultWorkspaceOffset + param_.topKIdexSize * 4;

        unpackGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(workSpace + unpackResultWorkspaceOffset));
        kRopeUnpackGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(workSpace + kNopeUnpackResultWorkspaceOffset));
        matmulGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workSpace + matmulResultWorkspaceOffset));
        topkValueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workSpace + topkValueResultWorkspaceOffset));
        topkIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workSpace + topkIdxResultWorkspaceOffset));
        qUnpackGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(workSpace + queryUnpackResultWorkspaceOffset));
    }

    __aicore__ inline void InitLocalBuffersForUnpackQ()
    {
        pipe_->InitBuffer(constBuf_,  param_.dimension * sizeof(half));
        pipe_->InitBuffer(selectBuf_, param_.headGroupNum * param_.dimension * sizeof(half));
        pipe_->InitBuffer(queryCompressedInQueue_, DOUBLE_BUFFER_NUM, param_.headGroupNum * param_.dimension * sizeof(int8_t) / COMPRESS_RATE);
        pipe_->InitBuffer(queryUnpackedOutQueue_, DOUBLE_BUFFER_NUM, param_.headGroupNum * param_.dimension * sizeof(int8_t) / INT4B_TYPE_SIZE_DIV_RATE);
        pipe_->InitBuffer(qReduceSumLastRowBuf_, param_.dimension * sizeof(half));
        pipe_->InitBuffer(qReduceSumBuf_, param_.headGroupNum * param_.dimension * sizeof(half));
    }

    __aicore__ inline void InitLocalBuffersForUnpackK()
    {
        pipe_->InitBuffer(keyCompressedInBuf_, DOUBLE_BUFFER_NUM, param_.tileN1 * param_.dimension * sizeof(int8_t) / COMPRESS_RATE);
        pipe_->InitBuffer(keyUnpackedOutBuf_, DOUBLE_BUFFER_NUM, param_.tileN1 * param_.dimension * sizeof(int8_t) / INT4B_TYPE_SIZE_DIV_RATE);
        pipe_->InitBuffer(constBuf_, param_.dimension * sizeof(half));
        pipe_->InitBuffer(selectBuf_, MAX_FP16_PROCESS_NUM * MAX_SELECT_AND_CAST_COUNT * sizeof(half));  // 一定要在最后
    }

    __aicore__ inline void InitLocalBuffersForTopK()
    {
        if (minChunkSize_ > MIN_CHUNK_SIZE) {
            pipe_->InitBuffer(chunkReduceMaxValueInQueue_, DOUBLE_BUFFER_NUM, param_.tileN2 * DivCeil(MAX_CHUNK_SIZE, minChunkSize_) * sizeof(half));
        }
        pipe_->InitBuffer(topKValueInQueue_, DOUBLE_BUFFER_NUM, DivCeil(param_.tileN2, minChunkSize_) * sizeof(half));
        pipe_->InitBuffer(topKValueInnerInQueue_, 1, param_.topKInnerSize * sizeof(half));
        pipe_->InitBuffer(topKIndexInnerInQueue_, 1, param_.topKInnerSize * sizeof(uint32_t));
        pipe_->InitBuffer(topKValueInnerOutQueue_, 1, param_.maxK * sizeof(half));
        pipe_->InitBuffer(topKIndexInnerOutQueue_, 1, param_.maxK * sizeof(uint32_t));
    }

    __aicore__ inline void InitLocalBuffersForTopKSort()
    {
        pipe_->InitBuffer(topKIndexSortInQueue_, 1, param_.maxK * sizeof(uint32_t));
        pipe_->InitBuffer(topKIndexSortTmpQueue_, 1, 3 * param_.maxK * sizeof(uint32_t));
        pipe_->InitBuffer(topKIndexSortCalcQueue_, 1, 3 * param_.maxK * sizeof(uint32_t));
    }

    __aicore__ inline void InitLocalBuffersForTableBlock()
    {
        pipe_->InitBuffer(tableBlockBuf_, param_.blockCount * sizeof(int32_t));
    }

    __aicore__ inline void WriteBlockTableFromTopK(
        uint32_t curBatchIdx,
        LocalTensor<int32_t>& topKIndexUb,
        uint32_t curKScalar,
        uint64_t outGmOffset)
    {
        if ASCEND_IS_AIC { return; }

        LocalTensor<int32_t> blockIdUb = topKIndexInnerInQueue_.AllocTensor<int32_t>();
        LocalTensor<int32_t> tableBlockTensor = tableBlockBuf_.template Get<int32_t>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(param_.blockCount * sizeof(int32_t)), 0, 0};
        DataCopy(tableBlockTensor, keyBlockTableGm_[curBatchIdx * param_.blockCount], copyParams);

        ::AscendC::WriteBlockTableFromTopK(curBatchIdx, topKIndexUb, blockIdUb, curKScalar, outGmOffset, 
            tableBlockTensor, indicesGm_, isContinuousBatch_, param_.blockCount);

        topKIndexInnerInQueue_.FreeTensor(blockIdUb);
    }
};
} // namespace AscendC
#endif // HAMMING_DIST_TOP_K_SPLIT_S_H