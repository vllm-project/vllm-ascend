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
 * \file hamming_dist_top_k_parallel.h
 * \brief
 */
#ifndef HAMMING_DIST_TOP_K_PARALLEL_H
#define HAMMING_DIST_TOP_K_PARALLEL_H

#include "hamming_dist_top_k_base.h"

namespace AscendC {
class HammingDistTopKParallelKernel {
public:
    __aicore__ inline HammingDistTopKParallelKernel() {}
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR keyCompressed, GM_ADDR keyCompressedRope, GM_ADDR k,
                                GM_ADDR seqLen, GM_ADDR chunkSize, GM_ADDR keyBlockTable, GM_ADDR mask,
                                GM_ADDR indices, GM_ADDR workSpace,
                                const HammingDistTopKTilingData &tilingData, TPipe *pipe)
    {
        const TCubeTiling &tiling = tilingData.matmulTiling;
        const TCubeTiling &tilingRope = tilingData.matmulTilingRope;
        const TopkTiling &topkTiling = tilingData.topkTiling;
        const HammingDistTopKTilingParams &tilingParam = tilingData.params;
        pipe_ = pipe;
        tilingData_ = tilingData;
        InitTilingParams(tiling, tilingRope, topkTiling, tilingParam);
        InitParams();
        InitGlobalBuffers(query, keyCompressed, keyCompressedRope, k, seqLen, chunkSize, keyBlockTable, mask, indices, workSpace);

        continFlag_ = keyBlockTableGm_.GetPhyAddr() != nullptr;
        mm_.SetSubBlockIdx(0);
        mm_.Init(&tiling, pipe_);
        if (param_.supportKeyRope) {
            mmRope_.SetSubBlockIdx(0);
            mmRope_.Init(&tilingRope, pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIndex = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            blockIndex = blockIndex / 2;
        }
        if (blockIndex >= param_.usedCoreNum) {
            return;
        }
        uint32_t batchNumPerLoop = 1;
        uint32_t curCoreLoop = (curCoreBatch_ + batchNumPerLoop - 1) / batchNumPerLoop;
        uint32_t tailBatchNumPerLoop = curCoreBatch_ - (curCoreLoop - 1) * batchNumPerLoop;
        uint32_t computeBatch;
        uint32_t nextLoopComputeBatch;
        
        if ASCEND_IS_AIV {
            pipe_->Reset();
        }

        // YF_LOG("ldeng 71, curCoreLoop=%d\n", curCoreLoop);

        for (uint32_t loopIdx = 0; loopIdx < curCoreLoop; loopIdx++) {
            //YF_LOG("ldeng 74\n");
            ComputeUnpackMM(loopIdx * batchNumPerLoop, batchNumPerLoop, loopIdx % BATCH_PING_PONG_NUM);
        }
        if ASCEND_IS_AIV {
            pipe_barrier(PIPE_ALL);
            pipe_->Reset();
            InitLocalBuffersForTopK();
        }

        //YF_LOG("ldeng 80\n");
        for (uint32_t loopIdx = 0; loopIdx < curCoreLoop; loopIdx++) {
            if ASCEND_IS_AIV {
                VectorWaitCube(SYNC_AIC_AIV_FLAG2 + loopIdx % BATCH_PING_PONG_NUM);
                if (GetSubBlockIdx() == loopIdx % 2) {
                    //YF_LOG("ldeng 83\n");
                    ComputeTopK(loopIdx, loopIdx % BATCH_PING_PONG_NUM);
                }
            }
        }
    }

protected:
    __aicore__ inline int32_t GetCurSeqLen(const GlobalTensor<int32_t> &seqLenGm_,
        const GlobalTensor<int32_t> &chunkSizeGm_, uint32_t realCurBatch)
    {
        int32_t curSeqLen = seqLenGm_.GetValue(realCurBatch);
        int32_t curChunkSize = 1;
        if (chunkSizeGm_.GetPhyAddr() != nullptr) {
            curChunkSize = chunkSizeGm_.GetValue(realCurBatch);
        }
        if (curChunkSize == 1 || curChunkSize == 8 || curChunkSize == 16) {
            if (curSeqLen <= 32) { // seqLen <= 32时，跳过hammingDistTopK，在kvSelect算子把所有的数据都选上
                curSeqLen = 0;
            } else {
                curSeqLen = curSeqLen - Min(curSeqLen, static_cast<int32_t>(curSeqLen % curChunkSize + 16)); /* 16: 用于计算lastlen长度 */
            }
        } else if (curChunkSize == 64) {
            curSeqLen = ((curSeqLen + 63) / 64) * 64;
        } else if(curChunkSize == 128){
            curSeqLen = ((curSeqLen + 127)/128)*128;
        } 
        else {
            curSeqLen = 0;
        }
        return curSeqLen;
    }

    __aicore__ inline void ComputeUnpackMM(uint32_t batchIdx, uint32_t batchNum, uint32_t pingPongFlag)
    {
        uint32_t maxLoopNum = (batchNum >> 1) << 1;  // 向下取偶数，等价于(batchNum / 2) * 2
        for (uint32_t j = 0; j < maxLoopNum; j++) {
            uint32_t curReducedBatch = (curCoreBatchStartIdx_ + batchIdx + j);
            uint32_t realCurBatch = curReducedBatch / param_.head;
            // 当前batch不计算, 直接跳过
            if (supportMask_) {
                bool batchMask = maskGm_.GetValue(realCurBatch);
                //YF_LOG("realCurBatch = %d, batchMask = %d\n", realCurBatch, batchMask);
                if (!batchMask) {
                    return;
                }
            }
            int32_t curSeqLen = GetCurSeqLen(seqLenGm_, chunkSizeGm_, realCurBatch);
            int32_t curK = kGm_.GetValue(realCurBatch);
            if (curK == 0 || curSeqLen == 0) {
                continue; /* 当前batch对应的seqLen或者k等于0时， 跳过该batch的unpack计算 */
            }
            if ASCEND_IS_AIV {
                if (GetSubBlockIdx() == (j % SUB_BLOCK_NUM)) {
                    UnpackOneBatch(curSeqLen, curReducedBatch, 0, true);
                }
                VectorNotifyCube<PIPE_MTE3>(SYNC_AIV_AIC_FLAG + pingPongFlag);
            }
            if ASCEND_IS_AIC {
                CubeWaitVector(SYNC_AIV_AIC_FLAG + pingPongFlag);
                ComputeMM(batchIdx + j, curSeqLen);
            }
        }
        if (batchNum > maxLoopNum) {
            ComputeUnpackMMForLastBatch(batchIdx, pingPongFlag);
        }
        if ASCEND_IS_AIC {
            CubeNotifyVector<PIPE_FIX>(SYNC_AIC_AIV_FLAG2 + pingPongFlag);
        }
    }

    __aicore__ inline void ComputeUnpackMMForLastBatch(uint32_t batchIdx, uint32_t pingPongFlag)
    {
        uint32_t curReducedBatch = (curCoreBatchStartIdx_ + batchIdx);
        uint32_t realCurBatch = curReducedBatch / param_.head;
        // 当前batch不计算, 直接跳过
        if (supportMask_) {
            bool batchMask = maskGm_.GetValue(realCurBatch);
            //YF_LOG("realCurBatch = %d, batchMask = %d\n", realCurBatch, batchMask);
            if (!batchMask) {
                return;
            }
        }
        int32_t curSeqLen = GetCurSeqLen(seqLenGm_, chunkSizeGm_, realCurBatch);
        int32_t curK = kGm_.GetValue(realCurBatch);
        if (curK == 0 || curSeqLen == 0) {
            return; /* 当前batch对应的seqLen或者k等于0时， 跳过该batch的unpack计算 */
        }
        if ASCEND_IS_AIV {
            uint32_t curBlockCount = matmul::CeilDiv(curSeqLen, param_.tileN1);
            uint32_t subBlockSeqOffset = 0;
            uint32_t subBlockSeqLen = 0;
            if (GetSubBlockIdx() == 0) {
                subBlockSeqLen = continFlag_ ? matmul::CeilDiv(curBlockCount, SUB_BLOCK_NUM) * param_.tileN1 :
                    matmul::CeilDiv(curSeqLen, SUB_BLOCK_NUM);
            } else if (GetSubBlockIdx() == 1) {
                subBlockSeqLen = continFlag_ ? (curBlockCount / SUB_BLOCK_NUM) * param_.tileN1 : curSeqLen / SUB_BLOCK_NUM;
                if (subBlockSeqLen <= 0) {
                    VectorNotifyCube<PIPE_MTE3>(SYNC_AIV_AIC_FLAG + pingPongFlag);
                    return;
                }
                subBlockSeqOffset += continFlag_ ? matmul::CeilDiv(curBlockCount, SUB_BLOCK_NUM) * param_.tileN1 :
                    matmul::CeilDiv(curSeqLen, SUB_BLOCK_NUM);
            }

            UnpackOneBatch(subBlockSeqLen, curReducedBatch, subBlockSeqOffset, GetSubBlockIdx() == 0);
            VectorNotifyCube<PIPE_MTE3>(SYNC_AIV_AIC_FLAG + pingPongFlag);
        }
        if ASCEND_IS_AIC {
            CubeWaitVector(SYNC_AIV_AIC_FLAG + pingPongFlag);
            ComputeMM(batchIdx, curSeqLen);
        }
    }

    __aicore__ inline void UnpackOneBatch(uint32_t sequenceLen, uint32_t batchIdx, uint32_t subBlockSeqOffset, bool unpackQuery)
    {
        if ASCEND_IS_AIC {
            return;
        }
        InitLocalBuffersForUnpackQuery();
        // unpack query
        UnpackQuery(batchIdx);
        pipe_->Reset();
        InitLocalBuffersForUnpackKey();
        // unpack key
        UnpackKey(sequenceLen, batchIdx, subBlockSeqOffset, false);
        if (param_.supportKeyRope) {
            UnpackKey(sequenceLen, batchIdx, subBlockSeqOffset, true);
        }
        pipe_->Reset();
    }

    __aicore__ inline void UnpackQuery(uint32_t batchIdx) {

        LocalTensor<half> constTensor = constBuf_.template Get<half>();
        LocalTensor<half> selectTensor = selectBuf_. template Get<half>();
        LocalTensor<half> qReduceSumTensor = qReduceSumBuf_. template Get<half>();
        LocalTensor<half> qReduceSumLastRowTensor = qReduceSumLastRowBuf_. template Get<half>();

        Duplicate<half>(constTensor, 1, param_.dimension);

        uint32_t compressedDimension = param_.dimension / COMPRESS_RATE;
        uint64_t queryGmOffset = batchIdx * param_.headGroupNum * compressedDimension;
        LocalTensor<uint8_t> queryCompressed = queryCompressedInQueue_.AllocTensor<uint8_t>();
        DataCopyExtParams queryCopyInParams{1, param_.headGroupNum * static_cast<uint32_t>(compressedDimension), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> queryCopyInPadParams{false, 0, 0, 0};
        DataCopyPad(queryCompressed, queryGm_[queryGmOffset], queryCopyInParams, queryCopyInPadParams);
        //DumpTensor(queryCompressed, 210, queryCompressed.GetSize());
        queryCompressedInQueue_.EnQue(queryCompressed);
        queryCompressed = queryCompressedInQueue_.DeQue<uint8_t>();
        // half 每次迭代128个元素
        uint32_t repeatedTimes = matmul::CeilDiv(param_.headGroupNum * param_.dimension, MAX_FP16_PROCESS_NUM);
        SelectCustom<half>(selectTensor, queryCompressed, constTensor, static_cast<uint8_t>(repeatedTimes));
        PipeBarrier<PIPE_V>();
        if (param_.supportKeyRope) {
            uint32_t pad_dim = param_.rope_dimension / 2;
            uint32_t valid_dim = param_.nope_dimension + pad_dim;
            // YF_LOG("param_.dimension=%d, valid_dim=%d, pad_dim=%d\n", param_.dimension, valid_dim, pad_dim);
            for (uint32_t i = 0; i < param_.headGroupNum; i++) {
                // DumpTensor(selectTensor[i * param_.dimension], 224, 672);
                Duplicate<half>(selectTensor[i * param_.dimension + valid_dim], 0, pad_dim);
                // DumpTensor(selectTensor[i * param_.dimension], 226, 672);
            }
        }
        queryCompressedInQueue_.FreeTensor(queryCompressed);

        uint64_t qMask = MAX_FP16_PROCESS_NUM;
        uint32_t repeatTimes = matmul::CeilDiv(param_.dimension, MAX_FP16_PROCESS_NUM);
        LocalTensor<half> qHashTensor = selectTensor;
        //DumpTensor(qHashTensor, 223, param_.dimension);
        if (param_.headGroupNum > 1) {
            static constexpr AscendC::CumSumConfig cumSumConfig{false, false, true};
            const AscendC::CumSumInfo cumSumInfo{param_.headGroupNum, param_.dimension};
            AscendC::CumSum<half, cumSumConfig>(qReduceSumTensor, qReduceSumLastRowTensor, selectTensor, cumSumInfo);
            PipeBarrier<PIPE_V>();

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
        //DumpTensor(qHashTensor, 240, param_.dimension);
        LocalTensor<int4b_t> queryUnpacked = queryUnpackedOutQueue_.AllocTensor<int4b_t>();
        Cast<int4b_t, half>(queryUnpacked, qHashTensor, RoundMode::CAST_CEIL, qMask, repeatTimes, {1, 1, 2, 8});
        queryUnpackedOutQueue_.EnQue(queryUnpacked);
        queryUnpacked = queryUnpackedOutQueue_.DeQue<int4b_t>();
        uint64_t unpackQGmOffset = queryGmOffset * 8 / param_.headGroupNum;
        DataCopyExtParams copyQOutParams{1, static_cast<uint32_t>(param_.dimension / 2), 0, 0, 0}; /* 2: 1 / size of int4b_t */
        DataCopyPad(qUnpackGm_[unpackQGmOffset], queryUnpacked, copyQOutParams);
        queryUnpackedOutQueue_.FreeTensor(queryUnpacked);
    }

    __aicore__ inline void UnpackKey(uint32_t sequenceLen, 
                                    uint32_t batchIdx,
                                    uint32_t subBlockSeqOffset,
                                    bool isKeyRope) {
        uint32_t realBatchIdx = batchIdx / param_.head; /* batchIdx without headNum */
        uint32_t headIdx = batchIdx % param_.head;

        uint32_t dimension = isKeyRope ? param_.rope_dimension : param_.nope_dimension;
        GlobalTensor<uint8_t> keyGm = isKeyRope ? keyRopeGm_ : keyGm_;

        uint32_t sequenceBlockNum = matmul::CeilDiv(sequenceLen, param_.tileN1);
        uint32_t tailN1 = sequenceLen - (sequenceBlockNum - 1) * param_.tileN1;
        uint32_t compressedDimension = dimension / COMPRESS_RATE;

        LocalTensor<half> constTensor = constBuf_.template Get<half>();
        LocalTensor<half> selectTensor = selectBuf_.template Get<half>();

        Duplicate<half>(constTensor, 1, dimension);
        // key select 迭代次数
        uint32_t selectRepeatedTimes = computeSelectRepeatedTimes(dimension);
        for (uint32_t j = 0; j < sequenceBlockNum; j++) {
            LocalTensor<uint8_t> keyCompressed = keyCompressedInQueue_.AllocTensor<uint8_t>();
            uint32_t copySeqLen = j == sequenceBlockNum - 1 ? tailN1 : param_.tileN1;
            DataCopyExtParams copyInParams{1, static_cast<uint32_t>(copySeqLen * compressedDimension), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> copyInPadParams{false, 0, 0, 0};
            uint64_t keyGmOffset = batchIdx * param_.maxSeqLen * compressedDimension +
                j * param_.tileN1 * compressedDimension + subBlockSeqOffset * compressedDimension;
            if (!continFlag_) {
                DataCopyPad(keyCompressed, keyGm[keyGmOffset], copyInParams, copyInPadParams);
            } else {
                int32_t blockTableVal = keyBlockTableGm_.GetValue(realBatchIdx * param_.blockCount + j +
                                                                  subBlockSeqOffset / param_.tileN1);
                uint64_t keyGmOffsetConti = (headIdx + static_cast<uint64_t>(blockTableVal) * param_.head) *
                    param_.tileN1 * compressedDimension;
                DataCopyPad(keyCompressed, keyGm[keyGmOffsetConti], copyInParams, copyInPadParams);
            }
            keyCompressedInQueue_.EnQue(keyCompressed);
            keyCompressed = keyCompressedInQueue_.DeQue<uint8_t>();
            //DumpTensor(keyCompressed, 285, static_cast<uint32_t>(copySeqLen * compressedDimension));
            //YF_LOG("sequenceBlockNum = %d j = %d copySeqLen = %d selectRepeatedTimes = %d subBlockSeqOffset = %d sequenceLen = %d\n", sequenceBlockNum, j, copySeqLen, selectRepeatedTimes, subBlockSeqOffset, sequenceLen);

            if (selectRepeatedTimes > MAX_SELECT_REPEATED_TIMES) {
                UnpackKeyCompressedWithBigDim(selectRepeatedTimes, selectTensor, constTensor, keyCompressed, keyGmOffset);
            } else {
                UnpackKeyCompressedWithLittleDim(selectTensor, constTensor, keyCompressed, dimension, keyGmOffset, copySeqLen);
            }
            keyCompressedInQueue_.FreeTensor(keyCompressed);
        }
    }

    __aicore__ inline void UnpackKeyCompressedWithBigDim(uint32_t selectRepeatedTimes,
                                                         LocalTensor<half> &selectTensor,
                                                         LocalTensor<half> &constTensor,
                                                         LocalTensor<uint8_t> &keyCompressed,
                                                         uint64_t keyGmOffset) {
        uint32_t keySelectCycleCount = (selectRepeatedTimes + MAX_SELECT_REPEATED_TIMES - 1) / MAX_SELECT_REPEATED_TIMES;
        uint32_t tailRepeateadTimes = selectRepeatedTimes % MAX_SELECT_REPEATED_TIMES;
        LocalTensor<int4b_t> keyUnpacked = keyUnpackedOutQueue_.AllocTensor<int4b_t>();
        // key offset
        uint64_t keyCompressedOffset = 0;
        uint32_t keyOffset = MAX_FP16_PROCESS_NUM * MAX_SELECT_REPEATED_TIMES / COMPRESS_RATE;
        for (uint32_t index = 0; index < keySelectCycleCount; index++) {
            uint32_t selectAndCastOffset = index * MAX_FP16_PROCESS_NUM * MAX_SELECT_REPEATED_TIMES;
            uint32_t maxRepeatedTimes = (tailRepeateadTimes > 0 && (index == keySelectCycleCount - 1))
                                            ? tailRepeateadTimes
                                            : MAX_SELECT_REPEATED_TIMES;
            SelectCustom<half>(selectTensor, keyCompressed[keyCompressedOffset], constTensor, static_cast<uint8_t>(maxRepeatedTimes));
            // DumpTensor(selectTensor, 366, 672);
            Cast<int4b_t, half>(keyUnpacked[selectAndCastOffset], selectTensor, RoundMode::CAST_CEIL, CAST_MASK, static_cast<uint8_t>(maxRepeatedTimes), {1, 1, 2, 8});
            keyCompressedOffset = keyOffset * (index + 1);
            // YF_LOG("keyGmOffset_ = %d tailRepeateadTimes = %d selectRepeatedTimes = %d \n", keyGmOffset_, tailRepeateadTimes, selectRepeatedTimes);
        }
        keyUnpackedOutQueue_.EnQue(keyUnpacked);
        keyUnpacked = keyUnpackedOutQueue_.DeQue<int4b_t>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(selectRepeatedTimes * MAX_FP16_PROCESS_NUM / 2 / BLOCK_CUBE), 0, 0};  // 2: 1/2, size of int4b_t
        DataCopy(unpackGm_[keyGmOffset * COMPRESS_RATE], keyUnpacked, copyParams);  // output to outQueue1 with DB_ON
        keyUnpackedOutQueue_.FreeTensor(keyUnpacked);
    }

    __aicore__ inline void UnpackKeyCompressedWithLittleDim(LocalTensor<half> &selectTensor,
                                                            LocalTensor<half> &constTensor,
                                                            const LocalTensor<uint8_t> &keyCompressed,
                                                            uint32_t dimension,
                                                            uint64_t keyGmOffset, uint32_t repeateadTimes) {
        SelectCustom<half>(selectTensor, keyCompressed, constTensor, static_cast<uint8_t>(repeateadTimes));
        PipeBarrier<PIPE_V>();
        LocalTensor<int4b_t> keyUnpacked = keyUnpackedOutQueue_.AllocTensor<int4b_t>();
        uint64_t mask = MAX_FP16_PROCESS_NUM;
        Cast<int4b_t, half>(keyUnpacked, selectTensor, RoundMode::CAST_CEIL, mask, static_cast<uint8_t>(repeateadTimes), {1, 1, 2, 8});
        keyUnpackedOutQueue_.EnQue(keyUnpacked);
        keyUnpacked = keyUnpackedOutQueue_.DeQue<int4b_t>();
        uint64_t unpackGmOffset = keyGmOffset * 8; /* 8: Original Dimension / Compressed Dimension */
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(repeateadTimes * dimension / 2), 0, 0, 0}; /* 2: 1 / size of int4b_t */
        if (param_.supportKeyRope) {
            DataCopyPad(kRopeUnpackGm_[unpackGmOffset], keyUnpacked, copyOutParams);
        } else {
            DataCopyPad(unpackGm_[unpackGmOffset], keyUnpacked, copyOutParams);
        }
        keyUnpackedOutQueue_.FreeTensor(keyUnpacked);
    }

    __aicore__ inline void ComputeMM(uint32_t batchIdx, uint32_t seqLen)
    {
        if ASCEND_IS_AIV {
            return;
        }
        mm_.SetOrgShape(param_.M, seqLen, param_.ka);
        mm_.SetSingleShape(param_.M, seqLen, param_.ka);
        float tmp = 1;
        uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
        mm_.SetQuantScalar(ans);

        uint32_t realBatchIdx = curCoreBatchStartIdx_ + batchIdx;
        mmOffsetA_ = realBatchIdx * (param_.ka + param_.rope_ka);
        mmOffsetB_ = realBatchIdx * param_.maxSeqLen * param_.kb;
        mmOffsetC_ = realBatchIdx * param_.maxSeqLen;
        mm_.SetTensorA(qUnpackGm_[mmOffsetA_], AMatmulType::isTrans);
        mm_.SetTensorB(unpackGm_[mmOffsetB_], BMatmulType::isTrans);
        mm_.IterateAll(matmulGm_[mmOffsetC_]);  // d

        if (param_.supportKeyRope) {
            SetFlag<HardEvent::FIX_MTE2>(eventIDFIX_MTE2);
            WaitFlag<HardEvent::FIX_MTE2>(eventIDFIX_MTE2);
            // DumpTensor(matmulGm_[mmOffsetC_], 435, 64);

            mmRope_.SetOrgShape(param_.M, seqLen, param_.rope_ka);
            mmRope_.SetSingleShape(param_.M, seqLen, param_.rope_ka);
            mmRope_.SetQuantScalar(ans);
            mmOffsetARope_ = mmOffsetA_ + param_.ka;
            mmOffsetBRope_ = realBatchIdx * param_.maxSeqLen * param_.rope_kb;
            mmRope_.SetTensorA(qUnpackGm_[mmOffsetARope_], AMatmulType::isTrans);
            mmRope_.SetTensorB(kRopeUnpackGm_[mmOffsetBRope_], BMatmulType::isTrans);
            mmRope_.IterateAll(matmulGm_[mmOffsetC_], 1);  // c
            // DumpTensor(matmulGm_[mmOffsetC_], 445, 64);
        }
    }

    __aicore__ inline void ComputeTopK(uint32_t batchIdx, uint32_t pingPongFlag)
    {
        if ASCEND_IS_AIC {
            return;
        }

        //YF_LOG("ldeng 299\n");

        uint32_t realReducedBatchIdx = curCoreBatchStartIdx_ + batchIdx;
        uint32_t realBatchIdx = realReducedBatchIdx / param_.head; /* batchIdx without headNum */

        // 当前batch是否需要skip
        if (supportMask_) {
            bool batchMask = maskGm_.GetValue(realBatchIdx);
            //YF_LOG("realBatchIdx = %d, batchMask = %d\n", realBatchIdx, batchMask);
            if (!batchMask) {
                // todo block table直接赋值给输出的Indices
                SetBlockTableForIndices(realBatchIdx, realReducedBatchIdx * param_.maxK);
                //YF_LOG("realBatchIdx = %d SetBlockTableForIndices\n", realBatchIdx);
                return;
            }
        }
        uint32_t curSeqLen = seqLenGm_.GetValue(realBatchIdx);
        uint32_t curK = kGm_.GetValue(realBatchIdx);
        uint32_t curChunkSize = 1;
        uint32_t curChunkNum = 0;
        if (chunkSizeGm_.GetPhyAddr() != nullptr) {
            curChunkSize = chunkSizeGm_.GetValue(realBatchIdx);
        }
        if (curChunkSize != 0) {
            if(curChunkSize == 1 || curChunkSize == 8 || curChunkSize == 16)
            {
                if (curSeqLen <= 32) { // seqLen <= 32时，跳过hammingDistTopK，在kvSelect算子把所有的数据都选上
                    curSeqLen = 0;
                } else {
                    curSeqLen = curSeqLen - Min(curSeqLen, static_cast<uint32_t>(curSeqLen % curChunkSize + 16)); /* 16: 用于计算lastlen长度 */
                }
                curChunkNum = curSeqLen / curChunkSize;
            }
            else if(curChunkSize == 64) {
                curChunkNum = ((curSeqLen + 63) / 64) * 64 / curChunkSize; //chunk_size=64时向上取整，通过recent强制取到尾chunk
            }
            else if(curChunkSize == 128)
            {
                curChunkNum =   ((curSeqLen + 127)/128)*128 / curChunkSize; //chunk_size=128时向上取整，通过recent强制取到尾chunk
            }
            
        } else {
            curSeqLen = 0;
        }
        if (curK == 0 || curSeqLen == 0) {
            return;
        }
        /* 重新校验K值，保证实际用到的K = min(kGm, curChunkNum, param_.maxK) */
        curK = Min(curK, curChunkNum);
        curK = Min(curK, param_.maxK);
        /* 
         * 若curK与curChunkNum相等，则说明curChunkNum不大于当前的param_.maxK，不会进入耗尽模式
         * 因此可以直接将topKBlockNum设置为1，tailN2和tileN2设置为curChunkNum，实现
         * 了tailN2计算时的除零保护。
         */
        uint32_t topKBlockNum = 1;
        /* param_.tileN2 > param_.maxK=> tileN2 >= curK */
        uint32_t tileN2 = Min(curChunkNum, param_.tileN2);  //param.tileN2=3328, it is very large


        uint32_t tailN2 = curChunkNum;
        if (curK < tileN2) {
            topKBlockNum = matmul::CeilDiv(curChunkNum - tileN2, tileN2 - curK) + 1;
            uint32_t effectiveTailN2 = (curChunkNum - tileN2) % (tileN2 - curK) > 0 ?
                (curChunkNum - tileN2) % (tileN2 - curK) : (tileN2 - curK);
            tailN2 = topKBlockNum > 1 ? (effectiveTailN2 + curK) : curChunkNum;
        }

        LocalTensor<half> topKOutValueTensor;
        LocalTensor<int32_t> topKOutIndexTensor;
        uint64_t mmOffset = realReducedBatchIdx * param_.N;
        uint32_t headChunkNum = 0;
        uint32_t tailChunkNum = 0;
    
        uint32_t chunkPerBlock = param_.tileN1 / curChunkSize;
        
        
        //uint32_t skipTailChunkNum = DivCeil(SKIP_TAIL_TOKEN_NUM, curChunkSize);
        //uint32_t skipHeadChunkNum = DivCeil(SKIP_HEAD_TOKEN_NUM, curChunkSize);
        uint32_t skipTailChunkNum = param_.recent;
        uint32_t skipHeadChunkNum = param_.sink;
        
        uint32_t last_block_tail_num = 0;
        uint32_t second_last_block_tail_num = 0;
        if (tailN2 >= skipTailChunkNum)
        {
            last_block_tail_num = skipTailChunkNum;
        }
        else
        {
            last_block_tail_num = tailN2;
            second_last_block_tail_num = skipTailChunkNum - tailN2;
        }

        //YF_LOG("batchIdx=%d, curSeqLen=%d, curChunkSize=%d, curChunkNum=%d, tileN2=%d,tailN2=%d,topKBlockNum=%d,skipTailChunkNum=%d,skipHeadChunkNum=%d,last_block_tail_num=%d,second_last_block_tail_num=%d, param_.tileN1=%d, \n", batchIdx, curSeqLen, curChunkSize, curChunkNum, tileN2, tailN2, topKBlockNum, skipTailChunkNum, skipHeadChunkNum, last_block_tail_num, second_last_block_tail_num,param_.tileN1);

        for (uint32_t i = 0; i < topKBlockNum; i++) {
            uint32_t copyLen = i == topKBlockNum - 1 ? tailN2 : tileN2;
            uint64_t matmulGmOffset = i == 0 ? mmOffset : mmOffset + i * (tileN2 - curK);
            GenerateTopKValueTensor(i, copyLen, tileN2, matmulGmOffset, curK, curChunkSize);
            GenerateTopKIndexTensor(i, copyLen, tileN2, matmulGmOffset - mmOffset, curK);
            LocalTensor<half> topKInValueTensor = topKInValueQueue_.DeQue<half>();

            //YF_LOG("ldeng 393 topKInValueTensor.GetSize()=%d\n", topKInValueTensor.GetSize());
            
            uint32_t chunkPerBlock = param_.tileN1 / curChunkSize;      // blocksize / chunksize = chunknum per block
            if (headChunkNum < skipHeadChunkNum) {
                uint32_t curHeadChunkNum = min(skipHeadChunkNum - headChunkNum, copyLen);
                headChunkNum += curHeadChunkNum;
                //YF_LOG("ldeng 399\n");
                //DumpTensor(topKInValueTensor, 400, topKInValueTensor.GetSize());
                Duplicate(topKInValueTensor, static_cast<half>(MAX_HALF_VALUE), curHeadChunkNum);
                //YF_LOG("ldeng 402\n");
                //DumpTensor(topKInValueTensor, 403, topKInValueTensor.GetSize());
            }


            if(i == topKBlockNum -2 && second_last_block_tail_num > 0)
            {
                //uint32_t offset = tileN2 - second_last_block_tail_num;
                //Maxs(topKInValueTensor[offset], topKInValueTensor[offset], MAX_HALF_VALUE, second_last_block_tail_num);
                //YF_LOG("ldeng 385\n");
                FillMaxValueFromTail(topKInValueTensor, tileN2, second_last_block_tail_num, curChunkSize);
            }

            if(i == topKBlockNum -1)
            {
                if(last_block_tail_num == tailN2)
                {
                    //Maxs(topKInValueTensor, topKInValueTensor, MAX_HALF_VALUE, tailN2);
                    //YF_LOG("ldeng 420\n");
                    //DumpTensor(topKInValueTensor, 421, topKInValueTensor.GetSize());
                    Duplicate(topKInValueTensor, static_cast<half>(MAX_HALF_VALUE), tailN2);
                    //YF_LOG("ldeng 423\n");
                    //DumpTensor(topKInValueTensor, 424, topKInValueTensor.GetSize());
                }
                else
                {
                    //uint32_t offset = tailN2 - last_block_tail_num;
                    //Maxs(topKInValueTensor[offset], topKInValueTensor[offset], MAX_HALF_VALUE, last_block_tail_num);
                    //YF_LOG("ldeng 430\n");
                    //DumpTensor(topKInValueTensor, 431, topKInValueTensor.GetSize());
                    FillMaxValueFromTail(topKInValueTensor, tailN2, last_block_tail_num, curChunkSize);
                    //DumpTensor(topKInValueTensor, 433, topKInValueTensor.GetSize());
                }
            }
            
            //YF_LOG("ldeng 406\n");

            LocalTensor<int32_t> topKInIndexTensor = topKInIndexQueue_.DeQue<int32_t>();
            topKOutValueTensor = topKOutValueQueue_.AllocTensor<half>();
            topKOutIndexTensor = topKOutIndexQueue_.AllocTensor<int32_t>();
            TopKCustom(topKOutValueTensor, topKOutIndexTensor, topKInValueTensor, topKInIndexTensor, curK, tilingData_, copyLen);
            topKInValueQueue_.FreeTensor(topKInValueTensor);
            topKInIndexQueue_.FreeTensor(topKInIndexTensor);
            topKOutValueQueue_.EnQue(topKOutValueTensor);
            topKOutIndexQueue_.EnQue(topKOutIndexTensor);
        }

        uint64_t topKOutGmOffset = static_cast<uint64_t>(realReducedBatchIdx) * param_.maxK;
        topKOutValueTensor = topKOutValueQueue_.DeQue<half>();
        topKOutIndexTensor = topKOutIndexQueue_.DeQue<int32_t>();
        if (!param_.supportOffload && (curChunkSize == 64 || curChunkSize == 128)) {
            // 将 TopK 的“chunk索引”映射成 block_id，并写回 GM(indices)
            WriteBlockTableFromTopK(realBatchIdx, topKOutIndexTensor, curK, topKOutGmOffset);
        } else {
            DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(curK * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(indicesGm_[topKOutGmOffset], topKOutIndexTensor, copyOutParams);
        }
        topKOutValueQueue_.FreeTensor(topKOutValueTensor);
        topKOutIndexQueue_.FreeTensor(topKOutIndexTensor);
    }

    __aicore__ inline void SetBlockTableForIndices(uint32_t curBatchIdx, uint64_t outGmOffset) {
        // 找到当前batch对应的tableblock
        LocalTensor<int32_t> tableBlockTensor = tableBlockBuf_.template Get<int32_t>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(param_.blockCount * sizeof(int32_t)), 0, 0};
        DataCopy(tableBlockTensor, keyBlockTableGm_[curBatchIdx * param_.blockCount], copyParams);
        //DumpTensor(tableBlockTensor, 500, 64);
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);

        // 直接复制
        uint32_t copyLen = param_.blockCount < param_.maxK ? param_.blockCount : param_.maxK;
        DataCopyExtParams cpOut{1, static_cast<uint32_t>(copyLen * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(indicesGm_[outGmOffset], tableBlockTensor, cpOut);
    }

    __aicore__ inline void GenerateTopKValueTensor(uint32_t i, uint32_t copyLen,
        uint32_t tileN2, uint64_t matmulGmOffset, uint32_t curK, uint32_t chunkSize)
    {
        LocalTensor<half> topKInValueTensor = topKInValueQueue_.AllocTensor<half>();
        uint32_t copyLenAligned = copyLen / BLOCK_CUBE * BLOCK_CUBE; /* floor aligned for datacopy */
        if (copyLenAligned < param_.tileN2) {
            Duplicate(topKInValueTensor[copyLenAligned], static_cast<half>(MIN_HALF_VALUE), param_.tileN2 - copyLenAligned);
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }

        if (chunkSize > 1) {
            uint16_t chunkNum = static_cast<uint16_t>(copyLen);
            LocalTensor<half> reduceInputTensor = topKInIndexQueue_.AllocTensor<int32_t>().ReinterpretCast<half>();
            ReduceMaxCustom(matmulGm_[matmulGmOffset], reduceInputTensor, topKInValueTensor, chunkNum, static_cast<uint8_t>(chunkSize));
            topKInValueQueue_.EnQue(topKInValueTensor);
            topKInIndexQueue_.FreeTensor(reduceInputTensor);
        } else {
            uint32_t copyLenCeilAligned = matmul::CeilDiv(copyLen * sizeof(half), BLOCK_CUBE)
                * BLOCK_CUBE / sizeof(half);
            DataCopyExtParams copyInParams{1, static_cast<uint32_t>(copyLen * sizeof(half)), 0, 0, 0};
            DataCopyPadExtParams<half> copyInPadParams{true, 0, static_cast<uint8_t>(copyLenCeilAligned - copyLen),
                static_cast<half>(MIN_HALF_VALUE)};
            DataCopyPad(topKInValueTensor, matmulGm_[matmulGmOffset], copyInParams, copyInPadParams);
            topKInValueQueue_.EnQue(topKInValueTensor);
            if (i > 0) {
                LocalTensor<half> topKOutValueTensor = topKOutValueQueue_.DeQue<half>();
                topKInValueTensor = topKInValueQueue_.DeQue<half>();
                uint64_t valueMask = curK > MAX_FP16_PROCESS_NUM ? MAX_FP16_PROCESS_NUM : curK;
                uint8_t valueRepeatTimes = curK / MAX_FP16_PROCESS_NUM;
                if (valueRepeatTimes > 0) {
                    Copy(topKInValueTensor, topKOutValueTensor, valueMask, valueRepeatTimes, {1, 1, 8, 8});
                }
                if (curK % MAX_FP16_PROCESS_NUM != 0) {
                    Copy(topKInValueTensor[valueRepeatTimes * MAX_FP16_PROCESS_NUM],
                        topKOutValueTensor[valueRepeatTimes * MAX_FP16_PROCESS_NUM],
                        curK % MAX_FP16_PROCESS_NUM, 1, {1, 1, 8, 8});
                }
                PipeBarrier<PIPE_V>();
                topKInValueQueue_.EnQue(topKInValueTensor);
                topKOutValueQueue_.FreeTensor(topKOutValueTensor);
            }
        }
    }

    __aicore__ inline void GenerateTopKIndexTensor(uint32_t i, uint32_t copyLen,
        uint32_t tileN2, uint64_t startIndex, uint32_t curK)
    {
        LocalTensor<int32_t> topKInIndexTensor = topKInIndexQueue_.AllocTensor<int32_t>();
        ArithProgression(topKInIndexTensor, static_cast<int32_t>(startIndex), 1, static_cast<int32_t>(copyLen));
        topKInIndexQueue_.EnQue(topKInIndexTensor);
        if (i > 0) {
            LocalTensor<int32_t> topKOutIndexTensor = topKOutIndexQueue_.DeQue<int32_t>();
            topKInIndexTensor = topKInIndexQueue_.DeQue<int32_t>();
            uint64_t indexMask = curK > MAX_INT32_PROCESS_NUM ? MAX_INT32_PROCESS_NUM : curK;
            uint8_t indexRepeatTimes = curK / MAX_INT32_PROCESS_NUM;
            if (indexRepeatTimes > 0) {
                Copy(topKInIndexTensor, topKOutIndexTensor, indexMask, indexRepeatTimes, {1, 1, 8, 8});
            } 
            if (curK % MAX_INT32_PROCESS_NUM != 0) {
                Copy(topKInIndexTensor[indexRepeatTimes * MAX_INT32_PROCESS_NUM],
                    topKOutIndexTensor[indexRepeatTimes * MAX_INT32_PROCESS_NUM],
                    curK % MAX_INT32_PROCESS_NUM, 1, {1, 1, 8, 8});
            }
            PipeBarrier<PIPE_V>();
            topKInIndexQueue_.EnQue(topKInIndexTensor);
            topKOutIndexQueue_.FreeTensor(topKOutIndexTensor);
        }
    }

    template <pipe_t pipe=PIPE_S>
    __aicore__ inline void SyncAicOnly(uint16_t eventId) {
        CrossCoreSetFlag<SYNC_MODE0, pipe>(eventId);
        CrossCoreWaitFlag(eventId);
    }

    template <pipe_t pipe=PIPE_S>
    __aicore__ inline void SyncAivOnly(uint16_t eventId) {
        CrossCoreSetFlag<SYNC_MODE0, pipe>(eventId);
        CrossCoreWaitFlag(eventId);
    }

    template <pipe_t pipe=PIPE_S>
    __aicore__ inline void VectorNotifyCube(uint16_t aiv2AicEventId) {
        CrossCoreSetFlag<SYNC_MODE2, pipe>(aiv2AicEventId);
    }

    __aicore__ inline void CubeWaitVector(uint16_t aiv2AicEventId) {
        CrossCoreWaitFlag(aiv2AicEventId);
    }

    template <pipe_t pipe=PIPE_S>
    __aicore__ inline void CubeNotifyVector(uint16_t aic2AivEventId) {
        CrossCoreSetFlag<SYNC_MODE2, pipe>(aic2AivEventId);
    }

    __aicore__ inline void VectorWaitCube(uint16_t aic2AivEventId) {
        CrossCoreWaitFlag(aic2AivEventId);
    }

protected:
    uint64_t innerSplitLoopTimes_ = 0;
    uint8_t innerSplitGMFlag_ = 0;

    static constexpr uint32_t BLOCK_CUBE = 32;
    static constexpr uint64_t SYNC_MODE0 = 0;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_AIV_ONLY_ALL_FLAG = 0;
    static constexpr uint64_t SYNC_AIC_ONLY_ALL_FLAG = 1;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 2;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 4;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG2 = 6;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG2 = 0;
    static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
    static constexpr uint32_t COMPRESSED_DIMENSION = 16;
    static constexpr uint32_t BATCH_PING_PONG_NUM = 8; // 核间同步flag的最大深度是15，
    static constexpr uint32_t SUB_BLOCK_NUM = 2;
    static constexpr uint32_t SUB_BLOCK_NUM_WITH_DB = 4;
    static constexpr float MIN_HALF_VALUE = -65535;
    int32_t eventIDFIX_MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_MTE2));

    static constexpr uint32_t MAX_SELECT_REPEATED_TIMES = 254; // Select最大迭代次数

    GlobalTensor<uint8_t> queryGm_;
    GlobalTensor<uint8_t> keyGm_;
    GlobalTensor<uint8_t> keyRopeGm_;
    GlobalTensor<int32_t> kGm_;
    GlobalTensor<int32_t> seqLenGm_;
    GlobalTensor<int32_t> chunkSizeGm_;
    GlobalTensor<int32_t> keyBlockTableGm_;
    GlobalTensor<bool> maskGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int4b_t> unpackGm_;
    GlobalTensor<int4b_t> kRopeUnpackGm_;
    GlobalTensor<half> matmulGm_;
    GlobalTensor<int4b_t> qUnpackGm_;

    TPipe *pipe_;
    TQue<TPosition::VECIN, 1> keyCompressedInQueue_;
    TQue<TPosition::VECOUT, 1> keyUnpackedOutQueue_;
    TQue<TPosition::VECIN, 1> topKInValueQueue_;
    TQue<TPosition::VECIN, 1> topKInIndexQueue_;
    TQue<TPosition::VECOUT, 1> topKOutValueQueue_;
    TQue<TPosition::VECOUT, 1> topKOutIndexQueue_;
    TBuf<TPosition::VECCALC> constBuf_;
    TBuf<TPosition::VECCALC> selectBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TQue<TPosition::VECIN, 1> queryCompressedInQueue_;
    TQue<TPosition::VECOUT, 1> queryUnpackedOutQueue_;
    TBuf<TPosition::VECCALC> qReduceSumLastRowBuf_;
    TBuf<TPosition::VECCALC> qReduceSumBuf_;

    TBuf<TPosition::VECIN> tableBlockBuf_;

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

    uint64_t mmOffsetA_;
    uint64_t mmOffsetB_;
    uint64_t mmOffsetARope_;
    uint64_t mmOffsetBRope_;
    uint64_t mmOffsetC_;
    uint32_t curCoreBatch_;
    uint32_t curCoreBatchStartIdx_;
    bool continFlag_;
    bool supportMask_ = true;

    __aicore__ inline void InitParams()
    {
        mmOffsetA_ = 0;
        mmOffsetB_ = 0;
        mmOffsetC_ = 0;

        param_.preCoreNum = param_.reducedBatch % param_.usedCoreNum;
        if (param_.preCoreNum == 0) {
            param_.preCoreNum = param_.usedCoreNum;
        }
        uint32_t blockIndex = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            blockIndex = blockIndex / 2;
        }
        if (blockIndex < param_.preCoreNum) {
            curCoreBatch_ = param_.singleCoreBatch;
        } else {
            curCoreBatch_ = param_.singleCoreBatch - 1;
        }

        curCoreBatchStartIdx_ = blockIndex * curCoreBatch_;
        if (blockIndex >= param_.preCoreNum) {
            curCoreBatchStartIdx_ += param_.preCoreNum;
        }
    }

    __aicore__ inline void InitTilingParams(const TCubeTiling &tiling, const TCubeTiling &tilingRope, const TopkTiling &topkTiling,
        const HammingDistTopKTilingParams &tilingParam)
    {
        // tiling data for select
        param_.usedCoreNum = tilingParam.usedCoreNum;
        param_.batch = tilingParam.batch;
        param_.head = tilingParam.head;
        param_.dimension = tilingParam.dimension;
        param_.nope_dimension = tilingParam.nope_dimension;
        param_.rope_dimension = tilingParam.rope_dimension;
        param_.reducedBatch = tilingParam.reducedBatch;
        param_.tileN1 = tilingParam.tileN1;
        param_.tileN2 = tilingParam.tileN2;
        param_.singleCoreBatch = tilingParam.singleCoreBatch;
        param_.qHead = tilingParam.qHead;
        param_.headGroupNum = tilingParam.headGroupNum;
        
        param_.maxK = tilingParam.maxK;

        // support key rope
        param_.supportKeyRope = tilingParam.supportKeyRope > 0;

        // tiling data for matmul
        param_.M = tiling.M;
        param_.N = tiling.N;
        param_.ka = tiling.Ka;
        param_.kb = tiling.Kb;
        if (param_.supportKeyRope) {
            param_.rope_ka = tilingRope.Ka;
            param_.rope_kb = tilingRope.Kb;
        }

        // tiling data for topk
        param_.mmGmOffset = tilingParam.mmGmOffset;
        param_.qUnpackGmOffset = tilingParam.qUnpackGmOffset;
        param_.kNopeUnpackGmOffset = tilingParam.kNopeUnpackGmOffset;
        topkTiling_ = topkTiling;
        param_.maxSeqLen = tilingParam.maxSeqLen;
        param_.sink = tilingParam.sink;
        param_.recent = tilingParam.recent;
        param_.blockCount = tilingParam.blockCount;

        // support offload
        param_.supportOffload = tilingParam.supportOffload > 0;
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR query, GM_ADDR keyCompressed, GM_ADDR keyCompressedRope, GM_ADDR k, GM_ADDR seqLen,
        GM_ADDR chunkSize, GM_ADDR keyBlockTable, GM_ADDR mask,GM_ADDR indices, GM_ADDR workSpace)
    {
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(query));
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(keyCompressed));
        keyRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(keyCompressedRope));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(k));
        seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLen));
        chunkSizeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunkSize));
        keyBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(keyBlockTable));
        maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(mask));
        supportMask_ = maskGm_.GetPhyAddr() != nullptr;
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        unpackGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(workSpace));
        kRopeUnpackGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(workSpace + param_.kNopeUnpackGmOffset));
        qUnpackGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(workSpace + param_.qUnpackGmOffset));
        matmulGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workSpace + param_.mmGmOffset));
    }

    __aicore__ inline void InitLocalBuffersForUnpackKey()
    {
        pipe_->InitBuffer(keyCompressedInQueue_, 1, param_.tileN1 * (param_.dimension / COMPRESS_RATE) * sizeof(int8_t)); /* 8: original dimension / compressed dimension */
        pipe_->InitBuffer(keyUnpackedOutQueue_, 1, param_.tileN1 * param_.dimension * sizeof(int8_t) / 2); /* 2: 1 / sizeof(int4b_t) */ 
        pipe_->InitBuffer(constBuf_, param_.dimension * sizeof(half));
        // 超过Select最大迭代限制次数
        uint64_t selectRepeatedTimes = computeSelectRepeatedTimes(param_.dimension);
        if (selectRepeatedTimes > MAX_SELECT_REPEATED_TIMES) {
            pipe_->InitBuffer(selectBuf_, MAX_FP16_PROCESS_NUM * MAX_SELECT_REPEATED_TIMES * sizeof(half));
        } else {
            pipe_->InitBuffer(selectBuf_, param_.tileN1 * param_.dimension * sizeof(half));
        }
    }

    __aicore__ inline uint64_t computeSelectRepeatedTimes(uint32_t dimension) {
        uint64_t selectElmentCount = param_.tileN1 * dimension;
        uint64_t selectRepeatedTimes = (selectElmentCount + MAX_FP16_PROCESS_NUM - 1) / MAX_FP16_PROCESS_NUM;
        return selectRepeatedTimes;
    }

    __aicore__ inline void InitLocalBuffersForUnpackQuery()
    {
        pipe_->InitBuffer(constBuf_, param_.dimension * sizeof(half));
        pipe_->InitBuffer(selectBuf_, param_.headGroupNum * param_.dimension * sizeof(half));
        pipe_->InitBuffer(queryCompressedInQueue_, 1, param_.headGroupNum * (param_.dimension / 8) * sizeof(int8_t)); /* 8: original dimension / compressed dimension */
        pipe_->InitBuffer(queryUnpackedOutQueue_, 1, param_.headGroupNum * param_.dimension * sizeof(int8_t) / 2); /* 2: 1 / sizeof(int4b_t) */  
        pipe_->InitBuffer(qReduceSumLastRowBuf_, param_.headGroupNum * param_.dimension * sizeof(half));
        pipe_->InitBuffer(qReduceSumBuf_, param_.headGroupNum * param_.dimension * sizeof(half));
    }

    __aicore__ inline void InitLocalBuffersForTopK() {
        pipe_->InitBuffer(topKInValueQueue_, 1, param_.tileN2 * sizeof(half));
        pipe_->InitBuffer(topKInIndexQueue_, 1, param_.tileN2 * 8 * sizeof(int32_t)); /* 8: 最大的chunkSize * sizeof(int32) / sizeof(half) */
        pipe_->InitBuffer(topKOutValueQueue_, 1, param_.maxK * sizeof(half));
        pipe_->InitBuffer(topKOutIndexQueue_, 1, param_.maxK * sizeof(int32_t));
        pipe_->InitBuffer(tableBlockBuf_, param_.blockCount * sizeof(int32_t));
    }
    
    __aicore__ inline void WriteBlockTableFromTopK(
        uint32_t curBatchIdx,
        LocalTensor<int32_t>& topKIndexUb,
        uint32_t curKScalar,
        uint64_t outGmOffset)
    {
        if ASCEND_IS_AIC { return; }
        // 复用现有的 int32 队列分配一块UB作为写回中转
        LocalTensor<int32_t> blockIdUb = topKInIndexQueue_.AllocTensor<int32_t>();

        LocalTensor<int32_t> tableBlockTensor = tableBlockBuf_.template Get<int32_t>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(param_.blockCount * sizeof(int32_t)), 0, 0};
        DataCopy(tableBlockTensor, keyBlockTableGm_[curBatchIdx * param_.blockCount], copyParams);

        ::AscendC::WriteBlockTableFromTopK(curBatchIdx, topKIndexUb, blockIdUb, curKScalar, outGmOffset,
            tableBlockTensor, indicesGm_, continFlag_, param_.blockCount);

        topKInIndexQueue_.FreeTensor(blockIdUb);
    }
};
} // namespace AscendC
#endif // HAMMING_DIST_TOP_K_PARALLEL_H