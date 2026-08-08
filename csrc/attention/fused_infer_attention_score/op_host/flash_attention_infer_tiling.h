/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_infer_tiling.h
 * \brief
 */

#ifndef FLASH_ATTN_INFER_TILING_H
#define FLASH_ATTN_INFER_TILING_H
#include "exe_graph/runtime/tiling_context.h"
#include "register/tilingdata_base.h"
#include "fused_infer_attention_score_tiling.h"

namespace optiling{
    constexpr int32_t MAX_CORE_NUM_FD = 26;

    BEGIN_TILING_DATA_DEF(coreNode)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startBIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startN1Idx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startS1Idx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startS2Idx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endBIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endN1Idx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endS1Idx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endS2Idx)
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, firstSplitKVTaskLseOffset)
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, firstSplitKVTaskOOffset)
    END_TILING_DATA_DEF
    REGISTER_TILING_DATA_CLASS(coreNodeOp, coreNode)

    BEGIN_TILING_DATA_DEF(splitNode)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, batchIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, headStartIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, headEndIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, qStartIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, qEndIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, splitNum)
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, lseTaskOffset)
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, oTaskOffset)
    END_TILING_DATA_DEF
    REGISTER_TILING_DATA_CLASS(splitNodeOp, splitNode)

    BEGIN_TILING_DATA_DEF(StridesConstInfo)
    TILING_DATA_FIELD_DEF(uint64_t, bnStride)
    END_TILING_DATA_DEF
    REGISTER_TILING_DATA_CLASS(StridesConstInfoOp, StridesConstInfo)

    BEGIN_TILING_DATA_DEF(FAInferTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numHeads)
    TILING_DATA_FIELD_DEF(uint32_t, embeddingSize)
    TILING_DATA_FIELD_DEF(uint32_t, embeddingSizeV)
    TILING_DATA_FIELD_DEF(uint32_t, numBlocks)
    TILING_DATA_FIELD_DEF(uint32_t, blockSize)
    TILING_DATA_FIELD_DEF(uint32_t, maxQSeqlen)
    TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqlen)
    TILING_DATA_FIELD_DEF(uint32_t, kvHeads)
    TILING_DATA_FIELD_DEF(uint32_t, batch)
    TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch)
    TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, maskType)
    TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize)
    TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize)
    TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize)
    TILING_DATA_FIELD_DEF(uint64_t, UpdateSize)
    TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize)
    TILING_DATA_FIELD_DEF(float, scaleValue)
    TILING_DATA_FIELD_DEF(uint64_t, pseQ)
    TILING_DATA_FIELD_DEF(uint64_t, pseKv)
    TILING_DATA_FIELD_DEF(uint32_t, padding3)
    TILING_DATA_FIELD_DEF(int64_t, preToken)
    TILING_DATA_FIELD_DEF(int64_t, nextToken)
    TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
    TILING_DATA_FIELD_DEF(uint64_t, splitLseTotalSize)
    TILING_DATA_FIELD_DEF(uint64_t, splitOTotalSize)
    TILING_DATA_FIELD_DEF(uint32_t, totalSplitNodeNum)
    TILING_DATA_FIELD_DEF(uint32_t, needCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, mainLoopTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, tailLoopTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, tailStartBatch)
    TILING_DATA_FIELD_DEF(uint32_t, tailStartN2)
    TILING_DATA_FIELD_DEF(uint32_t, tailKvNBlockTile)
    TILING_DATA_FIELD_DEF_STRUCT(StridesConstInfo, keyStrides)
    TILING_DATA_FIELD_DEF_STRUCT(StridesConstInfo, valueStrides)
    TILING_DATA_FIELD_DEF_STRUCT(coreNode, coreInfo)
    TILING_DATA_FIELD_DEF_STRUCT(splitNode, splitInfo)
    END_TILING_DATA_DEF

    const uint32_t SIZE_OF_16BIT = 2;
    const uint32_t SIZE_OF_32BIT = 4;
    const uint32_t N_SPLIT_HELPER = 2;
    const uint32_t MAX_KV_STACK_LEN = 512;
    const uint32_t Q_TILE_CEIL = 128;
    const uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
    const uint32_t BASE_KV_SIZE = 128;
    const uint32_t PRELANCH_NUM = 3;
    const int64_t SPARSE_MODE_INT_MAX = 2147483647;
    const uint32_t TAIL_TASK_DIVISOR = 2;

    enum class MaskType : uint32_t {
        NO_MASK = 0,
        MASK_SPEC = 1,
        SWA_MASK = 3,
        FULL_MASK = 4
    };

    enum class DataType : uint32_t {
        FP16 = 0,
        BF16 = 1
    };

    struct BatchParams {
        uint32_t qSeqlen;
        uint32_t kvSeqlen;
        uint32_t curQNBlockTile;
        uint32_t qNBlockNumPerGroup;
        uint32_t curQNBlockNum;
        uint32_t curQSBlockTile;
        uint32_t curQSBlockNum;
        uint32_t curKSBlockTile;
        uint32_t curKSBlockNum;
    };

    struct FAInferContext {
        int32_t numTokens = 0;
        int32_t numHeads = 0;
        int32_t embeddingSize = 0;
        int32_t embeddingSizeV = 0;
        int32_t numBlocks = 0;
        int32_t blockSize = 0;
        int32_t kvHeads = 0;
        int32_t batch = 0;
        int32_t innerPrecise = 0;
        int64_t maxQSeqlen = 0;
        int64_t maxKvSeqlen = 0;
        int64_t pseQ = 0;
        int64_t pseKv = 0;
        int64_t preToken = 0;
        int64_t nextToken = 0;
        int32_t sparseMode = 0;
        uint32_t maxNumBlocksPerBatch = 0;
        const int64_t *qSeqlenList{nullptr};
        const int64_t *kvSeqlenList{nullptr};
        float scaleValue = 0.0;
        size_t* workspaces{nullptr};
        MaskType maskType = MaskType::MASK_SPEC;
        DataType dataType = DataType::FP16;
        bool pagedCacheFlag = false;
        bool lseFlag = false;
        bool learnableSinkFlag = false;
        bool isTilingSink = false;
        bool flashDecodeFlag = false;
        bool decodingFlag = false;
        bool kvcacheNzFlag = false;
        string layout;
        uint64_t keyBnStride = 0;
        uint64_t valueBnStride = 0;
    };

    class FAInferTiling {
    public:
        FAInferTiling() = default;
        explicit FAInferTiling(const FAInferContext &faInfo);
        ge::graphStatus DoTiling(FAInferTilingData &tilingdata);
        void SetCoreNum(uint32_t blockNum) {
            this->blockNum_ = blockNum;
        }
        uint32_t GetCoreNum() {
            return this->blockNum_;
        }
        uint64_t GetTilingKey();
    private:
        void FillSplitCoreTilingData(FAInferTilingData &tilingdata);
        void FillWorkSpaceTilingData(FAInferTilingData &faTilingData);
        uint32_t GetQSBlockTile(int64_t kvSeqlen);
        uint32_t GetQSBlockTileDecode(int64_t qSeqlen);
        uint32_t GetKvNBlockTile(uint32_t rowNumPerQSGTile, uint32_t kvHead);
        uint32_t GetKSBlockTile(int64_t kvSeqlen);
        uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize);
        void FillBasicTilingData(FAInferTilingData &faTilingData);
        void splitBN2S1GS2(FAInferTilingData &faTilingData);
        void SplitCoreDecodeBS1GN2(FAInferTilingData &faTilingData);
        BatchParams getBatchParams(uint32_t bIdx, uint32_t groupSize);
        void fillCoreInfoForFlashDecode(FAInferTilingData &faTilingData, uint32_t groupSize, uint64_t perCoreTaskNum);
        void fillSplitInfoForFlashDecode(FAInferTilingData &faTilingData, uint32_t groupSize);
        void InitCoreInfoArrays(FAInferTilingData &faTilingData);
        void ConsumeS2BlocksFD(uint32_t groupSize, int64_t &resTaskNum,
            uint32_t &nowBIdx, uint32_t &nowS1Idx, uint32_t &nowS2Idx);
        void ConsumeRemainingBatchesFD(uint32_t groupSize, int64_t &resTaskNum,
            uint32_t &nowBIdx, uint32_t &nowN1Idx, uint32_t &nowS1Idx, uint32_t &nowS2Idx);
        void ConsumeRemainingN1GroupsFD(int64_t &resTaskNum, const BatchParams &p,
            uint32_t &nowN1Idx, uint32_t &nowS1Idx, uint32_t &nowS2Idx);
        void ConsumeRemainingS1GroupsFD(int64_t &resTaskNum, const BatchParams &p,
            uint32_t &nowS1Idx, uint32_t &nowS2Idx);
        void InitSplitInfoArrays(FAInferTilingData &faTilingData);
        void ProcessCoreSplitInfo(FAInferTilingData &faTilingData, uint32_t groupSize,
            uint32_t coreIdx, int32_t &splitIdx, int32_t &prevBIdx,
            int32_t &prevN1Idx, int32_t &prevS1Idx,
            int64_t &currentLseTaskOffset, int64_t &currentOTaskOffset);
    private:
        FAInferContext faInfo_;
        uint32_t blockNum_;
    };

    FAInferTiling::FAInferTiling(const FAInferContext &faInfo): faInfo_(faInfo) {}

    uint32_t FAInferTiling::GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        uint32_t qRowNumCeil = Q_TILE_CEIL;
        uint32_t qNBlockTile = (qSeqlen != 0) ?
            (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
        qNBlockTile = std::min(qNBlockTile, groupSize);
        qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
        return qNBlockTile;
    }

    uint32_t FAInferTiling::GetQSBlockTile(int64_t kvSeqlen)
    {
        uint32_t qSBlockTile = Q_TILE_CEIL;
        return qSBlockTile;
    }

    uint32_t FAInferTiling::GetQSBlockTileDecode(int64_t qSeqlen)
    {
        uint32_t qSBlockTile = std::min(Q_TILE_CEIL, static_cast<uint32_t>(qSeqlen));
        return qSBlockTile;
    }

    uint32_t FAInferTiling::GetKvNBlockTile(uint32_t rowNumPerQSGTile, uint32_t kvHead)
    {
        uint32_t rowNumCeilPerQSGKvNTile = Q_TILE_CEIL;
        uint32_t kvNBlockTile = rowNumCeilPerQSGKvNTile / rowNumPerQSGTile;
        kvNBlockTile = std::min(kvNBlockTile, kvHead);
        kvNBlockTile = std::max(kvNBlockTile, static_cast<uint32_t>(1));
        return kvNBlockTile;
    }
    uint32_t FAInferTiling::GetKSBlockTile(int64_t kvSeqlen)
    {
        uint32_t kSBlockTile = MAX_KV_STACK_LEN;
        return kSBlockTile;
    }

    void FAInferTiling::FillBasicTilingData(FAInferTilingData &faTilingData)
    {
        faTilingData.set_batch(static_cast<uint32_t>(faInfo_.batch));
        faTilingData.set_numHeads(static_cast<uint32_t>(faInfo_.numHeads));
        faTilingData.set_kvHeads(static_cast<uint32_t>(faInfo_.kvHeads));

        faTilingData.set_embeddingSize(static_cast<uint32_t>(faInfo_.embeddingSize));
        faTilingData.set_embeddingSizeV(static_cast<uint32_t>(faInfo_.embeddingSizeV));
        faTilingData.set_numBlocks(static_cast<uint32_t>(faInfo_.numBlocks));
        if (faInfo_.pagedCacheFlag) {
            faTilingData.set_blockSize(static_cast<uint32_t>(faInfo_.blockSize));
        } else {
            faTilingData.set_blockSize(BASE_KV_SIZE);
        }
        faTilingData.set_maxQSeqlen(faInfo_.maxQSeqlen);
        faTilingData.set_maxKvSeqlen(faInfo_.maxKvSeqlen);
        faTilingData.set_maxNumBlocksPerBatch(faInfo_.maxNumBlocksPerBatch);
        faTilingData.set_maskType(static_cast<uint32_t>(faInfo_.maskType));
        faTilingData.set_scaleValue(faInfo_.scaleValue);
        faTilingData.set_sparseMode(faInfo_.sparseMode);
        faTilingData.set_preToken(static_cast<int64_t>(faInfo_.preToken));
        faTilingData.set_nextToken(static_cast<int64_t>(faInfo_.nextToken));
        faTilingData.set_pseQ(faInfo_.pseQ);
        faTilingData.set_pseKv(faInfo_.pseKv);
        faTilingData.keyStrides.set_bnStride(faInfo_.keyBnStride);
        faTilingData.valueStrides.set_bnStride(faInfo_.valueBnStride);
    }

    uint64_t FAInferTiling::GetTilingKey()
    {
        constexpr uint64_t SPLIT_FUSE_BASE_KEY = 5000000000000000000;
        constexpr uint64_t PAGED_CACHE_KEY = 10000000;
        constexpr uint64_t COMP_CAUSAL_MASK_KEY = 3;
        constexpr uint64_t COMP_SWA_MASK_KEY = 5;
        constexpr uint64_t FULL_MASK_KEY = 6;
        constexpr uint64_t KVCACHE_NZ_KEY = 10;
        constexpr uint64_t LAYOUTQ_TND_KEY = 200000;
        constexpr uint64_t DTYPE_FP16_KEY = 100;
        constexpr uint64_t DTYPE_BF16_KEY = 200;
        constexpr uint64_t LSE_OUT_ONLY_KEY = 1000;
        constexpr uint64_t INNER_LOW_PREC_KEY = 10000;
        constexpr uint64_t LEARNABLE_SINK_KEY = 100000000;
        constexpr uint64_t FLASH_DECODE_KEY = 100000000000000000;
        constexpr uint64_t DECODING_KEY = 200000000000000000;
        uint64_t tilingKey = SPLIT_FUSE_BASE_KEY;
        if (faInfo_.pagedCacheFlag) {
            tilingKey += static_cast<uint64_t>(PAGED_CACHE_KEY);
        }
        if (faInfo_.maskType == MaskType::MASK_SPEC) {
            tilingKey += static_cast<uint64_t>(COMP_CAUSAL_MASK_KEY);
        } else if (faInfo_.maskType == MaskType::SWA_MASK) {
            tilingKey += static_cast<uint64_t>(COMP_SWA_MASK_KEY);
        } else if (faInfo_.maskType == MaskType::FULL_MASK) {
            tilingKey += static_cast<uint64_t>(FULL_MASK_KEY);
        }
        if (faInfo_.layout == "TND") {
            tilingKey += static_cast<uint64_t>(LAYOUTQ_TND_KEY);
        }
        if (faInfo_.kvcacheNzFlag) {
            tilingKey += static_cast<uint64_t>(KVCACHE_NZ_KEY);
        }
        if (faInfo_.dataType == DataType::FP16) {
            tilingKey += static_cast<uint64_t>(DTYPE_FP16_KEY);
        } else if (faInfo_.dataType == DataType::BF16) {
            tilingKey += static_cast<uint64_t>(DTYPE_BF16_KEY);
        }
        if (faInfo_.lseFlag) {
            tilingKey += static_cast<uint64_t>(LSE_OUT_ONLY_KEY);
        }
        if (faInfo_.learnableSinkFlag) {
            tilingKey += static_cast<uint64_t>(LEARNABLE_SINK_KEY);
        }
        if (faInfo_.innerPrecise == 1) {
            tilingKey += static_cast<uint64_t>(INNER_LOW_PREC_KEY);
        }
        if ((faInfo_.pagedCacheFlag) &&
            !(faInfo_.maskType == MaskType::SWA_MASK) &&
            !faInfo_.learnableSinkFlag &&
            !(faInfo_.innerPrecise == 1) &&
            faInfo_.flashDecodeFlag) {
            tilingKey += static_cast<uint64_t>(FLASH_DECODE_KEY);
        } else if (faInfo_.decodingFlag) {
            tilingKey += static_cast<uint64_t>(DECODING_KEY);
        }
        return tilingKey;
    }

    void FAInferTiling::FillWorkSpaceTilingData(FAInferTilingData &faTilingData)
    {
        uint64_t mm1OutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        uint64_t smOnlineOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_16BIT * PRELANCH_NUM;
        uint64_t mm2OutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        uint64_t UpdateSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        uint64_t splitLseTotalSize = 0;
        uint64_t splitOTotalSize = 0;
        if (faInfo_.isTilingSink) {
            splitLseTotalSize = 2 * static_cast<uint64_t>(blockNum_) * Q_TILE_CEIL * SIZE_OF_32BIT * faInfo_.numHeads;
            uint32_t embeddingSizeV = static_cast<uint32_t>(faInfo_.embeddingSizeV);
            splitOTotalSize = 2 * static_cast<uint64_t>(blockNum_) * Q_TILE_CEIL * embeddingSizeV * SIZE_OF_32BIT * faInfo_.numHeads;
            faTilingData.set_splitLseTotalSize(splitLseTotalSize);
            faTilingData.set_splitOTotalSize(splitOTotalSize);
            faTilingData.set_needCoreNum(blockNum_);
        } else {
            splitLseTotalSize = faTilingData.get_splitLseTotalSize();
            splitOTotalSize = faTilingData.get_splitOTotalSize();
        }
        uint64_t workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize + splitLseTotalSize + splitOTotalSize;
        faTilingData.set_mm1OutSize(mm1OutSize);
        faTilingData.set_smOnlineOutSize(smOnlineOutSize);
        faTilingData.set_mm2OutSize(mm2OutSize);
        faTilingData.set_UpdateSize(UpdateSize);
        faTilingData.set_workSpaceSize(workSpaceSize);
    }

    void FAInferTiling::FillSplitCoreTilingData(FAInferTilingData &faTilingData)
    {
        uint32_t totalTaskNum = 0;
        uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
        for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; batchIdx++) {
            uint32_t qSeqlen = *(faInfo_.qSeqlenList + batchIdx);
            uint32_t kvSeqlen = *(faInfo_.kvSeqlenList + batchIdx);
            if (batchIdx > 0) {
                uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + batchIdx - 1);
                qSeqlen = qSeqlen - prevQSeqlenSum;
                if (!faInfo_.pagedCacheFlag) {
                    uint64_t prevKvSeqlenSum = *(faInfo_.kvSeqlenList + batchIdx - 1);
                    kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                }
            }
            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
            uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
            uint32_t curQNBlockNum = qNBlockNumPerGroup * faInfo_.kvHeads;
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
            uint32_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
            uint32_t curTaskNum = curQNBlockNum * curQSBlockNum;
            if (batchIdx == 0) {
                faTilingData.set_firstBatchTaskNum(curTaskNum);
            }
            totalTaskNum += curTaskNum;
        }
        faTilingData.set_totalTaskNum(totalTaskNum);
    }


    BatchParams FAInferTiling::getBatchParams(uint32_t bIdx, uint32_t groupSize) {
        BatchParams p;
        p.qSeqlen = *(faInfo_.qSeqlenList + bIdx);
        p.kvSeqlen = *(faInfo_.kvSeqlenList + bIdx);
        if (bIdx > 0 && faInfo_.layout == "TND") {
            uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + bIdx - 1);
            p.qSeqlen = p.qSeqlen - prevQSeqlenSum;
            if (!faInfo_.pagedCacheFlag) {
                uint64_t prevKvSeqlenSum = *(faInfo_.kvSeqlenList + bIdx - 1);
                p.kvSeqlen = p.kvSeqlen - prevKvSeqlenSum;
            }
        }
        p.curQNBlockTile = GetQNBlockTile(p.qSeqlen, groupSize);
        p.qNBlockNumPerGroup = (groupSize + p.curQNBlockTile - 1) / p.curQNBlockTile;
        p.curQNBlockNum = p.qNBlockNumPerGroup * faInfo_.kvHeads;
        p.curQSBlockTile = GetQSBlockTile(p.kvSeqlen);
        p.curQSBlockNum = (p.qSeqlen + p.curQSBlockTile - 1) / p.curQSBlockTile;
        p.curKSBlockTile = GetKSBlockTile(p.kvSeqlen);
        p.curKSBlockNum = (p.kvSeqlen + p.curKSBlockTile - 1) / p.curKSBlockTile;
        return p;
    }

    void FAInferTiling::InitCoreInfoArrays(FAInferTilingData &faTilingData)
    {
        for (uint32_t coreIdx = 0; coreIdx < blockNum_; coreIdx++) {
            faTilingData.coreInfo.get_startBIdx()[coreIdx] = 0;
            faTilingData.coreInfo.get_startN1Idx()[coreIdx] = 0;
            faTilingData.coreInfo.get_startS1Idx()[coreIdx] = 0;
            faTilingData.coreInfo.get_startS2Idx()[coreIdx] = 0;
            faTilingData.coreInfo.get_endBIdx()[coreIdx] = 0;
            faTilingData.coreInfo.get_endN1Idx()[coreIdx] = 0;
            faTilingData.coreInfo.get_endS1Idx()[coreIdx] = 0;
            faTilingData.coreInfo.get_endS2Idx()[coreIdx] = 0;
        }
    }

    void FAInferTiling::ConsumeS2BlocksFD(uint32_t groupSize, int64_t &resTaskNum,
        uint32_t &nowBIdx, uint32_t &nowS1Idx, uint32_t &nowS2Idx)
    {
        while (nowS2Idx < getBatchParams(nowBIdx, groupSize).curKSBlockNum && resTaskNum > 0) {
            BatchParams p = getBatchParams(nowBIdx, groupSize);
            uint32_t remainingQ = (nowS1Idx < p.curQSBlockNum - 1) ? p.curQSBlockTile :
                (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint32_t remainingKV = (nowS2Idx < p.curKSBlockNum - 1) ? p.curKSBlockTile :
                (p.kvSeqlen - nowS2Idx * p.curKSBlockTile);
            uint64_t singleS2Task = remainingQ * remainingKV;
            resTaskNum -= singleS2Task;
            nowS2Idx += 1;
        }
    }

    void FAInferTiling::ConsumeRemainingBatchesFD(uint32_t groupSize, int64_t &resTaskNum,
        uint32_t &nowBIdx, uint32_t &nowN1Idx, uint32_t &nowS1Idx, uint32_t &nowS2Idx)
    {
        while (nowBIdx < static_cast<uint32_t>(faInfo_.batch) && resTaskNum > 0) {
            BatchParams p = getBatchParams(nowBIdx, groupSize);
            uint32_t remainingQ = p.qSeqlen * (faInfo_.numHeads - p.curQNBlockTile * nowN1Idx) -
                nowS1Idx * p.curQSBlockTile;
            uint32_t remainingKV = p.kvSeqlen;
            uint32_t remainingInBatch = remainingQ * remainingKV;
            if (resTaskNum >= static_cast<int64_t>(remainingInBatch)) {
                resTaskNum -= remainingInBatch;
                nowBIdx++;
                nowN1Idx = 0;
                nowS1Idx = 0;
                nowS2Idx = 0;
            } else {
                break;
            }
        }
    }

    void FAInferTiling::ConsumeRemainingN1GroupsFD(int64_t &resTaskNum, const BatchParams &p,
        uint32_t &nowN1Idx, uint32_t &nowS1Idx, uint32_t &nowS2Idx)
    {
        while (nowN1Idx < p.curQNBlockNum && resTaskNum > 0) {
            uint32_t remainingQ = p.qSeqlen * p.curQNBlockTile - nowS1Idx * p.curQSBlockTile;
            uint32_t remainingInN1 = remainingQ * p.kvSeqlen;
            if (resTaskNum >= static_cast<int64_t>(remainingInN1)) {
                resTaskNum -= remainingInN1;
                nowN1Idx++;
                nowS1Idx = 0;
                nowS2Idx = 0;
            } else {
                break;
            }
        }
    }

    void FAInferTiling::ConsumeRemainingS1GroupsFD(int64_t &resTaskNum, const BatchParams &p,
        uint32_t &nowS1Idx, uint32_t &nowS2Idx)
    {
        while (nowS1Idx < p.curQSBlockNum && resTaskNum > 0) {
            uint32_t remainingQ = (nowS1Idx < p.curQSBlockNum - 1) ?
                p.curQSBlockTile : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint64_t remainingInS1 = remainingQ * p.kvSeqlen;
            if (resTaskNum >= static_cast<int64_t>(remainingInS1)) {
                resTaskNum -= remainingInS1;
                nowS1Idx++;
                nowS2Idx = 0;
            } else {
                break;
            }
        }
    }

    void FAInferTiling::fillCoreInfoForFlashDecode(FAInferTilingData &faTilingData, uint32_t groupSize, uint64_t perCoreTaskNum) {
        uint32_t nowBIdx = 0;
        uint32_t nowN1Idx = 0;
        uint32_t nowS1Idx = 0;
        uint32_t nowS2Idx = 0;

        InitCoreInfoArrays(faTilingData);

        auto finishBatch = [&](uint32_t coreIdx) {
            BatchParams p = getBatchParams(faInfo_.batch - 1, groupSize);
            faTilingData.coreInfo.get_endBIdx()[coreIdx] = faInfo_.batch - 1;
            faTilingData.coreInfo.get_endN1Idx()[coreIdx] = p.curQNBlockNum - 1;
            faTilingData.coreInfo.get_endS1Idx()[coreIdx] = p.curQSBlockNum - 1;
            faTilingData.coreInfo.get_endS2Idx()[coreIdx] = p.curKSBlockNum;
            faTilingData.set_needCoreNum(coreIdx + 1);
        };

        for (uint32_t coreIdx = 0; coreIdx < blockNum_; coreIdx++) {
            int64_t resTaskNum = perCoreTaskNum;
            faTilingData.coreInfo.get_startBIdx()[coreIdx] = nowBIdx;
            faTilingData.coreInfo.get_startN1Idx()[coreIdx] = nowN1Idx;
            faTilingData.coreInfo.get_startS1Idx()[coreIdx] = nowS1Idx;
            faTilingData.coreInfo.get_startS2Idx()[coreIdx] = nowS2Idx;

            BatchParams p = getBatchParams(nowBIdx, groupSize);
            auto advanceCounters = [&]() {
                if (nowS2Idx == p.curKSBlockNum) { nowS1Idx++; nowS2Idx = 0; }
                if (nowS1Idx == p.curQSBlockNum) { nowN1Idx++; nowS1Idx = 0; nowS2Idx = 0; }
                if (nowN1Idx == p.curQNBlockNum) { nowBIdx++; nowN1Idx = 0; nowS1Idx = 0; nowS2Idx = 0; }
            };

            ConsumeS2BlocksFD(groupSize, resTaskNum, nowBIdx, nowS1Idx, nowS2Idx);

            if (resTaskNum <= 0) {
                faTilingData.coreInfo.get_endBIdx()[coreIdx] = nowBIdx;
                faTilingData.coreInfo.get_endN1Idx()[coreIdx] = nowN1Idx;
                faTilingData.coreInfo.get_endS1Idx()[coreIdx] = nowS1Idx;
                faTilingData.coreInfo.get_endS2Idx()[coreIdx] = nowS2Idx;
            }

            advanceCounters();
            if (nowBIdx < static_cast<uint32_t>(faInfo_.batch) && resTaskNum <= 0) continue;
            if (nowBIdx == static_cast<uint32_t>(faInfo_.batch)) { finishBatch(coreIdx); break; }

            ConsumeRemainingBatchesFD(groupSize, resTaskNum, nowBIdx, nowN1Idx, nowS1Idx, nowS2Idx);
            if (nowBIdx == static_cast<uint32_t>(faInfo_.batch)) { finishBatch(coreIdx); break; }
            p = getBatchParams(nowBIdx, groupSize);

            ConsumeRemainingN1GroupsFD(resTaskNum, p, nowN1Idx, nowS1Idx, nowS2Idx);
            advanceCounters();
            if (nowBIdx == static_cast<uint32_t>(faInfo_.batch)) { finishBatch(coreIdx); break; }
            p = getBatchParams(nowBIdx, groupSize);

            ConsumeRemainingS1GroupsFD(resTaskNum, p, nowS1Idx, nowS2Idx);
            advanceCounters();
            if (nowBIdx == static_cast<uint32_t>(faInfo_.batch)) { finishBatch(coreIdx); break; }
            p = getBatchParams(nowBIdx, groupSize);

            ConsumeS2BlocksFD(groupSize, resTaskNum, nowBIdx, nowS1Idx, nowS2Idx);
            if (nowBIdx == static_cast<uint32_t>(faInfo_.batch)) { finishBatch(coreIdx); break; }

            faTilingData.coreInfo.get_endBIdx()[coreIdx] = nowBIdx;
            faTilingData.coreInfo.get_endN1Idx()[coreIdx] = nowN1Idx;
            faTilingData.coreInfo.get_endS1Idx()[coreIdx] = nowS1Idx;
            faTilingData.coreInfo.get_endS2Idx()[coreIdx] = nowS2Idx;

            advanceCounters();
        }
    }

    void FAInferTiling::InitSplitInfoArrays(FAInferTilingData &faTilingData)
    {
        for (uint32_t splitIdx = 0; splitIdx < blockNum_ + 1; splitIdx++) {
            faTilingData.splitInfo.get_batchIdx()[splitIdx] = 0;
            faTilingData.splitInfo.get_headStartIdx()[splitIdx] = 0;
            faTilingData.splitInfo.get_headEndIdx()[splitIdx] = 0;
            faTilingData.splitInfo.get_qStartIdx()[splitIdx] = 0;
            faTilingData.splitInfo.get_qEndIdx()[splitIdx] = 0;
            faTilingData.splitInfo.get_splitNum()[splitIdx] = 0;
            faTilingData.splitInfo.get_lseTaskOffset()[splitIdx] = 0;
            faTilingData.splitInfo.get_oTaskOffset()[splitIdx] = 0;
        }
    }

    void FAInferTiling::ProcessCoreSplitInfo(FAInferTilingData &faTilingData, uint32_t groupSize,
        uint32_t coreIdx, int32_t &splitIdx, int32_t &prevBIdx,
        int32_t &prevN1Idx, int32_t &prevS1Idx,
        int64_t &currentLseTaskOffset, int64_t &currentOTaskOffset)
    {
        int32_t startBIdx = faTilingData.coreInfo.get_startBIdx()[coreIdx];
        int32_t startN1Idx = faTilingData.coreInfo.get_startN1Idx()[coreIdx];
        int32_t startS1Idx = faTilingData.coreInfo.get_startS1Idx()[coreIdx];
        int32_t startS2Idx = faTilingData.coreInfo.get_startS2Idx()[coreIdx];
        int32_t endBIdx = faTilingData.coreInfo.get_endBIdx()[coreIdx];
        int32_t endN1Idx = faTilingData.coreInfo.get_endN1Idx()[coreIdx];
        int32_t endS1Idx = faTilingData.coreInfo.get_endS1Idx()[coreIdx];
        int32_t endS2Idx = faTilingData.coreInfo.get_endS2Idx()[coreIdx];

        faTilingData.coreInfo.get_firstSplitKVTaskLseOffset()[coreIdx] = 0;
        faTilingData.coreInfo.get_firstSplitKVTaskOOffset()[coreIdx] = 0;
        bool foundFirstSplitKV = false;

        for (int32_t BIdx = startBIdx; BIdx <= endBIdx; BIdx++) {
            BatchParams p = getBatchParams(BIdx, groupSize);
            int32_t curStartN1 = (BIdx == startBIdx) ? startN1Idx : 0;
            int32_t curEndN1 = (BIdx == endBIdx) ? endN1Idx : p.curQNBlockNum - 1;

            for (int32_t N1Idx = curStartN1; N1Idx <= curEndN1; N1Idx++) {
                int32_t curStartS1 =
                    (BIdx == startBIdx && N1Idx == startN1Idx) ? startS1Idx : 0;
                int32_t curEndS1 = (BIdx == endBIdx && N1Idx == endN1Idx) ?
                    endS1Idx : p.curQSBlockNum - 1;

                for (int32_t S1Idx = curStartS1; S1Idx <= curEndS1; S1Idx++) {
                    int32_t curStartS2 = (BIdx == startBIdx && N1Idx == startN1Idx && S1Idx == startS1Idx) ?
                        startS2Idx : 0;
                    int32_t curEndS2 = (BIdx == endBIdx && N1Idx == endN1Idx && S1Idx == endS1Idx) ?
                        endS2Idx : p.curKSBlockNum;

                    uint32_t coveredS2 = curEndS2 - curStartS2;
                    bool isSplitKV = (coveredS2 > 0 && coveredS2 < p.curKSBlockNum);
                    if (!isSplitKV) continue;

                    int64_t tmpLseOffset = currentLseTaskOffset;
                    int64_t tmpOOffset = currentOTaskOffset;

                    uint32_t N1IdxPerGroup = N1Idx % p.qNBlockNumPerGroup;
                    uint32_t kvHeadIdx = N1Idx / p.qNBlockNumPerGroup;
                    uint32_t currentHeadStart = kvHeadIdx * groupSize + N1IdxPerGroup * p.curQNBlockTile;
                    uint32_t currentHeadEnd =
                        std::min(currentHeadStart + p.curQNBlockTile, (kvHeadIdx + 1) * groupSize);
                    uint32_t currentQStart = S1Idx * p.curQSBlockTile;
                    uint32_t currentQEnd = std::min(currentQStart + p.curQSBlockTile, p.qSeqlen);
                    uint32_t headLen = currentHeadEnd - currentHeadStart;
                    uint32_t qLen = currentQEnd - currentQStart;

                    if (BIdx != prevBIdx || N1Idx != prevN1Idx || S1Idx != prevS1Idx) {
                        splitIdx++;
                        if (splitIdx >= 0 && splitIdx < (int32_t)(blockNum_ + 1)) {
                            faTilingData.splitInfo.get_batchIdx()[splitIdx] = BIdx;
                            faTilingData.splitInfo.get_splitNum()[splitIdx] = 0;
                            faTilingData.splitInfo.get_headStartIdx()[splitIdx] = currentHeadStart;
                            faTilingData.splitInfo.get_headEndIdx()[splitIdx] = currentHeadEnd;
                            faTilingData.splitInfo.get_qStartIdx()[splitIdx] = currentQStart;
                            faTilingData.splitInfo.get_qEndIdx()[splitIdx] = currentQEnd;
                            faTilingData.splitInfo.get_lseTaskOffset()[splitIdx] = currentLseTaskOffset;
                            faTilingData.splitInfo.get_oTaskOffset()[splitIdx] = currentOTaskOffset;
                        }
                        prevBIdx = BIdx;
                        prevN1Idx = N1Idx;
                        prevS1Idx = S1Idx;
                    }
                    if (splitIdx >= 0 && splitIdx < (int32_t)(blockNum_ + 1)) {
                        faTilingData.splitInfo.get_splitNum()[splitIdx]++;
                        currentLseTaskOffset += (int64_t)headLen * qLen;
                        currentOTaskOffset += (int64_t)headLen * qLen * faInfo_.embeddingSizeV;
                    }
                    if (!foundFirstSplitKV) {
                        foundFirstSplitKV = true;
                        faTilingData.coreInfo.get_firstSplitKVTaskLseOffset()[coreIdx] = tmpLseOffset;
                        faTilingData.coreInfo.get_firstSplitKVTaskOOffset()[coreIdx] = tmpOOffset;
                    }
                }
            }
        }
    }

    void FAInferTiling::fillSplitInfoForFlashDecode(FAInferTilingData &faTilingData, uint32_t groupSize) {
        InitSplitInfoArrays(faTilingData);

        int64_t currentLseTaskOffset = 0;
        int64_t currentOTaskOffset = 0;
        int32_t splitIdx = -1;
        int32_t prevBIdx = -1;
        int32_t prevN1Idx = -1;
        int32_t prevS1Idx = -1;

        for (uint32_t coreIdx = 0; coreIdx < blockNum_; coreIdx++) {
            ProcessCoreSplitInfo(faTilingData, groupSize, coreIdx,
                splitIdx, prevBIdx, prevN1Idx, prevS1Idx,
                currentLseTaskOffset, currentOTaskOffset);
        }

        uint32_t actualSplitNum = (splitIdx + 1 > (int32_t)blockNum_) ? blockNum_ : (splitIdx + 1);
        faTilingData.set_totalSplitNodeNum(actualSplitNum);
        faTilingData.set_splitLseTotalSize(currentLseTaskOffset * SIZE_OF_32BIT);
        faTilingData.set_splitOTotalSize(currentOTaskOffset * SIZE_OF_32BIT);
    }

    void FAInferTiling::splitBN2S1GS2(FAInferTilingData &faTilingData) {
        uint64_t totalTaskNum = 0;
        uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;

        for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; batchIdx++) {
            BatchParams p = getBatchParams(batchIdx, groupSize);
            totalTaskNum += static_cast<uint64_t>(faInfo_.numHeads) * p.qSeqlen * p.kvSeqlen;
        }
        uint64_t perCoreTaskNum = (totalTaskNum + blockNum_ - 1) / blockNum_;
        fillCoreInfoForFlashDecode(faTilingData, groupSize, perCoreTaskNum);
        fillSplitInfoForFlashDecode(faTilingData, groupSize);
    }

    void FAInferTiling::SplitCoreDecodeBS1GN2(FAInferTilingData &faTilingData)
    {
        uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
        uint32_t coreNum = this->blockNum_;

        uint32_t curKvNBlockTile = GetKvNBlockTile(groupSize, faInfo_.kvHeads);
        uint32_t curKvNBlockNum = (faInfo_.kvHeads + curKvNBlockTile - 1) / curKvNBlockTile;
        uint32_t taskNumPerBatch = curKvNBlockNum;

        faTilingData.set_firstBatchTaskNum(taskNumPerBatch);
        uint32_t totalTaskNum = taskNumPerBatch * faInfo_.batch;
        faTilingData.set_totalTaskNum(totalTaskNum);

        uint32_t mainLoopCount = totalTaskNum / coreNum;
        uint32_t mainLoopTaskNum = mainLoopCount * coreNum;
        uint32_t tailLoopTaskNum = totalTaskNum - mainLoopTaskNum;

        uint32_t tailStartBatch = mainLoopTaskNum / taskNumPerBatch;
        uint32_t remainInBatch = mainLoopTaskNum - tailStartBatch * taskNumPerBatch;
        uint32_t tailStartN2 = remainInBatch * curKvNBlockTile;

        uint32_t tailKvNBlockTile = 0;
        uint32_t tailLoopTaskNumRework = tailLoopTaskNum;

        if (tailLoopTaskNum > 0 && tailLoopTaskNum <= coreNum / TAIL_TASK_DIVISOR) {
            tailKvNBlockTile = 1;
            uint32_t remainingKvHeadsFirstBatch = faInfo_.kvHeads - tailStartN2;
            uint32_t remainingFullBatches = faInfo_.batch - tailStartBatch - 1;
            tailLoopTaskNumRework = remainingKvHeadsFirstBatch + remainingFullBatches * faInfo_.kvHeads;
        } else if (tailLoopTaskNum > 0) {
            tailKvNBlockTile = 0;
            tailLoopTaskNumRework = tailLoopTaskNum;
        } else {
            tailKvNBlockTile = 0;
            tailLoopTaskNumRework = 0;
        }

        faTilingData.set_mainLoopTaskNum(mainLoopTaskNum);
        faTilingData.set_tailLoopTaskNum(tailLoopTaskNumRework);
        faTilingData.set_tailStartBatch(tailStartBatch);
        faTilingData.set_tailStartN2(tailStartN2);
        faTilingData.set_tailKvNBlockTile(tailKvNBlockTile);
    }

    ge::graphStatus FAInferTiling::DoTiling(FAInferTilingData &tilingdata)
    {
        FillBasicTilingData(tilingdata);
        if (!faInfo_.isTilingSink) {
            FillSplitCoreTilingData(tilingdata);
            if (faInfo_.flashDecodeFlag) {
                splitBN2S1GS2(tilingdata);
            } else if (faInfo_.decodingFlag) {
                SplitCoreDecodeBS1GN2(tilingdata);
            }
        }
        FillWorkSpaceTilingData(tilingdata);
        return ge::GRAPH_SUCCESS;
    }
}
#endif