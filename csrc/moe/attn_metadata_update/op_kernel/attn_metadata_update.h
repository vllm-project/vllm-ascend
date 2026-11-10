/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef ATTN_METADATA_UPDATE_H
#define ATTN_METADATA_UPDATE_H

#include "kernel_operator.h"
#include "attn_metadata_update_tiling.h"

using namespace AscendC;

namespace AttnMetadataUpdateImpl {

/* UB alignment requirement for AscendC (32 bytes) */
constexpr uint32_t AMU_UB_ALIGN = 32U;
constexpr uint32_t UB_ALIGN_SIZE = 32U;
static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
#define DATA_FLUSH(_gm_tensor, _type) \
    Barrier(); \
    DataCacheCleanAndInvalid<_type, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(_gm_tensor);\
    __asm__("NOP"); \
    dsb(DSB_ALL);
/*
 * AttnMetadataUpdate kernel class.
 *
 * Execution is divided into three stages:
 *
 * Stage 0 — Pre-computation (all cores, pure UB, no GM writes):
 *   Each core independently loads reqIds / discardLabel / discardReqIds[layerIdx]
 *   from GM into its own UB, then computes:
 *     (a) totalValid  — last non-(-1) index + 1 in reqIds.
 *     (b) prefixSum[] — exclusive prefix sum of discardLabel over [0, totalValid).
 *         prefixSum[i] is the compact output index for label==1 slot i.
 *     (c) discardSlotAssign[] — for each label==0 slot in [0, originalBs), the
 *         pre-assigned empty bucket slot index. Computed by scanning the local
 *         discardReqIds copy and marking each allocated slot to prevent intra-core
 *         double-allocation. Because all cores start from the same GM snapshot and
 *         apply the same deterministic scan, every core arrives at identical
 *         assignments — so no cross-core conflict is possible.
 *   SyncAll() after Stage 0 ensures all cores have finished before any GM write.
 *
 * Stage 1 — Current part (all cores parallel, i < originalBs):
 *   label==1 (running req):
 *     Copy positions/slotMapping/seqLens/blockTable[i] to compact output at
 *     prefixSum[i], using UB as intermediate buffer.
 *   label==0 (discarded req):
 *     Write metadata to the pre-assigned bucket slot discardSlotAssign[i].
 *     Update discardReqIds[emptySlot] and discardAttnCnt on GM.
 *   SyncAll() after Stage 1 ensures all bucket writes are visible to Stage 2.
 *
 * Stage 2 — History part (all cores parallel, i >= originalBs):
 *   Reload discardReqIds from GM (now updated by Stage 1).
 *   label==1 (refilled req):
 *     Search discardReqIds for reqIds[i] to find pos_in_bucket.
 *     Copy bucket metadata to compact output at prefixSum[i].
 *     Erase bucket entry (set discardReqIds[pos] = -1, decrement discardAttnCnt).
 *   label==0 (empty slot): skip.
 */
class AttnMetadataUpdate {
public:
    __aicore__ inline AttnMetadataUpdate() {}

    /* Initialise GM pointers and tiling parameters. */
    __aicore__ inline void Init(
        GM_ADDR reqIds,
        GM_ADDR discardLabel,
        GM_ADDR positions,
        GM_ADDR slotMapping,
        GM_ADDR blockTable,
        GM_ADDR seqLens,
        GM_ADDR discardPositions,
        GM_ADDR discardSlotMapping,
        GM_ADDR discardContextLens,
        GM_ADDR discardBlockTables,
        GM_ADDR discardAttnCnt,
        GM_ADDR discardReqIds,
        GM_ADDR originalBs,
        GM_ADDR outPositions,
        GM_ADDR outSlots,
        GM_ADDR outSeqLens,
        GM_ADDR outBlockTable,
        TPipe *pipe,
        const AttnMetadataUpdateTilingData *tilingData)
    {
        /* Tiling parameters */
        maxBs_       = tilingData->info.maxBs;
        
        tableSize_   = tilingData->info.tableSize;
        layerIdx_    = tilingData->info.layerIdx;
        /* maxDiscardBs_: stride for 2-D discard tensors (dim 1 of discard_positions) */
        maxDiscardBs_ = tilingData->info.maxDiscardBs;
        pipe_        = pipe;

        /* Current-batch input tensors */
        reqIdsGM_.SetGlobalBuffer((__gm__ int32_t *)reqIds);
        discardLabelGM_.SetGlobalBuffer((__gm__ int32_t *)discardLabel);
        positionsGM_.SetGlobalBuffer((__gm__ int64_t *)positions);
        slotMappingGM_.SetGlobalBuffer((__gm__ int32_t *)slotMapping);
        blockTableGM_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        seqLensGM_.SetGlobalBuffer((__gm__ int32_t *)seqLens);
        originalGM_.SetGlobalBuffer((__gm__ int32_t *)originalBs);

        /*
         * Per-layer discard bucket tensors.
         * 2-D tensors (layer_num x maxDiscardBs) are offset to the current
         * layer's row: base + layerIdx_ * maxDiscardBs_.
         */
        discardPositionsGM_.SetGlobalBuffer(
            (__gm__ int64_t *)discardPositions + (int64_t)layerIdx_ * maxDiscardBs_);
        discardSlotMappingGM_.SetGlobalBuffer(
            (__gm__ int32_t *)discardSlotMapping + (int64_t)layerIdx_ * maxDiscardBs_);
        discardContextLensGM_.SetGlobalBuffer(
            (__gm__ int32_t *)discardContextLens + (int64_t)layerIdx_ * maxDiscardBs_);
        discardBlockTablesGM_.SetGlobalBuffer(
            (__gm__ int32_t *)discardBlockTables +
            (int64_t)layerIdx_ * maxDiscardBs_ * tableSize_);

        /*
         * discardAttnCnt: shape (layer_num,), int32.
         * Current number of valid entries in the bucket for this layer.
         * Offset to the current layer's element.
         */
        discardAttnCntGM_.SetGlobalBuffer(
            (__gm__ int32_t *)discardAttnCnt + (int64_t)layerIdx_);

        /*
         * discardReqIds: shape (layer_num, maxDiscardBs), int32.
         * req_id of each bucket entry; unused entries are -1.
         * Offset to the current layer's row.
         */
        discardReqIdsGM_.SetGlobalBuffer(
            (__gm__ int32_t *)discardReqIds + (int64_t)layerIdx_ * maxDiscardBs_);

        /* Output tensors (compact / dense) */
        outPositionsGM_.SetGlobalBuffer((__gm__ int64_t *)outPositions);
        outSlotsGM_.SetGlobalBuffer((__gm__ int32_t *)outSlots);
        outSeqLensGM_.SetGlobalBuffer((__gm__ int32_t *)outSeqLens);
        outBlockTableGM_.SetGlobalBuffer((__gm__ int32_t *)outBlockTable);
    }

    template <typename T>
    __aicore__ inline void CpGMtoUB(LocalTensor<T> outputLt, GlobalTensor<T> inputGt, uint32_t size)
    {
        DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
        DataCopyPadExtParams<T> padParams;
        DataCopyPad(outputLt, inputGt, dataCopyParams, padParams);
    }

    template <typename T>
    __aicore__ inline void CpUBtoGM(GlobalTensor<T> outputGt, LocalTensor<T> inputLt, uint16_t count, uint32_t size,
                                        uint32_t srcStride = 0, uint32_t dstStride = 0)
    {
        DataCopyExtParams dataCopyParams(count, size, srcStride / UB_ALIGN_SIZE, dstStride, 0);
        DataCopyPad(outputGt, inputLt, dataCopyParams);
    }

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}
    __aicore__ inline void Process()
    {
        /* ================================================================== *
         * Upfront setup — all values that do NOT depend on GM-loaded data.
         *
         * Buffer sizes derived from tiling parameters (set in Init):
         *   reqIdsBufLen, discardReqIdsBufLen, slotAssignBufLen, blkLen.
         * All corresponding InitBuffers are called here so that AllocTensor
         * can be called later without interleaving with DataCopy sequences.
         *
         * Core-partitioning scalars (hardware intrinsics, no GM dependency):
         *   numCores, coreIdx, chunkCur, startCur, endCur.
         *
         * NOTE: prefixSumBufLen / InitBuffer(prefixSumBuf_) must stay after
         * totalValid is computed (totalValid depends on GM-loaded reqIds).
         * ================================================================== */

        /* Buffer sizes from tiling params */
        DATA_FLUSH(originalGM_, int32_t);
        originalBs_   = originalGM_.GetValue(0);
       // PRINTF("originalBs_=%d",originalBs_);
        pipe_barrier(PIPE_ALL);
       // SyncFunc<AscendC::HardEvent::MTE2_S>();
        const uint32_t reqIdsBufLen =
            (maxBs_ * sizeof(int32_t) + AMU_UB_ALIGN - 1U) / AMU_UB_ALIGN * AMU_UB_ALIGN;
        const uint32_t discardReqIdsBufLen =
            (maxDiscardBs_ * sizeof(int32_t) + AMU_UB_ALIGN - 1U) / AMU_UB_ALIGN * AMU_UB_ALIGN;
        const uint32_t slotAssignBufLen =
            (originalBs_ * sizeof(int32_t) + AMU_UB_ALIGN - 1U) / AMU_UB_ALIGN * AMU_UB_ALIGN;
        /* blkLen is the same on every loop iteration; compute once and hoist. */
        const uint32_t blkLen =
            (tableSize_ * sizeof(int32_t) + AMU_UB_ALIGN - 1U) / AMU_UB_ALIGN * AMU_UB_ALIGN;

        /* InitBuffers that have fixed / tiling-param-derived sizes */
        pipe_->InitBuffer(reqIdsBuf_,            reqIdsBufLen);
        pipe_->InitBuffer(discardLabelBuf_,      reqIdsBufLen);
        pipe_->InitBuffer(discardReqIdsBuf_,     discardReqIdsBufLen);
        pipe_->InitBuffer(discardSlotAssignBuf_, slotAssignBufLen);
        pipe_->InitBuffer(tmpSlotI32Buf_,            AMU_UB_ALIGN);
        pipe_->InitBuffer(tmpSeqLenI32Buf_,            AMU_UB_ALIGN);
        pipe_->InitBuffer(tmpPositionI64Buf_,            AMU_UB_ALIGN);
        pipe_->InitBuffer(tmpBlkBuf_,            blkLen);
        pipe_->InitBuffer(tmpReqIdI32Buf_,            AMU_UB_ALIGN);
        pipe_->InitBuffer(tmpEraseBuf_,            AMU_UB_ALIGN);
        pipe_->InitBuffer(cleanBuf_,            BUFFER_SIZE_BYTE_8K);
        

        /* Core-partitioning scalars (current-part range; history range computed
         * after totalValid is known) */
        const uint32_t numCores = GetBlockNum();
        const uint32_t coreIdx  = GetBlockIdx();
        const uint32_t chunkCur = (originalBs_ + numCores - 1U) / numCores;
        const uint32_t startCur = coreIdx * chunkCur;
        const uint32_t endCur   = startCur + chunkCur < originalBs_
                                  ? startCur + chunkCur : originalBs_;
        // PRINTF("current core is %d, start req %d, end req %d (excluded)", coreIdx, startCur, endCur);
        /* ================================================================== *
         * Stage 0: Pre-computation — pure UB, no GM writes.
         *
         * Each core independently performs the same deterministic computation
         * on the same read-only GM data, so all cores reach identical results.
         * ================================================================== */

        /* --- Load reqIds and discardLabel into UB --- */
        // LocalTensor<int32_t> reqIdsLocal       = reqIdsBuf_.AllocTensor<int32_t>();
        // LocalTensor<int32_t> discardLabelLocal = discardLabelBuf_.AllocTensor<int32_t>();
        LocalTensor<int32_t> reqIdsLocal       = reqIdsBuf_.Get<int32_t>();
        LocalTensor<int32_t> discardLabelLocal = discardLabelBuf_.Get<int32_t>();
        LocalTensor<int32_t> cleanLt = cleanBuf_.Get<int32_t>();
       // Duplicate(cleanLt, static_cast<int32_t>(0), maxBs_ * 2);
        LocalTensor<int64_t> cleanLt2 = cleanBuf_.Get<int64_t>();
        Duplicate(cleanLt, static_cast<int32_t>(-1), maxBs_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        CpUBtoGM(outSlotsGM_, cleanLt, 1, maxBs_ * sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Duplicate(cleanLt, static_cast<int32_t>(0), maxBs_ * 2);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        CpUBtoGM(outSeqLensGM_, cleanLt, 1, maxBs_ * sizeof(int32_t));
        CpUBtoGM(outBlockTableGM_, cleanLt, 1, maxBs_ * sizeof(int32_t));
        CpUBtoGM(outPositionsGM_, cleanLt2, 1, maxBs_ * sizeof(int64_t));

        /* Issue both loads simultaneously, sync with two event IDs. */
        CpGMtoUB(reqIdsLocal,       reqIdsGM_,       maxBs_ * sizeof (int32_t));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        CpGMtoUB(discardLabelLocal, discardLabelGM_, maxBs_ * sizeof (int32_t));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        // if (coreIdx == 0) {
        //     AscendC::DumpTensor(reqIdsLocal, 11111, maxBs_);
        //     AscendC::DumpTensor(discardLabelLocal, 22222, maxBs_);
        // }

        /* --- Load discardReqIds[layerIdx] into UB (initial snapshot) --- */
        // LocalTensor<int32_t> discardReqIdsLocal = discardReqIdsBuf_.AllocTensor<int32_t>();
        LocalTensor<int32_t> discardReqIdsLocal = discardReqIdsBuf_.Get<int32_t>();
        CpGMtoUB(discardReqIdsLocal, discardReqIdsGM_, maxDiscardBs_ * sizeof (int32_t));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);

        /* --- Determine totalValid ---
         * Scan reqIds from the tail; entries with value -1 are padding.
         * Depends on GM-loaded reqIdsLocal — must stay here. */
        uint32_t totalValid = maxBs_;
        while (totalValid > 0U && reqIdsLocal.GetValue(totalValid - 1U) == -1) {
            --totalValid;
        }

        /* --- Compute prefixSum[0..totalValid] ---
         * prefixSum[i] = number of label==1 slots in [0, i).
         * Buffer size depends on totalValid (GM-derived) — InitBuffer stays here. */
        const uint32_t prefixSumBufLen =
            ((totalValid + 1U) * sizeof(int32_t) + AMU_UB_ALIGN - 1U) / AMU_UB_ALIGN * AMU_UB_ALIGN;
        pipe_->InitBuffer(prefixSumBuf_, prefixSumBufLen);
        // LocalTensor<int32_t> prefixSumLocal = prefixSumBuf_.AllocTensor<int32_t>();
        LocalTensor<int32_t> prefixSumLocal = prefixSumBuf_.Get<int32_t>();

        uint32_t acc = 0U;
        prefixSumLocal.SetValue(0U, static_cast<int32_t>(acc));
        for (uint32_t i = 0U; i < totalValid; ++i) {
            if (discardLabelLocal.GetValue(i) == 1) {
                ++acc;
            }
            prefixSumLocal.SetValue(i + 1U, static_cast<int32_t>(acc));
        }

        /* --- Compute discardSlotAssign[0..originalBs-1] ---
         * For each label==0 slot in [0, originalBs), pre-assign an empty bucket
         * slot by scanning the local discardReqIds snapshot.  Each allocated slot
         * is immediately marked in the local copy (set to any non-(-1) value) so
         * that subsequent label==0 entries in the same scan do not reuse it.
         *
         * Because every core starts from the same GM snapshot and applies the
         * same deterministic first-fit scan, all cores produce identical
         * assignments — eliminating cross-core write conflicts in Stage 1. */
        // LocalTensor<int32_t> discardSlotAssignLocal = discardSlotAssignBuf_.AllocTensor<int32_t>();
        LocalTensor<int32_t> discardSlotAssignLocal = discardSlotAssignBuf_.Get<int32_t>();

        for (uint32_t i = 0U; i < originalBs_; ++i) {
            if (reqIdsLocal.GetValue(i) == -1 || discardLabelLocal.GetValue(i) != 0) {
                /* padding or running req — no bucket slot needed */
                discardSlotAssignLocal.SetValue(i, -1);
                continue;
            }

            /* Find the first empty slot (value == -1) in the local snapshot. */
            int32_t emptySlot = -1;
            for (uint32_t j = 0U; j < maxDiscardBs_; ++j) {
                if (discardReqIdsLocal.GetValue(j) == -1) {
                    emptySlot = static_cast<int32_t>(j);
                    break;
                }
            }

            discardSlotAssignLocal.SetValue(i, emptySlot);

            if (emptySlot >= 0) {
                /* Mark slot as occupied in the local copy so the next label==0
                 * entry in this scan does not pick the same slot. */
                discardReqIdsLocal.SetValue(static_cast<uint32_t>(emptySlot),
                                            reqIdsLocal.GetValue(i));
            }
        }

        /* All cores must finish Stage 0 before any GM write begins. */
        SyncAll();

        /* ================================================================== *
         * Stage 1: Current part (i < originalBs), all cores parallel.
         * ================================================================== */
        // if (coreIdx == 0)
        // {
        //     AscendC::DumpTensor(positionsGM_, 77777, maxBs_);
        //     AscendC::DumpTensor(slotMappingGM_, 88888, maxBs_);
        //     AscendC::DumpTensor(seqLensGM_, 99999, maxBs_);
        // }
        LocalTensor<int64_t> tmpPositionLt = tmpPositionI64Buf_.Get<int64_t>();
        LocalTensor<int32_t> tmpSlotLt = tmpSlotI32Buf_.Get<int32_t>();
        LocalTensor<int32_t> tmpSeqLenLt = tmpSeqLenI32Buf_.Get<int32_t>();
        LocalTensor<int32_t> tmpBlk = tmpBlkBuf_.Get<int32_t>();
        LocalTensor<int32_t> tmpReqIdLt = tmpReqIdI32Buf_.Get<int32_t>();
        LocalTensor<int32_t> tmpEraseLt = tmpEraseBuf_.Get<int32_t>();

        for (uint32_t i = startCur; i < endCur; ++i) {
            if (reqIdsLocal.GetValue(i) == -1) {
                continue; /* padding, skip */
            }

            if (discardLabelLocal.GetValue(i) == 1) {
                /* ---------------------------------------------------------- *
                 * Running req: copy metadata to compact output.
                 * All GM transfers go through a UB intermediate buffer.
                 * ---------------------------------------------------------- */
                uint32_t outIdx = static_cast<uint32_t>(prefixSumLocal.GetValue(i));

                /* Overlap positions (int64) and slotMapping (int32) loads —
                 * they use different UB buffers so MTE2 can run in parallel. */
                
                /* seqLens[i] → outSeqLens[outIdx]: reuse tmpI32 after MTE3 drains */


                // DataCopy(tmpI32, slotMappingGM_[i], 1U);
                CpGMtoUB(tmpSlotLt, slotMappingGM_[i], sizeof(int32_t));
                SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
                AscendC::PipeBarrier<PIPE_MTE2>();

                // PRINTF("before datacopy, core %d, outIdx %d, position lt %lld, slot lt %d\n", coreIdx, outIdx, tmpPositionLt.GetValue(0), tmpSlotLt.GetValue(0));
                // DataCopy(tmpI64, positionsGM_[i], 1U);
                CpGMtoUB(tmpPositionLt, positionsGM_[i], sizeof(int64_t));
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

                /* slotMapping[i] → outSlots[outIdx] */
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
                // DataCopy(outSlotsGM_[outIdx], tmpI32, 1U);
                CpUBtoGM(outSlotsGM_[outIdx], tmpSlotLt, 1, sizeof(int32_t));
                // PRINTF("after datacopy, core %d, outIdx %d, position lt %lld, slot lt %d\n", coreIdx, outIdx, tmpPositionLt.GetValue(0), tmpSlotLt.GetValue(0));

                // PRINTF("core idx %d valid, outIdx %d, req %d, position %lld, seqlens %d\n", 
                //     coreIdx, outIdx, reqIdsLocal(i), tmpPositionLt(0), tmpSeqLenLt(0));
                /* blockTable[i*tableSize..] → outBlockTable[outIdx*tableSize..] */

                CpGMtoUB(tmpSeqLenLt, seqLensGM_[i], sizeof(int32_t));
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
                CpUBtoGM(outSeqLensGM_[outIdx], tmpSeqLenLt, 1U, sizeof(int32_t));

                CpGMtoUB(tmpBlk, blockTableGM_[i * tableSize_], tableSize_ * sizeof(int32_t));
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
                CpUBtoGM(outBlockTableGM_[outIdx * tableSize_], tmpBlk, 1, tableSize_ * sizeof(int32_t));

                /* positions[i] → outPositions[outIdx] */
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                // DataCopy(outPositionsGM_[outIdx], tmpI64, 1U);
                CpUBtoGM(outPositionsGM_[outIdx], tmpPositionLt, 1, sizeof(int64_t));
                
                

                
                // if (coreIdx == 0) {
                //     SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
                //     WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
                //     PRINTF("core %d dump current table info, table size %d\n", coreIdx, tableSize_);
                //     AscendC::DumpTensor(tmpBlk, 141414, tableSize_);
                //     uint32_t array[] = {static_cast<uint32_t>(maxBs_), static_cast<uint32_t>(tableSize_)};
                //     AscendC::ShapeInfo shapeInfo(2, array);
                //     AscendC::DumpTensor(outBlockTableGM_, 151515, maxBs_ * tableSize_, shapeInfo);
                // }

            } else {
                /* ---------------------------------------------------------- *
                 * Discarded req: write metadata into the pre-assigned bucket
                 * slot.  The slot index was computed in Stage 0 and is
                 * guaranteed to be unique across all cores.
                 * ---------------------------------------------------------- */
                int32_t assignedSlot = discardSlotAssignLocal.GetValue(i);
                if (assignedSlot < 0) {
                    continue; /* bucket full (should not happen) */
                }
                uint32_t emptySlot = static_cast<uint32_t>(assignedSlot);

                
                CpGMtoUB(tmpSlotLt, slotMappingGM_[i], sizeof(int32_t));
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);



                /* slotMapping[i] → discardSlotMapping[layerIdx][emptySlot] */
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
                CpUBtoGM(discardSlotMappingGM_[emptySlot], tmpSlotLt, 1, sizeof(int32_t));

                /* seqLens[i] → discardContextLens[layerIdx][emptySlot]: reuse tmpI32 */
                
                CpGMtoUB(tmpSeqLenLt, seqLensGM_[i], sizeof(int32_t));
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
                CpUBtoGM(discardContextLensGM_[emptySlot], tmpSeqLenLt, 1, sizeof(int32_t));

                /* blockTable[i*tableSize..] → discardBlockTables[layerIdx][emptySlot*tableSize..] */
                CpGMtoUB(tmpBlk, blockTableGM_[i * tableSize_], tableSize_ * sizeof(int32_t));
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
                CpUBtoGM(discardBlockTablesGM_[emptySlot * tableSize_], tmpBlk, 1, tableSize_ * sizeof(int32_t));

                /* Write req_id to discardReqIds[layerIdx][emptySlot]: S → MTE3 */
                /* Overlap positions (int64) and slotMapping (int32) loads. */
                CpGMtoUB(tmpPositionLt, positionsGM_[i], sizeof(int64_t));
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);

                tmpReqIdLt.SetValue(0U, reqIdsLocal.GetValue(i));
                SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
                CpUBtoGM(discardReqIdsGM_[emptySlot], tmpReqIdLt, 1, sizeof(int32_t));
                /* discardAttnCnt is NOT updated here; core 0 recomputes it
                 * after Stage 2 by counting non-(-1) entries in discardReqIds. */
                                 /* positions[i] → discardPositions[layerIdx][emptySlot] */
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                CpUBtoGM(discardPositionsGM_[emptySlot], tmpPositionLt, 1, sizeof(int64_t));
            }
	    SyncFunc<AscendC::HardEvent::MTE3_S>();
        }

        /* All Stage-1 bucket writes must be globally visible before Stage 2. */
        SyncAll();
        // if (coreIdx == 0) {
        //     PRINTF("core %d dump current info\n", coreIdx);
        //     AscendC::DumpTensor(outPositionsGM_, 33333, maxBs_);
        //     AscendC::DumpTensor(outSlotsGM_, 44444, maxBs_);
        //     AscendC::DumpTensor(outSeqLensGM_, 55555, maxBs_);
        //     uint32_t array[] = {static_cast<uint32_t>(maxBs_), static_cast<uint32_t>(tableSize_)};
        //     AscendC::ShapeInfo shapeInfo(2, array);
        //     AscendC::DumpTensor(outBlockTableGM_, 666666, maxBs_ * tableSize_, shapeInfo);
        // }

        /* ================================================================== *
         * Stage 2: History part (i >= originalBs), all cores parallel.
         *
         * Reload discardReqIds from GM — it has been updated by Stage 1.
         * ================================================================== */
        CpGMtoUB(discardReqIdsLocal, discardReqIdsGM_, maxDiscardBs_ * sizeof(int32_t));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        uint32_t histTotal = totalValid > originalBs_ ? totalValid - originalBs_ : 0U;
        uint32_t chunkHist = histTotal > 0U ? (histTotal + numCores - 1U) / numCores : 0U;
        uint32_t startHist = originalBs_ + coreIdx * chunkHist;
        uint32_t endHist   = startHist + chunkHist < totalValid ? startHist + chunkHist : totalValid;

        for (uint32_t i = startHist; i < endHist; ++i) {
            if (reqIdsLocal.GetValue(i) == -1) {
                continue; /* padding, skip */
            }
            if (discardLabelLocal.GetValue(i) != 1) {
                continue; /* empty history slot, skip */
            }

            /* Find pos_in_bucket by searching discardReqIds for reqIds[i]. */
            int32_t  targetReqId = reqIdsLocal.GetValue(i);
            uint32_t posInBucket = maxDiscardBs_; /* sentinel: not found */
            for (uint32_t j = 0U; j < maxDiscardBs_; ++j) {
                if (discardReqIdsLocal.GetValue(j) == targetReqId) {
                    posInBucket = j;
                    break;
                }
            }
            if (posInBucket >= maxDiscardBs_) {
                continue; /* req_id not found, should not happen */
            }

            uint32_t outIdx = static_cast<uint32_t>(prefixSumLocal.GetValue(i));

            CpGMtoUB(tmpPositionLt, discardPositionsGM_[posInBucket], sizeof(int64_t));
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            CpGMtoUB(tmpSlotLt, discardSlotMappingGM_[posInBucket], sizeof(int32_t));
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);

            /* discardPositions[layerIdx][posInBucket] → outPositions[outIdx] */
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            CpUBtoGM(outPositionsGM_[outIdx], tmpPositionLt, 1U, sizeof(int64_t));

            /* discardSlotMapping[layerIdx][posInBucket] → outSlots[outIdx] */
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
            CpUBtoGM(outSlotsGM_[outIdx], tmpSlotLt, 1, sizeof(int32_t));

            /* discardContextLens[layerIdx][posInBucket] → outSeqLens[outIdx]: reuse tmpI32 */
            CpGMtoUB(tmpSeqLenLt, discardContextLensGM_[posInBucket], sizeof(int32_t));
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
            CpUBtoGM(outSeqLensGM_[outIdx], tmpSeqLenLt, 1U, sizeof(int32_t));

            /* discardBlockTables[layerIdx][posInBucket*tableSize..] → outBlockTable[outIdx*tableSize..] */
            // LocalTensor<int32_t> tmpBlk = tmpBlkBuf_.AllocTensor<int32_t>();
            CpGMtoUB(tmpBlk, discardBlockTablesGM_[posInBucket * tableSize_], tableSize_ * sizeof(int32_t));
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
            CpUBtoGM(outBlockTableGM_[outIdx * tableSize_], tmpBlk, 1, tableSize_ * sizeof(int32_t));

            /* Erase bucket entry: discardReqIds[layerIdx][posInBucket] = -1 (S → MTE3) */
            // LocalTensor<int32_t> tmpErase = tmpI32Buf_.AllocTensor<int32_t>();
            
            tmpEraseLt.SetValue(0U, -1);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
            CpUBtoGM(discardReqIdsGM_[posInBucket], tmpEraseLt, 1, sizeof(int32_t));
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            /* discardAttnCnt is NOT decremented here; core 0 recomputes it below. */
        }

        /* All Stage-2 bucket erasures must be globally visible before the
         * counter recomputation below. */
        SyncAll();

        /* ================================================================== *
         * Counter recomputation — core 0 only, serial.
         *
         * After Stage 1 (writes) and Stage 2 (erasures), the true count of
         * valid bucket entries equals the number of non-(-1) values in
         * discardReqIds[layerIdx].  Core 0 reloads the array and counts them,
         * then writes the result back to discardAttnCnt[layerIdx].
         *
         * All other cores skip this section.
         * ================================================================== */
        if (coreIdx == 0U) {
            /* Reload discardReqIds — Stage 2 may have erased entries. */
            CpGMtoUB(discardReqIdsLocal, discardReqIdsGM_, maxDiscardBs_ * sizeof(int32_t));
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

            uint32_t validCnt = 0U;
            for (uint32_t j = 0U; j < maxDiscardBs_; ++j) {
                if (discardReqIdsLocal.GetValue(j) != -1) {
                    ++validCnt;
                }
            }
            // PRINTF("discard cnt %d\n", validCnt);
            
            tmpEraseLt.SetValue(0U, static_cast<int32_t>(validCnt));
            SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
            CpUBtoGM(discardAttnCntGM_, tmpEraseLt, 1, sizeof(int32_t));
        }
    }

private:
    /* Current-batch input GM tensors */
    GlobalTensor<int32_t> reqIdsGM_;
    GlobalTensor<int32_t> discardLabelGM_;
    GlobalTensor<int64_t> positionsGM_;
    GlobalTensor<int32_t> slotMappingGM_;
    GlobalTensor<int32_t> blockTableGM_;
    GlobalTensor<int32_t> seqLensGM_;

    /* Per-layer discard bucket GM tensors (already offset to layerIdx row) */
    GlobalTensor<int64_t> discardPositionsGM_;
    GlobalTensor<int32_t> discardSlotMappingGM_;
    GlobalTensor<int32_t> discardContextLensGM_;
    GlobalTensor<int32_t> discardBlockTablesGM_;
    GlobalTensor<int32_t> discardAttnCntGM_;  /* single element: count for this layer */
    GlobalTensor<int32_t> discardReqIdsGM_;   /* req_id array of the bucket */

    /* Output GM tensors (compact / dense) */
    GlobalTensor<int64_t> outPositionsGM_;
    GlobalTensor<int32_t> outSlotsGM_;
    GlobalTensor<int32_t> outSeqLensGM_;
    GlobalTensor<int32_t> outBlockTableGM_;
    GlobalTensor<int32_t> originalGM_;

    /* UB buffers */
    TBuf<> reqIdsBuf_;
    TBuf<> discardLabelBuf_;
    TBuf<> discardReqIdsBuf_;
    TBuf<> prefixSumBuf_;
    TBuf<> discardSlotAssignBuf_;
    TBuf<> tmpSlotI32Buf_;
    TBuf<> tmpSeqLenI32Buf_;
    TBuf<> tmpPositionI64Buf_;
    TBuf<> tmpBlkBuf_;
    TBuf<> tmpReqIdI32Buf_;
    TBuf<> tmpEraseBuf_;
    TBuf<> cleanBuf_;

    TPipe *pipe_{nullptr};

    uint32_t maxBs_{0};
    uint32_t originalBs_{0};
    uint32_t tableSize_{0};
    uint32_t layerIdx_{0};
    uint32_t maxDiscardBs_{0};  /* stride for 2-D bucket tensors (max entries per layer) */
};

}  // namespace AttnMetadataUpdateImpl

#endif  // ATTN_METADATA_UPDATE_H
