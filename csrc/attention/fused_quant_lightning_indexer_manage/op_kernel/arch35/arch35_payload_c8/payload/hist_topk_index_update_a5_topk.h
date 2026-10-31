/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the CANN Open Software License Agreement Version 2.0.
 */

/*!
 * \file hist_topk_index_update_a5_topk.h
 * \brief LITopk wrapper for hist_topk_index_update_a5 (vectorized classify)
 *
 * Payload-aware LITopk wrapper.  The shared temporary layout keeps two
 * survivor-payload buffers, histogram scratch, and tmp indices in one
 * contiguous UB region so the final TopK survivors can carry hit/miss payload.
 */
#ifndef VLLM_FAST_HIST_TOPK_INDEX_UPDATE_A5_TOPK_H
#define VLLM_FAST_HIST_TOPK_INDEX_UPDATE_A5_TOPK_H

#include "kernel_operator.h"
#include "hist_topk_index_update_a5_common.h"
#include "li_update_ablation_config.h"
#include "vf_topk_16_gather.h"
#include "lim_normalize_pool_slots_vf.h"
#include "../../lim_payload_layout.h"

namespace vllm_fast_hist_topk_index_update_a5_payload {

using namespace VllmFastHistTopkIndexUpdateA5Common;
using namespace VllmFastTopkb16gather;

static_assert(LI_UPDATE_ABLATION_MODE >= 0 && LI_UPDATE_ABLATION_MODE <= 5,
              "LI_UPDATE_ABLATION_MODE must be in [0, 5]");

static constexpr uint32_t INDEXER_PAYLOAD_TOKEN_BITS = lim_payload_layout::SOURCE_BITS;
static constexpr uint32_t INDEXER_PAYLOAD_SLOT_SHIFT = INDEXER_PAYLOAD_TOKEN_BITS;
static constexpr uint32_t INDEXER_PAYLOAD_TOKEN_MASK = lim_payload_layout::SOURCE_MASK;
static constexpr uint32_t INDEXER_PAYLOAD_SLOT_MASK = lim_payload_layout::SLOT_MASK;
static constexpr uint32_t INDEXER_PAYLOAD_MISS_SLOT = INDEXER_PAYLOAD_SLOT_MASK;
static constexpr uint32_t TOPK_PAYLOAD_V_MTE2_EVENT = EVENT_ID4;
static constexpr uint32_t TOPK_PAYLOAD_MTE2_V_EVENT = EVENT_ID5;

__aicore__ inline uint32_t EncodeIndexerPayload(uint32_t tokenId, int32_t slotId)
{
    uint32_t slotBits = slotId >= 0
        ? (static_cast<uint32_t>(slotId) & INDEXER_PAYLOAD_SLOT_MASK)
        : INDEXER_PAYLOAD_MISS_SLOT;
    return (tokenId & INDEXER_PAYLOAD_TOKEN_MASK) |
           (slotBits << INDEXER_PAYLOAD_SLOT_SHIFT);
}

__aicore__ inline void LoadCurrentSlotsCompactRange(
    const LocalTensor<uint16_t>& currentSlotsLocal,
    const LocalTensor<int32_t>& slotStageLocal,
    GlobalTensor<int32_t>& cachedSlotsGm,
    uint64_t currentSlotOffset,
    uint32_t startOffset,
    uint32_t validLen,
    uint32_t stageLen)
{
    LocalTensor<int16_t> signedSlotsLocal = currentSlotsLocal.template ReinterpretCast<int16_t>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};

    for (uint32_t offset = startOffset; offset < validLen; offset += stageLen) {
        uint32_t count = (offset + stageLen > validLen) ? validLen - offset : stageLen;
        SetFlag<HardEvent::V_MTE2>(TOPK_PAYLOAD_V_MTE2_EVENT);
        WaitFlag<HardEvent::V_MTE2>(TOPK_PAYLOAD_V_MTE2_EVENT);
        copyParams.blockLen = count * sizeof(int32_t);
        AscendC::DataCopyPad(slotStageLocal, cachedSlotsGm[currentSlotOffset + offset],
                             copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(TOPK_PAYLOAD_MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(TOPK_PAYLOAD_MTE2_V_EVENT);
        asc_vf_call<lim_normalize_pool_slots_vf::Normalize>(
            (__ubuf__ uint32_t *)slotStageLocal.GetPhyAddr(),
            (count + 63U) / 64U);
        PipeBarrier<PIPE_V>();
        Cast(signedSlotsLocal[offset], slotStageLocal, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void LoadCurrentSlotsCompact(
    const LocalTensor<uint16_t>& currentSlotsLocal,
    const LocalTensor<int32_t>& slotStageLocal,
    GlobalTensor<int32_t>& cachedSlotsGm,
    uint64_t currentSlotOffset,
    uint32_t validLen,
    uint32_t stageLen)
{
    LoadCurrentSlotsCompactRange(
        currentSlotsLocal, slotStageLocal, cachedSlotsGm,
        currentSlotOffset, 0, validLen, stageLen);
}

__aicore__ inline void StartCurrentSlotsPrefetch(
    const LocalTensor<int32_t>& slotPrefetchLocal,
    GlobalTensor<int32_t>& cachedSlotsGm,
    uint64_t currentSlotOffset,
    uint32_t validLen)
{
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = validLen * sizeof(int32_t);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};

    // The dedicated prefetch scratch is reused by consecutive trunks and
    // requests. Ensure the preceding vector read has retired before MTE2
    // overwrites it with the current trunk's INT32 slots.
    SetFlag<HardEvent::V_MTE2>(TOPK_PAYLOAD_V_MTE2_EVENT);
    WaitFlag<HardEvent::V_MTE2>(TOPK_PAYLOAD_V_MTE2_EVENT);
    AscendC::DataCopyPad(
        slotPrefetchLocal, cachedSlotsGm[currentSlotOffset],
        copyParams, padParams);
}

__aicore__ inline void FinishCurrentSlotsPrefetch(
    const LocalTensor<uint16_t>& currentSlotsLocal,
    const LocalTensor<int32_t>& slotPrefetchLocal,
    uint32_t validLen)
{
    // The MTE2 copy above was deliberately left in flight while LiTopKVF
    // occupied PIPE_V. Wait only when the slot data is actually consumed.
    SetFlag<HardEvent::MTE2_V>(TOPK_PAYLOAD_MTE2_V_EVENT);
    WaitFlag<HardEvent::MTE2_V>(TOPK_PAYLOAD_MTE2_V_EVENT);
    asc_vf_call<lim_normalize_pool_slots_vf::Normalize>(
        (__ubuf__ uint32_t *)slotPrefetchLocal.GetPhyAddr(),
        (validLen + 63U) / 64U);
    PipeBarrier<PIPE_V>();
    LocalTensor<int16_t> signedSlotsLocal =
        currentSlotsLocal.template ReinterpretCast<int16_t>();
    Cast(signedSlotsLocal, slotPrefetchLocal, RoundMode::CAST_NONE, validLen);
    PipeBarrier<PIPE_V>();
}

template<typename T>
class LITopk {
public:
    __aicore__ inline LITopk() {}

    __aicore__ inline void Init(uint32_t topkTarget, uint32_t topkOut, uint32_t trunkLen)
    {
        topkTarget_ = topkTarget;
        topkOut_ = topkOut;
        trunkLen_ = trunkLen;
    }

    __aicore__ inline uint64_t GetSharedTmpBufferSize()
    {
        // Layout: survivorPayloadLocal[2] + histograms + idxHigh + idxLow + nkValue + tmpIndex(uint16).
        // Uses topkOut_ for buffer sizing and topkTarget_ for K search.
        uint64_t bufferSize1 =
            (2 * VllmFastHistTopkIndexUpdateA5Common::Align(topkOut_, (uint32_t)256) + 3 * 256 + 64) * sizeof(uint32_t);
        uint64_t bufferSize2 =
            (VllmFastHistTopkIndexUpdateA5Common::Align(topkOut_, (uint32_t)256) + trunkLen_) * sizeof(uint16_t);
        return bufferSize1 + bufferSize2;
    }

    __aicore__ inline void InitBuffers(const LocalTensor<uint32_t>& sharedTmp)
    {
        LocalTensor<uint32_t> survivorPayloadLocal1 = sharedTmp[0];
        LocalTensor<uint32_t> survivorPayloadLocal2 =
            survivorPayloadLocal1[VllmFastHistTopkIndexUpdateA5Common::Align(topkOut_, (uint32_t)256)];
        survivorPayloadLocal_[0] = survivorPayloadLocal1;
        survivorPayloadLocal_[1] = survivorPayloadLocal2;
        histogramsLocal_ =
            survivorPayloadLocal2[VllmFastHistTopkIndexUpdateA5Common::Align(topkOut_, (uint32_t)256)];
        idxHighLocal_ = histogramsLocal_[256];
        idxLowLocal_ = idxHighLocal_[256];
        nkValueLocal_ = idxLowLocal_[256];
        LocalTensor<uint32_t> tmpIndexLocalTmp = nkValueLocal_[64];
        tmpIndexLocal_ = tmpIndexLocalTmp.template ReinterpretCast<uint16_t>();
    }

    // LiTopKVF materializes the final 16-bit histogram boundary in the low
    // half of nkValueLocal_[0]. Callers with private candidate scratch can
    // consume it directly instead of reducing all survivor scores again.
    __aicore__ inline uint16_t GetLastKthValue()
    {
        return static_cast<uint16_t>(nkValueLocal_.GetValue(0) & 0xffffu);
    }

    __aicore__ inline void GatherSourceTags(
        const LocalTensor<uint16_t>& outputTags,
        const LocalTensor<uint16_t>& previousTags,
        const LocalTensor<uint16_t>& currentTags,
        uint32_t previousLen, uint32_t tokenBase,
        bool stagedCurrent)
    {
        LiTopKGatherSourceTagsVF(
            outputTags, tmpIndexLocal_, previousTags, currentTags,
            topkTarget_, previousLen, tokenBase, stagedCurrent);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SavePrimaryPayload(
        const LocalTensor<uint32_t>& dst,
        const LocalTensor<uint32_t>& finalPayload,
        uint32_t loopIndex, uint32_t s2LoopNum)
    {
        if (loopIndex + 1 == s2LoopNum) {
            DataCopy(dst, finalPayload,
                     VllmFastHistTopkIndexUpdateA5Common::Align(topkTarget_, static_cast<uint32_t>(256)));
        } else {
            DataCopy(dst, survivorPayloadLocal_[(loopIndex + 1) % 2],
                     VllmFastHistTopkIndexUpdateA5Common::Align(topkTarget_, static_cast<uint32_t>(256)));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RestorePrimaryPayload(
        const LocalTensor<uint32_t>& src,
        const LocalTensor<uint32_t>& finalPayload,
        uint32_t loopIndex, uint32_t s2LoopNum)
    {
        if (loopIndex + 1 == s2LoopNum) {
            DataCopy(finalPayload, src,
                     VllmFastHistTopkIndexUpdateA5Common::Align(topkTarget_, static_cast<uint32_t>(256)));
        } else {
            DataCopy(survivorPayloadLocal_[(loopIndex + 1) % 2], src,
                     VllmFastHistTopkIndexUpdateA5Common::Align(topkTarget_, static_cast<uint32_t>(256)));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DebugCopyTmpIndices(const LocalTensor<uint32_t>& dst)
    {
        Cast(dst, tmpIndexLocal_, RoundMode::CAST_NONE, topkTarget_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DebugCopyPreviousPayloadToGm(
        GlobalTensor<int32_t>& dst, uint32_t loopIndex)
    {
        DataCopyPad(dst, survivorPayloadLocal_[loopIndex % 2].template ReinterpretCast<int32_t>(),
                    {1, static_cast<uint16_t>(topkTarget_ * sizeof(int32_t)), 0, 0});
    }

    __aicore__ inline uint32_t DebugGetPreviousPayloadValue(
        uint32_t loopIndex, uint32_t index)
    {
        return survivorPayloadLocal_[loopIndex % 2].GetValue(index);
    }

    __aicore__ inline uint32_t DebugGetTmpIndexValue(uint32_t index)
    {
        return static_cast<uint32_t>(tmpIndexLocal_.GetValue(index));
    }

    __aicore__ inline void operator()(
        const LocalTensor<uint16_t>& inputValueLocal,
        const LocalTensor<uint32_t>& indicesOutLocal,
        const LocalTensor<uint16_t>& scoreOutLocal,
        const LocalTensor<uint16_t>& currentSlotsLocal,
        const LocalTensor<int32_t>& slotStageLocal,
        GlobalTensor<int32_t>& cachedSlotsGm,
        uint64_t slotOffset,
        uint32_t tokenBase,
        uint32_t slotStageLen,
        uint32_t validLen,
        uint32_t currentValidLen,
        uint32_t loopIndex,
        uint32_t s2LoopNum,
        bool isFinal,
        bool emitPayload)
    {
        (void)isFinal;
        // All LiTopKVF calls use topkTarget_ as the K search parameter.
        if (s2LoopNum == 1) {
            // This fused update consumes the survivor scores after TopK to
            // derive the protected kth boundary and to validate eviction
            // candidates.  The one-trunk path must therefore materialize
            // scoreOutLocal_ just like the multi-trunk path.
            LiTopKVF<true>(tmpIndexLocal_, scoreOutLocal,
                           inputValueLocal, histogramsLocal_,
                           idxHighLocal_, idxLowLocal_, nkValueLocal_,
                           topkTarget_, validLen);
            PipeBarrier<PIPE_V>();
            if (emitPayload) {
                LoadCurrentSlotsCompact(currentSlotsLocal, slotStageLocal, cachedSlotsGm,
                                        slotOffset + tokenBase, currentValidLen, slotStageLen);
                LiTopKGatherPayloadVF(indicesOutLocal, scoreOutLocal, inputValueLocal,
                                      tmpIndexLocal_, indicesOutLocal, currentSlotsLocal,
                                      topkTarget_, 0, tokenBase, validLen);
                PipeBarrier<PIPE_V>();
            } else {
                Cast(indicesOutLocal, tmpIndexLocal_, RoundMode::CAST_NONE, topkTarget_);
                PipeBarrier<PIPE_V>();
            }
            return;
        }

        if (loopIndex == 0) {
            LiTopKVF<true>(tmpIndexLocal_, scoreOutLocal,
                           inputValueLocal, histogramsLocal_,
                           idxHighLocal_, idxLowLocal_, nkValueLocal_,
                           topkTarget_, validLen);
            PipeBarrier<PIPE_V>();
            if (emitPayload) {
                LoadCurrentSlotsCompact(currentSlotsLocal, slotStageLocal, cachedSlotsGm,
                                        slotOffset + tokenBase, currentValidLen, slotStageLen);
                LiTopKGatherPayloadVF(survivorPayloadLocal_[(loopIndex + 1) % 2], scoreOutLocal,
                                      inputValueLocal, tmpIndexLocal_,
                                      survivorPayloadLocal_[loopIndex % 2], currentSlotsLocal,
                                      topkTarget_, 0, tokenBase, validLen);
                PipeBarrier<PIPE_V>();
            } else {
                Cast(survivorPayloadLocal_[(loopIndex + 1) % 2], tmpIndexLocal_,
                     RoundMode::CAST_NONE, topkTarget_);
                PipeBarrier<PIPE_V>();
            }
        } else {
            LiTopKVF<true>(tmpIndexLocal_, scoreOutLocal,
                           inputValueLocal, histogramsLocal_,
                           idxHighLocal_, idxLowLocal_, nkValueLocal_,
                           topkTarget_, validLen);
            PipeBarrier<PIPE_V>();
            if (emitPayload) {
                LoadCurrentSlotsCompact(currentSlotsLocal, slotStageLocal, cachedSlotsGm,
                                        slotOffset + tokenBase, currentValidLen, slotStageLen);
                if (loopIndex == s2LoopNum - 1) {
                    LiTopKGatherPayloadVF(indicesOutLocal, scoreOutLocal,
                                          inputValueLocal, tmpIndexLocal_,
                                          survivorPayloadLocal_[loopIndex % 2],
                                          currentSlotsLocal,
                                          topkTarget_,
                                          VllmFastHistTopkIndexUpdateA5Common::Align(topkTarget_, (uint32_t)256), tokenBase,
                                          validLen);
                    PipeBarrier<PIPE_V>();
                } else {
                    LiTopKGatherPayloadVF(survivorPayloadLocal_[(loopIndex + 1) % 2], scoreOutLocal,
                                          inputValueLocal, tmpIndexLocal_,
                                          survivorPayloadLocal_[loopIndex % 2],
                                          currentSlotsLocal,
                                          topkTarget_,
                                          VllmFastHistTopkIndexUpdateA5Common::Align(topkTarget_, (uint32_t)256), tokenBase,
                                          validLen);
                    PipeBarrier<PIPE_V>();
                }
            } else {
                // Non-payload fallback must be bit-for-bit equivalent to
                // hist_topk_topk.h.  LiTopKGatherPayloadVF carries payload
                // semantics and can leave invalid/shifted survivor token ids
                // when currentPayloadLocal is just indicesOutLocal_ scratch.
                LiTopKGatherVF(survivorPayloadLocal_[(loopIndex + 1) % 2], scoreOutLocal,
                               inputValueLocal, tmpIndexLocal_, survivorPayloadLocal_[loopIndex % 2],
                               topkTarget_,
                               loopIndex * trunkLen_ -
                                   VllmFastHistTopkIndexUpdateA5Common::Align(topkTarget_, (uint32_t)256),
                               validLen);
                PipeBarrier<PIPE_V>();
            }
            if (!emitPayload && loopIndex == s2LoopNum - 1) {
                PipeBarrier<PIPE_V>();
                DataCopy(indicesOutLocal, survivorPayloadLocal_[(loopIndex + 1) % 2],
                         VllmFastHistTopkIndexUpdateA5Common::Align(topkTarget_, (uint32_t)256));
                PipeBarrier<PIPE_V>();
            }
        }
    }

    // Native 16K histogram followed by survivor-only payload construction.
    // cache_slots are read as contiguous slotStageLen chunks; tmpIndexLocal_
    // selects either the previous survivors or an entry in the streamed
    // current-trunk slot chunk.
    __aicore__ inline void RunStreamingPayload(
        const LocalTensor<uint16_t>& inputValueLocal,
        const LocalTensor<uint32_t>& payloadOutLocal,
        const LocalTensor<uint16_t>& scoreOutLocal,
        const LocalTensor<int32_t>& slotStageLocal,
        GlobalTensor<int32_t>& cachedSlotsGm,
        uint64_t slotOffset,
        uint32_t tokenBase,
        uint32_t slotStageLen,
        uint32_t validLen,
        uint32_t currentValidLen,
        uint32_t loopIndex,
        uint32_t s2LoopNum)
    {
        LiTopKVF<true>(tmpIndexLocal_, scoreOutLocal,
                       inputValueLocal, histogramsLocal_,
                       idxHighLocal_, idxLowLocal_, nkValueLocal_,
                       topkTarget_, validLen);
        PipeBarrier<PIPE_V>();

        uint32_t previousLen = loopIndex == 0
            ? 0
            : VllmFastHistTopkIndexUpdateA5Common::Align(topkTarget_, static_cast<uint32_t>(256));
        LocalTensor<uint32_t> outputPayload = payloadOutLocal;
        if (loopIndex + 1 != s2LoopNum) {
            outputPayload = survivorPayloadLocal_[(loopIndex + 1) % 2];
        }

        if (previousLen == 0) {
            InitFirstStreamingPayloadVF(
                outputPayload, tmpIndexLocal_, topkTarget_, tokenBase);
        } else {
            InitMergedStreamingPayloadVF(
                outputPayload, tmpIndexLocal_,
                survivorPayloadLocal_[loopIndex % 2],
                topkTarget_, previousLen, tokenBase);
        }
        PipeBarrier<PIPE_V>();

        AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        for (uint32_t chunkBase = 0;
             chunkBase < currentValidLen;
             chunkBase += slotStageLen) {
            uint32_t chunkLen =
                chunkBase + slotStageLen > currentValidLen
                    ? currentValidLen - chunkBase
                    : slotStageLen;
            SetFlag<HardEvent::V_MTE2>(TOPK_PAYLOAD_V_MTE2_EVENT);
            WaitFlag<HardEvent::V_MTE2>(TOPK_PAYLOAD_V_MTE2_EVENT);
            copyParams.blockLen = chunkLen * sizeof(int32_t);
            AscendC::DataCopyPad(
                slotStageLocal,
                cachedSlotsGm[slotOffset + tokenBase + chunkBase],
                copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(TOPK_PAYLOAD_MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(TOPK_PAYLOAD_MTE2_V_EVENT);
            ApplyStreamingSlotChunkVF(
                outputPayload, tmpIndexLocal_, slotStageLocal,
                topkTarget_, previousLen, chunkBase, chunkLen);
            PipeBarrier<PIPE_V>();
        }
    }

    // Native 16K histogram followed by a dense per-trunk slot sidecar.
    //
    // LiTopKVF has already materialized the survivor scores in scoreOutLocal,
    // so inputValueLocal is dead after the histogram pass.  Reuse that 36 KiB
    // merge-score allocation as a 32 KiB uint16 slot sidecar instead of
    // reserving another UB buffer.  tmpIndexLocal_ can then gather each
    // survivor's slot directly in one SIMD pass; no per-2K chunk survivor
    // matching is required.
    __aicore__ inline void RunSidecarPayload(
        const LocalTensor<uint16_t>& inputValueLocal,
        const LocalTensor<uint32_t>& payloadOutLocal,
        const LocalTensor<uint16_t>& scoreOutLocal,
        const LocalTensor<int32_t>& slotPrefetchLocal,
        const LocalTensor<int32_t>& slotStageLocal,
        GlobalTensor<int32_t>& cachedSlotsGm,
        uint64_t slotOffset,
        uint32_t tokenBase,
        uint32_t slotStageLen,
        uint32_t validLen,
        uint32_t currentValidLen,
        uint32_t loopIndex,
        uint32_t s2LoopNum,
        bool allowSlotPrefetch,
        uint32_t slotPrefetchCap,
        uint32_t offloadedLen,
        uint32_t cacheBudget)
    {
        // Only the durable prefix has reverse-map entries. Tail keys take
        // part in exactly the same score TopK, but their slots are resident
        // consecutively after cacheBudget, never misses or eviction candidates.
        const uint32_t prefixLen = tokenBase >= offloadedLen ? 0U :
            VllmFastHistTopkIndexUpdateA5Common::Min(currentValidLen, offloadedLen - tokenBase);
        uint32_t prefetchLen = prefixLen;
        if (prefetchLen > slotPrefetchCap) {
            prefetchLen = slotPrefetchCap;
        }
        const bool prefetch = allowSlotPrefetch && prefetchLen != 0U;
        if (prefetch) {
            // Final requests may prefetch the full 16K trunk through resMm1;
            // earlier requests prefetch a 2K prefix through slotStage scratch.
            StartCurrentSlotsPrefetch(
                slotPrefetchLocal, cachedSlotsGm,
                slotOffset + tokenBase, prefetchLen);
        }

        LiTopKVF<true>(tmpIndexLocal_, scoreOutLocal,
                       inputValueLocal, histogramsLocal_,
                       idxHighLocal_, idxLowLocal_, nkValueLocal_,
                       topkTarget_, validLen);
        PipeBarrier<PIPE_V>();

        // inputValueLocal is no longer consumed after LiTopKVF, so compact
        // the slot prefetch into it before gathering survivor payloads.
        if (prefetch) {
            FinishCurrentSlotsPrefetch(
                inputValueLocal, slotPrefetchLocal, prefetchLen);
            if (prefetchLen < prefixLen) {
                LoadCurrentSlotsCompactRange(
                    inputValueLocal, slotStageLocal, cachedSlotsGm,
                    slotOffset + tokenBase, prefetchLen,
                    prefixLen, slotStageLen);
            }
        } else {
            LoadCurrentSlotsCompact(
                inputValueLocal, slotStageLocal, cachedSlotsGm,
                slotOffset + tokenBase, prefixLen, slotStageLen);
        }

        if (prefixLen < currentValidLen) {
            // Fill this trunk's tail slots. Order scalar sidecar stores after
            // the histogram/slot casts and before payload vector gather.
            SetFlag<HardEvent::V_S>(EVENT_ID6);
            WaitFlag<HardEvent::V_S>(EVENT_ID6);
            for (uint32_t i = prefixLen; i < currentValidLen; ++i) {
                inputValueLocal.SetValue(i, static_cast<uint16_t>(
                    cacheBudget + (tokenBase + i - offloadedLen)));
            }
            SetFlag<HardEvent::S_V>(EVENT_ID6);
            WaitFlag<HardEvent::S_V>(EVENT_ID6);
        }

        uint32_t previousLen = loopIndex == 0
            ? 0
            : VllmFastHistTopkIndexUpdateA5Common::Align(
                topkTarget_, static_cast<uint32_t>(256));
        LocalTensor<uint32_t> outputPayload = payloadOutLocal;
        if (loopIndex + 1 != s2LoopNum) {
            outputPayload = survivorPayloadLocal_[(loopIndex + 1) % 2];
        }

        // scoreOutLocal already contains the final scores.  The two score
        // tensor arguments are intentionally ignored by this payload-only
        // wrapper; inputValueLocal now denotes the dense slot sidecar.
        LiTopKGatherPayloadVF(
            outputPayload, scoreOutLocal, inputValueLocal,
            tmpIndexLocal_, survivorPayloadLocal_[loopIndex % 2],
            inputValueLocal, topkTarget_, previousLen, tokenBase, validLen);
        PipeBarrier<PIPE_V>();
        (void)slotStageLocal;
        (void)slotStageLen;
    }

    // P1 ablation: retain the exact contiguous cache_slots DMA/event pattern
    // used by RunStreamingPayload, but omit survivor matching and payload
    // construction. The destination is intentionally consumed by the MTE2_V
    // event before it is reused so the measured stage cannot be optimized
    // into an asynchronous tail.
    __aicore__ inline void RunSlotStreamOnly(
        const LocalTensor<int32_t>& slotStageLocal,
        GlobalTensor<int32_t>& cachedSlotsGm,
        uint64_t slotOffset,
        uint32_t tokenBase,
        uint32_t slotStageLen,
        uint32_t currentValidLen)
    {
        AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        for (uint32_t chunkBase = 0;
             chunkBase < currentValidLen;
             chunkBase += slotStageLen) {
            uint32_t chunkLen =
                chunkBase + slotStageLen > currentValidLen
                    ? currentValidLen - chunkBase
                    : slotStageLen;
            SetFlag<HardEvent::V_MTE2>(TOPK_PAYLOAD_V_MTE2_EVENT);
            WaitFlag<HardEvent::V_MTE2>(TOPK_PAYLOAD_V_MTE2_EVENT);
            copyParams.blockLen = chunkLen * sizeof(int32_t);
            AscendC::DataCopyPad(
                slotStageLocal,
                cachedSlotsGm[slotOffset + tokenBase + chunkBase],
                copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(TOPK_PAYLOAD_MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(TOPK_PAYLOAD_MTE2_V_EVENT);
            PipeBarrier<PIPE_V>();
        }
    }

    // P0/P1 ablation: run the same 16K histogram and survivor propagation
    // structure as the payload path, but keep only the token_id in each
    // uint32 survivor. This avoids the legacy non-payload fallback while
    // removing every cache_slots access.
    __aicore__ inline void RunTokenPayloadOnly(
        const LocalTensor<uint16_t>& inputValueLocal,
        const LocalTensor<uint32_t>& payloadOutLocal,
        const LocalTensor<uint16_t>& scoreOutLocal,
        uint32_t tokenBase,
        uint32_t validLen,
        uint32_t loopIndex,
        uint32_t s2LoopNum)
    {
        LiTopKVF<true>(tmpIndexLocal_, scoreOutLocal,
                       inputValueLocal, histogramsLocal_,
                       idxHighLocal_, idxLowLocal_, nkValueLocal_,
                       topkTarget_, validLen);
        PipeBarrier<PIPE_V>();

        uint32_t previousLen = loopIndex == 0
            ? 0
            : VllmFastHistTopkIndexUpdateA5Common::Align(
                topkTarget_, static_cast<uint32_t>(256));
        LocalTensor<uint32_t> outputPayload = payloadOutLocal;
        if (loopIndex + 1 != s2LoopNum) {
            outputPayload = survivorPayloadLocal_[(loopIndex + 1) % 2];
        }

        if (previousLen == 0) {
            InitFirstStreamingPayloadVF(
                outputPayload, tmpIndexLocal_, topkTarget_, tokenBase);
        } else {
            InitMergedStreamingPayloadVF(
                outputPayload, tmpIndexLocal_,
                survivorPayloadLocal_[loopIndex % 2],
                topkTarget_, previousLen, tokenBase);
        }
        PipeBarrier<PIPE_V>();
    }

    // Compile-time ablation wrapper:
    // P0: native 16K histogram TopK only.
    // P1: P0 + contiguous cache_slots stream/events.
    // P2..P5: dense sidecar + direct tmpIndex payload gather.
    __aicore__ inline void RunAblationStage(
        const LocalTensor<uint16_t>& inputValueLocal,
        const LocalTensor<uint32_t>& payloadOutLocal,
        const LocalTensor<uint16_t>& scoreOutLocal,
        const LocalTensor<int32_t>& slotPrefetchLocal,
        const LocalTensor<int32_t>& slotStageLocal,
        GlobalTensor<int32_t>& cachedSlotsGm,
        uint64_t slotOffset,
        uint32_t tokenBase,
        uint32_t slotStageLen,
        uint32_t validLen,
        uint32_t currentValidLen,
        uint32_t loopIndex,
        uint32_t s2LoopNum,
        bool allowSlotPrefetch,
        uint32_t slotPrefetchCap,
        uint32_t offloadedLen,
        uint32_t cacheBudget)
    {
#if LI_UPDATE_ABLATION_MODE >= 2
        RunSidecarPayload(
            inputValueLocal, payloadOutLocal, scoreOutLocal,
            slotPrefetchLocal, slotStageLocal,
            cachedSlotsGm, slotOffset, tokenBase,
            slotStageLen, validLen, currentValidLen, loopIndex, s2LoopNum,
            allowSlotPrefetch, slotPrefetchCap, offloadedLen, cacheBudget);
#else
        RunTokenPayloadOnly(
            inputValueLocal, payloadOutLocal, scoreOutLocal,
            tokenBase, validLen, loopIndex, s2LoopNum);
#if LI_UPDATE_ABLATION_MODE == 1
        RunSlotStreamOnly(
            slotStageLocal, cachedSlotsGm, slotOffset, tokenBase,
            slotStageLen, currentValidLen);
#endif
#endif
        (void)allowSlotPrefetch;
        (void)slotPrefetchCap;
    }

    // Histogram TopK over a compact score stream whose packed payload has
    // already been materialized.  The score path is identical to operator(),
    // but survivor gather reads currentPayloadLocal instead of reconstructing
    // payload from dense token positions and cachedSlots.
    __aicore__ inline void RunExistingPayload(
        const LocalTensor<uint16_t>& inputValueLocal,
        const LocalTensor<uint32_t>& payloadOutLocal,
        const LocalTensor<uint16_t>& scoreOutLocal,
        const LocalTensor<uint32_t>& currentPayloadLocal,
        uint32_t validLen,
        uint32_t previousLen,
        uint32_t loopIndex,
        uint32_t s2LoopNum)
    {
        LiTopKVF<true>(tmpIndexLocal_, scoreOutLocal,
                       inputValueLocal, histogramsLocal_,
                       idxHighLocal_, idxLowLocal_, nkValueLocal_,
                       topkTarget_, validLen);
        PipeBarrier<PIPE_V>();

        if (s2LoopNum == 1) {
            LiTopKGatherCurrentPayloadVF(payloadOutLocal, tmpIndexLocal_,
                                         currentPayloadLocal, topkTarget_);
            PipeBarrier<PIPE_V>();
            return;
        }

        if (loopIndex == 0) {
            LiTopKGatherCurrentPayloadVF(survivorPayloadLocal_[1], tmpIndexLocal_,
                                         currentPayloadLocal, topkTarget_);
            PipeBarrier<PIPE_V>();
            return;
        }

        if (loopIndex == s2LoopNum - 1) {
            LiTopKGatherExistingPayloadVF(payloadOutLocal, tmpIndexLocal_,
                                          survivorPayloadLocal_[loopIndex % 2],
                                          currentPayloadLocal, topkTarget_, previousLen);
        } else {
            LiTopKGatherExistingPayloadVF(survivorPayloadLocal_[(loopIndex + 1) % 2],
                                          tmpIndexLocal_,
                                          survivorPayloadLocal_[loopIndex % 2],
                                          currentPayloadLocal, topkTarget_, previousLen);
        }
        PipeBarrier<PIPE_V>();
    }

private:
    LocalTensor<uint32_t> survivorPayloadLocal_[2];
    LocalTensor<uint32_t> histogramsLocal_;
    LocalTensor<uint32_t> idxHighLocal_;
    LocalTensor<uint32_t> idxLowLocal_;
    LocalTensor<uint32_t> nkValueLocal_;
    LocalTensor<uint16_t> tmpIndexLocal_;

    uint32_t topkTarget_ = 0;
    uint32_t topkOut_ = 0;
    uint32_t trunkLen_ = 0;
};

} // namespace vllm_fast_hist_topk_index_update_a5_payload

#endif // VLLM_FAST_HIST_TOPK_INDEX_UPDATE_A5_TOPK_H
