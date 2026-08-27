/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// This translation unit follows the validated direct-invoke Vector template.
#include "kernel_operator.h"

namespace {

constexpr uint32_t ERROR_RECORD_LANES = 8;

template <typename index_t, typename enabled_t>
class MoeLoraPrefillRouteAllGather {
public:
    __aicore__ inline explicit MoeLoraPrefillRouteAllGather(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void* expandedRowIdx, __gm__ void* routedTopkIds,
        __gm__ void* tokenLoraIndices, __gm__ void* adapterEnabled,
        __gm__ void* localCount, __gm__ void* errorPerCore,
        uint32_t canonicalRows, uint32_t localRows, uint32_t numTokens,
        uint32_t numAdapters, uint32_t topK, uint32_t numExperts,
        uint32_t groupPitch, int64_t firstExpertIdx, uint32_t blockDim,
        uint32_t routeTileRows)
    {
        canonicalRows_ = canonicalRows;
        localRows_ = localRows;
        numTokens_ = numTokens;
        numAdapters_ = numAdapters;
        topK_ = topK;
        numExperts_ = numExperts;
        groupPitch_ = groupPitch;
        firstExpertIdx_ = firstExpertIdx;
        blockDim_ = blockDim;
        routeTileRows_ = routeTileRows;

        expandedGm_.SetGlobalBuffer((__gm__ index_t*)expandedRowIdx, canonicalRows);
        routedTopkGm_.SetGlobalBuffer((__gm__ index_t*)routedTopkIds, canonicalRows);
        tokenLoraGm_.SetGlobalBuffer((__gm__ int64_t*)tokenLoraIndices, numTokens);
        adapterEnabledGm_.SetGlobalBuffer((__gm__ enabled_t*)adapterEnabled, numAdapters);
        localCountGm_.SetGlobalBuffer((__gm__ int32_t*)localCount,
                                      static_cast<uint64_t>(blockDim) * groupPitch);
        errorPerCoreGm_.SetGlobalBuffer((__gm__ int32_t*)errorPerCore,
                                        static_cast<uint64_t>(blockDim) * ERROR_RECORD_LANES);

        pipe_->InitBuffer(expandedQueue_, 1, routeTileRows_ * sizeof(index_t));
        pipe_->InitBuffer(topkQueue_, 1, routeTileRows_ * sizeof(index_t));
        pipe_->InitBuffer(tokenLoraQueue_, 1, routeTileRows_ * sizeof(int64_t));
        pipe_->InitBuffer(adapterEnabledQueue_, 1, groupPitch * sizeof(enabled_t));
        pipe_->InitBuffer(countAndErrorQueue_, 1,
                          (groupPitch + ERROR_RECORD_LANES) * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockDim_) {
            return;
        }

        AscendC::LocalTensor<int32_t> countAndError = countAndErrorQueue_.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> counts = countAndError;
        AscendC::LocalTensor<int32_t> error = countAndError[groupPitch_];
        AscendC::Duplicate(counts, static_cast<int32_t>(0), groupPitch_);
        AscendC::Duplicate(error, static_cast<int32_t>(0), ERROR_RECORD_LANES);

        AscendC::LocalTensor<enabled_t> adapterEnabled =
            adapterEnabledQueue_.AllocTensor<enabled_t>();
        AscendC::DataCopyExtParams enabledCopy{
            1, numAdapters_ * static_cast<uint32_t>(sizeof(enabled_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<enabled_t> enabledPad{
            false, 0, 0, static_cast<enabled_t>(0)};
        AscendC::DataCopyPad(adapterEnabled, adapterEnabledGm_, enabledCopy, enabledPad);
        adapterEnabledQueue_.EnQue(adapterEnabled);
        adapterEnabled = adapterEnabledQueue_.DeQue<enabled_t>();

        const uint32_t rowsPerCore = (canonicalRows_ + blockDim_ - 1) / blockDim_;
        const uint64_t rawBegin = static_cast<uint64_t>(blockIdx) * rowsPerCore;
        const uint32_t begin = rawBegin < canonicalRows_ ? static_cast<uint32_t>(rawBegin) : canonicalRows_;
        const uint32_t rawEnd = begin + rowsPerCore;
        const uint32_t end = rawEnd < canonicalRows_ ? rawEnd : canonicalRows_;

        for (uint32_t tileBegin = begin; tileBegin < end; tileBegin += routeTileRows_) {
            const uint32_t remaining = end - tileBegin;
            const uint32_t tileRows = remaining < routeTileRows_ ? remaining : routeTileRows_;
            ProcessTile(tileBegin, tileRows, adapterEnabled, counts, error);
        }

        adapterEnabledQueue_.FreeTensor(adapterEnabled);
        countAndErrorQueue_.EnQue(countAndError);
        countAndError = countAndErrorQueue_.DeQue<int32_t>();

        AscendC::DataCopyExtParams countCopy{
            1, groupPitch_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(localCountGm_[static_cast<uint64_t>(blockIdx) * groupPitch_],
                             countAndError, countCopy);
        AscendC::DataCopyExtParams errorCopy{
            1, ERROR_RECORD_LANES * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(
            errorPerCoreGm_[static_cast<uint64_t>(blockIdx) * ERROR_RECORD_LANES],
            countAndError[groupPitch_], errorCopy);
        countAndErrorQueue_.FreeTensor(countAndError);
    }

private:
    __aicore__ inline void ProcessTile(
        uint32_t tileBegin, uint32_t tileRows,
        const AscendC::LocalTensor<enabled_t>& adapterEnabled,
        AscendC::LocalTensor<int32_t>& counts,
        AscendC::LocalTensor<int32_t>& error)
    {
        const uint32_t tokenBegin = tileBegin / topK_;
        const uint32_t tokenEnd = (tileBegin + tileRows - 1) / topK_ + 1;
        const uint32_t tokenRows = tokenEnd - tokenBegin;

        AscendC::LocalTensor<index_t> expanded = expandedQueue_.AllocTensor<index_t>();
        AscendC::LocalTensor<index_t> topk = topkQueue_.AllocTensor<index_t>();
        AscendC::LocalTensor<int64_t> tokenLora = tokenLoraQueue_.AllocTensor<int64_t>();

        AscendC::DataCopyExtParams routeCopy{
            1, tileRows * static_cast<uint32_t>(sizeof(index_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<index_t> routePad{
            false, 0, 0, static_cast<index_t>(0)};
        AscendC::DataCopyPad(expanded, expandedGm_[tileBegin], routeCopy, routePad);
        AscendC::DataCopyPad(topk, routedTopkGm_[tileBegin], routeCopy, routePad);

        AscendC::DataCopyExtParams loraCopy{
            1, tokenRows * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> loraPad{false, 0, 0, static_cast<int64_t>(0)};
        AscendC::DataCopyPad(tokenLora, tokenLoraGm_[tokenBegin], loraCopy, loraPad);

        expandedQueue_.EnQue(expanded);
        topkQueue_.EnQue(topk);
        tokenLoraQueue_.EnQue(tokenLora);
        expanded = expandedQueue_.DeQue<index_t>();
        topk = topkQueue_.DeQue<index_t>();
        tokenLora = tokenLoraQueue_.DeQue<int64_t>();

        for (uint32_t local = 0; local < tileRows; ++local) {
            const int64_t destination = static_cast<int64_t>(expanded.GetValue(local));
            if (destination < 0) {
                continue;
            }
            const uint32_t canonicalRow = tileBegin + local;
            const uint32_t tokenLocal = canonicalRow / topK_ - tokenBegin;
            const int64_t loraIdx = tokenLora.GetValue(tokenLocal);
            const int64_t dispatchExpert = static_cast<int64_t>(topk.GetValue(local));
            const int64_t localExpert = dispatchExpert - firstExpertIdx_;
            bool active = destination < static_cast<int64_t>(localRows_) &&
                          loraIdx >= 0 && loraIdx < static_cast<int64_t>(numAdapters_) &&
                          localExpert >= 0 && localExpert < static_cast<int64_t>(numExperts_);
            if (active) {
                active = static_cast<int64_t>(adapterEnabled.GetValue(loraIdx)) != 0;
            }
            uint32_t effectiveGroup = 0;
            if (active) {
                effectiveGroup = static_cast<uint32_t>(loraIdx) * numExperts_ +
                                 static_cast<uint32_t>(localExpert);
            } else if (destination >= static_cast<int64_t>(localRows_)) {
                error.SetValue(0, 1);
            }
            counts.SetValue(effectiveGroup, counts.GetValue(effectiveGroup) + 1);
        }

        expandedQueue_.FreeTensor(expanded);
        topkQueue_.FreeTensor(topk);
        tokenLoraQueue_.FreeTensor(tokenLora);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> expandedQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> topkQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> tokenLoraQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> adapterEnabledQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> countAndErrorQueue_;
    AscendC::GlobalTensor<index_t> expandedGm_;
    AscendC::GlobalTensor<index_t> routedTopkGm_;
    AscendC::GlobalTensor<int64_t> tokenLoraGm_;
    AscendC::GlobalTensor<enabled_t> adapterEnabledGm_;
    AscendC::GlobalTensor<int32_t> localCountGm_;
    AscendC::GlobalTensor<int32_t> errorPerCoreGm_;
    uint32_t canonicalRows_;
    uint32_t localRows_;
    uint32_t numTokens_;
    uint32_t numAdapters_;
    uint32_t topK_;
    uint32_t numExperts_;
    uint32_t groupPitch_;
    int64_t firstExpertIdx_;
    uint32_t blockDim_;
    uint32_t routeTileRows_;
};

#define MOE_LORA_PREFILL_ROUTE_AG_DECLARE(INDEX_TYPE, INDEX_NAME, ENABLED_TYPE, ENABLED_NAME)            \
    extern "C" __global__ __aicore__ void moe_lora_prefill_route_ag_##INDEX_NAME##_##ENABLED_NAME(     \
        __gm__ void* expandedRowIdx, __gm__ void* routedTopkIds, __gm__ void* tokenLoraIndices,          \
        __gm__ void* adapterEnabled, __gm__ void* localCount, __gm__ void* errorPerCore,                 \
        uint32_t canonicalRows, uint32_t localRows, uint32_t numTokens, uint32_t numAdapters,            \
        uint32_t topK, uint32_t numExperts, uint32_t groupPitch, int64_t firstExpertIdx,                 \
        uint32_t blockDim, uint32_t routeTileRows)                                                        \
    {                                                                                                     \
        AscendC::TPipe pipe;                                                                               \
        MoeLoraPrefillRouteAllGather<INDEX_TYPE, ENABLED_TYPE> op(&pipe);                                 \
        op.Init(expandedRowIdx, routedTopkIds, tokenLoraIndices, adapterEnabled, localCount,              \
                errorPerCore, canonicalRows, localRows, numTokens, numAdapters, topK, numExperts,        \
                groupPitch, firstExpertIdx, blockDim, routeTileRows);                                      \
        op.Process();                                                                                      \
    }

MOE_LORA_PREFILL_ROUTE_AG_DECLARE(int32_t, int32, bool, bool)
MOE_LORA_PREFILL_ROUTE_AG_DECLARE(int32_t, int32, int32_t, int32)
MOE_LORA_PREFILL_ROUTE_AG_DECLARE(int32_t, int32, int64_t, int64)
MOE_LORA_PREFILL_ROUTE_AG_DECLARE(int64_t, int64, bool, bool)
MOE_LORA_PREFILL_ROUTE_AG_DECLARE(int64_t, int64, int32_t, int32)
MOE_LORA_PREFILL_ROUTE_AG_DECLARE(int64_t, int64, int64_t, int64)

template <typename count_t, typename enabled_t>
class MoeLoraPrefillRouteAllToAll {
public:
    __aicore__ inline explicit MoeLoraPrefillRouteAllToAll(AscendC::TPipe* pipe)
        : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void* expertCount, __gm__ void* exchangedLoraIndices,
        __gm__ void* adapterEnabled, __gm__ void* localCount,
        __gm__ void* errorPerCore, uint32_t numRows, uint32_t numAdapters,
        uint32_t numExperts, uint32_t groupPitch, uint32_t blockDim,
        uint32_t routeTileRows)
    {
        numRows_ = numRows;
        numAdapters_ = numAdapters;
        numExperts_ = numExperts;
        groupPitch_ = groupPitch;
        blockDim_ = blockDim;
        routeTileRows_ = routeTileRows;
        expertCountGm_.SetGlobalBuffer((__gm__ count_t*)expertCount, numExperts);
        exchangedLoraGm_.SetGlobalBuffer(
            (__gm__ int64_t*)exchangedLoraIndices, numRows);
        adapterEnabledGm_.SetGlobalBuffer(
            (__gm__ enabled_t*)adapterEnabled, numAdapters);
        localCountGm_.SetGlobalBuffer(
            (__gm__ int32_t*)localCount,
            static_cast<uint64_t>(blockDim) * groupPitch);
        errorPerCoreGm_.SetGlobalBuffer(
            (__gm__ int32_t*)errorPerCore,
            static_cast<uint64_t>(blockDim) * ERROR_RECORD_LANES);

        pipe_->InitBuffer(expertCountQueue_, 1,
                          groupPitch * static_cast<uint32_t>(sizeof(count_t)));
        pipe_->InitBuffer(loraQueue_, 1,
                          routeTileRows_ * static_cast<uint32_t>(sizeof(int64_t)));
        pipe_->InitBuffer(adapterEnabledQueue_, 1,
                          groupPitch * static_cast<uint32_t>(sizeof(enabled_t)));
        pipe_->InitBuffer(countAndErrorQueue_, 1,
                          (groupPitch + ERROR_RECORD_LANES) *
                              static_cast<uint32_t>(sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockDim_) {
            return;
        }
        AscendC::LocalTensor<count_t> endpoints =
            expertCountQueue_.AllocTensor<count_t>();
        AscendC::DataCopyExtParams expertCopy{
            1, numExperts_ * static_cast<uint32_t>(sizeof(count_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<count_t> expertPad{
            false, 0, 0, static_cast<count_t>(0)};
        AscendC::DataCopyPad(endpoints, expertCountGm_, expertCopy, expertPad);
        expertCountQueue_.EnQue(endpoints);
        endpoints = expertCountQueue_.DeQue<count_t>();

        int32_t producerError = 0;
        int64_t endpoint = 0;
        for (uint32_t expert = 0; expert < numExperts_; ++expert) {
            int64_t count = static_cast<int64_t>(endpoints.GetValue(expert));
            if (count < 0) {
                producerError |= 1;
                count = 0;
            }
            const int64_t remaining = static_cast<int64_t>(numRows_) - endpoint;
            if (count > remaining) {
                endpoint = static_cast<int64_t>(numRows_);
                producerError |= 1;
            } else {
                endpoint += count;
            }
            endpoints.SetValue(expert, static_cast<count_t>(endpoint));
        }
        if (endpoint != static_cast<int64_t>(numRows_)) {
            producerError |= 2;
        }

        AscendC::LocalTensor<enabled_t> adapterEnabled =
            adapterEnabledQueue_.AllocTensor<enabled_t>();
        AscendC::DataCopyExtParams enabledCopy{
            1, numAdapters_ * static_cast<uint32_t>(sizeof(enabled_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<enabled_t> enabledPad{
            false, 0, 0, static_cast<enabled_t>(0)};
        AscendC::DataCopyPad(adapterEnabled, adapterEnabledGm_, enabledCopy, enabledPad);
        adapterEnabledQueue_.EnQue(adapterEnabled);
        adapterEnabled = adapterEnabledQueue_.DeQue<enabled_t>();

        AscendC::LocalTensor<int32_t> countAndError =
            countAndErrorQueue_.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> counts = countAndError;
        AscendC::LocalTensor<int32_t> error = countAndError[groupPitch_];
        AscendC::Duplicate(counts, static_cast<int32_t>(0), groupPitch_);
        AscendC::Duplicate(error, static_cast<int32_t>(0), ERROR_RECORD_LANES);
        error.SetValue(0, producerError);

        const uint32_t rowsPerCore = (numRows_ + blockDim_ - 1U) / blockDim_;
        const uint64_t rawBegin = static_cast<uint64_t>(blockIdx) * rowsPerCore;
        const uint32_t begin =
            rawBegin < numRows_ ? static_cast<uint32_t>(rawBegin) : numRows_;
        const uint32_t rawEnd = begin + rowsPerCore;
        const uint32_t end = rawEnd < numRows_ ? rawEnd : numRows_;
        uint32_t localExpert = 0;
        while (localExpert < numExperts_ &&
               static_cast<int64_t>(begin) >=
                   static_cast<int64_t>(endpoints.GetValue(localExpert))) {
            ++localExpert;
        }
        for (uint32_t tileBegin = begin; tileBegin < end; tileBegin += routeTileRows_) {
            const uint32_t remaining = end - tileBegin;
            const uint32_t tileRows =
                remaining < routeTileRows_ ? remaining : routeTileRows_;
            AscendC::LocalTensor<int64_t> lora = loraQueue_.AllocTensor<int64_t>();
            AscendC::DataCopyExtParams loraCopy{
                1, tileRows * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int64_t> loraPad{
                false, 0, 0, static_cast<int64_t>(0)};
            AscendC::DataCopyPad(
                lora, exchangedLoraGm_[tileBegin], loraCopy, loraPad);
            loraQueue_.EnQue(lora);
            lora = loraQueue_.DeQue<int64_t>();
            for (uint32_t local = 0; local < tileRows; ++local) {
                const uint32_t row = tileBegin + local;
                while (localExpert < numExperts_ &&
                       static_cast<int64_t>(row) >=
                           static_cast<int64_t>(endpoints.GetValue(localExpert))) {
                    ++localExpert;
                }
                const int64_t loraIdx = lora.GetValue(local);
                bool active = localExpert < numExperts_ && loraIdx >= 0 &&
                              loraIdx < static_cast<int64_t>(numAdapters_);
                if (active) {
                    active = static_cast<int64_t>(
                                 adapterEnabled.GetValue(loraIdx)) != 0;
                }
                uint32_t effectiveGroup = 0;
                if (active) {
                    effectiveGroup = static_cast<uint32_t>(loraIdx) * numExperts_ +
                                     static_cast<uint32_t>(localExpert);
                }
                counts.SetValue(
                    effectiveGroup, counts.GetValue(effectiveGroup) + 1);
            }
            loraQueue_.FreeTensor(lora);
        }

        expertCountQueue_.FreeTensor(endpoints);
        adapterEnabledQueue_.FreeTensor(adapterEnabled);
        countAndErrorQueue_.EnQue(countAndError);
        countAndError = countAndErrorQueue_.DeQue<int32_t>();
        AscendC::DataCopyExtParams countCopy{
            1, groupPitch_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyExtParams errorCopy{
            1, ERROR_RECORD_LANES * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(
            localCountGm_[static_cast<uint64_t>(blockIdx) * groupPitch_],
            countAndError, countCopy);
        AscendC::DataCopyPad(
            errorPerCoreGm_[static_cast<uint64_t>(blockIdx) * ERROR_RECORD_LANES],
            countAndError[groupPitch_], errorCopy);
        countAndErrorQueue_.FreeTensor(countAndError);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> expertCountQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> loraQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> adapterEnabledQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> countAndErrorQueue_;
    AscendC::GlobalTensor<count_t> expertCountGm_;
    AscendC::GlobalTensor<int64_t> exchangedLoraGm_;
    AscendC::GlobalTensor<enabled_t> adapterEnabledGm_;
    AscendC::GlobalTensor<int32_t> localCountGm_;
    AscendC::GlobalTensor<int32_t> errorPerCoreGm_;
    uint32_t numRows_;
    uint32_t numAdapters_;
    uint32_t numExperts_;
    uint32_t groupPitch_;
    uint32_t blockDim_;
    uint32_t routeTileRows_;
};

#define MOE_LORA_PREFILL_ROUTE_A2A_DECLARE(COUNT_TYPE, COUNT_NAME, ENABLED_TYPE, ENABLED_NAME) \
    extern "C" __global__ __aicore__ void moe_lora_prefill_route_a2a_##COUNT_NAME##_##ENABLED_NAME( \
        __gm__ void* expertCount, __gm__ void* exchangedLoraIndices, __gm__ void* adapterEnabled, \
        __gm__ void* localCount, __gm__ void* errorPerCore, uint32_t numRows, \
        uint32_t numAdapters, uint32_t numExperts, uint32_t groupPitch, uint32_t blockDim, \
        uint32_t routeTileRows) \
    { \
        AscendC::TPipe pipe; \
        MoeLoraPrefillRouteAllToAll<COUNT_TYPE, ENABLED_TYPE> op(&pipe); \
        op.Init(expertCount, exchangedLoraIndices, adapterEnabled, localCount, errorPerCore, \
                numRows, numAdapters, numExperts, groupPitch, blockDim, routeTileRows); \
        op.Process(); \
    }

MOE_LORA_PREFILL_ROUTE_A2A_DECLARE(int32_t, int32, bool, bool)
MOE_LORA_PREFILL_ROUTE_A2A_DECLARE(int32_t, int32, int32_t, int32)
MOE_LORA_PREFILL_ROUTE_A2A_DECLARE(int32_t, int32, int64_t, int64)
MOE_LORA_PREFILL_ROUTE_A2A_DECLARE(int64_t, int64, bool, bool)
MOE_LORA_PREFILL_ROUTE_A2A_DECLARE(int64_t, int64, int32_t, int32)
MOE_LORA_PREFILL_ROUTE_A2A_DECLARE(int64_t, int64, int64_t, int64)

class MoeLoraPrefillPrefixB1 {
public:
    __aicore__ inline explicit MoeLoraPrefillPrefixB1(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void* localCount, __gm__ void* corePrefix,
        __gm__ void* groupTotal, uint32_t numGroups, uint32_t groupPitch,
        uint32_t numCores, uint32_t blockDim, uint32_t prefixTileGroups)
    {
        numGroups_ = numGroups;
        groupPitch_ = groupPitch;
        numCores_ = numCores;
        blockDim_ = blockDim;
        prefixTileGroups_ = prefixTileGroups;

        localCountGm_.SetGlobalBuffer(
            (__gm__ int32_t*)localCount,
            static_cast<uint64_t>(numCores) * groupPitch);
        corePrefixGm_.SetGlobalBuffer(
            (__gm__ int32_t*)corePrefix,
            static_cast<uint64_t>(numCores) * groupPitch);
        groupTotalGm_.SetGlobalBuffer((__gm__ int32_t*)groupTotal, numGroups);

        const uint32_t packedBytes =
            numCores * prefixTileGroups_ * static_cast<uint32_t>(sizeof(int32_t));
        pipe_->InitBuffer(countQueue_, 1, packedBytes);
        pipe_->InitBuffer(prefixQueue_, 1, packedBytes);
        pipe_->InitBuffer(totalQueue_, 1,
                          prefixTileGroups_ * static_cast<uint32_t>(sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockDim_) {
            return;
        }
        const uint32_t groupStep = blockDim_ * prefixTileGroups_;
        for (uint32_t groupBegin = blockIdx * prefixTileGroups_;
             groupBegin < numGroups_; groupBegin += groupStep) {
            const uint32_t remaining = numGroups_ - groupBegin;
            const uint32_t validGroups =
                remaining < prefixTileGroups_ ? remaining : prefixTileGroups_;
            ProcessTile(groupBegin, validGroups);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t groupBegin, uint32_t validGroups)
    {
        const uint32_t ubPitchGroups = (validGroups + 7U) & ~7U;
        const uint32_t rowBytes = validGroups * static_cast<uint32_t>(sizeof(int32_t));
        const uint32_t gmStrideBytes =
            (groupPitch_ - validGroups) * static_cast<uint32_t>(sizeof(int32_t));

        AscendC::LocalTensor<int32_t> packedCount = countQueue_.AllocTensor<int32_t>();
        AscendC::DataCopyExtParams countCopy{
            static_cast<uint16_t>(numCores_), rowBytes, gmStrideBytes, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> countPad{
            true, 0, static_cast<uint8_t>(ubPitchGroups - validGroups),
            static_cast<int32_t>(0)};
        AscendC::DataCopyPad(
            packedCount, localCountGm_[groupBegin], countCopy, countPad);
        countQueue_.EnQue(packedCount);
        packedCount = countQueue_.DeQue<int32_t>();

        AscendC::LocalTensor<int32_t> packedPrefix =
            prefixQueue_.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> totals = totalQueue_.AllocTensor<int32_t>();
        for (uint32_t group = 0; group < validGroups; ++group) {
            int32_t running = 0;
            for (uint32_t core = 0; core < numCores_; ++core) {
                const uint32_t offset = core * ubPitchGroups + group;
                packedPrefix.SetValue(offset, running);
                running += packedCount.GetValue(offset);
            }
            totals.SetValue(group, running);
        }
        countQueue_.FreeTensor(packedCount);

        prefixQueue_.EnQue(packedPrefix);
        totalQueue_.EnQue(totals);
        packedPrefix = prefixQueue_.DeQue<int32_t>();
        totals = totalQueue_.DeQue<int32_t>();

        AscendC::DataCopyExtParams prefixCopy{
            static_cast<uint16_t>(numCores_), rowBytes, 0, gmStrideBytes, 0};
        AscendC::DataCopyPad(
            corePrefixGm_[groupBegin], packedPrefix, prefixCopy);
        AscendC::DataCopyExtParams totalCopy{1, rowBytes, 0, 0, 0};
        AscendC::DataCopyPad(groupTotalGm_[groupBegin], totals, totalCopy);

        prefixQueue_.FreeTensor(packedPrefix);
        totalQueue_.FreeTensor(totals);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> countQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> prefixQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> totalQueue_;
    AscendC::GlobalTensor<int32_t> localCountGm_;
    AscendC::GlobalTensor<int32_t> corePrefixGm_;
    AscendC::GlobalTensor<int32_t> groupTotalGm_;
    uint32_t numGroups_;
    uint32_t groupPitch_;
    uint32_t numCores_;
    uint32_t blockDim_;
    uint32_t prefixTileGroups_;
};

extern "C" __global__ __aicore__ void moe_lora_prefill_prefix_b1(
    __gm__ void* localCount, __gm__ void* corePrefix,
    __gm__ void* groupTotal, uint32_t numGroups, uint32_t groupPitch,
    uint32_t numCores, uint32_t blockDim, uint32_t prefixTileGroups)
{
    AscendC::TPipe pipe;
    MoeLoraPrefillPrefixB1 op(&pipe);
    op.Init(localCount, corePrefix, groupTotal, numGroups, groupPitch,
            numCores, blockDim, prefixTileGroups);
    op.Process();
}

class MoeLoraPrefillPrefixB2 {
public:
    __aicore__ inline explicit MoeLoraPrefillPrefixB2(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void* groupTotal, __gm__ void* errorPerCore,
        __gm__ void* groupStart, __gm__ void* groupCountI64,
        __gm__ void* routeError, uint32_t numGroups, uint32_t groupPitch,
        uint32_t numCores, uint32_t numRows)
    {
        numGroups_ = numGroups;
        groupPitch_ = groupPitch;
        numCores_ = numCores;
        numRows_ = numRows;

        groupTotalGm_.SetGlobalBuffer((__gm__ int32_t*)groupTotal, numGroups);
        errorPerCoreGm_.SetGlobalBuffer(
            (__gm__ int32_t*)errorPerCore,
            static_cast<uint64_t>(numCores) * ERROR_RECORD_LANES);
        groupStartGm_.SetGlobalBuffer((__gm__ int32_t*)groupStart, numGroups);
        groupCountI64Gm_.SetGlobalBuffer((__gm__ int64_t*)groupCountI64, numGroups);
        routeErrorGm_.SetGlobalBuffer((__gm__ int32_t*)routeError, ERROR_RECORD_LANES);

        pipe_->InitBuffer(groupTotalQueue_, 1,
                          groupPitch * static_cast<uint32_t>(sizeof(int32_t)));
        pipe_->InitBuffer(errorPerCoreQueue_, 1,
                          numCores * ERROR_RECORD_LANES *
                              static_cast<uint32_t>(sizeof(int32_t)));
        pipe_->InitBuffer(groupStartQueue_, 1,
                          groupPitch * static_cast<uint32_t>(sizeof(int32_t)));
        pipe_->InitBuffer(groupCountQueue_, 1,
                          groupPitch * static_cast<uint32_t>(sizeof(int64_t)));
        pipe_->InitBuffer(routeErrorQueue_, 1,
                          ERROR_RECORD_LANES * static_cast<uint32_t>(sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }

        AscendC::LocalTensor<int32_t> totals = groupTotalQueue_.AllocTensor<int32_t>();
        AscendC::DataCopyExtParams totalCopy{
            1, numGroups_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> totalPad{
            true, 0, static_cast<uint8_t>(groupPitch_ - numGroups_),
            static_cast<int32_t>(0)};
        AscendC::DataCopyPad(totals, groupTotalGm_, totalCopy, totalPad);

        AscendC::LocalTensor<int32_t> perCoreError =
            errorPerCoreQueue_.AllocTensor<int32_t>();
        AscendC::DataCopyExtParams errorCopy{
            1,
            numCores_ * ERROR_RECORD_LANES *
                static_cast<uint32_t>(sizeof(int32_t)),
            0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> errorPad{
            false, 0, 0, static_cast<int32_t>(0)};
        AscendC::DataCopyPad(perCoreError, errorPerCoreGm_, errorCopy, errorPad);

        groupTotalQueue_.EnQue(totals);
        errorPerCoreQueue_.EnQue(perCoreError);
        totals = groupTotalQueue_.DeQue<int32_t>();
        perCoreError = errorPerCoreQueue_.DeQue<int32_t>();

        AscendC::LocalTensor<int32_t> starts = groupStartQueue_.AllocTensor<int32_t>();
        AscendC::LocalTensor<int64_t> counts = groupCountQueue_.AllocTensor<int64_t>();
        AscendC::LocalTensor<int32_t> routeError = routeErrorQueue_.AllocTensor<int32_t>();
        AscendC::Duplicate(routeError, static_cast<int32_t>(0), ERROR_RECORD_LANES);

        int64_t observedRows = 0;
        int32_t error = 0;
        for (uint32_t group = 0; group < numGroups_; ++group) {
            const int32_t count = totals.GetValue(group);
            if (count < 0) {
                error |= 1;
            } else {
                observedRows += static_cast<int64_t>(count);
            }
        }
        for (uint32_t core = 0; core < numCores_; ++core) {
            error |= perCoreError.GetValue(core * ERROR_RECORD_LANES);
        }
        if (observedRows != static_cast<int64_t>(numRows_)) {
            error |= 2;
        }
        // Keep the GMM contract graph-safe even for malformed producer
        // metadata. Missing rows become an inactive group-0 sink; excess
        // rows are truncated at M. Thus the exported count always sums to M.
        const int64_t sinkRows = observedRows < static_cast<int64_t>(numRows_)
            ? static_cast<int64_t>(numRows_) - observedRows
            : 0;
        int64_t running = 0;
        for (uint32_t group = 0; group < numGroups_; ++group) {
            int64_t count = static_cast<int64_t>(totals.GetValue(group));
            if (count < 0) {
                count = 0;
            }
            if (group == 0) {
                count += sinkRows;
            }
            const int64_t remaining = static_cast<int64_t>(numRows_) - running;
            const int64_t normalized = count < remaining ? count : remaining;
            starts.SetValue(group, static_cast<int32_t>(running));
            counts.SetValue(group, normalized);
            running += normalized;
        }
        routeError.SetValue(0, error);

        groupTotalQueue_.FreeTensor(totals);
        errorPerCoreQueue_.FreeTensor(perCoreError);
        groupStartQueue_.EnQue(starts);
        groupCountQueue_.EnQue(counts);
        routeErrorQueue_.EnQue(routeError);
        starts = groupStartQueue_.DeQue<int32_t>();
        counts = groupCountQueue_.DeQue<int64_t>();
        routeError = routeErrorQueue_.DeQue<int32_t>();

        AscendC::DataCopyExtParams startCopy{
            1, numGroups_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyExtParams countCopy{
            1, numGroups_ * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyExtParams routeErrorCopy{
            1, ERROR_RECORD_LANES * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(groupStartGm_, starts, startCopy);
        AscendC::DataCopyPad(groupCountI64Gm_, counts, countCopy);
        AscendC::DataCopyPad(routeErrorGm_, routeError, routeErrorCopy);

        groupStartQueue_.FreeTensor(starts);
        groupCountQueue_.FreeTensor(counts);
        routeErrorQueue_.FreeTensor(routeError);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> groupTotalQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> errorPerCoreQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> groupStartQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> groupCountQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> routeErrorQueue_;
    AscendC::GlobalTensor<int32_t> groupTotalGm_;
    AscendC::GlobalTensor<int32_t> errorPerCoreGm_;
    AscendC::GlobalTensor<int32_t> groupStartGm_;
    AscendC::GlobalTensor<int64_t> groupCountI64Gm_;
    AscendC::GlobalTensor<int32_t> routeErrorGm_;
    uint32_t numGroups_;
    uint32_t groupPitch_;
    uint32_t numCores_;
    uint32_t numRows_;
};

extern "C" __global__ __aicore__ void moe_lora_prefill_prefix_b2(
    __gm__ void* groupTotal, __gm__ void* errorPerCore,
    __gm__ void* groupStart, __gm__ void* groupCountI64,
    __gm__ void* routeError, uint32_t numGroups, uint32_t groupPitch,
    uint32_t numCores, uint32_t numRows)
{
    AscendC::TPipe pipe;
    MoeLoraPrefillPrefixB2 op(&pipe);
    op.Init(groupTotal, errorPerCore, groupStart, groupCountI64,
            routeError, numGroups, groupPitch, numCores, numRows);
    op.Process();
}

template <typename data_t, typename index_t, typename enabled_t>
class MoeLoraPrefillScatterAllGather {
public:
    __aicore__ inline explicit MoeLoraPrefillScatterAllGather(AscendC::TPipe* pipe)
        : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void* x, __gm__ void* expandedRowIdx,
        __gm__ void* routedTopkIds, __gm__ void* tokenLoraIndices,
        __gm__ void* adapterEnabled, __gm__ void* corePrefix,
        __gm__ void* groupStart, __gm__ void* groupTotal, __gm__ void* groupedX,
        __gm__ void* permRecord, uint32_t canonicalRows, uint32_t numRows,
        uint32_t numTokens, uint32_t numAdapters, uint32_t topK,
        uint32_t numExperts, uint32_t numGroups, uint32_t groupPitch,
        uint32_t inputWidth, uint32_t groupedStride,
        int64_t firstExpertIdx, uint32_t blockDim, uint32_t routeTileRows,
        uint32_t columnTileElements)
    {
        canonicalRows_ = canonicalRows;
        numRows_ = numRows;
        numTokens_ = numTokens;
        numAdapters_ = numAdapters;
        topK_ = topK;
        numExperts_ = numExperts;
        numGroups_ = numGroups;
        groupPitch_ = groupPitch;
        inputWidth_ = inputWidth;
        groupedStride_ = groupedStride;
        firstExpertIdx_ = firstExpertIdx;
        blockDim_ = blockDim;
        routeTileRows_ = routeTileRows;
        columnTileElements_ = columnTileElements;

        xGm_.SetGlobalBuffer(
            (__gm__ data_t*)x, static_cast<uint64_t>(numRows) * inputWidth);
        expandedGm_.SetGlobalBuffer((__gm__ index_t*)expandedRowIdx, canonicalRows);
        routedTopkGm_.SetGlobalBuffer((__gm__ index_t*)routedTopkIds, canonicalRows);
        tokenLoraGm_.SetGlobalBuffer((__gm__ int64_t*)tokenLoraIndices, numTokens);
        adapterEnabledGm_.SetGlobalBuffer((__gm__ enabled_t*)adapterEnabled, numAdapters);
        corePrefixGm_.SetGlobalBuffer(
            (__gm__ int32_t*)corePrefix,
            static_cast<uint64_t>(blockDim) * groupPitch);
        groupStartGm_.SetGlobalBuffer((__gm__ int32_t*)groupStart, numGroups);
        groupTotalGm_.SetGlobalBuffer((__gm__ int32_t*)groupTotal, numGroups);
        groupedXGm_.SetGlobalBuffer(
            (__gm__ data_t*)groupedX,
            static_cast<uint64_t>(numRows) * groupedStride);
        permRecordGm_.SetGlobalBuffer(
            (__gm__ int32_t*)permRecord,
            static_cast<uint64_t>(numRows) * ERROR_RECORD_LANES);

        pipe_->InitBuffer(metadataQueue_, 1,
                          4U * groupPitch * static_cast<uint32_t>(sizeof(int32_t)));
        pipe_->InitBuffer(adapterEnabledQueue_, 1,
                          groupPitch * static_cast<uint32_t>(sizeof(enabled_t)));
        pipe_->InitBuffer(expandedQueue_, 1,
                          routeTileRows_ * static_cast<uint32_t>(sizeof(index_t)));
        pipe_->InitBuffer(topkQueue_, 1,
                          routeTileRows_ * static_cast<uint32_t>(sizeof(index_t)));
        pipe_->InitBuffer(tokenLoraQueue_, 1,
                          routeTileRows_ * static_cast<uint32_t>(sizeof(int64_t)));
        pipe_->InitBuffer(xInputQueue_, 2,
                          columnTileElements_ * static_cast<uint32_t>(sizeof(data_t)));
        pipe_->InitBuffer(xOutputQueue_, 2,
                          columnTileElements_ * static_cast<uint32_t>(sizeof(data_t)));
        pipe_->InitBuffer(permRecordQueue_, 2,
                          ERROR_RECORD_LANES * static_cast<uint32_t>(sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockDim_) {
            return;
        }

        AscendC::LocalTensor<int32_t> metadata = metadataQueue_.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> corePrefix = metadata;
        AscendC::LocalTensor<int32_t> groupStart = metadata[groupPitch_];
        AscendC::LocalTensor<int32_t> seen = metadata[2U * groupPitch_];
        AscendC::LocalTensor<int32_t> groupTotal = metadata[3U * groupPitch_];
        AscendC::DataCopy(
            corePrefix,
            corePrefixGm_[static_cast<uint64_t>(blockIdx) * groupPitch_],
            groupPitch_);
        AscendC::DataCopyExtParams startCopy{
            1, numGroups_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> startPad{
            true, 0, static_cast<uint8_t>(groupPitch_ - numGroups_),
            static_cast<int32_t>(0)};
        AscendC::DataCopyPad(groupStart, groupStartGm_, startCopy, startPad);
        AscendC::DataCopyPad(groupTotal, groupTotalGm_, startCopy, startPad);
        metadataQueue_.EnQue(metadata);
        metadata = metadataQueue_.DeQue<int32_t>();
        corePrefix = metadata;
        groupStart = metadata[groupPitch_];
        seen = metadata[2U * groupPitch_];
        groupTotal = metadata[3U * groupPitch_];
        AscendC::Duplicate(seen, static_cast<int32_t>(0), groupPitch_);

        AscendC::LocalTensor<enabled_t> adapterEnabled =
            adapterEnabledQueue_.AllocTensor<enabled_t>();
        AscendC::DataCopyExtParams enabledCopy{
            1, numAdapters_ * static_cast<uint32_t>(sizeof(enabled_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<enabled_t> enabledPad{
            false, 0, 0, static_cast<enabled_t>(0)};
        AscendC::DataCopyPad(adapterEnabled, adapterEnabledGm_, enabledCopy, enabledPad);
        adapterEnabledQueue_.EnQue(adapterEnabled);
        adapterEnabled = adapterEnabledQueue_.DeQue<enabled_t>();

        const uint32_t rowsPerCore = (canonicalRows_ + blockDim_ - 1U) / blockDim_;
        const uint64_t rawBegin = static_cast<uint64_t>(blockIdx) * rowsPerCore;
        const uint32_t begin =
            rawBegin < canonicalRows_ ? static_cast<uint32_t>(rawBegin) : canonicalRows_;
        const uint32_t rawEnd = begin + rowsPerCore;
        const uint32_t end = rawEnd < canonicalRows_ ? rawEnd : canonicalRows_;

        for (uint32_t tileBegin = begin; tileBegin < end; tileBegin += routeTileRows_) {
            const uint32_t remaining = end - tileBegin;
            const uint32_t tileRows =
                remaining < routeTileRows_ ? remaining : routeTileRows_;
            ProcessRouteTile(tileBegin, tileRows, adapterEnabled,
                             corePrefix, groupStart, seen);
        }

        // B2 inserts a group-0 sink when fewer than M local destinations are
        // observed. Core 0 materializes those rows as zeros and marks their
        // permutation record inactive, so later W13/W2 launches remain safe.
        if (blockIdx == 0) {
            const int32_t rawGroup0 = groupTotal.GetValue(0) > 0
                ? groupTotal.GetValue(0)
                : 0;
            const int32_t group0End = numGroups_ > 1
                ? groupStart.GetValue(1)
                : static_cast<int32_t>(numRows_);
            for (int32_t row = rawGroup0; row < group0End; ++row) {
                ZeroRow(static_cast<uint32_t>(row));
                WritePermRecord(static_cast<uint32_t>(row),
                                static_cast<int32_t>(0x80000000U));
            }
        }

        adapterEnabledQueue_.FreeTensor(adapterEnabled);
        metadataQueue_.FreeTensor(metadata);
    }

private:
    __aicore__ inline void ProcessRouteTile(
        uint32_t tileBegin, uint32_t tileRows,
        const AscendC::LocalTensor<enabled_t>& adapterEnabled,
        const AscendC::LocalTensor<int32_t>& corePrefix,
        const AscendC::LocalTensor<int32_t>& groupStart,
        AscendC::LocalTensor<int32_t>& seen)
    {
        const uint32_t tokenBegin = tileBegin / topK_;
        const uint32_t tokenEnd = (tileBegin + tileRows - 1U) / topK_ + 1U;
        const uint32_t tokenRows = tokenEnd - tokenBegin;

        AscendC::LocalTensor<index_t> expanded = expandedQueue_.AllocTensor<index_t>();
        AscendC::LocalTensor<index_t> topk = topkQueue_.AllocTensor<index_t>();
        AscendC::LocalTensor<int64_t> tokenLora = tokenLoraQueue_.AllocTensor<int64_t>();
        AscendC::DataCopyExtParams routeCopy{
            1, tileRows * static_cast<uint32_t>(sizeof(index_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<index_t> routePad{
            false, 0, 0, static_cast<index_t>(0)};
        AscendC::DataCopyPad(expanded, expandedGm_[tileBegin], routeCopy, routePad);
        AscendC::DataCopyPad(topk, routedTopkGm_[tileBegin], routeCopy, routePad);
        AscendC::DataCopyExtParams loraCopy{
            1, tokenRows * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> loraPad{
            false, 0, 0, static_cast<int64_t>(0)};
        AscendC::DataCopyPad(tokenLora, tokenLoraGm_[tokenBegin], loraCopy, loraPad);
        expandedQueue_.EnQue(expanded);
        topkQueue_.EnQue(topk);
        tokenLoraQueue_.EnQue(tokenLora);
        expanded = expandedQueue_.DeQue<index_t>();
        topk = topkQueue_.DeQue<index_t>();
        tokenLora = tokenLoraQueue_.DeQue<int64_t>();

        for (uint32_t local = 0; local < tileRows; ++local) {
            const int64_t destination = static_cast<int64_t>(expanded.GetValue(local));
            if (destination < 0) {
                continue;
            }
            const uint32_t canonicalRow = tileBegin + local;
            const uint32_t tokenLocal = canonicalRow / topK_ - tokenBegin;
            const int64_t loraIdx = tokenLora.GetValue(tokenLocal);
            const int64_t dispatchExpert = static_cast<int64_t>(topk.GetValue(local));
            const int64_t localExpert = dispatchExpert - firstExpertIdx_;
            bool active = destination < static_cast<int64_t>(numRows_) &&
                          loraIdx >= 0 && loraIdx < static_cast<int64_t>(numAdapters_) &&
                          localExpert >= 0 && localExpert < static_cast<int64_t>(numExperts_);
            if (active) {
                active = static_cast<int64_t>(adapterEnabled.GetValue(loraIdx)) != 0;
            }
            uint32_t effectiveGroup = 0;
            if (active) {
                effectiveGroup = static_cast<uint32_t>(loraIdx) * numExperts_ +
                                 static_cast<uint32_t>(localExpert);
            }
            const int32_t pos = groupStart.GetValue(effectiveGroup) +
                                corePrefix.GetValue(effectiveGroup) +
                                seen.GetValue(effectiveGroup);
            seen.SetValue(effectiveGroup, seen.GetValue(effectiveGroup) + 1);

            const int32_t groupEnd = effectiveGroup + 1U < numGroups_
                ? groupStart.GetValue(effectiveGroup + 1U)
                : static_cast<int32_t>(numRows_);
            if (pos < groupStart.GetValue(effectiveGroup) || pos >= groupEnd) {
                continue;
            }

            const uint32_t sourceRow =
                destination < static_cast<int64_t>(numRows_)
                    ? static_cast<uint32_t>(destination)
                    : 0U;
            CopyRow(sourceRow, static_cast<uint32_t>(pos));
            WritePermRecord(static_cast<uint32_t>(pos),
                            active ? static_cast<int32_t>(sourceRow)
                                   : -static_cast<int32_t>(sourceRow) - 1);
        }

        expandedQueue_.FreeTensor(expanded);
        topkQueue_.FreeTensor(topk);
        tokenLoraQueue_.FreeTensor(tokenLora);
    }

    __aicore__ inline void CopyRow(uint32_t sourceRow, uint32_t groupedRow)
    {
        constexpr uint32_t elementsPerBlock = 32U / sizeof(data_t);
        for (uint32_t column = 0; column < inputWidth_; column += columnTileElements_) {
            const uint32_t remaining = inputWidth_ - column;
            const uint32_t validElements =
                remaining < columnTileElements_ ? remaining : columnTileElements_;
            const uint32_t alignedElements =
                (validElements + elementsPerBlock - 1U) / elementsPerBlock * elementsPerBlock;
            AscendC::LocalTensor<data_t> input = xInputQueue_.AllocTensor<data_t>();
            if (validElements == alignedElements) {
                AscendC::DataCopy(
                    input,
                    xGm_[static_cast<uint64_t>(sourceRow) * inputWidth_ + column],
                    validElements);
            } else {
                AscendC::DataCopyExtParams copy{
                    1, validElements * static_cast<uint32_t>(sizeof(data_t)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<data_t> pad{
                    true, 0, static_cast<uint8_t>(alignedElements - validElements),
                    static_cast<data_t>(0)};
                AscendC::DataCopyPad(
                    input,
                    xGm_[static_cast<uint64_t>(sourceRow) * inputWidth_ + column],
                    copy, pad);
            }
            xInputQueue_.EnQue(input);
            input = xInputQueue_.DeQue<data_t>();
            AscendC::LocalTensor<data_t> output = xOutputQueue_.AllocTensor<data_t>();
            // A2 Adds does not accept BF16. Both supported payload dtypes are
            // 16-bit, so use an int16 bitwise identity staging operation.
            AscendC::Adds(
                output.template ReinterpretCast<int16_t>(),
                input.template ReinterpretCast<int16_t>(),
                static_cast<int16_t>(0),
                static_cast<int32_t>(alignedElements));
            xOutputQueue_.EnQue(output);
            xInputQueue_.FreeTensor(input);
            output = xOutputQueue_.DeQue<data_t>();
            if (validElements == alignedElements) {
                AscendC::DataCopy(
                    groupedXGm_[static_cast<uint64_t>(groupedRow) * groupedStride_ + column],
                    output, validElements);
            } else {
                AscendC::DataCopyExtParams copy{
                    1, validElements * static_cast<uint32_t>(sizeof(data_t)), 0, 0, 0};
                AscendC::DataCopyPad(
                    groupedXGm_[static_cast<uint64_t>(groupedRow) * groupedStride_ + column],
                    output, copy);
            }
            xOutputQueue_.FreeTensor(output);
        }
    }

    __aicore__ inline void ZeroRow(uint32_t groupedRow)
    {
        constexpr uint32_t elementsPerBlock = 32U / sizeof(data_t);
        for (uint32_t column = 0; column < inputWidth_; column += columnTileElements_) {
            const uint32_t remaining = inputWidth_ - column;
            const uint32_t validElements = remaining < columnTileElements_
                ? remaining
                : columnTileElements_;
            const uint32_t alignedElements =
                (validElements + elementsPerBlock - 1U) / elementsPerBlock * elementsPerBlock;
            AscendC::LocalTensor<data_t> output = xOutputQueue_.AllocTensor<data_t>();
            AscendC::Duplicate(output.template ReinterpretCast<int16_t>(),
                               static_cast<int16_t>(0), alignedElements);
            xOutputQueue_.EnQue(output);
            output = xOutputQueue_.DeQue<data_t>();
            AscendC::DataCopyExtParams copy{
                1, validElements * static_cast<uint32_t>(sizeof(data_t)), 0, 0, 0};
            AscendC::DataCopyPad(
                groupedXGm_[static_cast<uint64_t>(groupedRow) * groupedStride_ + column],
                output, copy);
            xOutputQueue_.FreeTensor(output);
        }
    }

    __aicore__ inline void WritePermRecord(uint32_t groupedRow, int32_t encodedRow)
    {
        AscendC::LocalTensor<int32_t> record = permRecordQueue_.AllocTensor<int32_t>();
        AscendC::Duplicate(record, static_cast<int32_t>(0), ERROR_RECORD_LANES);
        record.SetValue(0, encodedRow);
        permRecordQueue_.EnQue(record);
        record = permRecordQueue_.DeQue<int32_t>();
        AscendC::DataCopy(
            permRecordGm_[static_cast<uint64_t>(groupedRow) * ERROR_RECORD_LANES],
            record, ERROR_RECORD_LANES);
        permRecordQueue_.FreeTensor(record);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> metadataQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> adapterEnabledQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> expandedQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> topkQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> tokenLoraQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> xInputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> xOutputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> permRecordQueue_;
    AscendC::GlobalTensor<data_t> xGm_;
    AscendC::GlobalTensor<index_t> expandedGm_;
    AscendC::GlobalTensor<index_t> routedTopkGm_;
    AscendC::GlobalTensor<int64_t> tokenLoraGm_;
    AscendC::GlobalTensor<enabled_t> adapterEnabledGm_;
    AscendC::GlobalTensor<int32_t> corePrefixGm_;
    AscendC::GlobalTensor<int32_t> groupStartGm_;
    AscendC::GlobalTensor<int32_t> groupTotalGm_;
    AscendC::GlobalTensor<data_t> groupedXGm_;
    AscendC::GlobalTensor<int32_t> permRecordGm_;
    uint32_t canonicalRows_;
    uint32_t numRows_;
    uint32_t numTokens_;
    uint32_t numAdapters_;
    uint32_t topK_;
    uint32_t numExperts_;
    uint32_t numGroups_;
    uint32_t groupPitch_;
    uint32_t inputWidth_;
    uint32_t groupedStride_;
    int64_t firstExpertIdx_;
    uint32_t blockDim_;
    uint32_t routeTileRows_;
    uint32_t columnTileElements_;
};

#define MOE_LORA_PREFILL_SCATTER_AG_DECLARE(                                                       \
    DATA_TYPE, DATA_NAME, INDEX_TYPE, INDEX_NAME, ENABLED_TYPE, ENABLED_NAME)                       \
    extern "C" __global__ __aicore__ void                                                         \
        moe_lora_prefill_scatter_ag_##DATA_NAME##_##INDEX_NAME##_##ENABLED_NAME(                    \
            __gm__ void* x, __gm__ void* expandedRowIdx, __gm__ void* routedTopkIds,                \
            __gm__ void* tokenLoraIndices, __gm__ void* adapterEnabled, __gm__ void* corePrefix,    \
            __gm__ void* groupStart, __gm__ void* groupTotal, __gm__ void* groupedX,                \
            __gm__ void* permRecord,                                                                \
            uint32_t canonicalRows, uint32_t numRows, uint32_t numTokens, uint32_t numAdapters,     \
            uint32_t topK, uint32_t numExperts, uint32_t numGroups, uint32_t groupPitch,            \
            uint32_t inputWidth, uint32_t groupedStride, int64_t firstExpertIdx, uint32_t blockDim, \
            uint32_t routeTileRows, uint32_t columnTileElements)                                    \
    {                                                                                                \
        AscendC::TPipe pipe;                                                                          \
        MoeLoraPrefillScatterAllGather<DATA_TYPE, INDEX_TYPE, ENABLED_TYPE> op(&pipe);                \
        op.Init(x, expandedRowIdx, routedTopkIds, tokenLoraIndices, adapterEnabled, corePrefix,       \
                groupStart, groupTotal, groupedX, permRecord, canonicalRows, numRows, numTokens,     \
                numAdapters,                                                                        \
                topK, numExperts, numGroups, groupPitch, inputWidth, groupedStride,                  \
                firstExpertIdx, blockDim, routeTileRows, columnTileElements);                          \
        op.Process();                                                                                  \
    }

#define MOE_LORA_PREFILL_SCATTER_AG_ENABLED_TYPES(DATA_TYPE, DATA_NAME, INDEX_TYPE, INDEX_NAME) \
    MOE_LORA_PREFILL_SCATTER_AG_DECLARE(DATA_TYPE, DATA_NAME, INDEX_TYPE, INDEX_NAME, bool, bool) \
    MOE_LORA_PREFILL_SCATTER_AG_DECLARE(DATA_TYPE, DATA_NAME, INDEX_TYPE, INDEX_NAME, int32_t, int32) \
    MOE_LORA_PREFILL_SCATTER_AG_DECLARE(DATA_TYPE, DATA_NAME, INDEX_TYPE, INDEX_NAME, int64_t, int64)

MOE_LORA_PREFILL_SCATTER_AG_ENABLED_TYPES(half, fp16, int32_t, int32)
MOE_LORA_PREFILL_SCATTER_AG_ENABLED_TYPES(half, fp16, int64_t, int64)
MOE_LORA_PREFILL_SCATTER_AG_ENABLED_TYPES(bfloat16_t, bf16, int32_t, int32)
MOE_LORA_PREFILL_SCATTER_AG_ENABLED_TYPES(bfloat16_t, bf16, int64_t, int64)

template <typename data_t, typename count_t, typename enabled_t>
class MoeLoraPrefillScatterAllToAll {
public:
    __aicore__ inline explicit MoeLoraPrefillScatterAllToAll(AscendC::TPipe* pipe)
        : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void* x, __gm__ void* expertCount,
        __gm__ void* exchangedLoraIndices, __gm__ void* adapterEnabled,
        __gm__ void* corePrefix, __gm__ void* groupStart,
        __gm__ void* groupedX, __gm__ void* permRecord,
        uint32_t numRows, uint32_t numAdapters, uint32_t numExperts,
        uint32_t numGroups, uint32_t groupPitch, uint32_t inputWidth,
        uint32_t groupedStride, uint32_t blockDim, uint32_t routeTileRows,
        uint32_t columnTileElements)
    {
        numRows_ = numRows;
        numAdapters_ = numAdapters;
        numExperts_ = numExperts;
        numGroups_ = numGroups;
        groupPitch_ = groupPitch;
        inputWidth_ = inputWidth;
        groupedStride_ = groupedStride;
        blockDim_ = blockDim;
        routeTileRows_ = routeTileRows;
        columnTileElements_ = columnTileElements;
        xGm_.SetGlobalBuffer(
            (__gm__ data_t*)x, static_cast<uint64_t>(numRows) * inputWidth);
        expertCountGm_.SetGlobalBuffer((__gm__ count_t*)expertCount, numExperts);
        exchangedLoraGm_.SetGlobalBuffer(
            (__gm__ int64_t*)exchangedLoraIndices, numRows);
        adapterEnabledGm_.SetGlobalBuffer(
            (__gm__ enabled_t*)adapterEnabled, numAdapters);
        corePrefixGm_.SetGlobalBuffer(
            (__gm__ int32_t*)corePrefix,
            static_cast<uint64_t>(blockDim) * groupPitch);
        groupStartGm_.SetGlobalBuffer((__gm__ int32_t*)groupStart, numGroups);
        groupedXGm_.SetGlobalBuffer(
            (__gm__ data_t*)groupedX,
            static_cast<uint64_t>(numRows) * groupedStride);
        permRecordGm_.SetGlobalBuffer(
            (__gm__ int32_t*)permRecord,
            static_cast<uint64_t>(numRows) * ERROR_RECORD_LANES);

        pipe_->InitBuffer(metadataQueue_, 1,
                          3U * groupPitch * static_cast<uint32_t>(sizeof(int32_t)));
        pipe_->InitBuffer(expertCountQueue_, 1,
                          groupPitch * static_cast<uint32_t>(sizeof(count_t)));
        pipe_->InitBuffer(loraQueue_, 1,
                          routeTileRows_ * static_cast<uint32_t>(sizeof(int64_t)));
        pipe_->InitBuffer(adapterEnabledQueue_, 1,
                          groupPitch * static_cast<uint32_t>(sizeof(enabled_t)));
        pipe_->InitBuffer(xInputQueue_, 2,
                          columnTileElements_ * static_cast<uint32_t>(sizeof(data_t)));
        pipe_->InitBuffer(xOutputQueue_, 2,
                          columnTileElements_ * static_cast<uint32_t>(sizeof(data_t)));
        pipe_->InitBuffer(permRecordQueue_, 2,
                          ERROR_RECORD_LANES * static_cast<uint32_t>(sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockDim_) {
            return;
        }
        AscendC::LocalTensor<int32_t> metadata = metadataQueue_.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> corePrefix = metadata;
        AscendC::LocalTensor<int32_t> groupStart = metadata[groupPitch_];
        AscendC::LocalTensor<int32_t> seen = metadata[2U * groupPitch_];
        AscendC::DataCopy(
            corePrefix,
            corePrefixGm_[static_cast<uint64_t>(blockIdx) * groupPitch_],
            groupPitch_);
        AscendC::DataCopyExtParams startCopy{
            1, numGroups_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> startPad{
            true, 0, static_cast<uint8_t>(groupPitch_ - numGroups_),
            static_cast<int32_t>(0)};
        AscendC::DataCopyPad(groupStart, groupStartGm_, startCopy, startPad);
        metadataQueue_.EnQue(metadata);
        metadata = metadataQueue_.DeQue<int32_t>();
        corePrefix = metadata;
        groupStart = metadata[groupPitch_];
        seen = metadata[2U * groupPitch_];
        AscendC::Duplicate(seen, static_cast<int32_t>(0), groupPitch_);

        AscendC::LocalTensor<count_t> endpoints =
            expertCountQueue_.AllocTensor<count_t>();
        AscendC::DataCopyExtParams expertCopy{
            1, numExperts_ * static_cast<uint32_t>(sizeof(count_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<count_t> expertPad{
            false, 0, 0, static_cast<count_t>(0)};
        AscendC::DataCopyPad(endpoints, expertCountGm_, expertCopy, expertPad);
        expertCountQueue_.EnQue(endpoints);
        endpoints = expertCountQueue_.DeQue<count_t>();
        int64_t endpoint = 0;
        for (uint32_t expert = 0; expert < numExperts_; ++expert) {
            int64_t count = static_cast<int64_t>(endpoints.GetValue(expert));
            if (count < 0) {
                count = 0;
            }
            const int64_t remaining = static_cast<int64_t>(numRows_) - endpoint;
            endpoint = count > remaining
                           ? static_cast<int64_t>(numRows_)
                           : endpoint + count;
            endpoints.SetValue(expert, static_cast<count_t>(endpoint));
        }

        AscendC::LocalTensor<enabled_t> adapterEnabled =
            adapterEnabledQueue_.AllocTensor<enabled_t>();
        AscendC::DataCopyExtParams enabledCopy{
            1, numAdapters_ * static_cast<uint32_t>(sizeof(enabled_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<enabled_t> enabledPad{
            false, 0, 0, static_cast<enabled_t>(0)};
        AscendC::DataCopyPad(adapterEnabled, adapterEnabledGm_, enabledCopy, enabledPad);
        adapterEnabledQueue_.EnQue(adapterEnabled);
        adapterEnabled = adapterEnabledQueue_.DeQue<enabled_t>();

        const uint32_t rowsPerCore = (numRows_ + blockDim_ - 1U) / blockDim_;
        const uint64_t rawBegin = static_cast<uint64_t>(blockIdx) * rowsPerCore;
        const uint32_t begin =
            rawBegin < numRows_ ? static_cast<uint32_t>(rawBegin) : numRows_;
        const uint32_t rawEnd = begin + rowsPerCore;
        const uint32_t end = rawEnd < numRows_ ? rawEnd : numRows_;
        uint32_t localExpert = 0;
        while (localExpert < numExperts_ &&
               static_cast<int64_t>(begin) >=
                   static_cast<int64_t>(endpoints.GetValue(localExpert))) {
            ++localExpert;
        }
        for (uint32_t tileBegin = begin; tileBegin < end; tileBegin += routeTileRows_) {
            const uint32_t remaining = end - tileBegin;
            const uint32_t tileRows =
                remaining < routeTileRows_ ? remaining : routeTileRows_;
            ProcessTile(tileBegin, tileRows, endpoints, adapterEnabled,
                        corePrefix, groupStart, seen, localExpert);
        }

        adapterEnabledQueue_.FreeTensor(adapterEnabled);
        expertCountQueue_.FreeTensor(endpoints);
        metadataQueue_.FreeTensor(metadata);
    }

private:
    __aicore__ inline void ProcessTile(
        uint32_t tileBegin, uint32_t tileRows,
        const AscendC::LocalTensor<count_t>& endpoints,
        const AscendC::LocalTensor<enabled_t>& adapterEnabled,
        const AscendC::LocalTensor<int32_t>& corePrefix,
        const AscendC::LocalTensor<int32_t>& groupStart,
        AscendC::LocalTensor<int32_t>& seen, uint32_t& localExpert)
    {
        AscendC::LocalTensor<int64_t> lora = loraQueue_.AllocTensor<int64_t>();
        AscendC::DataCopyExtParams loraCopy{
            1, tileRows * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> loraPad{
            false, 0, 0, static_cast<int64_t>(0)};
        AscendC::DataCopyPad(lora, exchangedLoraGm_[tileBegin], loraCopy, loraPad);
        loraQueue_.EnQue(lora);
        lora = loraQueue_.DeQue<int64_t>();
        for (uint32_t local = 0; local < tileRows; ++local) {
            const uint32_t row = tileBegin + local;
            while (localExpert < numExperts_ &&
                   static_cast<int64_t>(row) >=
                       static_cast<int64_t>(endpoints.GetValue(localExpert))) {
                ++localExpert;
            }
            const int64_t loraIdx = lora.GetValue(local);
            bool active = localExpert < numExperts_ && loraIdx >= 0 &&
                          loraIdx < static_cast<int64_t>(numAdapters_);
            if (active) {
                active = static_cast<int64_t>(
                             adapterEnabled.GetValue(loraIdx)) != 0;
            }
            uint32_t effectiveGroup = 0;
            if (active) {
                effectiveGroup = static_cast<uint32_t>(loraIdx) * numExperts_ +
                                 static_cast<uint32_t>(localExpert);
            }
            const int32_t pos = groupStart.GetValue(effectiveGroup) +
                                corePrefix.GetValue(effectiveGroup) +
                                seen.GetValue(effectiveGroup);
            seen.SetValue(effectiveGroup, seen.GetValue(effectiveGroup) + 1);
            const int32_t groupEnd = effectiveGroup + 1U < numGroups_
                ? groupStart.GetValue(effectiveGroup + 1U)
                : static_cast<int32_t>(numRows_);
            if (pos < groupStart.GetValue(effectiveGroup) || pos >= groupEnd) {
                continue;
            }
            CopyRow(row, static_cast<uint32_t>(pos));
            WritePermRecord(
                static_cast<uint32_t>(pos),
                active ? static_cast<int32_t>(row)
                       : -static_cast<int32_t>(row) - 1);
        }
        loraQueue_.FreeTensor(lora);
    }

    __aicore__ inline void CopyRow(uint32_t sourceRow, uint32_t groupedRow)
    {
        constexpr uint32_t elementsPerBlock = 32U / sizeof(data_t);
        for (uint32_t column = 0; column < inputWidth_; column += columnTileElements_) {
            const uint32_t remaining = inputWidth_ - column;
            const uint32_t validElements =
                remaining < columnTileElements_ ? remaining : columnTileElements_;
            const uint32_t alignedElements =
                (validElements + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            AscendC::LocalTensor<data_t> input = xInputQueue_.AllocTensor<data_t>();
            if (validElements == alignedElements) {
                AscendC::DataCopy(
                    input,
                    xGm_[static_cast<uint64_t>(sourceRow) * inputWidth_ + column],
                    validElements);
            } else {
                AscendC::DataCopyExtParams copy{
                    1, validElements * static_cast<uint32_t>(sizeof(data_t)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<data_t> pad{
                    true, 0, static_cast<uint8_t>(alignedElements - validElements),
                    static_cast<data_t>(0)};
                AscendC::DataCopyPad(
                    input,
                    xGm_[static_cast<uint64_t>(sourceRow) * inputWidth_ + column],
                    copy, pad);
            }
            xInputQueue_.EnQue(input);
            input = xInputQueue_.DeQue<data_t>();
            AscendC::LocalTensor<data_t> output = xOutputQueue_.AllocTensor<data_t>();
            AscendC::Adds(
                output.template ReinterpretCast<int16_t>(),
                input.template ReinterpretCast<int16_t>(),
                static_cast<int16_t>(0), static_cast<int32_t>(alignedElements));
            xOutputQueue_.EnQue(output);
            xInputQueue_.FreeTensor(input);
            output = xOutputQueue_.DeQue<data_t>();
            if (validElements == alignedElements) {
                AscendC::DataCopy(
                    groupedXGm_[static_cast<uint64_t>(groupedRow) * groupedStride_ + column],
                    output, validElements);
            } else {
                AscendC::DataCopyExtParams copy{
                    1, validElements * static_cast<uint32_t>(sizeof(data_t)), 0, 0, 0};
                AscendC::DataCopyPad(
                    groupedXGm_[static_cast<uint64_t>(groupedRow) * groupedStride_ + column],
                    output, copy);
            }
            xOutputQueue_.FreeTensor(output);
        }
    }

    __aicore__ inline void WritePermRecord(uint32_t groupedRow, int32_t encodedRow)
    {
        AscendC::LocalTensor<int32_t> record = permRecordQueue_.AllocTensor<int32_t>();
        AscendC::Duplicate(record, static_cast<int32_t>(0), ERROR_RECORD_LANES);
        record.SetValue(0, encodedRow);
        permRecordQueue_.EnQue(record);
        record = permRecordQueue_.DeQue<int32_t>();
        AscendC::DataCopy(
            permRecordGm_[static_cast<uint64_t>(groupedRow) * ERROR_RECORD_LANES],
            record, ERROR_RECORD_LANES);
        permRecordQueue_.FreeTensor(record);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> metadataQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> expertCountQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> loraQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> adapterEnabledQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> xInputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> xOutputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> permRecordQueue_;
    AscendC::GlobalTensor<data_t> xGm_;
    AscendC::GlobalTensor<count_t> expertCountGm_;
    AscendC::GlobalTensor<int64_t> exchangedLoraGm_;
    AscendC::GlobalTensor<enabled_t> adapterEnabledGm_;
    AscendC::GlobalTensor<int32_t> corePrefixGm_;
    AscendC::GlobalTensor<int32_t> groupStartGm_;
    AscendC::GlobalTensor<data_t> groupedXGm_;
    AscendC::GlobalTensor<int32_t> permRecordGm_;
    uint32_t numRows_;
    uint32_t numAdapters_;
    uint32_t numExperts_;
    uint32_t numGroups_;
    uint32_t groupPitch_;
    uint32_t inputWidth_;
    uint32_t groupedStride_;
    uint32_t blockDim_;
    uint32_t routeTileRows_;
    uint32_t columnTileElements_;
};

#define MOE_LORA_PREFILL_SCATTER_A2A_DECLARE(                                                 \
    DATA_TYPE, DATA_NAME, COUNT_TYPE, COUNT_NAME, ENABLED_TYPE, ENABLED_NAME)                 \
    extern "C" __global__ __aicore__ void                                                    \
        moe_lora_prefill_scatter_a2a_##DATA_NAME##_##COUNT_NAME##_##ENABLED_NAME(             \
            __gm__ void* x, __gm__ void* expertCount, __gm__ void* exchangedLoraIndices,     \
            __gm__ void* adapterEnabled, __gm__ void* corePrefix, __gm__ void* groupStart,   \
            __gm__ void* groupedX, __gm__ void* permRecord, uint32_t numRows,                \
            uint32_t numAdapters, uint32_t numExperts, uint32_t numGroups,                   \
            uint32_t groupPitch, uint32_t inputWidth, uint32_t groupedStride,                \
            uint32_t blockDim, uint32_t routeTileRows, uint32_t columnTileElements)            \
    {                                                                                         \
        AscendC::TPipe pipe;                                                                   \
        MoeLoraPrefillScatterAllToAll<DATA_TYPE, COUNT_TYPE, ENABLED_TYPE> op(&pipe);          \
        op.Init(x, expertCount, exchangedLoraIndices, adapterEnabled, corePrefix, groupStart,  \
                groupedX, permRecord, numRows, numAdapters, numExperts, numGroups,            \
                groupPitch, inputWidth, groupedStride, blockDim, routeTileRows,                \
                columnTileElements);                                                           \
        op.Process();                                                                          \
    }

#define MOE_LORA_PREFILL_SCATTER_A2A_ENABLED_TYPES(DATA_TYPE, DATA_NAME, COUNT_TYPE, COUNT_NAME) \
    MOE_LORA_PREFILL_SCATTER_A2A_DECLARE(DATA_TYPE, DATA_NAME, COUNT_TYPE, COUNT_NAME, bool, bool) \
    MOE_LORA_PREFILL_SCATTER_A2A_DECLARE(DATA_TYPE, DATA_NAME, COUNT_TYPE, COUNT_NAME, int32_t, int32) \
    MOE_LORA_PREFILL_SCATTER_A2A_DECLARE(DATA_TYPE, DATA_NAME, COUNT_TYPE, COUNT_NAME, int64_t, int64)

MOE_LORA_PREFILL_SCATTER_A2A_ENABLED_TYPES(half, fp16, int32_t, int32)
MOE_LORA_PREFILL_SCATTER_A2A_ENABLED_TYPES(half, fp16, int64_t, int64)
MOE_LORA_PREFILL_SCATTER_A2A_ENABLED_TYPES(bfloat16_t, bf16, int32_t, int32)
MOE_LORA_PREFILL_SCATTER_A2A_ENABLED_TYPES(bfloat16_t, bf16, int64_t, int64)

template <typename data_t>
class MoeLoraPrefillGatherByPerm {
public:
    __aicore__ inline explicit MoeLoraPrefillGatherByPerm(AscendC::TPipe* pipe)
        : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void* source, __gm__ void* permRecord, __gm__ void* groupedX,
        uint32_t numRows, uint32_t inputWidth, uint32_t groupedStride,
        uint32_t blockDim, uint32_t routeTileRows, uint32_t columnTileElements)
    {
        numRows_ = numRows;
        inputWidth_ = inputWidth;
        groupedStride_ = groupedStride;
        blockDim_ = blockDim;
        routeTileRows_ = routeTileRows;
        columnTileElements_ = columnTileElements;
        sourceGm_.SetGlobalBuffer(
            (__gm__ data_t*)source, static_cast<uint64_t>(numRows) * inputWidth);
        permRecordGm_.SetGlobalBuffer(
            (__gm__ int32_t*)permRecord,
            static_cast<uint64_t>(numRows) * ERROR_RECORD_LANES);
        groupedXGm_.SetGlobalBuffer(
            (__gm__ data_t*)groupedX,
            static_cast<uint64_t>(numRows) * groupedStride);

        pipe_->InitBuffer(
            permQueue_, 1,
            routeTileRows_ * ERROR_RECORD_LANES *
                static_cast<uint32_t>(sizeof(int32_t)));
        pipe_->InitBuffer(inputQueue_, 2,
                          columnTileElements_ * static_cast<uint32_t>(sizeof(data_t)));
        pipe_->InitBuffer(outputQueue_, 2,
                          columnTileElements_ * static_cast<uint32_t>(sizeof(data_t)));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockDim_) {
            return;
        }
        const uint32_t rowsPerCore = (numRows_ + blockDim_ - 1U) / blockDim_;
        const uint64_t rawBegin = static_cast<uint64_t>(blockIdx) * rowsPerCore;
        const uint32_t begin =
            rawBegin < numRows_ ? static_cast<uint32_t>(rawBegin) : numRows_;
        const uint32_t rawEnd = begin + rowsPerCore;
        const uint32_t end = rawEnd < numRows_ ? rawEnd : numRows_;
        for (uint32_t tileBegin = begin; tileBegin < end;
             tileBegin += routeTileRows_) {
            const uint32_t remaining = end - tileBegin;
            const uint32_t tileRows =
                remaining < routeTileRows_ ? remaining : routeTileRows_;
            AscendC::LocalTensor<int32_t> perm =
                permQueue_.AllocTensor<int32_t>();
            AscendC::DataCopy(
                perm,
                permRecordGm_[static_cast<uint64_t>(tileBegin) *
                              ERROR_RECORD_LANES],
                tileRows * ERROR_RECORD_LANES);
            permQueue_.EnQue(perm);
            perm = permQueue_.DeQue<int32_t>();
            for (uint32_t local = 0; local < tileRows; ++local) {
                const int32_t encoded =
                    perm.GetValue(local * ERROR_RECORD_LANES);
                if (encoded == static_cast<int32_t>(0x80000000U)) {
                    ZeroRow(tileBegin + local);
                    continue;
                }
                const uint32_t sourceRow = encoded >= 0
                    ? static_cast<uint32_t>(encoded)
                    : static_cast<uint32_t>(-encoded - 1);
                CopyRow(sourceRow, tileBegin + local);
            }
            permQueue_.FreeTensor(perm);
        }
    }

private:
    __aicore__ inline void CopyRow(uint32_t sourceRow, uint32_t groupedRow)
    {
        constexpr uint32_t elementsPerBlock = 32U / sizeof(data_t);
        for (uint32_t column = 0; column < inputWidth_; column += columnTileElements_) {
            const uint32_t remaining = inputWidth_ - column;
            const uint32_t validElements =
                remaining < columnTileElements_ ? remaining : columnTileElements_;
            const uint32_t alignedElements =
                (validElements + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            AscendC::LocalTensor<data_t> input =
                inputQueue_.AllocTensor<data_t>();
            if (validElements == alignedElements) {
                AscendC::DataCopy(
                    input,
                    sourceGm_[static_cast<uint64_t>(sourceRow) * inputWidth_ +
                              column],
                    validElements);
            } else {
                AscendC::DataCopyExtParams copy{
                    1, validElements * static_cast<uint32_t>(sizeof(data_t)),
                    0, 0, 0};
                AscendC::DataCopyPadExtParams<data_t> pad{
                    true, 0,
                    static_cast<uint8_t>(alignedElements - validElements),
                    static_cast<data_t>(0)};
                AscendC::DataCopyPad(
                    input,
                    sourceGm_[static_cast<uint64_t>(sourceRow) * inputWidth_ +
                              column],
                    copy, pad);
            }
            inputQueue_.EnQue(input);
            input = inputQueue_.DeQue<data_t>();
            AscendC::LocalTensor<data_t> output =
                outputQueue_.AllocTensor<data_t>();
            AscendC::Adds(
                output.template ReinterpretCast<int16_t>(),
                input.template ReinterpretCast<int16_t>(),
                static_cast<int16_t>(0),
                static_cast<int32_t>(alignedElements));
            outputQueue_.EnQue(output);
            inputQueue_.FreeTensor(input);
            output = outputQueue_.DeQue<data_t>();
            if (validElements == alignedElements) {
                AscendC::DataCopy(
                    groupedXGm_[static_cast<uint64_t>(groupedRow) *
                                groupedStride_ + column],
                    output, validElements);
            } else {
                AscendC::DataCopyExtParams copy{
                    1, validElements * static_cast<uint32_t>(sizeof(data_t)),
                    0, 0, 0};
                AscendC::DataCopyPad(
                    groupedXGm_[static_cast<uint64_t>(groupedRow) *
                                groupedStride_ + column],
                    output, copy);
            }
            outputQueue_.FreeTensor(output);
        }
    }

    __aicore__ inline void ZeroRow(uint32_t groupedRow)
    {
        constexpr uint32_t elementsPerBlock = 32U / sizeof(data_t);
        for (uint32_t column = 0; column < inputWidth_; column += columnTileElements_) {
            const uint32_t remaining = inputWidth_ - column;
            const uint32_t validElements = remaining < columnTileElements_
                ? remaining
                : columnTileElements_;
            const uint32_t alignedElements =
                (validElements + elementsPerBlock - 1U) / elementsPerBlock * elementsPerBlock;
            AscendC::LocalTensor<data_t> output = outputQueue_.AllocTensor<data_t>();
            AscendC::Duplicate(output.template ReinterpretCast<int16_t>(),
                               static_cast<int16_t>(0), alignedElements);
            outputQueue_.EnQue(output);
            output = outputQueue_.DeQue<data_t>();
            AscendC::DataCopyExtParams copy{
                1, validElements * static_cast<uint32_t>(sizeof(data_t)), 0, 0, 0};
            AscendC::DataCopyPad(
                groupedXGm_[static_cast<uint64_t>(groupedRow) * groupedStride_ + column],
                output, copy);
            outputQueue_.FreeTensor(output);
        }
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> permQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outputQueue_;
    AscendC::GlobalTensor<data_t> sourceGm_;
    AscendC::GlobalTensor<int32_t> permRecordGm_;
    AscendC::GlobalTensor<data_t> groupedXGm_;
    uint32_t numRows_;
    uint32_t inputWidth_;
    uint32_t groupedStride_;
    uint32_t blockDim_;
    uint32_t routeTileRows_;
    uint32_t columnTileElements_;
};

#define MOE_LORA_PREFILL_GATHER_DECLARE(DATA_TYPE, DATA_NAME)                 \
    extern "C" __global__ __aicore__ void                                   \
        moe_lora_prefill_gather_##DATA_NAME(                                  \
            __gm__ void* source, __gm__ void* permRecord,                    \
            __gm__ void* groupedX, uint32_t numRows,                         \
            uint32_t inputWidth, uint32_t groupedStride, uint32_t blockDim,  \
            uint32_t routeTileRows, uint32_t columnTileElements)             \
    {                                                                         \
        AscendC::TPipe pipe;                                                   \
        MoeLoraPrefillGatherByPerm<DATA_TYPE> op(&pipe);                       \
        op.Init(source, permRecord, groupedX, numRows, inputWidth,            \
                groupedStride, blockDim, routeTileRows, columnTileElements);   \
        op.Process();                                                          \
    }

MOE_LORA_PREFILL_GATHER_DECLARE(half, fp16)
MOE_LORA_PREFILL_GATHER_DECLARE(bfloat16_t, bf16)

template <typename data_t>
class MoeLoraPrefillScatterAdd {
public:
    __aicore__ inline explicit MoeLoraPrefillScatterAdd(AscendC::TPipe* pipe)
        : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void* delta, __gm__ void* permRecord, __gm__ void* y,
        uint32_t numRows, uint32_t deltaWidth, uint32_t outputWidth,
        uint32_t outputOffset, uint32_t blockDim, uint32_t routeTileRows,
        uint32_t scatterAddTileElements)
    {
        numRows_ = numRows;
        deltaWidth_ = deltaWidth;
        outputWidth_ = outputWidth;
        outputOffset_ = outputOffset;
        blockDim_ = blockDim;
        routeTileRows_ = routeTileRows;
        scatterAddTileElements_ = scatterAddTileElements;
        deltaGm_.SetGlobalBuffer(
            (__gm__ data_t*)delta,
            static_cast<uint64_t>(numRows) * deltaWidth);
        permRecordGm_.SetGlobalBuffer(
            (__gm__ int32_t*)permRecord,
            static_cast<uint64_t>(numRows) * ERROR_RECORD_LANES);
        yGm_.SetGlobalBuffer(
            (__gm__ data_t*)y,
            static_cast<uint64_t>(numRows) * outputWidth);

        pipe_->InitBuffer(
            permQueue_, 1,
            routeTileRows_ * ERROR_RECORD_LANES *
                static_cast<uint32_t>(sizeof(int32_t)));
        pipe_->InitBuffer(
            deltaQueue_, 2,
            scatterAddTileElements_ * static_cast<uint32_t>(sizeof(data_t)));
        pipe_->InitBuffer(
            yQueue_, 2,
            scatterAddTileElements_ * static_cast<uint32_t>(sizeof(data_t)));
        pipe_->InitBuffer(
            outputQueue_, 2,
            scatterAddTileElements_ * static_cast<uint32_t>(sizeof(data_t)));
        pipe_->InitBuffer(
            deltaFp32Buffer_,
            scatterAddTileElements_ * static_cast<uint32_t>(sizeof(float)));
        pipe_->InitBuffer(
            yFp32Buffer_,
            scatterAddTileElements_ * static_cast<uint32_t>(sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockDim_) {
            return;
        }
        const uint32_t rowsPerCore = (numRows_ + blockDim_ - 1U) / blockDim_;
        const uint64_t rawBegin = static_cast<uint64_t>(blockIdx) * rowsPerCore;
        const uint32_t begin =
            rawBegin < numRows_ ? static_cast<uint32_t>(rawBegin) : numRows_;
        const uint32_t rawEnd = begin + rowsPerCore;
        const uint32_t end = rawEnd < numRows_ ? rawEnd : numRows_;
        for (uint32_t tileBegin = begin; tileBegin < end;
             tileBegin += routeTileRows_) {
            const uint32_t remaining = end - tileBegin;
            const uint32_t tileRows =
                remaining < routeTileRows_ ? remaining : routeTileRows_;
            AscendC::LocalTensor<int32_t> perm =
                permQueue_.AllocTensor<int32_t>();
            AscendC::DataCopy(
                perm,
                permRecordGm_[static_cast<uint64_t>(tileBegin) *
                              ERROR_RECORD_LANES],
                tileRows * ERROR_RECORD_LANES);
            permQueue_.EnQue(perm);
            perm = permQueue_.DeQue<int32_t>();
            for (uint32_t local = 0; local < tileRows; ++local) {
                const int32_t encoded =
                    perm.GetValue(local * ERROR_RECORD_LANES);
                if (encoded >= 0) {
                    AddRow(tileBegin + local, static_cast<uint32_t>(encoded));
                }
            }
            permQueue_.FreeTensor(perm);
        }
    }

private:
    __aicore__ inline void AddRow(uint32_t groupedRow, uint32_t outputRow)
    {
        constexpr uint32_t elementsPerBlock = 32U / sizeof(data_t);
        for (uint32_t column = 0; column < deltaWidth_;
             column += scatterAddTileElements_) {
            const uint32_t remaining = deltaWidth_ - column;
            const uint32_t validElements = remaining < scatterAddTileElements_
                ? remaining
                : scatterAddTileElements_;
            const uint32_t alignedElements =
                (validElements + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            AscendC::LocalTensor<data_t> delta =
                deltaQueue_.AllocTensor<data_t>();
            AscendC::LocalTensor<data_t> y = yQueue_.AllocTensor<data_t>();
            CopyIn(
                delta,
                deltaGm_[static_cast<uint64_t>(groupedRow) * deltaWidth_ +
                         column],
                validElements, alignedElements);
            CopyIn(
                y,
                yGm_[static_cast<uint64_t>(outputRow) * outputWidth_ +
                     outputOffset_ + column],
                validElements, alignedElements);
            deltaQueue_.EnQue(delta);
            yQueue_.EnQue(y);
            delta = deltaQueue_.DeQue<data_t>();
            y = yQueue_.DeQue<data_t>();

            AscendC::LocalTensor<float> deltaFp32 =
                deltaFp32Buffer_.Get<float>();
            AscendC::LocalTensor<float> yFp32 = yFp32Buffer_.Get<float>();
            AscendC::Cast(
                deltaFp32, delta, AscendC::RoundMode::CAST_NONE,
                alignedElements);
            AscendC::Cast(
                yFp32, y, AscendC::RoundMode::CAST_NONE,
                alignedElements);
            AscendC::Add(
                yFp32, yFp32, deltaFp32,
                static_cast<int32_t>(alignedElements));
            AscendC::LocalTensor<data_t> output =
                outputQueue_.AllocTensor<data_t>();
            AscendC::Cast(
                output, yFp32, AscendC::RoundMode::CAST_RINT,
                alignedElements);
            outputQueue_.EnQue(output);
            deltaQueue_.FreeTensor(delta);
            yQueue_.FreeTensor(y);
            output = outputQueue_.DeQue<data_t>();
            CopyOut(
                yGm_[static_cast<uint64_t>(outputRow) * outputWidth_ +
                     outputOffset_ + column],
                output, validElements, alignedElements);
            outputQueue_.FreeTensor(output);
        }
    }

    __aicore__ inline void CopyIn(
        AscendC::LocalTensor<data_t>& local,
        const AscendC::GlobalTensor<data_t>& global,
        uint32_t validElements, uint32_t alignedElements)
    {
        if (validElements == alignedElements) {
            AscendC::DataCopy(local, global, validElements);
        } else {
            AscendC::DataCopyExtParams copy{
                1, validElements * static_cast<uint32_t>(sizeof(data_t)),
                0, 0, 0};
            AscendC::DataCopyPadExtParams<data_t> pad{
                true, 0,
                static_cast<uint8_t>(alignedElements - validElements),
                static_cast<data_t>(0)};
            AscendC::DataCopyPad(local, global, copy, pad);
        }
    }

    __aicore__ inline void CopyOut(
        const AscendC::GlobalTensor<data_t>& global,
        const AscendC::LocalTensor<data_t>& local,
        uint32_t validElements, uint32_t alignedElements)
    {
        if (validElements == alignedElements) {
            AscendC::DataCopy(global, local, validElements);
        } else {
            AscendC::DataCopyExtParams copy{
                1, validElements * static_cast<uint32_t>(sizeof(data_t)),
                0, 0, 0};
            AscendC::DataCopyPad(global, local, copy);
        }
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> permQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> deltaQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> yQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> deltaFp32Buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yFp32Buffer_;
    AscendC::GlobalTensor<data_t> deltaGm_;
    AscendC::GlobalTensor<int32_t> permRecordGm_;
    AscendC::GlobalTensor<data_t> yGm_;
    uint32_t numRows_;
    uint32_t deltaWidth_;
    uint32_t outputWidth_;
    uint32_t outputOffset_;
    uint32_t blockDim_;
    uint32_t routeTileRows_;
    uint32_t scatterAddTileElements_;
};

#define MOE_LORA_PREFILL_SCATTER_ADD_DECLARE(DATA_TYPE, DATA_NAME)             \
    extern "C" __global__ __aicore__ void                                    \
        moe_lora_prefill_scatter_add_##DATA_NAME(                              \
            __gm__ void* delta, __gm__ void* permRecord, __gm__ void* y,      \
            uint32_t numRows, uint32_t deltaWidth, uint32_t outputWidth,       \
            uint32_t outputOffset, uint32_t blockDim, uint32_t routeTileRows, \
            uint32_t scatterAddTileElements)                                  \
    {                                                                          \
        AscendC::TPipe pipe;                                                    \
        MoeLoraPrefillScatterAdd<DATA_TYPE> op(&pipe);                          \
        op.Init(delta, permRecord, y, numRows, deltaWidth, outputWidth,        \
                outputOffset, blockDim, routeTileRows, scatterAddTileElements); \
        op.Process();                                                           \
    }

MOE_LORA_PREFILL_SCATTER_ADD_DECLARE(half, fp16)
MOE_LORA_PREFILL_SCATTER_ADD_DECLARE(bfloat16_t, bf16)

}  // namespace

namespace vllm_ascend {
extern void moe_lora_prefill_route_allgather_impl(
    void* stream, void* expandedRowIdx, void* routedTopkIds,
    void* tokenLoraIndices, void* adapterEnabled, void* localCount,
    void* errorPerCore, uint32_t canonicalRows, uint32_t localRows,
    uint32_t numTokens, uint32_t numAdapters, uint32_t topK,
    uint32_t numExperts, uint32_t groupPitch, int64_t firstExpertIdx,
    uint32_t blockDim, uint32_t routeTileRows, bool index64, uint32_t enabledType)
{
    if (!index64 && enabledType == 0) {
        moe_lora_prefill_route_ag_int32_bool<<<blockDim, nullptr, stream>>>(
            expandedRowIdx, routedTopkIds, tokenLoraIndices, adapterEnabled,
            localCount, errorPerCore, canonicalRows, localRows, numTokens,
            numAdapters, topK, numExperts, groupPitch, firstExpertIdx, blockDim,
            routeTileRows);
    } else if (!index64 && enabledType == 1) {
        moe_lora_prefill_route_ag_int32_int32<<<blockDim, nullptr, stream>>>(
            expandedRowIdx, routedTopkIds, tokenLoraIndices, adapterEnabled,
            localCount, errorPerCore, canonicalRows, localRows, numTokens,
            numAdapters, topK, numExperts, groupPitch, firstExpertIdx, blockDim,
            routeTileRows);
    } else if (!index64) {
        moe_lora_prefill_route_ag_int32_int64<<<blockDim, nullptr, stream>>>(
            expandedRowIdx, routedTopkIds, tokenLoraIndices, adapterEnabled,
            localCount, errorPerCore, canonicalRows, localRows, numTokens,
            numAdapters, topK, numExperts, groupPitch, firstExpertIdx, blockDim,
            routeTileRows);
    } else if (enabledType == 0) {
        moe_lora_prefill_route_ag_int64_bool<<<blockDim, nullptr, stream>>>(
            expandedRowIdx, routedTopkIds, tokenLoraIndices, adapterEnabled,
            localCount, errorPerCore, canonicalRows, localRows, numTokens,
            numAdapters, topK, numExperts, groupPitch, firstExpertIdx, blockDim,
            routeTileRows);
    } else if (enabledType == 1) {
        moe_lora_prefill_route_ag_int64_int32<<<blockDim, nullptr, stream>>>(
            expandedRowIdx, routedTopkIds, tokenLoraIndices, adapterEnabled,
            localCount, errorPerCore, canonicalRows, localRows, numTokens,
            numAdapters, topK, numExperts, groupPitch, firstExpertIdx, blockDim,
            routeTileRows);
    } else {
        moe_lora_prefill_route_ag_int64_int64<<<blockDim, nullptr, stream>>>(
            expandedRowIdx, routedTopkIds, tokenLoraIndices, adapterEnabled,
            localCount, errorPerCore, canonicalRows, localRows, numTokens,
            numAdapters, topK, numExperts, groupPitch, firstExpertIdx, blockDim,
            routeTileRows);
    }
}

extern void moe_lora_prefill_route_alltoall_impl(
    void* stream, void* expertCount, void* exchangedLoraIndices,
    void* adapterEnabled, void* localCount, void* errorPerCore,
    uint32_t numRows, uint32_t numAdapters, uint32_t numExperts,
    uint32_t groupPitch, uint32_t blockDim, uint32_t routeTileRows, bool count64,
    uint32_t enabledType)
{
    if (!count64 && enabledType == 0) {
        moe_lora_prefill_route_a2a_int32_bool<<<blockDim, nullptr, stream>>>(
            expertCount, exchangedLoraIndices, adapterEnabled, localCount,
            errorPerCore, numRows, numAdapters, numExperts, groupPitch, blockDim,
            routeTileRows);
    } else if (!count64 && enabledType == 1) {
        moe_lora_prefill_route_a2a_int32_int32<<<blockDim, nullptr, stream>>>(
            expertCount, exchangedLoraIndices, adapterEnabled, localCount,
            errorPerCore, numRows, numAdapters, numExperts, groupPitch, blockDim,
            routeTileRows);
    } else if (!count64) {
        moe_lora_prefill_route_a2a_int32_int64<<<blockDim, nullptr, stream>>>(
            expertCount, exchangedLoraIndices, adapterEnabled, localCount,
            errorPerCore, numRows, numAdapters, numExperts, groupPitch, blockDim,
            routeTileRows);
    } else if (enabledType == 0) {
        moe_lora_prefill_route_a2a_int64_bool<<<blockDim, nullptr, stream>>>(
            expertCount, exchangedLoraIndices, adapterEnabled, localCount,
            errorPerCore, numRows, numAdapters, numExperts, groupPitch, blockDim,
            routeTileRows);
    } else if (enabledType == 1) {
        moe_lora_prefill_route_a2a_int64_int32<<<blockDim, nullptr, stream>>>(
            expertCount, exchangedLoraIndices, adapterEnabled, localCount,
            errorPerCore, numRows, numAdapters, numExperts, groupPitch, blockDim,
            routeTileRows);
    } else {
        moe_lora_prefill_route_a2a_int64_int64<<<blockDim, nullptr, stream>>>(
            expertCount, exchangedLoraIndices, adapterEnabled, localCount,
            errorPerCore, numRows, numAdapters, numExperts, groupPitch, blockDim,
            routeTileRows);
    }
}

extern void moe_lora_prefill_prefix_b1_impl(
    void* stream, void* localCount, void* corePrefix, void* groupTotal,
    uint32_t numGroups, uint32_t groupPitch, uint32_t numCores,
    uint32_t blockDim, uint32_t prefixTileGroups)
{
    moe_lora_prefill_prefix_b1<<<blockDim, nullptr, stream>>>(
        localCount, corePrefix, groupTotal, numGroups, groupPitch,
        numCores, blockDim, prefixTileGroups);
}

extern void moe_lora_prefill_prefix_b2_impl(
    void* stream, void* groupTotal, void* errorPerCore, void* groupStart,
    void* groupCountI64, void* routeError, uint32_t numGroups,
    uint32_t groupPitch, uint32_t numCores, uint32_t numRows)
{
    moe_lora_prefill_prefix_b2<<<1, nullptr, stream>>>(
        groupTotal, errorPerCore, groupStart, groupCountI64, routeError,
        numGroups, groupPitch, numCores, numRows);
}

#define MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(DATA_NAME, INDEX_NAME, ENABLED_NAME)                    \
    moe_lora_prefill_scatter_ag_##DATA_NAME##_##INDEX_NAME##_##ENABLED_NAME                        \
        <<<blockDim, nullptr, stream>>>(                                                           \
            x, expandedRowIdx, routedTopkIds, tokenLoraIndices, adapterEnabled, corePrefix,        \
            groupStart, groupTotal, groupedX, permRecord, canonicalRows, numRows, numTokens,       \
            numAdapters,                                                                           \
            topK, numExperts, numGroups, groupPitch, inputWidth, groupedStride,                    \
            firstExpertIdx, blockDim, routeTileRows, columnTileElements)

extern void moe_lora_prefill_scatter_allgather_impl(
    void* stream, void* x, void* expandedRowIdx, void* routedTopkIds,
    void* tokenLoraIndices, void* adapterEnabled, void* corePrefix,
    void* groupStart, void* groupTotal, void* groupedX, void* permRecord,
    uint32_t canonicalRows, uint32_t numRows, uint32_t numTokens,
    uint32_t numAdapters, uint32_t topK, uint32_t numExperts,
    uint32_t numGroups, uint32_t groupPitch, uint32_t inputWidth,
    uint32_t groupedStride, int64_t firstExpertIdx, uint32_t blockDim,
    uint32_t routeTileRows, uint32_t columnTileElements,
    bool isBfloat16, bool index64, uint32_t enabledType)
{
    if (!isBfloat16 && !index64 && enabledType == 0) {
        MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(fp16, int32, bool);
    } else if (!isBfloat16 && !index64 && enabledType == 1) {
        MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(fp16, int32, int32);
    } else if (!isBfloat16 && !index64) {
        MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(fp16, int32, int64);
    } else if (!isBfloat16 && index64 && enabledType == 0) {
        MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(fp16, int64, bool);
    } else if (!isBfloat16 && index64 && enabledType == 1) {
        MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(fp16, int64, int32);
    } else if (!isBfloat16) {
        MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(fp16, int64, int64);
    } else if (!index64 && enabledType == 0) {
        MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(bf16, int32, bool);
    } else if (!index64 && enabledType == 1) {
        MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(bf16, int32, int32);
    } else if (!index64) {
        MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(bf16, int32, int64);
    } else if (enabledType == 0) {
        MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(bf16, int64, bool);
    } else if (enabledType == 1) {
        MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(bf16, int64, int32);
    } else {
        MOE_LORA_PREFILL_SCATTER_AG_LAUNCH(bf16, int64, int64);
    }
}

#define MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(DATA_NAME, COUNT_NAME, ENABLED_NAME)              \
    moe_lora_prefill_scatter_a2a_##DATA_NAME##_##COUNT_NAME##_##ENABLED_NAME                  \
        <<<blockDim, nullptr, stream>>>(                                                       \
            x, expertCount, exchangedLoraIndices, adapterEnabled, corePrefix, groupStart,     \
            groupedX, permRecord, numRows, numAdapters, numExperts, numGroups, groupPitch,    \
            inputWidth, groupedStride, blockDim, routeTileRows, columnTileElements)

extern void moe_lora_prefill_scatter_alltoall_impl(
    void* stream, void* x, void* expertCount, void* exchangedLoraIndices,
    void* adapterEnabled, void* corePrefix, void* groupStart,
    void* groupedX, void* permRecord, uint32_t numRows,
    uint32_t numAdapters, uint32_t numExperts, uint32_t numGroups,
    uint32_t groupPitch, uint32_t inputWidth, uint32_t groupedStride,
    uint32_t blockDim, uint32_t routeTileRows, uint32_t columnTileElements,
    bool isBfloat16, bool count64,
    uint32_t enabledType)
{
    if (!isBfloat16 && !count64 && enabledType == 0) {
        MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(fp16, int32, bool);
    } else if (!isBfloat16 && !count64 && enabledType == 1) {
        MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(fp16, int32, int32);
    } else if (!isBfloat16 && !count64) {
        MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(fp16, int32, int64);
    } else if (!isBfloat16 && count64 && enabledType == 0) {
        MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(fp16, int64, bool);
    } else if (!isBfloat16 && count64 && enabledType == 1) {
        MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(fp16, int64, int32);
    } else if (!isBfloat16) {
        MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(fp16, int64, int64);
    } else if (!count64 && enabledType == 0) {
        MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(bf16, int32, bool);
    } else if (!count64 && enabledType == 1) {
        MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(bf16, int32, int32);
    } else if (!count64) {
        MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(bf16, int32, int64);
    } else if (enabledType == 0) {
        MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(bf16, int64, bool);
    } else if (enabledType == 1) {
        MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(bf16, int64, int32);
    } else {
        MOE_LORA_PREFILL_SCATTER_A2A_LAUNCH(bf16, int64, int64);
    }
}

extern void moe_lora_prefill_gather_by_perm_impl(
    void* stream, void* source, void* permRecord, void* groupedX,
    uint32_t numRows, uint32_t inputWidth, uint32_t groupedStride,
    uint32_t blockDim, uint32_t routeTileRows, uint32_t columnTileElements,
    bool isBfloat16)
{
    if (isBfloat16) {
        moe_lora_prefill_gather_bf16<<<blockDim, nullptr, stream>>>(
            source, permRecord, groupedX, numRows, inputWidth,
            groupedStride, blockDim, routeTileRows, columnTileElements);
    } else {
        moe_lora_prefill_gather_fp16<<<blockDim, nullptr, stream>>>(
            source, permRecord, groupedX, numRows, inputWidth,
            groupedStride, blockDim, routeTileRows, columnTileElements);
    }
}

extern void moe_lora_prefill_scatter_add_impl(
    void* stream, void* delta, void* permRecord, void* y,
    uint32_t numRows, uint32_t deltaWidth, uint32_t outputWidth,
    uint32_t outputOffset, uint32_t blockDim, uint32_t routeTileRows,
    uint32_t scatterAddTileElements, bool isBfloat16)
{
    if (isBfloat16) {
        moe_lora_prefill_scatter_add_bf16<<<blockDim, nullptr, stream>>>(
            delta, permRecord, y, numRows, deltaWidth, outputWidth,
            outputOffset, blockDim, routeTileRows, scatterAddTileElements);
    } else {
        moe_lora_prefill_scatter_add_fp16<<<blockDim, nullptr, stream>>>(
            delta, permRecord, y, numRows, deltaWidth, outputWidth,
            outputOffset, blockDim, routeTileRows, scatterAddTileElements);
    }
}
}  // namespace vllm_ascend
