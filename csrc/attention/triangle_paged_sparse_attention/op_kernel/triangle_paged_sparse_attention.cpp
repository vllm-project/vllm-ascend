/*
 * Copyright (c) 2026 TriangleMix contributors.
 * This program is licensed under CANN Open Software License Agreement
 * Version 2.0. See LICENSE in the repository root.
 */
#include "kernel_operator.h"

#include "fia_paged_kv_address.h"
#include "triangle_paged_fia_fast_path.h"
#include "triangle_schedule.h"
#include "triangle_paged_sparse_attention_tiling.h"

namespace TrianglePaged {

using namespace AscendC;

/*
 * This is a deliberately bounded numerical reference kernel.  Its purpose is
 * to establish the paged-KV and Triangle softmax correctness contract before
 * the hot path is replaced by the FIA Cube/Vector pipeline.
 *
 * It is a real attention implementation: skipped Middle tokens are never read
 * for QK or PV, and sink/local logits share one softmax normalization domain.
 * It is not a performance claim: the dot-product and scalar reductions use
 * LocalTensor::GetValue().
 */
#ifdef TRIANGLE_ENABLE_AIV_REFERENCE
constexpr uint32_t kReferenceImplementation = 1;
constexpr uint32_t kReferenceMaxSequenceLength = 2048;

struct RowSchedule {
    KvInterval interval[2];
    uint32_t count;
    uint32_t keyCount;
};

__aicore__ inline uint32_t MaxU32(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

__aicore__ inline RowSchedule BuildRowSchedule(
    uint32_t queryPosition,
    const TrianglePagedSparseAttentionTilingData& tiling)
{
    RowSchedule result{};
    const uint32_t causalEnd = MinU32(queryPosition + 1, tiling.seqLen);
    const bool sparse =
        queryPosition >= tiling.sparseBegin &&
        queryPosition < tiling.sparseEnd;

    if (!sparse) {
        result.interval[0] = {0, causalEnd};
        result.count = causalEnd == 0 ? 0 : 1;
        result.keyCount = causalEnd;
        return result;
    }

    const uint32_t sinkEnd = MinU32(tiling.sinkTokens, causalEnd);
    const uint32_t localBeginRaw =
        queryPosition > tiling.localWindow
            ? queryPosition - tiling.localWindow
            : 0;

    // Merge overlap so a sink token is normalized and accumulated only once.
    if (localBeginRaw <= sinkEnd) {
        result.interval[0] = {0, causalEnd};
        result.count = causalEnd == 0 ? 0 : 1;
        result.keyCount = causalEnd;
        return result;
    }

    result.interval[0] = {0, sinkEnd};
    result.interval[1] = {localBeginRaw, causalEnd};
    result.count = (sinkEnd == 0 ? 0 : 1) +
                   (localBeginRaw < causalEnd ? 1 : 0);
    result.keyCount =
        sinkEnd + (causalEnd > localBeginRaw
                       ? causalEnd - localBeginRaw
                       : 0);
    if (sinkEnd == 0 && localBeginRaw < causalEnd) {
        result.interval[0] = result.interval[1];
    }
    return result;
}

class TrianglePagedSparseAttentionReference {
public:
    __aicore__ inline void Init(
        GM_ADDR query,
        GM_ADDR keyCache,
        GM_ADDR valueCache,
        GM_ADDR blockTable,
        GM_ADDR attentionOut,
        const TrianglePagedSparseAttentionTilingData& tilingData,
        TPipe* pipe)
    {
        pipe_ = pipe;
        tilingData_ = tilingData;

        const uint64_t queryElements =
            static_cast<uint64_t>(tilingData_.queryTokens) *
            tilingData_.queryHeads * tilingData_.headDim;
        const uint64_t cacheElements =
            static_cast<uint64_t>(tilingData_.physicalPageCount) *
            tilingData_.pageSize * tilingData_.kvHeads * tilingData_.headDim;

        query_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(query), queryElements);
        keyCache_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(keyCache), cacheElements);
        valueCache_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(valueCache), cacheElements);
        blockTable_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(blockTable),
            tilingData_.blockTablePageCapacity);
        output_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(attentionOut),
            queryElements);

        pipe_->InitBuffer(
            queryInQueue_, 1, kHeadDim * sizeof(bfloat16_t));
        pipe_->InitBuffer(
            keyInQueue_, 1, kHeadDim * sizeof(bfloat16_t));
        pipe_->InitBuffer(
            valueInQueue_, 1, kHeadDim * sizeof(bfloat16_t));
        pipe_->InitBuffer(
            outputQueue_, 1, kHeadDim * sizeof(bfloat16_t));

        pipe_->InitBuffer(queryFloatBuffer_, kHeadDim * sizeof(float));
        pipe_->InitBuffer(kvFloatBuffer_, kHeadDim * sizeof(float));
        pipe_->InitBuffer(productBuffer_, kHeadDim * sizeof(float));
        pipe_->InitBuffer(outputFloatBuffer_, kHeadDim * sizeof(float));
        pipe_->InitBuffer(
            scoreBuffer_,
            kReferenceMaxSequenceLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (tilingData_.implementationStatus != kReferenceImplementation ||
            tilingData_.seqLen > kReferenceMaxSequenceLength) {
            return;
        }

        const uint32_t core = GetBlockIdx();
        for (uint32_t task = core; task < tilingData_.taskCount;
             task += tilingData_.blockDim) {
            ProcessTask(task);
        }
    }

private:
    __aicore__ inline float ComputeScore(
        const LocalTensor<float>& queryFloat,
        uint32_t logicalToken,
        uint32_t kvHead)
    {
        const PagedKvLocation location = ResolvePagedBsndLocation(
            blockTable_,
            logicalToken,
            kvHead,
            tilingData_.pageSize,
            tilingData_.kvHeads,
            tilingData_.headDim);

        LocalTensor<bfloat16_t> key =
            keyInQueue_.AllocTensor<bfloat16_t>();
        DataCopy(key, keyCache_[location.elementOffset], kHeadDim);
        keyInQueue_.EnQue<bfloat16_t>(key);
        key = keyInQueue_.DeQue<bfloat16_t>();

        LocalTensor<float> kvFloat = kvFloatBuffer_.Get<float>();
        LocalTensor<float> product = productBuffer_.Get<float>();
        Cast(kvFloat, key, RoundMode::CAST_NONE, kHeadDim);
        PipeBarrier<PIPE_V>();
        Mul(product, queryFloat, kvFloat, kHeadDim);
        PipeBarrier<PIPE_V>();

        float score = 0.0F;
        for (uint32_t d = 0; d < kHeadDim; ++d) {
            score += product.GetValue(d);
        }
        keyInQueue_.FreeTensor(key);
        return score * tilingData_.scale;
    }

    __aicore__ inline void AccumulateValue(
        LocalTensor<float>& outputFloat,
        uint32_t logicalToken,
        uint32_t kvHead,
        float weight)
    {
        const PagedKvLocation location = ResolvePagedBsndLocation(
            blockTable_,
            logicalToken,
            kvHead,
            tilingData_.pageSize,
            tilingData_.kvHeads,
            tilingData_.headDim);

        LocalTensor<bfloat16_t> value =
            valueInQueue_.AllocTensor<bfloat16_t>();
        DataCopy(value, valueCache_[location.elementOffset], kHeadDim);
        valueInQueue_.EnQue<bfloat16_t>(value);
        value = valueInQueue_.DeQue<bfloat16_t>();

        LocalTensor<float> kvFloat = kvFloatBuffer_.Get<float>();
        Cast(kvFloat, value, RoundMode::CAST_NONE, kHeadDim);
        PipeBarrier<PIPE_V>();
        Muls(kvFloat, kvFloat, weight, kHeadDim);
        PipeBarrier<PIPE_V>();
        Add(outputFloat, outputFloat, kvFloat, kHeadDim);
        PipeBarrier<PIPE_V>();
        valueInQueue_.FreeTensor(value);
    }

    __aicore__ inline void ProcessTask(uint32_t task)
    {
        const uint32_t queryRow = task / tilingData_.queryHeads;
        const uint32_t queryHead = task % tilingData_.queryHeads;
        const uint32_t kvHead =
            queryHead / (tilingData_.queryHeads / tilingData_.kvHeads);
        const uint32_t queryPosition = tilingData_.queryStart + queryRow;
        const uint64_t queryOffset =
            (static_cast<uint64_t>(queryRow) * tilingData_.queryHeads +
             queryHead) *
            tilingData_.headDim;

        const RowSchedule schedule =
            BuildRowSchedule(queryPosition, tilingData_);
        if (schedule.count == 0 || schedule.keyCount == 0 ||
            schedule.keyCount > kReferenceMaxSequenceLength) {
            return;
        }

        LocalTensor<bfloat16_t> query =
            queryInQueue_.AllocTensor<bfloat16_t>();
        DataCopy(query, query_[queryOffset], kHeadDim);
        queryInQueue_.EnQue<bfloat16_t>(query);
        query = queryInQueue_.DeQue<bfloat16_t>();

        LocalTensor<float> queryFloat = queryFloatBuffer_.Get<float>();
        LocalTensor<float> scores = scoreBuffer_.Get<float>();
        Cast(queryFloat, query, RoundMode::CAST_NONE, kHeadDim);
        PipeBarrier<PIPE_V>();

        uint32_t scoreIndex = 0;
        float rowMax = -3.402823466e+38F;
        for (uint32_t intervalIndex = 0;
             intervalIndex < schedule.count;
             ++intervalIndex) {
            const KvInterval interval = schedule.interval[intervalIndex];
            for (uint32_t token = interval.begin; token < interval.end;
                 ++token) {
                const float score =
                    ComputeScore(queryFloat, token, kvHead);
                scores.SetValue(scoreIndex, score);
                rowMax = score > rowMax ? score : rowMax;
                ++scoreIndex;
            }
        }
        queryInQueue_.FreeTensor(query);

        Adds(scores, scores, -rowMax, scoreIndex);
        PipeBarrier<PIPE_V>();
        Exp(scores, scores, scoreIndex);
        PipeBarrier<PIPE_V>();

        float rowSum = 0.0F;
        for (uint32_t i = 0; i < scoreIndex; ++i) {
            rowSum += scores.GetValue(i);
        }
        Muls(scores, scores, 1.0F / rowSum, scoreIndex);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> outputFloat =
            outputFloatBuffer_.Get<float>();
        Duplicate(outputFloat, 0.0F, kHeadDim);
        PipeBarrier<PIPE_V>();

        scoreIndex = 0;
        for (uint32_t intervalIndex = 0;
             intervalIndex < schedule.count;
             ++intervalIndex) {
            const KvInterval interval = schedule.interval[intervalIndex];
            for (uint32_t token = interval.begin; token < interval.end;
                 ++token) {
                AccumulateValue(
                    outputFloat,
                    token,
                    kvHead,
                    scores.GetValue(scoreIndex));
                ++scoreIndex;
            }
        }

        LocalTensor<bfloat16_t> output =
            outputQueue_.AllocTensor<bfloat16_t>();
        Cast(output, outputFloat, RoundMode::CAST_RINT, kHeadDim);
        outputQueue_.EnQue<bfloat16_t>(output);
        output = outputQueue_.DeQue<bfloat16_t>();
        DataCopy(output_[queryOffset], output, kHeadDim);
        outputQueue_.FreeTensor(output);
    }

    TPipe* pipe_{nullptr};
    TrianglePagedSparseAttentionTilingData tilingData_{};

    GlobalTensor<bfloat16_t> query_;
    GlobalTensor<bfloat16_t> keyCache_;
    GlobalTensor<bfloat16_t> valueCache_;
    GlobalTensor<int32_t> blockTable_;
    GlobalTensor<bfloat16_t> output_;

    TQue<QuePosition::VECIN, 1> queryInQueue_;
    TQue<QuePosition::VECIN, 1> keyInQueue_;
    TQue<QuePosition::VECIN, 1> valueInQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;

    TBuf<TPosition::VECCALC> queryFloatBuffer_;
    TBuf<TPosition::VECCALC> kvFloatBuffer_;
    TBuf<TPosition::VECCALC> productBuffer_;
    TBuf<TPosition::VECCALC> outputFloatBuffer_;
    TBuf<TPosition::VECCALC> scoreBuffer_;
};
#endif  // TRIANGLE_ENABLE_AIV_REFERENCE

}  // namespace TrianglePaged

extern "C" __global__ __aicore__ void triangle_paged_sparse_attention(
    GM_ADDR query,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR block_table,
    GM_ADDR attention_out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(TrianglePagedSparseAttentionTilingData);
    GET_TILING_DATA(tilingData, tiling);
    __gm__ uint8_t *userWorkspace = AscendC::GetUserWorkspace(workspace);

    TrianglePaged::TrianglePagedFiaFastPath kernel;
    kernel.Init(
        query,
        key_cache,
        value_cache,
        block_table,
        attention_out,
        userWorkspace,
        tilingData);
    kernel.Process();
}
