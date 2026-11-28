// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace {
using namespace AscendC;

constexpr uint32_t kKeyDim = 576;
constexpr uint32_t kValueDim = 512;
constexpr uint32_t kTile = 16;
constexpr uint32_t kFloatBlock = 32 / sizeof(float);
constexpr float kInfinity = __builtin_inff();

template <HardEvent event>
__aicore__ inline void SyncScalarVector()
{
    const auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

struct Shape {
    uint32_t rows, heads, batches, pages, blockSize, tableWidth;
};

// One AIV owns an entire (query, head) result. It streams tiles bounded by a
// physical page, so a row can cross any number of nonconsecutive cache pages.
template <typename T>
class FlashMla {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR cache, GM_ADDR table, GM_ADDR bounds,
                                GM_ADDR sinks, GM_ADDR output, GM_ADDR lse, Shape shape,
                                float scale, bool hasSinks, bool writeLse)
    {
        shape_ = shape;
        scale_ = scale;
        hasSinks_ = hasSinks;
        writeLse_ = writeLse;
        query_.SetGlobalBuffer((__gm__ T*)q);
        cache_.SetGlobalBuffer((__gm__ T*)cache);
        table_.SetGlobalBuffer((__gm__ int32_t*)table);
        bounds_.SetGlobalBuffer((__gm__ int32_t*)bounds);
        output_.SetGlobalBuffer((__gm__ T*)output);
        if (hasSinks_) {
            sinks_.SetGlobalBuffer((__gm__ float*)sinks);
        }
        if (writeLse_) {
            lse_.SetGlobalBuffer((__gm__ float*)lse);
        }
        pipe_.InitBuffer(inputQ_, 1, kTile * kKeyDim * sizeof(T));
        pipe_.InitBuffer(outputQ_, 1, kValueDim * sizeof(T));
        pipe_.InitBuffer(qBuf_, kKeyDim * sizeof(float));
        pipe_.InitBuffer(kvBuf_, kTile * kKeyDim * sizeof(float));
        pipe_.InitBuffer(accBuf_, kValueDim * sizeof(float));
        pipe_.InitBuffer(productBuf_, kKeyDim * sizeof(float));
        pipe_.InitBuffer(reduceBuf_, kKeyDim * sizeof(float));
        pipe_.InitBuffer(scoresBuf_, kTile * kFloatBlock * sizeof(float));
        // One extra weight rescales all previously accumulated tiles.
        pipe_.InitBuffer(weightsBuf_, (kTile + kFloatBlock) * sizeof(float));
    }

    __aicore__ inline void Run()
    {
        const uint64_t jobs = static_cast<uint64_t>(shape_.rows) * shape_.heads;
        for (uint64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum()) {
            const uint64_t row = job / shape_.heads;
            const int32_t batch = bounds_.GetValue(3 * row);
            const int32_t begin = bounds_.GetValue(3 * row + 1);
            const int32_t end = bounds_.GetValue(3 * row + 2);
            auto acc = accBuf_.Get<float>();
            Duplicate(acc, 0.0f, kValueDim);
            PipeBarrier<PIPE_V>();
            float total = 0.0f;
            float maximum = -kInfinity;
            bool valid = batch >= 0 && static_cast<uint32_t>(batch) < shape_.batches &&
                         begin >= 0 && end > begin && shape_.blockSize != 0 &&
                         static_cast<uint64_t>(end) <=
                             static_cast<uint64_t>(shape_.tableWidth) * shape_.blockSize;
            if (valid) {
                if (hasSinks_) {
                    maximum = sinks_.GetValue(job % shape_.heads);
                    total = maximum == -kInfinity ? 0.0f : 1.0f;
                }
                CopyCast(query_[job * kKeyDim], qBuf_.Get<float>(), kKeyDim);
                valid = AccumulatePages(batch, begin, end, maximum, total);
            }
            Store(job, valid, maximum, total);
        }
    }

private:
    __aicore__ inline void CopyCast(GlobalTensor<T> source, LocalTensor<float> dest, uint32_t count)
    {
        auto input = inputQ_.template AllocTensor<T>();
        DataCopy(input, source, count);
        inputQ_.EnQue(input);
        input = inputQ_.template DeQue<T>();
        Cast(dest, input, RoundMode::CAST_NONE, count);
        inputQ_.FreeTensor(input);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool AccumulatePages(int32_t batch, uint32_t begin, uint32_t end,
                                          float& maximum, float& total)
    {
        uint32_t pos = begin;
        while (pos < end) {
            const uint32_t pageOffset = pos % shape_.blockSize;
            const uint64_t tableIndex = static_cast<uint64_t>(batch) * shape_.tableWidth +
                                        pos / shape_.blockSize;
            const int32_t page = table_.GetValue(tableIndex);
            if (page < 0 || static_cast<uint32_t>(page) >= shape_.pages) {
                return false;
            }
            const uint32_t pageRemaining = shape_.blockSize - pageOffset;
            const uint32_t length = end - pos < pageRemaining ? end - pos : pageRemaining;
            const uint32_t pageEnd = pos + length;
            uint64_t physical = static_cast<uint64_t>(page) * shape_.blockSize + pageOffset;
            while (pos < pageEnd) {
                const uint32_t count = pageEnd - pos < kTile ? pageEnd - pos : kTile;
                CopyCast(cache_[physical * kKeyDim], kvBuf_.Get<float>(), count * kKeyDim);
                UpdateTile(count, maximum, total);
                pos += count;
                physical += count;
            }
        }
        return true;
    }

    __aicore__ inline void UpdateTile(uint32_t count, float& maximum, float& total)
    {
        auto q = qBuf_.Get<float>();
        auto kv = kvBuf_.Get<float>();
        auto tmp = productBuf_.Get<float>();
        auto scores = scoresBuf_.Get<float>();
        auto reduce = reduceBuf_.Get<float>();
        auto weights = weightsBuf_.Get<float>();
        auto acc = accBuf_.Get<float>();

        for (uint32_t i = 0; i < count; ++i) {
            Mul(tmp, q, kv[i * kKeyDim], kKeyDim);
            PipeBarrier<PIPE_V>();
            // Each reduction destination starts on a 32-byte vector block.
            ReduceSum<float>(scores[i * kFloatBlock], tmp, reduce, kKeyDim);
            PipeBarrier<PIPE_V>();
        }
        SyncScalarVector<HardEvent::V_S>();
        float nextMax = maximum;
        for (uint32_t i = 0; i < count; ++i) {
            const float score = scores.GetValue(i * kFloatBlock) * scale_;
            weights.SetValue(i, score);
            nextMax = score > nextMax ? score : nextMax;
        }
        // A tile containing only -inf contributes no mass. Subtracting -inf
        // would produce NaNs and corrupt subsequent tiles with finite logits.
        const float safeMax = nextMax == -kInfinity ? 0.0f : nextMax;
        for (uint32_t i = 0; i < count; ++i) {
            weights.SetValue(i, weights.GetValue(i) - safeMax);
        }
        weights.SetValue(count, total == 0.0f ? -kInfinity : maximum - safeMax);
        SyncScalarVector<HardEvent::S_V>();
        Exp(weights, weights, count + 1);
        SyncScalarVector<HardEvent::V_S>();

        const float previousWeight = weights.GetValue(count);
        Muls(acc, acc, previousWeight, kValueDim);
        total *= previousWeight;
        for (uint32_t i = 0; i < count; ++i) {
            const float weight = weights.GetValue(i);
            total += weight;
            // Value is the first 512 channels of the SAME cache row as Key.
            Muls(tmp, kv[i * kKeyDim], weight, kValueDim);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, tmp, kValueDim);
            PipeBarrier<PIPE_V>();
        }
        maximum = nextMax;
        SyncScalarVector<HardEvent::S_V>();
    }

    __aicore__ inline void Store(uint64_t job, bool valid, float maximum, float total)
    {
        auto acc = accBuf_.Get<float>();
        const bool empty = !valid || total == 0.0f;
        if (empty) {
            Duplicate(acc, 0.0f, kValueDim);
        } else {
            Muls(acc, acc, 1.0f / total, kValueDim);
        }
        PipeBarrier<PIPE_V>();
        auto out = outputQ_.template AllocTensor<T>();
        Cast(out, acc, RoundMode::CAST_RINT, kValueDim);
        outputQ_.EnQue(out);
        out = outputQ_.template DeQue<T>();
        DataCopy(output_[job * kValueDim], out, kValueDim);
        outputQ_.FreeTensor(out);
        if (writeLse_) {
            auto value = outputQ_.template AllocTensor<float>();
            if (empty) {
                Duplicate(value, kInfinity, kFloatBlock);
            } else {
                Duplicate(value, total, kFloatBlock);
                PipeBarrier<PIPE_V>();
                Ln(value, value, kFloatBlock);
                PipeBarrier<PIPE_V>();
                Adds(value, value, maximum, kFloatBlock);
            }
            outputQ_.EnQue(value);
            value = outputQ_.template DeQue<float>();
            // Write exactly one FP32 even when adjacent heads share a GM block.
            const DataCopyExtParams oneValue{1, sizeof(float), 0, 0, 0};
            DataCopyPad(lse_[job], value, oneValue);
            outputQ_.FreeTensor(value);
        }
    }

    Shape shape_;
    float scale_;
    bool hasSinks_, writeLse_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQ_;
    TQue<QuePosition::VECOUT, 1> outputQ_;
    TBuf<QuePosition::VECCALC> qBuf_, kvBuf_, accBuf_, productBuf_, reduceBuf_, scoresBuf_, weightsBuf_;
    GlobalTensor<T> query_, cache_, output_;
    GlobalTensor<int32_t> table_, bounds_;
    GlobalTensor<float> sinks_, lse_;
};
}  // namespace

extern "C" __global__ __aicore__ void flash_mla_a3_fp16(
    GM_ADDR q, GM_ADDR cache, GM_ADDR table, GM_ADDR row_bounds, GM_ADDR sinks, GM_ADDR output,
    GM_ADDR lse, uint32_t rows, uint32_t heads, uint32_t batches, uint32_t pages,
    uint32_t block_size, uint32_t table_width, float scale, uint32_t has_sinks, uint32_t write_lse)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FlashMla<half> op;
    op.Init(q, cache, table, row_bounds, sinks, output, lse,
            {rows, heads, batches, pages, block_size, table_width}, scale, has_sinks, write_lse);
    op.Run();
}

extern "C" __global__ __aicore__ void flash_mla_a3_bf16(
    GM_ADDR q, GM_ADDR cache, GM_ADDR table, GM_ADDR row_bounds, GM_ADDR sinks, GM_ADDR output,
    GM_ADDR lse, uint32_t rows, uint32_t heads, uint32_t batches, uint32_t pages,
    uint32_t block_size, uint32_t table_width, float scale, uint32_t has_sinks, uint32_t write_lse)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FlashMla<bfloat16_t> op;
    op.Init(q, cache, table, row_bounds, sinks, output, lse,
            {rows, heads, batches, pages, block_size, table_width}, scale, has_sinks, write_lse);
    op.Run();
}
