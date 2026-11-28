// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr uint32_t kHeads = 4;
constexpr uint32_t kPage = 128;
constexpr uint32_t kTile = 32;
constexpr uint32_t kD = 576;
constexpr uint32_t kV = 512;
constexpr uint32_t kLane = 64;  // FP32 elements in one AIV repeat
constexpr uint32_t kBlock = 8;  // FP32 elements per 32-byte vector block
constexpr uint32_t kPart = kV + kBlock;
constexpr float kInfinity = __builtin_inff();

template <HardEvent event>
__aicore__ inline void Fence()
{
    const auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

// Each split emits an unnormalized numerator and its softmax statistics.
// A negative denominator flags an invalid physical page to the merge stage.
class Writer {
public:
    __aicore__ inline void Init(TPipe& pipe, GM_ADDR out, GM_ADDR lse, GM_ADDR partials, bool writeLse)
    {
        out_.SetGlobalBuffer((__gm__ bfloat16_t*)out);
        writeLse_ = writeLse;
        if (writeLse) lse_.SetGlobalBuffer((__gm__ float*)lse);
        if (partials != nullptr) parts_.SetGlobalBuffer((__gm__ float*)partials);
        pipe.InitBuffer(queue_, 1, kPart * sizeof(float));
    }

    __aicore__ inline void Partial(uint64_t index, LocalTensor<float> acc, float maximum, float sum)
    {
        auto data = queue_.AllocTensor<float>();
        Adds(data, acc, 0.0f, kV);
        Duplicate(data[kV], 0.0f, kBlock);
        Fence<HardEvent::V_S>();
        data.SetValue(kV, maximum);
        data.SetValue(kV + 1, sum);
        Fence<HardEvent::S_V>();
        queue_.EnQue(data);
        data = queue_.DeQue<float>();
        DataCopy(parts_[index * kPart], data, kPart);
        queue_.FreeTensor(data);
    }

    __aicore__ inline void Final(uint64_t job, LocalTensor<float> acc, float maximum, float sum)
    {
        if (sum <= 0.0f) {
            Duplicate(acc, 0.0f, kV);
        } else {
            Muls(acc, acc, 1.0f / sum, kV);
        }
        PipeBarrier<PIPE_V>();
        auto out = queue_.AllocTensor<bfloat16_t>();
        Cast(out, acc, RoundMode::CAST_RINT, kV);
        queue_.EnQue(out);
        out = queue_.DeQue<bfloat16_t>();
        DataCopy(out_[job * kV], out, kV);
        queue_.FreeTensor(out);
        if (writeLse_) {
            auto data = queue_.AllocTensor<float>();
            if (sum <= 0.0f) {
                Duplicate(data, kInfinity, kBlock);
            } else {
                Duplicate(data, sum, kBlock);
                PipeBarrier<PIPE_V>();
                Ln(data, data, kBlock);
                PipeBarrier<PIPE_V>();
                Adds(data, data, maximum, kBlock);
            }
            queue_.EnQue(data);
            data = queue_.DeQue<float>();
            DataCopyPad(lse_[job], data, DataCopyExtParams{1, sizeof(float), 0, 0, 0});
            queue_.FreeTensor(data);
        }
    }
private:
    bool writeLse_;
    GlobalTensor<bfloat16_t> out_;
    GlobalTensor<float> lse_, parts_;
    TQue<QuePosition::VECOUT, 1> queue_;
};

class DecodeSplit {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR cache, GM_ADDR table, GM_ADDR lengths,
                                GM_ADDR partials, GM_ADDR output, GM_ADDR lse,
                                uint32_t batches, uint32_t seq, uint32_t pages, uint32_t width,
                                uint32_t splits, float scale, bool causal, bool writeLse)
    {
        q_.SetGlobalBuffer((__gm__ bfloat16_t*)q);
        cache_.SetGlobalBuffer((__gm__ bfloat16_t*)cache);
        table_.SetGlobalBuffer((__gm__ int32_t*)table);
        lengths_.SetGlobalBuffer((__gm__ int32_t*)lengths);
        batches_ = batches;
        seq_ = seq;
        pages_ = pages;
        width_ = width;
        splits_ = splits;
        scale_ = scale;
        causal_ = causal;
        writer_.Init(pipe_, output, lse, partials, writeLse);
        pipe_.InitBuffer(input_, 1, kTile * kD * sizeof(bfloat16_t));
        pipe_.InitBuffer(keys_, kTile * kD * sizeof(float));
        pipe_.InitBuffer(query_, kD * sizeof(float));
        pipe_.InitBuffer(dot_, kTile * kLane * sizeof(float));
        pipe_.InitBuffer(acc_, kV * sizeof(float));
        pipe_.InitBuffer(scores_, (kTile + kBlock) * sizeof(float));
        pipe_.InitBuffer(stats_, kBlock * sizeof(float));
        pipe_.InitBuffer(broadcast_, kTile * kBlock * sizeof(float));
    }

    __aicore__ inline void Run()
    {
        const uint64_t rows = static_cast<uint64_t>(batches_) * seq_ * kHeads;
        for (uint64_t task = GetBlockIdx(); task < rows * splits_; task += GetBlockNum()) {
            const uint64_t job = task / splits_;
            const uint32_t split = task % splits_;
            const uint32_t batch = job / (seq_ * kHeads);
            const uint32_t position = (job / kHeads) % seq_;
            const int32_t length = lengths_.GetValue(batch);
            const bool valid = length >= 0 && static_cast<uint32_t>(length) <=
                               static_cast<uint64_t>(width_) * kPage;
            const int64_t end = causal_ ? static_cast<int64_t>(length) - seq_ + position + 1 : length;
            const uint32_t tokens = valid && end > 0 ? static_cast<uint32_t>(end) : 0;
            const uint32_t tiles = (tokens + kTile - 1) / kTile;
            const uint32_t first = static_cast<uint64_t>(tiles) * split / splits_;
            const uint32_t last = static_cast<uint64_t>(tiles) * (split + 1) / splits_;
            auto acc = acc_.Get<float>();
            Duplicate(acc, 0.0f, kV);
            PipeBarrier<PIPE_V>();
            float maximum = -kInfinity;
            float sum = 0.0f;
            if (first < last) Read(q_[job * kD], query_.Get<float>(), kD);
            uint32_t logicalPage = ~0U;
            uint32_t physicalPage = 0;
            for (uint32_t t = first; t < last; ++t) {
                const uint32_t begin = t * kTile;
                const uint32_t p = begin / kPage;
                if (p != logicalPage) {
                    const int32_t physical = table_.GetValue(static_cast<uint64_t>(batch) * width_ + p);
                    if (physical < 0 || static_cast<uint32_t>(physical) >= pages_) {
                        sum = -1.0f;
                        break;
                    }
                    logicalPage = p;
                    physicalPage = physical;
                }
                const uint32_t n = tokens - begin < kTile ? tokens - begin : kTile;
                const uint64_t k = static_cast<uint64_t>(physicalPage) * kPage + (begin % kPage);
                Read(cache_[k * kD], keys_.Get<float>(), n * kD);
                Compute(n, maximum, sum);
            }
            if (splits_ == 1) {
                writer_.Final(job, acc, maximum, sum);
            } else {
                writer_.Partial(task, acc, maximum, sum);
            }
        }
    }
private:
    __aicore__ inline void Read(GlobalTensor<bfloat16_t> source, LocalTensor<float> dst, uint32_t n)
    {
        auto in = input_.AllocTensor<bfloat16_t>();
        DataCopy(in, source, n);
        input_.EnQue(in);
        in = input_.DeQue<bfloat16_t>();
        Cast(dst, in, RoundMode::CAST_NONE, n);
        input_.FreeTensor(in);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint32_t n, float& maximum, float& sum)
    {
        auto q = query_.Get<float>();
        auto keys = keys_.Get<float>();
        auto dot = dot_.Get<float>();
        auto scores = scores_.Get<float>();
        auto stats = stats_.Get<float>();
        auto broad = broadcast_.Get<float>();
        auto acc = acc_.Get<float>();
        // Each repeat handles a different cache row. The query repeat stride
        // of zero broadcasts its 64 consecutive channels to all 32 tokens.
        const BinaryRepeatParams qk{1, 1, 1, kLane / kBlock, kD / kBlock, 0};
        Mul(dot, keys, q, kLane, n, qk);
        for (uint32_t d = kLane; d < kD; d += kLane) {
            PipeBarrier<PIPE_V>();
            MulAddDst(dot, keys[d], q[d], kLane, n, qk);
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(scores, dot, kLane, n, 1, 1, kLane / kBlock);
        PipeBarrier<PIPE_V>();
        Muls(scores, scores, scale_, n);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(stats, scores, n, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        Fence<HardEvent::V_S>();
        const float tileMax = stats.GetValue(0);
        const float nextMax = tileMax > maximum ? tileMax : maximum;
        const float safeMax = nextMax == -kInfinity ? 0.0f : nextMax;
        scores.SetValue(n, sum == 0.0f ? -kInfinity : maximum);
        Fence<HardEvent::S_V>();
        Adds(scores, scores, -safeMax, n + 1);
        PipeBarrier<PIPE_V>();
        Exp(scores, scores, n + 1);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(stats, scores, n, 1, 1, 1, 8);
        Brcb(broad, scores, (n + kBlock - 1) / kBlock, {1, 8});
        Fence<HardEvent::V_S>();
        const float previous = scores.GetValue(n);
        sum = sum * previous + stats.GetValue(0);
        maximum = nextMax;
        Muls(acc, acc, previous, kV);
        // Brcb supplies one equal-valued FP32 block per token. Repeat steps
        // to the next weight and block-stride zero reuses it across 64 lanes.
        const BinaryRepeatParams pv{1, 1, 0, kD / kBlock, kD / kBlock, 1};
        for (uint32_t d = 0; d < kV; d += kLane) {
            Mul(keys[d], keys[d], broad, kLane, n, pv);
        }
        // Tree reduction of complete (strided) cache rows. The 64 unused rope
        // lanes are added as padding and never enter the output accumulator.
        for (uint32_t halfRows = kTile / 2; halfRows != 0; halfRows /= 2) {
            if (n > halfRows) {
                PipeBarrier<PIPE_V>();
                Add(keys, keys, keys[halfRows * kD], (n - halfRows) * kD);
                n = halfRows;
            }
        }
        PipeBarrier<PIPE_V>();
        Add(acc, acc, keys, kV);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::S_V>();
    }

    uint32_t batches_, seq_, pages_, width_, splits_;
    float scale_;
    bool causal_;
    TPipe pipe_;
    Writer writer_;
    GlobalTensor<bfloat16_t> q_, cache_;
    GlobalTensor<int32_t> table_, lengths_;
    TQue<QuePosition::VECIN, 1> input_;
    TBuf<QuePosition::VECCALC> keys_, query_, dot_, acc_, scores_, stats_, broadcast_;
};

class DecodeMerge {
public:
    __aicore__ inline void Run(GM_ADDR partials, GM_ADDR output, GM_ADDR lse, uint32_t rows,
                               uint32_t splits, bool writeLse)
    {
        parts_.SetGlobalBuffer((__gm__ float*)partials);
        writer_.Init(pipe_, output, lse, nullptr, writeLse);
        pipe_.InitBuffer(input_, 1, kV * sizeof(float));
        pipe_.InitBuffer(acc_, kV * sizeof(float));
        pipe_.InitBuffer(weight_, kTile * sizeof(float));
        pipe_.InitBuffer(temp_, kV * sizeof(float));
        auto acc = acc_.Get<float>();
        auto w = weight_.Get<float>();
        for (uint64_t j = GetBlockIdx(); j < rows; j += GetBlockNum()) {
            const uint64_t base = j * splits * kPart;
            float maximum = -kInfinity;
            bool valid = true;
            for (uint32_t s = 0; s < splits; ++s) {
                const uint64_t stats = base + static_cast<uint64_t>(s) * kPart + kV;
                const float m = parts_.GetValue(stats);
                const float total = parts_.GetValue(stats + 1);
                maximum = m > maximum ? m : maximum;
                valid = valid && !(total < 0.0f);
            }
            const float safeMax = maximum == -kInfinity ? 0.0f : maximum;
            float sum = 0.0f;
            Duplicate(acc, 0.0f, kV);
            PipeBarrier<PIPE_V>();
            for (uint32_t first = 0; valid && first < splits; first += kTile) {
                const uint32_t n = splits - first < kTile ? splits - first : kTile;
                for (uint32_t i = 0; i < n; ++i) {
                    const uint64_t stats = base + static_cast<uint64_t>(first + i) * kPart + kV;
                    w.SetValue(i, parts_.GetValue(stats) - safeMax);
                }
                Fence<HardEvent::S_V>();
                Exp(w, w, n);
                Fence<HardEvent::V_S>();
                for (uint32_t i = 0; i < n; ++i) {
                    const uint64_t offset = base + static_cast<uint64_t>(first + i) * kPart;
                    const float factor = w.GetValue(i);
                    sum += factor * parts_.GetValue(offset + kV + 1);
                    auto data = input_.AllocTensor<float>();
                    DataCopy(data, parts_[offset], kV);
                    input_.EnQue(data);
                    data = input_.DeQue<float>();
                    auto tmp = temp_.Get<float>();
                    Muls(tmp, data, factor, kV);
                    input_.FreeTensor(data);
                    PipeBarrier<PIPE_V>();
                    Add(acc, acc, tmp, kV);
                    PipeBarrier<PIPE_V>();
                }
            }
            writer_.Final(j, acc, maximum, valid ? sum : -1.0f);
        }
    }
private:
    TPipe pipe_;
    Writer writer_;
    GlobalTensor<float> parts_;
    TQue<QuePosition::VECIN, 1> input_;
    TBuf<QuePosition::VECCALC> acc_, weight_, temp_;
};
}

extern "C" __global__ __aicore__ void mla_a3_decode_split(
    GM_ADDR q, GM_ADDR cache, GM_ADDR table, GM_ADDR lengths, GM_ADDR partials, GM_ADDR output, GM_ADDR lse,
    uint32_t batches, uint32_t query_len, uint32_t pages, uint32_t table_width, uint32_t splits,
    float scale, uint32_t causal, uint32_t write_lse)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    DecodeSplit op;
    op.Init(q, cache, table, lengths, partials, output, lse, batches, query_len, pages, table_width,
            splits, scale, causal, write_lse);
    op.Run();
}

extern "C" __global__ __aicore__ void mla_a3_decode_merge(
    GM_ADDR partials, GM_ADDR output, GM_ADDR lse, uint32_t rows, uint32_t splits, uint32_t write_lse)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    DecodeMerge op;
    op.Run(partials, output, lse, rows, splits, write_lse);
}
