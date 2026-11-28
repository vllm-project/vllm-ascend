// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace {
using namespace AscendC;
constexpr uint32_t kTile = 16;
constexpr int32_t kMagic = 0x4D4C4133;

struct BatchPlan {
    int32_t start, end, used, kv;
};

__aicore__ inline BatchPlan ReadBatch(const GlobalTensor<int32_t>& plan, uint32_t b)
{
    return {plan.GetValue(4 * b), plan.GetValue(4 * b + 1),
            plan.GetValue(4 * b + 2), plan.GetValue(4 * b + 3)};
}

__aicore__ inline int32_t Clip(int64_t position, int32_t length)
{
    return position < 0 ? 0 : (position > length ? length : static_cast<int32_t>(position));
}

__aicore__ inline void GetBounds(const GlobalTensor<int32_t>& plan, uint32_t batches,
                                 uint32_t tokens, uint32_t token, bool packed,
                                 uint32_t mode, int64_t left, int64_t right, uint32_t capacity,
                                 int32_t& batch, int32_t& begin, int32_t& end)
{
    uint32_t b = 0;
    int32_t physical = 0;
    int32_t local = 0;
    if (packed) {
        // Locate the first sequence whose exclusive end exceeds token.
        uint32_t hi = batches;
        while (b < hi) {
            const uint32_t mid = b + (hi - b) / 2;
            if (plan.GetValue(4 * mid + 1) <= static_cast<int32_t>(token)) {
                b = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (b == batches) {
            return;
        }
        const BatchPlan seq = ReadBatch(plan, b);
        if (seq.start < 0 || seq.start > static_cast<int32_t>(token) ||
            seq.end <= static_cast<int32_t>(token) || seq.end > static_cast<int32_t>(tokens)) {
            return;
        }
        physical = seq.end - seq.start;
        local = static_cast<int32_t>(token) - seq.start;
    } else {
        physical = static_cast<int32_t>(tokens / batches);
        b = token / physical;
        local = token % physical;
    }
    const BatchPlan seq = ReadBatch(plan, b);
    if (seq.used < 0 || seq.used > physical || local >= seq.used ||
        seq.kv <= 0 || static_cast<uint32_t>(seq.kv) > capacity) {
        return;
    }
    batch = static_cast<int32_t>(b);
    const int64_t position = static_cast<int64_t>(seq.kv) - seq.used + local;
    begin = (mode == 4 && left >= 0) ? Clip(position - left, seq.kv) : 0;
    end = mode == 3 ? Clip(position + 1, seq.kv) :
          (mode == 4 && right >= 0 ? Clip(position + right + 1, seq.kv) : seq.kv);
}

template <HardEvent event>
__aicore__ inline void SyncScalar()
{
    const auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

template <uint32_t width>
__aicore__ inline void WriteTile(GlobalTensor<int32_t>& dst, uint32_t start, uint32_t count,
                                TQue<QuePosition::VECOUT, 1>& queue, LocalTensor<int32_t>& data)
{
    // Scalar writes into UB precede MTE3. EnQue/DeQue is a V->MTE3
    // dependency, so establish S->V explicitly before publishing the tile.
    SyncScalar<HardEvent::S_V>();
    queue.EnQue(data);
    data = queue.DeQue<int32_t>();
    const DataCopyExtParams copy{1, count * width * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(dst[static_cast<uint64_t>(start) * width], data, copy);
    queue.FreeTensor(data);
}
}

extern "C" __global__ __aicore__ void mla_a3_plan(
    GM_ADDR kv, GM_ADDR cu, GM_ADDR used, GM_ADDR plan, uint32_t batches,
    int64_t max_q, int64_t max_kv, uint32_t has_cu, uint32_t has_used)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    GlobalTensor<int32_t> lengths, cumulative, effective, result;
    lengths.SetGlobalBuffer((__gm__ int32_t*)kv);
    if (has_cu) cumulative.SetGlobalBuffer((__gm__ int32_t*)cu);
    if (has_used) effective.SetGlobalBuffer((__gm__ int32_t*)used);
    result.SetGlobalBuffer((__gm__ int32_t*)plan);
    TPipe pipe;
    TQue<QuePosition::VECOUT, 1> queue;
    pipe.InitBuffer(queue, 1, kTile * 4 * sizeof(int32_t));
    for (uint32_t start = GetBlockIdx() * kTile; start < batches + 1; start += GetBlockNum() * kTile) {
        const uint32_t count = batches + 1 - start < kTile ? batches + 1 - start : kTile;
        auto data = queue.AllocTensor<int32_t>();
        SyncScalar<HardEvent::V_S>();
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t b = start + i;
            BatchPlan seq{kMagic, static_cast<int32_t>(has_cu), 0, 0};
            if (b < batches) {
                seq.start = has_cu ? cumulative.GetValue(b) : -1;
                seq.end = has_cu ? cumulative.GetValue(b + 1) : -1;
                const int64_t physical = has_cu ? static_cast<int64_t>(seq.end) - seq.start : max_q;
                seq.used = has_used ? effective.GetValue(b) : static_cast<int32_t>(physical);
                seq.kv = lengths.GetValue(b);
                if ((has_cu && (seq.start < 0 || physical < 0 || seq.used > physical)) ||
                    (max_q >= 0 && seq.used > max_q) || (max_kv >= 0 && seq.kv > max_kv)) {
                    seq.used = -1;
                }
            }
            data.SetValue(4 * i, seq.start);
            data.SetValue(4 * i + 1, seq.end);
            data.SetValue(4 * i + 2, seq.used);
            data.SetValue(4 * i + 3, seq.kv);
        }
        WriteTile<4>(result, start, count, queue, data);
    }
}

extern "C" __global__ __aicore__ void mla_a3_bounds(
    GM_ADDR plan, GM_ADDR bounds, uint32_t batches, uint32_t tokens, uint32_t packed,
    uint32_t mode, int64_t left, int64_t right, uint32_t capacity)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    GlobalTensor<int32_t> desc, result;
    desc.SetGlobalBuffer((__gm__ int32_t*)plan);
    result.SetGlobalBuffer((__gm__ int32_t*)bounds);
    if (batches == 0 || tokens == 0) return;
    const BatchPlan header = ReadBatch(desc, batches);
    const bool valid = header.start == kMagic && header.end == static_cast<int32_t>(packed) &&
        (packed ? desc.GetValue(0) == 0 && desc.GetValue(4 * (batches - 1) + 1) == static_cast<int32_t>(tokens)
                : tokens % batches == 0);
    TPipe pipe;
    TQue<QuePosition::VECOUT, 1> queue;
    pipe.InitBuffer(queue, 1, kTile * 3 * sizeof(int32_t));
    for (uint32_t start = GetBlockIdx() * kTile; start < tokens; start += GetBlockNum() * kTile) {
        const uint32_t count = tokens - start < kTile ? tokens - start : kTile;
        auto data = queue.AllocTensor<int32_t>();
        SyncScalar<HardEvent::V_S>();
        for (uint32_t i = 0; i < count; ++i) {
            int32_t b = 0, begin = 0, end = 0;
            if (valid) GetBounds(desc, batches, tokens, start + i, packed, mode, left, right, capacity, b, begin, end);
            data.SetValue(3 * i, b);
            data.SetValue(3 * i + 1, begin);
            data.SetValue(3 * i + 2, end);
        }
        WriteTile<3>(result, start, count, queue, data);
    }
}
