// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr int kMaxHot = 2048;
constexpr int kHashSize = 8192;
constexpr int kTailWidth = 2;
constexpr int kLayerFields = 16;

template <HardEvent event>
__aicore__ inline void SyncPipes()
{
    const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}


__aicore__ inline int Lookup(const LocalTensor<int32_t>& tokens, int token)
{
    uint32_t i = (static_cast<uint32_t>(token) * 2654435761U) & (kHashSize - 1);
    while (tokens.GetValue(i) >= 0 && tokens.GetValue(i) != token) {
        i = (i + 1) & (kHashSize - 1);
    }
    return i;
}
} // namespace

// The normal LIM entry couples replacement to score thresholds. Here the
// completed TopK gives the exact set that must be retained, so unselected
// resident slots and empty slots are sufficient eviction candidates.
extern "C" __global__ __aicore__ void selective_lim_group_topk(
    GM_ADDR topk, GM_ADDR layers, int layerCount, int rows, int width,
    int hostBlocks, int blockSize, int hot, int rowStride, int kBytes, int vBytes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (width > kMaxHot || hot > kMaxHot || width > hot) {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    TBuf<TPosition::VECCALC> cacheBuf;
    TBuf<TPosition::VECCALC> physicalBuf;
    pipe.InitBuffer(buf, (2 * kHashSize + 4 * kMaxHot) * sizeof(int32_t));
    pipe.InitBuffer(cacheBuf, (kMaxHot + 4) * sizeof(int64_t));
    pipe.InitBuffer(physicalBuf, kMaxHot * sizeof(int32_t));
    auto tokens = buf.Get<int32_t>();
    auto slots = tokens[kHashSize];
    auto resident = slots[kHashSize];
    auto used = resident[kMaxHot];
    auto free = used[kMaxHot];
    auto sources = free[kMaxHot];
    auto logical = reinterpret_cast<__gm__ int64_t*>(topk);
    auto descriptors = reinterpret_cast<__gm__ uint64_t*>(layers);
    GlobalTensor<int32_t> anchor;
    anchor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(topk));
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(anchor);
    // A layer/request owner retains the original per-row cache and duplicate-selection semantics.
    for (int owner = GetBlockIdx(); owner < layerCount * rows; owner += GetBlockNum()) {
        const int row = owner % rows;
        const auto layer = descriptors + (owner / rows) * kLayerFields;
        GM_ADDR table = reinterpret_cast<GM_ADDR>(layer[0]);
        GM_ADDR lengths = reinterpret_cast<GM_ADDR>(layer[1]);
        GM_ADDR requests = reinterpret_cast<GM_ADDR>(layer[2]);
        GM_ADDR tails = reinterpret_cast<GM_ADDR>(layer[3]);
        GM_ADDR keys = reinterpret_cast<GM_ADDR>(layer[4]);
        GM_ADDR previous = reinterpret_cast<GM_ADDR>(layer[5]);
        GM_ADDR indices = reinterpret_cast<GM_ADDR>(layer[6]);
        GM_ADDR src = reinterpret_cast<GM_ADDR>(layer[7]);
        GM_ADDR dst = reinterpret_cast<GM_ADDR>(layer[8]);
        GM_ADDR sizes = reinterpret_cast<GM_ADDR>(layer[9]);
        const uint64_t hostK = layer[10], hostV = layer[11];
        const uint64_t hotK = layer[12], hotV = layer[13];
        const int columns = static_cast<int>(layer[14]);
        auto cache = reinterpret_cast<__gm__ int64_t*>(keys);
        auto cacheLocal = cacheBuf.Get<int64_t>();
        auto selectedLocal = physicalBuf.Get<int32_t>();
        auto addresses = buf.Get<int64_t>();
        auto sizesLocal = cacheBuf.Get<int32_t>();
        GlobalTensor<int64_t> cacheGm, sourceGm, destinationGm, previousGm;
        GlobalTensor<int32_t> physicalGm, sizeGm;
        cacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(keys));
        sourceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(src));
        destinationGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(dst));
        previousGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(previous));
        physicalGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        sizeGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sizes));
        auto blocks = reinterpret_cast<__gm__ int32_t*>(table);
        auto seq = reinterpret_cast<__gm__ int32_t*>(lengths);
        auto req = reinterpret_cast<__gm__ int64_t*>(requests);
        auto tail = reinterpret_cast<__gm__ int32_t*>(tails);
        auto old = reinterpret_cast<__gm__ int64_t*>(previous);
        const int base = row * width;
        const int cacheBase = row * (hot + 1);
        const int length = seq[row];
        const int64_t request = req[row];
        const bool keep = request != 0 && old[row] == request;
        const int alignedWidth = (width + 7) / 8 * 8;
        Duplicate(tokens, -1, kHashSize);
        Duplicate(used, 0, hot);
        SyncPipes<HardEvent::V_S>();
        for (int i = 0; i < hot; ++i) {
            const int token = keep ? static_cast<int>(cache[cacheBase + i]) : -1;
            resident.SetValue(i, token);
            if (token >= 0) {
                const int h = Lookup(tokens, token);
                tokens.SetValue(h, token);
                slots.SetValue(h, i);
            }
        }
        bool pendingSource = false;
        for (int col = 0; col < width; ++col) {
            const int64_t id = logical[base + col];
            int source = -1;
            int selected = -1;
            if (request != 0 && id >= 0 && id < length && id / blockSize < columns) {
                const int block = id / blockSize;
                const int offset = id % blockSize;
                const int page = blocks[row * columns + block];
                if (page >= 0 && page < hostBlocks) {
                    if (tail[row * kTailWidth] >= 0 && block == (length - 1) / blockSize &&
                        offset >= tail[row * kTailWidth + 1]) {
                        selected = hot + offset;
                    } else if (id != length - 1) {
                        source = page * blockSize + offset;
                        const int h = Lookup(tokens, source);
                        if (tokens.GetValue(h) == source) {
                            selected = slots.GetValue(h);
                            used.SetValue(selected, 1);
                        }
                    }
                }
            }
            selectedLocal.SetValue(col, selected);
            sources.SetValue(col, source);
            pendingSource = pendingSource || (source >= 0 && selected < 0);
        }
        // Preserve the slow path's sentinel normalization before skipping cache writeback.
        const bool allHits = keep && !pendingSource && cache[cacheBase + hot] == -1;
        if (allHits) {
            // The lookup pass used slot-indexed flags; outputs need fresh column-indexed miss flags.
            SyncPipes<HardEvent::S_V>();
            Duplicate(used, 0, hot);
            SyncPipes<HardEvent::V_S>();
        } else {
            int available = 0;
            for (int i = 0; i < hot; ++i) {
                if (resident.GetValue(i) < 0) {
                    free.SetValue(available++, i);
                }
            }
            for (int i = 0; i < hot; ++i) {
                if (resident.GetValue(i) >= 0 && used.GetValue(i) == 0) {
                    free.SetValue(available++, i);
                }
            }
            int next = 0;
            for (int col = 0; col < width; ++col) {
                const int source = sources.GetValue(col);
                int selected = selectedLocal.GetValue(col);
                bool miss = false;
                if (source >= 0 && selected < 0) {
                    const int h = Lookup(tokens, source);
                    if (tokens.GetValue(h) == source) {
                        selected = slots.GetValue(h);
                    } else {
                        selected = free.GetValue(next++);
                        resident.SetValue(selected, source);
                        tokens.SetValue(h, source);
                        slots.SetValue(h, selected);
                        miss = true;
                    }
                    selectedLocal.SetValue(col, selected);
                }
                used.SetValue(col, miss ? 1 : 0);
            }
            for (int i = 0; i < hot; ++i) {
                cacheLocal.SetValue(i, resident.GetValue(i));
            }
            cacheLocal.SetValue(hot, -1);
        }
        SyncPipes<HardEvent::S_MTE3>();
        DataCopyPad(physicalGm[base], selectedLocal, {1, static_cast<uint32_t>(width * sizeof(int32_t)), 0, 0, 0});
        if (!allHits) {
            DataCopyPad(cacheGm[cacheBase], cacheLocal, {1, static_cast<uint32_t>((hot + 1) * sizeof(int64_t)), 0, 0, 0});
        }
        SyncPipes<HardEvent::MTE3_S>();

        // Lookup is finished, so reuse its hash storage for the four address vectors.
        for (int col = 0; col < width; ++col) {
            const int source = sources.GetValue(col);
            const bool miss = used.GetValue(col) != 0;
            const int64_t sourceRow = source < 0 ? 0 : source;
            const int64_t targetRow = row * rowStride + (miss ? selectedLocal.GetValue(col) : 0);
            addresses.SetValue(col, hostK + sourceRow * kBytes);
            addresses.SetValue(alignedWidth + col, hostV + sourceRow * vBytes);
            addresses.SetValue(2 * alignedWidth + col, hotK + targetRow * kBytes);
            addresses.SetValue(3 * alignedWidth + col, hotV + targetRow * vBytes);
            sizesLocal.SetValue(col, miss ? kBytes : 0);
            sizesLocal.SetValue(alignedWidth + col, miss ? vBytes : 0);
        }
        SyncPipes<HardEvent::S_MTE3>();
        const DataCopyExtParams addressCopy{1, static_cast<uint32_t>(width * sizeof(int64_t)), 0, 0, 0};
        const DataCopyExtParams sizeCopy{1, static_cast<uint32_t>(width * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(sourceGm[base], addresses, addressCopy);
        DataCopyPad(sourceGm[rows * width + base], addresses[alignedWidth], addressCopy);
        DataCopyPad(destinationGm[base], addresses[2 * alignedWidth], addressCopy);
        DataCopyPad(destinationGm[rows * width + base], addresses[3 * alignedWidth], addressCopy);
        DataCopyPad(sizeGm[base], sizesLocal, sizeCopy);
        DataCopyPad(sizeGm[rows * width + base], sizesLocal[alignedWidth], sizeCopy);
        SyncPipes<HardEvent::MTE3_S>();
        if (!allHits) {
            addresses.SetValue(0, request);
            SyncPipes<HardEvent::S_MTE3>();
            DataCopyPad(previousGm[row], addresses, {1, sizeof(int64_t), 0, 0, 0});
            SyncPipes<HardEvent::MTE3_S>();
        }
    }
}

namespace vllm_ascend {
void selective_lim_group_impl(void* stream, void* topk, void* layers,
    int layerCount, int rows, int width, int hostBlocks, int blockSize,
    int hot, int rowStride, int kBytes, int vBytes)
{
    selective_lim_group_topk<<<layerCount * rows, nullptr, stream>>>(
        static_cast<GM_ADDR>(topk), static_cast<GM_ADDR>(layers), layerCount,
        rows, width, hostBlocks, blockSize, hot, rowStride, kBytes, vBytes);
}
} // namespace vllm_ascend
