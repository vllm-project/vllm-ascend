// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr int kMaxHot = 2048;
constexpr int kHashSize = 8192;
constexpr int kTailWidth = 2;

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
extern "C" __global__ __aicore__ void selective_lim_topk(
    GM_ADDR topk, GM_ADDR table, GM_ADDR lengths, GM_ADDR requests,
    GM_ADDR tails, GM_ADDR keys, GM_ADDR previous, GM_ADDR indices,
    GM_ADDR src, GM_ADDR dst, GM_ADDR sizes,
    int rows, int width, int columns, int hostBlocks, int blockSize, int hot,
    int rowStride, int kBytes, int vBytes, uint64_t hostK, uint64_t hostV,
    uint64_t hotK, uint64_t hotV)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (width > kMaxHot || hot > kMaxHot || width > hot) {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, (2 * kHashSize + 4 * kMaxHot) * sizeof(int32_t));
    auto tokens = buf.Get<int32_t>();
    auto slots = tokens[kHashSize];
    auto resident = slots[kHashSize];
    auto used = resident[kMaxHot];
    auto free = used[kMaxHot];
    auto sources = free[kMaxHot];
    auto logical = reinterpret_cast<__gm__ int64_t*>(topk);
    auto blocks = reinterpret_cast<__gm__ int32_t*>(table);
    auto seq = reinterpret_cast<__gm__ int32_t*>(lengths);
    auto req = reinterpret_cast<__gm__ int64_t*>(requests);
    auto tail = reinterpret_cast<__gm__ int32_t*>(tails);
    auto cache = reinterpret_cast<__gm__ int64_t*>(keys);
    auto old = reinterpret_cast<__gm__ int64_t*>(previous);
    auto physical = reinterpret_cast<__gm__ int32_t*>(indices);
    auto from = reinterpret_cast<__gm__ int64_t*>(src);
    auto to = reinterpret_cast<__gm__ int64_t*>(dst);
    auto count = reinterpret_cast<__gm__ int32_t*>(sizes);
    GlobalTensor<int32_t> anchor;
    anchor.SetGlobalBuffer(physical);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(anchor);
    for (int row = 0; row < rows; ++row) {
        const int base = row * width;
        const int cacheBase = row * (hot + 1);
        const int length = seq[row];
        const int64_t request = req[row];
        for (int i = 0; i < kHashSize; ++i) {
            tokens.SetValue(i, -1);
        }
        for (int i = 0; i < hot; ++i) {
            const int token = request != 0 && old[row] == request ? cache[cacheBase + i] : -1;
            resident.SetValue(i, token);
            used.SetValue(i, 0);
            if (token >= 0) {
                const int h = Lookup(tokens, token);
                tokens.SetValue(h, token);
                slots.SetValue(h, i);
            }
        }
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
            physical[base + col] = selected;
            sources.SetValue(col, source);
        }
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
            const int pos = base + col;
            const int source = sources.GetValue(col);
            int selected = physical[pos];
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
                physical[pos] = selected;
            }
            const int64_t sourceRow = source < 0 ? 0 : source;
            const int64_t targetRow = row * rowStride + (miss ? selected : 0);
            const int vPos = rows * width + pos;
            from[pos] = hostK + sourceRow * kBytes;
            from[vPos] = hostV + sourceRow * vBytes;
            to[pos] = hotK + targetRow * kBytes;
            to[vPos] = hotV + targetRow * vBytes;
            count[pos] = miss ? kBytes : 0;
            count[vPos] = miss ? vBytes : 0;
        }
        for (int i = 0; i < hot; ++i) {
            cache[cacheBase + i] = resident.GetValue(i);
        }
        cache[cacheBase + hot] = -1;
        old[row] = request;
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(anchor);
}
