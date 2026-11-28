// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "grouped_softmax.h"
using namespace AscendC;
namespace {
using BFloat16 = bfloat16_t;
template <HardEvent event> __aicore__ inline void fence() {
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}
__aicore__ inline void loadNdToNz(LocalTensor<BFloat16> dst, GlobalTensor<BFloat16> src, uint16_t m,
                                  uint16_t n) {
    Nd2NzParams p{};
    p.ndNum = 1;
    p.nValue = m;
    p.dValue = n;
    p.srcDValue = n;
    p.dstNzC0Stride = (m + 15) / 16 * 16;
    p.dstNzNStride = 1;
    DataCopy(dst, src, p);
}
__aicore__ inline void loadFractals(LocalTensor<BFloat16> dst, LocalTensor<BFloat16> src, uint8_t repeat) {
    LoadData2DParams p{};
    p.repeatTimes = repeat;
    p.srcStride = 1;
    LoadData(dst, src, p);
}
__aicore__ inline void multiply(LocalTensor<float> c, LocalTensor<BFloat16> a, LocalTensor<BFloat16> b,
                                uint16_t n, uint16_t k, bool init) {
    MmadParams p{};
    p.m = 16;
    p.n = n;
    p.k = k;
    p.cmatrixInitVal = init;
    Mmad(c, a, b, p);
}
__aicore__ inline void storeRowMajor(GlobalTensor<float> out, LocalTensor<float> c, uint16_t n,
                                     uint32_t pitch) {
    FixpipeParamsV220 p{};
    p.mSize = 16;
    p.nSize = n;
    p.srcStride = 16;
    p.dstStride = pitch;
    p.ndNum = 1;
    Fixpipe(out, c, p);
}

// The page is [128,576] in GM. Both QK and PV reuse its NZ representation
// in L1. All 16 query rows remain in L0A until the batch changes.
class GroupCube {
    GlobalTensor<BFloat16> queryGm_, cacheGm_, weightsGm_;
    GlobalTensor<float> scoreGm_, valueGm_;
    TBuf<TPosition::A1> queryL1_, cacheL1_, weightsL1_;
    TBuf<TPosition::A2> operandA_;
    TBuf<TPosition::B2> operandB_;
    TBuf<TPosition::CO1> accumulator_;

  public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR q, GM_ADDR cache, GM_ADDR weights, GM_ADDR score,
                                GM_ADDR val, uint32_t group) {
        queryGm_.SetGlobalBuffer((__gm__ BFloat16 *)q);
        cacheGm_.SetGlobalBuffer((__gm__ BFloat16 *)cache);
        weightsGm_.SetGlobalBuffer((__gm__ BFloat16 *)weights);
        scoreGm_.SetGlobalBuffer((__gm__ float *)score);
        valueGm_.SetGlobalBuffer((__gm__ float *)val);
        pipe.InitBuffer(queryL1_, 16 * 576 * 2);
        pipe.InitBuffer(cacheL1_, group * 128 * 576 * 2);
        pipe.InitBuffer(weightsL1_, group * 16 * 128 * 2);
        pipe.InitBuffer(operandA_, (16 * 576 + group * 16 * 128) * 2);
        pipe.InitBuffer(operandB_, 2 * 128 * 128 * 2);
        pipe.InitBuffer(accumulator_, (16 * 512 + group * 16 * 128) * 4);
        const uint8_t pads[4] = {0, 0, 0, 0};
        SetFmatrix(1, 128, pads, FmatrixMode::FMATRIX_RIGHT);
    }
    __aicore__ inline void Query(uint32_t firstToken, uint32_t queryLength) {
        auto q1 = queryL1_.Get<BFloat16>();
        if (queryLength < 4) {
            InitConstValueParams<half> z{1, 576, 0, static_cast<half>(0)};
            InitConstValue(q1.ReinterpretCast<half>(), z);
            PipeBarrier<PIPE_MTE2>();
        }
        loadNdToNz(q1, queryGm_[static_cast<uint64_t>(firstToken) * 4 * 576], queryLength * 4, 576);
        fence<HardEvent::MTE2_MTE1>();
        loadFractals(operandA_.Get<BFloat16>(), q1, 36);
        fence<HardEvent::MTE1_M>();
    }
    __aicore__ inline void Run(GlobalTensor<int32_t> &table, uint64_t slot, uint32_t first, uint32_t count,
                               uint64_t groupSlot, uint32_t numPhysicalPages, uint32_t kvLength,
                               uint64_t tableSlot) {
        auto a0 = operandA_.Get<BFloat16>();
        auto b0 = operandB_.Get<BFloat16>();
        auto c0 = accumulator_.Get<float>();
        for (uint32_t p = 0; p < count; p++) {
            auto key = cacheL1_.Get<BFloat16>()[p * 128 * 576];
            const int32_t page = table.GetValue(tableSlot + p);
            const bool valid = page >= 0 && static_cast<uint32_t>(page) < numPhysicalPages;
            int64_t remaining = static_cast<int64_t>(kvLength) - (first + p) * 128;
            const uint32_t live = valid && remaining > 0 ? (remaining > 128 ? 128 : remaining) : 0;
            if (live) {
                loadNdToNz(key, cacheGm_[static_cast<uint64_t>(page) * 128 * 576], 128, 576);
            }
            if (live < 128) {
                PipeBarrier<PIPE_MTE2>();
                InitConstValueParams<half> z{36, static_cast<uint16_t>(128 - live),
                                             static_cast<uint16_t>(live), static_cast<half>(0)};
                InitConstValue(key.ReinterpretCast<half>()[live * 16], z);
            }
        }
        fence<HardEvent::MTE2_MTE1>();
        // QK accumulates six 96-channel tiles for each of 16 x 128 logits.
        for (uint32_t p = 0; p < count; p++) {
            auto key = cacheL1_.Get<BFloat16>()[p * 128 * 576];
            auto result = c0[16 * 512 + p * 16 * 128];
            for (uint32_t seg = 0; seg < 6; seg++) {
                auto tile = b0[(seg % 2) * 96 * 128];
                loadFractals(tile, key[seg * 96 * 128], 48);
                fence<HardEvent::MTE1_M>();
                multiply(result, a0[seg * 16 * 96], tile, 128, 96, seg == 0);
                PipeBarrier<PIPE_M>();
                fence<HardEvent::M_MTE1>();
            }
            fence<HardEvent::M_FIX>();
            storeRowMajor(scoreGm_[(slot + p) * 16 * 128], result, 128, 128);
        }
        // The paired AIV computes group-relative probabilities from these logits.
        CrossCoreSetFlag<2, PIPE_FIX>(1);
        CrossCoreWaitFlag<2>(2);
        for (uint32_t p = 0; p < count; p++)
            loadNdToNz(weightsL1_.Get<BFloat16>()[p * 16 * 128], weightsGm_[(slot + p) * 16 * 128], 16, 128);
        fence<HardEvent::MTE2_MTE1>();
        for (uint32_t p = 0; p < count; p++)
            loadFractals(a0[16 * 576 + p * 16 * 128], weightsL1_.Get<BFloat16>()[p * 16 * 128], 8);
        // Each 128-channel output column tile owns a distinct region in L0C.
        // Accumulate all pages in FP32 before a single 16 x 512 Fixpipe store.
        for (uint32_t p = 0; p < count; p++) {
            auto key = cacheL1_.Get<BFloat16>()[p * 128 * 576];
            for (uint32_t n = 0; n < 512; n += 128) {
                LoadData3DParamsV2Pro x{};
                x.channelSize = 128;
                x.extConfig = (static_cast<uint64_t>(128) << 16) | 128;
                x.fMatrixCtrl = true;
                auto tile = b0[((n / 128) % 2) * 128 * 128];
                LoadData<BFloat16>(tile, key[n * 128], x);
                fence<HardEvent::MTE1_M>();
                multiply(c0[n * 16], a0[16 * 576 + p * 16 * 128], tile, 128, 128, p == 0);
                PipeBarrier<PIPE_M>();
                fence<HardEvent::M_MTE1>();
            }
        }
        fence<HardEvent::M_FIX>();
        storeRowMajor(valueGm_[groupSlot * 16 * 512], c0, 512, 512);
        fence<HardEvent::FIX_M>();
        fence<HardEvent::MTE1_MTE2>();
    }
};
} // namespace
extern "C" __global__ __aicore__ void flash_mla_a3_grouped_kernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR table, GM_ADDR len, GM_ADDR scores, GM_ADDR weights,
                            GM_ADDR partial, GM_ADDR stats, uint32_t batches, uint32_t width, uint32_t pages,
                            uint32_t queryLength, float scale, uint32_t causal, uint32_t group,
                            uint32_t cores, GM_ADDR plan, GM_ADDR cumulative, GM_ADDR used, uint32_t flags,
                            GM_ADDR map) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TPipe pipe;
    GroupCube cube;
    GroupSoftmax soft;
    if ASCEND_IS_AIC {
        cube.Init(pipe, q, k, weights, scores, partial, group);
    }
    if ASCEND_IS_AIV {
        soft.Init(pipe, scores, weights, stats, group);
    }
    GlobalTensor<int32_t> lens, ids;
    lens.SetGlobalBuffer((__gm__ int32_t *)len);
    ids.SetGlobalBuffer((__gm__ int32_t *)table);
    const uint32_t groups = (width + group - 1) / group;
    uint32_t core = GetBlockIdx();
    if ASCEND_IS_AIV {
        core = GetBlockIdx() / GetSubBlockNum();
    }
    // Map physical tiles directly to the request(s) owning their tokens. Only a
    // tile that crosses a sequence boundary needs more than one QK/PV segment.
    // The usual first segment uses tile as its output slot. Extra segments
    // reserve disjoint slots after batches, including across simultaneous cores.
    if (flags & 16u) {
        GlobalTensor<int32_t> desc, cu, effective, outputMap;
        if (flags & 1u)
            desc.SetGlobalBuffer((__gm__ int32_t *)plan);
        if ((flags & 2u) && !(flags & 1u))
            cu.SetGlobalBuffer((__gm__ int32_t *)cumulative);
        if (flags & 4u)
            effective.SetGlobalBuffer((__gm__ int32_t *)used);
        outputMap.SetGlobalBuffer((__gm__ int32_t *)map);
        if ASCEND_IS_AIV {
            if (queryLength > 1)
                soft.InitMap(pipe);
        }
        const int32_t tokens = batches * queryLength;
        const bool hasPlan = flags & 1u, packed = flags & 2u;
        bool valid = true;
        if (hasPlan) {
            valid = desc.GetValue(4 * batches) == 0x4D4C4133 &&
                    desc.GetValue(4 * batches + 1) == static_cast<int32_t>(packed);
        }
        if (packed) {
            int32_t begin = hasPlan ? desc.GetValue(0) : cu.GetValue(0);
            int32_t end = hasPlan ? desc.GetValue(4 * batches - 3) : cu.GetValue(batches);
            valid = valid && begin == 0 && end == tokens;
        }
        for (uint64_t job = core; job < static_cast<uint64_t>(batches) * groups; job += cores) {
            uint32_t tile = job / groups, g = job % groups, first = g * group;
            uint32_t count = width - first < group ? width - first : group;
            const int32_t tileStart = tile * queryLength, tileEnd = tileStart + queryLength;
            const uint64_t scratch = static_cast<uint64_t>(tile) * width + first;
            int32_t mapRows[4] = {};
            int32_t token = tileStart, b = tile;
            while (token < tileEnd) {
                int32_t lo = tileStart, hi = tileEnd;
                if (packed) {
                    lo = hasPlan ? desc.GetValue(4 * b) : cu.GetValue(b);
                    hi = hasPlan ? desc.GetValue(4 * b + 1) : cu.GetValue(b + 1);
                    if (token < lo || token >= hi) {
                        int32_t left = 0, right = batches;
                        while (left < right) {
                            const int32_t mid = left + (right - left) / 2;
                            int32_t end = hasPlan ? desc.GetValue(4 * mid + 1) : cu.GetValue(mid + 1);
                            if (end <= token)
                                left = mid + 1;
                            else
                                right = mid;
                        }
                        b = left < static_cast<int32_t>(batches) ? left : batches - 1;
                        lo = hasPlan ? desc.GetValue(4 * b) : cu.GetValue(b);
                        hi = hasPlan ? desc.GetValue(4 * b + 1) : cu.GetValue(b + 1);
                    }
                }
                const bool physical = valid && lo >= 0 && hi <= tokens && lo <= token && token < hi;
                const uint32_t size = physical ? ((hi < tileEnd ? hi : tileEnd) - token) : 1;
                int32_t liveQ =
                    hasPlan ? desc.GetValue(4 * b + 2) : (flags & 4u ? effective.GetValue(b) : hi - lo);
                int32_t liveKv = hasPlan ? desc.GetValue(4 * b + 3) : lens.GetValue(b);
                const uint32_t remaining = physical && liveQ >= 0 && liveQ <= hi - lo && liveQ > token - lo
                                               ? liveQ - (token - lo)
                                               : 0;
                const uint32_t kv =
                    remaining && liveKv > 0 && liveKv <= static_cast<int64_t>(width) * 128 ? liveKv : 0;
                uint32_t offset = token - tileStart;
                uint32_t outSlot = offset == 0 ? tile : batches + tile * (queryLength - 1) + offset - 1;
                uint64_t outJob = static_cast<uint64_t>(outSlot) * groups + g;
                uint64_t tableSlot = static_cast<uint64_t>(b) * width + first;
                if ASCEND_IS_AIC {
                    cube.Query(token, size);
                    cube.Run(ids, scratch, first, count, outJob, pages, kv, tableSlot);
                }
                if ASCEND_IS_AIV {
                    uint32_t bad = 0xffffffffu;
                    for (uint32_t j = 0; j < count; j++) {
                        int32_t page = ids.GetValue(tableSlot + j);
                        if (page < 0 || static_cast<uint32_t>(page) >= pages) {
                            bad = (first + j) * 128;
                            break;
                        }
                    }
                    CrossCoreWaitFlag<2>(1);
                    soft.Run(scratch, first, count, outJob, kv, size, remaining, scale, causal, bad);
                    CrossCoreSetFlag<2, PIPE_MTE3>(2);
                    if (g == 0 && queryLength > 1) {
                        for (uint32_t r = 0; r < size; r++)
                            mapRows[offset + r] = outSlot * 16 + r * 4;
                    }
                }
                token += size;
            }
            if ASCEND_IS_AIV {
                if (g == 0 && queryLength > 1)
                    soft.WriteMap(outputMap, tile, mapRows);
            }
        }
        return;
    }
    // A single packed or dense sequence always has queryLength physical rows.
    // Its effective prefix may change on every graph replay.
    uint32_t activeQuery = queryLength;
    int32_t dynamicKv = 0;
    if (flags & 8u) {
        const bool packed = flags & 2u;
        bool valid = true;
        int32_t effective = queryLength;
        if (flags & 1u) {
            GlobalTensor<int32_t> descriptor;
            descriptor.SetGlobalBuffer((__gm__ int32_t *)plan);
            effective = descriptor.GetValue(2);
            dynamicKv = descriptor.GetValue(3);
            valid = descriptor.GetValue(4) == 0x4D4C4133 &&
                    descriptor.GetValue(5) == static_cast<int32_t>(packed);
            if (packed)
                valid = valid && descriptor.GetValue(0) == 0 &&
                        descriptor.GetValue(1) == static_cast<int32_t>(queryLength);
        } else {
            dynamicKv = lens.GetValue(0);
            if (packed) {
                GlobalTensor<int32_t> offsets;
                offsets.SetGlobalBuffer((__gm__ int32_t *)cumulative);
                valid = offsets.GetValue(0) == 0 && offsets.GetValue(1) == static_cast<int32_t>(queryLength);
            }
            if (flags & 4u) {
                GlobalTensor<int32_t> live;
                live.SetGlobalBuffer((__gm__ int32_t *)used);
                effective = live.GetValue(0);
            }
        }
        activeQuery =
            valid && effective > 0 && effective <= static_cast<int32_t>(queryLength) ? effective : 0;
        if (!activeQuery)
            dynamicKv = 0;
    }
    uint32_t previousBatch = ~0U;
    for (uint64_t job = core; job < static_cast<uint64_t>(batches) * groups; job += cores) {
        uint32_t b = job / groups, first = (job % groups) * group;
        uint32_t count = width - first < group ? width - first : group;
        uint64_t slot = static_cast<uint64_t>(b) * width + first;
        int32_t n = (flags & 8u) ? dynamicKv : lens.GetValue(b);
        uint32_t kvLength = n > 0 && n <= static_cast<int64_t>(width) * 128 ? n : 0;
        if ASCEND_IS_AIC {
            if (previousBatch != b) {
                cube.Query(b * queryLength, queryLength);
                previousBatch = b;
            }
            cube.Run(ids, slot, first, count, job, pages, kvLength, slot);
        }
        if ASCEND_IS_AIV {
            uint32_t bad = 0xffffffffu;
            for (uint32_t p = 0; p < count; p++) {
                int32_t page = ids.GetValue(slot + p);
                if (page < 0 || static_cast<uint32_t>(page) >= pages) {
                    bad = (first + p) * 128;
                    break;
                }
            }
            CrossCoreWaitFlag<2>(1);
            soft.Run(slot, first, count, job, kvLength, queryLength, activeQuery, scale, causal, bad);
            CrossCoreSetFlag<2, PIPE_MTE3>(2);
        }
    }
}
