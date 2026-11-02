#ifndef SPLITQUANT_DENSE_FUSED__A2_ADAPTER_H
#define SPLITQUANT_DENSE_FUSED__A2_ADAPTER_H

// CANN 8.5's public Cube headers do not provide an A2 packed-INT4
// BlockMmad. This adapter supplies that missing block boundary over the
// supported AscendC MatmulImpl INT4 primitive. Higher layers keep Cube' tile
// scheduler and per-group epilogue contracts, while execution remains native
// A2 Cube code rather than a floating-point dequantization path.
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling.h"

namespace SplitQuantDenseA2 {
using namespace AscendC;
using matmul::MatmulType;

struct GroupedProblem {
    uint32_t m;
    uint32_t n;
    uint32_t groups;
    uint32_t k_per_group;
};

struct Task {
    uint32_t group;
    uint32_t m_start;
    uint32_t n_start;
    uint32_t valid_m;
    uint32_t valid_n;
};

struct BlockScheduler {
    static constexpr uint32_t kTileM = 16;
    // A2 has a wide K dimension here (40 independent 128-wide groups).  A
    // 1024-column tile reduces scheduler and MatmulImpl iteration overhead for
    // decode-sized M=1 while remaining within the generated Cube tiling.
    static constexpr uint32_t kTileN = 1024;

    __aicore__ static inline uint32_t TaskCount(const GroupedProblem& p)
    {
        return p.groups * ((p.m + kTileM - 1) / kTileM) * ((p.n + kTileN - 1) / kTileN);
    }

    __aicore__ static inline Task GetTask(const GroupedProblem& p, uint32_t task_id)
    {
        const uint32_t n_tiles = (p.n + kTileN - 1) / kTileN;
        const uint32_t m_tiles = (p.m + kTileM - 1) / kTileM;
        const uint32_t per_group = m_tiles * n_tiles;
        const uint32_t group = task_id / per_group;
        const uint32_t tile = task_id % per_group;
        const uint32_t m_start = (tile / n_tiles) * kTileM;
        const uint32_t n_start = (tile % n_tiles) * kTileN;
        return {group, m_start, n_start,
                (p.m - m_start < kTileM) ? p.m - m_start : kTileM,
                (p.n - n_start < kTileN) ? p.n - n_start : kTileN};
    }
};

class BlockMmad {
public:
    static constexpr auto kConfig = GetMDLConfig(false, false, 0, false, false, false, true);
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, true>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;
    using Matmul = MatmulImpl<AType, BType, CType, CType, kConfig>;

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
    {
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(a));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(b));
        c_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(c));
        const auto* src = reinterpret_cast<__gm__ int32_t*>(tiling);
        auto* dst = reinterpret_cast<int32_t*>(&tiling_);
        for (uint32_t i = 0; i < (sizeof(TCubeTiling) + 3) / 4; ++i) {
            dst[i] = src[i];
        }
        mm_.SetSubBlockIdx(0);
        mm_.Init(&tiling_, &pipe_);
    }

    __aicore__ inline void Compute(const GroupedProblem& p, const Task& task)
    {
        const uint32_t total_k = p.groups * p.k_per_group;
        // A is one independent 128-wide group per tile; B keeps its original
        // [N, K_total] row stride while the group pointer selects its K slice.
        mm_.SetOrgShape(p.m, p.n, p.k_per_group, p.k_per_group, p.n);
        mm_.SetSingleShape(task.valid_m, task.valid_n, p.k_per_group);
        // GlobalTensor indices for int4b_t count logical nibbles.
        mm_.SetTensorA(a_[(task.group * p.m + task.m_start) * p.k_per_group], false);
        mm_.SetTensorB(b_[(task.group * p.n + task.n_start) * p.k_per_group], true);
        mm_.IterateAll(c_[(task.group * p.m + task.m_start) * p.n + task.n_start], 0);
    }

    __aicore__ inline void End() { mm_.End(); }

private:
    TPipe pipe_;
    TCubeTiling tiling_{};
    GlobalTensor<int4b_t> a_, b_;
    GlobalTensor<int32_t> c_;
    Matmul mm_;
};

// The Cube stage boundary is per-group INT32 partials. SplitQuant's exact
// offset correction and dtype-rounded scaling are intentionally performed by
// the existing following AIV epilogue, which consumes this workspace ABI.
struct BlockEpilogue {
    using ElementAccumulator = int32_t;
    static constexpr bool kProducesPerGroupPartial = true;
};

class GroupedKernel {
public:
    __aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling,
                               uint32_t m, uint32_t n, uint32_t groups)
    {
        if ASCEND_IS_AIV { return; }
        GroupedProblem problem{m, n, groups, 128};
        BlockMmad mmad;
        mmad.Init(a, b, c, tiling);
        for (uint32_t task_id = GetBlockIdx(); task_id < BlockScheduler::TaskCount(problem);
             task_id += GetBlockNum()) {
            const Task task = BlockScheduler::GetTask(problem, task_id);
            mmad.Compute(problem, task);
            PipeBarrier<PIPE_ALL>();
            SetAtomicNone();
        }
        mmad.End();
    }
};
} // namespace SplitQuantDenseA2

#endif
