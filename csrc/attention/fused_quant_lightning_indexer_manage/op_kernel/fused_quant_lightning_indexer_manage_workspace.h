#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_WORKSPACE_H
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_WORKSPACE_H

#include <cstdint>

// 本头同时被 host 侧 (tiling.cpp) 包含以共享 S2_BASE_SIZE 等布局常量——
// s2 chunk 粒度必须 kernel/host 单一来源, 双份定义曾靠注释同步(易漂移)。
// host 编译单元无 __aicore__ 宏, 置空使下方函数可解析; 设备侧已由
// kernel_operator.h 先行定义, 不受影响。
#ifndef __aicore__
#define __aicore__
#endif

namespace MtpC8Workspace {
// s2 chunk 粒度。2026-09-07 起对齐 lightning_indexer_quant(2048): 打分段
// chunk 数 ÷4, 摊薄逐 chunk 的跨核握手/ProcessVec prolog/kScale gather 固定
// 开销(aiv_scalar 2.50 vs F 0.65us/chunk 的主项)。512 是 bf16 血统 per-head
// mm1Res ×H 的 workspace 遗产, C8 头归约进 cube 后约束已不存在。
// 关联: kernel.h S2_BASE_SIZE / tiling.cpp workspace 公式均引此处。
constexpr uint64_t S2_BASE_SIZE = 2048U;
constexpr uint64_t S1_BASE_SIZE = 8U;
constexpr uint64_t DOUBLE_BUFFER = 2U;
constexpr uint64_t LD_HEAD_TAIL = 2U;
constexpr uint64_t VALUE_AND_INDEX = 2U;
constexpr uint64_t TOPK = 2048U;
constexpr uint64_t LD_PARAM_NUM = 16U;
constexpr uint64_t PAIR_CAPACITY = 8192U;
constexpr uint64_t THRESHOLD_STRIDE = 8U;
constexpr uint64_t ROUTE_COUNT_STRIDE = 8U;

__aicore__ inline uint64_t LiWorkspaceBytes(uint64_t blockNum, uint64_t headNum)
{
    const uint64_t mBaseSize = S1_BASE_SIZE * headNum;
    // C8: mm1Res 是每核 [S1_BASE_SIZE, S2_BASE_SIZE] 的 fp32 最终分数行 (两段
    // mma, 双缓冲), 后跟 vec0 折叠的 w*q_scale fp16 workspace (mBaseSize * 16
    // elements, 每核单缓冲)。与 kernel.h Init / tiling DoTiling 保持同序。
    constexpr uint64_t BLOCK_CUBE = 16U;
    uint64_t bytes = blockNum * DOUBLE_BUFFER * S1_BASE_SIZE * S2_BASE_SIZE * sizeof(float);
    bytes += blockNum * mBaseSize * BLOCK_CUBE * sizeof(uint16_t);
    bytes += blockNum * S1_BASE_SIZE * LD_HEAD_TAIL * VALUE_AND_INDEX * TOPK * sizeof(float);
    bytes += blockNum * S1_BASE_SIZE * LD_HEAD_TAIL * LD_PARAM_NUM * sizeof(int64_t);
    return bytes;
}

__aicore__ inline uint64_t Pair0Offset(uint64_t blockNum, uint64_t headNum)
{
    return LiWorkspaceBytes(blockNum, headNum);
}

__aicore__ inline uint64_t Pair1Offset(uint64_t blockNum, uint64_t headNum, uint64_t batch)
{
    return Pair0Offset(blockNum, headNum) + batch * PAIR_CAPACITY * sizeof(float);
}

__aicore__ inline uint64_t ScoreStride(uint64_t sourceCapacity)
{
    return ((sourceCapacity + S2_BASE_SIZE - 1U) / S2_BASE_SIZE) * S2_BASE_SIZE;
}

__aicore__ inline uint64_t ScoreOffset(uint64_t blockNum, uint64_t headNum, uint64_t batch)
{
    return Pair1Offset(blockNum, headNum, batch) + batch * PAIR_CAPACITY * sizeof(float);
}

__aicore__ inline uint64_t ThresholdOffset(uint64_t blockNum, uint64_t headNum,
                                            uint64_t batch, uint64_t totalQueries,
                                            uint64_t sourceCapacity)
{
    return ScoreOffset(blockNum, headNum, batch) +
           totalQueries * ScoreStride(sourceCapacity) * sizeof(float);
}

__aicore__ inline uint64_t RouteCountOffset(uint64_t blockNum, uint64_t headNum,
                                             uint64_t batch, uint64_t totalQueries,
                                             uint64_t sourceCapacity)
{
    return ThresholdOffset(blockNum, headNum, batch, totalQueries,
                           sourceCapacity) +
           totalQueries * THRESHOLD_STRIDE * sizeof(float);
}

} // namespace MtpC8Workspace

#endif
