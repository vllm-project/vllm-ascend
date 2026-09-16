#ifndef FUSED_LI_MANAGE_MTP_WORKSPACE_H
#define FUSED_LI_MANAGE_MTP_WORKSPACE_H

#include <cstdint>

namespace MtpWorkspace {
constexpr uint64_t S2_BASE_SIZE = 512U;
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
    uint64_t bytes = blockNum * DOUBLE_BUFFER * mBaseSize * S2_BASE_SIZE * sizeof(float);
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

} // namespace MtpWorkspace

#endif
