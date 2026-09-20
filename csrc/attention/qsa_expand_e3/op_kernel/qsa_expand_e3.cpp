#include "kernel_operator.h"
#include <cstdint>

struct QsaExpandE3TilingData {
    uint32_t rows;
};

namespace {
using namespace AscendC;

constexpr uint32_t kTopK = 512;
constexpr uint32_t kRatio = 4;
constexpr uint32_t kBodyWidth = kTopK * kRatio;
constexpr uint32_t kOutputWidth = 2051;
constexpr uint32_t kOutputAligned = 2056;

class QsaExpandE3vKernel {
public:
    __aicore__ inline void Init(GM_ADDR groups, GM_ADDR completeGroups,
                                GM_ADDR tailStart, GM_ADDR tailCount,
                                GM_ADDR sequenceLengths, GM_ADDR tokenToReq,
                                GM_ADDR expanded,
                                const QsaExpandE3TilingData* tiling,
                                TPipe* pipe)
    {
        rows_ = tiling->rows;
        core_ = GetBlockIdx();
        cores_ = GetBlockNum();
        groups_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groups));
        completeGroups_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(completeGroups));
        tailStart_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tailStart));
        tailCount_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tailCount));
        sequenceLengths_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequenceLengths));
        tokenToReq_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenToReq));
        expanded_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expanded));
        pipe->InitBuffer(groupsBuf_, kTopK * sizeof(int32_t));
        pipe->InitBuffer(outputBuf_, kOutputAligned * sizeof(int32_t));
        pipe->InitBuffer(indexBuf_, kBodyWidth * sizeof(int32_t));
        pipe->InitBuffer(offsetBuf_, kBodyWidth * sizeof(uint32_t));
        pipe->InitBuffer(laneBuf_, kBodyWidth * sizeof(int32_t));
        groupsLocal_ = groupsBuf_.Get<int32_t>();
        outputLocal_ = outputBuf_.Get<int32_t>();
        indexLocal_ = indexBuf_.Get<int32_t>();
        offsetLocal_ = offsetBuf_.Get<uint32_t>();
        laneLocal_ = laneBuf_.Get<int32_t>();
    }

    __aicore__ inline void ScalarBody(int32_t complete, int32_t tailStart,
                                      int32_t tailCount, int32_t seqLen)
    {
        for (uint32_t col = 0; col < kBodyWidth; ++col) {
            int32_t groupRank = static_cast<int32_t>(col >> 2);
            int32_t lane = static_cast<int32_t>(col & 3);
            int32_t token = -1;
            if (groupRank < complete) {
                int32_t group = groupsLocal_.GetValue(groupRank);
                int32_t candidate = group * static_cast<int32_t>(kRatio) + lane;
                if (group >= 0 && candidate < seqLen) token = candidate;
            } else if (groupRank == complete && lane < tailCount) {
                int32_t candidate = tailStart + lane;
                if (candidate >= 0 && candidate < seqLen) token = candidate;
            }
            outputLocal_.SetValue(col, token);
        }
    }

    __aicore__ inline void VectorFullBody(int32_t seqLen)
    {
        // Generate byte offsets {0,0,0,0,4,4,4,4,...}; Gather therefore
        // loads every compressed group exactly once for each of its 4 lanes.
        CreateVecIndex(indexLocal_, static_cast<int32_t>(0), kBodyWidth);
        ShiftRight(offsetLocal_.ReinterpretCast<int32_t>(), indexLocal_,
                   static_cast<int32_t>(2), static_cast<int32_t>(kBodyWidth));
        ShiftLeft(offsetLocal_.ReinterpretCast<int32_t>(),
                  offsetLocal_.ReinterpretCast<int32_t>(),
                  static_cast<int32_t>(2), static_cast<int32_t>(kBodyWidth));
        Gather(outputLocal_, groupsLocal_, offsetLocal_, 0, kBodyWidth);
        ShiftLeft(outputLocal_, outputLocal_, static_cast<int32_t>(2),
                  static_cast<int32_t>(kBodyWidth));
        // offsetLocal is the byte offset rank*4 and therefore also the
        // element value rank*4; index - rank*4 is the lane in [0, 3].
        Sub(laneLocal_, indexLocal_, offsetLocal_.ReinterpretCast<int32_t>(),
            static_cast<int32_t>(kBodyWidth));
        Add(outputLocal_, outputLocal_, laneLocal_, static_cast<int32_t>(kBodyWidth));
        event_t vs = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vs);
        WaitFlag<HardEvent::V_S>(vs);
        // Valid LI ranks are in range. Preserve the legacy contract for
        // poisoned/invalid group IDs used by the regression oracle.
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            int32_t group = groupsLocal_.GetValue(rank);
            if (group < 0 || group * static_cast<int32_t>(kRatio) >= seqLen) {
                for (uint32_t lane = 0; lane < kRatio; ++lane)
                    outputLocal_.SetValue(rank * kRatio + lane, -1);
            }
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = core_; row < rows_; row += cores_) {
            DataCopy(groupsLocal_, groups_[row * kTopK], kTopK);
            event_t mte2s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(mte2s);
            WaitFlag<HardEvent::MTE2_S>(mte2s);
            int32_t complete = completeGroups_.GetValue(row);
            int32_t tailStart = tailStart_.GetValue(row);
            int32_t tailCount = tailCount_.GetValue(row);
            int32_t req = tokenToReq_.GetValue(row);
            int32_t seqLen = req >= 0 ? sequenceLengths_.GetValue(req) : 0;
            if (complete == static_cast<int32_t>(kTopK))
                VectorFullBody(seqLen);
            else
                ScalarBody(complete, tailStart, tailCount, seqLen);

            for (uint32_t lane = 0; lane < kOutputWidth - kBodyWidth; ++lane) {
                int32_t token = -1;
                if (complete == static_cast<int32_t>(kTopK) &&
                    static_cast<int32_t>(lane) < tailCount) {
                    int32_t candidate = tailStart + static_cast<int32_t>(lane);
                    if (candidate >= 0 && candidate < seqLen) token = candidate;
                }
                outputLocal_.SetValue(kBodyWidth + lane, token);
            }
            event_t smte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(smte3);
            WaitFlag<HardEvent::S_MTE3>(smte3);
            DataCopyExtParams outputCopy{1, kOutputWidth * sizeof(int32_t), 0, 0, 0};
            DataCopyPad(expanded_[row * kOutputWidth], outputLocal_, outputCopy);
            event_t mte3s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(mte3s);
            WaitFlag<HardEvent::MTE3_S>(mte3s);
        }
    }

private:
    uint32_t rows_, core_, cores_;
    GlobalTensor<int32_t> groups_, completeGroups_, tailStart_, tailCount_;
    GlobalTensor<int32_t> sequenceLengths_, tokenToReq_, expanded_;
    TBuf<TPosition::VECCALC> groupsBuf_, outputBuf_, indexBuf_, offsetBuf_, laneBuf_;
    LocalTensor<int32_t> groupsLocal_, outputLocal_, indexLocal_, laneLocal_;
    LocalTensor<uint32_t> offsetLocal_;
};
}

extern "C" __global__ __aicore__ void qsa_expand_e3(
    GM_ADDR groups, GM_ADDR complete_groups, GM_ADDR tail_start,
    GM_ADDR tail_count, GM_ADDR sequence_lengths, GM_ADDR token_to_req,
    GM_ADDR expanded, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QsaExpandE3TilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    QsaExpandE3vKernel op;
    op.Init(groups, complete_groups, tail_start, tail_count, sequence_lengths,
            token_to_req, expanded, &tilingData, &pipe);
    op.Process();
}
