#ifndef MATMUL_ALLREDUCE_ADD_RMSNORM_910C_AIV_KERNEL_H
#define MATMUL_ALLREDUCE_ADD_RMSNORM_910C_AIV_KERNEL_H

#include "kernel_operator.h"
#include "moe_distribute_base.h"
#include "matmul_allreduce_add_rmsnorm_910c_tiling.h"
#include "matmul_allreduce_add_rmsnorm_910c_utils.h"

namespace MatmulAllreduceAddRmsnorm910cImpl {

using namespace AscendC;

constexpr uint32_t UB_ALIGN_BYTES = 32;
constexpr uint32_t TARGET_ROW_ELEMENTS = 5120;
constexpr float TARGET_ROW_ELEMENTS_RECIPROCAL = 1.0f / 5120.0f;
constexpr uint32_t FP32_ELEMENTS_PER_REPEAT = 64;
constexpr uint32_t MAX_FINE_GRAIN_CORES = 24;
constexpr uint32_t AIC_TILES_PER_AIV_GROUP = 8;
constexpr uint32_t AIV_GROUP_ELEMENTS = FINE_GRAIN_TILE_ELEMENTS * AIC_TILES_PER_AIV_GROUP;
constexpr uint32_t PROTOCOL_READY_FLAG_INDEX = MAX_FINE_GRAIN_CORES - 1;
constexpr uint32_t ADD_READY_FLAG_BASE = MAX_FINE_GRAIN_CORES;
constexpr uint32_t MAX_AIV_GROUPS =
    (TARGET_ROW_ELEMENTS + AIV_GROUP_ELEMENTS - 1) / AIV_GROUP_ELEMENTS;
constexpr uint32_t RMS_READY_FLAG_INDEX = ADD_READY_FLAG_BASE + MAX_AIV_GROUPS;
constexpr uint64_t RANK_DATA_STRIDE = 1024UL * 1024UL;
constexpr uint64_t DATA_BANK_STRIDE = 2UL * RANK_DATA_STRIDE;
constexpr uint64_t STATUS_BANK_STRIDE = 512UL * 1024UL;
constexpr uint64_t STATUS_RANK_STRIDE = 1024UL;
constexpr uint64_t RECIPROCAL_RMS_OFFSET = 192UL * 1024UL;
constexpr uint64_t STATUS_CONTROL_OFFSET = 256UL * 1024UL;
constexpr uint64_t DATA_BANK_CONTROL_OFFSET = 900UL * 1024UL;
constexpr uint32_t MAX_SUPPORTED_RANKS = 2;

template <HardEvent event>
__aicore__ inline void SyncPipe()
{
    int32_t eventId = GetTPipePtr()->FetchEventID(event);
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T>
class MatmulAllreduceAddRmsnorm910cAivKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR residual, GM_ADDR gamma, GM_ADDR output, GM_ADDR addOutput,
        GM_ADDR workspace,
        const MatmulAllreduceAddRmsnorm910cTilingData *tilingData,
        __gm__ HcclOpResParam *windowContext)
    {
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
        residual_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(residual));
        gamma_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gamma));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
        addOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(addOutput));
        readyFlags_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace),
            FINE_GRAIN_READY_WORKSPACE_BYTES / sizeof(int32_t));
        windowContext_ = windowContext;

        const auto &shape = tilingData->matmulAllreduceAddRmsnormInfo.ppTilingData.opShape;
        const auto &comm = tilingData->matmulAllreduceAddRmsnormInfo.commTilingData;
        rowCount_ = static_cast<uint32_t>(shape.m);
        rowElements_ = static_cast<uint32_t>(shape.n);
        epsilon_ = tilingData->matmulAllreduceAddRmsnormInfo.rmsnormTilingData.epsilon;
        rank_ = static_cast<uint32_t>(comm.rank);
        rankSize_ = static_cast<uint32_t>(comm.rankSize);
        coreIndex_ = GetBlockIdx() / GetTaskRation();
        subBlockIndex_ = GetSubBlockIdx();
        activeCoreCount_ = DivCeil(rowElements_, AIV_GROUP_ELEMENTS);

        pipe_.InitBuffer(copyBuffer_, TARGET_ROW_ELEMENTS * sizeof(T));
        pipe_.InitBuffer(source0Buffer_, TARGET_ROW_ELEMENTS * sizeof(T));
        pipe_.InitBuffer(source1Buffer_, TARGET_ROW_ELEMENTS * sizeof(T));
        pipe_.InitBuffer(accBuffer_, TARGET_ROW_ELEMENTS * sizeof(float));
        pipe_.InitBuffer(tmpBuffer_, TARGET_ROW_ELEMENTS * sizeof(float));
        pipe_.InitBuffer(reduceBuffer_, FP32_ELEMENTS_PER_REPEAT * sizeof(float));
        pipe_.InitBuffer(statusBuffer_, FINE_GRAIN_READY_WORKSPACE_BYTES);
        pipe_.InitBuffer(gatherBuffer_, MAX_FINE_GRAIN_CORES * sizeof(float));
        pipe_.InitBuffer(gatherTmpBuffer_, sizeof(uint32_t));
        pipe_.InitBuffer(statusSumBuffer_, UB_ALIGN_BYTES);
        pipe_.InitBuffer(statusFlagBuffer_, UB_ALIGN_BYTES);
    }

    __aicore__ inline void Process()
    {
        if constexpr (FINE_GRAIN_DEBUG_STAGE >= 0) {
            if (IsLeader()) {
                ClearReadyFlags();
            }
        }
        FFTSCrossCoreSync<PIPE_MTE3>(FFTS_SYNC_AICORE_GROUP_MODE, AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID);
        WaitEvent(0);
        if (rankSize_ != MAX_SUPPORTED_RANKS || rowCount_ != 1 ||
            rowElements_ != TARGET_ROW_ELEMENTS || activeCoreCount_ > MAX_FINE_GRAIN_CORES) {
            return;
        }

        if constexpr (FINE_GRAIN_DEBUG_STAGE < 0) {
            WaitForAllAicCores();
            return;
        }

        if constexpr (FINE_GRAIN_DEBUG_STAGE == 0) {
            if (IsActiveWorker()) {
                WaitForAicTile();
            }
            WaitForAllAicCores();
            return;
        }

        if (IsLeader()) {
            InitProtocolState();
            PublishProtocolState();
        }

        if (IsActiveWorker()) {
            if (!IsLeader()) {
                WaitForProtocolState();
                LoadProtocolState();
            }
            WaitForAicTile();
            PublishChunk();
            PublishStatus();
            WaitForAllRanks();
            ReduceAdd();
        }

        WaitForAllAicCores();
        if (!IsActiveWorker()) {
            return;
        }
        PublishReadyFlag(ADD_READY_FLAG_BASE + coreIndex_);
        if (IsLeader()) {
            WaitForAllAddOutputs();
            FinishRmsReduction();
            PublishReadyFlag(RMS_READY_FLAG_INDEX);
        } else {
            WaitForReadyFlag(RMS_READY_FLAG_INDEX);
        }
        NormalizeAndStore();
    }

private:
    __aicore__ inline void ClearReadyFlags()
    {
        LocalTensor<int32_t> ready = statusBuffer_.Get<int32_t>();
        constexpr uint32_t elementCount = FINE_GRAIN_READY_WORKSPACE_BYTES / sizeof(int32_t);
        Duplicate(ready, 0, elementCount);
        SyncPipe<HardEvent::V_MTE3>();
        DataCopy(readyFlags_, ready, elementCount);
        SyncPipe<HardEvent::MTE3_S>();
    }

    __aicore__ inline void WaitForAicTile()
    {
        uint32_t firstTile = coreIndex_ * AIC_TILES_PER_AIV_GROUP;
        uint32_t tileCount = DivCeil(rowElements_, FINE_GRAIN_TILE_ELEMENTS);
        uint32_t endTile = firstTile + AIC_TILES_PER_AIV_GROUP;
        endTile = endTile < tileCount ? endTile : tileCount;
        for (uint32_t tile = firstTile; tile < endTile; ++tile) {
            WaitForReadyFlag(tile);
        }
    }

    __aicore__ inline void WaitForProtocolState()
    {
        WaitForReadyFlag(PROTOCOL_READY_FLAG_INDEX);
    }

    __aicore__ inline void WaitForReadyFlag(uint32_t flagIndex)
    {
        LocalTensor<int32_t> ready = statusFlagBuffer_.Get<int32_t>();
        uint32_t offset = flagIndex * (FINE_GRAIN_READY_FLAG_STRIDE_BYTES / sizeof(int32_t));
        int32_t value = 0;
        while (value != 1) {
            DataCopy(ready, readyFlags_[offset], UB_ALIGN_BYTES / sizeof(int32_t));
            SyncPipe<HardEvent::MTE2_S>();
            value = ready.GetValue(0);
        }
    }

    __aicore__ inline void PublishReadyFlag(uint32_t flagIndex)
    {
        uint32_t offset = flagIndex * (FINE_GRAIN_READY_FLAG_STRIDE_BYTES / sizeof(int32_t));
        readyFlags_.SetValue(offset, 1);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
            DcciDst::CACHELINE_OUT>(readyFlags_[offset]);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WaitForAllAddOutputs()
    {
        for (uint32_t core = 0; core < activeCoreCount_; ++core) {
            WaitForReadyFlag(ADD_READY_FLAG_BASE + core);
        }
    }

    __aicore__ inline bool IsLeader() const
    {
        return coreIndex_ == 0 && subBlockIndex_ == 0;
    }

    __aicore__ inline bool IsActiveWorker() const
    {
        return subBlockIndex_ == 0 && coreIndex_ < activeCoreCount_;
    }

    __aicore__ inline uint32_t ChunkOffset() const
    {
        return coreIndex_ * AIV_GROUP_ELEMENTS;
    }

    __aicore__ inline uint32_t ChunkCount() const
    {
        uint32_t remaining = rowElements_ - ChunkOffset();
        return remaining < AIV_GROUP_ELEMENTS ? remaining : AIV_GROUP_ELEMENTS;
    }

    __aicore__ inline void WaitForAllAicCores()
    {
        FFTSCrossCoreSync<PIPE_MTE3>(FFTS_SYNC_INTERNEL_MODE, MAX_BLOCK_COUNT);
        WaitEvent(MAX_BLOCK_COUNT);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ReduceSumLikeAddRmsNorm(
        const LocalTensor<float> &dst, const LocalTensor<float> &src,
        const LocalTensor<float> &work, uint32_t count)
    {
        constexpr uint32_t repeatBytes = 256;
        constexpr uint32_t blockBytes = 32;
        uint32_t repeatTimes = count / FP32_ELEMENTS_PER_REPEAT;
        uint32_t tailCount = count % FP32_ELEMENTS_PER_REPEAT;
        uint32_t bodyCount = repeatTimes * FP32_ELEMENTS_PER_REPEAT;
        BinaryRepeatParams params;
        params.src0RepStride = repeatBytes / blockBytes;
        params.src0BlkStride = 1;
        params.src1RepStride = 0;
        params.src1BlkStride = 1;
        params.dstRepStride = 0;
        params.dstBlkStride = 1;

        Duplicate(work, 0.0f, FP32_ELEMENTS_PER_REPEAT);
        PipeBarrier<PIPE_V>();
        if (repeatTimes > 0) {
            Add(work, src, work, FP32_ELEMENTS_PER_REPEAT, repeatTimes, params);
            PipeBarrier<PIPE_V>();
        }
        if (tailCount != 0) {
            Add(work, src[bodyCount], work, tailCount, 1, params);
            PipeBarrier<PIPE_V>();
        }
        AscendCUtils::SetMask<float>(FP32_ELEMENTS_PER_REPEAT);
        WholeReduceSum<float, false>(dst, work, MASK_PLACEHOLDER, 1, 0, 1, 0);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline GM_ADDR GetWindowAddress(uint32_t rankId) const
    {
        if (rankId == rank_) {
            return reinterpret_cast<GM_ADDR>(windowContext_->localWindowsIn);
        }
        auto relation = reinterpret_cast<__gm__ HcclRankRelationResV2 *>(
            windowContext_->remoteRes[rankId].nextDevicePtr);
        return reinterpret_cast<GM_ADDR>(relation->windowsIn);
    }

    __aicore__ inline GM_ADDR GetStatusAddress(uint32_t rankId) const
    {
        GM_ADDR base;
        if (rankId == rank_) {
            base = reinterpret_cast<GM_ADDR>(windowContext_->localWindowsExp);
        } else {
            auto relation = reinterpret_cast<__gm__ HcclRankRelationResV2 *>(
                windowContext_->remoteRes[rankId].nextDevicePtr);
            base = reinterpret_cast<GM_ADDR>(relation->windowsExp);
        }
        return base + dataBank_ * STATUS_BANK_STRIDE;
    }

    __aicore__ inline void InitProtocolState()
    {
        GM_ADDR localStatusBase = reinterpret_cast<GM_ADDR>(windowContext_->localWindowsExp);
        GlobalTensor<int32_t> bankControl;
        bankControl.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            localStatusBase + DATA_BANK_CONTROL_OFFSET));
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(bankControl[0]);
        dataBank_ = static_cast<uint32_t>(bankControl.GetValue(0) & 1);
        bankControl.SetValue(0, static_cast<int32_t>(1U - dataBank_));
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(bankControl[0]);
        PipeBarrier<PIPE_ALL>();

        GlobalTensor<int32_t> stateControl;
        stateControl.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            localStatusBase + dataBank_ * STATUS_BANK_STRIDE + STATUS_CONTROL_OFFSET));
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(stateControl[0]);
        int32_t previous = stateControl.GetValue(0);
        stateValue_ = previous == 0 ? 0x3F800000 : 0;
        stateTarget_ = previous == 0 ? 1.0f : 0.0f;
        stateControl.SetValue(0, stateValue_);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(stateControl[0]);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PublishProtocolState()
    {
        PublishReadyFlag(PROTOCOL_READY_FLAG_INDEX);
    }

    __aicore__ inline void LoadProtocolState()
    {
        GM_ADDR localStatusBase = reinterpret_cast<GM_ADDR>(windowContext_->localWindowsExp);
        GlobalTensor<int32_t> bankControl;
        bankControl.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            localStatusBase + DATA_BANK_CONTROL_OFFSET));
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(bankControl[0]);
        dataBank_ = 1U - static_cast<uint32_t>(bankControl.GetValue(0) & 1);

        GlobalTensor<int32_t> stateControl;
        stateControl.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            localStatusBase + dataBank_ * STATUS_BANK_STRIDE + STATUS_CONTROL_OFFSET));
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(stateControl[0]);
        stateValue_ = stateControl.GetValue(0);
        stateTarget_ = stateValue_ == 0 ? 0.0f : 1.0f;
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PublishChunk()
    {
        LocalTensor<T> copy = copyBuffer_.Get<T>();
        uint32_t offset = ChunkOffset();
        uint32_t count = ChunkCount();
        uint64_t bankOffset = dataBank_ * DATA_BANK_STRIDE + rank_ * RANK_DATA_STRIDE;
        DataCopy(copy, input_[offset], count);
        SyncPipe<HardEvent::MTE2_MTE3>();
        uint32_t peer = 1U - rank_;
        GlobalTensor<T> peerWindow;
        peerWindow.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(
            GetWindowAddress(peer) + bankOffset));
        DataCopy(peerWindow[offset], copy, count);
        SyncPipe<HardEvent::MTE3_S>();
    }

    __aicore__ inline void PublishStatus()
    {
        LocalTensor<int32_t> statusFlag = statusFlagBuffer_.Get<int32_t>();
        statusFlag.SetValue(0, stateValue_);
        SyncPipe<HardEvent::S_MTE3>();
        uint64_t offset = rank_ * STATUS_RANK_STRIDE + coreIndex_ * UB_ALIGN_BYTES;
        for (uint32_t destination = 0; destination < rankSize_; ++destination) {
            GlobalTensor<int32_t> destinationStatus;
            destinationStatus.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
                GetStatusAddress(destination) + offset));
            DataCopy(destinationStatus, statusFlag, UB_ALIGN_BYTES / sizeof(int32_t));
        }
        SyncPipe<HardEvent::MTE3_S>();
    }

    __aicore__ inline void WaitForAllRanks()
    {
        LocalTensor<float> status = statusBuffer_.Get<float>();
        LocalTensor<float> gathered = gatherBuffer_.Get<float>();
        LocalTensor<uint32_t> gatherTmp = gatherTmpBuffer_.Get<uint32_t>();
        LocalTensor<float> statusSum = statusSumBuffer_.Get<float>();
        GlobalTensor<float> localStatus;
        localStatus.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            GetStatusAddress(rank_) + coreIndex_ * UB_ALIGN_BYTES));

        gatherTmp.SetValue(0, 1);
        SyncPipe<HardEvent::S_V>();
        constexpr uint16_t rankStrideBlocks = STATUS_RANK_STRIDE / UB_ALIGN_BYTES;
        DataCopyParams copyParams{static_cast<uint16_t>(rankSize_), 1,
            static_cast<uint16_t>(rankStrideBlocks - 1), 0};
        SumParams sumParams{1, rankSize_, rankSize_};
        float minTarget = stateTarget_ * rankSize_ - 0.5f;
        float maxTarget = stateTarget_ * rankSize_ + 0.5f;
        float sum = -1.0f;
        while (sum < minTarget || sum > maxTarget) {
            DataCopy<float>(status, localStatus, copyParams);
            SyncPipe<HardEvent::MTE2_V>();
            uint64_t reservedCount = 0;
            GatherMask(gathered, status, gatherTmp, true, 1,
                {1, static_cast<uint16_t>(rankSize_), 1, 0}, reservedCount);
            PipeBarrier<PIPE_V>();
            Sum(statusSum, gathered, sumParams);
            SyncPipe<HardEvent::V_S>();
            sum = statusSum.GetValue(0);
        }
    }

    __aicore__ inline void ReduceAdd()
    {
        uint32_t offset = ChunkOffset();
        uint32_t count = ChunkCount();
        uint64_t bankOffset = dataBank_ * DATA_BANK_STRIDE;
        uint32_t peer = 1U - rank_;
        GlobalTensor<T> peerSource;
        peerSource.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(
            GetWindowAddress(rank_) + bankOffset + peer * RANK_DATA_STRIDE));

        LocalTensor<T> input0 = source0Buffer_.Get<T>();
        LocalTensor<T> input1 = source1Buffer_.Get<T>();
        LocalTensor<T> staging = copyBuffer_.Get<T>();
        LocalTensor<float> acc = accBuffer_.Get<float>();
        LocalTensor<float> tmp = tmpBuffer_.Get<float>();
        DataCopy(input0, input_[offset], count);
        DataCopy(input1, peerSource[offset], count);
        DataCopy(staging, residual_[offset], count);
        SyncPipe<HardEvent::MTE2_V>();
        Cast(acc, input0, RoundMode::CAST_NONE, count);
        Cast(tmp, input1, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Add(acc, acc, tmp, count);
        PipeBarrier<PIPE_V>();

        Cast(input0, acc, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Cast(acc, input0, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Cast(tmp, staging, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Add(acc, acc, tmp, count);
        PipeBarrier<PIPE_V>();

        Cast(input0, acc, RoundMode::CAST_RINT, count);
        SyncPipe<HardEvent::V_MTE3>();
        DataCopy(addOutput_[offset], input0, count);
        SyncPipe<HardEvent::MTE3_S>();
    }

    __aicore__ inline void FinishRmsReduction()
    {
        LocalTensor<float> statusSum = statusSumBuffer_.Get<float>();
        LocalTensor<float> acc = accBuffer_.Get<float>();
        LocalTensor<float> tmp = tmpBuffer_.Get<float>();
        LocalTensor<float> reduceWork = reduceBuffer_.Get<float>();
        LocalTensor<T> addInput = source0Buffer_.Get<T>();
        DataCopy(addInput, addOutput_, rowElements_);
        SyncPipe<HardEvent::MTE2_V>();
        Cast(acc, addInput, RoundMode::CAST_NONE, rowElements_);
        PipeBarrier<PIPE_V>();
        Mul(tmp, acc, acc, rowElements_);
        PipeBarrier<PIPE_V>();
        Muls(tmp, tmp, TARGET_ROW_ELEMENTS_RECIPROCAL, rowElements_);
        PipeBarrier<PIPE_V>();
        ReduceSumLikeAddRmsNorm(statusSum, tmp, reduceWork, rowElements_);
        PipeBarrier<PIPE_V>();
        Adds(statusSum, statusSum, epsilon_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(statusSum, statusSum, 1);
        Duplicate(acc, 1.0f, 1);
        PipeBarrier<PIPE_V>();
        Div(statusSum, acc, statusSum, 1);

        GlobalTensor<float> reciprocalRms;
        reciprocalRms.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            GetStatusAddress(rank_) + RECIPROCAL_RMS_OFFSET));
        SyncPipe<HardEvent::V_MTE3>();
        DataCopy(reciprocalRms, statusSum, UB_ALIGN_BYTES / sizeof(float));
        SyncPipe<HardEvent::MTE3_S>();
    }

    __aicore__ inline void NormalizeAndStore()
    {
        uint32_t offset = ChunkOffset();
        uint32_t count = ChunkCount();
        LocalTensor<T> input0 = source0Buffer_.Get<T>();
        LocalTensor<T> staging = copyBuffer_.Get<T>();
        LocalTensor<float> acc = accBuffer_.Get<float>();
        LocalTensor<float> tmp = tmpBuffer_.Get<float>();
        LocalTensor<float> statusSum = statusSumBuffer_.Get<float>();
        GlobalTensor<float> reciprocalRmsGlobal;
        reciprocalRmsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            GetStatusAddress(rank_) + RECIPROCAL_RMS_OFFSET));
        DataCopy(statusSum, reciprocalRmsGlobal, UB_ALIGN_BYTES / sizeof(float));
        SyncPipe<HardEvent::MTE2_S>();
        float reciprocalRms = statusSum.GetValue(0);

        SyncPipe<HardEvent::S_V>();
        Cast(acc, input0, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Muls(acc, acc, reciprocalRms, count);
        PipeBarrier<PIPE_V>();
        Cast(input0, acc, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Cast(acc, input0, RoundMode::CAST_NONE, count);
        DataCopy(staging, gamma_[offset], count);
        SyncPipe<HardEvent::MTE2_V>();
        Cast(tmp, staging, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Mul(acc, acc, tmp, count);
        PipeBarrier<PIPE_V>();
        Cast(input0, acc, RoundMode::CAST_RINT, count);
        SyncPipe<HardEvent::V_MTE3>();
        DataCopy(output_[offset], input0, count);
        SyncPipe<HardEvent::MTE3_MTE2>();
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> copyBuffer_;
    TBuf<QuePosition::VECCALC> source0Buffer_;
    TBuf<QuePosition::VECCALC> source1Buffer_;
    TBuf<QuePosition::VECCALC> accBuffer_;
    TBuf<QuePosition::VECCALC> tmpBuffer_;
    TBuf<QuePosition::VECCALC> reduceBuffer_;
    TBuf<QuePosition::VECCALC> statusBuffer_;
    TBuf<QuePosition::VECCALC> gatherBuffer_;
    TBuf<QuePosition::VECCALC> gatherTmpBuffer_;
    TBuf<QuePosition::VECCALC> statusSumBuffer_;
    TBuf<QuePosition::VECCALC> statusFlagBuffer_;
    GlobalTensor<T> input_;
    GlobalTensor<T> residual_;
    GlobalTensor<T> gamma_;
    GlobalTensor<T> output_;
    GlobalTensor<T> addOutput_;
    GlobalTensor<int32_t> readyFlags_;
    __gm__ HcclOpResParam *windowContext_ = nullptr;
    uint32_t rowCount_ = 0;
    uint32_t rowElements_ = 0;
    uint32_t rank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIndex_ = 0;
    uint32_t subBlockIndex_ = 0;
    uint32_t activeCoreCount_ = 0;
    uint32_t dataBank_ = 0;
    int32_t stateValue_ = 0;
    float stateTarget_ = 0.0f;
    float epsilon_ = 0.0f;
};

}  // namespace MatmulAllreduceAddRmsnorm910cImpl

using MatmulAllreduceAddRmsnorm910cImpl::MatmulAllreduceAddRmsnorm910cAivKernel;

#endif
