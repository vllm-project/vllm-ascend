#ifndef COMBINE_AND_DISPATCH_REBATCH_H
#define COMBINE_AND_DISPATCH_REBATCH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "combine_and_dispatch_rebatch_tiling.h"

namespace CombineAndDispatchRebatchImpl {

using namespace AscendC;
static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
static constexpr int UB_ALIGN_SIZE = 32;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = 128 * 1024 / 2;
constexpr static int32_t UB_HEAD_OFFSET = 60 * 1024;
constexpr static int32_t UB_MID_OFFSET = UB_HEAD_OFFSET + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + UB_ALIGN_SIZE;
// static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
#define DATA_FLUSH(_gm_tensor, _type) \
    Barrier(); \
    DataCacheCleanAndInvalid<_type, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(_gm_tensor);\
    __asm__("NOP"); \
    dsb(DSB_ALL);
template <typename T>
class CombineAndDispatchRebatch {
public:
    __aicore__ inline CombineAndDispatchRebatch() {}

    __aicore__ inline void Init(
        GM_ADDR batchBeforeDispatch,
        GM_ADDR residualsBeforeDispatch,
        GM_ADDR prefixIndex,
        GM_ADDR discardLabel,
        GM_ADDR discardTokens,
        GM_ADDR discardMoeReqIdTensor,
        GM_ADDR reqIdList,
        GM_ADDR discardResiduals,
        GM_ADDR runningTopk,
        GM_ADDR discardTopk,
        GM_ADDR runningTopkWeight,
        GM_ADDR discardTopkWeight,
        GM_ADDR moeCntInput,
        GM_ADDR actualBs,
        GM_ADDR historyBs,
        GM_ADDR runningBs,
        GM_ADDR originalBs,
        // GM_ADDR rebatchOutput,
        GM_ADDR rebatchResiduals,
        GM_ADDR workspaceGM,
        TPipe *pipe,
        const CombineAndDispatchRebatchTilingData *tilingData);
__aicore__ inline void Process()
{
    DATA_FLUSH(runningBsGm_, int32_t);
    DATA_FLUSH(historyBsGm_, int32_t);
    DATA_FLUSH(originalBsGm_, int32_t);
    runningBs_    = runningBsGm_.GetValue(0);
    historyBs_    = historyBsGm_.GetValue(0);
    originalBs_   = originalBsGm_.GetValue(0);
    pipe_barrier(PIPE_ALL);
    if (isCombineRebatch_ == 1) {

        // ---- combine_rebatch 多核并行 ----
        // 将 actualBs_ 个 token 按 blockNum 均分，每个 core 处理 [start, end) 范围。
        // 每个 core 独立扫描 cutIdx[0..start) 计算自己的输出起始偏移，
        // 再处理自己的 token 范围，各 core 写入位置天然不重叠，无需 core 间同步。
        //找到discardLabel中每个1的位置
        CombineRebatch();
        
    } else {

        // ---- dispatch_rebatch 多核并行 ----
        // moeCntInput_: 需要重新插入的 token 数（add_cnt）
        // runningBs_  : 重新插入前 batchBeforeDispatch 中已有的 token 数
        // historyBs  : 历史累计已重新插入的 token 数
        icache_preload(8);
        DATA_FLUSH(moeCntInputGm_, int32_t);
        uint32_t addCnt       = moeCntInputGm_.GetValue(0);
        //printf("addCnt=%d\n",addCnt);
        uint32_t newRunningBs = runningBs_ + addCnt;
        SyncAll();


        if (addCnt > 0) {
            // running_label_bs: reqIdList / discardLabel 中新插入 token 元数据的起始偏移
            uint32_t runningLabelBs = originalBs_ + historyBs_;
            // 阶段一：将 addCnt 个 token 的重新插入操作按 blockNum 均分
            pipe_barrier(PIPE_ALL);

            // 步骤 1c/1d: 并行更新 reqIdList 和 discardLabel 元数据
            LocalTensor<int32_t> labelLocal = tBuf.Get<int32_t>();
            Duplicate(labelLocal, static_cast<int32_t>(1), addCnt);
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            copyGmToGmWithSelectBlocks(reqIdListGm_[runningLabelBs],
                                        discardMoeReqIdTensorGm_,addCnt,blockNum,0);
            CpUBtoGM(discardLabelGm_[runningLabelBs],labelLocal, 1, addCnt * sizeof(int32_t));
           // if (blockIdx==0){AscendC::DumpTensor(discardLabelGm_, 4444444, 16);printf("%d\n",addCnt);}

            

            // 步骤 1e/1f/1g: 并行将 discard 缓冲区 token 写回 batchBeforeDispatch
        
            copyGmToGmWithSelectBlocks(batchBeforeDispatchGm_[runningBs_ * hiddenSize_],
                                        discardTokensGm_,addCnt * hiddenSize_,blockNum/4,0);
            copyGmToGmWithSelectBlocks(residualsBeforeDispatchGm_[runningBs_ * hiddenSize_],
                                        discardResidualsGm_,addCnt * hiddenSize_,blockNum/4,blockNum/4);
            copyGmToGmWithSelectBlocks(runningTopkGm_[runningBs_ * k_],
                                        discardTopkGm_,addCnt * k_,blockNum/4,blockNum/2);
             //  if (blockIdx==0) AscendC::DumpTensor(runningTopkGm_, 55555, 16);                         
            copyGmToGmWithSelectBlocks(runningTopkWeightGm_[runningBs_ * k_],
                                        discardTopkWeightGm_,addCnt * k_,blockNum/4,blockNum*3/4);
            
            if (blockIdx == 0) {
                LocalTensor<int32_t> runningBsLt = runningBsbuf.Get<int32_t>();
                runningBsLt.SetValue(0, runningBs_ + addCnt);
                //printf("addCnt=%d\n",runningBs_ + addCnt);
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                CpUBtoGM(runningBsGm_, runningBsLt, 1, sizeof(int32_t));
            } else if (blockIdx == 1) {
                LocalTensor<int32_t> historyBsLt = historyBsbuf.Get<int32_t>();
                historyBsLt.SetValue(0, historyBs_ + addCnt);
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                CpUBtoGM(historyBsGm_, historyBsLt, 1, sizeof(int32_t));
            } else {
                SyncFunc<AscendC::HardEvent::MTE3_S>();
                LocalTensor<int32_t> discardCntLt = moeCntBuf.Get<int32_t>();
                discardCntLt.SetValue(0, 0);
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                CpUBtoGM(moeCntInputGm_, discardCntLt, 1, sizeof(int32_t));
            }
        }

    }
   // if (blockIdx==0) AscendC::DumpTensor(runningTopkGm_, 6666, 16);
    // TODO: implement combine_and_dispatch_rebatch kernel logic
}

private:
    __aicore__ inline void InitGlobalBuffer(
        GM_ADDR batchBeforeDispatch,
        GM_ADDR residualsBeforeDispatch,
        GM_ADDR prefixIndex,
        GM_ADDR discardLabel,
        GM_ADDR discardTokens,
        GM_ADDR discardMoeReqIdTensor,
        GM_ADDR reqIdList,
        GM_ADDR discardResiduals,
        GM_ADDR runningTopk,
        GM_ADDR discardTopk,
        GM_ADDR runningTopkWeight,
        GM_ADDR discardTopkWeight,
        GM_ADDR moeCntInput,
        // GM_ADDR rebatchOutput,
        GM_ADDR rebatchResiduals,
        GM_ADDR actualBs,
        GM_ADDR historyBs,
        GM_ADDR runningBs,
        GM_ADDR originalBs
        );

    __aicore__ inline void InitTilingData(
        const CombineAndDispatchRebatchTilingData *tilingData);

__aicore__ inline void InitBuffers(TPipe *pipe)
{
    icache_preload(8);
    pipe->Reset();
    if (isCombineRebatch_ == 1) {
        pipe->InitBuffer(rowBuf, BUFFER_SIZE_BYTE_8K);
        pipe->InitBuffer(residualsBuf, BUFFER_SIZE_BYTE_8K);
        pipe->InitBuffer(runningTopkBuf, BUFFER_SIZE_BYTE_8K);
        pipe->InitBuffer(runningTopkWeightBuf, BUFFER_SIZE_BYTE_8K);
        
        
        pipe->InitBuffer(curIdBuf, BUFFER_SIZE_BYTE_8K);  //8
        pipe->InitBuffer(discardLabelBuf, BUFFER_SIZE_BYTE_8K);  //8
        pipe->InitBuffer(moeCntBuf, BUFFER_SIZE_BYTE_8K); //16
        pipe->InitBuffer(reqIdListBuf, BUFFER_SIZE_BYTE_8K); //16
        pipe->InitBuffer(tBuf1, BUFFER_SIZE_BYTE_8K);
        pipe->InitBuffer(tBuf, BUFFER_SIZE_BYTE_8K);
    } else {
        pipe->InitBuffer(moeCntBuf, BUFFER_SIZE_BYTE_8K);
        pipe->InitBuffer(tBuf, BUFFER_SIZE_BYTE_8K);

    }
    pipe->InitBuffer(runningBsbuf, BUFFER_SIZE_BYTE_8K);
    pipe->InitBuffer(historyBsbuf, BUFFER_SIZE_BYTE_8K);
}

template <typename TT>
__aicore__ inline void CpGMtoUB(LocalTensor<TT> outputLt, GlobalTensor<TT> inputGt, uint32_t size)
{
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    DataCopyPadExtParams<TT> padParams;
    DataCopyPad(outputLt, inputGt, dataCopyParams, padParams);
}

template <typename TT>
__aicore__ inline void CpUBtoGM(GlobalTensor<TT> outputGt, LocalTensor<TT> inputLt, uint16_t count, uint32_t size,
                                    uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams(count, size, srcStride / UB_ALIGN_SIZE, dstStride, 0);
    DataCopyPad(outputGt, inputLt, dataCopyParams);
}

template <typename TT>
__aicore__ inline void CpUB2GM(__gm__ TT *gmAddr, __ubuf__ TT *ubAddr, uint32_t size)
{
    LocalTensor<uint8_t> ubTensor;
    GlobalTensor<uint8_t> gmTensor;
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
    DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

template <typename TT>
__aicore__ inline void CpGM2UB(__ubuf__ TT *ubAddr, __gm__ TT *gmAddr, uint32_t size)
{
    LocalTensor<uint8_t> ubTensor;
    GlobalTensor<uint8_t> gmTensor;
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
    DataCopyPadExtParams<uint8_t> padParams;
    DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}

template<HardEvent eventType>
__aicore__ inline void SetWaitEvent(event_t eventId)
{
    AscendC::SetFlag<eventType>(eventId);
    AscendC::WaitFlag<eventType>(eventId);
}

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

template <typename TT, typename U = TT>
__aicore__ inline void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<U>& inputGT,
                                            const GlobalTensor<TT>& outputGT)
{
    // 只有在需要转换(U->T)时，UB会按比例(sizeof(U):sizeof(T))分成input/output两块，并32对齐
    constexpr int32_t ubBlockSize = UB_SINGLE_PING_PONG_ADD_SIZE_MAX;
    constexpr int32_t ubAlignNum = ubBlockSize / (sizeof(TT) + sizeof(U)) / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
    constexpr int32_t inputUbBlockSize = std::is_same_v<TT, U> ? ubBlockSize : ubAlignNum * sizeof(U);
    constexpr int32_t outputUbBlockSize = std::is_same_v<TT, U> ? ubBlockSize : ubAlignNum * sizeof(TT);
    __gm__ U *input = const_cast<__gm__ U *>(inputGT.GetPhyAddr());
    __gm__ TT *output = const_cast<__gm__ TT *>(outputGT.GetPhyAddr());
    __ubuf__ U* inputUB[2] = {(__ubuf__ U*)(UB_HEAD_OFFSET), (__ubuf__ U*)(UB_MID_OFFSET)};
    __ubuf__ TT* outputUB[2] = {(__ubuf__ TT*)inputUB[0], (__ubuf__ TT*)inputUB[1]};
    if constexpr (!std::is_same_v<TT, U>) {
        outputUB[0] = (__ubuf__ TT*)(inputUB[0] + inputUbBlockSize / sizeof(U));
        outputUB[1] = (__ubuf__ TT*)(inputUB[1] + inputUbBlockSize / sizeof(U));
    }
    int inputOffsetNum = 0;
    int outputOffsetNum = 0;
    if (dataSizeRemain <= 0) {
        return;
    }
    PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
    for (int64_t i = 0; dataSizeRemain > 0; i++) {
        // size 与 dataSizeRemain 都指的是output size
        uint32_t size = dataSizeRemain > outputUbBlockSize ? outputUbBlockSize : dataSizeRemain;
        event_t eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        CpGM2UB((i & 1) ? inputUB[0] : inputUB[1], input + inputOffsetNum, size / sizeof(TT) * sizeof(U));
        if constexpr (!std::is_same_v<TT, U>) {
            SetWaitEvent<HardEvent::MTE2_V>(eventId);
            CastImpl((i & 1) ? outputUB[0] : outputUB[1], (i & 1) ? inputUB[0] : inputUB[1], RoundMode::CAST_NONE,
                size / sizeof(TT));
            SetWaitEvent<HardEvent::V_MTE3>(eventId);
        }
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        CpUB2GM(output + outputOffsetNum, (i & 1) ? outputUB[0] : outputUB[1], size);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
        dataSizeRemain -= size;
        inputOffsetNum += (size / sizeof(TT));
        outputOffsetNum += (size / sizeof(TT));
    }
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
    AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID3); // Scalar等MTE3
    AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    PipeBarrier<PIPE_ALL>();
    return;
}
template <typename TT>
__aicore__ inline void copyGmToGmWithSelectBlocks(
        const GlobalTensor<TT> &outputGt, const GlobalTensor<TT> &inputGt, int elementNum, int usedBlockNum, int firstBlock) {
    int elementPerBlock = Ceil(elementNum, usedBlockNum);
    pipe_barrier(PIPE_ALL);
    int copyOffset = elementPerBlock * (blockIdx - firstBlock);
    if (copyOffset < 0) return;
    int copyLen = elementNum - copyOffset < elementPerBlock ? elementNum - copyOffset : elementPerBlock;
    pipe_barrier(PIPE_ALL);
    if (copyLen <= 0) {
        return;
    }
    // xcclCpGM2GM(outputGt[copyOffset], inputGt[copyOffset], copyLen * sizeof(T));
    CpGM2GMPingPong(copyLen * sizeof(TT), inputGt[copyOffset], outputGt[copyOffset]);
}
__aicore__ inline void CombineRebatch()
{

    actualBs_ = actualBsGm_.GetValue(0);
    if (actualBs_ == runningBs_) {
        // if (!copyResidualBack_) {
        //     copyGmToGmWithSelectBlocks(
        //         rebatchResidualsGm_, residualsBeforeDispatchGm_,
        //         runningBs_ * hiddenSize_, blockNum, 0);
        // }
        return;
    }
    icache_preload(8);
    DATA_FLUSH(moeCntInputGm_, int32_t);
    int32_t paddingCnt = moeCntInputGm_.GetValue(0);
    uint32_t labelScanEnd = originalBs_ + historyBs_;
    int32_t posBuf[100] = {0};
    uint32_t nzCnt = 0;
    LocalTensor<int32_t> discardTokenLocal = discardLabelBuf.Get<int32_t>();
    CpGMtoUB(discardTokenLocal, discardLabelGm_, maxBs_ * sizeof(int32_t));
    SyncAll<true>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t j = 0; j < maxBs_; j++) {
        if (discardTokenLocal.GetValue(j) != 0) {
            posBuf[nzCnt] = j;
            
            nzCnt++;
        }
    }

    LocalTensor<int32_t> prefixLocal = curIdBuf.Get<int32_t>();
    CpGMtoUB(prefixLocal, prefixIndexGm_, maxBs_ * sizeof(int32_t));





    SyncFunc<AscendC::HardEvent::MTE2_S>();

    LocalTensor<int32_t> reqIdListLocal = reqIdListBuf.Get<int32_t>();
    CpGMtoUB(reqIdListLocal, reqIdListGm_, maxBs_ * sizeof(int32_t));
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

    // 处理本 core 负责的 [start, end) token 范围
    LocalTensor<T> rowLocal = rowBuf.Get<T>();
    LocalTensor<T> residualsLt = residualsBuf.Get<T>();
    LocalTensor<int32_t> runningTopkLt = runningTopkBuf.Get<int32_t>();
    LocalTensor<bfloat16_t> runningTopkWeightLt = runningTopkWeightBuf.Get<bfloat16_t>();

    // 只按被截断 token 分核。cutCount <= blockNum 时，一个 AIV
    // 最多处理一个 token；截断数更大时才由一个 AIV 顺序处理多个。
    const uint32_t cutCount =
        static_cast<uint32_t>(runningBs_ - actualBs_);
    const uint32_t baseCutsPerCore = cutCount / blockNum;
    const uint32_t remainderCores = cutCount % blockNum;
    const uint32_t localCutCount =
        baseCutsPerCore + (blockIdx < remainderCores ? 1U : 0U);
    const uint32_t startCut =
        blockIdx * baseCutsPerCore +
        (blockIdx < remainderCores ? blockIdx : remainderCores);
    const uint32_t endCut = startCut + localCutCount;
    

    uint32_t discardOffset = 0;
    if (startCut < cutCount) {
      for (uint32_t i = 0; i < static_cast<uint32_t>(runningBs_); ++i) {
        const int32_t prefix = prefixLocal.GetValue(i);
        const int32_t prevPrefix =
            i > 0U ? prefixLocal.GetValue(i - 1U) : 0;
        const bool isCut = prefix == prevPrefix;
        if (!isCut) {
            continue;
        }
        if (discardOffset >= endCut) {
            break;
        }
        if (discardOffset >= startCut) {
            const int32_t idx = posBuf[i];
            int32_t appendOffset = paddingCnt + discardOffset;


            CpGMtoUB(rowLocal, batchBeforeDispatchGm_[i * hiddenSize_], hiddenSize_ * sizeof(T));
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
            CpUBtoGM(discardTokensGm_[appendOffset * hiddenSize_], rowLocal, 1, hiddenSize_ * sizeof(T));

            CpGMtoUB(residualsLt, residualsBeforeDispatchGm_[i * hiddenSize_], hiddenSize_ * sizeof(T));
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
            CpUBtoGM(discardResidualsGm_[appendOffset * hiddenSize_], residualsLt, 1, hiddenSize_ * sizeof(T));


            CpGMtoUB(runningTopkLt, runningTopkGm_[i * k_], k_ * sizeof(int32_t));
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID3);
            CpUBtoGM(discardTopkGm_[appendOffset * k_], runningTopkLt, 1, k_ * sizeof(int32_t));


            CpGMtoUB(runningTopkWeightLt, runningTopkWeightGm_[i * k_], k_ * sizeof(bfloat16_t));
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID4);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID4);
            CpUBtoGM(discardTopkWeightGm_[appendOffset * k_], runningTopkWeightLt, 1, k_ * sizeof(bfloat16_t));

	    

            int32_t discardReqId = reqIdListLocal.GetValue(idx);
            LocalTensor<int32_t> disCardLabelLt = tBuf.Get<int32_t>();
            disCardLabelLt.SetValue(0, 0);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            LocalTensor<int32_t> reqIdLt = tBuf1.Get<int32_t>();
            reqIdLt.SetValue(0, discardReqId);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            CpUBtoGM(discardMoeReqIdTensorGm_[appendOffset], reqIdLt, 1, sizeof(int32_t));
            CpUBtoGM(discardLabelGm_[idx], disCardLabelLt, 1, sizeof(int32_t));
            
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            SyncFunc<AscendC::HardEvent::MTE3_S>();

        }
        ++discardOffset;
      }

    }
    // if (copyResidualBack_) {
    //     SyncAll();
    //     copyGmToGmWithSelectBlocks(
    //         residualsBeforeDispatchGm_, rebatchResidualsGm_,
    //         maxBs_ * hiddenSize_ * sizeof(T), blockNum, 0);
    // }

    if (blockIdx == 0) {
        LocalTensor<int32_t> discardCntLt = moeCntBuf.Get<int32_t>();
        discardCntLt.SetValue(0, static_cast<int32_t>(paddingCnt + cutCount));
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        CpUBtoGM(moeCntInputGm_, discardCntLt, 1, sizeof(int32_t));
    }
	if (blockIdx == 1) {
        LocalTensor<int32_t> runningBsLt = runningBsbuf.Get<int32_t>();
        runningBsLt.SetValue(0, actualBs_);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        CpUBtoGM(runningBsGm_, runningBsLt, 1, sizeof(int32_t));
    }
    //printf("finish combine!");
}



    TPipe *pipe_;
    
    uint32_t isCombineRebatch_;
    uint32_t moeCntInput_;
    uint32_t hiddenSize_;
	uint32_t originalBs_;
    uint32_t maxBs_;
    uint32_t k_;
    uint32_t blockIdx;
    uint32_t blockNum;
    int32_t runningBs_;
    int32_t actualBs_;
    int32_t historyBs_;

    GlobalTensor<T>     batchBeforeDispatchGm_;
    GlobalTensor<T>     residualsBeforeDispatchGm_;
    GlobalTensor<int32_t> prefixIndexGm_;
    GlobalTensor<int32_t> discardLabelGm_;
    GlobalTensor<T>     discardTokensGm_;
    GlobalTensor<int32_t> discardMoeReqIdTensorGm_;
    GlobalTensor<int32_t> reqIdListGm_;
    GlobalTensor<T>     discardResidualsGm_;
    GlobalTensor<int32_t> runningTopkGm_;
    GlobalTensor<int32_t> discardTopkGm_;
    GlobalTensor<int32_t> moeCntInputGm_;
    GlobalTensor<int32_t> historyBsGm_;
    GlobalTensor<int32_t> originalBsGm_;
    GlobalTensor<int32_t> runningBsGm_;
    GlobalTensor<int32_t> actualBsGm_;
    GlobalTensor<T>     rebatchOutputGm_;
    GlobalTensor<T>     rebatchResidualsGm_;
    GlobalTensor<bfloat16_t> runningTopkWeightGm_;
    GlobalTensor<bfloat16_t> discardTopkWeightGm_;

    TBuf<> rowBuf;
    TBuf<> curIdBuf;
    TBuf<> discardLabelBuf;
    TBuf<> discardResBuf;
    TBuf<> moeCntBuf;
    TBuf<> reqIdListBuf;
    TBuf<> tBuf;
    TBuf<> tBuf1;
    TBuf<> residualsBuf;
    TBuf<> runningTopkBuf;
    TBuf<> runningTopkWeightBuf;
    TBuf<> runningBsbuf;
    TBuf<> historyBsbuf;
};

} // namespace CombineAndDispatchRebatchImpl

#endif // COMBINE_AND_DISPATCH_REBATCH_H
