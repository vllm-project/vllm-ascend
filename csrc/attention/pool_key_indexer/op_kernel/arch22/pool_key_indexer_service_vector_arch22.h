/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_KEY_INDEXER_SERVICE_VECTOR_ARCH22_H
#define POOL_KEY_INDEXER_SERVICE_VECTOR_ARCH22_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../pool_key_indexer_common.h"
#include "pool_key_indexer_vector.h"

namespace PkiKernel {
using namespace PkiCommon;
using namespace PkiServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t SPARSE_COUNT_4K = 4096;
constexpr uint32_t LD_PARAM_NUM = 16;
// 展开输出分段长度(int32 元素数): 展开/清理按此分批生成并写 GM, 使
// expandOutBuf_ 固定 4KB, 不随 outputLen 增长越 arch22 192KB UB 上限
constexpr uint32_t EXPAND_CHUNK = 1024;
// vgather 展开路径的段长(独立于标量/vecPath 的 EXPAND_CHUNK; Gather/Muls/Add
// 均为 Level-2 count 形式, 2201 下软件展开为 count-mask, repeatTime=count/64)。
constexpr uint32_t EXPAND_GATHER_CHUNK = 512;
// A2/A3 AIV TPipe UB 池硬上限(release 模式 InitBuffer 越池不报错), 超限回退标量展开。
constexpr uint32_t PKI_UB_POOL_LIMIT_BYTES = 196608;
// poolSize%8!=0 展开路径向量化开关: 1=vgather 向量展开(Gather+Muls+Add);
// 0=回退全标量展开路径(逐 token GetValue/SetValue)。
#ifndef PKI_EXPAND_VEC
#define PKI_EXPAND_VEC 1
#endif
constexpr uint32_t EVENTID_V_TO_MTE2_PING = 0;
constexpr uint32_t EVENTID_V_TO_MTE2_PONG = 1;
constexpr uint32_t EVENTID_V_TO_MTE2_TMPUB = 2;

// 主模板：Q_T必选，W_T可选（默认void），无论W_T传什么，默认weightsType=Q_T
template <typename Q_T, typename W_T = void>
struct PkiTypeTraits {
    using weightsType = Q_T; // 默认：weightsType绑定Q_T
};

// 偏特化1：固定第二个参数W_T=float，Q_T保留泛型
template <typename Q_T>
struct PkiTypeTraits<Q_T, float> {
    using weightsType = float; // W_T=float时，强制weightsType为float
};

// FP8模式（仅arch35实际实例化，此处保持同构以防误用）：weights张量为FP16，
// W_T绑定half，避免vector侧weights路径实例化fp8标量语义
template <>
struct PkiTypeTraits<fp8_e4m3fn_t> {
    using weightsType = half;
};

template <typename LIT>
class PoolKeyIndexerServiceVector {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    static constexpr PkiLayout LAYOUT_T = LIT::layout;
    using W_DERIVED_T =
        typename PkiTypeTraits<Q_T, typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;
    // 量化场景: PkiType 显式携带 weightsType(half/bfloat16_t), 优先于推导
    using W_EXPLICIT_T = typename LIT::weightsType;
    using W_T = typename std::conditional<std::is_same<W_EXPLICIT_T, void>::value, W_DERIVED_T, W_EXPLICIT_T>::type;

    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline PoolKeyIndexerServiceVector(){};
    __aicore__ inline void ProcessVec(const PkiCommon::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct PkiCommon::ConstInfo &constInfo,
                                      const PoolKeyIndexerTilingData *__restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<W_T> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<float> valueOutGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);

private:
    __aicore__ inline void ExpandAndAppendIndices(LocalTensor<int32_t> poolIndices, LocalTensor<int32_t> &tokenIndices,
                                                  LocalTensor<int32_t> &workBuf, GlobalTensor<int32_t> &idxOutGm,
                                                  int64_t idxOutBase, uint32_t sparseCount, uint32_t poolSize,
                                                  uint32_t validS2Len, int32_t poolTailK, int32_t L_orig,
                                                  uint32_t curS1Idx, uint32_t curS1Size);
    // vgather 展开模板(qOff/pTpl)在 workLocal_ 中的起始偏移(poolIndices 区之后)
    __aicore__ inline uint32_t ExpandGatherTplOffset()
    {
        uint32_t alignedPoolSize = PkiCommon::Align(poolSize_, (uint32_t)8);
        return alignedPoolSize + PkiCommon::Align(static_cast<uint32_t>(constInfo_.sparseCount), (uint32_t)8);
    }
    // 一次性构建 qOff[i]=(i/ps)*4 / pTpl[i]=i%ps 展开模板, 常驻 workLocal_ 尾部
    __aicore__ inline void BuildExpandGatherTpl();

    // 标量(S pipe)与向量(V pipe)/MTE3 间必须显式硬同步,
    // PipeBarrier<PIPE_V> 不保证 S 侧读写顺序
    __aicore__ inline void VToSSync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
    }
    __aicore__ inline void SToVSync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }
    __aicore__ inline void SToMTE3Sync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventID);
        WaitFlag<HardEvent::S_MTE3>(eventID);
    }
    // 向量写(V pipe)后由 MTE3 读出前的真跨流水同步(SetFlag 排入 V 队尾,
    // WaitFlag 在 MTE3 队等待), 分段展开输出每段 DataCopyPad 前使用
    __aicore__ inline void VToMTE3Sync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
    }
    // 上段 DataCopyPad(MTE3 读 tokenIndices) 完成后再覆写 tokenIndices 的同步
    __aicore__ inline void MTE3ToVSync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventID);
        WaitFlag<HardEvent::MTE3_V>(eventID);
    }

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<float> valueOutGm;
    // =================================常量区=================================

private:
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    TBuf<TPosition::VECCALC> expandOutBuf_;
    LocalTensor<int32_t> expandOutLocal_;
    TBuf<TPosition::VECCALC> workBuf_;
    LocalTensor<int32_t> workLocal_;

    LocalTensor<float> tmpUb_;
    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> SortedBasicBlock_;

    int32_t blockId_ = -1;
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    uint32_t poolSize_ = 1;
    uint32_t outputLen_ = 0;
    // poolSize%8!=0 且 UB 预算允许时置 true(InitBuffers), 展开走 vgather 路径
    bool expandGatherOn_ = false;

    // para for LD
    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = 16;
    int32_t virTopK = 0;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    struct PkiCommon::ConstInfo constInfo_;
};

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;
    virTopK = constInfo_.isSparseCountOver2K ? constInfo_.sparseCount : BASE_TOPK;

    // 尺寸落具名局部量, 供下方 UB 预算按字节核算(与 InitBuffer 调用同源)
    uint32_t tmpBufSize = (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float);
    uint32_t sortOutBufSize = CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float);
    uint32_t indexBufSize = s2BaseSize_ * sizeof(int32_t);
    uint32_t reduceOutBufSize = s2BaseSize_ * 2 * sizeof(float);
    uint32_t brcBufSize = groupInner_ * 8 * sizeof(float);
    uint32_t paramBufSize = LD_PARAM_NUM * sizeof(int64_t);

    pipe->InitBuffer(outQueue_, 1, outNeedBufSize); // 32KB  extract
    // 68KB 在搬运cube核计算得到的结果和weight时，分成两块34KB，用于db；在mrgsort时，用作临时UB
    pipe->InitBuffer(tmpBuf_, tmpBufSize);
    pipe->InitBuffer(sortOutBuf_, sortOutBufSize);     // 64KB
    pipe->InitBuffer(indexBuf_, indexBufSize);         // 2KB
    pipe->InitBuffer(reduceOutBuf_, reduceOutBufSize); // 4KB
    pipe->InitBuffer(brcBuf_, brcBufSize);
    pipe->InitBuffer(paramBuf_, paramBufSize);

    if (poolSize_ > 1) {
        // expandOutLocal_ 固定 EXPAND_CHUNK(1024) 元素, 展开/清理按此分段写 GM,
        // 避免整行 outputLen 分配越 arch22 192KB UB 上限
        uint32_t expandOutLen = PkiCommon::Align(static_cast<uint64_t>(EXPAND_CHUNK), (uint64_t)8);
        pipe->InitBuffer(expandOutBuf_, expandOutLen * sizeof(uint32_t));
        expandOutLocal_ = expandOutBuf_.Get<int32_t>();
        uint32_t workSize = PkiCommon::Align(static_cast<uint64_t>(poolSize_ + 64), (uint64_t)8) +
                            PkiCommon::Align(static_cast<uint64_t>(constInfo_.sparseCount + 64), (uint64_t)8);
#if PKI_EXPAND_VEC
        // ps%8!=0 时 workLocal_ 尾部常驻 qOff/pTpl 模板; 按 UB 池上限核算,
        // 超限(如 ps=2/topk=8192→sc=4096)回退标量路径。
        expandGatherOn_ = (poolSize_ % 8 != 0);
        if (expandGatherOn_) {
            uint32_t tplEnd = ExpandGatherTplOffset() + EXPAND_GATHER_CHUNK * 2;
            uint32_t fixedBytes = outNeedBufSize + tmpBufSize + sortOutBufSize + indexBufSize + reduceOutBufSize +
                                  brcBufSize + paramBufSize + expandOutLen * sizeof(uint32_t);
            uint32_t gatherBytes = PkiCommon::Max(workSize, tplEnd) * sizeof(uint32_t);
            if (fixedBytes + gatherBytes <= PKI_UB_POOL_LIMIT_BYTES) {
                workSize = PkiCommon::Max(workSize, tplEnd);
            } else {
                expandGatherOn_ = false; // UB 预算不足, 回退标量路径
            }
        }
#else
        expandGatherOn_ = false;
#endif
        pipe->InitBuffer(workBuf_, workSize * sizeof(uint32_t));
        workLocal_ = workBuf_.Get<int32_t>();
        // 向量路径的展开模板经 CreateVecIndex 一次性构建常驻 workLocal_ 头部,
        // 替代每行标量重建的逐行开销(与消费方同 pipe, 常驻安全)
        if (poolSize_ % 8 == 0) {
            AscendC::CreateVecIndex(workLocal_, static_cast<int32_t>(0), poolSize_);
        }
#if PKI_EXPAND_VEC
        else if (expandGatherOn_) {
            BuildExpandGatherTpl();
        }
#endif
    }

    tmpUb_ = tmpBuf_.Get<float>();

    tmpUb_ = tmpBuf_.Get<float>();
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    SortedBasicBlock_ = globalTopkUb_[virTopK * 2 * 2];
    globalTopkNum_ = 0;

    // 基本块执行前初始化UB和GM
    // step1. 初始化一个有序索引 0 - s2BaseSize_
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // step2. globalTopkUb_ [CeilDiv(s1BaseSize_, 2), BASE_TOPK, 2]   -inf,-1
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);

    // step3. 初始化 vec1ParamGm, LD 标志位设为 -1(needFd=-1), 布局 [aic, 2, s1BaseSize_, 16]
    LocalTensor<float> tmpBuff = outQueue_.AllocTensor<float>();
    Duplicate(tmpBuff.template ReinterpretCast<int32_t>(), -1, 2 * (s1BaseSize_ / 2) * paramNum_ * 2);
    outQueue_.EnQue<float>(tmpBuff);
    tmpBuff = outQueue_.DeQue<float>();
    int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +      // 2个AIV共同地址偏移
                           (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_; // 每个AIV的地址偏移，S1方向
    DataCopyPad(vec1ParamGm[wsInfoOffset], tmpBuff.template ReinterpretCast<int64_t>(),
                {1, static_cast<uint16_t>((s1BaseSize_ / 2) * 2 * paramNum_ * sizeof(int64_t)), 0, 0});
    outQueue_.FreeTensor(tmpBuff);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    // Reset 后按 LD 布局重分配缓冲(地址与主集不同): 先排空主集在途的
    // MTE3 读与 V 写, 再重建展开模板, 避免与模板 S 写竞态
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    VToSSync();
    pipe->InitBuffer(ldToBeMrgBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float)); // 2：value + index
    pipe->InitBuffer(ldTmpBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float));     // 2：value + index
    pipe->InitBuffer(ldOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
    if (poolSize_ > 1) {
        uint32_t expandOutLen = PkiCommon::Align(static_cast<uint64_t>(EXPAND_CHUNK), (uint64_t)8);
        pipe->InitBuffer(expandOutBuf_, expandOutLen * sizeof(uint32_t));
        expandOutLocal_ = expandOutBuf_.Get<int32_t>();
        uint32_t workSize = PkiCommon::Align(static_cast<uint64_t>(poolSize_ + 64), (uint64_t)8) +
                            PkiCommon::Align(static_cast<uint64_t>(constInfo_.sparseCount + 64), (uint64_t)8);
#if PKI_EXPAND_VEC
        // 与 InitBuffers 同式重算模板尺寸(expandGatherOn_ 已在 InitBuffers 门禁置位)
        if (expandGatherOn_) {
            uint32_t tplEnd = ExpandGatherTplOffset() + EXPAND_GATHER_CHUNK * 2;
            workSize = PkiCommon::Max(workSize, tplEnd);
        }
#endif
        pipe->InitBuffer(workBuf_, workSize * sizeof(uint32_t));
        workLocal_ = workBuf_.Get<int32_t>();
        // Reset 后重配的 LD 阶段 workLocal_ 需重建展开模板(见 InitBuffers)
        if (poolSize_ % 8 == 0) {
            AscendC::CreateVecIndex(workLocal_, static_cast<int32_t>(0), poolSize_);
        }
#if PKI_EXPAND_VEC
        else if (expandGatherOn_) {
            BuildExpandGatherTpl();
        }
#endif
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitParams(
    const struct PkiCommon::ConstInfo &constInfo, const PoolKeyIndexerTilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;

    groupInner_ = 16;

    blockId_ = GetBlockIdx();
    poolSize_ = static_cast<uint32_t>(constInfo.poolSize);
    outputLen_ = constInfo.sparseCount * poolSize_ + poolSize_ - 1;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm, GlobalTensor<int64_t> vec1ParamGm,
    GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm, GlobalTensor<float> valueOutGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

#if PKI_EXPAND_VEC
// 一次性构建 vgather 展开模板常驻 workLocal_ 尾部只读: qOff[i]=(i/ps)*4,
// pTpl[i]=i%ps; S pipe 标量写, SToVSync 保证后续 V pipe 读可见
template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::BuildExpandGatherTpl()
{
    uint32_t ps = poolSize_;
    uint32_t tplOff = ExpandGatherTplOffset();
    LocalTensor<int32_t> qOffI32 = workLocal_[tplOff];
    LocalTensor<int32_t> pTplI32 = workLocal_[tplOff + EXPAND_GATHER_CHUNK];
    for (uint32_t i = 0; i < EXPAND_GATHER_CHUNK; i++) {
        qOffI32.SetValue(i, static_cast<int32_t>((i / ps) * sizeof(int32_t)));
        pTplI32.SetValue(i, static_cast<int32_t>(i % ps));
    }
    SToVSync(); // 标量写(S pipe) -> 后续 V pipe Gather/Muls/Add 读可见
}
#endif

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::ExpandAndAppendIndices(
    LocalTensor<int32_t> poolIndices, LocalTensor<int32_t> &tokenIndices, LocalTensor<int32_t> &workBuf,
    GlobalTensor<int32_t> &idxOutGm, int64_t idxOutBase, uint32_t sparseCount, uint32_t poolSize, uint32_t validS2Len,
    int32_t poolTailK, int32_t L_orig, uint32_t curS1Idx, uint32_t curS1Size)
{
    uint32_t topk = sparseCount * poolSize;
    uint32_t totalOut = topk + poolSize - 1;

    // 尾块可见 token 数
    int32_t visibleTailK = 0;
    if (poolTailK > 0) {
        if (constInfo_.maskMode == 0) {
            visibleTailK = poolTailK;
        } else {
            int32_t globalPosQ = L_orig - static_cast<int32_t>(curS1Size) + static_cast<int32_t>(curS1Idx);
            visibleTailK = PkiCommon::Max(0, PkiCommon::Min(poolTailK, globalPosQ - static_cast<int32_t>(topk) + 1));
        }
    }

    uint32_t expandRounds = (validS2Len > 0) ? PkiCommon::Min(validS2Len, sparseCount) : 0;
    bool vecPath = (poolSize % 8 == 0);
#if PKI_EXPAND_VEC
    bool gatherPath = !vecPath && expandGatherOn_; // ps%8!=0 时走 vgather 向量展开
#else
    bool gatherPath = false;
#endif
    uint32_t alignedPoolSize = PkiCommon::Align(poolSize, (uint32_t)8);

    // 展开模板已一次性构建常驻 workBuf 头部, 无需逐行标量重建
    LocalTensor<int32_t> offsetTpl = workBuf;
    // vgather 展开模板 qOff/pTpl 常驻 workLocal_ 尾部(见 BuildExpandGatherTpl);
    // 声明置于 #if 外保证宏关时仍可编译(gatherPath 恒 false 不进入)
    LocalTensor<uint32_t> qOffTpl;
    LocalTensor<int32_t> pTpl;
#if PKI_EXPAND_VEC
    if (gatherPath) {
        uint32_t tplOff = ExpandGatherTplOffset();
        qOffTpl = workBuf[tplOff].template ReinterpretCast<uint32_t>();
        pTpl = workBuf[tplOff + EXPAND_GATHER_CHUNK];
    }
#endif
    if (expandRounds > 0 && !gatherPath) {
        VToSSync(); // poolIndices 向量写 → 后续标量 GetValue 可见(gatherPath 无 S 侧读)
    }

    // 分段生成写出: expandOutBuf_ 固定 4KB, 不随 outputLen 增长越 UB 上限;
    // 段边界对齐到池粒度(向量路径保持 32B 对齐)
    uint32_t segChunk = gatherPath ? EXPAND_GATHER_CHUNK : EXPAND_CHUNK;
    uint32_t poolsPerSeg = PkiCommon::Max((uint32_t)1, segChunk / poolSize);

    // ---- 展开区 [0, topk): 按池对齐分段 ----
    uint32_t segDone = 0;
    bool outStarted = false;
    while (segDone < topk) {
        uint32_t kLen = PkiCommon::Min(poolsPerSeg, (topk - segDone) / poolSize);
        uint32_t segLen = kLen * poolSize; // 池对齐, <= EXPAND_CHUNK

        if (outStarted) {
            MTE3ToVSync(); // 上一段 DataCopyPad 读完 tokenIndices 方可覆写
        }
        outStarted = true;
        Duplicate<int32_t>(tokenIndices, -1, segLen);
        PipeBarrier<PIPE_V>();
        if (!vecPath && !gatherPath) {
            // V→S: 标量 SetValue 须等本段 -1 填充(V 写)落地, 否则会被迟到的 V 写覆盖
            VToSSync();
        }

        uint32_t kLo = segDone / poolSize;
        uint32_t kHi = PkiCommon::Min(kLo + kLen, expandRounds);
        if (gatherPath) {
            // vgather 向量展开(全 V pipe): tokenIndices[i] = poolIndices[kLo+i/ps]*ps + i%ps;
            // count 模式下 [validLen,segLen) 不写, kHi<kLo 时 validLen 归零防下溢
            uint32_t validLen = (kHi > kLo) ? (kHi - kLo) * poolSize : 0;
            if (validLen > 0) {
                Gather(tokenIndices, poolIndices, qOffTpl, static_cast<uint32_t>(kLo * sizeof(int32_t)), validLen);
                PipeBarrier<PIPE_V>(); // Gather 写 → Muls 原地读
                Muls(tokenIndices, tokenIndices, static_cast<int32_t>(poolSize), validLen);
                PipeBarrier<PIPE_V>(); // Muls 写 → Add 原地读
                Add<int32_t>(tokenIndices, tokenIndices, pTpl, validLen);
            }
        } else {
            for (uint32_t k = kLo; k < kHi; k++) {
                int32_t base = poolIndices.GetValue(k) * static_cast<int32_t>(poolSize);
                uint32_t off = (k - kLo) * poolSize;
                if (vecPath) {
                    // 向量展开: Duplicate(广播 base) + Add(加模板), 全 SIMD
                    Duplicate<int32_t>(tokenIndices[off], base, alignedPoolSize);
                    PipeBarrier<PIPE_V>();
                    Add<int32_t>(tokenIndices[off], tokenIndices[off], offsetTpl, alignedPoolSize);
                } else {
                    // 非 8 倍数 poolSize: 无法用 32B 对齐向量写, 逐 token 标量写
                    for (uint32_t p = 0; p < poolSize; p++) {
                        tokenIndices.SetValue(off + p, base + static_cast<int32_t>(p));
                    }
                }
            }
        }

        // V 写(Duplicate -1 / 向量展开)对 MTE3(DataCopyPad) 可见; 标量路径另需 S→MTE3
        VToMTE3Sync();
        if (!vecPath && !gatherPath) {
            SToMTE3Sync();
        }
        DataCopyPad(idxOutGm[idxOutBase + segDone], tokenIndices,
                    {1, static_cast<uint16_t>(segLen * sizeof(int32_t)), 0, 0});
        segDone += segLen;
    }

    // ---- 尾块区 [topk, totalOut): 长度 poolSize-1, 单独处理 ----
    uint32_t tailLen = totalOut - topk; // = poolSize - 1
    if (tailLen > 0) {
        if (outStarted) {
            MTE3ToVSync();
        }
        Duplicate<int32_t>(tokenIndices, -1, PkiCommon::Align(tailLen, (uint32_t)8));
        PipeBarrier<PIPE_V>();
        if (visibleTailK > 0) {
            // V→S: -1 填充先落地, 再标量精确写 visibleTailK 个尾 token
            VToSSync();
            for (int32_t t = 0; t < visibleTailK; t++) {
                tokenIndices.SetValue(t, L_orig - poolTailK + t);
            }
            SToMTE3Sync();
        }
        VToMTE3Sync();
        DataCopyPad(idxOutGm[idxOutBase + topk], tokenIndices,
                    {1, static_cast<uint16_t>(tailLen * sizeof(int32_t)), 0, 0});
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    uint32_t idxOutLen = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
    if (poolSize_ > 1) {
        // 分段写 -1: 固定 EXPAND_CHUNK 分段 Duplicate -1 + DataCopyPad 写 GM,
        // 避免按整行 idxOutLen 铺满 expandOutLocal_ 越界
        uint32_t segDone = 0;
        while (segDone < idxOutLen) {
            uint32_t segLen = PkiCommon::Min(EXPAND_CHUNK, idxOutLen - segDone);
            if (segDone > 0) {
                MTE3ToVSync();
            }
            Duplicate(expandOutLocal_, constInfo_.INVALID_IDX, PkiCommon::Align(segLen, (uint32_t)8));
            // V(Duplicate) 写 expandOutLocal_ 后由 MTE3(DataCopyPad) 搬出,
            // 必须真跨流水同步(仅 PipeBarrier<PIPE_V> 不够, 与主输出路径一致)
            VToMTE3Sync();
            DataCopyPad(indiceOutGm[invalidS1offset + segDone], expandOutLocal_,
                        {1, static_cast<uint16_t>(segLen * sizeof(int32_t)), 0, 0});
            segDone += segLen;
        }
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    } else {
        LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
        LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
        Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
        outQueue_.EnQue<float>(valueULocal);
        valueULocal = outQueue_.DeQue<float>();
        PkiServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
        outQueue_.FreeTensor(valueULocal);
    }

    if (constInfo_.returnValue) {
        uint32_t negInf = constInfo_.INVALID_VAL;
        LocalTensor<uint32_t> valueULocal = outQueue_.AllocTensor<uint32_t>();
        Duplicate(valueULocal, negInf, constInfo_.sparseCount);
        outQueue_.EnQue<uint32_t>(valueULocal);
        valueULocal = outQueue_.DeQue<uint32_t>();
        GlobalTensor<uint32_t> valueOutGmTmp;
        valueOutGmTmp.SetGlobalBuffer((__gm__ uint32_t *)valueOutGm.GetPhyAddr());
        // invalidS1offset 是 indices 行偏移(行宽 outputLen_/sparseCount);
        // value 行宽为 sparseCount, 需换算行号后重算偏移, 否则越界写且本行 value 漏写
        uint64_t idxStride = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
        uint64_t valueOffset = (invalidS1offset / idxStride) * constInfo_.sparseCount;
        PkiServiceVec::CopyOut(valueOutGmTmp[valueOffset], valueULocal, constInfo_.sparseCount);
        outQueue_.FreeTensor(valueULocal);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::ProcessVec(const PkiCommon::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 512*512 + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * (constInfo_.mBaseSizeAlign * s2BaseSize_);
    // (B,S1,N1,1);(T,N1,1) -> (B,S1,N2,G,1) 当前只切分到S1轴
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    PipeBarrier<PIPE_V>();
    // cuS1BeginIdxPerAiv: 每个AIV的S1起始偏移
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // cuS1ProcNumPerAiv: 每个AIv的S1计算量
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    // 基本块基地址偏移奇数核加一个S1地址偏移
    weightGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    // cut G
    int32_t outerG = CeilDiv(gSize_, groupInner_);

    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
        blockS2StartIdx_ = 0;
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq
    int32_t cuRealAcSeq = info.actS2Size;
    int32_t cuRealAcSeqCount = 0;
    if (constInfo_.attenMaskFlag) {
        // attenMask true场景
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + cuS1BeginIdxPerAiv;
    }
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    int32_t cuRealAcSeqIni = cuRealAcSeq;
    // LD输出S1方向偏移，保证2个Vector输出的内容连续
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeqCount += 1;
            cuRealAcSeq = (cuRealAcSeqCount + cuRealAcSeqIni) / static_cast<int32_t>(constInfo_.poolSize);
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            int32_t mmUbStride = (cuS2LenVecAlign - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
            LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
            PipeBarrier<PIPE_V>();
            LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
            if (constInfo_.isSparseCountOver2K) {
                WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }
            for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
                int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;

                int32_t pingpong = outerGidx % 2;
                LocalTensor<float> dbTmpUb = tmpUb_[pingpong * (groupInner_ * s2BaseSize_ + s2BaseSize_)];
                LocalTensor<float> weightsInUb = dbTmpUb[procGnum * s2BaseSize_];
                WaitFlag<HardEvent::V_MTE2>(pingpong);
                LocalTensor<W_T> weightsInTUb = weightsInUb.template ReinterpretCast<W_T>();
                if constexpr (!IsSameType<W_T, float>::value) {
                    weightsInTUb = weightsInTUb[groupInner_];
                }
                int64_t mmGmAllOffet = mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                       outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign;
                int64_t weightGmAllOffset = weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_;

                PkiServiceVec::CopyIn(dbTmpUb, weightsInTUb, mm1ResGm, weightsGm, mmGmAllOffet, weightGmAllOffset,
                                      procGnum, info.actualSingleProcessSInnerSizeAlign, mmUbStride);

                SetFlag<HardEvent::MTE2_V>(pingpong);
                WaitFlag<HardEvent::MTE2_V>(pingpong);
                PkiServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], dbTmpUb, weightsInUb, weightsInTUb,
                                       brcBuf, procGnum, s2BaseSize_, outerGidx);
                // confused reduceOp in DoScale
                // neednot use PkiServiceVec::doReduce(mmInUb, reduceOutInner, procGnum, (s2BaseSize_+8));
                SetFlag<HardEvent::V_MTE2>(pingpong);
            }

            int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            PkiServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, gRedCnt, s2BaseSize_);
            outQueue_.FreeTensor(reduceCacheBuf);

            LocalTensor<float> sortScoreUb = reduceOutBuff;
            LocalTensor<float> sortIndiceUb = reduceOutBuff[cuS2LenVecAlign];
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), PkiServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            // 池级分数乘 1/sqrt(headDim)(文档公式 S = Q@K^T/sqrt(headDim));
            // 缩放为正数, 与 ReLU/加权求和可交换, 在聚合后统一应用
            Muls(sortScoreUb, reduceOutInner, constInfo_.qkScale, cuS2Len);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            // 无效数据索引填充为-1
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
            }
            PipeBarrier<PIPE_V>();
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K) {
                // info.actS1Size > 4 则单个vector核内处理的 s1>2，缓存方案无法处理
                PkiServiceVec::SortAll(reduceOutBuff, tmpSortBuf,
                                       cuS2LenVecAlign); //  cuS2LenVecAlign <= s2BaseSize_, fill -inf
                PipeBarrier<PIPE_V>();
                LocalTensor<float> UbTmpSort = constInfo_.isSparseCountOver2K ? tmpUb_ : tmpSortBuf;
                PkiServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff,
                                         cuS2LenVecAlign, UbTmpSort);
            } else {
                int64_t globalTopkUbCacheIdx = (info.s2Idx - blockS2StartIdx_) % 4;
                Sort<float, true>(
                    SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2 + globalTopkUbCacheIdx * s2BaseSize_ * 2],
                    reduceOutBuff, sortIndiceUbInt.template ReinterpretCast<uint32_t>(), tmpSortBuf,
                    cuS2LenVecAlign / 32);
                AscendC::PipeBarrier<PIPE_V>();
                // 缓存4块512或者S2结束, 需要进行精排
                if (globalTopkUbCacheIdx == 3 || isS2End || info.isAllLoopEnd) {
                    LocalTensor<float> tt = SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2];
                    // 前4块直接精排覆盖到globalTopkUb_
                    if (info.s2Idx - blockS2StartIdx_ < 4) {
                        MrgBasicBlock(globalTopkUb_[innerS1Idx * BASE_TOPK * 2], tt,
                                      static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                    } else {
                        if (globalTopkUbCacheIdx > 0) {
                            MrgBasicBlock(tmpSortBuf, tt, static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                            PipeBarrier<PIPE_V>();
                            DataCopy(SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf,
                                     (globalTopkUbCacheIdx + 1) * s2BaseSize_ * 2);
                        }
                        PipeBarrier<PIPE_V>();
                        SparseTopK(globalTopkUb_[innerS1Idx * BASE_TOPK * 2],
                                   SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf, BASE_TOPK,
                                   s2BaseSize_ * (globalTopkUbCacheIdx + 1));
                    }
                }
            }
            if (constInfo_.isSparseCountOver2K) {
                SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }

            PipeBarrier<PIPE_V>();
            outQueue_.FreeTensor(tmpSortBuf);

            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;

            // 中间结果保存
            bool needCopyWsGm = info.isAllLoopEnd || isS2End;

            if (needCopyOutGm) {
                int64_t offset = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? virTopK : constInfo_.sparseCount / 2;
                int64_t copyLen =
                    (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? constInfo_.sparseCount : constInfo_.sparseCount / 2;
                int64_t copyNum = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? 1 : 2;
                if (poolSize_ > 1) {
                    uint32_t alignedPoolSize = PkiCommon::Align(poolSize_, (uint32_t)8);
                    for (int64_t i = 0; i < copyNum; i++) {
                        LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                        LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                        Extract(outValueUb, outIdxUb, globalTopkUb_[innerS1Idx * virTopK * 2 + 2 * i * offset],
                                (offset / 32));

                        PipeBarrier<PIPE_V>();
                        LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                        DataCopy(workLocal_[alignedPoolSize + i * copyLen], idxULocal1,
                                 PkiCommon::Align(copyLen, (int64_t)8));

                        if (constInfo_.returnValue) {
                            outQueue_.EnQue<float>(outValueUb);
                            outValueUb = outQueue_.DeQue<float>();
                            PkiServiceVec::CopyOut(
                                valueOutGm[info.valueOutOffset + cuS1Idx * constInfo_.sparseCount + i * offset],
                                outValueUb, copyLen);
                        }
                        outQueue_.FreeTensor(outValueUb);
                    }
                    PipeBarrier<PIPE_V>();
                    // MTE3→V: 等上一行 DataCopyPad 读完 expandOutLocal_ 再覆写
                    // (行尾 SetWaitFlag 仅登记不等待, 此处显式等待)
                    event_t evtMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                    SetFlag<HardEvent::MTE3_V>(evtMte3V);
                    WaitFlag<HardEvent::MTE3_V>(evtMte3V);
                    // cuRealAcSeq 为有效池数上界, 不可回退 sparseCount(实际池数
                    // 不足时 TopK 尾部 -1 标记会展开成负索引残留)
                    uint32_t validS2Len = static_cast<uint32_t>(cuRealAcSeq);
                    ExpandAndAppendIndices(workLocal_[alignedPoolSize], expandOutLocal_, workLocal_, indiceOutGm,
                                           info.indiceOutOffset + cuS1Idx * outputLen_, constInfo_.sparseCount,
                                           poolSize_, validS2Len, info.poolTailK,
                                           static_cast<int32_t>(info.actS2SizeOrig), static_cast<uint32_t>(cuS1Idx),
                                           info.actS1Size);
                    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                } else {
                    for (int64_t i = 0; i < copyNum; i++) {
                        LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                        LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                        Extract(outValueUb, outIdxUb, globalTopkUb_[innerS1Idx * virTopK * 2 + 2 * i * offset],
                                (offset / 32));

                        if (constInfo_.returnValue) {
                            PipeBarrier<PIPE_V>();
                        }

                        LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                        outQueue_.EnQue<float>(outValueUb);
                        outValueUb = outQueue_.DeQue<float>();

                        PkiServiceVec::CopyOut(
                            indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount + i * offset],
                            idxULocal1, copyLen);
                        if (constInfo_.returnValue) {
                            PkiServiceVec::CopyOut(
                                valueOutGm[info.valueOutOffset + cuS1Idx * constInfo_.sparseCount + i * offset],
                                outValueUb, copyLen);
                        }
                        outQueue_.FreeTensor(outValueUb);
                    }
                }
            } else if (needCopyWsGm) {
                // vec1Res Gm = [aic, s1BaseSize_, 2, 2, topkOut_] float32;
                // vec1Param Gm = [aic, s1BaseSize_, 2, 16] int64(LD 参数)

                int64_t wsOffset = (blockId_ / 2) * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                                   (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * 2 * BASE_TOPK + // 每个AIV的地址偏移，S1方向
                                   (ldS1Offset + innerS1Idx) * 2 * 2 * BASE_TOPK;
                int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ + // 2个AIV共同地址偏移
                                       (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_ + // 每个AIV的地址偏移，S1方向
                                       (ldS1Offset + innerS1Idx) * 2 * paramNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                tmpiBuff.SetValue(0, static_cast<int64_t>(1));
                tmpiBuff.SetValue(1, static_cast<int64_t>(cuRealAcSeq));
                tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
                tmpiBuff.SetValue(3, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(4, static_cast<int64_t>(isS2End));
                tmpiBuff.SetValue(5, static_cast<int64_t>(info.bN2Idx));
                tmpiBuff.SetValue(6, static_cast<int64_t>(cuS1Idx));
                tmpiBuff.SetValue(7, static_cast<int64_t>(cuS1ProcNum));
                uint64_t idxStride = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
                tmpiBuff.SetValue(8, static_cast<int64_t>(info.indiceOutOffset + cuS1Idx * idxStride));
                tmpiBuff.SetValue(9, static_cast<int64_t>(info.poolTailK));
                tmpiBuff.SetValue(10, static_cast<int64_t>(info.actS2SizeOrig));
                tmpiBuff.SetValue(11, static_cast<int64_t>(info.actS1Size));
                // slot 12: values 输出偏移(行宽 sparseCount, 与 slot 8 的 indices
                // 偏移行宽不同, poolSize>1 时不可复用 slot 8)
                tmpiBuff.SetValue(12, static_cast<int64_t>(info.valueOutOffset + cuS1Idx * constInfo_.sparseCount));
                // 写入头尾判断: head 与前面块规约, tail 与后面块规约
                bool isTailReduce = blockS2StartIdx_ == 0; // 一定是isLastTile
                // WS偏移规则: 与前面块规约写 0 偏移; 与后面块规约写 1 偏移
                if (isTailReduce) { // S2不是最后结束的数据就需要往后做规约，放入第二块ws
                    wsInfoOffset += paramNum_;
                    wsOffset += 2 * BASE_TOPK;
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                PkiServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                PkiServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * BASE_TOPK * 2], 2 * BASE_TOPK);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        } else if (cuRealAcSeq <= 0) {
            uint64_t idxStride = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * idxStride);
        }
    }

    // BNSD场景无效S1 输出-1
    if (LAYOUT_T == PkiLayout::BSND) {
        // 最后一个S1的基本块, 需要 >= info.actS1Size
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        // blockS2StartIdx_ == 0 控制S2从开始的核去做冗余清理
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            uint64_t idxStride = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * idxStride);
            }
        }

        int32_t invalidS1Num2 =
            static_cast<int32_t>(info.actS1Size - info.actS2SizeOrig) / static_cast<int32_t>(constInfo_.poolSize);
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            uint64_t idxStride = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) * idxStride);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        // S2最后一个Loop后, 下一个基本块初始从0开始
        blockS2StartIdx_ = 0;
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::ProcessLD()
{
    int32_t curCubeId = blockId_ / 2;
    int32_t tmpCubeId = curCubeId;

    int64_t s2ActSeq = 0;
    int64_t s2Start;
    int64_t s2End;
    int64_t isS2End;
    int64_t bn2Idx;
    int64_t s1Idx;
    int64_t poolTailKLd = 0;
    int64_t L_origLd = 0;
    int64_t actS1SizeLd = 0;
    uint32_t acc_list_num = 0;
    int64_t bIdx = 0;
    int64_t needFd;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    int64_t nextneedFd;
    int64_t valueOffset = 0;
    int64_t outOffset = 0;
    int64_t valueOutOffsetLd = 0; // slot 12: values 输出偏移(行宽 sparseCount)

    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    // S2开头信息: 从尾规约开始, while 读取后续核头规约, 存满 4 个 list 或
    // S2 结尾则 merge; 每个核忽略自己的头规约(由前面核完成)
    uint32_t s1LdStartIdx = 0;
    uint32_t s1ProcNum = 0;
    uint64_t paramGmCoreOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_;
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        needFd = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_);
        if (needFd == 1) {
            s1LdStartIdx = (s1ProcNum == 0) ? innerS1Idx : s1LdStartIdx;
            s1ProcNum++;
        }
    }

    if (s1ProcNum == 0) {
        return;
    }

    // S1逐行计算
    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) {
        s1LdStartIdx = s1LdStartIdx + s1VecNum;
        s1VecNum = s1ProcNum - s1VecNum;
    }
    for (uint32_t innerS1Idx = s1LdStartIdx; innerS1Idx < s1LdStartIdx + s1VecNum; innerS1Idx++) {
        // 重置偏移
        tmpCubeId = curCubeId;
        acc_list_num = 0;
        valueOffset = 0;

        // 搬入数据
        wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                   innerS1Idx * 2 * 2 * BASE_TOPK + 2 * BASE_TOPK;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(curValueIdxUb, vec1ResGm[wsOffset],
                    {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
        acc_list_num++;
        valueOffset += 2 * BASE_TOPK;

        // 获取下一个核规约信息
        tmpCubeId++;
        wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
        needFd = vec1ParamGm.GetValue(wsInfoOffset);
        isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        s1Idx = vec1ParamGm.GetValue(wsInfoOffset + 6);
        outOffset = vec1ParamGm.GetValue(wsInfoOffset + 8);
        s2ActSeq = vec1ParamGm.GetValue(wsInfoOffset + 1);
        poolTailKLd = vec1ParamGm.GetValue(wsInfoOffset + 9);
        L_origLd = vec1ParamGm.GetValue(wsInfoOffset + 10);
        actS1SizeLd = vec1ParamGm.GetValue(wsInfoOffset + 11);
        valueOutOffsetLd = vec1ParamGm.GetValue(wsInfoOffset + 12);

        while (needFd == 1) {
            // 搬入头规约数据
            wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                       innerS1Idx * 2 * 2 * BASE_TOPK;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset],
                        {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
            valueOffset += 2 * BASE_TOPK;
            acc_list_num++;

            // 每满4个list，聚合  前2K为mrg结果
            if (acc_list_num == mrgListNum_) {
                // MrgSort 四条2048的队列，Mrg成一条
                AscendC::MrgSort4Info params;
                params.elementLengths[0] = BASE_TOPK;
                params.elementLengths[1] = BASE_TOPK;
                params.elementLengths[2] = BASE_TOPK;
                params.elementLengths[3] = BASE_TOPK;
                params.ifExhaustedSuspension = true;
                params.validBit = 0b1111;
                params.repeatTimes = 1;

                AscendC::MrgSortSrcList<float> srcList;
                srcList.src1 = curValueIdxUb[0];
                srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
                srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
                srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                MrgSort(tmpUb, srcList, params);
                PipeBarrier<PIPE_V>();
                DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
                PipeBarrier<PIPE_V>();
                acc_list_num = 1;
                valueOffset = 2 * BASE_TOPK;
            }

            // reduce到S2末尾，则跳出
            if (isS2End == 1) {
                break;
            }

            tmpCubeId++;
            wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
            needFd = vec1ParamGm.GetValue(wsInfoOffset);
            isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        }

        // mrg不足4个list的数据
        if (acc_list_num != 1) {
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = BASE_TOPK;
            params.elementLengths[1] = BASE_TOPK;
            params.elementLengths[2] = BASE_TOPK;
            params.elementLengths[3] = BASE_TOPK;
            params.ifExhaustedSuspension = true;
            if (acc_list_num == 2) {
                params.validBit = 0b0011;
            } else if (acc_list_num == 3) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
            srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
            srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
            PipeBarrier<PIPE_V>();
        }

        // 搬出
        LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
        LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();
        uint16_t idxOutBytes =
            static_cast<uint16_t>((poolSize_ > 1 ? outputLen_ : constInfo_.sparseCount) * sizeof(int32_t));
        if (!constInfo_.returnValue) {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
            if (poolSize_ > 1) {
                PipeBarrier<PIPE_V>();
                // MTE3→V: 等上一行 DataCopyPad 读完 expandOutLocal_ 再覆写
                // (与 ProcessVec 主输出路径相同)
                event_t evtMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(evtMte3V);
                WaitFlag<HardEvent::MTE3_V>(evtMte3V);
                // s2ActSeq 为有效池数上界, 不可回退 sparseCount
                // (否则 -1 无效池标记展开成负索引残留)
                uint32_t validS2Len = static_cast<uint32_t>(s2ActSeq);
                ExpandAndAppendIndices(outIdxUb.template ReinterpretCast<int32_t>(), expandOutLocal_, workLocal_,
                                       indiceOutGm, outOffset, constInfo_.sparseCount, poolSize_, validS2Len,
                                       static_cast<int32_t>(poolTailKLd), static_cast<int32_t>(L_origLd),
                                       static_cast<uint32_t>(s1Idx), static_cast<uint32_t>(actS1SizeLd));
                PipeBarrier<PIPE_V>();
            } else {
                LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(indiceOutGm[outOffset], idxULocal1, {1, idxOutBytes, 0, 0});
            }
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        } else {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
            PipeBarrier<PIPE_V>();
            if (poolSize_ > 1) {
                // MTE3→V: 等上一行 DataCopyPad 读完 expandOutLocal_ 再覆写
                // (与 ProcessVec 主输出路径相同)
                event_t evtMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(evtMte3V);
                WaitFlag<HardEvent::MTE3_V>(evtMte3V);
                uint32_t validS2Len = static_cast<uint32_t>(s2ActSeq);
                ExpandAndAppendIndices(outIdxUb.template ReinterpretCast<int32_t>(), expandOutLocal_, workLocal_,
                                       indiceOutGm, outOffset, constInfo_.sparseCount, poolSize_, validS2Len,
                                       static_cast<int32_t>(poolTailKLd), static_cast<int32_t>(L_origLd),
                                       static_cast<uint32_t>(s1Idx), static_cast<uint32_t>(actS1SizeLd));
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
            if (poolSize_ > 1) {
                // ExpandAndAppendIndices 已内部分段写 GM, 此处无需再写 indices
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            } else {
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(indiceOutGm[outOffset], idxULocal1, {1, idxOutBytes, 0, 0});
            }
            DataCopyPad(valueOutGm[valueOutOffsetLd], outValueUb,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(float)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}
} // namespace PkiKernel
#endif
