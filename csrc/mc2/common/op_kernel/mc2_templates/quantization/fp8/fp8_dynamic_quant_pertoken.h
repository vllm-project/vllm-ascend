/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fp8_dynamic_quant_pertoken.h
 * \brief
 */

#ifndef FP8_DYNAMIC_QUANT_PERTOKEN_H
#define FP8_DYNAMIC_QUANT_PERTOKEN_H

namespace MC2KernelTemplate {
struct MC2PertokenDQuantContext {
    GM_ADDR quantInputAddr;
    GM_ADDR quantOutputAddr;
    GM_ADDR transposeDstAddr;
    GM_ADDR quantOutputScaleAddr;
    uint64_t rowNum;
    uint64_t colNumPerBlock;
    uint64_t blockNum;
    uint64_t nextBlockDataOffset;
    uint64_t quantInputAddrOffset;
    uint64_t quantOutputAddrOffset;
    uint64_t transposeDstAddrOffset;
    uint64_t quantOutputScaleAddrOffset;
};

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
class Fp8DynamicQuantPertoken {
public:
    __aicore__ inline Fp8DynamicQuantPertoken(TPipe *tPipe) : tPipe_(tPipe){};
    __aicore__ inline MC2PertokenDQuantContext *GetContextPtr();
    __aicore__ inline void Init() {};
    __aicore__ inline void Process(uint32_t taskIndex);

protected:
    static constexpr uint32_t ALIGN_NUM = 8;
    static constexpr uint32_t TWO_FACTOR = 2;
    static constexpr uint32_t ONE_FACTOR = 1;
    static constexpr uint32_t UB_DATABLOCK = 32;
    static constexpr uint32_t COMPARE_ALIGN_LEN = 256;
    static constexpr uint32_t VECTOR_UB_SIZE = AscendC::TOTAL_UB_SIZE;

    static constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
    static constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;

    constexpr static AscendC::MicroAPI::CastTrait castTraitB16ToB32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToI16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTraitI16ToF16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF16ToI8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32tofp8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32toh8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    template <typename T>
    __aicore__ static inline T Max(T a, T b)
    {
        return (a > b) ? (a) : (b);
    }

    template <typename T>
    __aicore__ static inline T Min(T a, T b)
    {
        return (a > b) ? (b) : (a);
    }

    template <typename T>
    __aicore__ static inline T Ceil(T x, T y)
    {
        return (x + y - 1) / y;
    }

    template <AscendC::HardEvent event>
    __aicore__ static inline void SyncFunc()
    {
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(eventID);
        AscendC::WaitFlag<event>(eventID);
    }

    TPipe *tPipe_;
    MC2PertokenDQuantContext context_;

    GlobalTensor<quantInputDataType> quantInputGM_;
    GlobalTensor<quantInputDataType> transOutGM_;
    GlobalTensor<quantOutputDataType> quantOutputGM_;
    GlobalTensor<float> quantOutputScaleGM_;

    TQue<QuePosition::VECIN, 1> rawInputQue_;     // 存放原始 float16 数据 (LargeK使用)
    TQue<QuePosition::VECOUT, 1> quantOutputQue_; // 存放量化后的 fp8 数据 (LargeK使用)
    TQue<QuePosition::VECOUT, 1> transOutQue_;    // 存放转置后的原始 float16 数据 (LargeK使用)

    TBuf<TPosition::VECCALC> rawInputBuf_[TWO_FACTOR];   // 存放原始 float16 数据 (SmallK使用)
    TBuf<TPosition::VECOUT> quantOutputBuf_[TWO_FACTOR]; // 存放量化后的 fp8 数据 (SmallK使用)

    TBuf<TPosition::VECOUT> quantScaleBuf_;
    TBuf<TPosition::VECCALC> maxValueBuf_;

    uint64_t usedCoreAivNum_ = 0;   // 使用的aiv核数
    uint64_t rowsThisCore_ = 0;     // 当前核负责的总行数
    uint64_t startRowThisCore_ = 0; // 当前核负责的起始行索引

    float recipFP8MaxLimit_ = 0.0f;
    float fp8MaxLimit_ = 0.0f;
    bool isTransOut_ = false;

    __aicore__ inline void SetMaxValue();
    __aicore__ inline void ProcessOneTokenRegBase();
    __aicore__ inline void ProcessOneTokenSmallK();
    __aicore__ inline void ProcessOneTokenLargeK();
    __aicore__ inline void CalculateMaxRegBase(__local_mem__ quantInputDataType *xAddr, uint64_t kAxisSize,
                                               __local_mem__ float *maxAddr);
    __aicore__ inline void DoQuantRegBase(__local_mem__ quantInputDataType *xAddr,
                                          __local_mem__ quantOutputDataType *yAddr, uint64_t dataCount, float scale);
    __aicore__ inline void Init(GM_ADDR quantInputAddr, GM_ADDR quantOutputAddr, GM_ADDR quantOutputScaleAddr,
                                GM_ADDR transposeDstAddr);
    __aicore__ inline void Process();
    __aicore__ inline void Destroy();
};

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline MC2PertokenDQuantContext *
Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::GetContextPtr()
{
    return &context_;
}

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline void
Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::Process(uint32_t taskIndex)
{
    GM_ADDR quantInputAddr = context_.quantInputAddr + taskIndex * context_.quantInputAddrOffset;
    GM_ADDR quantOutputAddr = context_.quantOutputAddr + taskIndex * context_.quantOutputAddrOffset;
    GM_ADDR transposeDstAddr = (context_.transposeDstAddr == nullptr) ?
                                   nullptr :
                                   (context_.transposeDstAddr + taskIndex * context_.transposeDstAddrOffset);
    GM_ADDR quantOutputScaleAddr = context_.quantOutputScaleAddr + taskIndex * context_.quantOutputScaleAddrOffset;
    Init(quantInputAddr, quantOutputAddr, quantOutputScaleAddr, transposeDstAddr);
    Process();
    Destroy();
}

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::Init(
    GM_ADDR quantInputAddr, GM_ADDR quantOutputAddr, GM_ADDR quantOutputScaleAddr, GM_ADDR transposeDstAddr)
{
    if ASCEND_IS_AIC {
        return;
    }

    tPipe_->Reset();

    uint64_t totalCores = static_cast<uint64_t>(GetBlockNum() * TWO_FACTOR);
    this->usedCoreAivNum_ = (context_.rowNum < totalCores) ? context_.rowNum : totalCores;
    // 2. 均匀分配任务到各个核
    uint32_t coreIdx = GetBlockIdx();
    uint64_t avgRows = context_.rowNum / this->usedCoreAivNum_;
    uint32_t tailRows = context_.rowNum % this->usedCoreAivNum_;

    // 每个核计算自己的任务范围，如10行数据，使用4个核（0,1,2,3），第1个aiv核处理3,4,5行，startRowThisCore_的起始索引为3
    this->rowsThisCore_ = avgRows + (coreIdx < tailRows ? 1 : 0);
    this->startRowThisCore_ = coreIdx * avgRows + (coreIdx < tailRows ? coreIdx : tailRows);

    SetMaxValue();

    quantInputGM_.SetGlobalBuffer((__gm__ quantInputDataType *)quantInputAddr);
    quantOutputGM_.SetGlobalBuffer((__gm__ quantOutputDataType *)quantOutputAddr);
    quantOutputScaleGM_.SetGlobalBuffer((__gm__ float *)quantOutputScaleAddr);
    if (transposeDstAddr != nullptr) {
        this->isTransOut_ = true;
        transOutGM_.SetGlobalBuffer((__gm__ quantInputDataType *)transposeDstAddr);
    } else {
        this->isTransOut_ = false;
    }
}

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::Process()
{
    if (GetBlockIdx() >= this->usedCoreAivNum_) {
        return;
    }
    ProcessOneTokenRegBase();
}

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::Destroy()
{
    rawInputQue_.FreeAllEvent();
    quantOutputQue_.FreeAllEvent();
    transOutQue_.FreeAllEvent();
}

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::SetMaxValue()
{
    if constexpr (IsSameType<quantOutputDataType, fp8_e5m2_t>::value) {
        this->recipFP8MaxLimit_ = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
        this->fp8MaxLimit_ = FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<quantOutputDataType, fp8_e4m3fn_t>::value) {
        this->recipFP8MaxLimit_ = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
        this->fp8MaxLimit_ = FP8_E4M3FN_MAX_VALUE;
    }
}

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline void
Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::ProcessOneTokenRegBase()
{
    if constexpr (isSmallK) {
        ProcessOneTokenSmallK();
    } else {
        ProcessOneTokenLargeK();
    }
}

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline void
Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::ProcessOneTokenSmallK()
{
    // 空间大小和偏移计算
    uint32_t perBlockDataLength = static_cast<uint32_t>(context_.colNumPerBlock * sizeof(quantInputDataType));
    uint32_t inputSizePerBlock = Ceil(perBlockDataLength, UB_DATABLOCK) * UB_DATABLOCK;
    uint32_t inputSize = inputSizePerBlock * context_.blockNum;
    constexpr uint8_t tempRShiftAmountIn = sizeof(quantInputDataType) / 2;
    uint64_t perBlockDataOffsetInUB = inputSizePerBlock >> tempRShiftAmountIn;

    uint32_t outputSizePerBlock =
        Ceil(static_cast<uint32_t>(context_.colNumPerBlock * sizeof(quantOutputDataType)), UB_DATABLOCK) * UB_DATABLOCK;
    uint32_t outputSize = outputSizePerBlock * context_.blockNum;
    constexpr uint8_t tempRShiftAmountOut = sizeof(quantOutputDataType) / 2;
    uint64_t perBlockQuantDataOffsetInUB = outputSizePerBlock >> tempRShiftAmountOut;

    // 空间分配
    tPipe_->InitBuffer(maxValueBuf_, UB_DATABLOCK);
    tPipe_->InitBuffer(quantScaleBuf_,
                       Ceil(static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), UB_DATABLOCK) * UB_DATABLOCK);
    for (uint32_t i = 0; i < TWO_FACTOR; ++i) {
        tPipe_->InitBuffer(rawInputBuf_[i], inputSize);
        tPipe_->InitBuffer(quantOutputBuf_[i], outputSize);
    }
    LocalTensor<float> maxValueData = maxValueBuf_.Get<float>();
    LocalTensor<float> coreQuantScales = quantScaleBuf_.Get<float>();
    LocalTensor<quantInputDataType> rawInputTensor[2];
    LocalTensor<quantOutputDataType> quantOut[2];
    for (uint32_t i = 0; i < TWO_FACTOR; ++i) {
        rawInputTensor[i] = rawInputBuf_[i].Get<quantInputDataType>();
        quantOut[i] = quantOutputBuf_[i].Get<quantOutputDataType>();
    }

    // 申请event同步数据搬入
    AscendC::TEventID copyInEvent[2];
    copyInEvent[0] = tPipe_->AllocEventID<AscendC::HardEvent::MTE2_S>();
    copyInEvent[1] = tPipe_->AllocEventID<AscendC::HardEvent::MTE2_S>();
    // 申请event同步量化数据搬出
    AscendC::TEventID quantOutEvent[2];
    quantOutEvent[0] = tPipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();
    quantOutEvent[1] = tPipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();
    // 申请event同步permute数据搬出
    AscendC::TEventID transOutEvent[2];
    transOutEvent[0] = tPipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();
    transOutEvent[1] = tPipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();

    // 数据搬入相关参数
    uint32_t gmStrideBytes =
        static_cast<uint32_t>((context_.nextBlockDataOffset - context_.colNumPerBlock) * sizeof(quantInputDataType));
    uint32_t ubStrideBlocks = (inputSizePerBlock - perBlockDataLength) / UB_DATABLOCK;
    DataCopyExtParams inputCopyParams = {static_cast<uint16_t>(context_.blockNum), perBlockDataLength, gmStrideBytes,
                                         ubStrideBlocks, 0};
    // 首轮数据搬入
    uint64_t globalInputIdx0 = this->startRowThisCore_ * context_.colNumPerBlock;
    DataCopyPad<quantInputDataType>(rawInputTensor[0], quantInputGM_[globalInputIdx0], inputCopyParams,
                                    {true, 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(copyInEvent[0]);
    // 保证循环中同步配对
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(transOutEvent[1]);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(quantOutEvent[0]);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(quantOutEvent[1]);

    for (uint64_t r = 0; r < this->rowsThisCore_; ++r) {
        uint64_t globalRowIdx = this->startRowThisCore_ + r;
        uint64_t globalInputIdx = globalRowIdx * context_.colNumPerBlock;
        uint64_t globalOutIdx = globalRowIdx * context_.colNumPerBlock * context_.blockNum;
        uint32_t bufIdx = r % 2;
        uint32_t nextBufIdx = (r + 1) % 2;
        // 提前启动下一轮搬入
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(transOutEvent[nextBufIdx]);
        if (r + 1 < this->rowsThisCore_) {
            uint64_t nextGlobalInputIdx = globalInputIdx + context_.colNumPerBlock;
            DataCopyPad<quantInputDataType>(rawInputTensor[nextBufIdx], quantInputGM_[nextGlobalInputIdx],
                                            inputCopyParams, {true, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(copyInEvent[nextBufIdx]);
        }
        // 启动本轮permute搬出
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(copyInEvent[bufIdx]);
        if (this->isTransOut_) {
            DataCopyPad<quantInputDataType, PaddingMode::Normal>(
                transOutGM_[globalOutIdx], rawInputTensor[bufIdx],
                {static_cast<uint16_t>(context_.blockNum), perBlockDataLength, ubStrideBlocks, 0, 0});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(transOutEvent[bufIdx]);
        // 求最大值
        __local_mem__ quantInputDataType *xAddr =
            (__local_mem__ quantInputDataType *)rawInputTensor[bufIdx].GetPhyAddr();
        __local_mem__ float *maxValueAddr = (__local_mem__ float *)maxValueData.GetPhyAddr();
        uint64_t totalKSize = context_.blockNum * perBlockDataOffsetInUB;
        CalculateMaxRegBase(xAddr, totalKSize, maxValueAddr);
        SyncFunc<AscendC::HardEvent::V_S>();
        float maxValue = maxValueData.GetValue(0);
        // 求量化系数
        float scale = maxValue * this->recipFP8MaxLimit_;
        coreQuantScales.SetValue(r, scale);
        SyncFunc<AscendC::HardEvent::S_V>();
        // 做量化
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(quantOutEvent[bufIdx]);
        __local_mem__ quantOutputDataType *yAddr = (__local_mem__ quantOutputDataType *)quantOut[bufIdx].GetPhyAddr();
        for (uint32_t k = 0; k < context_.blockNum; ++k) {
            DoQuantRegBase(xAddr + k * perBlockDataOffsetInUB, yAddr + k * perBlockQuantDataOffsetInUB,
                           perBlockDataOffsetInUB, scale);
        }
        SyncFunc<AscendC::HardEvent::V_S>();
        // 量化结果搬出
        DataCopyPad<quantOutputDataType, PaddingMode::Normal>(
            quantOutputGM_[globalOutIdx], quantOut[bufIdx],
            {static_cast<uint16_t>(context_.blockNum),
             static_cast<uint32_t>(context_.colNumPerBlock * sizeof(quantOutputDataType)), 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(quantOutEvent[bufIdx]);
    }
    // 搬出scales
    DataCopyExtParams outScaleParams = {1, static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float, PaddingMode::Normal>(quantOutputScaleGM_[this->startRowThisCore_], coreQuantScales,
                                            outScaleParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    // 保证循环中同步配对
    uint64_t lastRow = this->rowsThisCore_ - 1;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(quantOutEvent[0]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(quantOutEvent[1]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(transOutEvent[lastRow % 2]);
    // 释放event
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(copyInEvent[0]);
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(copyInEvent[1]);
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE3_S>(quantOutEvent[0]);
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE3_S>(quantOutEvent[1]);
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE3_S>(transOutEvent[0]);
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE3_S>(transOutEvent[1]);
}

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline void
Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::ProcessOneTokenLargeK()
{
    uint32_t perBlockDataLength = static_cast<uint32_t>(context_.colNumPerBlock * sizeof(quantInputDataType));
    uint32_t inputSizePerBlock = Ceil(perBlockDataLength, UB_DATABLOCK) * UB_DATABLOCK;
    uint32_t inputSize = inputSizePerBlock * context_.blockNum;
    constexpr uint8_t tempRShiftAmountIn = sizeof(quantInputDataType) / 2;
    uint64_t perBlockDataOffsetInUB = inputSizePerBlock >> tempRShiftAmountIn;
    uint32_t outputSizePerBlock =
        Ceil(static_cast<uint32_t>(perBlockDataOffsetInUB * sizeof(quantOutputDataType)), UB_DATABLOCK) * UB_DATABLOCK;
    uint32_t outputSize = perBlockDataOffsetInUB * context_.blockNum;
    constexpr uint8_t tempRShiftAmountOut = sizeof(quantOutputDataType) / 2;
    uint64_t perBlockQuantDataOffsetInUB = outputSizePerBlock >> tempRShiftAmountOut;

    tPipe_->InitBuffer(maxValueBuf_, UB_DATABLOCK);
    tPipe_->InitBuffer(quantScaleBuf_,
                       Ceil(static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), UB_DATABLOCK) * UB_DATABLOCK);
    tPipe_->InitBuffer(rawInputQue_, ONE_FACTOR, inputSize);
    tPipe_->InitBuffer(quantOutputQue_, ONE_FACTOR, outputSize);

    LocalTensor<float> maxValueData = maxValueBuf_.Get<float>();
    LocalTensor<float> coreQuantScales = quantScaleBuf_.Get<float>();
    LocalTensor<quantInputDataType> rawInputTensor = rawInputQue_.AllocTensor<quantInputDataType>();
    LocalTensor<quantOutputDataType> quantOut = quantOutputQue_.AllocTensor<quantOutputDataType>();
    transOutQue_.EnQue(rawInputTensor);
    quantOutputQue_.EnQue(quantOut);

    // 逐token处理
    float scale;
    float maxValue;
    float maxValuePerRank;
    // 申请两个event管理非连续block数据的并行搬入
    AscendC::TEventID rawInputEvent[2];
    rawInputEvent[0] = tPipe_->AllocEventID<AscendC::HardEvent::MTE2_S>();
    rawInputEvent[1] = tPipe_->AllocEventID<AscendC::HardEvent::MTE2_S>();
    for (uint64_t r = 0; r < this->rowsThisCore_; ++r) {
        uint64_t globalRowIdx = this->startRowThisCore_ + r;
        uint64_t globalInputIdx = globalRowIdx * context_.colNumPerBlock;
        transOutQue_.DeQue();
        // 提前启动数据搬运
        DataCopyPad<quantInputDataType>(rawInputTensor, quantInputGM_[globalInputIdx], {1, perBlockDataLength, 0, 0, 0},
                                        {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(rawInputEvent[0]);
        DataCopyPad<quantInputDataType>(rawInputTensor[perBlockDataOffsetInUB],
                                        quantInputGM_[globalInputIdx + context_.nextBlockDataOffset],
                                        {1, perBlockDataLength, 0, 0, 0}, {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(rawInputEvent[1]);

        // 非连续K循环处理
        __local_mem__ quantInputDataType *xAddr = (__local_mem__ quantInputDataType *)rawInputTensor.GetPhyAddr();
        __local_mem__ float *maxValueAddr = (__local_mem__ float *)maxValueData.GetPhyAddr();
        maxValue = 0.0f;
        maxValuePerRank = 0.0f;
        for (uint32_t k = 0; k < context_.blockNum; ++k) {
            // 等待一块K搬入完成
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(rawInputEvent[k % 2]);
            // 提前启动数据搬运
            uint32_t nextK = k + 2;
            if (nextK < context_.blockNum) {
                DataCopyPad<quantInputDataType, PaddingMode::Normal>(
                    rawInputTensor[nextK * perBlockDataOffsetInUB],
                    quantInputGM_[globalInputIdx + nextK * context_.nextBlockDataOffset],
                    {1, perBlockDataLength, 0, 0, 0}, {false, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(rawInputEvent[k % 2]);
            }

            // 计算最大绝对值
            CalculateMaxRegBase(xAddr + k * perBlockDataOffsetInUB, context_.colNumPerBlock, maxValueAddr);
            SyncFunc<AscendC::HardEvent::V_S>();
            maxValuePerRank = maxValueData.GetValue(0);
            maxValue = Max(maxValue, maxValuePerRank);
        }
        // 转置结果搬出
        uint64_t globalOutIdx = globalRowIdx * context_.colNumPerBlock * context_.blockNum;
        if (this->isTransOut_) {
            DataCopyPad<quantInputDataType, PaddingMode::Normal>(
                transOutGM_[globalOutIdx], rawInputTensor,
                {static_cast<uint16_t>(context_.blockNum), perBlockDataLength, 0, 0, 0});
        }
        transOutQue_.EnQue(rawInputTensor);

        // 计算量化系数
        scale = maxValue * this->recipFP8MaxLimit_;
        coreQuantScales.SetValue(r, scale);
        SyncFunc<AscendC::HardEvent::S_V>();
        // 量化
        quantOutputQue_.DeQue();
        __local_mem__ quantOutputDataType *yAddr = (__local_mem__ quantOutputDataType *)quantOut.GetPhyAddr();
        for (uint32_t k = 0; k < context_.blockNum; ++k) {
            DoQuantRegBase(xAddr + k * perBlockDataOffsetInUB, yAddr + k * perBlockQuantDataOffsetInUB,
                           perBlockDataOffsetInUB, scale);
        }
        SyncFunc<AscendC::HardEvent::V_S>();
        // 搬出量化结果
        DataCopyPad<quantOutputDataType, PaddingMode::Normal>(
            quantOutputGM_[globalOutIdx], quantOut,
            {static_cast<uint16_t>(context_.blockNum),
             static_cast<uint32_t>(context_.colNumPerBlock * sizeof(quantOutputDataType)), 0, 0, 0});
        quantOutputQue_.EnQue(quantOut);
    }
    // 搬出量化系数
    DataCopyExtParams outScaleParams = {1, static_cast<uint32_t>(this->rowsThisCore_ * sizeof(float)), 0, 0, 0};
    DataCopyPad<float, PaddingMode::Normal>(quantOutputScaleGM_[this->startRowThisCore_], coreQuantScales,
                                            outScaleParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(rawInputEvent[0]);
    tPipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(rawInputEvent[1]);
    transOutQue_.DeQue();
    quantOutputQue_.DeQue();
    rawInputQue_.FreeTensor(rawInputTensor);
    quantOutputQue_.FreeTensor(quantOut);
}

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::CalculateMaxRegBase(
    __local_mem__ quantInputDataType *xAddr, uint64_t kAxisSize, __local_mem__ float *maxAddr)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<quantInputDataType> vregX;
        AscendC::MicroAPI::RegTensor<float> vregFloatX;
        AscendC::MicroAPI::RegTensor<float> vregAbsX;
        AscendC::MicroAPI::RegTensor<float> vregMaxAbsX;
        AscendC::MicroAPI::RegTensor<float> vregReduceMax;

        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg ureg0;

        AscendC::MicroAPI::Duplicate(vregMaxAbsX, 0.0f, preg1);
        uint32_t dtypeSize = sizeof(float);
        uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        // kAxisSize aligned to 16 for vector processing
        uint32_t colNum = (kAxisSize + 15) / 16 * 16;
        uint16_t vfLoop = (colNum + VL - 1) / VL;
        uint32_t sreg0 = colNum;

        for (uint16_t j = 0; j < vfLoop; j++) {
            preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
            AscendC::MicroAPI::DataCopy<quantInputDataType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vregX, xAddr + j * VL);
            AscendC::MicroAPI::Cast<float, quantInputDataType, castTraitB16ToB32>(vregFloatX, vregX, preg0);
            AscendC::MicroAPI::Abs(vregAbsX, vregFloatX, preg0);
            AscendC::MicroAPI::Max(vregMaxAbsX, vregAbsX, vregMaxAbsX, preg1);
        }
        AscendC::MicroAPI::ReduceMax(vregReduceMax, vregMaxAbsX, preg1);
        AscendC::MicroAPI::DataCopyUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            maxAddr, vregReduceMax, ureg0, 1);
        AscendC::MicroAPI::DataCopyUnAlignPost(maxAddr, ureg0, 0);
    }
}

template <typename quantInputDataType, typename quantOutputDataType, bool isSmallK>
__aicore__ inline void Fp8DynamicQuantPertoken<quantInputDataType, quantOutputDataType, isSmallK>::DoQuantRegBase(
    __local_mem__ quantInputDataType *xAddr, __local_mem__ quantOutputDataType *yAddr, uint64_t dataCount, float scale)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<quantInputDataType> vregX;
        AscendC::MicroAPI::RegTensor<float> vregFloatX;
        AscendC::MicroAPI::RegTensor<float> vregScale;
        AscendC::MicroAPI::RegTensor<float> vregScaledX;
        AscendC::MicroAPI::RegTensor<int16_t> vregYInt16;
        AscendC::MicroAPI::RegTensor<half> vregYFp16;
        AscendC::MicroAPI::RegTensor<quantOutputDataType> vregY;

        AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg preg2;
        AscendC::MicroAPI::MaskReg preg3 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::H>();

        AscendC::MicroAPI::Duplicate(vregScale, scale, preg1);

        uint32_t dtypeSize = sizeof(float);
        uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize; // 每个向量寄存器能存的元素数
        // colNum对齐到16
        uint32_t colNum = (dataCount + 15) / 16 * 16;
        uint16_t vfLoop = (colNum + VL - 1) / VL;
        uint32_t sreg1 = colNum;
        for (uint16_t j = 0; j < vfLoop; j++) {
            auto addr = yAddr + j * VL;
            preg2 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
            // 1. 重新加载数据（或可以重用之前加载的）
            AscendC::MicroAPI::DataCopy<quantInputDataType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vregX, xAddr + j * VL);
            // 2. 转换为float并应用平滑缩放
            AscendC::MicroAPI::Cast<float, quantInputDataType, castTraitB16ToB32>(vregFloatX, vregX, preg2);
            // 3. 除以scale进行缩放
            AscendC::MicroAPI::Div(vregScaledX, vregFloatX, vregScale, preg2);
            // 4. 根据目标类型进行量化
            if constexpr (IsSameType<quantOutputDataType, int8_t>::value) {
                AscendC::MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(vregYInt16, vregScaledX, preg2);
                AscendC::MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(vregYFp16, vregYInt16, preg2);
                AscendC::MicroAPI::Cast<quantOutputDataType, half, castTraitF16ToI8>(vregY, vregYFp16, preg2);
            } else if constexpr (IsSameType<quantOutputDataType, hifloat8_t>::value) {
                AscendC::MicroAPI::Cast<quantOutputDataType, float, castTraitF32toh8>(vregY, vregScaledX, preg2);
            } else if constexpr (IsSameType<quantOutputDataType, fp8_e4m3fn_t>::value ||
                                 IsSameType<quantOutputDataType, fp8_e5m2_t>::value) {
                AscendC::MicroAPI::Cast<quantOutputDataType, float, castTraitF32tofp8>(vregY, vregScaledX, preg2);
            } else if constexpr (IsSameType<quantOutputDataType, int4b_t>::value) {
                AscendC::MicroAPI::RegTensor<uint16_t> vregPackedHalf;
                AscendC::MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(vregYInt16, vregScaledX, preg2);
                AscendC::MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(vregYFp16, vregYInt16, preg2);
                AscendC::MicroAPI::Pack(vregPackedHalf, (AscendC::MicroAPI::RegTensor<uint32_t> &)vregYFp16);
                AscendC::MicroAPI::Cast<int4x2_t, half, castTraitF16ToI8>(
                    (AscendC::MicroAPI::RegTensor<int4x2_t> &)vregY,
                    (AscendC::MicroAPI::RegTensor<half> &)vregPackedHalf, preg2);
                addr = yAddr + (j * VL) / 2;
            }
            // 5. 存储量化结果
            if constexpr (IsSameType<quantOutputDataType, int4b_t>::value) {
                AscendC::MicroAPI::DataCopy<quantOutputDataType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    addr, vregY, preg3);
            } else {
                AscendC::MicroAPI::DataCopy<quantOutputDataType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    addr, vregY, preg2);
            }
        }
    }
}

// fp8DynamicQuant的动态实现
#ifndef DEFINE_MC2_FP8_DYNAMIC_QUANT_PERTOKEN
#define DEFINE_MC2_FP8_DYNAMIC_QUANT_PERTOKEN(QuantInputDataType, QuantOutputDataType, DynamicQuantType, isSmallK)     \
    using DynamicQuantType =                                                                                           \
        MC2KernelTemplate::Fp8DynamicQuantPertoken<QuantInputDataType, QuantOutputDataType, isSmallK>
#endif
} // namespace MC2KernelTemplate
#endif
