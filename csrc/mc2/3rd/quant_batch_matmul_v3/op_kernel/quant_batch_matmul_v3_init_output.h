/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_init_output.h
 * \brief
 */
#ifndef MC2_QUANT_BATCH_MATMUL_V3_INIT_OUTPUT_H
#define MC2_QUANT_BATCH_MATMUL_V3_INIT_OUTPUT_H

#include "quant_batch_matmul_v3_base.h"

namespace AscendC {
template <typename yType>
class BmmDequantInitOutput {
public:
    __aicore__ inline BmmDequantInitOutput()
    {
    }
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR workSpace, const Mc2QuantBatchMatmulV3TilingData *tilingData,
                                TPipe *tPipe)
    {
        InitTilingData(tilingData);
        // init global buffer
        yGm_.SetGlobalBuffer((__gm__ yType *)y);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        workspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workSpace);
        tPipe->InitBuffer(localBuffer_, tilingData->params.ubSize);
        ClearWorkSpace();
#else
        tPipe->InitBuffer(localBuffer_, TOTAL_L1_SIZE);
#endif
    }

    /** main logical function
     */
    __aicore__ inline void Process()
    {
        uint32_t usedCoreNum = GetBlockNum();
        uint64_t clearSizePerCore = DequantBmm::Align(DequantBmm::CeilDiv(outputSize_, usedCoreNum), MIN_CLEAR_SIZE);
        usedCoreNum = DequantBmm::CeilDiv(outputSize_, clearSizePerCore);
        if (GetBlockIdx() >= usedCoreNum) {
            SyncAllCores();
            return;
        }

        LocalTensor<yType> tmpBuf = localBuffer_.Get<yType>();

        uint64_t dstOffset = clearSizePerCore * GetBlockIdx();
        uint64_t realClearSize = DequantBmm::Min<uint64_t>(outputSize_ - dstOffset, clearSizePerCore);
        uint64_t clearBlkNum = realClearSize * sizeof(yType) / ONE_BLK_SIZE;

        DataCopyParams dataCopyParams(1, 1, 0, 0);
        if (unlikely(clearBlkNum == 0U)) {
            ClearLocalTensor(tmpBuf, ONE_BLK_ITEM_NUM);
            // 只支持32B对齐搬出，这里尽量避免越界，往前凑。如果是小于32B的极小case，那只能越界了
            if (dstOffset > ONE_BLK_ITEM_NUM - realClearSize) {
                dstOffset -= (ONE_BLK_ITEM_NUM - realClearSize);
            }

            DataCopy(yGm_[dstOffset], tmpBuf, dataCopyParams);
        } else {
            dataCopyParams.blockLen =
                DequantBmm::Min<uint64_t>(tmpBuf.GetSize() * sizeof(yType) / ONE_BLK_SIZE, clearBlkNum);
            uint64_t burstItemNum = dataCopyParams.blockLen * ONE_BLK_ITEM_NUM;
            ClearLocalTensor(tmpBuf, burstItemNum);
            uint64_t loop = clearBlkNum / dataCopyParams.blockLen;
            for (uint64_t idx = 0; idx < loop; ++idx) {
                DataCopy(yGm_[dstOffset], tmpBuf, dataCopyParams);
                dstOffset += burstItemNum;
            }

            dataCopyParams.blockLen = clearBlkNum % dataCopyParams.blockLen;
            if (dataCopyParams.blockLen > 0) {
                DataCopy(yGm_[dstOffset], tmpBuf, dataCopyParams);
                dstOffset += (dataCopyParams.blockLen * ONE_BLK_ITEM_NUM);
            }

            uint64_t tailClearSize = realClearSize - clearBlkNum * ONE_BLK_ITEM_NUM;
            if (tailClearSize > 0U) {
                dataCopyParams.blockLen = 1;
                // 同前，尽量避免越界，往前凑
                if (dstOffset > ONE_BLK_ITEM_NUM - tailClearSize) {
                    dstOffset -= (ONE_BLK_ITEM_NUM - tailClearSize);
                }

                DataCopy(yGm_[dstOffset], tmpBuf, dataCopyParams);
            }
        }

        SyncAllCores();
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    __aicore__ inline void ClearWorkSpace()
    {
        LocalTensor<int32_t> tmpUb = localBuffer_.Get<int32_t>();
        uint32_t size = GetBlockNum() * ONE_BLK_SIZE / sizeof(int32_t);
        Duplicate(tmpUb, 0, size);

        TEventID eventIdVtoMte3 = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventIdVtoMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVtoMte3);

        DataCopy(workspaceGm_, tmpUb, size);
    }

    __aicore__ inline void ClearLocalTensor(LocalTensor<yType> &tmpBuf, uint32_t size)
    {
        LocalTensor<int16_t> tmpTensor = tmpBuf.template ReinterpretCast<int16_t>();
        // Duplicate not support int8
        Duplicate(tmpTensor, static_cast<int16_t>(0), DequantBmm::CeilDiv(size * sizeof(yType), sizeof(int16_t)));

        TEventID eventIdVtoMte3 = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventIdVtoMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVtoMte3);
    }

    __aicore__ inline void SyncAllCores()
    {
        LocalTensor<int32_t> ubWorkspace = localBuffer_.Get<int32_t>();
        SyncAll(workspaceGm_, ubWorkspace);
    }
#else
    __aicore__ inline void ClearLocalTensor(LocalTensor<yType> &tmpBuf, uint32_t size)
    {
        uint16_t blockNum = size / ONE_BLK_ITEM_NUM;
        InitConstValue(tmpBuf, {1, blockNum, 0, 0U});

        TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE2_MTE3>();
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
    }

    __aicore__ inline void SyncAllCores()
    {
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_AIC_FLAG);
        CrossCoreWaitFlag(SYNC_AIC_FLAG);
    }
#endif

private:
    uint64_t outputSize_;
    GlobalTensor<yType> yGm_;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    GlobalTensor<int32_t> workspaceGm_;
    TBuf<TPosition::VECCALC> localBuffer_;
#else
    TBuf<TPosition::TSCM> localBuffer_;
#endif

    static constexpr uint64_t MIN_CLEAR_BYTE = 512UL;
    static constexpr uint64_t MIN_CLEAR_SIZE = MIN_CLEAR_BYTE / sizeof(yType);
    static constexpr uint64_t ONE_BLK_ITEM_NUM = ONE_BLK_SIZE / sizeof(yType);

    __aicore__ inline void InitTilingData(const Mc2QuantBatchMatmulV3TilingData *tilingData)
    {
        uint32_t batch = tilingData->params.batchC;
        uint32_t mSize = tilingData->matmulTiling.M;
        uint32_t nSize = tilingData->matmulTiling.N;
        outputSize_ = static_cast<uint64_t>(batch) * mSize * nSize;
    }
};
} // namespace AscendC

#endif // QUANT_BATCH_MATMUL_V3_INIT_OUTPUT_H
