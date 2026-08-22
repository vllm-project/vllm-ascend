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
 * \file mm1_processor.h
 * \brief
 */
#ifndef MM1_PROCESSOR_H
#define MM1_PROCESSOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../matmul_modules/ifa_flag_data.h"

using namespace matmul;

struct Mm1TaskParam {
    uint32_t sigM;
    uint32_t sigN;
    uint32_t sigK;
    uint32_t orgM;
    uint32_t orgN;
    uint32_t orgKa;
    uint32_t orgKb;
    uint32_t orgKc;
    uint32_t tscmIdx;
    int64_t tensorAOffset;
    int64_t tensorBOffset;
};
template <typename IFAT, typename mmType>
class Mm1Processor
{
public:
    using T = float;
    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using MM_OUT_T = T;
    using MM_IN_T = Q_T;
    __aicore__ inline Mm1Processor(){};
    // 非量化
    __aicore__ inline void Send(LocalTensor<MM_OUT_T>& mm1OutResUb, GlobalTensor<Q_T>& queryGm,
                                GlobalTensor<KV_T>& keyGm, const Mm1TaskParam& taskParam);
    // 伪量化
    __aicore__ inline void Send(LocalTensor<MM_OUT_T>& mm1OutResUb, TSCM<TPosition::VECIN, 1>& queryScmQueue,
                                TSCM<TPosition::VECIN, 1>& keyScmQueue, const Mm1TaskParam& taskParam);
    // 非量化L1自管理
    __aicore__ inline void Send(LocalTensor<MM_OUT_T>& mm1OutResUb, TSCM<TPosition::GM, 1>& queryScmQueue,
                                TSCM<TPosition::GM, 1>& keyScmQueue, const Mm1TaskParam& taskParam);
    __aicore__ inline void Wait();
    mmType mm;
};

template <typename IFAT, typename mmType>
__aicore__ inline void Mm1Processor<IFAT, mmType>::Send(LocalTensor<MM_OUT_T>& mm1OutResUb, GlobalTensor<Q_T>& queryGm,
                                                        GlobalTensor<KV_T>& keyGm, const Mm1TaskParam& taskParam)
{
    mm.SetOrgShape(taskParam.orgM, taskParam.orgN, taskParam.orgKa, taskParam.orgKb, taskParam.orgKc);
    mm.SetTensorA(queryGm[taskParam.tensorAOffset]);
    mm.SetTensorB(keyGm[taskParam.tensorBOffset], true);
    mm.SetTail(taskParam.sigM, taskParam.sigN, taskParam.sigK);

    mm.template IterateAll<false>(mm1OutResUb, false, false, true);
}

template <typename IFAT, typename mmType>
__aicore__ inline void Mm1Processor<IFAT, mmType>::Send(LocalTensor<MM_OUT_T>& mm1OutResUb,
                                                        TSCM<TPosition::GM, 1>& queryScmQueue,
                                                        TSCM<TPosition::GM, 1>& keyScmQueue,
                                                        const Mm1TaskParam& taskParam)
{
    LocalTensor<Q_T> queryScmTensor = queryScmQueue.DeQue<Q_T>();
    LocalTensor<KV_T> keyScmTensor = keyScmQueue.DeQue<KV_T>();
    mm.SetTensorA(queryScmTensor);
    mm.SetTensorB(keyScmTensor, true);
    mm.SetTail(taskParam.sigM, taskParam.sigN, taskParam.sigK);
    mm.template IterateAll<false>(mm1OutResUb, false, false, true);

    keyScmQueue.FreeTensor(keyScmTensor);
    queryScmQueue.FreeTensor(queryScmTensor);
}

template <typename IFAT, typename mmType>
__aicore__ inline void Mm1Processor<IFAT, mmType>::Send(LocalTensor<MM_OUT_T>& mm1OutResUb,
                                                        TSCM<TPosition::VECIN, 1>& queryScmQueue,
                                                        TSCM<TPosition::VECIN, 1>& keyScmQueue,
                                                        const Mm1TaskParam& taskParam)
{
    LocalTensor<Q_T> queryScmTensor = queryScmQueue.DeQue<Q_T>();
    LocalTensor<Q_T> keyScmTensor = keyScmQueue.DeQue<Q_T>();
    mm.SetTensorA(queryScmTensor);
    mm.SetTensorB(keyScmTensor, true);
    mm.SetTail(taskParam.sigM, taskParam.sigN, taskParam.sigK);
    mm.template IterateAll<false>(mm1OutResUb, false, false, true);

    keyScmQueue.FreeTensor(keyScmTensor);
    queryScmQueue.FreeTensor(queryScmTensor);
}

template <typename IFAT, typename mmType>
__aicore__ inline void Mm1Processor<IFAT, mmType>::Wait()
{
    mm.WaitIterateAll();
    mm.End();
}
#endif
