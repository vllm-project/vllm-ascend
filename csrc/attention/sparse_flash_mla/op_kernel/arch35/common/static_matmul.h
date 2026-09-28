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
 * \file matmul_static.h
 * \brief 静态 tensor 版本的 MatmulK/MatmulN。
 *        与 common/op_kernel/matmul.h 中的 MatmulK/MatmulN 等价，但 L0A/L0B 通过 RingBuffer
 *        管理，且核内同步使用显式 bufferId 的 SetFlag/WaitFlag（M_MTE1 / MTE1_M），
 *        不再依赖 BuffersPolicyDB + AllocEventID。
 *        L0A/L0B 合并为单个 bufferId（INNERCORE_L0AB 语义），因为 A/B 总是成对加载、
 *        被同一个 M 管道消费。
 */
#ifndef SPARSE_FLASH_MLA_MATMUL_STATIC_H
#define SPARSE_FLASH_MLA_MATMUL_STATIC_H

#if __has_include("../../../../common/op_kernel/matmul.h")
#include "../../../../common/op_kernel/matmul.h"
#elif __has_include("../../../common/op_kernel/matmul.h")
#include "../../../common/op_kernel/matmul.h"
#elif __has_include("../../common/op_kernel/matmul.h")
#include "../../common/op_kernel/matmul.h"
#else
#include "../common/matmul.h"
#endif
#include "static_buffer.h"
using namespace AscendC;

// L0A/L0B 合并槽位对应的 flag id 映射，默认直接用槽位号 (0/1)。
// 若使用方需要将 M_MTE1/MTE1_M 的 flag id 偏移，可在 include 前覆盖此宏。
#ifndef MATMUL_STATIC_L0AB_ID
#define MATMUL_STATIC_L0AB_ID(s) (s)
#endif

namespace fa_base_matmul {

// 切K
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulKStatic(const LocalTensor<A> &aL1Tensor, const LocalTensor<B> &bL1Tensor,
                                     RingBuffer<L0AType> &l0A, RingBuffer<L0BType> &l0B,
                                     const LocalTensor<C> &cL0Tensor, const MMParam &param,
                                     const LocalTensor<AScaleType> &aScaleL1Tensor = LocalTensor<AScaleType>(),
                                     const LocalTensor<BScaleType> &bScaleL1Tensor = LocalTensor<AScaleType>())
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK = tailSize ? tailSize : baseK;
    uint64_t L1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
    uint64_t L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4;

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;

        StaticBuffer<L0AType> &aBuf = l0A.GetNext();
        StaticBuffer<L0BType> &bBuf = l0B.GetNext();

        WaitFlag<HardEvent::M_MTE1>(MATMUL_STATIC_L0AB_ID(aBuf.idx)); // 等 M 用完该 AB 槽
        LocalTensor<L0ADType> L0ATensor = aBuf.tensor.template ReinterpretCast<L0ADType>();
        LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);

        LocalTensor<L0BDType> L0BTensor = bBuf.tensor.template ReinterpretCast<L0BDType>();
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
        LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);

        SetFlag<HardEvent::MTE1_M>(MATMUL_STATIC_L0AB_ID(aBuf.idx));  // MTE1 搬运完，通知 M
        WaitFlag<HardEvent::MTE1_M>(MATMUL_STATIC_L0AB_ID(aBuf.idx)); // M 等数据就绪

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) {
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                      UNITFLAG_EN_OUTER_LAST :
                                      UNITFLAG_ENABLE;
        }
        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

        SetFlag<HardEvent::M_MTE1>(MATMUL_STATIC_L0AB_ID(aBuf.idx)); // M 用完，释放该 AB 槽
    }
}

// 切N
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulNStatic(const LocalTensor<A> &aL1Tensor, const LocalTensor<B> &bL1Tensor,
                                     RingBuffer<L0AType> &l0A, RingBuffer<L0BType> &l0B,
                                     const LocalTensor<C> &cL0Tensor, const MMParam &param,
                                     const LocalTensor<AScaleType> &aScaleL1Tensor = LocalTensor<AScaleType>(),
                                     const LocalTensor<BScaleType> &bScaleL1Tensor = LocalTensor<AScaleType>())
{
    uint32_t nLoops = (param.singleN + baseN - 1) / baseN;
    uint32_t tailSize = param.singleN % baseN;
    uint32_t tailN = tailSize ? tailSize : baseN;
    uint64_t L1Boffset = param.isRightTranspose ? (baseN << 4) : ((param.singleK + 15) >> 4 << 4) * baseN;
    uint64_t L0Coffset = ((param.singleM + 15) >> 4 << 4) * baseN;
    if (param.realM != 0) {
        L0Coffset = ((param.realM + 15) >> 4 << 4) * baseN;
    }

    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    if (param.realM != 0) {
        mmadParams.m = param.realM;
    }
    mmadParams.k = param.singleK;
    if (mmadParams.m == 1) {
        mmadParams.m = FP16_ONE_FRACTAL_ELEMENT;
    }
    mmadParams.cmatrixInitVal = param.isOutKFisrt;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = param.unitFlag;

    for (uint32_t n = 0; n < nLoops; n++) {
        uint32_t tileN = (n == (nLoops - 1)) ? tailN : baseN;
        mmadParams.n = tileN;

        StaticBuffer<L0AType> &aBuf = l0A.GetNext();
        StaticBuffer<L0BType> &bBuf = l0B.GetNext();

        WaitFlag<HardEvent::M_MTE1>(MATMUL_STATIC_L0AB_ID(aBuf.idx)); // 等 M 用完该 AB 槽
        if (n == 0 || n == 1) { // 每个 ping-pong slot 首次访问各装一份 A, 之后复用不再覆盖
            LocalTensor<L0ADType> L0ATensor = aBuf.tensor.template ReinterpretCast<L0ADType>();
            LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM);
        }

        LocalTensor<L0BDType> L0BTensor = bBuf.tensor.template ReinterpretCast<L0BDType>();
        uint64_t loopNum = param.isRightTranspose ? nLoops : 1;
        LoadDataToL0B(L0BTensor, bL1Tensor, param, n * L1Boffset, param.singleK, tileN, loopNum);

        SetFlag<HardEvent::MTE1_M>(MATMUL_STATIC_L0AB_ID(aBuf.idx));  // MTE1 搬运完，通知 M
        WaitFlag<HardEvent::MTE1_M>(MATMUL_STATIC_L0AB_ID(aBuf.idx)); // M 等数据就绪

        Mmad(cL0Tensor[n * L0Coffset], aBuf.tensor, bBuf.tensor, mmadParams);

        SetFlag<HardEvent::M_MTE1>(MATMUL_STATIC_L0AB_ID(aBuf.idx)); // M 用完，释放该 AB 槽
    }
}

} // namespace fa_base_matmul
#endif // SPARSE_FLASH_MLA_MATMUL_STATIC_H
