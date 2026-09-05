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
 * \file ffn_high_precision.h
 * \brief
 */

#ifndef ASCENDC_FFN_HIGH_PRECISION_H
#define ASCENDC_FFN_HIGH_PRECISION_H

#include "ffn_base.h"

namespace FFN {
template <typename T, class mm1Type, class mm2Type = mm1Type, typename c1T = T, typename c2T = c1T, typename BiasT = T>
class FFNHighPrecision : public FFNBase<T, mm1Type, mm2Type, c1T, c2T, BiasT> {
public:
    __aicore__ inline FFNHighPrecision(typename mm1Type::MT &mm1_, typename mm2Type::MT &mm2_)
        : FFNBase<T, mm1Type, mm2Type, c1T, c2T, BiasT>(mm1_, mm2_)
    {
    }

    /** @brief main entry function.
     */
    __aicore__ inline void Process()
    {
        if constexpr (IsSameType<BiasT, float>::value) {
            if (unlikely(this->ProcessZeroN1())) {
                return;
            }
        }

        this->ProcessNormal();
    }
};

} // namespace FFN

#endif // ASCENDC_FFN_HIGH_PRECISION_H
