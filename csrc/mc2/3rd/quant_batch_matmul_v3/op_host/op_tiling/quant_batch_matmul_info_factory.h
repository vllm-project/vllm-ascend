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
 * \file quant_batch_matmul_info_factory.h
 * \brief
 */
#ifndef MC2_QUANT_BATCH_MATMUL_INFO_FACTORY_H
#define MC2_QUANT_BATCH_MATMUL_INFO_FACTORY_H

#include <pthread.h>

#include "common/op_host/op_tiling/lock.h"
#include "quant_batch_matmul_v3_tiling.h"

namespace optiling {
class Mc2QuantBatchMatmulInfoFactory {
public:
    Mc2QuantBatchMatmulInfoFactory() = default;
    ~Mc2QuantBatchMatmulInfoFactory() = default;

    Mc2QuantBatchMatmulInfo *Get()
    {
        Mc2QuantBatchMatmulInfo *ptr = nullptr;
        auto threadId = pthread_self();
        lock_.rdlock();
        auto it = inst_.find(threadId);
        if (it == inst_.end()) {
            lock_.unlock();
            lock_.wrlock();
            ptr = &(inst_[threadId]);
        } else {
            ptr = &(it->second);
        }

        lock_.unlock();
        return ptr;
    }

private:
    std::map<pthread_t, Mc2QuantBatchMatmulInfo> inst_;
    Ops::Transformer::Optiling::RWLock lock_;
};

} // namespace optiling
#endif // QUANT_BATCH_MATMUL_INFO_FACTORY_H
