/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file collective_comm_api.h
 * \brief Hcomm 集合通信统一 API
 */

#pragma once

#include "collective_comm_base.h"
#include "all_to_all/all_to_all_udma_get.h"
#include "all_to_all/all_to_all_udma_put.h"
#include "all_gather/all_gather_udma_put.h"

namespace Apace {
namespace AivComm {

using namespace AscendC;

enum class CommCollectiveOp {
    AllToAll,
    AllGather,
    ReduceScatter
};

enum class CommMode {
    GET,
    PUT
};

template<CommCollectiveOp Op, CommMode Mode, typename T, typename Barrier>
struct CollectiveCommHelper;

template<typename T, typename Barrier>
struct CollectiveCommHelper<CommCollectiveOp::AllToAll, CommMode::GET, T, Barrier> {
    using type = AllToAllCommGetImpl<T, Barrier>;
};
template<typename T, typename Barrier>
struct CollectiveCommHelper<CommCollectiveOp::AllToAll, CommMode::PUT, T, Barrier> {
    using type = AllToAllCommPutImpl<T, Barrier>;
};
template<typename T, typename Barrier>
struct CollectiveCommHelper<CommCollectiveOp::AllGather, CommMode::PUT, T, Barrier> {
    using type = AllGatherCommPutImpl<T, Barrier>;
};

template<CommCollectiveOp Op, CommMode Mode, typename T, typename Barrier>
using CollectiveComm = typename CollectiveCommHelper<Op, Mode, T, Barrier>::type;

} // namespace AivComm
} // namespace Apace
