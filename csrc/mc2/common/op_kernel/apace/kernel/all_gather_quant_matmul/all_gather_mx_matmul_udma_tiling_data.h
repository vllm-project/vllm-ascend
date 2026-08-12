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
 * \file all_gather_mx_matmul_udma_tiling_data.h
 * \brief Tiling data for AllGather + QuantMatmul fusion kernel (prefill variant)
 */

#pragma once

#include <cstdint>
#include "apace/tiling/quant_matmul_tiling_data.h"
#include "apace/tiling/comm_tiling_data.h"
#include "apace/block/aiv_comm/collective_comm_context.h"

namespace Apace {
namespace AivComm {
// Convenience wrapper so the kernel receives one pointer instead of two.
struct CommContext {
    CommUdmaContext udmaCtx;
    CommUbmemContext ubmemCtx;
};
} // namespace AivComm
} // namespace Apace

using Apace::AivComm::CommContext;

#pragma pack(push, 8)
struct alignas(8) AllGatherMxMatmulUdmaTilingData {
    QuantMatmulTilingData mmTile;
    CommTilingData commTile;
};
#pragma pack(pop)
