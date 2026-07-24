/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_wu_fwd_struct.h
 * \brief Shared tiling data for recompute_wu_fwd.
 */

#ifndef RECOMPUTE_WU_FWD_STRUCT_H
#define RECOMPUTE_WU_FWD_STRUCT_H

#include <cstdint>

namespace GDN {

struct RecomputeWUFwdTilingData {
    int64_t B;
    int64_t Hk;
    int64_t Hv;
    int64_t hvPerHk;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t chunkNum;
    int64_t chunkSize;
    int64_t vbVecRow;
    int64_t kbgExpVecRow;
    int64_t isVariable;
};

} // namespace GDN

#endif // RECOMPUTE_WU_FWD_STRUCT_H
