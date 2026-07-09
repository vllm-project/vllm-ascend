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
 * \file scatter_nd_update.cpp
 * \brief scatter_nd_update
 */
#include "./arch35/scatter_nd_update_simt.h"
#include "./arch35/scatter_nd_update_simd.h"
#include "./arch35/scatter_nd_update_simd_mask.h"
#include "./arch35/scatter_nd_update_simt_sort.h"
#include "./arch35/scatter_nd_update_deterministic_simd.h"
#include "./arch35/scatter_nd_update_deterministic_simt.h"
#define TILING_KEY_NOT_EXCEED_INT32 100
#define TILING_KEY_EXCEED_INT32 200

static constexpr uint64_t B8 = 1;

template <typename T>
struct GetComputeType {
    using type = typename std::conditional<sizeof(T) == B8, int8_t, T>::type;
};

using namespace ScatterNdUpdate;

extern "C" __global__ __aicore__ void scatter_nd_update(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ScatterNdUpdateRegBaseTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using updateType = typename GetComputeType<DTYPE_UPDATES>::type;
    if (TILING_KEY_IS(TILING_KEY_NOT_EXCEED_INT32)) {
        if (tilingData.isMask == 1) {
            ScatterNdUpdateSimdMask<updateType, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else if (tilingData.isDeterministic == 1 && tilingData.isDeterminSimt == 1) {
            ScatterNdUpdateDeterministicSimt<updateType, DTYPE_INDICES, uint32_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else if (tilingData.isDeterministic == 1 && tilingData.isDeterminSimt != 1) {
            ScatterNdUpdateDeterministicSimd<updateType, DTYPE_INDICES, uint32_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else if (tilingData.isSimdNonDeterministic == 1) {
            ScatterNdUpdateSimd<updateType, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else if (tilingData.isSimtWithSort == 1) {
            ScatterNdUpdateSimtSort<updateType, DTYPE_INDICES, uint32_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else {
            ScatterNdUpdateSimt<updateType, DTYPE_INDICES, uint32_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_EXCEED_INT32) && tilingData.outputStorageShapeSize < INT32_MAX) {
        // outputStorageShapeSize 在 int32 范围内不需要升精度，使用 DTYPE_INDICES 作为 OFFSET_T
        if (tilingData.isMask == 1) {
            ScatterNdUpdateSimdMask<updateType, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else if (tilingData.isSimdNonDeterministic == 1) {
            ScatterNdUpdateSimd<updateType, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else if (tilingData.isDeterministic == 1 && tilingData.isDeterminSimt == 1) {
            ScatterNdUpdateDeterministicSimt<updateType, DTYPE_INDICES, uint64_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else if (tilingData.isDeterministic == 1 && tilingData.isDeterminSimt != 1) {
            ScatterNdUpdateDeterministicSimd<updateType, DTYPE_INDICES, uint64_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else if (tilingData.isSimtWithSort == 1) {
            ScatterNdUpdateSimtSort<updateType, DTYPE_INDICES, uint64_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else {
            ScatterNdUpdateSimt<updateType, DTYPE_INDICES, uint64_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_EXCEED_INT32)) {
        // outputStorageShapeSize 超出 int32 范围，必须使用 int64_t 作为 OFFSET_T
        if (tilingData.isMask == 1) {
            ScatterNdUpdateSimdMask<updateType, DTYPE_INDICES, int64_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else if (tilingData.isSimdNonDeterministic == 1) {
            ScatterNdUpdateSimd<updateType, DTYPE_INDICES, int64_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else if (tilingData.isDeterministic == 1 && tilingData.isDeterminSimt == 1) {
            ScatterNdUpdateDeterministicSimt<updateType, DTYPE_INDICES, uint64_t, int64_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else if (tilingData.isDeterministic == 1 && tilingData.isDeterminSimt != 1) {
            ScatterNdUpdateDeterministicSimd<updateType, DTYPE_INDICES, uint64_t, int64_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else if (tilingData.isSimtWithSort == 1) {
            ScatterNdUpdateSimtSort<updateType, DTYPE_INDICES, uint64_t, int64_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        } else {
            ScatterNdUpdateSimt<updateType, DTYPE_INDICES, uint64_t, int64_t> op(tilingData, pipe);
            op.Init(var, indices, updates, varRef, workspace);
            op.Process();
        }
    }
}
