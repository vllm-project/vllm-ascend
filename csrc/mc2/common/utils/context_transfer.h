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
 * \file context_transfer.h
 * \brief
 */
#ifndef _CONTEXT_TRANSFER_H_
#define _CONTEXT_TRANSFER_H_
#include "runtime/tiling_context.h"

namespace optiling {

constexpr uint64_t DIM_THREE = 3;
constexpr uint64_t DIM_ONE = 1;

// MMR代表MatMulAllReduce缩写
struct MMRCtxInfo {
    const char *group{nullptr};
    const char *reduceOp{nullptr};
    const bool *isTransA{nullptr};
    const bool *isTransB{nullptr};
    int32_t commTurn{0};
    const int64_t *antiquantGroupSizePtr{nullptr};
    const int64_t *groupSizePtr{nullptr};
    const int64_t *yDtypePtr{nullptr};
    const int64_t *commQuantModePtr{nullptr};
    const gert::CompileTimeTensorDesc *x1{nullptr};
    const gert::CompileTimeTensorDesc *x2{nullptr};
    const gert::CompileTimeTensorDesc *bias{nullptr};
    const gert::CompileTimeTensorDesc *x3{nullptr};
    const gert::CompileTimeTensorDesc *antiquant_scale{nullptr};
    const gert::CompileTimeTensorDesc *antiquant_offset{nullptr};
    const gert::CompileTimeTensorDesc *dequant_scale{nullptr};
    const gert::CompileTimeTensorDesc *pertoken_scale{nullptr};
    const gert::CompileTimeTensorDesc *comm_quant_scale_1{nullptr};
    const gert::CompileTimeTensorDesc *comm_quant_scale_2{nullptr};
    const gert::CompileTimeTensorDesc *y{nullptr};
    const gert::StorageShape *x1_shape{nullptr};
    const gert::StorageShape *x2_shape{nullptr};
    const gert::StorageShape *bias_shape{nullptr};
    const gert::StorageShape *x3_shape{nullptr};
    const gert::StorageShape *antiquant_scale_shape{nullptr};
    const gert::StorageShape *antiquant_offset_shape{nullptr};
    const gert::StorageShape *dequant_scale_shape{nullptr};
    const gert::StorageShape *pertoken_scale_shape{nullptr};
    const gert::StorageShape *comm_quant_scale_1_shape{nullptr};
    const gert::StorageShape *comm_quant_scale_2_shape{nullptr};
    const gert::StorageShape *y_shape{nullptr};
};
// ARN代表AddResNorm的缩写
struct ARNCtxInfo {
    const float *epsilon{nullptr};
    const gert::CompileTimeTensorDesc *x1{nullptr};
    const gert::CompileTimeTensorDesc *x2{nullptr};
    const gert::CompileTimeTensorDesc *gamma{nullptr};
    const gert::CompileTimeTensorDesc *y{nullptr};
    const gert::CompileTimeTensorDesc *rstd{nullptr};
    const gert::CompileTimeTensorDesc *x{nullptr};
    const gert::StorageShape *x1_shape{nullptr};
    const gert::StorageShape *x2_shape{nullptr};
    const gert::StorageShape *gamma_shape{nullptr};
    const gert::StorageShape *y_shape{nullptr};
    const gert::StorageShape *rstd_shape{nullptr};
    const gert::StorageShape *x_shape{nullptr};
};
// MRN代表MatMulAllReduceAddResNorm的缩写
struct MRNCtxInfo {
    MMRCtxInfo mmrCtxInfo;
    ARNCtxInfo arnCtxInfo;
};
using IMRNCtxInfo = MRNCtxInfo;
class ContextTransfer {
public:
    // MRN代表MatMulAllReduceAddResNorm的缩写
    static ge::graphStatus AssembleMMRCtxInfoFromMRNCtx(const gert::TilingContext *const context,
                                                        MMRCtxInfo &mmrCtxInfo);

    static ge::graphStatus AssembleARNCtxInfoFromMRNCtx(const gert::TilingContext *const context,
                                                        ARNCtxInfo &arnCtxInfo);

    // IMRN代表InplaceMatMulAllReduceAddResNorm的缩写
    static ge::graphStatus AssembleMMRCtxInfoFromIMRNCtx(const gert::TilingContext *const context,
                                                         MMRCtxInfo &mmrCtxInfo);

    static ge::graphStatus AssembleARNCtxInfoFromIMRNCtx(const gert::TilingContext *const context,
                                                         ARNCtxInfo &arnCtxInfo);

    static ge::graphStatus AssembleMRNCtxInfoFromMRNCtx(const gert::TilingContext *const context,
                                                        MRNCtxInfo &mrnCtxInfo);
    static ge::graphStatus AssembleIMRNCtxInfoFromIMRNCtx(const gert::TilingContext *const context,
                                                          IMRNCtxInfo &imrnCtxInfo);
    static ge::graphStatus AssembleMMRCtxInfoFromMMRCtx(const gert::TilingContext *const context,
                                                        MMRCtxInfo &mmrCtxInfo);
    static ge::graphStatus CheckMRNCtxInfo(const gert::TilingContext *context, const MRNCtxInfo &mrnCtxInfo);
};
} // namespace optiling
#endif // _CONTEXT_TRANSFER_H_
