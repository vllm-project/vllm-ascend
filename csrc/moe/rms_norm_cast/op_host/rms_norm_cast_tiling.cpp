/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "rms_norm_cast_tiling.h"
#include "log/ops_log.h"
#include <algorithm>

namespace optiling {
namespace {
constexpr uint32_t FP16_KEY = 1;
constexpr uint32_t BF16_KEY = 3;
constexpr uint32_t B16_PER_BLOCK = 16;
constexpr uint64_t RESERVED_UB_BYTES = 1024;
constexpr uint64_t BYTES_PER_COLUMN = 12;  // two b16 and two fp32 buffers

ge::graphStatus Tiling4RmsNormCast(gert::TilingContext* context)
{
    const auto* x_shape_ptr = context->GetInputShape(0);
    const auto* gamma_shape_ptr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape_ptr);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape_ptr);
    const gert::Shape& x_shape = x_shape_ptr->GetStorageShape();
    const gert::Shape& gamma_shape = gamma_shape_ptr->GetStorageShape();
    OP_CHECK_IF(x_shape.GetDimNum() < 1 || gamma_shape.GetDimNum() != 1,
                OP_LOGE(context, "x must have rank >= 1 and gamma must have rank 1"),
                return ge::GRAPH_FAILED);

    const uint32_t num_col = gamma_shape.GetShapeSize();
    OP_CHECK_IF(num_col == 0 || x_shape.GetDim(x_shape.GetDimNum() - 1) != num_col,
                OP_LOGE(context, "gamma must match the last dimension of x"),
                return ge::GRAPH_FAILED);
    const uint64_t total = x_shape.GetShapeSize();
    OP_CHECK_IF(total == 0 || total % num_col != 0,
                OP_LOGE(context, "x must be non-empty"), return ge::GRAPH_FAILED);
    const uint32_t num_row = total / num_col;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    const uint32_t num_col_aligned = (num_col + B16_PER_BLOCK - 1) / B16_PER_BLOCK * B16_PER_BLOCK;
    OP_CHECK_IF(static_cast<uint64_t>(num_col_aligned) * BYTES_PER_COLUMN > ub_size - RESERVED_UB_BYTES,
                OP_LOGE(context, "last dimension is too large for the fused RMSNorm kernel"),
                return ge::GRAPH_FAILED);

    const uint32_t core_num = platform.GetCoreNumAiv();
    const uint32_t used_core_num = std::min(num_row, core_num);
    const uint32_t rows_per_core = (num_row + used_core_num - 1) / used_core_num;
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* epsilon = attrs->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsilon);
    OP_CHECK_IF(*epsilon < 0.0f, OP_LOGE(context, "epsilon must be non-negative"), return ge::GRAPH_FAILED);

    const auto dtype = context->GetInputDesc(0)->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16,
                OP_LOGE(context, "x only supports float16 and bfloat16"), return ge::GRAPH_FAILED);
    context->SetTilingKey(dtype == ge::DT_FLOAT16 ? FP16_KEY : BF16_KEY);
    context->SetBlockDim(used_core_num);

    RmsNormCastTilingData tiling;
    tiling.set_num_row(num_row);
    tiling.set_num_col(num_col);
    tiling.set_num_col_aligned(num_col_aligned);
    tiling.set_rows_per_core(rows_per_core);
    tiling.set_inv_num_col(1.0f / static_cast<float>(num_col));
    tiling.set_epsilon(*epsilon);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(RmsNormCast).Tiling(Tiling4RmsNormCast);
}  // namespace optiling
