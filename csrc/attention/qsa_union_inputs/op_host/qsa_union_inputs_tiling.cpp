/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "qsa_union_inputs_tiling.h"

#include <algorithm>

#include "log/ops_log.h"
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/sort/sort_tiling_intf.h"

namespace optiling {
namespace {
constexpr uint32_t UNION_SIZE = 8;
constexpr uint32_t BLOCK_Y = 4;
constexpr uint32_t OUTPUT_TILE = 512;
constexpr uint32_t RESERVED_UB_BYTES = 2048;

uint32_t AlignUp32(uint32_t value)
{
    return (value + 31U) / 32U * 32U;
}

ge::graphStatus Tiling4QsaUnionInputs(gert::TilingContext* context)
{
    const auto* compact_shape_ptr = context->GetInputShape(0);
    const auto* visible_shape_ptr = context->GetInputShape(1);
    const auto* row_map_shape_ptr = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, compact_shape_ptr);
    OP_CHECK_NULL_WITH_CONTEXT(context, visible_shape_ptr);
    OP_CHECK_NULL_WITH_CONTEXT(context, row_map_shape_ptr);
    const auto& compact_shape = compact_shape_ptr->GetStorageShape();
    const auto& visible_shape = visible_shape_ptr->GetStorageShape();
    const auto& row_map_shape = row_map_shape_ptr->GetStorageShape();
    OP_CHECK_IF(compact_shape.GetDimNum() != 2 ||
                    visible_shape.GetDimNum() != 1 ||
                    row_map_shape.GetDimNum() != 2,
                OP_LOGE(context, "QsaUnionInputs expects rank-2/rank-1/rank-2 inputs"),
                return ge::GRAPH_FAILED);

    const uint32_t rows = compact_shape.GetDim(0);
    const uint32_t compact_columns = compact_shape.GetDim(1);
    const uint32_t groups = row_map_shape.GetDim(0);
    OP_CHECK_IF(compact_columns <= 3 || visible_shape.GetDim(0) != rows ||
                    row_map_shape.GetDim(1) != UNION_SIZE,
                OP_LOGE(context, "QsaUnionInputs input shapes are inconsistent"),
                return ge::GRAPH_FAILED);
    const uint32_t input_groups = compact_columns - 3;
    const uint32_t topk = input_groups + (input_groups < 512 ? 1 : 0);
    OP_CHECK_IF(topk == 0 || (topk & (topk - 1)) != 0 || topk > 512,
                OP_LOGE(context, "QsaUnionInputs supports power-of-two topk up to 512"),
                return ge::GRAPH_FAILED);
    const uint32_t width = UNION_SIZE * topk;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    const bool is_a3 = platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_93;
    const uint32_t sort_tmp_bytes = is_a3 ? width * sizeof(int32_t) * 3 :
        AscendC::GetSortTmpSize(platform, width, sizeof(int32_t));
    const uint32_t prepare_vector_bytes =
        AlignUp32(input_groups * sizeof(int32_t));
    const uint32_t prepare_mask_bytes =
        AlignUp32((input_groups + 7U) / 8U);
    const uint64_t required_ub =
        static_cast<uint64_t>(width) * sizeof(int32_t) * 2 +
        static_cast<uint64_t>(width) * sizeof(int32_t) * 2 +
        sort_tmp_bytes + AlignUp32(input_groups * sizeof(int32_t)) +
        prepare_vector_bytes * 4 + prepare_mask_bytes * 2 +
        (is_a3 ? width * sizeof(float) * 2 : 0) +
        OUTPUT_TILE * (1 + BLOCK_Y) * sizeof(int32_t) +
        32 * sizeof(int32_t) + 512 * sizeof(int32_t) + RESERVED_UB_BYTES;
    OP_CHECK_IF(required_ub > ub_size,
                OP_LOGE(context, "QsaUnionInputs requires %lu UB bytes but only %lu are available",
                        required_ub, ub_size),
                return ge::GRAPH_FAILED);

    const uint32_t core_num = platform.GetCoreNumAiv();
    OP_CHECK_IF(groups == 0 || core_num == 0,
                OP_LOGE(context, "QsaUnionInputs requires at least one union group"),
                return ge::GRAPH_FAILED);
    const uint32_t groups_per_core =
        (groups + core_num - 1) / core_num;

    QsaUnionInputsTilingData tiling;
    tiling.set_rows(rows);
    tiling.set_groups(groups);
    tiling.set_input_groups(input_groups);
    tiling.set_topk(topk);
    tiling.set_width(width);
    tiling.set_compact_row_stride(compact_columns);
    tiling.set_sort_tmp_bytes(sort_tmp_bytes);
    tiling.set_groups_per_core(groups_per_core);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    // Keep blockDim shape-invariant. The ACLNN executor can be reused across
    // dynamic row/group shapes; a first call with one group must not pin later
    // calls to one AIV core. Extra cores naturally take an empty begin/end
    // range in the device kernel.
    context->SetBlockDim(core_num);
    // This operator emits one generic AscendC entry, so the runtime function
    // entry must use the default tiling key.
    context->SetTilingKey(0);
    // One 32-byte slot per group prevents multi-core false sharing while
    // publishing the compact union counts.
    context->GetWorkspaceSizes(1)[0] = is_a3 ? 0 :
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) +
        static_cast<size_t>(groups) * 8 * sizeof(int32_t);
    return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(QsaUnionInputs).Tiling(Tiling4QsaUnionInputs);
}  // namespace optiling
