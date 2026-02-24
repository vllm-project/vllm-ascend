
#include "../op_kernel/matmul_gelu_tiling.h"
#include "register/op_def_registry.h"

#include "graph/utils/type_utils.h"
#include "tiling/tiling_api.h"

#include "log/ops_log.h"
#include "error/ops_error.h"

//static ge::graphStatus TilingFunc(gert::TilingContext* context)
//{
//  Matmul_geluTilingData tiling;
//  const gert::StorageShape* x1_shape = context->GetInputShape(0);
//  int32_t data_sz = 1;
//  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
//    data_sz *= x1_shape->GetStorageShape().GetDim(i);
//  tiling.set_size(data_sz);
//  context->SetBlockDim(8);
//  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
//  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
//
//  return ge::GRAPH_SUCCESS;
//}

constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INPUT_INDEX_WEIGHT = 1;
constexpr size_t INPUT_INDEX_BIAS = 2;
constexpr uint32_t ONE_DIMS = 1;
constexpr uint32_t TWO_DIMS = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const char *nodeName = context->GetNodeName();
    MatmulGeluTilingData *tiling = context->GetTilingData<MatmulGeluTilingData>();
    OPS_ERR_IF(tiling == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    const gert::InputDesc *inputDesc = context->GetInputDesc(INPUT_INDEX_X);
    OPS_ERR_IF(inputDesc == nullptr, OPS_LOG_E(nodeName, "x shape is null."), return ge::GRAPH_FAILED);
    const ge::DataType xDataType = inputDesc->GetDataType();

    const gert::StorageShape *xStorageShape = context->GetInputShape(INPUT_INDEX_X);
    OPS_ERR_IF(xStorageShape == nullptr, OPS_LOG_E(nodeName, "x shape is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OPS_LOG_E(nodeName, "x shape dims must be 2, but current dim num is %lu.",
                            xStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *weightStorageShape = context->GetInputShape(INPUT_INDEX_WEIGHT);
    OPS_ERR_IF(weightStorageShape == nullptr, OPS_LOG_E(nodeName, "weight shape is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(weightStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OPS_LOG_E(nodeName, "weight shape dims must be 2, but current dim num is %lu.",
                            weightStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *biasStorageShape = context->GetInputShape(INPUT_INDEX_BIAS);
    OPS_ERR_IF(biasStorageShape == nullptr, OPS_LOG_E(nodeName, "bias shape is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(biasStorageShape->GetStorageShape().GetDimNum() != ONE_DIMS,
                    OPS_LOG_E(nodeName, "bias shape dims must be 1, but current dim num is %lu.",
                            biasStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);

    size_t  m = xShape.GetDim(0);
    size_t  k = xShape.GetDim(1);
    size_t  n = BiasShape.GetDim(0);
    tiling->m = m;
    tiling->n = n;
    tiling->k = k;

    if (k == weightShape.GetDim(0)) {
        tiling->transB = false;
    } else if (k == weightShape.GetDim(1)) {
        tiling->transB = true;
    } else {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t totalCoreNum = ascendcPlatform.GetCoreNumAic();

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    const size_t elementSize = (xDataType == ge::DT_FLOAT) ? sizeof(float) : 2;
    const size_t userWorkspaceSize = m * n * 4;
    currentWorkspace[0] = systemWorkspacesSize + userWorkspaceSize;

    context->SetBlockDim(totalCoreNum);
    // context->SetTilingKey(static_cast<uint64_t>(xDataType));

    return ge::GRAPH_SUCCESS;
}

struct MatmulGeluCompileInfo {};
ge::graphStatus TilingParseForMatmulGelu(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulGelu)
    .Tiling(TilingFunc)
    .TilingParse<MatmulGeluCompileInfo>(TilingParseForMatmulGelu);