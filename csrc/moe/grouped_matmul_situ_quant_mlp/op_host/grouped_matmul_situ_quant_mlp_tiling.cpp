#include "grouped_matmul_situ_quant_mlp_tiling_data.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <register/op_impl_registry.h>

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

using namespace matmul_tiling;
using namespace GroupedMatmulSituQuantTiling;

enum InputIndex { X, X_SCALE, GROUP_LIST, UP_WEIGHT, UP_SCALE, UP_COMPENSATION,
                  DOWN_WEIGHT, DOWN_SCALE, DOWN_COMPENSATION };

constexpr int64_t REPORT_EXPERTS = 14;
constexpr int64_t REPORT_WIDTH = 7168;
constexpr int64_t MAX_CAPACITY = 7168;
constexpr int64_t EXPERT_SLOT_ROWS = 640;
constexpr int64_t CUBE_OUTPUT_BYTES_PER_ROW = 4;
constexpr int64_t SYSTEM_WORKSPACE_BYTES = 16 * 1024 * 1024;
constexpr uint64_t MLP_TILING_KEY = 0;

struct CubeTile {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t kSteps;
};

bool HasShape(const gert::StorageShape* storage, std::initializer_list<int64_t> sizes, bool scale = false)
{
    if (storage == nullptr) {
        return false;
    }
    const auto& shape = storage->GetStorageShape();
    if (scale && shape.GetDimNum() == 3) {
        return shape.GetDim(0) == *sizes.begin() && shape.GetDim(1) == 1 &&
            shape.GetDim(2) == *(sizes.begin() + 1);
    }
    if (shape.GetDimNum() != sizes.size()) {
        return false;
    }
    size_t dim = 0;
    for (int64_t size : sizes) {
        if (shape.GetDim(dim++) != size) {
            return false;
        }
    }
    return true;
}

bool HasNzStorage(const gert::TilingContext* context, InputIndex index, int64_t experts, int64_t k, int64_t n)
{
    const auto* tensor = context->GetInputShape(index);
    const auto* desc = context->GetInputDesc(index);
    if (tensor == nullptr || desc == nullptr || desc->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ) {
        return false;
    }
    const auto& shape = tensor->GetStorageShape();
    // The ACLNN individual-op path may expose the NZ tensor's logical rank.
    if (shape.GetDimNum() == 3) {
        return HasShape(tensor, {experts, k, n});
    }
    return shape.GetDimNum() == 5 && shape.GetDim(0) == experts && shape.GetDim(1) == n / 64 &&
        shape.GetDim(2) == k / 16 && shape.GetDim(3) == 16 && shape.GetDim(4) == 64;
}

bool BuildCube(platform_ascendc::PlatformAscendC& platform, TCubeTiling& out, CubeTile tile,
               int64_t capacity, int64_t k, int64_t n)
{
    MatmulApiTiling tiling(platform);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT4);
    tiling.SetBType(TPosition::GM, CubeFormat::NZ, DataType::DT_INT4);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    tiling.SetBias(false);
    tiling.SetShape(tile.m, tile.n, k);
    tiling.SetFixSplit(tile.m, tile.n, tile.k);
    tiling.SetOrgShape(capacity, n, k);
    tiling.SetBufferSpace(-1, -1, -1);
    if (tiling.GetTiling(out) == -1) {
        return false;
    }
    out.set_baseM(tile.m);
    out.set_baseN(tile.n);
    out.set_baseK(tile.k);
    out.set_dbL0B(2);
    out.set_stepKa(tile.kSteps);
    out.set_stepKb(tile.kSteps);
    out.set_depthA1(2 * tile.kSteps);
    out.set_depthB1(2 * tile.kSteps);
    out.set_stepM(1);
    out.set_stepN(1);
    return true;
}

void FillParams(GMMSituQuantBaseParams& params, int64_t cores, int64_t experts, int64_t capacity,
                int64_t k, int64_t n, int64_t slotRows, int64_t groupListType, int64_t aBytes,
                int64_t cBytes)
{
    params.set_coreNum(cores);
    params.set_groupNum(experts);
    params.set_M(capacity);
    params.set_K(k);
    params.set_N(n);
    params.set_mLimit(slotRows);
    params.set_workSpaceOffset1(aBytes);
    params.set_workSpaceOffset2(cBytes);
    params.set_quantGroupNum(1);
    params.set_isSingleTensor(1);
    params.set_groupListType(groupListType);
    params.set_assistRowsPerExpert(0);
}

int64_t VectorRows(uint64_t ubBytes, uint64_t width)
{
    // The shared epilogue uses 8.5*width bytes per row, two width-wide
    // FP32 temporaries and a 32-byte aligned FP32 scale queue.
    const uint64_t fixedBytes = 8 * width + 64;
    if (ubBytes <= fixedBytes) {
        return 0;
    }
    const auto rowBytes = 17 * width / 2;
    uint64_t rows = (ubBytes - fixedBytes) / rowBytes;
    while (rows != 0 && fixedBytes + rowBytes * rows + 32 * ((rows + 7) / 8) > ubBytes) {
        --rows;
    }
    return rows;
}

ge::graphStatus Tiling(gert::TilingContext* context)
{
    const auto* x = context->GetInputShape(X);
    const auto* groups = context->GetInputShape(GROUP_LIST);
    if (x == nullptr || groups == nullptr || x->GetOriginShape().GetDimNum() != 2 ||
        groups->GetOriginShape().GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t capacity = x->GetOriginShape().GetDim(0);
    const int64_t width = x->GetOriginShape().GetDim(1);
    const int64_t experts = groups->GetOriginShape().GetDim(0);
    const bool report = experts == REPORT_EXPERTS && width == REPORT_WIDTH;
    const bool routed = experts == A8W4_KIMI_EP128_EXPERTS && width == A8W4_KIMI_ROUTED_K;
    if ((!report && !routed) || capacity <= 0 || capacity > MAX_CAPACITY) {
        OP_LOGE(context->GetNodeName(), "Unsupported A2/A3 SiTU MLP shape");
        return ge::GRAPH_FAILED;
    }
    const int64_t upWidth = report ? REPORT_WIDTH : A8W4_KIMI_ROUTED_N;
    const int64_t activationWidth = upWidth / 2;
    const bool validInputs[] = {
        HasShape(context->GetInputShape(X_SCALE), {capacity}),
        HasNzStorage(context, UP_WEIGHT, experts, width, upWidth),
        HasNzStorage(context, DOWN_WEIGHT, experts, activationWidth, width),
        HasShape(context->GetInputShape(UP_SCALE), {experts, upWidth}, true),
        HasShape(context->GetInputShape(DOWN_SCALE), {experts, width}, true),
        HasShape(context->GetInputShape(UP_COMPENSATION), {experts, upWidth}),
        HasShape(context->GetInputShape(DOWN_COMPENSATION), {experts, width})
    };
    constexpr InputIndex validatedInputs[] = {
        X_SCALE, UP_WEIGHT, DOWN_WEIGHT, UP_SCALE, DOWN_SCALE, UP_COMPENSATION, DOWN_COMPENSATION
    };
    for (size_t index = 0; index < sizeof(validInputs) / sizeof(validInputs[0]); ++index) {
        if (!validInputs[index]) {
            OP_LOGE(context->GetNodeName(), "Invalid SiTU MLP input shape at index %d", validatedInputs[index]);
            return ge::GRAPH_FAILED;
        }
    }
    const auto* attrs = context->GetAttrs();
    const auto* groupType = attrs->GetAttrPointer<int64_t>(0);
    const auto* beta = attrs->GetAttrPointer<float>(1);
    const auto* linearBeta = attrs->GetAttrPointer<float>(2);
    if (groupType == nullptr || beta == nullptr || linearBeta == nullptr || (*groupType != 0 && *groupType != 1) ||
        !std::isfinite(*beta) || *beta <= 0 || !std::isfinite(*linearBeta)) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t cores = platform.GetCoreNumAic();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const int64_t upRows = VectorRows(ubBytes, upWidth);
    const int64_t downRows = VectorRows(ubBytes, width);
    if (cores == 0 || upRows == 0 || downRows == 0) {
        return ge::GRAPH_FAILED;
    }

    // Each phase has two reusable partial-result slots. The activation and
    // row-scale buffers cover the entire captured capacity.
    const int64_t slotRows = std::min(capacity, EXPERT_SLOT_ROWS);
    const int64_t slots = capacity <= slotRows ? 1 : 2;
    const int64_t aBytes = slots * slotRows * width;
    const int64_t cBytes = slots * slotRows * upWidth * CUBE_OUTPUT_BYTES_PER_ROW;
    const int64_t packedBytes = capacity * activationWidth;
    const int64_t scaleBytes = capacity * sizeof(float);

    GroupedMatmulSituQuantMlpTilingData data;
    if (!BuildCube(platform, data.upDefault, {256, 128, 256, 4}, capacity, width, upWidth) ||
        !BuildCube(platform, data.upDecode, {128, 256, 256, 4}, capacity, width, upWidth) ||
        !BuildCube(platform, data.upSingleToken, {64, 384, 128, 8}, capacity, width, upWidth) ||
        !BuildCube(platform, data.downCube, report ? CubeTile{64, 384, 128, 8} : CubeTile{128, 256, 256, 4},
                   capacity, activationWidth, width)) {
        return ge::GRAPH_FAILED;
    }
    FillParams(data.upParams, cores, experts, capacity, width, upWidth, slotRows, *groupType, aBytes, cBytes);
    FillParams(data.downParams, cores, experts, capacity, activationWidth, width, slotRows,
               *groupType, aBytes, cBytes);
    data.upVector.set_tokenLen(upWidth);
    data.upVector.set_groupListLen(experts);
    data.upVector.set_maxProcessRowNum(upRows);
    data.upVector.set_beta(*beta);
    data.upVector.set_linearBeta(*linearBeta);
    data.upVector.set_activateLeft(1);
    data.downVector.set_tokenLen(width);
    data.downVector.set_groupListLen(experts);
    data.downVector.set_maxProcessRowNum(std::min(upRows, downRows));

    auto* workspace = context->GetWorkspaceSizes(1);
    auto* raw = context->GetRawTilingData();
    if (workspace == nullptr || raw == nullptr || raw->GetCapacity() < data.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = SYSTEM_WORKSPACE_BYTES + aBytes + cBytes + packedBytes + scaleBytes;
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(data.GetDataSize());
    context->SetTilingKey(MLP_TILING_KEY);
    context->SetBlockDim(cores);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_OPTILING(GroupedMatmulSituQuantMlp).Tiling(Tiling);

} // namespace optiling
