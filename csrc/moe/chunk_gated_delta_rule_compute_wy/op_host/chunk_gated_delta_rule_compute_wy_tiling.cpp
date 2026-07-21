#include "chunk_gated_delta_rule_compute_wy_tiling.h"

#include <algorithm>

#include <register/op_impl_registry.h>
#include <tiling/tiling_api.h>

namespace optiling {
static constexpr size_t INPUT_Q_IDX = 0;
static constexpr size_t INPUT_V_IDX = 2;
static constexpr size_t ATTR_CHUNK_SIZE_IDX = 0;

static constexpr size_t DIM_B = 0;
static constexpr size_t DIM_T = 1;
static constexpr size_t DIM_H = 2;
static constexpr size_t DIM_D = 3;

static constexpr int64_t FIXED_CHUNK = 64;
static constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
// Atlas inference Matmul needs a UB scratch region for internal temps.
static constexpr uint32_t LOCAL_WORKSPACE_BYTES = 32 * 1024;
// Per-core GM staging: A_half(64*128) + B_half(64*128) + C_float(64*128)
static constexpr uint32_t STAGING_A_BYTES = FIXED_CHUNK * 128 * sizeof(uint16_t);
static constexpr uint32_t STAGING_B_BYTES = FIXED_CHUNK * 128 * sizeof(uint16_t);
static constexpr uint32_t STAGING_C_BYTES = FIXED_CHUNK * 128 * sizeof(float);
static constexpr uint32_t PER_CORE_STAGING_BYTES = STAGING_A_BYTES + STAGING_B_BYTES + STAGING_C_BYTES;

static ge::graphStatus FillCubeTiling(gert::TilingContext *context, int64_t m, int64_t n, int64_t k, bool bTranspose,
                                      TCubeTiling &out)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                bTranspose);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm.SetBias(false);
    mm.SetOrgShape(m, n, k);
    mm.SetShape(m, n, k);
    mm.SetBufferSpace(-1, -1, -1);
    if (mm.GetTiling(out) == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ChunkGatedDeltaRuleComputeWy(gert::TilingContext *context)
{
    ChunkGatedDeltaRuleComputeWyTilingData tiling;

    const auto qShape = context->GetInputShape(INPUT_Q_IDX)->GetStorageShape();
    const auto vShape = context->GetInputShape(INPUT_V_IDX)->GetStorageShape();
    const auto attrs = context->GetAttrs();

    const int64_t b = qShape.GetDim(DIM_B);
    const int64_t t = qShape.GetDim(DIM_T);
    const int64_t hk = qShape.GetDim(DIM_H);
    const int64_t kdim = qShape.GetDim(DIM_D);
    const int64_t hv = vShape.GetDim(DIM_H);
    const int64_t vdim = vShape.GetDim(DIM_D);
    const auto chunkSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX);
    if (chunkSizePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t chunkSize = *chunkSizePtr;

    if (chunkSize <= 0 || t <= 0 || hk <= 0 || hv <= 0 || kdim <= 0 || vdim <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (chunkSize != FIXED_CHUNK) {
        return ge::GRAPH_FAILED;
    }
    if ((t % chunkSize) != 0 || (hv % hk) != 0) {
        return ge::GRAPH_FAILED;
    }
    if ((kdim % 16) != 0 || (vdim % 16) != 0) {
        return ge::GRAPH_FAILED;
    }
    if (kdim > 128 || vdim > 128) {
        return ge::GRAPH_FAILED;
    }
    if (b > 32 || hv > 64) {
        return ge::GRAPH_FAILED;
    }

    const int64_t numChunks = t / chunkSize;
    const int64_t groupSize = hv / hk;
    const int64_t totalTasks = b * hv * numChunks;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t usedCoreNum = std::max(aicNum, aivNum);
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    context->SetBlockDim(usedCoreNum);

    // mmAttn: kBeta[64,K] @ K[64,K]^T -> [64,64]
    if (FillCubeTiling(context, FIXED_CHUNK, FIXED_CHUNK, kdim, /*bTranspose=*/true, tiling.mmAttn) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // mmU: attn[64,64] @ V[64,V] -> [64,V]
    if (FillCubeTiling(context, FIXED_CHUNK, vdim, FIXED_CHUNK, /*bTranspose=*/false, tiling.mmU) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // mmW: attn[64,64] @ kBetaExp[64,K] -> [64,K]
    if (FillCubeTiling(context, FIXED_CHUNK, kdim, FIXED_CHUNK, /*bTranspose=*/false, tiling.mmW) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_batch(b);
    tiling.set_seqlen(t);
    tiling.set_kNumHead(hk);
    tiling.set_vNumHead(hv);
    tiling.set_kHeadDim(kdim);
    tiling.set_vHeadDim(vdim);
    tiling.set_chunkSize(chunkSize);
    tiling.set_numChunks(numChunks);
    tiling.set_groupSize(groupSize);
    tiling.set_totalTasks(totalTasks);
    tiling.set_localWorkspaceSize(LOCAL_WORKSPACE_BYTES);
    tiling.set_perCoreWorkspaceBytes(PER_CORE_STAGING_BYTES);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_reserved0(0);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = static_cast<size_t>(SYS_WORKSPACE_SIZE) +
                   static_cast<size_t>(usedCoreNum) * static_cast<size_t>(PER_CORE_STAGING_BYTES);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkGatedDeltaRuleComputeWy(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleComputeWy)
    .Tiling(Tiling4ChunkGatedDeltaRuleComputeWy)
    .TilingParse<ChunkGatedDeltaRuleComputeWyCompileInfo>(TilingPrepareForChunkGatedDeltaRuleComputeWy);

} // namespace optiling
