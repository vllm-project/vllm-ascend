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
// 310P dav_m200 UB is 192KB. K=V=128 fits since the kernel solves W and U in two
// passes with a single fp32 RHS resident (~178KB peak); see compute_wy_kernel.h.
static constexpr int64_t MAX_HEAD_DIM = 128;
// 32KB: the matmul lib stages VECCALC operands through this workspace; a
// 128-wide apply needs 16KB for B alone. 8KB (a prior shrink) silently
// overruns at head dim 128 and emits NaN — K=64 fit exactly and stayed green,
// which hid the breakage for a day.
static constexpr uint32_t LOCAL_WORKSPACE_BYTES = 32 * 1024;
// Per-core GM staging for Cube inputs: A_half(64*MAX_HEAD_DIM) + B_half(64*MAX_HEAD_DIM).
static constexpr uint32_t STAGING_A_BYTES = FIXED_CHUNK * MAX_HEAD_DIM * sizeof(uint16_t);
static constexpr uint32_t STAGING_B_BYTES = FIXED_CHUNK * MAX_HEAD_DIM * sizeof(uint16_t);
// A_half + B_half staging + fp32 A-snapshot slot for the two-pass solve.
static constexpr uint32_t SNAP_BYTES = FIXED_CHUNK * FIXED_CHUNK * sizeof(float);
static constexpr uint32_t PER_CORE_STAGING_BYTES = STAGING_A_BYTES + STAGING_B_BYTES + SNAP_BYTES;

static ge::graphStatus FillCubeTiling(gert::TilingContext *context, int64_t m, int64_t n, int64_t k, bool bTranspose,
                                      TCubeTiling &out, bool abFromUb = false)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    const auto abPos = abFromUb ? matmul_tiling::TPosition::VECCALC : matmul_tiling::TPosition::GM;
    mm.SetAType(abPos, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                false);
    mm.SetBType(abPos, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                bTranspose);
    mm.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT);
    mm.SetBias(false);
    mm.SetOrgShape(m, n, k);
    // Cap the tiled block at the 64-wide chunk: on 310P a cube base dim of 128 is
    // mis-driven by this kernel (baseN=128 -> wrong results, baseK=128 -> aicore timeout).
    // OrgShape keeps the real sizes so strides are right; the kernel re-sets
    // SetOrgShape/SetSingleShape at runtime and the matmul iterates internally.
    mm.SetShape(std::min<int64_t>(m, FIXED_CHUNK), std::min<int64_t>(n, FIXED_CHUNK), std::min<int64_t>(k, FIXED_CHUNK));
    mm.SetBufferSpace(-1, -1, -1);
    mm.SetFixSplit(FIXED_CHUNK, FIXED_CHUNK, FIXED_CHUNK);
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
    if (kdim > MAX_HEAD_DIM || vdim > MAX_HEAD_DIM) {
        return ge::GRAPH_FAILED;
    }
    if (b > 32 || hv > 64) {
        return ge::GRAPH_FAILED;
    }

    const int64_t numChunks = t / chunkSize;
    const int64_t groupSize = hv / hk;
    const int64_t totalTasks = b * hv * numChunks;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t usedCoreNum = ascendcPlatform.GetCoreNumAic();
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    context->SetBlockDim(usedCoreNum);

    // All cube work runs as <=64-wide tiles: the kernel slices K and N itself and
    // re-sets OrgShape/SingleShape per call. Tilings generated at any dim above 64
    // wedge MatmulImpl::Init on the device (bisected: Init alone hangs at K=128),
    // so every tiling is generated for the 64^3 tile.
    // mmAttn: kBeta[64,<=64] @ K[64,<=64]^T -> [64,64], K-slices accumulated in UB.
    if (FillCubeTiling(context, FIXED_CHUNK, FIXED_CHUNK, FIXED_CHUNK, /*bTranspose=*/true, tiling.mmAttn,
                       /*abFromUb=*/true) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // mmSquare: P[64,64] @ P[64,64] -> [64,64]
    if (FillCubeTiling(context, FIXED_CHUNK, FIXED_CHUNK, FIXED_CHUNK, /*bTranspose=*/false, tiling.mmSquare) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // mmApplyU / mmApplyW: P/T[64,64] @ R[64,<=64] column slices, operands fed
    // straight from UB (no GM staging round-trip).
    if (FillCubeTiling(context, FIXED_CHUNK, FIXED_CHUNK, FIXED_CHUNK, /*bTranspose=*/false, tiling.mmApplyU,
                       /*abFromUb=*/true) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (FillCubeTiling(context, FIXED_CHUNK, FIXED_CHUNK, FIXED_CHUNK, /*bTranspose=*/false, tiling.mmApplyW,
                       /*abFromUb=*/true) != ge::GRAPH_SUCCESS) {
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
    const uint64_t workspaceOffset = ascendcPlatform.GetLibApiWorkSpaceSize();
    tiling.set_workspaceOffset(workspaceOffset);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = static_cast<size_t>(workspaceOffset) +
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
