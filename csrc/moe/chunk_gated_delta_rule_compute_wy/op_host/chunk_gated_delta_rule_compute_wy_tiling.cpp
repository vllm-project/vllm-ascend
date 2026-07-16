#include "chunk_gated_delta_rule_compute_wy_tiling.h"

#include <register/op_impl_registry.h>

namespace optiling {
static constexpr size_t INPUT_Q_IDX = 0;
static constexpr size_t INPUT_V_IDX = 2;
static constexpr size_t ATTR_CHUNK_SIZE_IDX = 0;

static constexpr size_t DIM_B = 0;
static constexpr size_t DIM_T = 1;
static constexpr size_t DIM_H = 2;
static constexpr size_t DIM_D = 3;

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
    if (chunkSize != 64) {
        return ge::GRAPH_FAILED;
    }
    if ((t % chunkSize) != 0 || (hv % hk) != 0) {
        return ge::GRAPH_FAILED;
    }
    if ((kdim % 16) != 0 || (vdim % 16) != 0) {
        return ge::GRAPH_FAILED;
    }
    // Current vector-only 310P kernel uses fixed local-memory budget for head dims
    // (per-task UB holds 64xK / 64xV float+half buffers plus attn 64x64).
    if (kdim > 128 || vdim > 128) {
        return ge::GRAPH_FAILED;
    }
    // B/Hv only affect task count and short staging for g/beta (64*Hv). Attn scratch
    // has 4096 floats, so Hv<=64 is safe; B is an independent outer index.
    if (b > 32 || hv > 64) {
        return ge::GRAPH_FAILED;
    }

    const int64_t numChunks = t / chunkSize;
    const int64_t groupSize = hv / hk;
    const int64_t totalTasks = b * hv * numChunks;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(ascendcPlatform.GetCoreNumAiv());

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

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
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
