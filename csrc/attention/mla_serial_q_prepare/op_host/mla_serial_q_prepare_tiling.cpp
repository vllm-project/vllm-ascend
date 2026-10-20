// SPDX-License-Identifier: Apache-2.0
#include "mla_serial_q_prepare_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* context)
{
    const auto* input = context->GetInputShape(0);
    if (input == nullptr || input->GetStorageShape().GetDimNum() != 2) return ge::GRAPH_FAILED;
    const auto& x = input->GetStorageShape();
    uint32_t tokens = x.GetDim(0);
    if (tokens == 0 || tokens > 1024 || context->GetInputDesc(0)->GetDataType() != ge::DT_BF16)
        return ge::GRAPH_FAILED;
    uint32_t rank = 0, heads = 1, nope = 0, headsPerTask = 1, useRope = 0, cacheBlocks = 0;
    uint64_t cacheStride = 0, ropeStride = 0;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const uint32_t cores = platform.GetCoreNumAiv();
    // Two row queues, one trig queue, two row FP32 vectors and two trig vectors.
    if (cores == 0 || ub < 128U*64U*(2U*2U+2U*4U)+2U*64U*(2U+4U)) return ge::GRAPH_FAILED;
    uint32_t tasks = tokens;
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr || attrs->GetInt(0) == nullptr || attrs->GetInt(1) == nullptr)
        return ge::GRAPH_FAILED;
    heads = *attrs->GetInt(0);
    if (heads == 0 || heads > 128 || x.GetDim(1)%heads) return ge::GRAPH_FAILED;
    nope = x.GetDim(1)/heads-64;
    if (nope < 16 || nope > 256 || nope%16) return ge::GRAPH_FAILED;
    useRope = *attrs->GetInt(1);
    headsPerTask = std::min(heads, std::min(8192U/nope, std::max(1U,(tokens*heads+cores-1)/cores)));
    tasks = tokens*((heads+headsPerTask-1)/headsPerTask);
    MlaSerialQPrepareTilingData t;
    t.set_tokens(tokens);
    t.set_rank(rank);
    t.set_heads(heads);
    t.set_nope(nope);
    t.set_headsPerTask(headsPerTask);
    t.set_useRope(useRope);
    t.set_cacheBlocks(cacheBlocks);
    t.set_cacheStride(cacheStride);
    t.set_ropeStride(ropeStride);
    t.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(t.GetDataSize());
    context->GetWorkspaceSizes(1)[0] = 0;
    context->SetBlockDim(std::min(tasks,cores));
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}
struct MlaSerialQPrepareCompileInfo {};
static ge::graphStatus Parse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(MlaSerialQPrepare).Tiling(Tiling).TilingParse<MlaSerialQPrepareCompileInfo>(Parse);
}
