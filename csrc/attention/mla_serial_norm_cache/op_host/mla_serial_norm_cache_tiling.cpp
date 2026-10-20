// SPDX-License-Identifier: Apache-2.0
#include "mla_serial_norm_cache_tiling.h"
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
    // Three b16 queues and five fp32 work vectors, each bounded by4096 elements.
    if (cores == 0 || ub < 4096U*(3U*2U+5U*4U)) return ge::GRAPH_FAILED;
    uint32_t tasks = tokens;
    rank = x.GetDim(1)-576;
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr || attrs->GetInt(0) == nullptr || attrs->GetInt(1) == nullptr ||
        attrs->GetInt(2) == nullptr || attrs->GetInt(3) == nullptr) return ge::GRAPH_FAILED;
    cacheStride = *attrs->GetInt(0); ropeStride = *attrs->GetInt(1);
    cacheBlocks = *attrs->GetInt(2); useRope = *attrs->GetInt(3);
    if (rank == 0 || rank > 4096 || rank%32 || cacheBlocks == 0 ||
        cacheStride < 128*512 || ropeStride < 128*64) return ge::GRAPH_FAILED;
    const auto* gammaQ=context->GetInputShape(5);
    const auto* gammaKv=context->GetInputShape(1);
    if(gammaQ==nullptr || gammaKv==nullptr || gammaQ->GetStorageShape().GetDimNum()!=1 ||
       gammaKv->GetStorageShape().GetDimNum()!=1 || gammaQ->GetStorageShape().GetDim(0)!=rank ||
       gammaKv->GetStorageShape().GetDim(0)!=512) return ge::GRAPH_FAILED;
    MlaSerialNormCacheTilingData t;
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
struct MlaSerialNormCacheCompileInfo {};
static ge::graphStatus Parse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(MlaSerialNormCache).Tiling(Tiling).TilingParse<MlaSerialNormCacheCompileInfo>(Parse);
}
