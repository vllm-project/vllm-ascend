// SPDX-License-Identifier: Apache-2.0
#include "mla_serial_q_finalize_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* context)
{
    constexpr uint32_t QUEUE_ELEMENTS=8192;
    constexpr uint32_t MAX_TILE_CELLS=128;
    constexpr uint64_t REQUIRED_UB_BYTES=180992;
    const auto* input=context->GetInputShape(0);
    const auto* attrs=context->GetAttrs();
    if(input==nullptr || input->GetStorageShape().GetDimNum()!=2 || attrs==nullptr || attrs->GetInt(0)==nullptr)
        return ge::GRAPH_FAILED;
    const auto& x=input->GetStorageShape();
    const int64_t h=*attrs->GetInt(0), width=x.GetDim(1);
    if(x.GetDim(0)<1 || x.GetDim(0)>1024 || h<1 || h>128 || width%h || width/h<=64)
        return ge::GRAPH_FAILED;
    const uint32_t tokens=x.GetDim(0),heads=h,nope=width/h-64;
    if(nope<16 || nope>256 || nope%16) return ge::GRAPH_FAILED;
    uint32_t useRope=0;
    const auto* flag=attrs->GetInt(1);
    const auto* key=context->GetInputShape(1);
    if(flag==nullptr || (*flag!=0 && *flag!=1) || key==nullptr) return ge::GRAPH_FAILED;
    const auto& k=key->GetStorageShape();
    if(k.GetDimNum()!=3 || k.GetDim(0)!=heads || k.GetDim(1)!=tokens || k.GetDim(2)!=512)
        return ge::GRAPH_FAILED;
    useRope=*flag;
    auto platform=platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub=0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    const uint32_t cores=platform.GetCoreNumAiv();
    if(cores==0 || ub<REQUIRED_UB_BYTES) return ge::GRAPH_FAILED;
    constexpr uint32_t TOKENS_PER_TILE=2;
    const uint32_t tileTokens=std::min(tokens,TOKENS_PER_TILE);
    const uint32_t tileHeads=std::min(heads,MAX_TILE_CELLS/tileTokens);
    const uint32_t tasks=((tokens+tileTokens-1)/tileTokens)*((heads+tileHeads-1)/tileHeads);
    MlaSerialQFinalizeTilingData data;
    data.set_tokens(tokens); data.set_heads(heads); data.set_nope(nope);
    data.set_tileHeads(tileHeads); data.set_tileTokens(tileTokens); data.set_useRope(useRope);
    data.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    context->GetWorkspaceSizes(1)[0]=0;
    context->SetBlockDim(std::min(tasks,cores));
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}
struct MlaSerialQFinalizeCompileInfo {};
static ge::graphStatus Parse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(MlaSerialQFinalize).Tiling(Tiling).TilingParse<MlaSerialQFinalizeCompileInfo>(Parse);
}
