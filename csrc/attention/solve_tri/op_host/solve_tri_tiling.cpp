/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
 #include "solve_tri_tiling.h"
 #include "register/op_impl_registry.h"
 #include "tiling/platform/platform_ascendc.h"
 #include <string>
 
 namespace optiling {
 
 constexpr uint32_t INPUT_X_IDX = 0;
 constexpr uint32_t INPUT_CU_SEQLENS_IDX = 1;
 constexpr uint32_t INPUT_CHUNK_INDICES_IDX = 2;
 constexpr uint32_t OUTPUT_X_OUT_IDX = 0;
constexpr uint32_t ATTR_LAYOUT_IDX = 0;
 
 static ge::graphStatus SolveTriTilingFunc(gert::TilingContext* context)
 {
     auto platformInfo = context->GetPlatformInfo();
     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
     int64_t coreNum = ascendcPlatform.GetCoreNumAic();
     if (coreNum == 0) return ge::GRAPH_FAILED;
 
     // Get input shape: [B, H, T, BT] (BHTD), [B, T, H, BT] (BSND), or [total_T, H, BT] (THD)
     auto inputShape = context->GetInputShape(INPUT_X_IDX);
     if (inputShape == nullptr) return ge::GRAPH_FAILED;
     auto shape = inputShape->GetStorageShape();
     int64_t ndim = shape.GetDimNum();
     if (ndim != 3 && ndim != 4) return ge::GRAPH_FAILED;

     // Get layout attribute to determine shape parsing
     auto attrs = context->GetAttrs();
     const char *layoutStr = attrs->GetStr(ATTR_LAYOUT_IDX);
     std::string layout = layoutStr ? layoutStr : "bsnd";

     // layoutMode: 0=BHTD, 1=BSND, 2=THD
     int64_t layoutMode = 1;  // default to BSND
     if (layout == "bhtd") {
         layoutMode = 0;
     } else if (layout == "bsnd") {
         layoutMode = 1;
     } else if (layout == "tnd") {
         layoutMode = 2;
     }

     int64_t B, H, T, BT;
     if (ndim == 4) {
         if (layoutMode == 0) {
             // BHTD: [B, H, T, BT]
             B = shape.GetDim(0);
             H = shape.GetDim(1);
             T = shape.GetDim(2);
             BT = shape.GetDim(3);
         } else {
             // BSND: [B, T, H, BT]
             B = shape.GetDim(0);
             T = shape.GetDim(1);
             H = shape.GetDim(2);
             BT = shape.GetDim(3);
         }
     } else {
         // 3D THD: [total_T, H, BT]
         B = 1;
         T = shape.GetDim(0);
         H = shape.GetDim(1);
         BT = shape.GetDim(2);
     }

     int64_t chunkSize = BT;

    // isVarlen only for THD mode
    int64_t isVarlen = (layoutMode == 2) ? 1 : 0;
    int64_t hasCuSeqlens = isVarlen;

    int64_t totalChunks = 0;
    int64_t numChunks = 0;
    int64_t totalTiles = 0;
    int64_t lastChunkValidSize = 0;

    if (isVarlen) {
        auto chunkIndicesShape = context->GetInputShape(INPUT_CHUNK_INDICES_IDX);
        // chunk_indices 是扁平 1D tensor: [seq0, chunk0, seq1, chunk1, ...]
        // 元素数 = total_chunks * 2
        int64_t chunkIndicesLen = chunkIndicesShape->GetStorageShape().GetDim(0);
        totalChunks = chunkIndicesLen / 2;
        totalTiles = totalChunks * H;
        numChunks = 0;
        lastChunkValidSize = 0;
    } else {
        totalChunks = 0;
        numChunks = (T + chunkSize - 1) / chunkSize;
        totalTiles = B * numChunks * H;
        int64_t remainder = T % chunkSize;
        lastChunkValidSize = (remainder == 0) ? chunkSize : remainder;
    }

    int64_t tilesPerCore = (totalTiles + coreNum - 1) / coreNum;

    // Get input dtype: 0=fp16, 1=bf16
    auto inputDtype = context->GetInputDesc(INPUT_X_IDX)->GetDataType();
    int64_t dtypeMode = 0;  // default fp16
    if (inputDtype == ge::DT_BF16) {
        dtypeMode = 1;
    }

    // Set tiling data
    SolveTriTilingData tiling;
    tiling.set_totalTiles(totalTiles);
    tiling.set_matrixSize(chunkSize);
    tiling.set_numHeads(H);
    tiling.set_seqLen(T);
    tiling.set_batchSize(B);
    tiling.set_isLower(1);
    tiling.set_hasCuSeqlens(hasCuSeqlens);
    tiling.set_tilesPerCore(tilesPerCore);
    tiling.set_chunkSize(chunkSize);
    tiling.set_numChunks(numChunks);
    tiling.set_lastChunkValidSize(lastChunkValidSize);
    tiling.set_isVarlen(isVarlen);
    tiling.set_totalChunks(totalChunks);
    tiling.set_layoutMode(layoutMode);
    tiling.set_dtypeMode(dtypeMode);
 
     context->SetTilingKey(1);
     tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                         context->GetRawTilingData()->GetCapacity());
     context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
 
     int64_t usedCoreNum = (totalTiles + tilesPerCore - 1) / tilesPerCore;
     if (usedCoreNum > coreNum) usedCoreNum = coreNum;
     context->SetBlockDim(usedCoreNum);
 
     // Workspace: 用于存储辅助矩阵 (-I, ZERO, +I) + 每核中转缓冲区
     // 需要加上系统 workspace 大小
     uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
     size_t sharedSize = 3 * chunkSize * chunkSize * sizeof(uint16_t);  // I + -I + ZERO
     size_t perCoreSize = 2 * chunkSize * chunkSize * sizeof(uint16_t);  // 每核 2 个中转区（X 流 + Y 流双缓冲）
     size_t userWorkspaceSize = sharedSize + usedCoreNum * perCoreSize;
     // 对齐到 512 字节
     userWorkspaceSize = ((userWorkspaceSize + 511) / 512) * 512;
     // 总 workspace = 用户 workspace + 系统 workspace
     size_t* ws = context->GetWorkspaceSizes(1);
     ws[0] = userWorkspaceSize + sysWorkspaceSize;
 
     return ge::GRAPH_SUCCESS;
 }
 
 static ge::graphStatus SolveTriTilingParse(gert::TilingParseContext* context)
 {
     return ge::GRAPH_SUCCESS;
 }
 
 struct SolveTriCompileInfo {};
 
 IMPL_OP_OPTILING(SolveTri)
     .Tiling(SolveTriTilingFunc)
     .TilingParse<SolveTriCompileInfo>(SolveTriTilingParse);
 
 }  // namespace optiling
 