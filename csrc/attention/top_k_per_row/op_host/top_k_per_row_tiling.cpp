/**
 * Copyright (c) 2026 SeaGaruda
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * Host tiling for TopKPerRow.
 *
 * The kernel runs one row per loop iteration on each AIV core and processes the
 * row window in chunks of `chunk_size` elements.  Each chunk is reduced to its
 * own top-k with the Ascend C high level TopK API; every `max_chunks` chunk
 * results are merged into a running candidate list of k entries (same API, one
 * row of `max_chunks * topk_pad` elements).
 *
 * Everything below only depends on shapes/attrs + the platform, never on tensor
 * values, so it is safe for graph mode / ACLGraph capture.
 */
#include "top_k_per_row_tiling.h"

#include <algorithm>

#include "log/ops_log.h"

namespace optiling {
namespace {
constexpr uint32_t FP32_KEY = 1;
constexpr uint32_t FP16_KEY = 2;
constexpr uint32_t BF16_KEY = 3;

constexpr uint32_t K_ALIGN = 32;           // TopK/Sort work on 32-element groups
constexpr uint32_t K_TOPK_MAX = 2048;
constexpr uint64_t RESERVED_UB_BYTES = 2048;
constexpr uint32_t K_MAX_CHUNKS = 4;       // merge width, 4 keeps the merge row small
constexpr uint32_t K_MAX_CHUNK_WIDTH = 32768;

uint32_t CeilAlign(uint32_t v, uint32_t a)
{
    return (v + a - 1) / a * a;
}

// UB bytes required by one row of work for a given (chunk, max_chunks).
//   raw input (T) + fp32 working copy + merge buffers (max_chunks + 1 slots)
//   + running candidates + TopK positions/byte-offsets + TopK tmp buffer.
uint64_t UbNeed(uint32_t chunk, uint32_t topkPad, uint32_t maxChunks, uint32_t dtypeSize, uint32_t tmpSize)
{
    const uint64_t srcRaw = static_cast<uint64_t>(chunk) * dtypeSize;
    const uint64_t srcFp32 = static_cast<uint64_t>(chunk) * sizeof(float);
    const uint64_t merge = static_cast<uint64_t>(maxChunks + 1) * topkPad * (sizeof(float) + sizeof(int32_t));
    const uint64_t cur = static_cast<uint64_t>(topkPad) * (sizeof(float) + sizeof(int32_t));
    const uint64_t pos = static_cast<uint64_t>(topkPad) * (sizeof(int32_t) + sizeof(uint32_t));
    const uint64_t dummy = static_cast<uint64_t>(std::max(chunk, (maxChunks + 1) * topkPad)) * sizeof(int32_t);
    return srcRaw + srcFp32 + merge + cur + pos + dummy + tmpSize;
}

ge::graphStatus Tiling4TopKPerRow(gert::TilingContext* context)
{
    const auto* logitsShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, logitsShapePtr);
    const gert::Shape& logitsShape = logitsShapePtr->GetStorageShape();
    OP_CHECK_IF(logitsShape.GetDimNum() != 2,
                OP_LOGE(context, "logits must be 2-D (num_rows, width)"), return ge::GRAPH_FAILED);

    const uint32_t numRows = static_cast<uint32_t>(logitsShape.GetDim(0));
    const uint32_t width = static_cast<uint32_t>(logitsShape.GetDim(1));
    OP_CHECK_IF(width == 0, OP_LOGE(context, "width must be > 0"), return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* topkPtr = attrs->GetInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkPtr);
    OP_CHECK_IF(*topkPtr <= 0 || *topkPtr > K_TOPK_MAX,
                OP_LOGE(context, "topk must be in (0, 2048]"), return ge::GRAPH_FAILED);
    const uint32_t topk = static_cast<uint32_t>(*topkPtr);
    const uint32_t topkPad = CeilAlign(topk, K_ALIGN);

    const auto dtype = context->GetInputDesc(0)->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16,
                OP_LOGE(context, "logits only supports float32/float16/bfloat16"), return ge::GRAPH_FAILED);
    const uint32_t dtypeSize = (dtype == ge::DT_FLOAT) ? 4U : 2U;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t ubBudget = (ubSize > RESERVED_UB_BYTES) ? (ubSize - RESERVED_UB_BYTES) : 0;

    // Probe the TopK tiling for the merge level first: it only depends on
    // (topk_pad, max_chunks) and bounds how wide the merge row may be.
    TopKPerRowTilingData tiling;
    uint32_t maxChunks = K_MAX_CHUNKS;
    uint32_t tmpSize = 0;
    uint32_t probeTmp = 0;
    while (maxChunks >= 1) {
        probeTmp = 0;
        const uint32_t mergeInner = (maxChunks + 1) * topkPad;  // +1 slot for the running candidates
        if (AscendC::TopKTilingFunc(platform, mergeInner, 1, topk, sizeof(float), false,
                                    AscendC::TopKMode::TOPK_NORMAL, true, tiling.topk_tiling_merge)) {
            uint32_t hi = 0;
            uint32_t lo = 0;
            if (AscendC::GetTopKMaxMinTmpSize(platform, mergeInner, 1, false, true,
                                              AscendC::TopKMode::TOPK_NORMAL, true, sizeof(float), hi, lo)) {
                probeTmp = hi;
                break;
            }
        }
        maxChunks >>= 1;
    }
    OP_CHECK_IF(maxChunks == 0, OP_LOGE(context, "TopK merge tiling is not feasible for this topk"),
                return ge::GRAPH_FAILED);
    tmpSize = probeTmp;

    // Largest chunk that still fits UB together with the merge level. A chunk
    // must be able to hold one full top-k (so one TopK call over the chunk is
    // well defined) and be K_ALIGN wide. Rows narrower than MIN_CHUNK -- down to
    // the single-column case -- are legal input, so the physical chunk is padded
    // up to K_ALIGN rather than rejected: LoadChunkToFp32 already pads every
    // chunk with -FLT_MAX up to chunk_size.
    uint32_t chunkMax = CeilAlign(std::min(width, K_MAX_CHUNK_WIDTH), K_ALIGN);
    const uint32_t chunkFloor = std::max(K_ALIGN, topkPad);
    uint32_t chunk = (chunkMax < chunkFloor) ? chunkFloor : chunkMax;
    bool chunkOk = false;
    while (chunk >= chunkFloor && chunk != 0U) {
        const uint64_t need = UbNeed(chunk, topkPad, maxChunks, dtypeSize, tmpSize);
        if (need <= ubBudget) {
            uint32_t hi = 0;
            uint32_t lo = 0;
            if (AscendC::TopKTilingFunc(platform, chunk, 1, std::min(topk, chunk), sizeof(float), false,
                                        AscendC::TopKMode::TOPK_NORMAL, true, tiling.topk_tiling_chunk) &&
                AscendC::GetTopKMaxMinTmpSize(platform, chunk, 1, false, true,
                                              AscendC::TopKMode::TOPK_NORMAL, true, sizeof(float), hi, lo)) {
                tmpSize = std::max(tmpSize, hi);
                // recompute with the final tmp size before accepting the chunk
                if (UbNeed(chunk, topkPad, maxChunks, dtypeSize, tmpSize) <= ubBudget) {
                    chunkOk = true;
                    break;
                }
            }
        }
        chunk >>= 1;
        chunk = chunk / K_ALIGN * K_ALIGN;
    }
    OP_CHECK_IF(!chunkOk, OP_LOGE(context, "no feasible chunk size for TopKPerRow on this platform"),
                return ge::GRAPH_FAILED);

    const uint32_t coreNum = platform.GetCoreNumAiv();
    const uint32_t usedCore = (numRows == 0) ? 1U : std::min(numRows, coreNum);
    context->SetBlockDim(usedCore);
    context->SetTilingKey(dtype == ge::DT_FLOAT ? FP32_KEY : (dtype == ge::DT_FLOAT16 ? FP16_KEY : BF16_KEY));

    tiling.set_num_rows(numRows);
    tiling.set_width(width);
    tiling.set_topk(topk);
    tiling.set_topk_pad(topkPad);
    tiling.set_rows_per_core((numRows + usedCore - 1) / usedCore);
    tiling.set_chunk_size(chunk);
    tiling.set_max_chunks(maxChunks);
    tiling.set_num_round((width + chunk - 1) / chunk);
    tiling.set_tmp_size(tmpSize);
    tiling.set_dtype_size(dtypeSize);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // TopK keeps its workspace inside UB; only an empty workspace is declared so
    // that graph mode still sees a valid (zero sized) workspace request.
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(TopKPerRow).Tiling(Tiling4TopKPerRow);
}  // namespace optiling
