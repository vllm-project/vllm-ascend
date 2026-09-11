#include "chunk_kda_fwd_tiling.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <register/op_impl_registry.h>
#include "arch35/chunk_kda_fwd_tiling_impl.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Q_IDX = 0;
constexpr size_t INPUT_V_IDX = 2;
constexpr size_t INPUT_G_IDX = 3;
constexpr size_t INPUT_A_LOG_IDX = 5;
constexpr size_t INPUT_DT_BIAS_IDX = 6;
constexpr size_t INPUT_INITIAL_STATE_IDX = 7;
constexpr size_t INPUT_CU_SEQLENS_IDX = 8;
constexpr size_t INPUT_CHUNK_INDICES_IDX = 9;

constexpr size_t OUTPUT_FINAL_STATE_IDX = 1;
constexpr size_t OUTPUT_GK_IDX = 2;
constexpr size_t OUTPUT_W_IDX = 5;
constexpr size_t OUTPUT_U_IDX = 6;
constexpr size_t OUTPUT_QG_IDX = 7;
constexpr size_t OUTPUT_KG_IDX = 8;
constexpr size_t OUTPUT_V_NEW_IDX = 9;
constexpr size_t OUTPUT_H_IDX = 10;

constexpr size_t ATTR_LAYOUT_IDX = 0;
constexpr size_t ATTR_SCALE_IDX = 1;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 2;
constexpr size_t ATTR_SAFE_GATE_IDX = 3;
constexpr size_t ATTR_LOWER_BOUND_IDX = 4;
constexpr size_t ATTR_USE_GATE_IDX = 5;
constexpr size_t ATTR_STAGE_IDX = 7;
constexpr int64_t KDA_STAGE_FULL = -1;
constexpr int64_t KDA_STAGE_FINALIZE = 3;

constexpr uint64_t KDA_ALIGN = 512;
constexpr uint64_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint64_t KDA_SOLVE_PIPELINE_DEPTH = 4;
constexpr uint64_t KDA_SCORE_QUEUE_SLOTS = 4;
constexpr uint64_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint64_t KDA_GDN_PIPELINE_DEPTH = 2;
constexpr uint32_t KDA_BATCH_MODE = 1;

uint64_t AlignWorkspace(uint64_t bytes)
{
    return (bytes + KDA_ALIGN - 1) / KDA_ALIGN * KDA_ALIGN;
}

uint64_t AllocateWorkspace(uint64_t &cursor, uint64_t bytes)
{
    const uint64_t offset = AlignWorkspace(cursor);
    cursor = offset + bytes;
    return offset;
}

bool HasOutput(gert::TilingContext *context, size_t index)
{
    const auto instanceInfo = context->GetIrOutputInstanceInfo(index);
    if (instanceInfo == nullptr || instanceInfo->GetInstanceNum() == 0) {
        return false;
    }
    const auto outputShape = context->GetOutputShape(instanceInfo->GetInstanceStart());
    return outputShape != nullptr &&
        outputShape->GetStorageShape().GetShapeSize() != 1;
}

struct ShapeInfo {
    int64_t rank = 0;
    int64_t batch = 0;
    int64_t seqlen = 0;
    int64_t qHeads = 0;
    int64_t vHeads = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
    bool sequenceMajor = false;
};

bool ResolveShape(gert::TilingContext *context, const char *layout, ShapeInfo &info)
{
    const auto qShapePtr = context->GetInputShape(INPUT_Q_IDX);
    const auto vShapePtr = context->GetInputShape(INPUT_V_IDX);
    if (qShapePtr == nullptr || vShapePtr == nullptr || layout == nullptr) {
        return false;
    }
    const auto &qShape = qShapePtr->GetStorageShape();
    const auto &vShape = vShapePtr->GetStorageShape();
    info.rank = qShape.GetDimNum();
    if (info.rank != vShape.GetDimNum() || (info.rank != 3 && info.rank != 4)) {
        return false;
    }

    info.sequenceMajor = std::strcmp(layout, "BSND") == 0 || std::strcmp(layout, "TND") == 0;
    if (info.rank == 4) {
        info.batch = qShape.GetDim(0);
        if (info.sequenceMajor) {
            info.seqlen = qShape.GetDim(1);
            info.qHeads = qShape.GetDim(2);
            info.vHeads = vShape.GetDim(2);
        } else {
            info.qHeads = qShape.GetDim(1);
            info.vHeads = vShape.GetDim(1);
            info.seqlen = qShape.GetDim(2);
        }
        info.kDim = qShape.GetDim(3);
        info.vDim = vShape.GetDim(3);
    } else {
        info.batch = 1;
        if (info.sequenceMajor) {
            info.seqlen = qShape.GetDim(0);
            info.qHeads = qShape.GetDim(1);
            info.vHeads = vShape.GetDim(1);
        } else {
            info.qHeads = qShape.GetDim(0);
            info.vHeads = vShape.GetDim(0);
            info.seqlen = qShape.GetDim(1);
        }
        info.kDim = qShape.GetDim(2);
        info.vDim = vShape.GetDim(2);
    }
    return info.batch > 0 && info.seqlen > 0 && info.qHeads > 0 && info.vHeads > 0 &&
           info.kDim > 0 && info.vDim > 0 && info.vHeads % info.qHeads == 0;
}

bool ResolveSequenceInfo(gert::TilingContext *context, int64_t seqlen, int64_t chunkSize,
                         int64_t batch, bool &isVarLen, int64_t &seqNum,
                         int64_t &totalChunks)
{
    const auto cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX);
    isVarLen = cuTensor != nullptr;
    seqNum = batch;
    totalChunks = (seqlen + chunkSize - 1) / chunkSize;
    if (!isVarLen) {
        return totalChunks > 0;
    }

    seqNum = cuTensor->GetStorageShape().GetDim(0) - 1;
    const int64_t *cu = cuTensor->GetData<int64_t>();
    if (seqNum <= 0 || cu == nullptr || cu[0] != 0 || cu[seqNum] > seqlen) {
        return false;
    }
    totalChunks = 0;
    for (int64_t seq = 0; seq < seqNum; ++seq) {
        if (cu[seq] < 0 || cu[seq + 1] < cu[seq]) {
            return false;
        }
        totalChunks += (cu[seq + 1] - cu[seq] + chunkSize - 1) / chunkSize;
    }

    const auto chunkShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
    if (chunkShape != nullptr &&
        chunkShape->GetStorageShape().GetShapeSize() != totalChunks * 2) {
        return false;
    }
    return totalChunks > 0;
}
} // namespace

ge::graphStatus Tiling4ChunkKdaFwd(gert::TilingContext *context)
{
    const auto qDesc = context->GetInputDesc(INPUT_Q_IDX);
    const auto gDesc = context->GetInputDesc(INPUT_G_IDX);
    const auto attrs = context->GetAttrs();
    if (qDesc == nullptr || gDesc == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const char *layout = attrs->GetStr(ATTR_LAYOUT_IDX);
    const float scale = static_cast<float>(*attrs->GetAttrPointer<double>(ATTR_SCALE_IDX));
    const int64_t chunkSize = *attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX);
    const bool safeGate = *attrs->GetAttrPointer<bool>(ATTR_SAFE_GATE_IDX);
    const float lowerBound = *attrs->GetAttrPointer<float>(ATTR_LOWER_BOUND_IDX);
    const bool useGateInKernel = *attrs->GetAttrPointer<bool>(ATTR_USE_GATE_IDX);
    const int64_t stage = *attrs->GetAttrPointer<int64_t>(ATTR_STAGE_IDX);
    if (chunkSize <= 0 || stage < KDA_STAGE_FULL ||
        stage > KDA_STAGE_FINALIZE) {
        return ge::GRAPH_FAILED;
    }

    ShapeInfo shape;
    if (!ResolveShape(context, layout, shape)) {
        return ge::GRAPH_FAILED;
    }
    bool isVarLen = false;
    int64_t seqNum = 0;
    int64_t totalChunks = 0;
    if (!ResolveSequenceInfo(context, shape.seqlen, chunkSize, shape.batch,
                             isVarLen, seqNum, totalChunks)) {
        return ge::GRAPH_FAILED;
    }

    const bool hasALog = context->GetOptionalInputDesc(INPUT_A_LOG_IDX) != nullptr;
    const bool hasDtBias = context->GetOptionalInputDesc(INPUT_DT_BIAS_IDX) != nullptr;
    const bool hasInitialState = context->GetOptionalInputDesc(INPUT_INITIAL_STATE_IDX) != nullptr;
    const bool storeFinalState = HasOutput(context, OUTPUT_FINAL_STATE_IDX);
    const bool storeGk = HasOutput(context, OUTPUT_GK_IDX);
    const bool storeW = HasOutput(context, OUTPUT_W_IDX);
    const bool storeU = HasOutput(context, OUTPUT_U_IDX);
    const bool storeQG = HasOutput(context, OUTPUT_QG_IDX);
    const bool storeKg = HasOutput(context, OUTPUT_KG_IDX);
    const bool storeVNew = HasOutput(context, OUTPUT_V_NEW_IDX);
    const bool storeH = HasOutput(context, OUTPUT_H_IDX);

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t blockDim = std::max<uint32_t>(platform.GetCoreNumAic(), 1);
    const bool isAscend950 =
        platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    const bool useChunk64K128V128Template =
        chunkSize == 64 && shape.kDim == 128 && shape.vDim == 128;
    const auto arch35Options = arch35::ConfigureChunkKdaFwdArch35(
        isAscend950, qDesc->GetDataType() == ge::DT_BF16,
        gDesc->GetDataType() == ge::DT_FLOAT, hasALog, useGateInKernel,
        safeGate, isVarLen, shape.seqlen, shape.vHeads, chunkSize,
        shape.kDim, shape.vDim, storeQG, storeVNew, storeH);

    const uint64_t dataBytes =
        qDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const uint64_t tokenHeads = static_cast<uint64_t>(shape.batch) *
        shape.vHeads * shape.seqlen;
    const uint64_t kTensorBytes = tokenHeads * shape.kDim * dataBytes;
    const uint64_t vTensorBytes = tokenHeads * shape.vDim * dataBytes;
    const uint64_t gkBytes = tokenHeads * shape.kDim * sizeof(float);
    const uint64_t stateElements = static_cast<uint64_t>(seqNum) *
        shape.vHeads * shape.kDim * shape.vDim;
    const uint64_t hChunkCount = isVarLen
        ? static_cast<uint64_t>(totalChunks)
        : static_cast<uint64_t>(shape.batch) * totalChunks;
    const uint64_t hBytes = hChunkCount * shape.vHeads * shape.kDim *
        shape.vDim * dataBytes;

    uint64_t cursor = 0;
    const uint64_t gkStorageOffset = storeGk ? 0 : AllocateWorkspace(cursor, gkBytes);
    const uint64_t finalStateStorageOffset = storeFinalState ? 0 :
        AllocateWorkspace(cursor, stateElements * sizeof(float));
    const uint64_t wStorageOffset = storeW ? 0 : AllocateWorkspace(cursor, kTensorBytes);
    const uint64_t uStorageOffset = storeU ? 0 : AllocateWorkspace(cursor, vTensorBytes);
    const uint64_t qgStorageOffset = storeQG ? 0 : AllocateWorkspace(cursor, kTensorBytes);
    const uint64_t kgStorageOffset = storeKg ? 0 : AllocateWorkspace(cursor, kTensorBytes);
    const uint64_t vNewStorageBytes =
        arch35Options.useDenseFwdH && !storeVNew
            ? static_cast<uint64_t>(shape.batch) * shape.vHeads * chunkSize *
                  shape.vDim * dataBytes
            : vTensorBytes;
    const uint64_t vNewStorageOffset = storeVNew ? 0 :
        AllocateWorkspace(cursor, vNewStorageBytes);
    const uint64_t hStorageBytes =
        arch35Options.useDenseFwdH && !storeH
            ? static_cast<uint64_t>(shape.batch) * shape.vHeads * shape.kDim *
                  shape.vDim * dataBytes
            : hBytes;
    const uint64_t hStorageOffset = storeH ? 0 : AllocateWorkspace(cursor, hStorageBytes);
    const uint64_t qgScaledOffset = AllocateWorkspace(cursor, kTensorBytes);

    const uint64_t matrixBytes = tokenHeads * chunkSize * sizeof(float);
    const uint64_t prepareAqkFp32Offset = AllocateWorkspace(cursor, matrixBytes);
    const uint64_t prepareAkkFp32Offset = AllocateWorkspace(cursor, matrixBytes);
    const uint64_t prepareScratchOffset = AlignWorkspace(cursor);
    const uint64_t solveDepth = safeGate ? KDA_SOLVE_PIPELINE_DEPTH : 1;
    const uint64_t solveBytes = static_cast<uint64_t>(blockDim) * solveDepth *
        KDA_SOLVE_SCRATCH_SLOTS * chunkSize * chunkSize * sizeof(float);
    const uint64_t scoreBytes = static_cast<uint64_t>(blockDim) *
        KDA_SCORE_QUEUE_SLOTS * KDA_SCORE_SCRATCH_PLANES * chunkSize *
        shape.kDim * dataBytes;
    cursor = prepareScratchOffset + AlignWorkspace(solveBytes) + scoreBytes;

    const uint64_t postWuScratchOffset = AlignWorkspace(cursor);
    if (!arch35Options.fusePostWu && !arch35Options.fusePostWuIntoFwdH) {
        cursor = postWuScratchOffset + tokenHeads * shape.kDim * sizeof(float);
    }

    const uint64_t fwdHWorkspaceBaseOffset = AlignWorkspace(cursor);
    uint64_t fwdHCursor = 0;
    const uint64_t vWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * chunkSize * shape.vDim *
                        sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t vUpdateWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * chunkSize * shape.vDim *
                        sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t kDecayWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * chunkSize * shape.kDim *
                        sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t hWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, static_cast<uint64_t>(blockDim) * shape.kDim * shape.vDim *
                        sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t tokenBatch = isVarLen ? static_cast<uint64_t>(seqNum) : 1;
    const uint64_t numSeqWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, (tokenBatch + 1) * sizeof(int64_t));
    const uint64_t numChunksWorkspaceOffset = AllocateWorkspace(
        fwdHCursor, (tokenBatch + 1) * sizeof(int64_t));
    cursor = fwdHWorkspaceBaseOffset + AlignWorkspace(fwdHCursor);

    const uint64_t outputScratchOffset = AllocateWorkspace(
        cursor, 2 * tokenHeads * shape.vDim * sizeof(float));
    const uint64_t totalWorkspace = AlignWorkspace(cursor);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(useChunk64K128V128Template ? 2 : 1);
    context->SetScheduleMode(KDA_BATCH_MODE);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + totalWorkspace;

    ChunkKdaFwdTilingData tiling;
    tiling.set_batch(shape.batch);
    tiling.set_seqNum(seqNum);
    tiling.set_qHeadNum(shape.qHeads);
    tiling.set_vHeadNum(shape.vHeads);
    tiling.set_seqlen(shape.seqlen);
    tiling.set_kHeadDim(shape.kDim);
    tiling.set_vHeadDim(shape.vDim);
    tiling.set_chunkSize(chunkSize);
    tiling.set_totalChunks(totalChunks);
    tiling.set_inputRank(shape.rank);
    tiling.set_scale(scale);
    tiling.set_lowerBound(lowerBound);
    tiling.set_hasInitialState(hasInitialState);
    tiling.set_isVarLen(isVarLen);
    tiling.set_safeGate(safeGate);
    tiling.set_inputSequenceMajor(shape.sequenceMajor);
    tiling.set_useGateInKernel(useGateInKernel);
    tiling.set_hasALog(hasALog);
    tiling.set_hasDtBias(hasDtBias);
    tiling.set_computeGateInPrepare(arch35Options.computeGateInPrepare);
    tiling.set_fusePostWu(arch35Options.fusePostWu);
    tiling.set_fusePostWuIntoFwdH(arch35Options.fusePostWuIntoFwdH);
    tiling.set_useDenseFwdH(arch35Options.useDenseFwdH);
    tiling.set_storeFinalState(storeFinalState);
    tiling.set_storeGk(storeGk);
    tiling.set_storeW(storeW);
    tiling.set_storeU(storeU);
    tiling.set_storeQG(storeQG);
    tiling.set_storeKg(storeKg);
    tiling.set_storeVNew(storeVNew);
    tiling.set_storeH(storeH);
    tiling.set_stage(stage);
    tiling.set_gateDataType(gDesc->GetDataType() == ge::DT_FLOAT ? 2 :
        (gDesc->GetDataType() == ge::DT_BF16 ? 1 : 0));
    tiling.set_gateUsedCoreNum(static_cast<int64_t>(blockDim) * 2);
    tiling.set_prepareUsedCoreNum(blockDim);
    tiling.set_postWuUsedCoreNum(blockDim);
    tiling.set_outputUsedCoreNum(blockDim);
    tiling.set_gkStorageOffset(gkStorageOffset);
    tiling.set_finalStateStorageOffset(finalStateStorageOffset);
    tiling.set_wStorageOffset(wStorageOffset);
    tiling.set_uStorageOffset(uStorageOffset);
    tiling.set_qgStorageOffset(qgStorageOffset);
    tiling.set_kgStorageOffset(kgStorageOffset);
    tiling.set_vNewStorageOffset(vNewStorageOffset);
    tiling.set_hStorageOffset(hStorageOffset);
    tiling.set_qgScaledOffset(qgScaledOffset);
    tiling.set_prepareAqkFp32Offset(prepareAqkFp32Offset);
    tiling.set_prepareAkkFp32Offset(prepareAkkFp32Offset);
    tiling.set_prepareScratchOffset(prepareScratchOffset);
    tiling.set_postWuScratchOffset(postWuScratchOffset);
    tiling.set_outputScratchOffset(outputScratchOffset);
    tiling.set_fwdHWorkspaceBaseOffset(fwdHWorkspaceBaseOffset);
    tiling.set_vWorkspaceOffset(vWorkspaceOffset);
    tiling.set_vUpdateWorkspaceOffset(vUpdateWorkspaceOffset);
    tiling.set_kDecayWorkspaceOffset(kDecayWorkspaceOffset);
    tiling.set_hWorkspaceOffset(hWorkspaceOffset);
    tiling.set_numSeqWorkspaceOffset(numSeqWorkspaceOffset);
    tiling.set_numChunksWorkspaceOffset(numChunksWorkspaceOffset);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaFwd(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwd)
    .Tiling(Tiling4ChunkKdaFwd)
    .TilingParse<ChunkKdaFwdCompileInfo>(TilingPrepare4ChunkKdaFwd);
} // namespace optiling
