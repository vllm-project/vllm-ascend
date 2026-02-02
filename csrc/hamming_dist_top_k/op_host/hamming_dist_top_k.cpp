
#include "hamming_dist_top_k_tiling.h"
#include "hamming_dist_top_k.h"
#include "register/op_def_registry.h"
#include <sstream>
namespace optiling {

namespace {

}

bool HammingDistTopKTiling::IsCapable() 
{
    return true; 
}

ge::graphStatus HammingDistTopKTiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

ge::graphStatus HammingDistTopKTiling::GetShapeAttrsInfo() {   
    inputParams_.opName = context_->GetNodeName();
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HammingDistTopKTiling::DoOpTiling() {
    auto keyBlockTablePtr = context_->GetOptionalInputShape(KEY_BLOCK_TABLE_INPUT_INDEX);
    continFlag_ = keyBlockTablePtr != nullptr;
    if (!this->SetPlatformInfoForTiling()) {
        return ge::GRAPH_FAILED;
    }

    uint32_t batch = GetShape(0).GetDim(0);
    uint32_t head = GetShape(1).GetDim(1);
    uint32_t qHead = GetShape(0).GetDim(1);
    uint32_t headGroupNum = qHead / head;
    seqLen_ = GetShape(1).GetDim(2); /* when continFlag==true, it is blockSize */
    if (continFlag_) {
        uint32_t blockSize = GetShape(1).GetDim(2); /* when continFlag==true, it is blockSize */
        seqLen_ = GetInputAttrData(0);
        seqLen_ = ops::CeilDiv(seqLen_, blockSize) * blockSize;
        uint32_t blockCount = GetShape(KEY_BLOCK_TABLE_INPUT_INDEX).GetDim(1);
        tilingData_.params.set_blockCount(blockCount);
        // printf("parallel: seqLen_ = %d\n", seqLen_);
    }
    uint32_t dimension = GetShape(0).GetDim(3) * COMPRESSED_RATE;
    uint64_t nope_dimension = GetShape(1).GetDim(3) * COMPRESSED_RATE;
    uint32_t reducedBatch = batch * head;
    uint32_t usedCoreNum = std::min(reducedBatch, coreNum_);
    uint32_t singleCoreBatch = ops::CeilDiv(reducedBatch, usedCoreNum);
    tilingData_.params.set_batch(batch);
    tilingData_.params.set_head(head);
    tilingData_.params.set_maxSeqLen(seqLen_);
    tilingData_.params.set_qHead(qHead);
    tilingData_.params.set_headGroupNum(headGroupNum);

    tilingData_.params.set_maxK(maxK);

    tilingData_.params.set_dimension(dimension);
    tilingData_.params.set_nope_dimension(nope_dimension);
    tilingData_.params.set_reducedBatch(reducedBatch);
    tilingData_.params.set_usedCoreNum(usedCoreNum);
    tilingData_.params.set_tileN1(TILE_N1);
    if (continFlag_) {
        uint32_t blockSize = GetShape(1).GetDim(2); /* 2: the dim of blockSize */
        tilingData_.params.set_tileN1(blockSize);
    }
    tilingData_.params.set_tileN2(TILE_N2);
    tilingData_.params.set_singleCoreBatch(singleCoreBatch);

    tilingData_.params.set_singleCoreSeqLen(seqLen_);
    tilingData_.params.set_kNopeUnpackGmOffset(static_cast<uint64_t>(reducedBatch) * seqLen_ * nope_dimension / 2); /* 2 : 1 / sizeof(int4b_t) */
    tilingData_.params.set_qUnpackGmOffset(static_cast<uint64_t>(reducedBatch) * seqLen_ * dimension / 2);
    tilingData_.params.set_mmGmOffset(static_cast<uint64_t>(reducedBatch) * seqLen_ * dimension / 2 + /* 2 : 1 / sizeof(int4b_t) */
        static_cast<uint64_t>(reducedBatch) * 1 * dimension / 2);

    this->SetMatmulTiling();

    bool supportKeyRope = context_->GetOptionalInputShape(KEY_ROPE_INPUT_INDEX) != nullptr;;
    tilingData_.params.set_supportKeyRope(supportKeyRope);
    if (supportKeyRope) {
        uint64_t rope_dimension = GetShape(KEY_ROPE_INPUT_INDEX).GetDim(3) * COMPRESSED_RATE;
        tilingData_.params.set_rope_dimension(rope_dimension);
        this->SetMatmulTilingRope();
    }

    this->SetTopKTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HammingDistTopKTiling::DoLibApiTiling() {
    PrintTilingData();
    PrintTilingDataRope();
    return ge::GRAPH_SUCCESS;
}

uint64_t HammingDistTopKTiling::GetTilingKey() { return 1; }

ge::graphStatus HammingDistTopKTiling::GetWorkspaceSize() {
    uint64_t *workspaces = context_->GetWorkspaceSizes(1);
    uint64_t sysWorkspaceSize = WORKSIZE;
    /* usrWorkspaceSize = workspace for Select + workspace for Topk */
    uint64_t usrWorkspaceSize = ops::CeilDiv(static_cast<uint64_t>(tilingData_.params.get_reducedBatch()) *
                                tilingData_.params.get_maxSeqLen() * tilingData_.params.get_dimension() *
                                sizeof(int8_t), static_cast<uint64_t>(2)) + /* 2: 1/2, size of int4 */
                                ops::CeilDiv(static_cast<uint64_t>(tilingData_.params.get_reducedBatch()) *
                                tilingData_.params.get_dimension() * sizeof(int8_t), static_cast<uint64_t>(2)) +
                                static_cast<uint64_t>(tilingData_.params.get_reducedBatch()) *
                                tilingData_.params.get_maxSeqLen() * sizeof(uint16_t);
    workspaces[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HammingDistTopKTiling::PostTiling() {
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    auto blockDim = tilingData_.params.get_usedCoreNum();
    context_->SetBlockDim(blockDim);
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void HammingDistTopKTiling::Reset() {
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    inputParams_.mSize = 0UL;
    inputParams_.kSize = 0UL;
    inputParams_.nSize = 0UL;
    inputParams_.queryDtype = ge::DT_INT4;
    inputParams_.keyDtype = ge::DT_UINT8;
    inputParams_.kDtype = ge::DT_INT32;
    inputParams_.seqLenDtype = ge::DT_INT32;
    inputParams_.indicesDtype = ge::DT_INT32;
    inputParams_.libApiWorkSpaceSize = 0U;
    inputParams_.opName = nullptr;
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
}

void HammingDistTopKTiling::SetMatmulTiling() {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);
    tiling.SetDim(1);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    if (seqLen_ >= SEQ_LEN_THRES) {
        tiling.SetFixSplit(-1, 512, -1); /* 512: BaseN = 512 and BaseK = 128 can fully utilize L0B */
    }
    uint64_t nope_dimension = GetShape(1).GetDim(3) * 8;
    tiling.SetShape(1, seqLen_, nope_dimension);
    tiling.SetSingleShape(1, seqLen_, nope_dimension);
    tiling.SetOrgShape(1, seqLen_, nope_dimension);
    tiling.SetBias(false);
    tiling.GetTiling(tilingData_.matmulTiling); /* if ret = -1, get tiling failed */
}

void HammingDistTopKTiling::SetMatmulTilingRope() {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);
    tiling.SetDim(1);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    if (seqLen_ >= SEQ_LEN_THRES) {
        tiling.SetFixSplit(-1, 512, -1); /* 512: BaseN = 512 and BaseK = 128 can fully utilize L0B */
    }
    uint64_t rope_dimension = GetShape(KEY_ROPE_INPUT_INDEX).GetDim(3) * 8;
    tiling.SetShape(1, seqLen_, rope_dimension);
    tiling.SetSingleShape(1, seqLen_, rope_dimension);
    tiling.SetOrgShape(1, seqLen_, rope_dimension);
    tiling.SetBias(false);
    tiling.GetTiling(tilingData_.matmulTilingRope); /* if ret = -1, get tiling failed */
}

void HammingDistTopKTiling::SetTopKTiling() {   
    uint32_t inner = std::min(ops::CeilDiv(seqLen_, TOP_K_ALIGN_NUM) * TOP_K_ALIGN_NUM, tilingData_.params.get_tileN2());
    uint32_t outter = 1;
    uint32_t k = std::min(seqLen_, maxK);
    uint32_t maxSize = 0;
    uint32_t minSize = 0;
    uint32_t dTypeSize = 2; /* 2:size of float16 */
    const bool IS_REUSESOURCE = false;
    const bool IS_INITINDEX = true;
    const bool IS_LARGEST = true;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    tilingData_.params.set_outter(outter);
    tilingData_.params.set_inner(inner);
    tilingData_.params.set_topkN(inner);
    AscendC::TopKTilingFunc(ascendcPlatform, inner, outter, k, dTypeSize, IS_INITINDEX, AscendC::TopKMode::TOPK_NORMAL, IS_LARGEST, tilingData_.topkTiling);
    AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, inner, outter, IS_REUSESOURCE, IS_INITINDEX, AscendC::TopKMode::TOPK_NORMAL, IS_LARGEST, dTypeSize, maxSize, minSize);
}

void HammingDistTopKTiling::PrintTilingData() {
    optiling::TCubeTiling &tiling = tilingData_.matmulTiling;
    std::stringstream ss;
    ss << "\n batch: " << tilingData_.params.get_batch()
       << "\n head: " << tilingData_.params.get_head() << " qHead: " << tilingData_.params.get_qHead()
       << "\n headGroupNum: " << tilingData_.params.get_headGroupNum()
       << "\n reducedBatch: " << tilingData_.params.get_reducedBatch()
       << "\n seqLen: " << tilingData_.params.get_maxSeqLen() << " dimension: " << tilingData_.params.get_dimension()
       << "\n tileN1: " << tilingData_.params.get_tileN1() << " tileN2: " << tilingData_.params.get_tileN2();
    ss << "\n usedCoreNum: " << tiling.get_usedCoreNum() << " M: " << tiling.get_M() << " N: " << tiling.get_N()
       << "\n Ka: " << tiling.get_Ka() << " Kb: " << tiling.get_Kb() << " singleCoreM: " << tiling.get_singleCoreM()
       << "\n singleCoreN: " << tiling.get_singleCoreN() << " singleCoreK: " << tiling.get_singleCoreK()
       << "\n baseM: " << tiling.get_baseM() << " baseN: " << tiling.get_baseN() << " baseK: " << tiling.get_baseK()
       << "\n depthA1: " << tiling.get_depthA1() << " depthB1: " << tiling.get_depthB1()
       << "\n stepM: " << tiling.get_stepM() << " stepN: " << tiling.get_stepN() << " stepka: " << tiling.get_stepKa()
       << "\n stepkb: " << tiling.get_stepKb();
}

void HammingDistTopKTiling::PrintTilingDataRope() {
    optiling::TCubeTiling &tiling = tilingData_.matmulTilingRope;
    std::stringstream ss;
    ss << "\n batch: " << tilingData_.params.get_batch()
       << "\n head: " << tilingData_.params.get_head() << " qHead: " << tilingData_.params.get_qHead()
       << "\n headGroupNum: " << tilingData_.params.get_headGroupNum()
       << "\n reducedBatch: " << tilingData_.params.get_reducedBatch()
       << "\n seqLen: " << tilingData_.params.get_maxSeqLen() << " dimension: " << tilingData_.params.get_dimension()
       << "\n tileN1: " << tilingData_.params.get_tileN1() << " tileN2: " << tilingData_.params.get_tileN2();
    ss << "\n usedCoreNum: " << tiling.get_usedCoreNum() << " M: " << tiling.get_M() << " N: " << tiling.get_N()
       << "\n Ka: " << tiling.get_Ka() << " Kb: " << tiling.get_Kb() << " singleCoreM: " << tiling.get_singleCoreM()
       << "\n singleCoreN: " << tiling.get_singleCoreN() << " singleCoreK: " << tiling.get_singleCoreK()
       << "\n baseM: " << tiling.get_baseM() << " baseN: " << tiling.get_baseN() << " baseK: " << tiling.get_baseK()
       << "\n depthA1: " << tiling.get_depthA1() << " depthB1: " << tiling.get_depthB1()
       << "\n stepM: " << tiling.get_stepM() << " stepN: " << tiling.get_stepN() << " stepka: " << tiling.get_stepKa()
       << "\n stepkb: " << tiling.get_stepKb();
}

const gert::Shape HammingDistTopKTiling::GetShape(const size_t index) {
    return context_->GetInputShape(index)->GetStorageShape();
}

const gert::Shape HammingDistTopKTiling::GetOutShape(const size_t index) {
    return context_->GetOutputShape(index)->GetStorageShape();
}

const uint32_t HammingDistTopKTiling::GetInputAttrData(const size_t index) {
    if (auto attrPtr = context_->GetAttrs()) {
        const int64_t* p = attrPtr->GetInt(index);
        if (p != nullptr) {
            return static_cast<uint32_t>(*p);
        }
    }
    return 0;
}

bool HammingDistTopKTiling::SetPlatformInfoForTiling() {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    coreNum_ = ascendcPlatform.GetCoreNumAic();
    return true;
}

}