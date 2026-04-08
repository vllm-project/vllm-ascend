
#include "hamming_dist_top_k_tiling.h"
#include "hamming_dist_top_k.h"
#include "hamming_dist_top_k_split.h"
#include <sstream>
#include <iostream> //for PrintTilingData

namespace optiling {
namespace {

}
bool HammingDistTopKSplitSTiling::IsCapable() {
    SetPlatformInfoForTiling();
    bool isContinuousBatch = context_->GetOptionalInputShape(KEY_BLOCK_TABLE_INPUT_INDEX) != nullptr;
    uint32_t batch = GetShape(0).GetDim(0);
    uint32_t maxSeqLen = GetShape(1).GetDim(2); /* when continFlag==false, it is maxSeqLen */
    if (isContinuousBatch) {
        uint32_t blockSize = GetShape(1).GetDim(2); /* when continFlag==true, it is blockSize */
        tilingData_.params.set_tileN1(blockSize);
        uint32_t blockCount = GetShape(KEY_BLOCK_TABLE_INPUT_INDEX).GetDim(1);
        tilingData_.params.set_blockCount(blockCount);

        maxSeqLen = GetInputAttrData(0);
        maxSeqLen = ((maxSeqLen + blockSize - 1) / blockSize) * blockSize;
        // printf("split: maxSeqLen = %d\n", maxSeqLen);
        if (maxSeqLen == 0) {  // 回退：blockCount * blockSize
            maxSeqLen = blockCount * blockSize;
        }
    } else {
        tilingData_.params.set_tileN1(TILE_N1);
    }
    tilingData_.params.set_maxSeqLen(maxSeqLen);
    uint32_t head = GetShape(1).GetDim(1);
    uint32_t usedCoreNum = coreNum_;
    if (head > usedCoreNum) {
        return false;
    }

    if (maxSeqLen > SUPER_LONG_SEQLEN || (batch < MAX_BATCH && maxSeqLen > MIN_SPLIT_S_SEQLEN)) {
        return true;
    }
    return false; 
}

ge::graphStatus HammingDistTopKSplitSTiling::GetWorkspaceSize() {
    uint64_t *workspaces = context_->GetWorkspaceSizes(1);
    uint64_t sysWorkspaceSize = WORKSIZE;
    //usrWorkspaceSize = workspace for Select + workspace for Topk
    uint64_t usrWorkspaceSize = ops::CeilDiv(static_cast<uint64_t>(tilingData_.params.get_layerSize() * COMPRESSED_RATE * sizeof(int8_t)), static_cast<uint64_t>(2)) +
                                 ops::CeilDiv(static_cast<uint64_t>(tilingData_.params.get_layerSizeRope() * COMPRESSED_RATE * sizeof(int8_t)), static_cast<uint64_t>(2)) + 
                                 ops::CeilDiv(static_cast<uint64_t>(tilingData_.params.get_matmulResultSize() * sizeof(float)), static_cast<uint64_t>(2)) + 
                                 ops::CeilDiv(static_cast<uint64_t>(tilingData_.params.get_topKValueSize() * sizeof(float)), static_cast<uint64_t>(2)) + 
                                 static_cast<uint64_t>(tilingData_.params.get_topKIdexSize() * sizeof(int32_t)) +
                                 ops::CeilDiv(static_cast<uint64_t>(tilingData_.params.get_batchN()) *
                                 tilingData_.params.get_dimension() * sizeof(int8_t), static_cast<uint64_t>(2));

    workspaces[0] = sysWorkspaceSize + WORKSPACE_SCALE * usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HammingDistTopKSplitSTiling::DoOpTiling() {   
    uint64_t batch = GetShape(0).GetDim(0);
    uint64_t qHead = GetShape(0).GetDim(1);
    uint64_t head = GetShape(1).GetDim(1);
    uint64_t dimension = GetShape(0).GetDim(3) * COMPRESSED_RATE;
    uint64_t nope_dimension = GetShape(1).GetDim(3) * COMPRESSED_RATE;
    uint64_t headGroupNum = qHead / head;
    uint64_t maxSeqLen = tilingData_.params.get_maxSeqLen();
    uint64_t usedCoreNum = coreNum_;
    uint64_t tileN2 = 4 * 1024; 
    tilingData_.params.set_batch(batch);
    tilingData_.params.set_head(head);
    tilingData_.params.set_qHead(qHead);
    tilingData_.params.set_headGroupNum(headGroupNum);
    tilingData_.params.set_batchN(batch * head);
    tilingData_.params.set_dimension(dimension);
    tilingData_.params.set_nope_dimension(nope_dimension);
    tilingData_.params.set_layerSize(batch * head * maxSeqLen * nope_dimension / COMPRESSED_RATE);     
    tilingData_.params.set_matmulResultSize(batch * head * maxSeqLen);
    tilingData_.params.set_topKValueSize(batch * head * ops::CeilDiv(maxSeqLen, tileN2) * maxK);
    tilingData_.params.set_topKIdexSize(batch * head * ops::CeilDiv(maxSeqLen, tileN2) * maxK);
    tilingData_.params.set_topKInnerSize(TOP_K_INNER_SIZE);
    tilingData_.params.set_maxK(maxK);
    tilingData_.params.set_usedCoreNum(usedCoreNum);
    tilingData_.params.set_sBlockSize(S_BLOCK_SIZE);
    tilingData_.params.set_tileN3(TILE_N3);
    tilingData_.params.set_tileN2(tileN2);
    bool supportKeyRope = context_->GetOptionalInputShape(KEY_ROPE_INPUT_INDEX) != nullptr;;
    tilingData_.params.set_supportKeyRope(supportKeyRope);
    SetMatmulTiling();
    if (supportKeyRope) {
        uint64_t rope_dimension = GetShape(KEY_ROPE_INPUT_INDEX).GetDim(3) * COMPRESSED_RATE;
        tilingData_.params.set_rope_dimension(rope_dimension);
        tilingData_.params.set_layerSizeRope(batch * head * maxSeqLen * rope_dimension / COMPRESSED_RATE);
        SetMatmulTilingRope();  
    }
    SetTopKTiling();
    return ge::GRAPH_SUCCESS;
}

uint64_t HammingDistTopKSplitSTiling::GetTilingKey() {
    return 10;
}

void HammingDistTopKSplitSTiling::SetMatmulTiling() {
    uint64_t nope_dimension = GetShape(1).GetDim(3) * COMPRESSED_RATE;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);
    tiling.SetDim(1);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetFixSplit(-1, L0B_BASE_SIZE, -1);
    tiling.SetShape(1, VECTOR_CUBE_RATIO * tilingData_.params.get_tileN2(), nope_dimension);
    tiling.SetSingleShape(1, VECTOR_CUBE_RATIO * tilingData_.params.get_tileN2(), nope_dimension);
    tiling.SetOrgShape(1, VECTOR_CUBE_RATIO * tilingData_.params.get_tileN2(), nope_dimension);
    tiling.SetBias(false);
    tiling.GetTiling(tilingData_.matmulTiling); // if ret = -1, get tiling failed
}

void HammingDistTopKSplitSTiling::SetMatmulTilingRope() {
    uint64_t rope_dimension = GetShape(KEY_ROPE_INPUT_INDEX).GetDim(3) * COMPRESSED_RATE;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);
    tiling.SetDim(1);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetFixSplit(-1, L0B_BASE_SIZE, -1);
    tiling.SetShape(1, VECTOR_CUBE_RATIO * tilingData_.params.get_tileN2(), rope_dimension);
    tiling.SetSingleShape(1, VECTOR_CUBE_RATIO * tilingData_.params.get_tileN2(), rope_dimension);
    tiling.SetOrgShape(1, VECTOR_CUBE_RATIO * tilingData_.params.get_tileN2(), rope_dimension);
    tiling.SetBias(false);
    tiling.GetTiling(tilingData_.matmulTilingRope); // if ret = -1, get tiling failed
}

void HammingDistTopKSplitSTiling::SetTopKTiling() {
    uint32_t inner = tilingData_.params.get_topKInnerSize();
    uint32_t outter = 1;
    uint32_t dTypeSize = 2; // 2:size of float16
    const bool IS_REUSESOURCE = false;
    const bool IS_INITINDEX = true;
    const bool IS_LARGEST = true;
    uint32_t maxSize = 0;
    uint32_t minSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    tilingData_.params.set_outter(outter);
    tilingData_.params.set_inner(inner);
    tilingData_.params.set_topkN(inner);
    AscendC::TopKTilingFunc(ascendcPlatform, inner, outter, maxK, dTypeSize, IS_INITINDEX, AscendC::TopKMode::TOPK_NORMAL, IS_LARGEST, tilingData_.topkTiling);
    AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, inner, outter, IS_REUSESOURCE, IS_INITINDEX, AscendC::TopKMode::TOPK_NORMAL, IS_LARGEST, dTypeSize, maxSize, minSize);
}

void HammingDistTopKSplitSTiling::PrintTilingData() {
    optiling::TCubeTiling &tiling = tilingData_.matmulTiling;
    std::stringstream ss;
    ss << "\n usedCoreNum: " << coreNum_ << " M: " << tiling.get_M() << " N: " << tiling.get_N()
       << "\n Ka: " << tiling.get_Ka() << " Kb: " << tiling.get_Kb() << " singleCoreM: " << tiling.get_singleCoreM()
       << "\n singleCoreN: " << tiling.get_singleCoreN() << " singleCoreK: " << tiling.get_singleCoreK()
       << "\n baseM: " << tiling.get_baseM() << " baseN: " << tiling.get_baseN() << " baseK: " << tiling.get_baseK()
       << "\n depthA1: " << tiling.get_depthA1() << " depthB1: " << tiling.get_depthB1()
       << "\n stepM: " << tiling.get_stepM() << " stepN: " << tiling.get_stepN() << " stepka: " << tiling.get_stepKa()
       << "\n stepkb: " << tiling.get_stepKb();
    
    std::cout << "HammingDistTopKSplitSTiling::PrintTilingData()" << ss.str() << std::endl;
}

void HammingDistTopKSplitSTiling::PrintTilingDataRope() {
    optiling::TCubeTiling &tiling = tilingData_.matmulTilingRope;
    std::stringstream ss;
    ss << "\n usedCoreNum: " << coreNum_ << " M: " << tiling.get_M() << " N: " << tiling.get_N()
       << "\n Ka: " << tiling.get_Ka() << " Kb: " << tiling.get_Kb() << " singleCoreM: " << tiling.get_singleCoreM()
       << "\n singleCoreN: " << tiling.get_singleCoreN() << " singleCoreK: " << tiling.get_singleCoreK()
       << "\n baseM: " << tiling.get_baseM() << " baseN: " << tiling.get_baseN() << " baseK: " << tiling.get_baseK()
       << "\n depthA1: " << tiling.get_depthA1() << " depthB1: " << tiling.get_depthB1()
       << "\n stepM: " << tiling.get_stepM() << " stepN: " << tiling.get_stepN() << " stepka: " << tiling.get_stepKa()
       << "\n stepkb: " << tiling.get_stepKb();
    
    std::cout << "HammingDistTopKSplitSTiling::PrintTilingDataRope()" << ss.str() << std::endl;
}

}