#ifndef HAMMING_DIST_TOP_K_TILING_H
#define HAMMING_DIST_TOP_K_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HammingDistTopKTilingParams)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
	TILING_DATA_FIELD_DEF(uint32_t, batchN);
    TILING_DATA_FIELD_DEF(uint32_t, head);
    TILING_DATA_FIELD_DEF(uint32_t, dimension);
    TILING_DATA_FIELD_DEF(uint32_t, nope_dimension);
    TILING_DATA_FIELD_DEF(uint32_t, rope_dimension);
    TILING_DATA_FIELD_DEF(uint32_t, reducedBatch);
	TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, sink);
    TILING_DATA_FIELD_DEF(uint32_t, recent);
    TILING_DATA_FIELD_DEF(uint32_t, supportOffload);
    TILING_DATA_FIELD_DEF(uint32_t, layerSize);
    TILING_DATA_FIELD_DEF(uint32_t, layerSizeRope);
    TILING_DATA_FIELD_DEF(uint32_t, matmulResultSize);
    TILING_DATA_FIELD_DEF(uint32_t, topKValueSize);
    TILING_DATA_FIELD_DEF(uint32_t, topKIdexSize);
    TILING_DATA_FIELD_DEF(uint32_t, topKInnerSize);
    TILING_DATA_FIELD_DEF(uint32_t, maxK);
    TILING_DATA_FIELD_DEF(uint32_t, tileN1);
    TILING_DATA_FIELD_DEF(uint32_t, sBlockSize);
    TILING_DATA_FIELD_DEF(uint32_t, blockCount);
    TILING_DATA_FIELD_DEF(uint32_t, tileN3);
    TILING_DATA_FIELD_DEF(uint32_t, tileN2);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreBatch);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, outter);
    TILING_DATA_FIELD_DEF(uint32_t, inner);
    TILING_DATA_FIELD_DEF(uint32_t, topkN);
    TILING_DATA_FIELD_DEF(uint64_t, kNopeUnpackGmOffset);
    TILING_DATA_FIELD_DEF(uint64_t, mmGmOffset);
    TILING_DATA_FIELD_DEF(uint32_t, qHead);
    TILING_DATA_FIELD_DEF(uint64_t, qUnpackGmOffset);
    TILING_DATA_FIELD_DEF(uint32_t, headGroupNum);
    TILING_DATA_FIELD_DEF(uint32_t, supportKeyRope);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HammingDistTopKTilingParamsOp, HammingDistTopKTilingParams)

BEGIN_TILING_DATA_DEF(HammingDistTopKTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(HammingDistTopKTilingParams, params);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingRope);
    TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HammingDistTopK, HammingDistTopKTilingData)
REGISTER_TILING_DATA_CLASS(HammingDistTopKTilingDataOp, HammingDistTopKTilingData)

struct HammingDistTopKMatmulInfo {
    bool transA = false;
    bool transB = false;
    bool hasBias = false;
    uint64_t mSize = 0UL;
    uint64_t kSize = 0UL;
    uint64_t nSize = 0UL;
    ge::DataType queryDtype = ge::DT_INT4;
    ge::DataType keyDtype = ge::DT_UINT8;
    ge::DataType kDtype = ge::DT_INT32;
    ge::DataType seqLenDtype = ge::DT_INT32;
    ge::DataType indicesDtype = ge::DT_INT32;
    int64_t outDtype = 0L;
    uint32_t libApiWorkSpaceSize = 0U;
    uint64_t bf16ExtreWorkSpaceSize = 0UL;
    const char *opName = nullptr;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    ge::Format cFormat = ge::FORMAT_ND;
};

struct AiCoreParams {
    uint64_t ubSize;
    uint64_t blockDim;
    uint64_t aicNum;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
};
// using HammingDistTopKCompileInfo = gert::GemmCompileInfo;
}

#endif