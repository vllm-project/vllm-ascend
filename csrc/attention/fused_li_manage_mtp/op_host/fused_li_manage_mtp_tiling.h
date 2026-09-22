/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef FUSED_LI_MANAGE_MTP_TILING_H_
#define FUSED_LI_MANAGE_MTP_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct MtpRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct MtpTensorParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

constexpr uint32_t WEIGHTS_INDEX = 0;
constexpr uint32_t QUERY_DEQUANT_SCALE_INDEX = 1;
constexpr uint32_t QUERY_INDEX = 2;
constexpr uint32_t KEY_DEQUANT_SCALE_INDEX = 3;
constexpr uint32_t KEY_INDEX = 4;
constexpr uint32_t BLOCK_TABLE_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 6;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 7;
constexpr uint32_t OFFLOAD_SEQ_K_INDEX = 8;
constexpr uint32_t CACHE_TOKENS_INDEX = 9;
constexpr uint32_t REQUEST_STATE_INDEX = 10;
constexpr uint32_t REQ_POOL_ENTRIES_INDEX = 11;
constexpr uint32_t CACHE_SLOTS_INDEX = 12;
constexpr uint32_t TOPK_INDEX = 0;
constexpr uint32_t TOPK_SLOTS_INDEX = 1;
constexpr uint32_t MISS_COUNT_INDEX = 2;
constexpr uint32_t CACHE_SLOTS_OUT_INDEX = 3;

constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

constexpr uint32_t DECODE_N2 = 1;
constexpr uint32_t DECODE_HEAD_DIM = 128;
constexpr uint32_t DECODE_SPARSE_COUNT = 2048;
constexpr uint32_t DECODE_OUTPUT_CAPACITY = 2048;

BEGIN_TILING_DATA_DEF(FusedLiManageMtpTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, tSize)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, poolSize)
TILING_DATA_FIELD_DEF(uint32_t, n1Size)
TILING_DATA_FIELD_DEF(uint32_t, cacheSlotsSize)
TILING_DATA_FIELD_DEF(uint32_t, scheduleMode)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedLiManageMtp, FusedLiManageMtpTilingData)

struct FusedLiManageMtpCompileInfo {};

struct FusedLiManageMtpParaInfo {
    MtpRequiredParaInfo query = {nullptr, nullptr};
    MtpRequiredParaInfo key = {nullptr, nullptr};
    MtpRequiredParaInfo weights = {nullptr, nullptr};
    MtpTensorParaInfo reqPoolEntries = {nullptr, nullptr};
    MtpRequiredParaInfo cacheSlots = {nullptr, nullptr};
    MtpTensorParaInfo cacheTokens = {nullptr, nullptr};
    MtpTensorParaInfo actualSeqLengths = {nullptr, nullptr};
    MtpTensorParaInfo blockTable = {nullptr, nullptr};
    MtpRequiredParaInfo topkIndexOut = {nullptr, nullptr};
    MtpRequiredParaInfo topkSlotsOut = {nullptr, nullptr};
    MtpRequiredParaInfo topkMissCountOut = {nullptr, nullptr};
    MtpRequiredParaInfo missCountOut = {nullptr, nullptr};
    MtpRequiredParaInfo missSrcOut = {nullptr, nullptr};
    MtpRequiredParaInfo missSlotsOut = {nullptr, nullptr};
    MtpRequiredParaInfo cacheSlotsOut = {nullptr, nullptr};
};

class FusedLiManageMtpTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    FusedLiManageMtpParaInfo opParamInfo;

    uint32_t bSize = 0;
    uint32_t tSize = 0;
    uint32_t n1Size = 32;
    uint32_t n2Size = DECODE_N2;
    uint32_t s2Size = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t poolSize = 0;
    uint32_t cacheSlotsSize = 0;
    uint32_t usedCoreNum = 0;

    ge::DataType inputQType = ge::DT_FLOAT16;
};

class FusedLiManageMtpTiling {
public:
    explicit FusedLiManageMtpTiling(gert::TilingContext *context, bool mtp = true)
        : context_(context), mtp_(mtp) {};
    ge::graphStatus ParseAndCheck(FusedLiManageMtpTilingInfo &tilingInfo);
    ge::graphStatus DoTiling(FusedLiManageMtpTilingInfo *tilingInfo);

private:
    ge::graphStatus GetNpuInfo(FusedLiManageMtpTilingInfo &tilingInfo) const;
    ge::graphStatus GetTensorInfo(FusedLiManageMtpTilingInfo &tilingInfo) const;
    ge::graphStatus CheckDtype(const FusedLiManageMtpTilingInfo &tilingInfo) const;
    ge::graphStatus CheckShape(FusedLiManageMtpTilingInfo &tilingInfo) const;

    gert::TilingContext *context_ = nullptr;
    FusedLiManageMtpTilingData tilingData_;
    bool mtp_ = true;
};

} // namespace optiling
#endif // FUSED_LI_MANAGE_MTP_TILING_H_
