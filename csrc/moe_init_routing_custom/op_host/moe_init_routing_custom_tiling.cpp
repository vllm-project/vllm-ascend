/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_custom_tiling.cpp
 * \brief
 */
#include "moe_init_routing_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_templates_registry.h"

namespace optiling {
const static int64_t NUM_TWO = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t NUM_FIVE = 5;
const static int64_t MRG_LIST_NUM = 4;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_BLOCK_BYTE = 32;
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static int32_t SIZE_16 = 16;
const static int32_t SIZE_31 = 31;
const static int32_t LENGTH_1024 = 1024;
const static int64_t MAX_COLS_ONE_LOOP = 16376;
const static int64_t ASSIST_NUM = 256;
const static int64_t SPLIT_K_THRESHOLD = 512;
const static int64_t KV_FACTOR = 2;
const static int64_t ONE_CORE_SORT_BUFFER = 6;
const static int64_t EXPERT_IDX_MAX = 10240;
const static int64_t KV_MODE_EXPERT_IDX_MAX = EXPERT_IDX_MAX / KV_FACTOR;
const static int64_t ACTIVE_NUM_MIN_VALUE = static_cast<int64_t>(-1);
const static int64_t EXPERT_CAPACITY_MIN_VALUE = static_cast<int64_t>(0);

const static int64_t INPUT_X_INDEX = 0;
const static int64_t INPUT_EXPERT_IDX_INDEX = 1;
const static int64_t INPUT_SCALE_INDEX = 2;
const static int64_t INPUT_OFFSET_INDEX = 3;
const static int64_t OUTPUT_EXPANDED_X_INDEX = 0;
const static int64_t OUTPUT_EXPANDED_ROW_IDX_INDEX = 1;
const static int64_t OUTPUT_EXPERT_TOKENS_COUNT_INDEX = 2;
const static int64_t OUTPUT_EXPANDED_SCALE_INDEX = 3;
const static int64_t ATTR_ACTIVE_NUM_INDEX = 0;
const static int64_t ATTR_EXPERT_CAPACITY_INDEX = 1;
const static int64_t ATTR_EXPERT_NUM_INDEX = 2;
const static int64_t ATTR_DROP_PAD_MODE_INDEX = 3;
const static int64_t ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX = 4;
const static int64_t ATTR_EXPERT_TOKEN_NUM_FLAG_INDEX = 5;
const static int64_t ATTR_QUANT_MODE_INDEX = 6;
const static int64_t ATTR_EXPERT_RANGE_INDEX = 7;
const static int64_t ATTR_ROW_IDX_TYPE_INDEX = 8;
const static int64_t ATTR_EXPERT_RANGE_DIM = 2;
const static int64_t GATHER = 0;
const static int64_t SCATTER = 1;
const static int64_t UN_QUANT = -1L;
const static int64_t STATIC_QUANT = 0;
const static int64_t DYNAMIC_QUANT = 1;
const static int64_t CUMSUM = 0;
const static int64_t COUNT = 1;
const static int64_t KEY_VALUE = 2;
const static int64_t DROP_LESS = 0;
const static int64_t DROP_PAD = 1;
const static int64_t DYNAMIC_QUANT_COLS_BUFFER = 21;
const static int64_t DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER = 13;
const static int64_t STATIC_QUANT_FULLLOAD_COLS_BUFFER = 11;

const static int64_t DYNAMIC_QUANT_SRC_TO_DST_BUFFER = 15;
const static int64_t DYNAMIC_QUANT_SCALE_SIZE_64 = 64;
const static int64_t MAX_COLS_DYNAMIC_QUANT = 6144;
const static int64_t SIZE_INT32 = 4;
const static int64_t SIZE_INT16 = 2;
const static int64_t SIZE_INT8 = 1;
const static int64_t SIZE_FP32 = 4;

const static uint64_t TILINGKEY_BASE = 1000000;
const static uint64_t SORT_CORE_TILINGKEY_BASE = 100000;
const static uint64_t QUANT_MODE_TILINGKEY_BASE = 10000;
const static uint64_t ROWIDX_TYPE_TILINGKEY_BASE = 1000;
const static uint64_t DROP_MODE_TILINGKEY_BASE = 100;

// Tiling Key for performance puncturing
const static uint64_t PERFORMANCE_TILINGKEY_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168 = 2000000;
const static uint64_t UNQUANTIZED_FULLLOAD_TILINGKEY = 2100000;
const static uint64_t STATIC_QUANT_FULLLOAD_TILINGKEY = 2200000;
const static uint64_t DYNAMIC_QUANT_FULLLOAD_TILINGKEY = 2300000;
const static uint64_t DYNAMIC_QUANT_EPFULLLOAD_TILINGKEY = 10000;
const static uint64_t DYNAMIC_QUANT_SMOOTHTYPE_FULLLOAD_TILINGKEY = 1000;

const static int64_t PERFORMANCE_MODE_TOP_K = 8;
const static int64_t PERFORMANCE_MODE_BS_MIN = 384;
const static int64_t PERFORMANCE_MODE_BS_MAX = 8192;
const static int64_t PERFORMANCE_MODE_RANGE_MAX = 32;
const static int64_t PERFORMANCE_MODE_MAX_BATCH_SIZE_TOP_K = PERFORMANCE_MODE_BS_MAX * PERFORMANCE_MODE_TOP_K;
const static int64_t PERFORMANCE_MODE_MAX_ONE_CORE_GATHER = 21845;

const static int64_t gatherFirstN = 100;
const static int64_t gatherFirstScale = 8;
const static int64_t scale1H = 1;
const static int64_t scaleEH = 2;
const static int64_t ONE_REPEAT_SORT_NUM = 32;

enum class PerformanceMode : int32_t {
    COMMON = 0,
    ONE_CORE_GATHER_SORT = 1,
    MULTI_CORE_GATHER_SORT = 2,
};

static constexpr int64_t KEY_VALUE_MODE_DIM0_NUM = 2;

#define unlikely(x) __builtin_expect((x), 0)

#define CHECK_FAIL(context, cond, ...)                      \
    do {                                                    \
        if (cond) {                                         \
            OPS_LOG_E(context->GetNodeName(), ##__VA_ARGS__); \
            return ge::GRAPH_FAILED;                        \
        }                                                   \
    } while (0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                           \
    do {                                                                                                   \
        if (unlikely((ptr) == nullptr)) {                                                                  \
            const char* name = (unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr)) ? \
                                   "nil" :                                                                 \
                                   (context)->GetNodeName();                                               \
            OPS_LOG_E(name, "%s is nullptr!", #ptr);                                                       \
            return ge::GRAPH_FAILED;                                                                       \
        }                                                                                                  \
    } while (0)

template <typename T1, typename T2>
static T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
typename std::enable_if <std::is_integral<T>::value, T>::type CeilAlign(T x, T align) {
    return CeilDiv(x, align) * align;
}

inline static int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline static int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE / bytes;
}

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
}

inline static int64_t GetPerOrLastValue(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    return x <= y ? x : x % y;
}

inline static int64_t AlignOneBlockByteCeil(int64_t x)
{
    return x / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
}

class MoeInitRountingCustomTilingBase : public TilingBaseClass {
public:
    explicit MoeInitRountingCustomTilingBase(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeInitRountingCustomTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckDtype();
    void Tiling4GatherOutCompute();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void Tiling4VBSCompute();
    void Tiling4ExpertTokensCountCompute();
    void ShowTilingData();
    void Tinlig4VBSMultiCoreCompute(MoeCustomVBSComputeTilingData *tilingData);
    void Tinlig4VBSOneCoreCompute(MoeCustomVBSComputeTilingData *tilingData);
    bool IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168() const;
    bool IsFullLoad();
    int64_t IsGatherFirstFullLoad();
    void SetGatherTilingData(MoeCustomSrcToDstCapacityComputeTilingData *tilingData, int64_t perCoreRows,
                             int64_t lastCoreRows, int64_t cols);
    void SetGatherTilingDataCols(MoeCustomSrcToDstCapacityComputeTilingData *tilingData, int64_t baseMaxCols, int64_t cols);
    void SetGatherTilingDataRows(MoeCustomSrcToDstCapacityComputeTilingData *tilingData, int64_t perCoreRows,
                                 int64_t lastCoreRows, int64_t basePerLoopMaxRows);
    void Tiling4SrcToDstDropPadCompute();
    void Tiling4SrcToDstDropPadDynamicCompute();
    void Tiling4SrcToDstCompute();
    PerformanceMode GetPerformanceMode() const;

    int64_t aivNum;
    int64_t sortLoopMaxElement = 0;
    int64_t mrgSortListMaxElement = 1504;
    int64_t totalLength_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t cols_ = 0;
    int64_t inuptXDtypeSize_;

    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t isInputScale_ = 0;
    int64_t isInputOffset_ = 0;

    int64_t sortMode_ = 0;
    int64_t rowIdxTytpe_ = 0;
    int64_t activeNum_ = -1L;
    int64_t expertCapacity_ = -1L;
    int64_t expertNum_ = -1L;
    int64_t dropPadMode_ = -1L;
    int64_t expertTokensNumType_ = -1L;
    bool expertTokensNumFlag_ = false;
    int64_t quantMode_ = 0;
    int64_t rowIdxType_ = -1L;

    bool isFullload_ = false;
    int64_t gatherFirstFullload_ = 0;
    int64_t ep_ = 0;
    int64_t smoothType_ = 0;

    const gert::StorageShape *xShapePtr_ = nullptr;
    const gert::StorageShape *expertIdxShapePtr_ = nullptr;
    const gert::StorageShape *scaleShapePtr_ = nullptr;
    const gert::StorageShape *offsetShapePtr_ = nullptr;

    const int64_t *activeNumPtr_ = nullptr;
    const int64_t *expertCapacityPtr_ = nullptr;
    const int64_t *expertNumPtr_ = nullptr;
    const int64_t *dropPadModePtr_ = nullptr;
    const int64_t *expertTokensNumTypePtr_ = nullptr;
    const bool *expertTokensNumFlagPtr_ = nullptr;
    const int64_t *quantModePtr_ = nullptr;
    const gert::ContinuousVector *activeExpertRangeListPtr_;
    const int64_t *rowIdxTypePtr_ = nullptr;

    const gert::StorageShape *expandedXShapePtr_ = nullptr;
    const gert::StorageShape *expandedRowIdxShapePtr_ = nullptr;
    const gert::StorageShape *expertTokensCountOrCumsumShapePtr_ = nullptr;
    const gert::StorageShape *expandedScaleShapePtr_ = nullptr;

    const gert::Shape performXShape = gert::Shape({1, 7168});
    const gert::Shape performExpertIdxShape = gert::Shape({1, 8});
    const gert::Shape performScaleShape = gert::Shape({256, 7168});

    const char *opName = "";
    MoeInitRoutingCustomTilingData moeInitRoutingCustomTilingData;
};

void MoeInitRountingCustomTilingBase::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus MoeInitRountingCustomTilingBase::GetPlatformInfo()
{
    auto compileInfoPtr = reinterpret_cast<const MoeInitRoutingCustomCompileInfo*>(context_->GetCompileInfo());
    CHECK_FAIL(context_, compileInfoPtr == nullptr, "fail to get platform info");
    aivNum = compileInfoPtr->aivNum;
    aicoreParams_.blockDim = aivNum;
    aicoreParams_.ubSize = compileInfoPtr->ubSize;
    moeInitRoutingCustomTilingData.set_coreNum(aivNum);
    OPS_LOG_I(context_->GetNodeName(), "---PlatformInfo--- aivNum is: %ld, ubSizePlatForm is: %ld ", aivNum, aicoreParams_.ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingCustomTilingBase::CheckAttr()
{
    quantMode_ = *quantModePtr_;
    moeInitRoutingCustomTilingData.set_quantMode(quantMode_);
    OPS_LOG_I(context_->GetNodeName(), "quant_mode is: %ld ", quantMode_);

    dropPadMode_ = *dropPadModePtr_;
    moeInitRoutingCustomTilingData.set_dropPadMode(dropPadMode_);
    CHECK_FAIL(context_, (dropPadMode_ != DROP_LESS) && (dropPadMode_ != DROP_PAD), 
               "drop_pad_mode should be %ld or %ld", DROP_LESS, DROP_PAD);

    rowIdxTytpe_ = *rowIdxTypePtr_;
    moeInitRoutingCustomTilingData.set_rowIdxType(rowIdxTytpe_);
    OPS_LOG_I(context_->GetNodeName(), "row_idx_type is: %ld ", rowIdxTytpe_);

    activeNum_ = *activeNumPtr_;
    if (dropPadMode_ == DROP_LESS) {
        CHECK_FAIL(context_, activeNum_ < ACTIVE_NUM_MIN_VALUE,
                   "active_num should be greater than or equal to 0");
    }

    expertNum_ = *expertNumPtr_;
    moeInitRoutingCustomTilingData.set_expertNum(expertNum_);
	if (expertNum_ <= 0) {
		OPS_LOG_E(context_->GetNodeName(), "expert_num should be greater than 0");
		return ge::GRAPH_FAILED;
	}
	if (activeExpertRangeListPtr_->GetSize() != ATTR_EXPERT_RANGE_DIM && activeExpertRangeListPtr_->GetSize() != 0) {
		OPS_LOG_E(context_, "The dim number of expert_range should be %ld or 0(no input)", ATTR_EXPERT_RANGE_DIM);
        return ge::GRAPH_FAILED;
	}
    if (activeExpertRangeListPtr_->GetSize() == 0) {
        expertStart_ = 0;
        expertEnd_ = expertNum_;
    } else {
        const int64_t *expertRangeList = reinterpret_cast<const int64_t *>(activeExpertRangeListPtr_->GetData());
        expertStart_ = expertRangeList[0];
        expertEnd_ = expertRangeList[1];
    }
    moeInitRoutingCustomTilingData.set_expertStart(expertStart_);
    moeInitRoutingCustomTilingData.set_expertEnd(expertEnd_);
    moeInitRoutingCustomTilingData.set_actualExpertNum(expertEnd_ - expertStart_);
    OPS_LOG_I(context_, "expert_start is: %ld, expert_end is: %ld, actualExpertNum is: %ld ", expertStart_, expertEnd_,
              expertEnd_ - expertStart_);
    
    n_ = xShapePtr_->GetStorageShape().GetDim(0);
    expertCapacity_ = *expertCapacityPtr_;
    moeInitRoutingCustomTilingData.set_expertCapacity(expertCapacity_);
    if (dropPadMode_ == DROP_PAD) {
        CHECK_FAIL(context_, expertCapacity_ <= EXPERT_CAPACITY_MIN_VALUE || expertCapacity_ > n_,
                   "expert_Capacity should be greater than 0 and less than %ld", n_);
        CHECK_FAIL(context_, rowIdxTytpe_ == SCATTER, "rowIdxTytpe should be 0 when droppadmode is 1");
		CHECK_FAIL(context_, expertStart_ != 0 || expertEnd_ != expertNum_,
		           "expert_range should be [0, %ld] when droppadmode is 1", expertNum_);
    }

    expertTokensNumType_ = *expertTokensNumTypePtr_;
    moeInitRoutingCustomTilingData.set_expertTokensNumType(expertTokensNumType_);
    CHECK_FAIL(context_, (expertTokensNumType_ != COUNT) && (expertTokensNumType_ != KEY_VALUE) && (expertTokensNumType_ != CUMSUM),
                "expert_tokens_num_type currently not support %ld", expertTokensNumType_);

    expertTokensNumFlag_ = *expertTokensNumFlagPtr_;
    if (dropPadMode_ == DROP_PAD && expertTokensNumFlag_) {
        CHECK_FAIL(context_, expertTokensNumType_ != COUNT, "In DROP_PAD mode and expert_tokens_num_flag is true, expert_tokens_num_type only supports COUNT, but got %ld", expertTokensNumType_);}
    if (expertTokensNumFlag_) {
        moeInitRoutingCustomTilingData.set_expertTokensNumFlag(1);
    } else {
        moeInitRoutingCustomTilingData.set_expertTokensNumFlag(0);
    }

    CHECK_FAIL(context_, expertStart_ < 0, "expert_start should be greater than or equal to 0");
    CHECK_FAIL(context_, expertStart_ >= expertEnd_, "expert_start should be less than expert_end");
    CHECK_FAIL(context_, expertEnd_ > expertNum_, "expert_end should be less than or equal to %ld", expertNum_);
    if (expertTokensNumType_ == KEY_VALUE) {
        CHECK_FAIL(context_, expertEnd_ > KV_MODE_EXPERT_IDX_MAX, "expert_end should be less than or equal to %ld in KEY_VALUE mode",
                   KV_MODE_EXPERT_IDX_MAX);
    } else {
        CHECK_FAIL(context_, expertEnd_ > EXPERT_IDX_MAX, "expert_end should be less than or equal to %ld", EXPERT_IDX_MAX);
    }
    CHECK_FAIL(context_, quantMode_ != UN_QUANT && quantMode_ != DYNAMIC_QUANT && quantMode_ != STATIC_QUANT, "quant_mode currently support %ld, %ld or %ld", UN_QUANT, DYNAMIC_QUANT, STATIC_QUANT);
    CHECK_FAIL(context_, rowIdxTytpe_ != SCATTER && rowIdxTytpe_ != GATHER, "row_idx_type currently support %ld or %ld", SCATTER, GATHER);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingCustomTilingBase::CheckInputShape()
{
    const gert::Shape xShape = xShapePtr_->GetStorageShape();
    OPS_LOG_I(context_->GetNodeName(), "input x shape: %s ", ops::Shape2String(xShape).c_str());
    const gert::Shape expertIdxShape = expertIdxShapePtr_->GetStorageShape();
    OPS_LOG_I(context_->GetNodeName(), "input expert_idx shape: %s.", ops::Shape2String(expertIdxShape).c_str());

    // 参数校验
    CHECK_FAIL(context_, xShape.GetDimNum() != DIM_TWO, "The dim number of x should be %lu.", DIM_TWO);
    CHECK_FAIL(context_, expertIdxShape.GetDimNum() != DIM_TWO, "The dim number of expert_idx should be %lu.", DIM_TWO);
    CHECK_FAIL(context_, xShape.GetDim(0) != expertIdxShape.GetDim(0), context_->GetNodeName(), "Input rows should be same.");

    n_ = expertIdxShape.GetDim(0);
    k_ = expertIdxShape.GetDim(1);
    cols_ = xShape.GetDim(1);
    moeInitRoutingCustomTilingData.set_n(n_);
    moeInitRoutingCustomTilingData.set_k(k_);
    moeInitRoutingCustomTilingData.set_cols(cols_);
    totalLength_ = n_ * k_;
    if (activeNum_ == 0 || activeNum_ == ACTIVE_NUM_MIN_VALUE) {
        activeNum_ = totalLength_;
    } else {
        activeNum_ = std::min(activeNum_, totalLength_);
    }
    moeInitRoutingCustomTilingData.set_activeNum(activeNum_);

    inuptXDtypeSize_ =
        static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(INPUT_X_INDEX)->GetDataType()));
    OPS_LOG_I(context_->GetNodeName(), "Input x dtype size is: %ld. ", inuptXDtypeSize_);

    if (quantMode_ == UN_QUANT && scaleShapePtr_ != nullptr) {
        auto scaleShape = scaleShapePtr_->GetStorageShape();
        OPS_LOG_I(context_->GetNodeName(), "input scale shape: %s", ops::Shape2String(scaleShape).c_str());
        auto scaleDimNum = static_cast<int64_t>(scaleShape.GetDimNum());
        CHECK_FAIL(context_,
            scaleDimNum != 1, 
            context_->GetNodeName(), "The dim number of scale should be 1, current is %ld", scaleDimNum);
        auto scaleDim0 = static_cast<int64_t>(scaleShape.GetDim(0));
        CHECK_FAIL(context_,
            scaleDim0 != n_, 
            "The first dim of scale should be n_, current is %ld", scaleDim0);
    }

    if (quantMode_ == STATIC_QUANT) {
        CHECK_FAIL(context_, scaleShapePtr_ == nullptr, "scale is null");
        CHECK_FAIL(context_, offsetShapePtr_ == nullptr, "offset is null");
        auto scaleShape = scaleShapePtr_->GetStorageShape();
        OPS_LOG_I(context_->GetNodeName(), "input scale shape: %s", ops::Shape2String(scaleShape).c_str());
        auto scaleDimNum = static_cast<int64_t>(scaleShape.GetDimNum());
        CHECK_FAIL(context_,
            scaleDimNum != 1, 
            "The dim number of scale should be 1, current is %ld", scaleDimNum);
        auto scaleDim0 = static_cast<int64_t>(scaleShape.GetDim(0));
        CHECK_FAIL(context_,
            scaleDim0 != 1, 
            "The first dim of scale should be 1, current is %ld", scaleDim0);
        auto offsetShape = offsetShapePtr_->GetStorageShape();
        OPS_LOG_I(context_->GetNodeName(), "input offset shape: %s", ops::Shape2String(offsetShape).c_str());
        auto offsetDimNum = static_cast<int64_t>(offsetShape.GetDimNum());
        CHECK_FAIL(context_,
            offsetDimNum != 1,
            "The dim number of offset should be 1, current is %ld", offsetDimNum);
        auto offsetDim0 = static_cast<int64_t>(offsetShape.GetDim(0));
        CHECK_FAIL(context_,
            offsetDim0 != 1, 
            "The first dim of offset should be 1, current is %ld", offsetDim0);
    }

    if (quantMode_ == DYNAMIC_QUANT && scaleShapePtr_ != nullptr) {
        auto scaleShape = scaleShapePtr_->GetStorageShape();
        OPS_LOG_I(context_->GetNodeName(), "input scale shape: %s", ops::Shape2String(scaleShape).c_str());
        auto scaleDimNum = static_cast<int64_t>(scaleShape.GetDimNum());
        CHECK_FAIL(context_,
            scaleDimNum != NUM_TWO, 
            "The dim number of scale should be 2, current is %ld", scaleDimNum);
        auto scaleDim0 = static_cast<int64_t>(scaleShape.GetDim(0));
        CHECK_FAIL(context_,
            scaleDim0 != (expertEnd_ - expertStart_) && scaleDim0 != 1, 
            "The first dim of scale should be %ld or 1, current is %ld", (expertEnd_ - expertStart_), scaleDim0);
        auto scaleDim1 = static_cast<int64_t>(scaleShape.GetDim(1));
        CHECK_FAIL(context_,
            scaleDim1 != cols_,
            "The second dim of scale should be %ld, current is %ld", cols_, scaleDim0);
        if (scaleDim0 == 1) {
            smoothType_ = scale1H;
        } else {
            smoothType_ = scaleEH;
        }
        moeInitRoutingCustomTilingData.set_smoothType(smoothType_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingCustomTilingBase::CheckOutShape()
{
    const gert::Shape expandedXShape = context_->GetOutputShape(0)->GetStorageShape();
    OPS_LOG_I(context_->GetNodeName(), "expanded_x shape: %s.", ops::Shape2String(expandedXShape).c_str());
    const gert::Shape expandedRowIdxShape = context_->GetOutputShape(1)->GetStorageShape();
    OPS_LOG_I(context_->GetNodeName(), "expanded_row_idx shape: %s.", ops::Shape2String(expandedRowIdxShape).c_str());
    const gert::Shape expertTokensCountOrCumsumShape = context_->GetOutputShape(NUM_TWO)->GetStorageShape();
    OPS_LOG_I(context_->GetNodeName(), "expert_tokens_count_or_cumsum shape: %s.", ops::Shape2String(expertTokensCountOrCumsumShape).c_str());

    size_t expandedXDimNum = expandedXShape.GetDimNum();
    if (dropPadMode_ > 0) {
        CHECK_FAIL(context_, expandedXDimNum != NUM_THREE, "The dim number of expandedX should be 3.");
        CHECK_FAIL(context_, expandedXShape.GetDim(0) != expertNum_, "The first dim of expandedX should be %ld.", expertNum_);
        CHECK_FAIL(context_, expandedXShape.GetDim(1) != expertCapacity_, "The second dim of expandedX should be %ld.",
                    expertCapacity_);
        CHECK_FAIL(context_,
                    expandedXShape.GetDim(NUM_TWO) != cols_,
                    "The third dim of expandedX should be %ld.", cols_);
    } else {
        CHECK_FAIL(context_,expandedXDimNum != DIM_TWO, "The dim number of expandedX should be 2.");
        int64_t firstDim = totalLength_;
        firstDim = activeNum_ == 0 ? firstDim : std::min(firstDim, activeNum_);
        CHECK_FAIL(context_, expandedXShape.GetDim(0) != firstDim, "The first dim of expandedX should be %ld.", firstDim);
        CHECK_FAIL(context_, expandedXShape.GetDim(1) != cols_,
                    "The second dim of expandedX should be %ld.", cols_);
    }

    CHECK_FAIL(context_, expandedRowIdxShape.GetDimNum() != DIM_ONE,
                "The dim number of expanded_row_idx should be 1.");
    CHECK_FAIL(context_,
        expandedRowIdxShape.GetDim(0) != totalLength_,
        "The first dim of expanded_row_idx and expanded_expert_idx should be %ld.", totalLength_);

    if(expertTokensNumFlag_){
        if (expertTokensNumType_ == KEY_VALUE) {
            CHECK_FAIL(context_,
                expertTokensCountOrCumsumShape.GetDimNum() != DIM_TWO,
                "The dim number of expert_tokens_count_or_cumsum should be 2 when in KEY_VALUE mode.");
            CHECK_FAIL(context_, expertTokensCountOrCumsumShape.GetDim(0) != expertNum_,
                        "The first dim of expert_tokens_count_or_cumsum should be %ld.", expertNum_);
            CHECK_FAIL(context_, expertTokensCountOrCumsumShape.GetDim(1) != KEY_VALUE_MODE_DIM0_NUM,
                        "The second dim of expert_tokens_count_or_cumsum should be %ld.",
                                KEY_VALUE_MODE_DIM0_NUM);
        } else {
            CHECK_FAIL(context_, expertTokensCountOrCumsumShape.GetDimNum() != DIM_ONE,
                        "The dim number of expert_tokens_count_or_cumsum should be 1 when not in KEY_VALUE mode.");
            CHECK_FAIL(context_, expertTokensCountOrCumsumShape.GetDim(0) != (expertEnd_ - expertStart_),
                       "The first dim of expert_tokens_count_or_cumsum should be %ld.", (expertEnd_ - expertStart_));
        }
    }

    if (quantMode_ != STATIC_QUANT && scaleShapePtr_ != nullptr) {
        const gert::Shape expandedScaleShape = context_->GetOutputShape(3)->GetStorageShape();
        OPS_LOG_I(context_->GetNodeName(), "expanded_scale shape: %s.", ops::Shape2String(expandedScaleShape).c_str());
        size_t expandedScaleDimNum = expandedScaleShape.GetDimNum();
        CHECK_FAIL(context_, expandedScaleDimNum != DIM_ONE, "The dim number of expanded_scale should be 1.");
        if (dropPadMode_ > 0) {
            CHECK_FAIL(context_, expandedScaleShape.GetDim(0) != expertNum_ * expertCapacity_,
                 "The first dim of expanded_scale should be %ld.", expertNum_ * expertCapacity_);
        } else {
            int64_t firstDim = totalLength_;
            firstDim = activeNum_ == 0 ? firstDim : std::min(firstDim, activeNum_);
            CHECK_FAIL(context_, expandedScaleShape.GetDim(0) != firstDim,
                "The first dim of expanded_scale should be %ld.", firstDim);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingCustomTilingBase::GetShapeAttrsInfo()
{
    OPS_LOG_I(context_->GetNodeName(), "TilingContext: %s.", context_->GetNodeName());

    xShapePtr_ = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr_);

    expertIdxShapePtr_ = context_->GetInputShape(INPUT_EXPERT_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxShapePtr_);

    scaleShapePtr_ = context_->GetOptionalInputShape(INPUT_SCALE_INDEX);
    if (scaleShapePtr_ == nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "optional input scale is null");
    } else {
        isInputScale_ = 1;
    }
    moeInitRoutingCustomTilingData.set_isInputScale(isInputScale_);

    offsetShapePtr_ = context_->GetOptionalInputShape(INPUT_OFFSET_INDEX);
    if (offsetShapePtr_ == nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "optional input offset is null");
    } else {
        isInputOffset_ = 1;
    }
    moeInitRoutingCustomTilingData.set_isInputOffset(isInputOffset_);

    expandedXShapePtr_ = context_->GetOutputShape(OUTPUT_EXPANDED_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXShapePtr_);
    expandedRowIdxShapePtr_ = context_->GetOutputShape(OUTPUT_EXPANDED_ROW_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxShapePtr_);
    expertTokensCountOrCumsumShapePtr_ = context_->GetOutputShape(OUTPUT_EXPERT_TOKENS_COUNT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokensCountOrCumsumShapePtr_);
    expandedScaleShapePtr_ = context_->GetOutputShape(OUTPUT_EXPANDED_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedScaleShapePtr_);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    activeNumPtr_ = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVE_NUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, activeNumPtr_);
    expertCapacityPtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_CAPACITY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertCapacityPtr_);
    expertNumPtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_NUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertNumPtr_);
    dropPadModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dropPadModePtr_);
    expertTokensNumTypePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokensNumTypePtr_);
    expertTokensNumFlagPtr_ = attrs->GetAttrPointer<bool>(ATTR_EXPERT_TOKEN_NUM_FLAG_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokensNumFlagPtr_);
    quantModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantModePtr_);
    activeExpertRangeListPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_EXPERT_RANGE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, activeExpertRangeListPtr_);
    rowIdxTypePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_ROW_IDX_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rowIdxTypePtr_);
    return ge::GRAPH_SUCCESS;
}

void MoeInitRountingCustomTilingBase::ShowTilingData()
{   
    int64_t isFullloadInt = 1 ? isFullload_ == true : 0;
    OPS_LOG_I(context_->GetNodeName(), "isFullload: %ld, gatherFirstFullload: %ld, ep: %ld", isFullloadInt, gatherFirstFullload_, ep_);
}

int64_t MoeInitRountingCustomTilingBase::IsGatherFirstFullLoad() {
    if (ep_ == 0) {
        return 0;
    } else if (n_ >= gatherFirstN && (expertEnd_-expertStart_) * gatherFirstScale <= expertNum_) {
        return 1;
    }
    return 0;
}

bool MoeInitRountingCustomTilingBase::IsFullLoad() {
    int64_t perCoreTokens = 1;
    if (expertStart_ == 0 && expertEnd_ == expertNum_) {
        ep_ = 0;
        if (quantMode_ != 1) {
            perCoreTokens = n_ / aivNum;
            int64_t remainder = n_ % aivNum;
            // NUM_TWO is Max xRows need add 2 becauseof the left and right row may be another row.
            perCoreTokens = remainder <= 1 ? perCoreTokens + 1 : perCoreTokens + NUM_TWO;
        }
    } else {
        ep_ = 1;
        perCoreTokens = 1;
    }
    moeInitRoutingCustomTilingData.set_ep(ep_);

    if (totalLength_ > sortLoopMaxElement || this->dropPadMode_ == 1) {
        return false;
    }
    
    gatherFirstFullload_ = IsGatherFirstFullLoad();
    moeInitRoutingCustomTilingData.set_gatherFirstFullload(gatherFirstFullload_);
    int64_t tileLength = Align(this->totalLength_, int64_t(sizeof(int32_t)));
    int64_t sortNum = CeilDiv(tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;

    int64_t sortSpace = sortNum * sizeof(int32_t) * ONE_CORE_SORT_BUFFER;
    int64_t rowIdxSpace = sortNum * sizeof(int32_t) * NUM_THREE;
    int64_t expertSpace = CeilDiv(this->expertNum_ * int64_t(sizeof(int64_t)), ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * NUM_TWO;
    int64_t gatherSpace = CeilDiv(cols_ * inuptXDtypeSize_, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * perCoreTokens;
    int64_t remainUb = aicoreParams_.ubSize - sortSpace - rowIdxSpace - expertSpace - LENGTH_1024;

    if (quantMode_ == -1) {
        remainUb -= (gatherSpace + ONE_BLOCK_BYTE);
    } else if (quantMode_ == 0) {
        int64_t quantSpace = 0;
        int64_t xAlignedCount = Align(this->cols_, int64_t(sizeof(int8_t)));
        quantSpace = xAlignedCount * STATIC_QUANT_FULLLOAD_COLS_BUFFER * perCoreTokens;
        remainUb -= (gatherSpace + quantSpace);
    } else {
        int64_t quantSpace = CeilDiv(cols_, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE  * DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER;
        int64_t scaleOutSpace = ONE_BLOCK_BYTE * NUM_TWO;
        remainUb -= (quantSpace + scaleOutSpace);
    }
    return remainUb > 0;
}

bool MoeInitRountingCustomTilingBase::IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168() const
{
    OPS_LOG_I(context_->GetNodeName(), "Begin IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168() ...");
    bool result = false;

    // expert_range [0,256), quant_mode=DYNAMIC_QUANT
    const gert::Shape performXShape_X_1_7168 = gert::Shape({1, 7168});
    const gert::Shape performExpertIdxShape_X_1_7168 = gert::Shape({1, 8});
    const gert::Shape performScaleShape_X_1_7168 = gert::Shape({256, 7168});

    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxShapePtr_);
    if (nullptr == scaleShapePtr_) {
        result = false;
    } else if (xShapePtr_->GetStorageShape() == performXShape_X_1_7168 &&
               expertIdxShapePtr_->GetStorageShape() == performExpertIdxShape_X_1_7168 &&
               scaleShapePtr_->GetStorageShape() == performScaleShape_X_1_7168 && offsetShapePtr_ == nullptr &&
               context_->GetInputDesc(INPUT_X_INDEX)->GetDataType() == ge::DT_BF16 && expertStart_ == 0 &&
               expertEnd_ == ASSIST_NUM && quantMode_ == DYNAMIC_QUANT && expertTokensNumType_ == KEY_VALUE) {
        result = true;
    }
    OPS_LOG_I(context_->GetNodeName(), "End IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168() ...");
    return result;
}

PerformanceMode MoeInitRountingCustomTilingBase::GetPerformanceMode() const
{
    PerformanceMode result = PerformanceMode::COMMON;
    if (expertNum_ != ASSIST_NUM || (expertEnd_ - expertStart_) > PERFORMANCE_MODE_RANGE_MAX ||
        n_ < PERFORMANCE_MODE_BS_MIN || n_ > PERFORMANCE_MODE_BS_MAX || k_ != PERFORMANCE_MODE_TOP_K) {
        return result;
    }

    // Judge performance mode according to totalLength_
    if (totalLength_ < PERFORMANCE_MODE_MAX_ONE_CORE_GATHER) {
        OPS_LOG_I(context_->GetNodeName(), "totalLength_: %ld, PerformanceMode::ONE_CORE_GATHER_SORT", totalLength_);
        result = PerformanceMode::ONE_CORE_GATHER_SORT;
    } else if (totalLength_ <= PERFORMANCE_MODE_MAX_BATCH_SIZE_TOP_K) {
        OPS_LOG_I(context_->GetNodeName(), "totalLength_: %ld, PerformanceMode::MULTI_CORE_GATHER_SORT", totalLength_);
        result = PerformanceMode::MULTI_CORE_GATHER_SORT;
    }
    return result;
}

ge::graphStatus MoeInitRountingCustomTilingBase::CheckDtype() 
{
    auto inputXDtype_ = context_->GetInputDesc(INPUT_X_INDEX)->GetDataType();
    CHECK_FAIL(context_, inputXDtype_ != ge::DT_INT8 && inputXDtype_ != ge::DT_FLOAT16 && inputXDtype_ != ge::DT_BF16 && inputXDtype_ != ge::DT_FLOAT,
                "The data type of input_X should be INT8, FLOAT16, BF16, FLOAT.");
    CHECK_FAIL(context_, inputXDtype_ == ge::DT_INT8 && quantMode_ != UN_QUANT,
                "When input_X is INT8, quantization is not supported.");

    auto expertIdxDtype_ = context_->GetInputDesc(INPUT_EXPERT_IDX_INDEX)->GetDataType();
    CHECK_FAIL(context_, expertIdxDtype_ != ge::DT_INT32,
        "The data type of input_expertIdx should be INT32.");
    
    if (quantMode_ == STATIC_QUANT) {
        auto scaleDtype_ = context_->GetOptionalInputDesc(INPUT_SCALE_INDEX)->GetDataType();
        CHECK_FAIL(context_, scaleDtype_ != ge::DT_FLOAT,
            "The data type of input_scale should be FLOAT.");
        
        auto offsetDtype_ = context_->GetOptionalInputDesc(INPUT_OFFSET_INDEX)->GetDataType();
        CHECK_FAIL(context_, offsetDtype_ != ge::DT_FLOAT,
            "The data type of input_offset should be FLOAT.");
    } else {
        if (scaleShapePtr_ != nullptr) {
            auto scaleDtype_ = context_->GetOptionalInputDesc(INPUT_SCALE_INDEX)->GetDataType();
            CHECK_FAIL(context_, scaleDtype_ != ge::DT_FLOAT,
                "The data type of input_scale should be FLOAT.");
        }
    }

    auto expandedXDtype_ = context_->GetOutputDesc(OUTPUT_EXPANDED_X_INDEX)->GetDataType();
    CHECK_FAIL(context_,expandedXDtype_ != ge::DT_INT8 && expandedXDtype_ != ge::DT_FLOAT16 && expandedXDtype_ != ge::DT_BF16 && expandedXDtype_ != ge::DT_FLOAT,
        "The data type of output_expanded_X should be INT8, FLOAT16, BF16, FLOAT.");
    
    auto expandedRowIdxDtype_ = context_->GetOutputDesc(OUTPUT_EXPANDED_ROW_IDX_INDEX)->GetDataType();
    CHECK_FAIL(context_,expandedRowIdxDtype_ != ge::DT_INT32,
        "The data type of output_expanded_row_idx should be INT32.");

    auto expertTokensCountOrCusumDtype_ = context_->GetOutputDesc(OUTPUT_EXPERT_TOKENS_COUNT_INDEX)->GetDataType();
    CHECK_FAIL(context_,expertTokensCountOrCusumDtype_ != ge::DT_INT64,
        "The data type of output_expert_tokens_count_or_cumsum should be INT64.");
    
    if (quantMode_ == DYNAMIC_QUANT || (quantMode_ == UN_QUANT && scaleShapePtr_ != nullptr)) {
        auto expandedScaleDtype_ = context_->GetOutputDesc(OUTPUT_EXPANDED_SCALE_INDEX)->GetDataType();
        CHECK_FAIL(context_,expandedScaleDtype_ != ge::DT_FLOAT,
            "The data type of input_expanded_scale should be FLOAT.");
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingCustomTilingBase::DoOpTiling()
{   
    auto ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckDtype();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168()) {
        aivNum = totalLength_;
    }

    sortLoopMaxElement = (aicoreParams_.ubSize - aivNum * ONE_BLOCK_BYTE) / (NUM_FOUR * NUM_TWO * NUM_FOUR) /
                         SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;

    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4ExpertTokensCountCompute();
    Tiling4SrcToDstCompute();
    Tiling4SrcToDstDropPadCompute();
    Tiling4GatherOutCompute();
    isFullload_ = IsFullLoad();
    ShowTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingCustomTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRountingCustomTilingBase::GetTilingKey() const
{   
    if (isFullload_) {
        if (quantMode_ == UN_QUANT) {
            return UNQUANTIZED_FULLLOAD_TILINGKEY;
        } else if (quantMode_ == STATIC_QUANT) {
            return STATIC_QUANT_FULLLOAD_TILINGKEY;
        } else {
            return (DYNAMIC_QUANT_FULLLOAD_TILINGKEY + ep_ * DYNAMIC_QUANT_EPFULLLOAD_TILINGKEY 
                    + smoothType_ * DYNAMIC_QUANT_SMOOTHTYPE_FULLLOAD_TILINGKEY);
        }
    }
    else if (IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168()) {
        return PERFORMANCE_TILINGKEY_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168;
    } else if (PerformanceMode::ONE_CORE_GATHER_SORT == GetPerformanceMode() && quantMode_ == UN_QUANT &&
               rowIdxTytpe_ == SCATTER && expertTokensNumType_ == COUNT) {
        uint64_t sortMode = NUM_TWO;
        return static_cast<uint64_t>(TILINGKEY_BASE + sortMode * SORT_CORE_TILINGKEY_BASE +
                                     static_cast<uint64_t>(quantMode_ + 1) * QUANT_MODE_TILINGKEY_BASE +
                                     static_cast<uint64_t>(rowIdxTytpe_) * ROWIDX_TYPE_TILINGKEY_BASE + 
                                     static_cast<uint64_t>(dropPadMode_) * DROP_MODE_TILINGKEY_BASE);
    } else if (PerformanceMode::MULTI_CORE_GATHER_SORT == GetPerformanceMode() && quantMode_ == UN_QUANT &&
               rowIdxTytpe_ == SCATTER && expertTokensNumType_ == COUNT) {
        uint64_t sortMode = 3;
        return static_cast<uint64_t>(TILINGKEY_BASE + sortMode * SORT_CORE_TILINGKEY_BASE +
                                     static_cast<uint64_t>(quantMode_ + 1) * QUANT_MODE_TILINGKEY_BASE +
                                     static_cast<uint64_t>(rowIdxTytpe_) * ROWIDX_TYPE_TILINGKEY_BASE + 
                                     static_cast<uint64_t>(dropPadMode_) * DROP_MODE_TILINGKEY_BASE);
    }
    return static_cast<uint64_t>(TILINGKEY_BASE + static_cast<uint64_t>(sortMode_) * SORT_CORE_TILINGKEY_BASE +
                                 static_cast<uint64_t>(quantMode_ + 1) * QUANT_MODE_TILINGKEY_BASE +
                                 static_cast<uint64_t>(rowIdxTytpe_) * ROWIDX_TYPE_TILINGKEY_BASE + 
                                 static_cast<uint64_t>(dropPadMode_) * DROP_MODE_TILINGKEY_BASE);
}

ge::graphStatus MoeInitRountingCustomTilingBase::GetWorkspaceSize()
{
    size_t sortWorkspaceSize =
        sizeof(float) * static_cast<size_t>(totalLength_ * NUM_TWO * NUM_THREE);
    size_t coreSyncWorkspaceSize =
        moeInitRoutingCustomTilingData.get_coreNum() * SORT32_ALIGN_ELEMENT * NUM_TWO;
    size_t scatterWorkspaceSize = sizeof(int32_t) * static_cast<size_t>(totalLength_);
    size_t expertIdxValueWorkspaceSize = sizeof(int32_t) * static_cast<size_t>(aivNum) * 2U;
    size_t expertTokensCountWorkspaceSize = sizeof(int32_t) * static_cast<size_t>((expertEnd_ - expertStart_));
    int64_t expertTokenTotalCountWorkspace = AlignBytes(1, static_cast<int64_t>(sizeof(int32_t)));
    int64_t quantTempWorkspaceSize = aivNum * cols_ * static_cast<int64_t>(sizeof(float));
    workspaceSize_ = sortWorkspaceSize + coreSyncWorkspaceSize + scatterWorkspaceSize + expertTokensCountWorkspaceSize +
                     expertTokenTotalCountWorkspace + SIZE_16 * LENGTH_1024 * LENGTH_1024;
    if (quantMode_ == DYNAMIC_QUANT) {
        workspaceSize_ += quantTempWorkspaceSize;
    }
    if (dropPadMode_ == DROP_PAD) {
        workspaceSize_ += expertIdxValueWorkspaceSize;
    }
    OPS_LOG_I(context_->GetNodeName(), "Allocate workspaceSize is: %ld.", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingCustomTilingBase::PostTiling()
{
    context_->SetBlockDim(aivNum);
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    moeInitRoutingCustomTilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                            context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(moeInitRoutingCustomTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
void MoeInitRountingCustomTilingBase::Tinlig4VBSOneCoreCompute(MoeCustomVBSComputeTilingData *tilingData)
{
    tilingData->set_needCoreNum(1);
    tilingData->set_perCoreElements(totalLength_);
    tilingData->set_perCoreLoops(1);
    tilingData->set_perCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(1);
    tilingData->set_lastCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLastLoopElements(tilingData->get_perCoreElements());
}

void MoeInitRountingCustomTilingBase::Tinlig4VBSMultiCoreCompute(MoeCustomVBSComputeTilingData *tilingData)
{
    int64_t needCoreNum = CeilDiv(totalLength_, sortLoopMaxElement);
    needCoreNum = static_cast<int64_t>(std::pow(NUM_FOUR, CeilLog4(needCoreNum)));
    needCoreNum = std::min(needCoreNum, aivNum);

    if (needCoreNum == 0) {
        OPS_LOG_E(context_->GetNodeName(), "Variate needCoreNum cannot be 0.");
        return;
    }
    int64_t perCoreElements = (needCoreNum == 0) ? 0 : (totalLength_ / needCoreNum);
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength_ - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = CeilDiv(totalLength_, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    tilingData->set_needCoreNum(needCoreNum);
    do {
        tilingData->set_perCoreElements(perCoreElements);
        tilingData->set_perCoreLoops(
            CeilDiv(tilingData->get_perCoreElements(), sortLoopMaxElement));
        tilingData->set_perCorePerLoopElements(std::min(tilingData->get_perCoreElements(), sortLoopMaxElement));

        tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements() -
                                                (tilingData->get_perCoreLoops() - 1) *
                                                    tilingData->get_perCorePerLoopElements());

        tilingData->set_lastCoreElements(totalLength_ -
                                         (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
        tilingData->set_lastCoreLoops(tilingData->get_perCoreLoops());
        int64_t lastCorePerLoopElements =
            CeilDiv(CeilDiv(tilingData->get_lastCoreElements(), tilingData->get_lastCoreLoops()),
                               SORT32_ALIGN_ELEMENT) *
            SORT32_ALIGN_ELEMENT;
        tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
        tilingData->set_lastCoreLastLoopElements(tilingData->get_lastCoreElements() -
                                                 (tilingData->get_lastCoreLoops() - 1) *
                                                     tilingData->get_lastCorePerLoopElements());
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (tilingData->get_lastCoreLastLoopElements() <= 0 && perCoreElements > 0);
    if (tilingData->get_lastCoreLastLoopElements() <= 0) {
        OPS_LOG_E(context_->GetNodeName(), "vbs tiling failed");
        return;
    }
}

void MoeInitRountingCustomTilingBase::Tiling4VBSCompute()
{
    if (totalLength_ <= sortLoopMaxElement) {
        sortMode_ = 0;
    } else {
        sortMode_ = 1;
    }

    auto tilingData = &moeInitRoutingCustomTilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    if (sortMode_ == 0UL) {
        Tinlig4VBSOneCoreCompute(tilingData);
        return;
    }
    Tinlig4VBSMultiCoreCompute(tilingData);
}

void MoeInitRountingCustomTilingBase::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &moeInitRoutingCustomTilingData.vbsComputeParamsOp;
    auto tilingData = &moeInitRoutingCustomTilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = CeilDiv(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM);
    tilingData->set_needCoreNum(needCoreNum);
}

void MoeInitRountingCustomTilingBase::Tiling4SortOutCompute()
{
    auto tilingData = &moeInitRoutingCustomTilingData.sortOutComputeParamsOp;
    tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}

void MoeInitRountingCustomTilingBase::Tiling4ExpertTokensCountCompute()
{
    auto tilingData = &moeInitRoutingCustomTilingData.expertTokensCountTilingDataOp;
    int64_t totalElements = moeInitRoutingCustomTilingData.get_n() * moeInitRoutingCustomTilingData.get_k();
    int64_t perCoreElements = CeilDiv(totalElements, aivNum);
    int64_t needCoreNum = CeilDiv(totalElements, perCoreElements);
    int64_t lastCoreElements = totalElements - (needCoreNum - 1) * perCoreElements;
    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreElements(perCoreElements);
    tilingData->set_lastCoreElements(lastCoreElements);

    int64_t expertNumElement = (moeInitRoutingCustomTilingData.get_expertTokensNumType() != KEY_VALUE) ?
                                   moeInitRoutingCustomTilingData.get_actualExpertNum() :
                                   (moeInitRoutingCustomTilingData.get_actualExpertNum() + 1) * DIM_TWO;

    int64_t maxElementsPerLoop =
        (static_cast<int64_t>(aicoreParams_.ubSize) -
         CeilAlign(expertNumElement, ONE_BLOCK_BYTE) *
             (static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO + static_cast<int64_t>(sizeof(int64_t))) -
         ONE_BLOCK_BYTE) / static_cast<int64_t>(sizeof(int32_t));
    int64_t perCoreLoops = CeilDiv(perCoreElements, maxElementsPerLoop);
    int64_t perCorePerLoopElements = CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;

    tilingData->set_perCoreLoops(perCoreLoops);
    tilingData->set_perCorePerLoopElements(perCorePerLoopElements);
    tilingData->set_perCoreLastLoopElements(perCoreLastLoopElements);

    int64_t lastCoreLoops = CeilDiv(lastCoreElements, maxElementsPerLoop);
    int64_t lastCorePerLoopElements = CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;

    tilingData->set_lastCoreLoops(lastCoreLoops);
    tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
    tilingData->set_lastCoreLastLoopElements(lastCoreLastLoopElements);

    OPS_LOG_I(context_->GetNodeName(),
            "ExpertTokensCountCompute Tilingdata, needCoreNum is: %ld, perCoreElements is: %ld, lastCoreElements is: "
            "%ld, maxElementsPerLoop is: %ld, perCoreLoops is: %ld, perCorePerLoopElements is: %ld, "
            "perCoreLastLoopElements "
            "is: %ld, lastCoreLoops is: %ld, lastCorePerLoopElements is: %ld, lastCoreLastLoopElements is: %ld.",
            needCoreNum, perCoreElements, lastCoreElements, maxElementsPerLoop, perCoreLoops, perCorePerLoopElements,
            perCoreLastLoopElements, lastCoreLoops, lastCorePerLoopElements, lastCoreLastLoopElements);
}

void MoeInitRountingCustomTilingBase::Tiling4SrcToDstDropPadCompute()
{
    if (quantMode_ == DYNAMIC_QUANT && dropPadMode_ == DROP_PAD) {
        MoeInitRountingCustomTilingBase::Tiling4SrcToDstDropPadDynamicCompute();
        return;
    }

    auto tilingData = &moeInitRoutingCustomTilingData.srcToDstDropPadParamsOp;

    int64_t perCoreRows = CeilDiv(totalLength_, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = CeilDiv(totalLength_, perCoreRows);
    tilingData->set_needCoreNum(needCoreNum);
    int64_t cols = moeInitRoutingCustomTilingData.get_cols();
    tilingData->set_perCoreRows(perCoreRows);
    int64_t lastCoreRows = totalLength_ - perCoreRows * (needCoreNum - 1);
    tilingData->set_lastCoreRows(lastCoreRows);
    bool needScaleCopy = (isInputScale_ != 0 && quantMode_ == -1);
    int64_t inuptXDtypeSize = inuptXDtypeSize_ == SIZE_INT8 ? SIZE_INT16 : inuptXDtypeSize_;

    int64_t rowSize =
        (perCoreRows * sizeof(int32_t) * NUM_TWO + ONE_BLOCK_BYTE + ONE_BLOCK_BYTE * needScaleCopy + ONE_BLOCK_BYTE - 1) /
        ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t colSize = (cols * inuptXDtypeSize + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

    if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
        SetGatherTilingData(tilingData, perCoreRows, lastCoreRows, cols);
    } else {
        int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
        int64_t baseMaxColsSize =
            (baseMaxCols * inuptXDtypeSize + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        int64_t basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - baseMaxColsSize - ONE_BLOCK_BYTE -
                                     ONE_BLOCK_BYTE * needScaleCopy) /static_cast<int64_t>(sizeof(int32_t))
                                     / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        if (cols < MAX_COLS_ONE_LOOP) {
            basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - colSize - ONE_BLOCK_BYTE -
                                 ONE_BLOCK_BYTE * needScaleCopy) / static_cast<int64_t>(sizeof(int32_t)) 
                                 / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = (static_cast<int64_t>(aicoreParams_.ubSize) - rowSize) / inuptXDtypeSize / ONE_BLOCK_BYTE *
                          ONE_BLOCK_BYTE;
        }
        tilingData->set_perLoopCols(std::min(baseMaxCols, cols));
        tilingData->set_lastLoopCols(GetPerOrLastValue(cols, baseMaxCols));
        tilingData->set_colLoops((cols + baseMaxCols - 1) / baseMaxCols);

        tilingData->set_perCorePerLoopRows(std::min(perCoreRows, basePerLoopMaxRows));
        tilingData->set_perCoreLastLoopRows(GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
        tilingData->set_perCoreLoops((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);

        tilingData->set_lastCorePerLoopRows(std::min(lastCoreRows, basePerLoopMaxRows));
        tilingData->set_lastCoreLastLoopRows(GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
        tilingData->set_lastCoreLoops((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
    }
}

void MoeInitRountingCustomTilingBase::SetGatherTilingData(
    MoeCustomSrcToDstCapacityComputeTilingData* tilingData, int64_t perCoreRows, int64_t lastCoreRows, int64_t cols)
{
    tilingData->set_perCorePerLoopRows(perCoreRows);
    tilingData->set_perCoreLastLoopRows(perCoreRows);
    tilingData->set_lastCorePerLoopRows(lastCoreRows);
    tilingData->set_lastCoreLastLoopRows(lastCoreRows);
    tilingData->set_perCoreLoops(1);
    tilingData->set_lastCoreLoops(1);
    tilingData->set_perLoopCols(cols);
    tilingData->set_lastLoopCols(cols);
    tilingData->set_colLoops(1);
}

void MoeInitRountingCustomTilingBase::SetGatherTilingDataCols(
    MoeCustomSrcToDstCapacityComputeTilingData* tilingData, int64_t baseMaxCols, int64_t cols)
{
    tilingData->set_perLoopCols(std::min(baseMaxCols, cols));
    tilingData->set_lastLoopCols(GetPerOrLastValue(cols, baseMaxCols));
    tilingData->set_colLoops(baseMaxCols == 0 ? 0 : (cols + baseMaxCols - 1) / baseMaxCols);
}

void MoeInitRountingCustomTilingBase::SetGatherTilingDataRows(
    MoeCustomSrcToDstCapacityComputeTilingData* tilingData, int64_t perCoreRows, int64_t lastCoreRows,
    int64_t basePerLoopMaxRows)
{
    tilingData->set_perCorePerLoopRows(std::min(perCoreRows, basePerLoopMaxRows));
    tilingData->set_perCoreLastLoopRows(GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
    tilingData->set_perCoreLoops(
        basePerLoopMaxRows == 0 ? 0 : (perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);

    tilingData->set_lastCorePerLoopRows(std::min(lastCoreRows, basePerLoopMaxRows));
    tilingData->set_lastCoreLastLoopRows(GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
    tilingData->set_lastCoreLoops(
        basePerLoopMaxRows == 0 ? 0 : (lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
}

void MoeInitRountingCustomTilingBase::Tiling4SrcToDstDropPadDynamicCompute()
{
    auto tilingData = &moeInitRoutingCustomTilingData.srcToDstDropPadDynamicParamsOp;

    int64_t perCoreRows = CeilDiv(totalLength_, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    tilingData->set_needCoreNum(CeilDiv(totalLength_, perCoreRows));
    int64_t cols = moeInitRoutingCustomTilingData.get_cols();
    tilingData->set_perCoreRows(perCoreRows);
    int64_t lastCoreRows = totalLength_ - perCoreRows * (tilingData->get_needCoreNum() - 1);
    tilingData->set_lastCoreRows(lastCoreRows);

    int64_t rowSize = AlignBytes(perCoreRows, static_cast<int64_t>(sizeof(int32_t))) * NUM_FOUR;
    int64_t colSize = AlignBytes(cols, static_cast<int64_t>(sizeof(int8_t))) * DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
    int64_t scaleSize = DYNAMIC_QUANT_SCALE_SIZE_64;
    if (rowSize + colSize + scaleSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
        SetGatherTilingData(tilingData, perCoreRows, lastCoreRows, cols);
    } else {
        int64_t baseMaxCols = MAX_COLS_DYNAMIC_QUANT;
        int64_t totalColSize = AlignBytes(baseMaxCols, static_cast<int64_t>(sizeof(int8_t))) * DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
        int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
        int64_t basePerLoopMaxRows = AlignOneBlockByteCeil((ubSize - totalColSize - scaleSize) / SIZE_INT32) / NUM_FOUR;
        if (cols < MAX_COLS_DYNAMIC_QUANT) {
            basePerLoopMaxRows = AlignOneBlockByteCeil((ubSize - colSize - scaleSize) / SIZE_INT32) / NUM_FOUR;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = AlignOneBlockByteCeil(ubSize - rowSize - scaleSize) / DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
        }
        SetGatherTilingDataCols(tilingData, baseMaxCols, cols);
        SetGatherTilingDataRows(tilingData, perCoreRows, lastCoreRows, basePerLoopMaxRows);
    }
}

void MoeInitRountingCustomTilingBase::Tiling4SrcToDstCompute()
{
    auto tilingData = &moeInitRoutingCustomTilingData.srcToDstComputeParamsOp;

    int64_t useCore = aivNum;
    int64_t remainUbSize = aicoreParams_.ubSize - ASSIST_NUM * sizeof(int32_t) - ONE_BLOCK_BYTE * (ASSIST_NUM + 1);
    int64_t perLoopMaxElements = remainUbSize / (ONE_BLOCK_BYTE + SIZE_INT32);
    int64_t perCoreElements = CeilDiv(totalLength_, useCore);
    if (perCoreElements <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = CeilDiv(totalLength_, perCoreElements);
    tilingData->set_needCoreNum(needCoreNum);
    int64_t lastCoreElements = totalLength_ - perCoreElements * (needCoreNum - 1);

    tilingData->set_perCoreElements(perCoreElements);
    tilingData->set_lastCoreElements(lastCoreElements);
    int64_t perCoreLoops = CeilDiv(perCoreElements, perLoopMaxElements);
    int64_t perCorePerLoopElements = CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;

    int64_t lastCoreLoops = CeilDiv(lastCoreElements, perLoopMaxElements);
    int64_t lastCorePerLoopElements = CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;

    tilingData->set_perCoreLoops(perCoreLoops);
    tilingData->set_perCorePerLoopElements(perCorePerLoopElements);
    tilingData->set_perCoreLastLoopElements(perCoreLastLoopElements);
    tilingData->set_lastCoreLoops(lastCoreLoops);
    tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
    tilingData->set_lastCoreLastLoopElements(lastCoreLastLoopElements);
}

void MoeInitRountingCustomTilingBase::Tiling4GatherOutCompute()
{
    auto tilingData = &moeInitRoutingCustomTilingData.gatherOutComputeParamsOp;
    int64_t perCoreIndicesElements = CeilDiv(totalLength_, aivNum);
    if (perCoreIndicesElements <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = CeilDiv(totalLength_, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = totalLength_ - (needCoreNum - 1) * perCoreIndicesElements;

    int64_t perLoopCols = moeInitRoutingCustomTilingData.get_cols();
    int64_t colMultiple = NUM_TWO * inuptXDtypeSize_;
    int64_t rowMultiple = NUM_TWO;
    if (quantMode_ == DYNAMIC_QUANT) {
        colMultiple = DYNAMIC_QUANT_COLS_BUFFER;
        rowMultiple = NUM_FOUR;
    }
    if (quantMode_ == STATIC_QUANT) {
        colMultiple = SIZE_INT8 * NUM_TWO + SIZE_FP32 + SIZE_INT16 + inuptXDtypeSize_ * NUM_TWO;
        rowMultiple = NUM_TWO;
    }
    int64_t perLoopMaxIndicesElements =
        (static_cast<int64_t>(aicoreParams_.ubSize) - Align(perLoopCols, inuptXDtypeSize_) * colMultiple -
         ONE_BLOCK_BYTE * NUM_TWO) /
        rowMultiple / static_cast<int64_t>(sizeof(int32_t));
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = CeilDiv(perLoopCols, NUM_TWO);
        perLoopMaxIndicesElements = (static_cast<int64_t>(aicoreParams_.ubSize) -
                                     Align(perLoopCols, inuptXDtypeSize_) * colMultiple - ONE_BLOCK_BYTE * NUM_TWO) /
                                    rowMultiple / static_cast<int64_t>(sizeof(int32_t));
        OPS_LOG_I(context_->GetNodeName(), "perLoopCols is: %ld, perLoopMaxIndicesElements is: %ld", perLoopCols,
                perLoopMaxIndicesElements);
    }
    int64_t colsLoops = CeilDiv(moeInitRoutingCustomTilingData.get_cols(), perLoopCols);
    int64_t lastLoopCols = moeInitRoutingCustomTilingData.get_cols() - (colsLoops - 1) * perLoopCols;
    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreIndicesElements(perCoreIndicesElements);
    tilingData->set_lastCoreIndicesElements(lastCoreIndicesElements);
    tilingData->set_colsLoops(colsLoops);
    tilingData->set_perLoopCols(perLoopCols);
    tilingData->set_lastLoopCols(lastLoopCols);

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    tilingData->set_perCoreIndicesLoops(perCoreIndicesLoops);
    tilingData->set_perCorePerLoopIndicesElements(perCorePerLoopIndicesElements);
    tilingData->set_perCoreLastLoopIndicesElements(perCoreLastLoopIndicesElements);

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    tilingData->set_lastCoreIndicesLoops(lastCoreIndicesLoops);
    tilingData->set_lastCorePerLoopIndicesElements(lastCorePerLoopIndicesElements);
    tilingData->set_lastCoreLastLoopIndicesElements(lastCoreLastLoopIndicesElements);

    OPS_LOG_I(
        context_->GetNodeName(),
        "GatherOut Tilingdata, needCoreNum is: %ld, perCoreIndicesElements is: %ld, lastCoreIndicesElements is: %ld, "
        "colsLoops is: %ld, perLoopCols is: %ld, lastLoopCols is: %ld, perCoreIndicesLoops is: %ld, "
        "perCorePerLoopIndicesElements is: %ld, perCoreLastLoopIndicesElements is: %ld, lastCoreIndicesLoops is: "
        "%ld, lastCorePerLoopIndicesElements is: "
        "%ld, lastCoreLastLoopIndicesElements is: %ld.",
        needCoreNum, perCoreIndicesElements, lastCoreIndicesElements, colsLoops, perLoopCols, lastLoopCols,
        perCoreIndicesLoops, perCorePerLoopIndicesElements, perCoreLastLoopIndicesElements, lastCoreIndicesLoops,
        lastCorePerLoopIndicesElements, lastCoreLastLoopIndicesElements);
}

REGISTER_TILING_TEMPLATE("MoeInitRoutingCustom", MoeInitRountingCustomTilingBase, 10000); // If not 910_95, fallback to this.
} // namespace optiling