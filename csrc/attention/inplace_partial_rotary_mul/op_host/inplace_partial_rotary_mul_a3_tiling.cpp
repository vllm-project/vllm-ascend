/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file inplace_partial_rotary_mul_tiling.cpp
 * \brief
 */

#include "inplace_partial_rotary_mul_tiling.h"

namespace optiling {
constexpr int64_t TILING_KEY_FLOAT16 = 0;
constexpr int64_t TILING_KEY_BFLOAT16 = 10;
constexpr int64_t TILING_KEY_FLOAT32 = 20;
constexpr int64_t TILING_KEY_UNPAD = 0;
constexpr int64_t TILING_KEY_PAD = 1;
constexpr int64_t TILING_KEY_SPLIT_S = 0;
constexpr int64_t TILING_KEY_SPLIT_BS = 100;
constexpr int64_t TILING_KEY_SPLIT_BSN = 200;
constexpr int64_t FP16_BF16_DTYPE_SIZE = 2;
constexpr int64_t FP32_DTYPE_SIZE = 4;
constexpr int64_t INT32_DTYPE_SIZE = 4;
constexpr int64_t REPEAT_FP32 = 64;
constexpr int64_t TBUF_SIZE = 0;
constexpr int64_t ALIGN_32 = 8;
constexpr int64_t ALIGN_16 = 16;
constexpr int64_t IO_NUM = 3; // sin、cos -> tri
constexpr int64_t BASE_KEY = 2000;
constexpr int64_t CONST_4 = 4;

class InplacePartialRotaryMulTiling
{
public:
    explicit InplacePartialRotaryMulTiling(gert::TilingContext* context) : context_(context){};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInput();
    ge::graphStatus CalTilingData();
    ge::graphStatus TilingSplitN(int64_t numHeads, int64_t headDimAlign, int64_t ubSize,
                                 ge::DataType dataDtype);
    ge::graphStatus TilingSplitB(int64_t batchSize, int64_t numHeads, int64_t headDimAlign,
                                 int64_t ubSize, ge::DataType dataDtype);
    ge::graphStatus TilingSplitS();
    ge::graphStatus TilingSplit();
    void FillTilingData();
    void PrintTilingData() const;
    void PrintInfo();

private:
    int64_t coreNum_ = 0;
    int64_t ubSize_=0;
    int64_t dtypeX = 0;
    int64_t repeatNum_ = 0;
    bool isBrc_ = true;
    int64_t dim0_ = 0;
    int64_t dim1_ = 0;
    int64_t dim2_ = 0;
    int64_t end_ =0;
    int64_t tilingKey_ =1;
    bool isAlign_ = false;
    bool isSpecail_ = false;
    int64_t oneBlockSize_ = 0;
    int64_t dtypeSize_ = 2;
    int64_t xdim0_ = 0;
    int64_t xdim1_ = 0;
    int64_t xdim2_ = 0;
    int64_t xdim3_ = 0;
    int64_t r1dim0_ = 0;
    int64_t r1dim1_ = 0;
    int64_t r1dim2_ = 0;
    int64_t r1dim3_ = 0;

    // tiingdata
    int64_t usedCoreNum_ = 0;
    int64_t numHead_ = 0;
    int64_t headDim_ = 0;
    int64_t allHeadDim_ = 0;
    int64_t coreTUbLoopTime_ = 0;
    int64_t coreBUbLoopTime_ = 0;
    int64_t coreTUbLoopTail_ = 0;
    int64_t coreBUbLoopTail_ = 0;
    int64_t ubFactor_ = 0;
    int64_t start_=0;
    int64_t blockFactor_=0;
    gert::TilingContext* context_ = nullptr;
    RopeRegbaseTilingData tilingData_;
};
int64_t GetCeilInt(int64_t value1, int64_t value2)
{
    if (value2 == 0)
        return value2;
    return (value1 + value2 - 1) / value2;
}

int64_t GetDiv(int64_t value1, int64_t value2)
{
    if (value2 == 0)
        return value2;
    return value1 / value2;
}

int64_t GetDivRem(int64_t value1, int64_t value2)
{
    if (value2 == 0)
        return value2;
    return value1 % value2;
}
void InplacePartialRotaryMulTiling::FillTilingData()
{   
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_numHead(numHead_);
    tilingData_.set_headDim(headDim_);
    tilingData_.set_allHeadDim(allHeadDim_);
    tilingData_.set_coreTUbLoopTime(coreTUbLoopTime_);
    tilingData_.set_coreBUbLoopTime(coreBUbLoopTime_);
    tilingData_.set_coreTUbLoopTail(coreTUbLoopTail_);
    tilingData_.set_coreBUbLoopTail(coreBUbLoopTail_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_start(start_);
    tilingData_.set_blockFactor(blockFactor_);
}
void InplacePartialRotaryMulTiling::PrintTilingData() const
{
    OPS_LOG_I(context_->GetNodeName(), "InplacePartialRotaryMulTiling  begin print.");
    OPS_LOG_I(context_->GetNodeName(), "usedCoreNum = %ld.", usedCoreNum_);
    OPS_LOG_I(context_->GetNodeName(), "numHead_ = %ld.", numHead_);
    OPS_LOG_I(context_->GetNodeName(), "headDim_ = %ld.", headDim_);
    OPS_LOG_I(context_->GetNodeName(), "allHeadDim_ = %ld.", allHeadDim_);
    OPS_LOG_I(context_->GetNodeName(), "coreTUbLoopTime_ = %ld.", coreTUbLoopTime_);
    OPS_LOG_I(context_->GetNodeName(), "coreBUbLoopTime_ = %ld.", coreBUbLoopTime_);
    OPS_LOG_I(context_->GetNodeName(), "coreTUbLoopTail_ = %ld.", coreTUbLoopTail_);
    OPS_LOG_I(context_->GetNodeName(), "coreBUbLoopTail_ = %ld.", coreBUbLoopTail_);
    OPS_LOG_I(context_->GetNodeName(), "ubFactor = %ld.", ubFactor_);
    OPS_LOG_I(context_->GetNodeName(), "start_ = %ld.", start_);
    OPS_LOG_I(context_->GetNodeName(), "blockFactor_ = %ld.", blockFactor_);
    OPS_LOG_I(context_->GetNodeName(), "tilingKey = %ld.", tilingKey_);
}
void InplacePartialRotaryMulTiling::PrintInfo()
{
    OPS_LOG_I(context_->GetNodeName(), "usedCoreNum = %ld.", tilingData_.get_usedCoreNum());
    OPS_LOG_I(context_->GetNodeName(), "start = %ld.", tilingData_.get_start());
    OPS_LOG_I(context_->GetNodeName(), "allHeadDim = %ld.", tilingData_.get_allHeadDim());
    OPS_LOG_I(context_->GetNodeName(), " batchSize=%ld.", tilingData_.get_batchSize());
    OPS_LOG_I(context_->GetNodeName(), " seqLen=%ld.", tilingData_.get_seqLen());
    OPS_LOG_I(context_->GetNodeName(), " numHeads=%ld.", tilingData_.get_numHeads());
    OPS_LOG_I(context_->GetNodeName(), " headDim=%ld.", tilingData_.get_headDim());
    OPS_LOG_I(context_->GetNodeName(), " frontCoreNum=%ld.", tilingData_.get_frontCoreNum());
    OPS_LOG_I(context_->GetNodeName(), " tailCoreNum=%ld.", tilingData_.get_tailCoreNum());
    OPS_LOG_I(context_->GetNodeName(), " coreCalcNum=%ld.", tilingData_.get_coreCalcNum());
    OPS_LOG_I(context_->GetNodeName(), " coreCalcTail=%ld.", tilingData_.get_coreCalcTail());
    OPS_LOG_I(context_->GetNodeName(), " ubCalcNum=%ld.", tilingData_.get_ubCalcNum());
    OPS_LOG_I(context_->GetNodeName(), " ubCalcLoop=%ld.", tilingData_.get_ubCalcLoop());
    OPS_LOG_I(context_->GetNodeName(), " ubCalcTail=%ld.", tilingData_.get_ubCalcTail());
    OPS_LOG_I(context_->GetNodeName(), " ubCalcTailNum=%ld.", tilingData_.get_ubCalcTailNum());
    OPS_LOG_I(context_->GetNodeName(), " ubCalcTailLoop=%ld.", tilingData_.get_ubCalcTailLoop());
    OPS_LOG_I(context_->GetNodeName(), " ubCalcTailTail=%ld.", tilingData_.get_ubCalcTailTail());
    OPS_LOG_I(context_->GetNodeName(), " ubCalcBNum=%ld.", tilingData_.get_ubCalcBNum());
    OPS_LOG_I(context_->GetNodeName(), " ubCalcBLoop=%ld.", tilingData_.get_ubCalcBLoop());
    OPS_LOG_I(context_->GetNodeName(), " ubCalcBTail=%ld.", tilingData_.get_ubCalcBTail());
    OPS_LOG_I(context_->GetNodeName(), " ubCalcNNum=%ld.", tilingData_.get_ubCalcNNum());
    OPS_LOG_I(context_->GetNodeName(), " ubCalcNLoop=%ld.", tilingData_.get_ubCalcNLoop());
    OPS_LOG_I(context_->GetNodeName(), " ubCalcNTail=%ld.", tilingData_.get_ubCalcNTail());
    OPS_LOG_I(context_->GetNodeName(), "tilingKey = %ld.", tilingKey_);
}
ge::graphStatus InplacePartialRotaryMulTiling::CalTilingData()
{
    OPS_ERR_IF(!isSpecail_, OPS_LOG_I("Tiling4InplacePartialRotaryMul", "not specail"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!isAlign_, OPS_LOG_I("Tiling4InplacePartialRotaryMul", " d not align"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(xdim3_ > REPEAT_FP32, OPS_LOG_I("Tiling4InplacePartialRotaryMul", "D is repeat one repeat"),
               return ge::GRAPH_FAILED);
    int64_t ubNum = ubSize_ / sizeof(float);
    int64_t last = ubNum - dim1_*dim2_;
    int64_t preCoreNumFactor = (dim0_ + coreNum_ - 1) / coreNum_;
    usedCoreNum_ = (dim0_ + preCoreNumFactor - 1) / preCoreNumFactor;
    int64_t tailCoreNum = dim0_ - preCoreNumFactor *(usedCoreNum_ -1);
    blockFactor_ = preCoreNumFactor;
    ubFactor_ = last / (CONST_4*dim1_*dim2_ + CONST_4*dim2_);
    if (ubFactor_ > preCoreNumFactor) {
        ubFactor_ = preCoreNumFactor;
    }

    OPS_LOG_I(context_->GetNodeName(), "ubFactor_ = %ld.", ubFactor_);
    OPS_ERR_IF(ubFactor_ <= 0, OPS_LOG_I("Tiling4InplacePartialRotaryMul", " is large nout support"),
               return ge::GRAPH_FAILED);
    coreBUbLoopTime_ = (preCoreNumFactor + ubFactor_ -1) /ubFactor_;
    coreBUbLoopTail_ = preCoreNumFactor % ubFactor_;
    if (coreBUbLoopTail_ == 0) {
        coreBUbLoopTail_ = ubFactor_;
    }
    coreTUbLoopTime_ = (tailCoreNum + ubFactor_ -1) / ubFactor_;
    coreTUbLoopTail_ = tailCoreNum % ubFactor_;
    if (coreTUbLoopTail_ == 0) {
        coreTUbLoopTail_ = ubFactor_;
    }
    
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InplacePartialRotaryMulTiling::Init()
{
    OPS_LOG_I(context_->GetNodeName(), "Tiling4InplacePartialRotaryMul Init running.");
    OPS_ERR_IF(context_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Tiling4InplacePartialRotaryMul", "Tiling context is null"),
               return ge::GRAPH_FAILED);
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Tiling4InplacePartialRotaryMul", "Tiling platformInfo is null"),
               return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(
        coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OPS_ERR_IF(
        ubSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);
    OPS_LOG_I(context_->GetNodeName(),"coreNum_ is %ld, ubSize_ %ld ",coreNum_, ubSize_);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InplacePartialRotaryMulTiling::CheckInput()
{
    auto xInput = context_->GetInputShape(0);
    auto inputR1 = context_->GetInputShape(1);
    auto inputR2 = context_->GetInputShape(1);
    auto dataDtype = context_->GetInputDesc(0)->GetDataType();
    if (dataDtype == ge::DT_FLOAT16 || dataDtype == ge::DT_BF16) {
        dtypeSize_ = FP16_BF16_DTYPE_SIZE;
        oneBlockSize_ = ALIGN_16;
    } else {
        dtypeSize_ = FP32_DTYPE_SIZE;
        oneBlockSize_ = ALIGN_32;
    }
     
    OPS_ERR_IF(xInput == nullptr || inputR1 == nullptr || inputR2 == nullptr, OPS_LOG_E(context_->GetNodeName(), "get input nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape xShape = xInput->GetStorageShape();
    int64_t dimNum = xShape.GetDimNum();
    gert::Shape inputR1Shape = inputR1->GetStorageShape();
    gert::Shape inputR2Shape = inputR2->GetStorageShape();
    int64_t dimNumR1 = inputR1Shape.GetDimNum();
    int64_t dimNumR2 = inputR2Shape.GetDimNum();
    auto inputShape = xInput->GetStorageShape();
    OPS_ERR_IF(dimNum != CONST_4,
        OPS_LOG_E(context_->GetNodeName(), "xInput dim:%ld, should be 4.", dimNum),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(dimNumR1 != CONST_4 || dimNumR2 != CONST_4,
        OPS_LOG_E(context_->GetNodeName(), "dimNumR1:%ld dimNumR2 %ld,  r1 r2 dim must be4", dimNumR1, dimNumR2),
        return ge::GRAPH_FAILED);
    for (int64_t i = 0 ; i < CONST_4; i++) {
        int64_t r1dim  = inputR1Shape.GetDim(i);
        int64_t r2dim  = inputR2Shape.GetDim(i);
        if (r1dim != r2dim) {
            OPS_LOG_E(context_->GetNodeName(), "i is %d r1dim is %ld, r2dim is %ld, not equal",i,r1dim,r2dim);
            return ge::GRAPH_FAILED;
        }
    }
    dim0_ = xShape.GetDim(0);
    xdim0_ = dim0_;
    xdim1_ = xShape.GetDim(1);
    xdim2_ = xShape.GetDim(2);
    allHeadDim_ = xShape.GetDim(dimNum - 1);
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "attrs is nullptr"),
        return ge::GRAPH_FAILED);
    int64_t mode = *(attrs->GetAttrPointer<int64_t>(0));
    OPS_ERR_IF(mode != 1,
        OPS_LOG_E(context_->GetNodeName(), "mode only support interleave"),
        return ge::GRAPH_FAILED);
    auto sliceListAttr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    auto sliceData = static_cast<const int64_t *>(sliceListAttr->GetData());
    start_ = sliceData[0];
    end_ = sliceData[1];
    OPS_LOG_I(context_->GetNodeName(), "end_ %ld, end_ %ld",start_, end_);

    headDim_ = end_ - start_;
    xdim3_ = headDim_;
    OPS_ERR_IF(headDim_ <= 0,
        OPS_LOG_E(context_->GetNodeName(), "slice not right"),
        return ge::GRAPH_FAILED);
    r1dim3_ =  inputR1Shape.GetDim(3);
    r1dim0_ =  inputR1Shape.GetDim(0);
    r1dim1_ =  inputR1Shape.GetDim(1);
    r1dim2_ =  inputR1Shape.GetDim(2);
    int64_t r1Dim2 = inputR1Shape.GetDim(2);
    int64_t xDim1 = xShape.GetDim(1);
    int64_t xDim2 = xShape.GetDim(2);
    OPS_ERR_IF(headDim_ != r1dim3_,
        OPS_LOG_E(context_->GetNodeName(), "slice not right, not equal r1 and r2 last dim num"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(r1dim0_ != dim0_,
        OPS_LOG_E(context_->GetNodeName(), "dim0 must be equal"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(r1dim2_ != 1,
        OPS_LOG_E(context_->GetNodeName(), "r1dim2_ must be 1"),
        return ge::GRAPH_FAILED);
    dim2_ = headDim_;
    dim1_ = xShape.GetDim(1) * xShape.GetDim(2);
    numHead_ = dim1_;
    if (r1dim1_ == 1 && r1dim2_ == 1 && (xdim1_ == 1 || xdim2_ == 1)) {
        tilingKey_ = 1;
        isSpecail_ = true;
        if (r1dim1_ == dim1_) {
            isBrc_ = false;
            tilingKey_ = tilingKey_ + 1;
        }
    }
    if (xdim3_ % oneBlockSize_ == 0){
        isAlign_ = true;
    }
    OPS_LOG_I(context_->GetNodeName(), "isSpecail_ %d", isSpecail_);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InplacePartialRotaryMulTiling::TilingSplitN(int64_t numHeads, int64_t headDimAlign, int64_t ubSize,
                             ge::DataType dataDtype)
{
    const int64_t bufferSize = ubSize - 0;
    int64_t totalHeadNum1Size = headDimAlign * IO_NUM * dtypeSize_ + headDimAlign * INT32_DTYPE_SIZE;
    if (dataDtype == ge::DT_BF16 || dataDtype == ge::DT_FLOAT16) {
        totalHeadNum1Size += headDimAlign * FP32_DTYPE_SIZE * IO_NUM;
    }
    uint32_t ubCalcNNum{1}, ubCalcNLoop{numHeads}, ubCalcNTail{0};
    OPS_ERR_IF(bufferSize < totalHeadNum1Size, OPS_LOG_E(context_->GetNodeName(), "The D dimension of the input shape is too large."),
                return ge::GRAPH_FAILED);
    ubCalcNNum = GetDiv(bufferSize, totalHeadNum1Size);
    ubCalcNLoop = GetCeilInt(numHeads, ubCalcNNum);
    ubCalcNTail = GetDivRem(numHeads, ubCalcNNum) != 0 ? numHeads - (ubCalcNLoop - 1) * ubCalcNNum : 0;
    tilingData_.set_ubCalcNNum(ubCalcNNum);
    tilingData_.set_ubCalcNLoop(ubCalcNLoop);
    tilingData_.set_ubCalcNTail(ubCalcNTail);
    tilingKey_ += TILING_KEY_SPLIT_BSN;
    context_->SetTilingKey(tilingKey_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulTiling::TilingSplitB(int64_t batchSize, int64_t numHeads, int64_t headDimAlign,
                             int64_t ubSize, ge::DataType dataDtype)
{
    const int64_t tBufferSize = numHeads * headDimAlign * FP32_DTYPE_SIZE;
    const int64_t bufferSize = ubSize - tBufferSize;
    int64_t totalBatch1Size = numHeads * headDimAlign * IO_NUM * dtypeSize_;

    if (dataDtype == ge::DT_BF16 || dataDtype == ge::DT_FLOAT16) {
        totalBatch1Size += numHeads * headDimAlign * FP32_DTYPE_SIZE * IO_NUM;
    }
    int64_t ubCalcBNum{1}, ubCalcBLoop{batchSize}, ubCalcBTail{0};
    if (ubSize < tBufferSize || bufferSize < totalBatch1Size) {
        OPS_ERR_IF(TilingSplitN(numHeads, headDimAlign, ubSize, dataDtype) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(context_->GetNodeName(), "TilingSplitN fail."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ubCalcBNum = GetDiv(bufferSize, totalBatch1Size);
    ubCalcBLoop = GetCeilInt(batchSize, ubCalcBNum);
    ubCalcBTail = GetDivRem(batchSize, ubCalcBNum) != 0 ? batchSize - (ubCalcBLoop - 1) * ubCalcBNum : 0;

    tilingData_.set_ubCalcBNum(ubCalcBNum);
    tilingData_.set_ubCalcBLoop(ubCalcBLoop);
    tilingData_.set_ubCalcBTail(ubCalcBTail);

    tilingKey_ += TILING_KEY_SPLIT_S;
    context_->SetTilingKey(tilingKey_);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InplacePartialRotaryMulTiling::TilingSplitS()
{
    auto xDesc = context_->GetInputDesc(0);
    auto dataDtype = xDesc->GetDataType();
    int64_t batchSize = tilingData_.get_batchSize();
    int64_t seqLen = tilingData_.get_seqLen();
    int64_t numHeads = tilingData_.get_numHeads();
    int64_t headDim = tilingData_.get_headDim();
    // block split
    int64_t frontCoreNum = GetDivRem(seqLen, coreNum_) != 0 ? GetDivRem(seqLen, coreNum_) : coreNum_;
    int64_t tailCoreNum = seqLen <= coreNum_ ? 0 : coreNum_ - frontCoreNum;
    usedCoreNum_ = frontCoreNum + tailCoreNum;
    int64_t coreCalcNum = GetCeilInt(seqLen, coreNum_);
    int64_t coreCalcTail = GetDiv(seqLen, coreNum_);
    tilingData_.set_frontCoreNum(frontCoreNum);
    tilingData_.set_tailCoreNum(tailCoreNum);
    tilingData_.set_coreCalcNum(coreCalcNum);
    tilingData_.set_coreCalcTail(coreCalcTail);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    context_->SetBlockDim(usedCoreNum_);
    int64_t headDimAlign = 0;
    if (isAlign_) {
        headDimAlign = headDim;
    } else {
        headDimAlign = GetCeilInt(headDim, oneBlockSize_) * oneBlockSize_;
        tilingKey_ += 1;
    }
    // ub split
    int64_t tBufferSize = numHeads * headDimAlign * FP32_DTYPE_SIZE;
    int64_t bufferSize = ubSize_ - tBufferSize;
    int64_t ioUbSize = batchSize * coreCalcNum * numHeads * headDimAlign * IO_NUM * dtypeSize_;
    int64_t totalSeq1Size = batchSize * numHeads * headDimAlign * IO_NUM * dtypeSize_;
    if (dataDtype == ge::DT_BF16 || dataDtype == ge::DT_FLOAT16) {
        ioUbSize += batchSize * coreCalcNum * numHeads * headDimAlign * FP32_DTYPE_SIZE * IO_NUM;
        totalSeq1Size += batchSize * numHeads * headDimAlign * FP32_DTYPE_SIZE * IO_NUM;
    }
    if (tBufferSize >= ubSize_) {
        OPS_ERR_IF(TilingSplitN(numHeads, headDimAlign, ubSize_, dataDtype) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(context_->GetNodeName(), "TilingSplitN fail."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (ubSize_ < tBufferSize || bufferSize < totalSeq1Size) {
        OPS_ERR_IF(TilingSplitB(batchSize, numHeads, headDimAlign, ubSize_, dataDtype) !=
                        ge::GRAPH_SUCCESS,
                    OPS_LOG_E(context_->GetNodeName(), "TilingSplitB fail."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    context_->SetTilingKey(tilingKey_);
    int64_t ubCalcNum, ubCalcLoop, ubCalcTail;
    if (bufferSize < ioUbSize) {
        ubCalcNum = GetDiv(bufferSize, totalSeq1Size);
        ubCalcLoop = GetCeilInt(coreCalcNum, ubCalcNum);
        ubCalcTail = GetDivRem(coreCalcNum, ubCalcNum) != 0 ? coreCalcNum - (ubCalcLoop - 1) * ubCalcNum : 0;
    } else {
        ubCalcNum = coreCalcNum;
        ubCalcLoop = 1;
        ubCalcTail = 0;
    }
    tilingData_.set_ubCalcNum(ubCalcNum);
    tilingData_.set_ubCalcLoop(ubCalcLoop);
    tilingData_.set_ubCalcTail(ubCalcTail);
    // ub split for tail core
    int64_t ubCalcTailNum{0}, ubCalcTailLoop{0}, ubCalcTailTail{0};
    if (coreCalcTail != 0) {
        ioUbSize = batchSize * coreCalcTail * numHeads * headDimAlign * IO_NUM * dtypeSize_;
        totalSeq1Size = batchSize * numHeads * headDimAlign * IO_NUM * dtypeSize_;
        if (dataDtype == ge::DT_BF16 || dataDtype == ge::DT_FLOAT16) {
            ioUbSize += batchSize * coreCalcNum * numHeads * headDimAlign * FP32_DTYPE_SIZE * IO_NUM;
            totalSeq1Size += batchSize * numHeads * headDimAlign * FP32_DTYPE_SIZE * IO_NUM;
        }
        if (bufferSize < ioUbSize) {
            ubCalcTailNum = GetDiv(bufferSize, totalSeq1Size);
            ubCalcTailLoop = GetCeilInt(coreCalcTail, ubCalcTailNum);
            ubCalcTailTail =
                GetDivRem(coreCalcTail, ubCalcTailNum) != 0 ? coreCalcTail - (ubCalcTailLoop - 1) * ubCalcTailNum : 0;
        } else {
            ubCalcTailNum = coreCalcTail;
            ubCalcTailLoop = 1;
            ubCalcTailTail = 0;
        }
    }
    tilingData_.set_ubCalcTailNum(ubCalcTailNum);
    tilingData_.set_ubCalcTailLoop(ubCalcTailLoop);
    tilingData_.set_ubCalcTailTail(ubCalcTailTail);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulTiling::TilingSplit()
{
    int64_t batchSizeOut{1}, seqLenOut{1}, numHeadsOut{1};
    if (r1dim1_ == 1 && r1dim2_ == 1 && xdim0_ == r1dim0_) {
        seqLenOut = r1dim0_;
        numHeadsOut = xdim1_ * xdim2_; // SBND -> 1S(BN)D -> 1SND
    } else if (r1dim0_ == 1 && r1dim2_ == 1 && xdim1_ == r1dim1_) {
        seqLenOut = r1dim1_;
        batchSizeOut = xdim0_; // BSND
        numHeadsOut = xdim2_;
    } else if (r1dim0_ == 1 && r1dim1_ == 1 && xdim2_ == r1dim2_) {
        seqLenOut = r1dim2_;
        batchSizeOut = xdim0_ * xdim1_; // BNSD -> (BN)S1D -> BS1D
    } else if (xdim0_ == r1dim0_ && xdim1_ == r1dim1_) {
        batchSizeOut = 1;
        seqLenOut = r1dim0_ * r1dim1_;
        numHeadsOut = xdim2_; // 1,BS,N,D cons/sin 1,BS,1,D
    } else {
        OPS_LOG_E(context_->GetNodeName(), "The shape of the input x, cos and sin is not supported.");
        return ge::GRAPH_FAILED;
    }
    if (batchSizeOut != 1) {
        OPS_LOG_E(context_->GetNodeName(), "batchSizeOut must be 1");
        return ge::GRAPH_FAILED;
    }
    tilingData_.set_batchSize(batchSizeOut);
    tilingData_.set_seqLen(seqLenOut);
    tilingData_.set_numHeads(numHeadsOut);
    tilingData_.set_headDim(xdim3_);

    OPS_ERR_IF(TilingSplitS() != ge::GRAPH_SUCCESS,
                OPS_LOG_E(context_->GetNodeName(), "TilingSplitS fail."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InplacePartialRotaryMulTiling::DoTiling()
{
    OPS_LOG_I(context_->GetNodeName(), "Enter InplacePartialRotaryMulTiling DoTiling");
    OPS_ERR_IF(
        CheckInput() != ge::GRAPH_SUCCESS,
        OPS_LOG_E(context_->GetNodeName(), "CheckInputShapes is failed"),
        return ge::GRAPH_FAILED);
    if (CalTilingData() == ge::GRAPH_SUCCESS) {
        FillTilingData();
        PrintTilingData();
        context_->SetBlockDim(usedCoreNum_);
        context_->SetTilingKey(tilingKey_);
        
        size_t* workspaces = context_->GetWorkspaceSizes(1);
        workspaces[0] = static_cast<size_t>(16 * 1024 * 1024);
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    } else {
        // 开始走原始的71的逻辑
        tilingKey_ = BASE_KEY;
        auto dataDtype = context_->GetInputDesc(0)->GetDataType();
        if (dataDtype == ge::DT_FLOAT16) {
            tilingKey_ += TILING_KEY_SPLIT_S;
        } else if (dataDtype == ge::DT_BF16) {
            tilingKey_ += TILING_KEY_BFLOAT16;
        } else if (dataDtype == ge::DT_FLOAT) {
            tilingKey_ += TILING_KEY_FLOAT32;
            dtypeSize_ = FP32_DTYPE_SIZE;
        }
        tilingData_.set_allHeadDim(allHeadDim_);
        tilingData_.set_start(start_);
        OPS_ERR_IF(TilingSplit() != ge::GRAPH_SUCCESS,
                OPS_LOG_E(context_->GetNodeName(), "TilingSplit fail."), return ge::GRAPH_FAILED);

        OPS_LOG_I(context_->GetNodeName(), "[tilingKey]: %ld", tilingKey_);
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
        size_t usrWorkspaceSize = 0;
        size_t sysWorkspaceSize = 16 * 1024 * 1024;
        size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
        currentWorkspace[0] = usrWorkspaceSize + sysWorkspaceSize;

        PrintInfo();
        return ge::GRAPH_SUCCESS;
    }
    
}
ge::graphStatus Tiling4InplacePartialRotaryMul(gert::TilingContext* context)
{
    InplacePartialRotaryMulTiling tilingImpl = InplacePartialRotaryMulTiling(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context, "Tiling4InplacePartialRotaryMul init failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context, "Tiling4InplacePartialRotaryMul do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(context->GetNodeName(), "end Tiling4InplacePartialRotaryMul");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
