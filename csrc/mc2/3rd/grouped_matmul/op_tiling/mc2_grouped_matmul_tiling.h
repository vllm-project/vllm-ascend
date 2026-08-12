/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file mc2_grouped_matmul_tiling.h
 * \brief
 */
#ifndef MC2_GROUPED_MATMUL_TILING_H
#define MC2_GROUPED_MATMUL_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>

#include "mc2_grouped_matmul_host_util.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
namespace Mc2GroupedMatmul {
BEGIN_TILING_DATA_DEF(MC2GMMBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, activeType);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseK);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseN);
TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
TILING_DATA_FIELD_DEF(uint32_t, singleWeight);
TILING_DATA_FIELD_DEF(uint32_t, singleX);
TILING_DATA_FIELD_DEF(uint32_t, singleY);
TILING_DATA_FIELD_DEF(int32_t, groupType);
TILING_DATA_FIELD_DEF(uint32_t, singleN);    // If sequential write， the value should be zero!
TILING_DATA_FIELD_DEF(uint32_t, quantParam); // in quant case, PerToken: 1; in antiquant case, represents PerGroupSize
TILING_DATA_FIELD_DEF(uint32_t, groupListType);
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, hasBias);
TILING_DATA_FIELD_DEF(uint64_t, workspaceSize);
TILING_DATA_FIELD_DEF(uint64_t, totalInGroup);  // for A8W4 MSD
TILING_DATA_FIELD_DEF(uint64_t, k);             // for A8W4 MSD
TILING_DATA_FIELD_DEF(uint64_t, n);             // for A8W4 MSD
TILING_DATA_FIELD_DEF(uint64_t, vBaseM);        // for A8W4 MSD
TILING_DATA_FIELD_DEF(uint64_t, parallNum);     // for A8W4 MSD
TILING_DATA_FIELD_DEF(uint64_t, quantGroupNum); // for A8W4 MSD
TILING_DATA_FIELD_DEF(uint64_t, isPreTiling);
TILING_DATA_FIELD_DEF(uint32_t, withOffset);
TILING_DATA_FIELD_DEF(uint32_t, isOutputDisableL2Cache);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMBaseParamsOp, MC2GMMBaseParams)

BEGIN_TILING_DATA_DEF(CommonGMMArray)
TILING_DATA_FIELD_DEF_ARR(int32_t, 128, mList); // 128 ：MAX_TENSOR_CONT
TILING_DATA_FIELD_DEF_ARR(int32_t, 128, kList);
TILING_DATA_FIELD_DEF_ARR(int32_t, 128, nList);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMArrayOp, CommonGMMArray)

// for autotiling w4a8
BEGIN_TILING_DATA_DEF(MC2A8W4HPTiling)
TILING_DATA_FIELD_DEF(uint32_t, group_num);
TILING_DATA_FIELD_DEF(int8_t, group_type);
TILING_DATA_FIELD_DEF(uint32_t, required_core_num);
TILING_DATA_FIELD_DEF(float, format_in);
TILING_DATA_FIELD_DEF(float, format_out);
TILING_DATA_FIELD_DEF(uint32_t, numAic);
TILING_DATA_FIELD_DEF(uint32_t, numAiv);
TILING_DATA_FIELD_DEF(uint64_t, szUb);
TILING_DATA_FIELD_DEF(uint64_t, szL0A);
TILING_DATA_FIELD_DEF(uint64_t, szL0C);
TILING_DATA_FIELD_DEF(uint8_t, pattern);
TILING_DATA_FIELD_DEF(uint8_t, kernel_index);
TILING_DATA_FIELD_DEF(uint32_t, splitTimes);
TILING_DATA_FIELD_DEF(int8_t, output_type);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 2, ori_in0_shape);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 2, ori_in1_shape);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 2, ori_out_shape);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, single_core_tiling);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 2, single_core_base_tiling);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, splitRecord);
TILING_DATA_FIELD_DEF(uint64_t, workspaceOffset);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(A8W4HPTilingOp, MC2A8W4HPTiling)

BEGIN_TILING_DATA_DEF(MC2GMMTilingData)
TILING_DATA_FIELD_DEF_STRUCT(MC2GMMBaseParams, gmmBaseParams);
TILING_DATA_FIELD_DEF_STRUCT(CommonGMMArray, gmmArray);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
// for autotiling
TILING_DATA_FIELD_DEF_STRUCT(MC2A8W4HPTiling, hpTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Mc2GroupedMatmul, MC2GMMTilingData)

struct GMMCompileInfo {
    uint32_t aicNum;
    uint32_t aivNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l2Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    platform_ascendc::SocVersion socVersion;
};

class GMMTiling {
public:
    MC2GMMTilingData tilingData;
    ge::graphStatus Init(const gert::TilingContext *context);
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext *context);
    ge::graphStatus A8W4Tiling(gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr);
    ge::graphStatus A16W4MsdTiling(gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr);

protected:
    bool IsAivAicRatioTwoRequired();
    bool IsFixedAxisMoveCondition();
    bool IsIntDataType();
    ge::graphStatus CalMMTiling(const gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr);
    ge::graphStatus GMMSetMMTiling(const gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr);
    ge::graphStatus GMMGetAttrs(const gert::TilingContext *context);
    ge::graphStatus GMMSetUbDivideBlk();
    ge::graphStatus GMMSetUbDivideBlkAntiquant();
    ge::graphStatus GMMSetUbDivideBlkQuant();
    ge::graphStatus GMMSetUbDivideBlkA4W4();
    ge::graphStatus GMMCalUbSize(const gert::TilingContext *context, uint32_t ubSize);
    int64_t GMMGetBS(const gert::Shape &xShape) const;
    ge::graphStatus PrepareTilingData(const gert::TilingContext *context);
    ge::graphStatus CheckWeightNZShape(const gert::TilingContext *context, int64_t numInOneBlk) const;
    ge::graphStatus GMMGetTensorShapeSplitM(const gert::TilingContext *context, const gert::Shape &xShape,
                                            const gert::Shape &wShape);
    ge::graphStatus GMMGetTensorShapeSplitK(const gert::TilingContext *context, const gert::Shape &xShape,
                                            const gert::Shape &wShape);
    ge::graphStatus SplitMSingleXSingleWeightSingleY(const gert::Shape &xShape, const gert::Shape &wShape);
    ge::graphStatus SplitMSingleXSeparatedWeight(const gert::TilingContext *context, const gert::Shape &xShape);
    ge::graphStatus SeparatedXSeparatedWeight(const gert::TilingContext *context);
    ge::graphStatus SeparatedXSingleWeight(const gert::TilingContext *context, const gert::Shape &wShape);
    ge::graphStatus SplitKSingleXSingleWeightSingleY(const gert::TilingContext *context, const gert::Shape &xShape,
                                                     const gert::Shape &wShape);
    ge::graphStatus SplitKSingleXSeparatedWeight(const gert::TilingContext *context, const gert::Shape &xShape,
                                                 const gert::Shape &wShape);
    ge::graphStatus DivideUbAndSetWorkspace(gert::TilingContext *context, const uint32_t &aicNum);
    ge::graphStatus DynamicTilingSingleN(gert::TilingContext *context, const uint32_t &aicNum,
                                         const GMMCompileInfo *compileInfoPtr);
    ge::graphStatus IsOutputDisableL2Cache(gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr);
    int32_t FindBestSingleN(const uint32_t &aicNum);
    bool TryFullLoadA(int32_t baseM, const GMMCompileInfo *compileInfoPtr);
    void DivideUbAndSetWorkspaceAntiquant(size_t *workspaces, const uint32_t &aicNum, uint32_t &ubSize);
    ge::graphStatus CalcStepKaKb(const gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr,
                                 int64_t mInMM, uint32_t &mmStepKa, uint32_t &mmStepKb);
    ge::graphStatus SetBias(const gert::TilingContext *context, matmul_tiling::MultiCoreMatmulTiling &mm) const;
    int32_t FindBestSingleNPertoken(const uint32_t aicNum) const;
    void FindBestUsedCoreNumOneGroup(const uint32_t aicNum);
    ge::graphStatus SetWorkspscesPerTokenQuant(const uint32_t aicNum, size_t *workspaces);
    void SetTilingDataIsSingleTensor();
    ge::graphStatus GetPerGroupNum(const gert::TilingContext *context);
    ge::graphStatus CheckMKN(const gert::TilingContext *context);
    void FullLoadK(const GMMCompileInfo *compileInfoPtr);
    void SetMMPreTiling();
    bool StaticTilingProcess(gert::TilingContext *context);
    bool CheckTilingMatchStaticValue();
    void PrintTilingInfo(gert::TilingContext *context);
    void GMMSetTplTilingKey(gert::TilingContext *context);
    uint32_t GetTplDataType(const ge::DataType &dtype);
    ge::graphStatus CheckA16W4MsdEnable(uint64_t mSize, uint64_t antiquantGroupNum, const gert::TilingContext *context,
                                        const GMMCompileInfo *compileInfoPtr);
    uint64_t GetWithOffset(const gert::TilingContext *context);
    bool CheckTensorListLength(const gert::TilingContext *context);

private:
    int32_t mList_[Mc2GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int32_t kList_[Mc2GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int32_t nList_[Mc2GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int64_t maxM_ = 0L;
    int64_t maxN_ = 0L;
    int64_t maxK_ = 0L;
    int32_t minK_ = INT32_MAX;
    int32_t baseM_ = 0;
    int32_t baseN_ = 0;
    int32_t baseK_ = 0;
    uint64_t ubSize_ = 0UL;
    uint32_t mmDataTypeSize_ = 0;
    uint32_t ubDivideBlkNum_ = 0;
    uint32_t ubIoBlkNum_ = 0;
    uint32_t ubBlockAlign_ = 0;
    uint64_t workspacesSize_ = 0UL; // for antiquant
    uint32_t groupNum_ = 0;
    bool transposeWeight_ = false;
    bool transposeX_ = false;
    bool isSingleWeight_ = false;
    bool isSingleX_ = false;
    bool isSingleY_ = false;
    bool isAllSingleTensor_ = false;
    bool hasBias_ = false;
    int32_t groupType_ = 0;
    int64_t splitItem_ = 0L;
    uint32_t groupListType_ = 0;
    uint32_t xKDim_ = 0;
    uint32_t weightNDim_ = 0;
    uint32_t weightKDim_ = 0;
    uint32_t xDimNum_ = 0;
    bool antiquantPerformance_ = false;
    uint32_t actType_ = 0;
    uint32_t usedCoreNum_ = 0;
    int64_t tuningConfig_ = 0L;
    int64_t tuningConfigWorkspace_ = 0L;
    uint64_t FixedAxisMoveWorkspace_ = 0L;
    bool isA4W4_ = false;
    bool isA16W16_ = false;
    bool isA8W4FakeA8W8_ = false;
    bool isFixedAxisMove_ = false;
    uint64_t A8W4noMsdSpace_ = 0;

    ge::DataType xDType_ = ge::DT_UNDEFINED;
    ge::DataType mmDType_ = ge::DT_UNDEFINED;
    ge::DataType weightDtype_ = ge::DT_UNDEFINED;
    ge::DataType scaleDtype_ = ge::DT_UNDEFINED;
    ge::DataType perTokenScaleDtype_ = ge::DT_UNDEFINED;
    ge::DataType yDtype_ = ge::DT_UNDEFINED;
    bool isA8W8_ = false;
    // in quant case, it indicates pertoken flag; in antiquant case, it represents pergroup size
    uint32_t perTokenOrPerGroupSize_ = 0;
    bool isA16W8Msd_ = false;
    uint32_t totalM_ = 0;
    matmul_tiling::CubeFormat wFormat_;
    int32_t nzFactor_; // for weight nz format
};
} // namespace Mc2GroupedMatmul
} // namespace optiling

#endif // MC2_GROUPED_MATMUL_TILING_H
