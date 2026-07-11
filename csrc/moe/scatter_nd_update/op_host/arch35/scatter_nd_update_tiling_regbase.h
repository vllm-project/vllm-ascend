/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file scatter_nd_update_tiling.h
 * \brief ascendc scatter ND update tiling h
 */

#ifndef SCATTER_ND_UPDATE_TILING_H_
#define SCATTER_ND_UPDATE_TILING_H_

#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/scatter_nd_update_struct.h"

namespace optiling {

struct ScatterNdUpdateCompileInfo {
    int64_t core_num{1};
    int64_t ub_size{1};
    bool is_950{false};
};

using namespace Ops::NN::Optiling;
class ScatterNdUpdateTiling : public TilingBaseClass {
public:
    explicit ScatterNdUpdateTiling(gert::TilingContext* context) : TilingBaseClass(context) {}
    uint32_t maxThread_{1024};
    uint64_t coreNum_{0};
    uint64_t ubSize_{0};

protected:
    bool IsCapable() override { return true; }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    uint32_t GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend);
    int64_t GetRestAvailableSize(int64_t sampleNum, int64_t valueTypeBytes, int64_t originalSize, int64_t postAxisSize,
                                 ge::DataType idType);
    ge::graphStatus CheckScatterNdUpdateTensorShape(const gert::Shape& indiceShape, const gert::Shape& updateShape,
                                                    const gert::Shape& outputShape);
    ge::graphStatus ValidateVarInfo(const gert::Tensor* var, const gert::Shape& varOriginShape);
    ge::graphStatus ValidateIndicesInfo(const gert::Tensor* indices, gert::Shape& indiceShape, int64_t& indiceDims);
    ge::graphStatus HandleNonContiguousCase(const gert::Tensor* var, const gert::Shape& varOriginShape);
    ge::graphStatus HandleContiguousCase();
    ge::graphStatus ValidateUpdatesInfo(const gert::Tensor* updates, gert::Shape& updateShape);
    ge::graphStatus CalculateDerivedParams(const gert::Shape& varOriginShape, gert::Shape& indiceShape,
                                           gert::Shape& updateShape);
    void DoOpTilingSplitAfter();
    void DoOpTilingSimdSplitIndices();
    void DoOpTilingForSimdNonDetermin();
    void DoOpTilingForSimdMask();
    void DoOpTilingForDeterministic();
    void CalcDeterministicCoreSplit();
    void CalcDeterministicUpdateSplit(int64_t ubBlock);
    void CalcDeterministicIndicesSplit(int64_t ubBlock);
    void CalculateMask();
    uint64_t GetTilingKey() const override;
    void ComputeCoreSplitAfterAxis();
    void InitFactors(int64_t halfUbSize, int64_t indicesSize, int64_t alignNum);
    void HandleIndicesFactorGtOne(int64_t halfUbSize, int64_t indicesSize, int64_t alignNum, int64_t ubBlock);
    void HandleIndicesFactorLeOne(int64_t halfUbSize, int64_t indicesSize, int64_t alignNum, int64_t ubBlock);
    int64_t UnitIdxAligned(int64_t idxFactor, int64_t ubBlock);
    int64_t UnitUpdAligned(int64_t afterAxisFactor, int64_t ubBlock);
    int64_t OccupyTotal(int64_t idxFactor, int64_t afterAxisFactor, int64_t ubBlock);
    int64_t RoughMaxIdxByUb(int64_t afterAxisFactor, int64_t halfUbSize, int64_t indicesSize);
    ge::graphStatus SortTiling();
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GenerateTilingKey();
    void DumpTilingInfo() override;

private:
    ge::graphStatus UbTiling();
    void BlockTiling();
    void SetTilingData();
    ge::graphStatus SetStride();

private:
    int64_t shapeRank_ = 0;
    int64_t totalCoreNum_{0};
    ge::DataType varDtype_ = ge::DT_UNDEFINED;
    ge::DataType updateDtype_ = ge::DT_UNDEFINED;
    ge::DataType indiceDtype_ = ge::DT_UNDEFINED;
    ge::DataType outOfSetDtype_ = ge::DT_UNDEFINED;

    uint64_t updateShapeSize{0};
    uint64_t indiceShapeSize{0};
    uint64_t outputStorageShapeSize_ = 1;
    uint64_t outputShapeSize = 1;
    uint64_t alignFactor{0};
    uint64_t blockNum{0};
    uint64_t blockTilingSize{0};
    uint64_t tailBlockTilingSize{0};
    uint32_t ubTilingSize{0};
    uint32_t rankSize_{0};
    uint64_t sliceSize{0};
    uint64_t strideList[MAX_SHAPE_RANK] = {0};
    uint64_t outPutShape[MAX_SHAPE_RANK] = {0};
    uint64_t workspaceSize{0};

    int64_t indicesAxis_ = 0;
    int64_t varInAxis_ = 1;
    int64_t varStorageInAxis_ = 1;
    int64_t updatesInAxis_ = 1;
    int64_t afterAxis_ = 1;
    int64_t varTypeSize_ = 0;
    int64_t indicesTypeSize_ = 0;
    int64_t outOfSetTypeSize_ = 0;
    int64_t indicesFactor_ = 0;
    int64_t ubRowFactor_ = 0;
    int64_t afterAxisFactor_ = 0;
    int64_t ubQuantaIndxFactor_ = 0;
    int64_t usedCoreNumBefore_ = 0;
    int64_t usedCoreNumAfter_ = 0;
    int64_t eachCorePreAxisCount_ = 0;
    int64_t tailCorePreAxisCount_ = 0;
    int64_t eachCoreAfterAxisCount_ = 0;
    int64_t tailCoreAfterAxisCount_ = 0;
    int64_t updateLoopSize_ = 0;
    int64_t updateTailNum_ = 0;
    int64_t indicesLoopSize_ = 0;
    int64_t indiceTailNum_ = 0;
    int64_t isSplitPreAxis_ = 0;
    int64_t tailUpdateLoopSize_ = 0;
    int64_t tailUpdateTailNum_ = 0;
    int64_t isSplitAfterAxis_ = 0;
    int64_t isSplitOneLine_ = 0;

    int64_t eachCoreIndexCount_ = 0;
    int64_t tailCoreIndexCount_ = 0;
    int64_t eachCoreVarCount_ = 0;
    int64_t tailCoreVarCount_ = 0;
    int64_t isDeterministic_ = 0;
    int64_t isSimtWithSort_ = 0;
    int64_t isSimdWithSort_ = 0;
    int64_t isMask_ = 0;
    int64_t isSimdNonDeterministic_ = 0;

    int64_t indicesUbFactor_ = 0;
    int64_t normBlockLoop_ = 0;
    int64_t tailBlockLoop_ = 0;
    int64_t normBlockTail_ = 0;
    int64_t tailBlockTail_ = 0;
    int64_t IsContiguous_ = 0;

    // 处理AtomicMax
    int64_t normCoreHandleIdx_ = 0;
    int64_t tailCoreHandleIdx_ = 0;
    int64_t calcMaskUsedCoreNum_ = 0;
    int64_t maskNormBlockLen_ = 0;
    int64_t maskTailBlockLen_ = 0;
    int64_t isDeterminSimt_ = 0;
    int64_t needInt64_ = 0;

    const char* opName = "ScatterNdUpdate";
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ND_ADD_TILING_H_
