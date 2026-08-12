/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// mc2编译依赖
//
// mc2编译依赖
/*!
 * \file quant_batch_matmul_v3_tiling.h
 * \brief
 */

#ifndef MC2_QUANT_BATCH_MATMUL_V3_TILING_H
#define MC2_QUANT_BATCH_MATMUL_V3_TILING_H

#include "quant_batch_matmul_v3_tiling_base.h"
#include "../../op_kernel/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {

class Mc2QuantBatchMatmulV3Tiling : public Mc2QuantBatchMatmulV3TilingBase {
public:
    explicit Mc2QuantBatchMatmulV3Tiling(gert::TilingContext *context);
    Mc2QuantBatchMatmulV3Tiling(gert::TilingContext *context, Mc2QuantBatchMatmulV3TilingData *out);
    ~Mc2QuantBatchMatmulV3Tiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData，mc2使用的直接接口
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void Reset();
    bool CheckDtypeOnOnlyL0c2ub() const;
    bool CheckDtypeOnOnlyL0c2out() const;
    bool CheckDtypeOnOnlyL0c2outForSupportedList() const;
    bool CheckDtypeOnOnlyL0c2outForA4W4() const;
    bool CheckDtypeOnOnlyL0c2outForPertoken() const;
    bool CheckDtypeOnOnlyL0c2outForX1NZ() const;
    bool CheckDtypeOnOnlyL0c2outForUnclassified() const;
    uint64_t GetTilingKey(bool isBasicTiling) const;
    bool GetUbDequantExtreSpace() override;
    ge::graphStatus CalcUbTiling() override;
    bool CheckDtype() const override;
    bool CheckShape(const std::vector<gert::Shape *> &mandtoryShape, const gert::StorageShape *biasShape,
                    const gert::StorageShape *pertokenShape, const std::vector<int64_t> &dimValueOfMKN) const override;
    bool SetMatmulTilingFromTbeTiling();
    bool GetTbeTiling();
    void ProcessMSmall();
    int32_t GetIteratorOrder();
    void PrintTilingData();
    void PrintTbeTiling();
    void PrintTilingParams() const;

    ge::graphStatus CalcPertokenOptUbTiling();
    ge::graphStatus CalcUbTiling(uint32_t baseN, uint32_t baseM);
    void SpiltSingleCore(int32_t &singleCoreM, int32_t &singleCoreN);
    void SpiltForWorkSpaceLimit(int32_t singleCoreM, int32_t singleCoreN, int32_t blockDim);
    bool SetBlockDimsAndSingleCore(AscendC::tiling::TCubeTiling &mt);
    bool CalcUsedL1AndUBSize(int32_t aL1Size, int32_t bL1Size, bool &fallback);
    bool CheckShapeInBoundary(const gert::Shape &shape, uint32_t shapeIdx) const;
    ge::graphStatus InitTilingData(matmul_tiling::MatmulApiTilingBase &mm, bool fallback = false);
    void Int4LowerAxisAlign(uint64_t &baseM, uint64_t &baseN) const;
    uint64_t CalcL1SizeForBiasAndScale();
    int32_t CalcND2NZSpace() const;
    void ConstructCacheParams(BatchmatmulCompileParas &compileParams, BatchmatmulRunParas &runParams) const;
    void ModifyCacheParams(BatchmatmulRunParas &runParams) const;
    bool NeedAtomiClean() const;

    bool CheckDimValue(const gert::Shape &scaleShape, const gert::StorageShape *biasShape,
                       const gert::StorageShape *pertokenShape, const std::vector<int64_t> &dimValueOfMKN) const;

    bool CheckShapeInRangeForOptionalInputs(const gert::StorageShape *biasShape,
                                            const gert::StorageShape *pertokenShape) const;
    bool BiasShapeCheck(const gert::Shape &biasShape) const;
    uint64_t GetTotalSize(uint64_t m, uint64_t k, uint64_t n) const;

    uint32_t GetABankConflictSize();
    void UpdateSmallMTbeTiling();
    void UpdateSmallMTbeTiling(uint64_t baseM, uint64_t baseN, uint64_t baseK);
    void SetQuantBatchMatmulRunParas(QuantBatchMatmulRunParas &runParams,
                                     const optiling::Mc2QuantBatchMatmulInfo &inputParams);
    // 新增数据成员请注意：如果是在GetShapeAttrsInfo函数过程中获取的，请放到Mc2QuantBatchMatmulInfo结构体中，或者保证在DoOpTiling赋值
    Mc2QuantBatchMatmulV3TilingData tilingDataSelf_{};
    Mc2QuantBatchMatmulV3TilingData &tilingData_;
};
} // namespace optiling
#endif // QUANT_BATCH_MATMUL_V3_TILING_H
