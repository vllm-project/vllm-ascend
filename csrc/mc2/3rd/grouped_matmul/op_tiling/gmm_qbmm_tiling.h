/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gmm_qbmm_tiling.h
 * \brief Copied from gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h
 */
#ifndef MC2_GMM_QBMM_TILING_H
#define MC2_GMM_QBMM_TILING_H

#include "mc2_grouped_matmul_tiling.h"
#include "../op_kernel/grouped_matmul_tiling_data_apt.h"
#include "op_host/tiling_base.h"

namespace optiling {
namespace Mc2GroupedMatmulTiling {
namespace GmmConstant {
constexpr uint64_t MX_GROUP_SIZE = 32;
constexpr uint64_t NUM_HALF = 2;
constexpr uint64_t EVEN_FACTOR = 2;
constexpr uint32_t DB_SIZE = 2;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
constexpr uint64_t UB_ALIGN_SIZE = 32;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint32_t SCALER_FACTOR_MAX = 127;
constexpr uint32_t SCALER_FACTOR_MIN = 1;
constexpr uint32_t SCALER_FACTOR_DEFAULT = 1;
constexpr uint32_t SCALER_FACTOR_B_BIT = 8;
constexpr uint32_t SCALER_FACTOR_M_BIT = 16;
constexpr uint32_t SCALER_FACTOR_N_BIT = 24;
constexpr uint64_t MTE2_MIN_LOAD_SIZE_V120 = 64 * 1024UL;
constexpr uint64_t MAX_REPEAT_TIMES = 255; // InitOutput接口取值
constexpr uint64_t GMM_MAX_GROUP_LIST_SIZE = 1024UL;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr uint64_t PER_BLOCK_GROUP_SIZE = 128;
constexpr uint64_t SPLIT_M_W_DIMS = 3;
constexpr uint64_t SPLIT_K_W_DIMS = 2;
constexpr uint64_t X_DIMS = 2;
constexpr uint64_t BIAS_DIMS = 2;
constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2;
constexpr uint64_t MXFP_BASEK_FACTOR = 64;
constexpr size_t MXFP_TYPE_K_SCALE_DIM_NUM = 3;
constexpr size_t MXFP_TYPE_M_SCALE_DIM_NUM = 4;
constexpr size_t MXFP_PER_TOKEN_SCALE_DIM_NUM = 3;
constexpr uint32_t RESERVED_LENGTH = 1024;

constexpr size_t WEIGHTNZ_DIM_NUM = 5;
constexpr size_t WEIGHTNZ_FIRST_DIM = 0;
constexpr size_t WEIGHTNZ_SECOND_DIM = 1;
constexpr size_t WEIGHTNZ_THIRD_DIM = 2;
constexpr size_t WEIGHTNZ_FORTH_DIM = 3;
constexpr size_t WEIGHTNZ_FIFTH_DIM = 4;

constexpr uint32_t WEIGHTNZ_K0_16 = 16;
constexpr uint32_t WEIGHTNZ_N0_16 = 16;
constexpr uint32_t WEIGHTNZ_K0_32 = 32;
constexpr uint32_t WEIGHTNZ_N0_32 = 32;
constexpr uint8_t BF16_VALUE = 27;
constexpr uint32_t BF16_WORKSIZE = 2;
constexpr uint32_t FP32_WORKSIZE = 4;
constexpr uint64_t DB_REQUIRED_BYTES_SIZE = 14;
constexpr uint32_t SYS_WORKSPACE_SIZES = 16 * 1024 * 1024;
} // namespace GmmConstant

enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERGROUP_MODE = 0x1U << 4,
    PERBLOCK_MODE = 0x1U << 5,
};

typedef enum {
    GMM_ACT_TYPE_NONE = 0L,
    GMM_ACT_TYPE_RELU = 1L,
    GMM_ACT_TYPE_GELU_TANH = 2L,
    GMM_ACT_TYPE_GELU_ERR_FUNC = 3L,
    GMM_ACT_TYPE_FAST_GELU = 4L,
    GMM_ACT_TYPE_SILU = 5L,
} GMMActType;

struct GQmmBasicTiling {
    uint32_t usedCoreNum = 1;
    uint32_t tilingMode = 0;
    uint64_t singleCoreM = 1;
    uint64_t singleCoreN = 1;
    uint64_t singleCoreK = 1;
    uint64_t baseM = 1;
    uint64_t baseN = 1;
    uint64_t baseK = 1;
    uint64_t stepKa = 1;
    uint64_t stepKb = 1;
    uint64_t depthA1 = 1;
    uint64_t depthB1 = 1;
    uint64_t stepM = 1;
    uint64_t stepN = 1;
    uint32_t iterateOrder = 0;
    uint32_t dbL0c = 1;
    uint32_t calOrder = 0;
    uint32_t scaleFactorA = 1;
    uint32_t scaleFactorB = 1;
};

struct GQmmInputInfo {
    uint64_t mSize = 0UL;
    uint64_t kSize = 0UL;
    uint64_t nSize = 0UL;
    uint64_t groupNum = 0UL;
    int64_t outDtype = 0L;
    uint64_t kernelType = 0UL;
    QuantMode aQuantMode = QuantMode::DEFAULT;
    QuantMode bQuantMode = QuantMode::DEFAULT;
    int8_t groupType = Mc2GroupedMatmul::NO_SPLIT;
    int8_t groupListType = 0;
    int8_t splitItem = 0;
    int8_t actType = 0;
    const char *opName = nullptr;
    ge::DataType aDtype = ge::DT_INT8;
    ge::DataType bDtype = ge::DT_INT8;
    ge::DataType cDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_INT32;
    ge::DataType scaleDtype = ge::DT_UINT64;
    ge::DataType perTokenScaleDtype = ge::DT_FLOAT;
    ge::DataType outDataDtype = ge::DT_FLOAT16;
    ge::DataType outScaleDtype = ge::DT_FLOAT;

    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    ge::Format cFormat = ge::FORMAT_ND;
    bool transA = false;
    bool transB = false;
    bool hasBias = false;
    bool isSingleX = false;
    bool isSingleW = false;
    bool isSingleY = false;
};

class Mc2GroupedQbmmTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit Mc2GroupedQbmmTiling(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~Mc2GroupedQbmmTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Ops::Transformer::OpTiling::TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    virtual void Reset();
    void CalBasicBlock();
    ge::graphStatus CalL1Tiling();
    ge::graphStatus CalL1Depth(uint64_t leftL1Size);
    bool SetGroupNum(uint32_t groupListIndex);
    bool SetMKN(const gert::Shape &xShape, const gert::Shape &wShape);
    void SetKernelType();
    virtual bool AnalyzeAttrs();
    virtual bool AnalyzeDtype();
    virtual bool AnalyzeInputs();
    virtual void PrintQuantParams();
    bool IsMicroScaling() const;
    bool CheckQuantParamsForMXTypeM(const gert::Shape &xScaleShape, const gert::Shape &wScaleShape) const;
    GQmmBasicTiling basicTiling_;
    GQmmInputInfo inputParams_;

    uint64_t GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit);
    void CalStepKs();
    void CalScaleFactors();
    uint64_t GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const;
    uint64_t GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const;
    bool SetQuantMode(const gert::Shape &wScaleShape, const gert::StorageShape *xScaleStorageShape,
                      const gert::Shape &wShape);
    void SetPerGroupQuantMode(const gert::Shape &xScaleShape, const gert::Shape &wScaleShape,
                              const gert::Shape &wShape);
    bool CheckQuantParamsForMXTypeK(const gert::Shape &xScaleShape, const gert::Shape &wScaleShape) const;
    bool CheckFp4Shape() const;
    bool CheckBiasDtype() const;
    bool CheckBiasShape(const gert::StorageShape *biasStorageShape) const;
    bool CheckQuantParamsForMxQuantMode(const gert::StorageShape *xScaleStorageShape,
                                        const gert::Shape &wScaleShape) const;
    bool CheckQuantParams(const gert::StorageShape *xScaleStorageShape, const gert::Shape &wScaleShape) const;
    bool CheckQuantParamsForNonKGroupQuantMode(const gert::Shape &wScaleShape) const;
    bool SetMKNList();
    bool IsBiasInL1() const;
    bool CheckDtypeForWeightNz(bool isPertokenScaleNull) const;
    bool CheckShapeForWeightNz(const gert::Shape &wShape) const;
    bool CheckActiveModeDtype(const gert::StorageShape *xScaleStorageShape) const;
    bool CheckActiveMode(const gert::Shape &wScaleShape, const gert::StorageShape *xScaleStorageShape);

    Mc2GroupedMatmulTilingData::GMMQuantTilingData tilingData_;
    bool isWeightNz_ = false;

    int32_t mList_[Mc2GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int32_t kList_[Mc2GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int32_t nList_[Mc2GroupedMatmul::MAX_TENSOR_CONT] = {0};
};
} // namespace Mc2GroupedMatmulTiling
} // namespace optiling

#endif // MC2_GMM_QBMM_TILING_H
