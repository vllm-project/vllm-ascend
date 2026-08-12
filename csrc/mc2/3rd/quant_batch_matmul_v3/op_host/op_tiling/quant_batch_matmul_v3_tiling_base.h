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
 * \file quant_batch_matmul_v3_tiling_base.h
 * \brief
 */

#ifndef MC2_QUANT_BATCH_MATMUL_V3_TILING_BASE_H
#define MC2_QUANT_BATCH_MATMUL_V3_TILING_BASE_H
#include <memory>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "quant_batch_matmul_v3_compile_info.h"
#include "op_host/tiling_base.h"
#include "ops_legacy/op_tiling/op_cache_tiling.h"
#include "common/op_host/math_util.h"

namespace optiling {

constexpr uint64_t BASIC_ALIGN_16 = 16;

/**
 * Mc2QuantBatchMatmulInfo 增改参数规则：
 *   1. 新增参数需要初始化
 *   2. 检查mc2依赖文件,清单列表:
 *     ops/built-in/op_tiling/runtime/matmul_all_reduce/quant_matmul_all_reduce_tiling.h
 *     ops/built-in/op_tiling/runtime/matmul_all_reduce/quant_matmul_all_reduce_tiling.cc
 *     ops/built-in/op_tiling/runtime/matmul_all_reduce/quant_matmul_all_reduce_tiling_310_general.h
 *     ops/built-in/op_tiling/runtime/matmul_all_reduce/quant_matmul_all_reduce_tiling_310_general.cc
 *     ops/built-in/op_tiling/runtime/quant_batch_matmul_v3/quant_batch_matmul_v3_tiling.cc
 *   3. 增加新参数需要和MC2开发者说明
 */
struct Mc2QuantBatchMatmulInfo {
public:
    uint64_t GetMatmulApiMSize() const; // mm api单次计算的M
    uint64_t GetTotalMatmulApiMSize(uint64_t baseM) const;
    uint64_t GetTotalBaseMCnt(uint64_t baseM) const;
    void Reset(); // 新增数据成员要修改Reset函数

    bool initFlag = false; // 避免重复解析flag
    bool transA = false;
    bool transB = false;
    bool hasBias = false;
    uint64_t mSize = 0UL;
    uint64_t mSizePerNpu = 0UL;
    uint64_t kSize = 0UL;
    uint64_t nSize = 0UL;
    uint64_t batchA = 0UL;
    uint64_t batchA1 = 0UL;
    uint64_t batchA2 = 0UL;
    uint64_t batchA3 = 0UL;
    uint64_t batchA4 = 0UL;
    uint64_t batchB = 0UL;
    uint64_t batchB1 = 0UL;
    uint64_t batchB2 = 0UL;
    uint64_t batchB3 = 0UL;
    uint64_t batchB4 = 0UL;
    uint64_t batchC = 0UL;
    uint64_t batchC1 = 0UL;
    uint64_t batchC2 = 0UL;
    uint64_t batchC3 = 0UL;
    uint64_t batchC4 = 0UL;
    uint64_t batchBias = 0UL;
    ge::DataType aDtype = ge::DT_INT8;
    ge::DataType bDtype = ge::DT_INT8;
    ge::DataType cDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_INT32;
    ge::DataType scaleDtype = ge::DT_UINT64;
    ge::DataType perTokenScaleDtype = ge::DT_FLOAT;
    bool isPerTensor = false;
    bool isPerChannel = false;
    bool isPertoken = false;
    bool isDoubleScale = false;
    bool isMxPerGroup = false;
    bool isPerBlock = false;
    bool isPerBlockPerToken = false;
    int64_t outDtype = 0L;
    uint32_t libApiWorkSpaceSize = 0U;
    uint64_t bf16ExtreWorkSpaceSize = 0UL;
    uint64_t groupSize = 0UL;
    uint64_t groupSizeM = 0UL;
    uint64_t groupSizeK = 0UL;
    uint64_t groupSizeN = 0UL;
    const char *opName = nullptr;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    ge::Format cFormat = ge::FORMAT_ND; // 新增数据成员要修改Reset函数
};

enum class Mc2QuantBatchMatmulV3Trans : uint32_t {
    NO_TRANS = 0, // transA 0 transB 0
    A_TRANS = 1,  // transA 1 transB 0
    B_TRANS = 2,  // transA 0 transB 1
    AB_TRANS = 3  // transA 1 transB 1
};

enum class BasicQuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
};

class Mc2QuantBatchMatmulV3TilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit Mc2QuantBatchMatmulV3TilingBase(gert::TilingContext *context, bool isTilingOut);
    ~Mc2QuantBatchMatmulV3TilingBase() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    void InitCompileInfo();

    // mc2使用的直接接口：begin
    virtual const gert::Shape GetX1Shape(const size_t index);
    virtual const gert::Shape GetX2Shape(const size_t index);
    virtual const gert::Shape GetOutputShape(const size_t index);
    virtual const gert::Shape &GetScaleShape(const size_t index);
    virtual const gert::StorageShape *GetPertokenShape(const size_t index);
    virtual const gert::StorageShape *GetBiasShape(const size_t index);
    virtual bool AnalyzeInputs();
    // mc2使用的直接接口：end

    virtual bool GetUbDequantExtreSpace();
    virtual ge::graphStatus CalcUbTiling();
    virtual bool CheckDtype() const;
    virtual bool CheckShape(const std::vector<gert::Shape *> &mandtoryShape, const gert::StorageShape *biasShape,
                            const gert::StorageShape *pertokenShape, const std::vector<int64_t> &dimValueOfMKN) const;

    virtual ge::graphStatus CheckContext();
    virtual bool AnalyzeDtype();
    virtual bool AnalyzeAttrs();
    void SetFormat();
    bool SetQuantMode(const gert::Shape &scaleShape, const gert::StorageShape *pertokenShape);
    bool SetX1QuantMode(BasicQuantMode &x1QuantMode, bool isFp8Hif8Input, const gert::StorageShape *pertokenShape);
    bool SetX2QuantMode(BasicQuantMode &x2QuantMode, bool isFp8Hif8Input, const gert::Shape &scaleShape);
    std::string QuantModeMapToString(const std::vector<BasicQuantMode> &quantModeList) const;
    bool IsMicroScaling() const;
    std::string QuantModeToString(BasicQuantMode quantMode) const;
    uint64_t GetBatchSize(const gert::Shape &shape) const;
    bool InferOutBatchDim(const gert::Shape &x1Shape, const gert::Shape &x2Shape);
    int8_t CheckFusionBatchA(const gert::Shape &x1Shape, const gert::Shape &x2Shape, const gert::Shape &biasShape,
                             uint64_t fusedDimValue) const;
    bool CheckOutputShapeAvailable() const;
    bool ReCalcGroupSize(uint64_t &groupSize, uint64_t inputSize, uint64_t scaleSize, const char *dimensionName);
    bool AnalyzeGroupInfo(const gert::Shape &scaleShape, const gert::StorageShape *pertokenShape);
    virtual void AnalyzeBatchInfo(const gert::Shape &oriShapeA, const gert::Shape &oriShapeB);
    void DoBatchFusion(uint64_t fusedDimValue);
    bool CheckShapeInRangeForMandtoryInputs(size_t x1ShapeLen, size_t x2ShapeLen) const;
    void SetTransAttr(Mc2QuantBatchMatmulV3Trans &trans) const;
    virtual bool SetPlatformInfoForTiling();

    template <typename T>
    inline bool CheckNumberIsValid(const T &num, const std::string &opName, const std::string &description) const
    {
        if (num > static_cast<uint64_t>(INT32_MAX)) {
            OP_LOGW(opName.c_str(), "%s size is greater than INT32_MAX or less than 0, num:%s", description.c_str(),
                    std::to_string(num).c_str());
            return true;
        }
        return false;
    };

    // 需要对齐16的参数需要判断是否大于floorAlign(uint32_max, 16)
    template <typename T>
    inline bool CheckNumberIsValid2(const T &num, const std::string &opName, const std::string &description) const
    {
        if (num > ops::FloorAlign(static_cast<uint64_t>(INT32_MAX), BASIC_ALIGN_16)) {
            OP_LOGW(opName.c_str(), "%s size is greater than floorAlign(INT32_MAX, 16) or less than 0, num:%s",
                    description.c_str(), std::to_string(num).c_str());
            return true;
        }
        return false;
    };

    template <typename T>
    T GetShapeWithDataType(T size, ge::DataType dtype) const
    {
        if (dtype == ge::DT_INT4 || dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
            return size + size;
        } else {
            return size / static_cast<T>(ge::GetSizeByDataType(dtype));
        }
    }

    template <typename T>
    T GetSizeWithDataType(T shape, ge::DataType dtype) const
    {
        if (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4) {
            return (shape + 1) >> 1;
        } else {
            return shape * static_cast<T>(ge::GetSizeByDataType(dtype));
        }
    }

    static constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
    static constexpr uint64_t KB_SIZE = 1024;
    static constexpr uint32_t ROW_FIRST = 1;
    static constexpr uint32_t COL_FIRST = 2;
    static constexpr uint64_t NUM_DB = 2;

    // 新增数据成员请注意：如果是在GetShapeAttrsInfo函数过程中获取的，请放到Mc2QuantBatchMatmulInfo结构体中，或者保证在DoOpTiling赋值
    Mc2QuantBatchMatmulInfo &inputParams_;
    bool isBf16Opt_ = false;
    bool isUbQuant_ = false;
    CacheTilingData tbeTiling_;
    Mc2QuantBatchMatmulV3CompileInfo compileInfo_;
    bool compileInfoInit_ = false;
    bool isTilingOut_ = false;
    size_t tilingDataSize_ = 0;
};
} // namespace optiling
#endif // QUANT_BATCH_MATMUL_V3_TILING_BASE_H
