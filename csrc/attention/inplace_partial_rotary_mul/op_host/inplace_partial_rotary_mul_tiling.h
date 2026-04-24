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
 * \file rotary_position_embedding.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_H


#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
// #include "tiling_base/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
// #include "tiling_base/tiling_base.h"
// #include "tiling_base/tiling_util.h"
#include "platform/platform_info.h"
// #include "util/math_util.h"
#include "error/ops_error.h"
namespace optiling {

BEGIN_TILING_DATA_DEF(RopeRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, CosB);
TILING_DATA_FIELD_DEF(int64_t, S);
TILING_DATA_FIELD_DEF(int64_t, D);
TILING_DATA_FIELD_DEF(int64_t, N);
TILING_DATA_FIELD_DEF(int64_t, blockNumB);
TILING_DATA_FIELD_DEF(int64_t, blockFactorB);
TILING_DATA_FIELD_DEF(int64_t, blockNumS);
TILING_DATA_FIELD_DEF(int64_t, blockFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNumS);
TILING_DATA_FIELD_DEF(int64_t, ubFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNumB);
TILING_DATA_FIELD_DEF(int64_t, ubFactorB);
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorB);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNumN);
TILING_DATA_FIELD_DEF(int64_t, ubFactorN);
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorN);
TILING_DATA_FIELD_DEF(int64_t, rotaryMode);
TILING_DATA_FIELD_DEF(int64_t, dAlign);
TILING_DATA_FIELD_DEF(int64_t, dSplitCoef);
TILING_DATA_FIELD_DEF(int64_t, blockNumBS);
TILING_DATA_FIELD_DEF(int64_t, blockFactorBS);
TILING_DATA_FIELD_DEF(int64_t, blockTailBS);
TILING_DATA_FIELD_DEF(int64_t, blockNumN);
TILING_DATA_FIELD_DEF(int64_t, blockFactorN);
TILING_DATA_FIELD_DEF(int64_t, blockTailN);
TILING_DATA_FIELD_DEF(int64_t, ubFactorBS);
TILING_DATA_FIELD_DEF(int64_t, sliceStart);
TILING_DATA_FIELD_DEF(int64_t, sliceEnd);
TILING_DATA_FIELD_DEF(int64_t, sliceLength);
// A3
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numHead);
TILING_DATA_FIELD_DEF(int64_t, headDim);
TILING_DATA_FIELD_DEF(int64_t, allHeadDim);
TILING_DATA_FIELD_DEF(int64_t, coreTUbLoopTime);
TILING_DATA_FIELD_DEF(int64_t, coreBUbLoopTime);
TILING_DATA_FIELD_DEF(int64_t, coreTUbLoopTail);
TILING_DATA_FIELD_DEF(int64_t, coreBUbLoopTail);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, start);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
//A3 主线
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, seqLen);
TILING_DATA_FIELD_DEF(int64_t, numHeads);
TILING_DATA_FIELD_DEF(int64_t, frontCoreNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);
TILING_DATA_FIELD_DEF(int64_t, coreCalcNum);
TILING_DATA_FIELD_DEF(int64_t, coreCalcTail);
TILING_DATA_FIELD_DEF(int64_t, ubCalcNum);
TILING_DATA_FIELD_DEF(int64_t, ubCalcLoop);
TILING_DATA_FIELD_DEF(int64_t, ubCalcTail);
TILING_DATA_FIELD_DEF(int64_t, ubCalcTailNum);
TILING_DATA_FIELD_DEF(int64_t, ubCalcTailLoop);
TILING_DATA_FIELD_DEF(int64_t, ubCalcTailTail);
TILING_DATA_FIELD_DEF(int64_t, ubCalcBNum);
TILING_DATA_FIELD_DEF(int64_t, ubCalcBLoop);
TILING_DATA_FIELD_DEF(int64_t, ubCalcBTail);
TILING_DATA_FIELD_DEF(int64_t, ubCalcNNum);
TILING_DATA_FIELD_DEF(int64_t, ubCalcNLoop);
TILING_DATA_FIELD_DEF(int64_t, ubCalcNTail);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InplacePartialRotaryMul, RopeRegbaseTilingData)

ge::graphStatus Tiling4InplacePartialRotaryMul(gert::TilingContext* context);
struct RotaryPositionEmbeddingCompileInfo {
    int64_t blockDim;
    uint64_t ubSize;
    platform_ascendc::SocVersion socVersion;
};

struct AiCoreParams {
    uint64_t ubSize = 0;
    uint64_t blockDim = 0;
    uint64_t aicNum = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
};

enum class RopeLayout : uint8_t {
    NO_BROADCAST = 1,
    BROADCAST_BSN = 2,
    BSND = 3,
    SBND = 4,
    BNSD = 5
};

enum class RotaryPosEmbeddingMode : uint8_t {
    HALF = 0,
    INTERLEAVE = 1,
    QUARTER = 2,
    DEEPSEEK_INTERLEAVE = 3
};

template <typename T>
static inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)));
}

template <typename T>
static inline T CeilAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)) * (rnd));
}

template <typename T>
static inline T FloorDiv(T x, T y)
{
    if (y == 0)
    {
        return 0;
    }
    return x / y;
}

template <typename T>
static inline T FloorAlign(T x, T y)
{
    if (y == 0)
    {
        return 0;
    }
    return x / y * y;
}

class RotaryPosEmbeddingMembaseTilingClass {
public:
    explicit RotaryPosEmbeddingMembaseTilingClass(gert::TilingContext *context) : context_(context)
    {
    }

    void Reset(gert::TilingContext *context)
    {
        RotaryPosEmbeddingMembaseTilingClass::Reset(context);
    }

    ge::graphStatus GetPlatformInfo();

    ge::graphStatus GetWorkspaceSize()
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling()
    {
        return ge::GRAPH_SUCCESS;
    }

    bool IsCapable()
    {
        return true;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling()
    {
        return ge::GRAPH_SUCCESS;
    }
    // 7、保存Tiling数据
    ge::graphStatus PostTiling()
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo();

    uint64_t GetTilingKey() const
    {
        return context_->GetTilingKey();
    }

protected:
    static const uint32_t MODE_ROTATE_INTERLEAVED = 1;
    uint32_t inputMode_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    gert::TilingContext* context_ = nullptr;
    std::unique_ptr<platform_ascendc::PlatformAscendC> ascendcPlatform_{nullptr};
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
    AiCoreParams aicoreParams_;
};

class RopeRegBaseTilingClass {
public:
    explicit RopeRegBaseTilingClass(gert::TilingContext *context) : context_(context)
    {
    }

    void Reset(gert::TilingContext *context)
    {
        RopeRegBaseTilingClass::Reset(context);
    }

    bool IsRotaryPosEmbeddingMode(const int32_t mode) const;
    ge::graphStatus CheckNullptr();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckDtypeAndAttr();
    ge::graphStatus CheckParam();
    ge::graphStatus JudgeLayoutByShape(const gert::Shape &xShape, const gert::Shape &cosShape);
    ge::graphStatus CheckRotaryModeShapeRelation(const int64_t d);
    ge::graphStatus CheckShapeAllPositive(const int64_t idx) const;
    ge::graphStatus CheckShapeAllPositive() const;
    std::string rotaryModeStr_;

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus JudgeSliceInfo();
    virtual ge::graphStatus GetWorkspaceSize() 
    {
        return ge::GRAPH_SUCCESS;
    }

    virtual ge::graphStatus DoLibApiTiling() 
    {
        return ge::GRAPH_SUCCESS;
    }

    virtual ge::graphStatus DoOpTiling() 
    {
        return ge::GRAPH_SUCCESS;
    }

    virtual uint64_t GetTilingKey() const 
    {
        return 0;
    }

    virtual ge::graphStatus PostTiling() 
    {
        return ge::GRAPH_SUCCESS;
    }

    virtual bool IsCapable() 
    {
        return true;
    }

    bool IsRegbaseSocVersion()
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        auto socVersion = ascendcPlatform.GetSocVersion();
        return socVersion == platform_ascendc::SocVersion::ASCEND910_95;
    }

    ge::graphStatus DoTiling()
    {
        auto ret = GetShapeAttrsInfo();
        if (ret != ge::GRAPH_SUCCESS)
        {
            return ret;
        }
        ret = GetPlatformInfo();
        if (ret != ge::GRAPH_SUCCESS)
        {
            return ret;
        }
        if (!IsCapable())
        {
            return ge::GRAPH_PARAM_INVALID;
        }
        ret = DoOpTiling();
        if (ret != ge::GRAPH_SUCCESS)
        {
            return ret;
        }
        ret = DoLibApiTiling();
        if (ret != ge::GRAPH_SUCCESS)
        {
            return ret;
        }
        ret = GetWorkspaceSize();
        if (ret != ge::GRAPH_SUCCESS)
        {
            return ret;
        }
        ret = PostTiling();
        if (ret != ge::GRAPH_SUCCESS)
        {
            return ret;
        }
        context_->SetTilingKey(GetTilingKey());
        return ge::GRAPH_SUCCESS;
    }

protected:
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910_95;
    int64_t b_{0};
    int64_t s_{0};
    int64_t n_{0};
    int64_t d_{0};
    int64_t cosb_{0};
    ge::DataType dtype_;
    RopeLayout layout_;
    RotaryPosEmbeddingMode rotaryMode_;

    int64_t blockSize_;
    int64_t dSplitCoef_;
    bool is1snd_ = false;
    gert::TilingContext* context_ = nullptr;
    std::unique_ptr<platform_ascendc::PlatformAscendC> ascendcPlatform_{nullptr};
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
    AiCoreParams aicoreParams_;
    int64_t sliceStart_{0};
    int64_t sliceEnd_{0};
    int64_t sliceLength_{0};
    int64_t cosd_{0};
    int64_t sind_{0};
};

class RopeRegBaseTilingClassAAndB : public RopeRegBaseTilingClass {
public:
    explicit RopeRegBaseTilingClassAAndB(gert::TilingContext *context) : RopeRegBaseTilingClass(context)
    {
    }

    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void SetTilingData();

private:
    ge::graphStatus MergeDim();
    ge::graphStatus SplitCore();
    ge::graphStatus ComputeUbFactor();

    int64_t blockNumB_{0};
    int64_t blockFactorB_{0};
    int64_t blockNumS_{0};
    int64_t blockFactorS_{0};
    int64_t ubFactorB_{0};
    int64_t ubLoopNumB_{0};
    int64_t ubTailFactorB_{0};
    int64_t ubFactorS_{0};
    int64_t ubLoopNumS_{0};
    int64_t ubTailFactorS_{0};
    int64_t ubFactorN_{0};
    int64_t ubLoopNumN_{0};
    int64_t ubTailFactorN_{0};
    RopeRegbaseTilingData tilingData_;
};

class RopeRegBaseTilingClassAB : public RopeRegBaseTilingClass {
public:
    explicit RopeRegBaseTilingClassAB(gert::TilingContext *context) : RopeRegBaseTilingClass(context)
    {
    }

    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    int64_t blockNumBS_ = 0;
    int64_t blockFactorBS_ = 0;
    int64_t blockTailBS_ = 0;
    int64_t blockNumN_ = 0;
    int64_t blockFactorN_ = 0;
    int64_t blockTailN_ = 0;
    int64_t ubFactorBS_ = 0;
    int64_t ubFactorN_ = 0;
    int64_t dAlign_ = 0;
    RopeRegbaseTilingData tilingData_;
};

class RopeRegBaseTilingClassABAAndBA : public RopeRegBaseTilingClass {
public:
    explicit RopeRegBaseTilingClassABAAndBA(gert::TilingContext *context) : RopeRegBaseTilingClass(context)
    {
    }

    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void SetTilingData();

private:
    ge::graphStatus SplitCore();
    ge::graphStatus ComputeUbFactor();

    int64_t blockNumB_{0};
    int64_t blockFactorB_{0};
    int64_t blockNumS_{0};
    int64_t blockFactorS_{0};
    int64_t ubFactorB_{0};
    int64_t ubLoopNumB_{0};
    int64_t ubTailFactorB_{0};
    int64_t ubFactorS_{0};
    int64_t ubLoopNumS_{0};
    int64_t ubTailFactorS_{0};
    int64_t ubFactorN_{0};
    int64_t ubLoopNumN_{0};
    int64_t ubTailFactorN_{0};
    RopeRegbaseTilingData tilingData_;
};

class RopeRegBaseTilingClassBAB : public RopeRegBaseTilingClass {
public:
    explicit RopeRegBaseTilingClassBAB(gert::TilingContext *context_) : RopeRegBaseTilingClass(context_)
    {
    }

    // 计算数据切分
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 设置Tiling数据
    ge::graphStatus PostTiling() override;

    bool IsCapable() override
    {
        // BSND format, 1s1d模版，后续可扩展支持所有bab类型的boardcast
        if (IsRegbaseSocVersion() && (layout_ == RopeLayout::BSND) &&
            (cosb_ == 1)) {
            return true;
        }
        return false;
    }

private:
    int64_t coreNum_ = 0;
    int64_t blockNumB_ = 0;
    int64_t blockFactorB_ = 0;
    int64_t blockNumS_ = 0;
    int64_t blockFactorS_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t ubLoopNumS_ = 0;
    int64_t ubFactorS_ = 1;
    int64_t ubTailFactorS_ = 0;
    int64_t ubLoopNumB_ = 0;
    int64_t ubFactorB_ = 1;
    int64_t ubTailFactorB_ = 0;
    int64_t ubLoopNumN_ = 0;    // 核内处理N循环了多少次
    int64_t ubFactorN_ = 1;     // 每次循环处理多少个N
    int64_t ubTailFactorN_ = 0; // 最后一次循环处理多少N
    int64_t ubSize_ = 0;
    uint64_t tilingKey_ = 0;

    ge::graphStatus SplitCore();
    ge::graphStatus SplitUb();
    void PrintTilingData();
    RopeRegbaseTilingData tilingData_;
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_H

