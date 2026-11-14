/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_situ_quant_tiling.h
 * \brief GroupedMatmul + dequant + SiTU-GLU + quant (A8W4 MSD) 的 tiling 数据结构与常量。
 *
 * 本算子只覆盖 A2/A3 的 A8W4 MSD 场景：
 *   x: INT8, weight: INT4, weight_scale: UINT64, y: INT8, y_scale: FLOAT
 * 计算流程与 GroupedMatmulSwigluQuantV2 的 A8W4 MSD 路径一致，仅把激活从 SwiGLU 换成 SiTU。
 */
#ifndef __OP_HOST_OP_TILING_GROUPED_MATMUL_SITU_QUANT_TILING_H__
#define __OP_HOST_OP_TILING_GROUPED_MATMUL_SITU_QUANT_TILING_H__

#include <cstdint>
#include <set>
#include "tiling_base/tiling_base.h"
#include "platform/platform_infos_def.h"
#include "tiling/tiling_api.h"

namespace optiling {

// GMM 基本信息（与 kernel 侧 GMMSituQuantBaseParams 一一对应）
BEGIN_TILING_DATA_DEF(GMMSituQuantBaseParams)
TILING_DATA_FIELD_DEF(int64_t, groupNum);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, K);
TILING_DATA_FIELD_DEF(int64_t, N);
TILING_DATA_FIELD_DEF(int64_t, M);
TILING_DATA_FIELD_DEF(int64_t, baseM);
TILING_DATA_FIELD_DEF(int64_t, baseN);
TILING_DATA_FIELD_DEF(int64_t, mLimit);
TILING_DATA_FIELD_DEF(int64_t, workSpaceOffset1);
TILING_DATA_FIELD_DEF(int64_t, workSpaceOffset2);
TILING_DATA_FIELD_DEF(int64_t, quantGroupNum);
TILING_DATA_FIELD_DEF(int64_t, isSingleTensor);
TILING_DATA_FIELD_DEF(int64_t, groupListType);
TILING_DATA_FIELD_DEF(int64_t, smoothScaleDimNum);
TILING_DATA_FIELD_DEF(int64_t, singleN);
// 每专家平均行数不超过该值时，kernel 改用 mmTilingDataAlt 那套分块（0 = 不用备选）。
// 放在这里而不是顶层结构体里，是因为 kernel 侧把 baseParams 指针一路传到 pre/mid/post，
// 决策点只在这一个结构体上，省一次参数透传。
TILING_DATA_FIELD_DEF(int64_t, altRowsPerExpertLimit);
// 补偿矩阵的自算开关：调用方没有提供 weight_assist_matrix 时置 1，kernel 自己算。
//
// 为什么要多一行：pre 把 int8 的 x 拆成高低两个 int4 时，低半字节带了一个固定的 -8
// （x = 16*high + low + 8），所以 cube 出来的是 (x-8)*w*scale，整体少了 8*sum_k(w)*scale。
// 这一项只和权重有关。调用方给的时候是 host 上算好的 [E, N]；不给的时候，kernel 在每个
// 专家块的有效行之后再打一行全 -8 的 int4 A 行（0x88），cube 顺手算出来
// cAux = (-8)*sum_k(w)*scale，post 减掉即可。这一行贴着专家自己的行放，是为了让它落在
// 同一个 M tile 里、跟那些行共用 B —— 否则要多流一遍该专家的权重，decode 档直接翻倍。
//
// 1 时每个「本 split 内非空」的专家块在 A/C 两个 workspace 里各多一行，pre/mid/post 三处
// 都按「专家块末尾多一行」来寻址，见各自文件里的注释。
TILING_DATA_FIELD_DEF(int64_t, assistRowsPerExpert);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMSituQuantBaseParamsOp, GMMSituQuantBaseParams)

// SiTU 激活 + 量化部分 tiling 基本信息
//
// 相比 GroupedMatmulSwigluQuantV2 的同名结构，尾部追加了三个 SiTU 参数。
// 追加在末尾以保证已有字段的序列化偏移不变。
BEGIN_TILING_DATA_DEF(GMMSituQuant)
TILING_DATA_FIELD_DEF(int64_t, maxProcessRowNum);
TILING_DATA_FIELD_DEF(int64_t, groupListLen);
TILING_DATA_FIELD_DEF(int64_t, tokenLen);
// ---- SiTU 参数（本算子新增）----
// beta:         SiTU 的 beta，必须 > 0。gate 分支为 beta * tanh(gate / beta) * sigmoid(gate)
// linearBeta:   up 分支的 linear_beta，<= 0 时不启用（up 保持原值）
// activateLeft: 1 -> gate 取前半、up 取后半；0 -> gate 取后半、up 取前半
TILING_DATA_FIELD_DEF(float, beta);
TILING_DATA_FIELD_DEF(float, linearBeta);
TILING_DATA_FIELD_DEF(int64_t, activateLeft);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMSituQuantOp, GMMSituQuant)

// 结构体集合
BEGIN_TILING_DATA_DEF(GMMSituQuantTilingData)
TILING_DATA_FIELD_DEF_STRUCT(GMMSituQuantBaseParams, gmmSituQuantBaseParams);
TILING_DATA_FIELD_DEF_STRUCT(GMMSituQuant, gmmSituQuant);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
// 备选分块（baseM/baseN 互换）的 Matmul tiling。kernel 按 live 行数在两者之间二选一，
// 见 A8W4_BASEM_NZ_ALT 处的注释；不用备选时这里填与 mmTilingData 相同的内容。
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingDataAlt);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupedMatmulSituQuant, GMMSituQuantTilingData)

struct GMMSituCompileInfo {
    uint64_t ubSize_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    uint32_t baseM_ = 128;
    uint32_t baseN_ = 256;
    bool supportL12BtBf16 = false;
    uint64_t l1Size_ = 0;
    uint64_t l0CSize_ = 0;
    int32_t npuArch_ = 0;
};

namespace GroupedMatmulSituQuantTiling {
// 槽位索引必须与 grouped_matmul_situ_quant_def.cpp 的声明顺序一致：
// A5 原有的 x/x_scale/weight/weight_scale/group_list 保持在前，
// A2A3 额外需要的 weight_assist_matrix / bias / smooth_scale 追加在尾部。
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t X_SCALE_INDEX = 1;
constexpr uint32_t WEIGHT_INDEX = 2;
constexpr uint32_t WEIGHT_SCALE_INDEX = 3;
constexpr uint32_t GROUPLIST_INDEX = 4;
constexpr uint32_t WEIGHT_ASSIST_MATRIX_INDEX = 5;
constexpr uint32_t BIAS_INDEX = 6;
constexpr uint32_t SMOOTH_SCALE_INDEX = 7;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t Y_SCALE_INDEX = 1;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;

// 属性顺序必须与 grouped_matmul_situ_quant_def.cpp 中 Attr 的声明顺序一致：
// A5 原有的 group_list_type / beta / linear_beta 在前，A2A3 扩展属性追加在尾部。
constexpr uint32_t ATTR_INDEX_GROUPLIST_TYPE = 0;
constexpr uint32_t ATTR_INDEX_BETA = 1;
constexpr uint32_t ATTR_INDEX_LINEAR_BETA = 2;
constexpr uint32_t ATTR_INDEX_DEQUANT_MODE = 3;
constexpr uint32_t ATTR_INDEX_DEQUANT_DTYPE = 4;
constexpr uint32_t ATTR_INDEX_QUANT_MODE = 5;
constexpr uint32_t ATTR_INDEX_QUANT_DTYPE = 6;
constexpr uint32_t ATTR_INDEX_TRANSPOSE_WEIGHT = 7;
constexpr uint32_t ATTR_INDEX_TUNING_CONFIG = 8;
constexpr uint32_t ATTR_INDEX_ACTIVATE_LEFT = 9;

constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
constexpr uint32_t DIM_4 = 4;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t NUM_EIGHT = 8;
constexpr uint32_t SYS_WORKSPACE_SIZE = static_cast<uint32_t>(16 * 1024 * 1024);
constexpr int64_t USER_WORKSPACE_LIMIT = static_cast<int64_t>(64 * 1024 * 1024);

// A8W4 MSD 的对齐约束，与 op_api 侧 GroupedMatmulSituQuantBaseHandler 保持一致。
constexpr int64_t A8W4_ALIGN_K = 64L;    // INT4 fractal 的 K 方向对齐
constexpr int64_t A8W4_ALIGN_N = 16L;    // per-token 量化落盘的 UB 地址对齐
constexpr int64_t A8W4_ALIGN_NZ_N = 64L; // NZ 排布 N 方向的 tile
constexpr int64_t DOUBLE_WORKSPACE_SPLIT = 2;
constexpr int64_t INT32_DTYPE_SIZE = 4;
constexpr int64_t FP32_DTYPE_SIZE = 4;
constexpr int64_t FP32_BLOCK_SIZE = 8;
constexpr int64_t BLOCK_BYTE = 32;
constexpr int64_t SWIGLU_REDUCE_FACTOR = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t ND_WEIGHT_DIM_LIMIT = 3;
constexpr int64_t NZ_WEIGHT_DIM_LIMIT = 5;
constexpr int64_t DOUBLE_ROW = 2;
constexpr int64_t PERCHANNEL_WSCALE_DIM_LIMIT = 2;
constexpr int64_t PERGROUP_WSCALE_DIM_LIMIT = 3;
constexpr int64_t A8W4_MSD_TILING_KEY_MODE = 2;
constexpr int64_t COMMON_TILING_KEY_MODE = 0;
// Keep each FP32 accumulation tile within the 128 KiB L0C budget.
// ND uses wider N tiles for contiguous weight loads. NZ chooses between
// two tile shapes using device-side live rows, including graph replay.
constexpr int64_t A8W4_BASEM = 64;
constexpr int64_t A8W4_BASEK = 128;
constexpr int64_t A8W4_BASEN = 512;
constexpr int64_t A8W4_BASEM_NZ = 256;
constexpr int64_t A8W4_BASEK_NZ = 256;
constexpr int64_t A8W4_BASEN_NZ = 128;
constexpr int64_t A8W4_BASEM_NZ_ALT = 128;
constexpr int64_t A8W4_BASEN_NZ_ALT = 256;
// Zero disables the alternate shape (ND); NZ switches at this live average.
constexpr int64_t A8W4_ALT_ROWS_PER_EXPERT_LIMIT = 32;
constexpr int64_t SIZE_OF_HALF_2 = 2;

// SiTU 属性默认值。与 dequant_situ_quant 的调用约定保持一致：
// beta=1.0、linear_beta=0.0（不启用 up 变换）、activate_left=true（gate 在前半）。
constexpr float SITU_DEFAULT_BETA = 1.0f;
constexpr float SITU_DEFAULT_LINEAR_BETA = 0.0f;
constexpr int64_t SITU_DEFAULT_ACTIVATE_LEFT = 1;

class GroupedMatmulSituQuantTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GroupedMatmulSituQuantTiling(gert::TilingContext* context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context) {};

    ~GroupedMatmulSituQuantTiling() override = default;

protected:
    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    };

    virtual void FillTilingData() = 0;
    virtual void PrintTilingData() = 0;
};

} // namespace GroupedMatmulSituQuantTiling
} // namespace optiling

#endif // __OP_HOST_OP_TILING_GROUPED_MATMUL_SITU_QUANT_TILING_H__
