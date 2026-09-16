/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_KEY_INDEXER_TILING_H
#define POOL_KEY_INDEXER_TILING_H

#include <vector>
#include "err/ops_err.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
enum class PkiDataLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

// ------------------ Input/Output Index ------------------
constexpr uint32_t PKI_QUERY_INDEX = 0;
constexpr uint32_t PKI_POOL_KEY_INDEX = 1;
constexpr uint32_t PKI_WEIGHTS_INDEX = 2;
constexpr uint32_t PKI_POOL_TAIL_K_INDEX = 3;
constexpr uint32_t PKI_ACTUAL_SEQ_Q_INDEX = 4;
constexpr uint32_t PKI_ACTUAL_SEQ_K_INDEX = 5;
constexpr uint32_t PKI_BLOCK_TABLE_INDEX = 6;
constexpr uint32_t PKI_Q_DESCALE_INDEX = 7;
constexpr uint32_t PKI_K_DESCALE_INDEX = 8;
constexpr uint32_t PKI_SPARSE_INDICES_INDEX = 0;
constexpr uint32_t PKI_SPARSE_VALUES_INDEX = 1;

// ------------------ Attribute Index ------------------
constexpr uint32_t PKI_ATTR_TOPK = 0;
constexpr uint32_t PKI_ATTR_POOL_SIZE = 1;
constexpr uint32_t PKI_ATTR_LAYOUT_Q = 2;
constexpr uint32_t PKI_ATTR_LAYOUT_K = 3;
constexpr uint32_t PKI_ATTR_MASK_MODE = 4;
constexpr uint32_t PKI_ATTR_QUANT_MODE = 5;
constexpr uint32_t PKI_ATTR_RETURN_VALUE = 6;
constexpr uint32_t PKI_ATTR_KEY_STRIDE0 = 7;
constexpr uint32_t PKI_ATTR_K_DESCALE_STRIDE0 = 8;

// ------------------ Limits ------------------
constexpr uint32_t PKI_HEAD_DIM = 128;
constexpr uint32_t PKI_N2_FIXED = 1;
constexpr uint32_t PKI_N1_LIMIT = 64;
constexpr uint32_t PKI_POOL_SIZE_LIMIT = 128;
// PA block_size 上限: PA 寻址按运行时 blockSize 动态计算, 支持 16 对齐的任意 blockSize
constexpr uint32_t PKI_BLOCK_SIZE_LIMIT = 2048;
constexpr uint32_t PKI_BLOCK_SIZE_FACTOR = 16;
constexpr uint32_t PKI_TOPK_DEFAULT = 2048;
// quant_mode host-side values (match def.cpp attribute defaults, may be negative)
// Mapping to template values: tpl = host + 1 (see template_tiling_key.h PKI_TPL_QUANT_*)
constexpr int32_t PKI_QUANT_NONE = -1;
constexpr int32_t PKI_QUANT_FP8_PER_TOKEN = 0;
constexpr int32_t PKI_QUANT_MXFP8 = 1;
constexpr int32_t PKI_MASK_DEFAULT = 0;
constexpr int32_t PKI_MASK_CAUSAL = 3;
// mxFP8 scale 布局: 每 MX_SCALE_GROUP_SIZE 个 D 元素共享一个
// E8M0 scale, 尾维 [headDim/MX_SCALE_GROUP_ALIGN, MX_E8M0_SCALE_PACK_NUM] 打包
constexpr uint32_t PKI_MX_SCALE_GROUP_SIZE = 32;   // 每 32 个 D 元素一个 scale
constexpr uint32_t PKI_MX_E8M0_SCALE_PACK_NUM = 2; // E8M0 尾维打包数(2 个 e8m0 打包寻址)
constexpr uint32_t PKI_MX_SCALE_SHAPE_ALIGN = 64;  // headDim 对齐约束(当前 headDim 恒为 128)

// ------------------ Tiling Constants ------------------
// 以下 S1/S2/M 基本块常量为 arch35/host 侧口径; arch22 kernel 以自有硬编码覆盖, 不消费这些字段。
constexpr uint32_t PKI_S1_BASE_SIZE = 4;
constexpr uint32_t PKI_S1_BASE_SIZE_SMALL = 2;
constexpr uint32_t PKI_S2_BASE_SIZE = 128;
constexpr uint32_t PKI_M_BASE_SIZE = 256;
constexpr uint32_t PKI_M_BASE_SIZE_SMALL = 128;
constexpr uint32_t PKI_D_BASE_BLOCK = 128;
constexpr uint32_t PKI_TOPK_6K = 6144;
constexpr uint32_t PKI_GM_ALIGN_BYTES = 512;
constexpr uint32_t PKI_SCORE_T_SIZE = 4; // sizeof(uint32_t) sortable key

// Trunk lengths for TopK
constexpr uint32_t PKI_TRUNK_LEN_16K = 16384;
constexpr uint32_t PKI_TRUNK_LEN_12K = 12288;
constexpr uint32_t PKI_TRUNK_LEN_8K = 8192;
constexpr uint32_t PKI_TRUNK_LEN_4K = 4096;
constexpr uint32_t PKI_TRUNK_LEN_2K = 2048;
constexpr uint32_t PKI_TOPK_2K = 2048;
constexpr uint32_t PKI_TOPK_3K = 3072;
constexpr uint32_t PKI_TOPK_4K = 4096;
constexpr uint32_t PKI_TOPK_5K = 5120;

// ------------------ TilingData Definition ------------------
BEGIN_TILING_DATA_DEF(PoolKeyIndexerTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(int64_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, topk)
TILING_DATA_FIELD_DEF(uint32_t, poolSize)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, s1BaseSize)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSizeMax)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
TILING_DATA_FIELD_DEF(uint32_t, trunkLen)
TILING_DATA_FIELD_DEF(uint32_t, maskMode)
TILING_DATA_FIELD_DEF(uint32_t, quantMode)
TILING_DATA_FIELD_DEF(uint32_t, returnValue)
TILING_DATA_FIELD_DEF(uint32_t, layoutQ)
TILING_DATA_FIELD_DEF(uint32_t, layoutK)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0)
TILING_DATA_FIELD_DEF(uint32_t, keyDequantScaleStride0)
TILING_DATA_FIELD_DEF(uint64_t, wsOffScore)
TILING_DATA_FIELD_DEF(uint64_t, wsOffLdScore)
TILING_DATA_FIELD_DEF(uint64_t, wsOffLdIdx)
TILING_DATA_FIELD_DEF(float, qkScale) // 1/sqrt(headDim), score 缩放系数
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(PoolKeyIndexer, PoolKeyIndexerTilingData)

// ------------------ CompileInfo ------------------
// 图模式必需: 注册 TilingParse 后框架才生成 compile info JSON 供 FE 解析, 未注册图编译报错
struct PoolKeyIndexerCompileInfo {
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch = NpuArch::DAV_2201;
    uint64_t aicNum = 0;
    uint64_t aivNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
};

// ------------------ TilingInfo ------------------
struct PoolKeyIndexerTilingInfo {
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    NpuArch npuArch = NpuArch::DAV_2201;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 1;
    uint32_t gSize = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t headDim = PKI_HEAD_DIM;
    uint32_t topk = PKI_TOPK_DEFAULT;
    uint32_t poolSize = 16;
    uint32_t sparseCount = 0;
    int32_t maskMode = PKI_MASK_CAUSAL;
    int32_t quantMode = PKI_QUANT_NONE;
    bool returnValue = false;
    PkiDataLayout layoutQ = PkiDataLayout::BSND;
    PkiDataLayout layoutK = PkiDataLayout::BSND;
    bool pageAttention = false;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t keyStride0 = 0;
    uint32_t keyDequantScaleStride0 = 0;
    uint32_t usedCoreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint64_t ubSize = 0;
    ge::DataType queryDtype = ge::DT_FLOAT16;
    ge::DataType keyDtype = ge::DT_FLOAT16;
    ge::DataType weightsDtype = ge::DT_FLOAT16;
};

// ------------------ Tiling Class ------------------
class PoolKeyIndexerTiling {
public:
    explicit PoolKeyIndexerTiling(gert::TilingContext *context)
        : context_(context)
    {}
    ge::graphStatus DoTiling(PoolKeyIndexerTilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    PoolKeyIndexerTilingData tilingData_;
    std::vector<uint32_t> keyStridesVec_;
    // k_descale 运行时 stride(PA 0 轴非连续支持, 与 keyStridesVec_ 同源时机捕获)
    std::vector<uint32_t> keyDequantScaleStridesVec_;

    ge::graphStatus ParseAndCheckParams(PoolKeyIndexerTilingInfo &info);
    ge::graphStatus CalcTilingParams(PoolKeyIndexerTilingInfo &info);
    ge::graphStatus CalcWorkspaceSize(PoolKeyIndexerTilingInfo &info, uint64_t &workspaceSize);
    ge::graphStatus CheckKeyContiguous(const PoolKeyIndexerTilingInfo &info) const;
    uint32_t GetTrunkLen(uint32_t sparseCount);
    void SetTilingData(PoolKeyIndexerTilingInfo &info);
};

} // namespace optiling
#endif // POOL_KEY_INDEXER_TILING_H
