/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_base_tiling.cc
 * \brief
 */
#include <cinttypes>
#include "matmul_v3_base_tiling.h"
#include "../../op_kernel/mat_mul_v3_tiling_key.h"

#include "matmul_v3_l2_cache.h"
#include "util/math_util.h"
#include "mc2_log_compat.h"
#include "op_host/tiling_key.h"
#include "mc2_log.h"
#include "ops_legacy/op_tiling/op_cache_tiling.h"
#include "runtime_kb_api.h"
#include "matmul_v3_tuning.h"
#include "common/op_host/math_util.h"
#include "common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"

using namespace optiling::mc2_matmul_v3;
using Ops::Transformer::MathUtil;
using Ops::Transformer::OpTiling::GET_TILINGKEY;
namespace tuningtiling {
REGISTER_TUNING_TILING_CLASS(Mc2MatMulV3, Mc2MatMulV3TunnerTiling);
}

namespace {
constexpr double CORE_RATIO = 0.9;
// 基本块二维矩阵定义{baseMShape, baseNShape, baseKByte}
// 参数一代表基本块中M的大小, 参数二代表N的大小
// 参数三代表K的Byte大小（需注意baseKByte = KShape * KDtypeSize), 后续输出baseK时，须在矩阵基础上除以dtypeSize
const std::vector<std::vector<uint64_t>> CALC_M_BASIC = {{128, 256, 128}, {128, 128, 256}, {64, 512, 64}};
const std::vector<std::vector<uint64_t>> CALC_MN_BASIC = {{64, 64, 512}, {128, 128, 256}, {128, 256, 128}};
const std::vector<std::vector<uint64_t>> CALC_M_BASIC_L0C_256 = {{256, 256, 128}, {128, 128, 256}, {128, 512, 64}};
const std::vector<std::vector<uint64_t>> CALC_MN_BASIC_L0C_256 = {
    {16, 16, 2048}, {32, 32, 1024}, {64, 64, 512}, {128, 128, 256}, {256, 256, 128}};
const std::vector<uint64_t> CALC_ND_BASIC = {6144, 4096, 2048};
const std::vector<uint64_t> SUPPORT_ND2NZ_GM2L0 = {32, 64, 96, 128, 160, 192, 224, 256, 384};
constexpr uint64_t SMALL_SHAPE_LOWER_THRES = 128;
constexpr uint64_t L0C_THRES = 32768;
constexpr uint64_t NUMBER_TWO = 2;
constexpr uint64_t BLOCK_BYTE_SIZE = 32;
constexpr uint64_t N_ALIGNED = 16;
constexpr uint64_t NCALC_THRES = 16;
constexpr uint64_t MIN_TAIL = 512;
constexpr uint64_t MULTI_CORE_SINGLE_K = 384;
constexpr uint64_t CACHELINE = 512;
constexpr uint64_t NUMBER_SIXTEEN = 16;
constexpr uint64_t ND2NZ_THRES = 384;
constexpr uint64_t VNCHW_UP_THRES = 72368;
constexpr uint64_t VECTOR_D_BASE = 2048;
constexpr size_t HF32_ATTR_NUM = 4;
constexpr size_t HF32_ATTR_INDEX = 3;
constexpr uint64_t ONE_BATCH_DIM = 1;
constexpr uint64_t TWO_BATCH_DIM = 2;
constexpr uint64_t THREE_BATCH_DIM = 3;
constexpr uint64_t FOUR_BATCH_DIM = 4;
constexpr uint64_t LAST_DIM = 1;
constexpr uint64_t LAST_SECOND_DIM = 2;
constexpr uint64_t L2_SIZE_2 = 192UL * 1024UL * 1024UL;
constexpr uint64_t BLOCK = 128;
constexpr uint64_t N_THRESHOLD = 256;
constexpr uint64_t M_THRESHOLD = 256;
constexpr uint64_t MIX_BLOCK_NUM = 24;
constexpr uint64_t MIN_SPLITK_K = 1000;
constexpr uint64_t MAX_SPLITK_K = 4608;
constexpr uint64_t UB_SIZE = 196352;
constexpr uint64_t MIN_CORE_SPLITK = 8;
constexpr uint64_t MIN_SPLITK_MN64 = 6144;
constexpr uint64_t DELTAK_PER_CORE = 1024;
constexpr uint64_t L1_BIAS_SIZE = 1UL * 1024UL;
constexpr uint64_t STEP_NUM_3 = 3;
constexpr uint64_t SINGLE_CORE_M_24 = 256UL;
constexpr uint64_t SHAPE_LIMIT = 65535;
static const int64_t SINGLE_CORE_SPLIT_SMALL_K = 1536;
static const int64_t SINGLE_CORE_SPLIT_SMALL_MN = 384;
static const int64_t SINGLE_CORE_SPLIT_LARGE_MN = 49152; // 128 * 384

#define DO_CACL_TILING_ENABLE(func)                                                                                    \
    if (func) {                                                                                                        \
        break;                                                                                                         \
    }

inline uint64_t CalBaseSize(uint64_t cnt, uint64_t totalCoreNum, uint64_t size, uint64_t maxBase)
{
    uint64_t curCnt = 0UL;
    if (cnt == 0UL) {
        curCnt = 1UL;
    } else {
        curCnt = totalCoreNum / cnt;
        curCnt = std::max(curCnt, 1UL);
    }
    uint64_t base = ops::CeilAlign(std::max(size / curCnt, 1UL), BASIC_ALIGN_16);
    base = std::min(base, maxBase);
    return base;
}
} // namespace

namespace optiling {
namespace mc2_matmul_v3 {
using ge::float32_t;
bool Mc2MatmulV3BaseTiling::CheckAoeTilingEnable(uint32_t aoeTilingEnable, const std::string &opName)
{
    if (aoeTilingEnable <= 1U) {
        return true;
    }
    uint32_t tilingSplitK = aoeTilingEnable % 10U; // aoe 的tilingEnable的个位
    uint32_t maxTilingEnable = static_cast<uint32_t>(Mc2TilingEnableSplitCore::MULTI_CORE_SPLIT_K);
    if (context_->GetDeterministic() == 1) {
        maxTilingEnable = static_cast<uint32_t>(Mc2TilingEnableSplitCore::DETERMINISTIC_SPLIT_K);
    }
    bool isInvalidTilingEnable = tilingSplitK == 1U || tilingSplitK > maxTilingEnable;
    if (isInvalidTilingEnable) {
        OP_LOGW(opName, "Invalid tilingEnable which the bit of split k %d is undefined", tilingSplitK);
        return false;
    }
    tilingEnable_.tilingEnableSplitCore = static_cast<Mc2TilingEnableSplitCore>(tilingSplitK);
    uint32_t tilingFullLoad = (aoeTilingEnable / 10U) % 10U; // aoe 的tilingEnable的十位
    if (tilingFullLoad > static_cast<uint32_t>(Mc2TilingEnableFullLoad::BL1_FULL_LOAD)) {
        OP_LOGW(opName, "Invalid tilingEnable which the bit of full load %d is undefined", tilingFullLoad);
        return false;
    }
    tilingEnable_.tilingEnableFullLoad = static_cast<Mc2TilingEnableFullLoad>(tilingFullLoad);
    uint32_t tilingFixOpti = (aoeTilingEnable / 1000U) % 10U; // aoe 的tilingEnable的千位
    if (tilingFixOpti > static_cast<uint32_t>(Mc2TilingEnableFixOpti::VEC_NZ2ND_UNALIGNOUT)) {
        OP_LOGW(opName, "Invalid tilingEnable which the bit of fix-Opti %d is undefined", tilingFixOpti);
        return false;
    }
    tilingEnable_.tilingEnableFixOpti = static_cast<Mc2TilingEnableFixOpti>(tilingFixOpti);
    return true;
}

ge::graphStatus Mc2MatmulV3BaseTiling::GetPlatformInfo() // 检查平台信息是否支持
{
    if (!compileInfoInit_) {
        auto compileInfoPtr = reinterpret_cast<const Mc2MatmulV3CompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
        compileInfo_ = *compileInfoPtr;
    }
    if (compileInfo_.aicNum == 0) {
        OP_LOGE(context_->GetNodeName(), "compileInfo.aicNum is zero.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void Mc2MatmulV3BaseTiling::InitCompileInfo() // 检查输入属性是否支持
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGW(context_->GetNodeName(), "platformInfo is null");
        return;
    }
    Mc2MatmulV3CompileInfo compileInfo;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", compileInfo.socVersionStr);
    std::string val;
    std::string dataMoveL12Bt;
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);
    compileInfo.supportL0c2out = !val.empty();
    compileInfo.supportL12BtBf16 = (dataMoveL12Bt.find("bf16") != string::npos);
    compileInfo.aicNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAic());
    compileInfo.aivNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
    compileInfo.socVersion = ascendcPlatform.GetSocVersion();
    compileInfo.btSize = compileInfo.supportL0c2out ? 1024UL : 0UL;                // 1024 is btSize
    compileInfo.btSize = compileInfo.supportL12BtBf16 ? 4096 : compileInfo.btSize; // 4096 is btSize
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo.l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo.l2Size);

    TilingPrepareForOpCache(context_);
    OP_LOGI(context_->GetNodeName(),
            "parse compile info soc:%d, l1Size:%lu, l2Size:%lu, coreNum:%lu, supportL0c2out:%d, supportL12BtBf16:%d",
            static_cast<int32_t>(compileInfo.socVersion), compileInfo.l1Size, compileInfo.l2Size, compileInfo.aicNum,
            compileInfo.supportL0c2out, compileInfo.supportL12BtBf16);
    compileInfoInit_ = true;
    compileInfo_ = compileInfo;
}


ge::graphStatus Mc2MatmulV3BaseTiling::GetShapeAttrsInfo() // 检查输入属性是否支持
{
    args_.opName = context_->GetNodeName();
    OP_TILING_CHECK(args_.opName == nullptr, OP_LOGE("matmul", "get op name invalid."), return ge::GRAPH_FAILED);
    OP_LOGI(args_.opName, "TilingContext: %s", Ops::Transformer::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK((CheckArgs() != ge::GRAPH_SUCCESS) || (GetArgs() != ge::GRAPH_SUCCESS),
                    OP_LOGE(args_.opName, "invalid context."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MatmulV3BaseTiling::CheckArgs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(idx));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(idx));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    // 区分Matmul和GemmV2，只有3个输入的为Matmul，并设置bias标志
    if (std::string(context_->GetNodeType()) != "TransposeBatchMatMul" && context_->GetInputDesc(idx) != nullptr &&
        context_->GetInputDesc(idx + 1) == nullptr) {
        args_.hasBias = true;
    }

    if (attrs->GetAttrNum() >= HF32_ATTR_NUM) {
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<int64_t>(HF32_ATTR_INDEX - 1));
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(HF32_ATTR_INDEX));
    }
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    return ge::GRAPH_SUCCESS;
}

static inline void GetFormat(const gert::TilingContext &context, Mc2MatmulV3Args &args)
{
    ge::Format formatA = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(0)->GetStorageFormat()));
    ge::Format formatB = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(1)->GetStorageFormat()));
    ge::Format formatOut = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetOutputDesc(0)->GetStorageFormat()));
    args.aFormat = (formatA != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatA;
    args.bFormat = (formatB != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatB;
    args.outFormat = (formatOut != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatOut;
}

static inline void GetDtype(const gert::TilingContext &context, Mc2MatmulV3Args &args)
{
    // op_impl_mode_enum: 0x1: default 0x2: high_performance 0x4: high_precision 0x8: super_performance
    // 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
    if (context.GetAttrs()->GetAttrNum() >= HF32_ATTR_NUM) {
        args.isHf32 = *context.GetAttrs()->GetAttrPointer<bool>(HF32_ATTR_INDEX) ? 1 : 0;
    }
    OP_LOGD(args.opName, "Hf32 flag is: %d", args.isHf32);

    args.aType = context.GetInputDesc(0)->GetDataType();
    args.bType = context.GetInputDesc(1)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    if (args.hasBias) {
        args.biasType = context.GetInputDesc(BIAS_IDX)->GetDataType();
    }
}

static ge::graphStatus GetInputDims(const gert::Shape &storageShape, ge::Format format, int64_t (&dims)[TWO_BATCH_DIM])
{
    const size_t dimNum = storageShape.GetDimNum();
    if (format == ge::FORMAT_ND) {
        if (dimNum < TWO_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
        dims[0] = storageShape[dimNum - TWO_BATCH_DIM];
        dims[1] = storageShape[dimNum - ONE_BATCH_DIM];
    } else {
        if (dimNum < FOUR_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
        dims[0] = storageShape[dimNum - THREE_BATCH_DIM] * storageShape[dimNum - TWO_BATCH_DIM];
        dims[1] = storageShape[dimNum - FOUR_BATCH_DIM] * storageShape[dimNum - ONE_BATCH_DIM];
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus OpSpecificCheck(const gert::TilingContext &context, const Mc2MatmulV3Args &args)
{
    const bool isMatMulV3 = (strcmp(context.GetNodeType(), "Mc2MatMulV3") == 0);
    const bool isBatchMatMulV3 = (strcmp(context.GetNodeType(), "Mc2BatchMatMulV3") == 0);
    if (!isBatchMatMulV3 && !isMatMulV3) {
        // apply no additional checks for ops other than MMV3, BMMV3, for now
        return ge::GRAPH_SUCCESS;
    }

    // check input dim num equals to 2
    if (isMatMulV3) {
        auto isMatrix = [](const gert::Shape &oriShape) { return oriShape.GetDimNum() == TWO_BATCH_DIM; };
        if (!isMatrix(context.GetInputShape(0)->GetOriginShape()) ||
            !isMatrix(context.GetInputShape(1)->GetOriginShape())) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                args.opName, "x1 and x2",
                std::to_string(context.GetInputShape(0)->GetOriginShape().GetDimNum()).c_str(),
                "The shape dim of x1 and x2 must be 2D.");
            return ge::GRAPH_FAILED;
        }
    }

    // check bias
    if (args.hasBias) {
        const gert::Shape &biasShape = context.GetInputShape(BIAS_IDX)->GetOriginShape();
        const int64_t biasValue = biasShape[biasShape.GetDimNum() - 1];
        if (args.nOriValue != static_cast<uint64_t>(biasValue)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(args.opName, "bias", std::to_string(biasValue).c_str(),
                                                  "The shape dim of bias must be equal to n.");
            return ge::GRAPH_FAILED;
        }
    }

    // dtype check
    std::vector<ge::DataType> dtype = {args.aType, args.bType, args.cType};
    if (args.hasBias) {
        dtype.push_back(args.biasType);
    }
    auto isValidDtype = [&args](const std::vector<ge::DataType> &dtypeList) -> ge::graphStatus {
        const std::vector<std::vector<ge::DataType>> dtypeSuportList = {
            // x1,              x2,             y,              bias
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT},
            {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
            {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT}};
        for (auto &supported : dtypeSuportList) {
            if (std::equal(dtypeList.begin(), dtypeList.end(), supported.begin())) {
                return ge::GRAPH_SUCCESS;
            }
        }
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(args.opName, "x1, x2 and output", Ops::Base::ToString(args.aType).c_str(),
                                              "The dtypes of x1, x2 and output must be within the supported range.");
        return ge::GRAPH_FAILED;
    };
    return isValidDtype(dtype);
}

static ge::graphStatus GetShape(const gert::TilingContext &context, Mc2MatmulV3Args &args)
{
    // get transpose
    args.isATrans = *context.GetAttrs()->GetAttrPointer<bool>(0);
    args.isBTrans = *context.GetAttrs()->GetAttrPointer<bool>(1);

    // get (m, k, n)
    int64_t mkDims[TWO_BATCH_DIM];
    int64_t knDims[TWO_BATCH_DIM];
    if ((GetInputDims(context.GetInputShape(0)->GetStorageShape(), args.aFormat, mkDims) != ge::GRAPH_SUCCESS) ||
        (GetInputDims(context.GetInputShape(1)->GetStorageShape(), args.bFormat, knDims) != ge::GRAPH_SUCCESS)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            args.opName, "x1 and x2", std::to_string(context.GetInputShape(0)->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of x1 and x2 is too small for given format.");
        return ge::GRAPH_FAILED;
    }
    int64_t kDimsIndex = args.isATrans ? 0 : 1;
    int64_t k = mkDims[kDimsIndex];
    int64_t kRight = 0;
    if (args.bFormat == ge::FORMAT_FRACTAL_NZ) {
        auto bOriginShape = context.GetInputShape(1)->GetOriginShape();
        int64_t bDimNum = bOriginShape.GetDimNum();
        kRight = bOriginShape.GetDim(args.isBTrans ? bDimNum - ONE_BATCH_DIM : bDimNum - TWO_BATCH_DIM);
    } else {
        int64_t dimIndex = args.isBTrans ? 1 : 0;
        kRight = knDims[dimIndex];
    }
    if (k != kRight) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(args.opName, "x1 and x2",
                                               (std::to_string(k) + " and " + std::to_string(kRight)).c_str(),
                                               "The shape of x1 and x2 must be the same at the k dimension.");
        return ge::GRAPH_FAILED;
    }
    int64_t mDimsIndex = args.isATrans ? 1 : 0;
    int64_t m = mkDims[mDimsIndex];
    int64_t nDimsIndex = args.isBTrans ? 0 : 1;
    int64_t n = knDims[nDimsIndex];
    auto isValidDimValue = [](int64_t dim) -> bool { return (dim > 0) && (dim <= INT32_MAX); };
    if (!isValidDimValue(m) || !isValidDimValue(k) || !isValidDimValue(n)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            args.opName, "m, k and n",
            (std::to_string(m) + ", " + std::to_string(k) + " and " + std::to_string(n)).c_str(),
            "The value of m, k and n must be positive and not exceed INT32_MAX.");
        return ge::GRAPH_FAILED;
    }
    args.mValue = static_cast<uint64_t>(m);
    args.kValue = static_cast<uint64_t>(k);
    args.nValue = static_cast<uint64_t>(n);

    // get origin (m, n)
    const gert::Shape &cShape = context.GetOutputShape(0)->GetOriginShape();
    const size_t cDimNum = cShape.GetDimNum();
    if (cDimNum < TWO_BATCH_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(args.opName, "output", (std::to_string(cDimNum) + "D").c_str(),
                                                 "The shape dim of output must be at least 2D.");
        return ge::GRAPH_FAILED;
    }
    args.nOriValue = cShape[cDimNum - LAST_DIM];
    args.mOriValue = cShape[cDimNum - LAST_SECOND_DIM];

    if (args.bFormat == ge::FORMAT_FRACTAL_NZ) {
        args.nValue = args.nOriValue;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MatmulV3BaseTiling::GetArgs()
{
    GetFormat(*context_, args_);
    GetDtype(*context_, args_);
    if (GetShape(*context_, args_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return OpSpecificCheck(*context_, args_);
}

inline bool GetNd2nzA(const Mc2MatmulV3Args &args_, const Mc2MatmulV3CompileInfo &compileInfo_)
{
    constexpr uint64_t nMataThread = 16384;
    constexpr uint64_t mMataThread = 4096;
    constexpr uint64_t kMataThread = 6656;
    constexpr uint64_t compileNum = 24;
    return !args_.isBTrans && args_.nValue % nMataThread == 0 &&
           (args_.mValue > mMataThread || (args_.mValue == mMataThread && compileInfo_.aicNum >= compileNum)) &&
           args_.kValue >= kMataThread && args_.bFormat == ge::FORMAT_ND &&
           (args_.aType == ge::DT_FLOAT16 || args_.aType == ge::DT_BF16);
}

inline bool GetNd2nzB(const Mc2MatmulV3Args &args_, const Mc2MatmulV3CompileInfo &compileInfo_)
{
    constexpr uint64_t kMataCond = 16384;
    constexpr uint64_t nMataCond = 7168;
    constexpr uint64_t mMataCondMax = 4480;
    constexpr uint64_t mMataCondMin = 4096;
    constexpr uint64_t compileNum = 24;
    return !args_.isATrans && args_.isBTrans && args_.kValue == kMataCond && args_.mValue >= mMataCondMin &&
           args_.mValue <= mMataCondMax && args_.nValue >= nMataCond && args_.bFormat == ge::FORMAT_ND &&
           (args_.aType == ge::DT_FLOAT16 || args_.aType == ge::DT_BF16) && compileInfo_.aicNum >= compileNum;
}

ge::graphStatus Mc2MatmulV3BaseTiling::GetMoreArgs()
{
    aDtypeSize_ = GetSizeByDataType(args_.aType);
    bDtypeSize_ = GetSizeByDataType(args_.bType);
    cDtypeSize_ = GetSizeByDataType(args_.cType);
    m256Align_ = Is256BAlign(args_.mValue, aDtypeSize_);  // A矩阵 m轴256B对齐
    kA256Align_ = Is256BAlign(args_.kValue, aDtypeSize_); // A矩阵 k轴256B对齐
    kB256Align_ = Is256BAlign(args_.kValue, bDtypeSize_); // B矩阵 k轴256B对齐
    n256Align_ = Is256BAlign(args_.nValue, bDtypeSize_);  // B矩阵 n轴256B对齐
    bool innerAlignA = kA256Align_;
    bool innerAlignB = n256Align_;
    uint64_t innerSizeA = args_.kValue;
    uint64_t innerSizeB = args_.nValue;
    uint64_t outerSizeA = args_.mValue;
    uint64_t outerSizeB = args_.kValue;
    trans_ = Mc2MatmulV3Trans::NO_TRANS;
    if (args_.isATrans) {
        trans_ = Mc2MatmulV3Trans::A_TRANS;
        innerAlignA = m256Align_;
        innerSizeA = args_.mValue;
        outerSizeA = args_.kValue;
    }
    if (args_.isBTrans) {
        trans_ = Mc2MatmulV3Trans::B_TRANS;
        innerAlignB = kB256Align_;
        innerSizeB = args_.kValue;
        outerSizeB = args_.nValue;
    }
    if (args_.isATrans && args_.isBTrans) {
        trans_ = Mc2MatmulV3Trans::AB_TRANS;
    }
    calcMBasic_ = compileInfo_.l0CSize == L0C_SIZE_256_KB ? CALC_MN_BASIC_L0C_256 : CALC_M_BASIC;
    calcMNBasic_ = compileInfo_.l0CSize == L0C_SIZE_256_KB ? CALC_MN_BASIC_L0C_256 : CALC_MN_BASIC;
    l2TileLength_ = L2_TILE_LENGTH;
    OP_TILING_CHECK(!compileInfo_.supportL0c2out, tilingSelect_ = Mc2TilingCalcSelect::BASE, return ge::GRAPH_SUCCESS);

    // check the size is equaled to {32, 64, 96, 128, 160, 192, 224, 256, 384}.
    bool supportNd2NzOnTheWayA = IsOnTheWay(args_.aFormat, innerSizeA, aDtypeSize_, SUPPORT_ND2NZ_GM2L0);
    bool supportNd2NzOnTheWayB = IsOnTheWay(args_.bFormat, innerSizeB, bDtypeSize_, SUPPORT_ND2NZ_GM2L0);
    args_.nd2nzA = ((!innerAlignA || innerSizeA > ND2NZ_ON_THE_FLY_LIMIT) && (args_.aFormat == ge::FORMAT_ND) &&
                    (!supportNd2NzOnTheWayA) &&
                    !(args_.aType == ge::DT_FLOAT && !args_.isHf32 && innerSizeA < ND2NZ_ON_THE_FLY_LIMIT) &&
                    !(args_.aType == ge::DT_FLOAT && args_.isHf32 && innerSizeA * aDtypeSize_ < CACHELINE));
    args_.nd2nzB = ((!innerAlignB || innerSizeB > ND2NZ_ON_THE_FLY_LIMIT) && (args_.bFormat == ge::FORMAT_ND) &&
                    (!supportNd2NzOnTheWayB) &&
                    !(args_.bType == ge::DT_FLOAT && !args_.isHf32 && innerSizeB < ND2NZ_ON_THE_FLY_LIMIT) &&
                    !(args_.bType == ge::DT_FLOAT && args_.isHf32 && innerSizeB * bDtypeSize_ < CACHELINE));

    OP_LOGD(args_.opName, "After judging nd2nz tiling condition, matrix A need normal mode nd2nz = %u, matrix B = %u.",
            static_cast<uint32_t>(args_.nd2nzA), static_cast<uint32_t>(args_.nd2nzB));
    args_.nd2nzA =
        args_.nd2nzA || NeedNd2NzVnchw(outerSizeA, innerSizeA, supportNd2NzOnTheWayA, aDtypeSize_, args_.aFormat);
    args_.nd2nzB =
        args_.nd2nzB || NeedNd2NzVnchw(outerSizeB, innerSizeB, supportNd2NzOnTheWayB, bDtypeSize_, args_.bFormat);
    // (k, n) n为16384的倍数时，mata冲突严重，m越大，右矩阵重复载入越多，冲突影响越大，将右矩阵先做nd2nz
    // 限制为fp16、bf16场景
    bool mataConflictFlag = GetNd2nzA(args_, compileInfo_);
    //  B 矩阵转置场景
    bool mataConflictFlag2 = GetNd2nzB(args_, compileInfo_);
    args_.nd2nzB = args_.nd2nzB || mataConflictFlag || mataConflictFlag2;
    OP_LOGI(args_.opName, "After judging nd2nz tiling condition, matrix A need vnchw mode nd2nz = %u, matrix B = %u.",
            static_cast<uint32_t>(args_.nd2nzA), static_cast<uint32_t>(args_.nd2nzB));
    if (args_.nd2nzA && NeedNd2NzVnchw(outerSizeA, innerSizeA, supportNd2NzOnTheWayA, aDtypeSize_, args_.aFormat)) {
        args_.unAlignProcessType = 1;
    } else {
        args_.unAlignProcessType = 0;
    }
    return ge::GRAPH_SUCCESS;
}

inline void ResetBase(Mc2MatmulV3RunInfo &runInfo, const uint64_t l0CSize, const uint64_t dTypeSize = 2)
{
    runInfo.baseM = (l0CSize == L0C_SIZE_256_KB) ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128;
    runInfo.baseN = BASIC_BLOCK_SIZE_256; // 256 is better base
    runInfo.baseK = BASIC_BLOCK_K_128_BYTE / dTypeSize;
    runInfo.stepM = BASE_STEP;
    runInfo.stepN = BASE_STEP;
    runInfo.iterateOrder = ITER_COL_FIRST;
    runInfo.dbL0c = DB_OFF_SIZE;
}

bool Mc2MatmulV3BaseTiling::IsOnTheWay(ge::Format matFormat, uint64_t innerSize, uint64_t dtypeSize,
                                       const std::vector<uint64_t> supportNd2nzList) const
{
    if (matFormat == ge::FORMAT_ND) {
        return (std::find(supportNd2nzList.begin(), supportNd2nzList.end(), innerSize * dtypeSize) !=
                supportNd2nzList.end());
    }
    return false;
}

bool Mc2MatmulV3BaseTiling::NeedNd2NzVnchw(uint64_t outerSize, uint64_t innerSize, bool supportNd2NzOnTheWay,
                                           uint64_t dtypeSize, ge::Format matFormat) const
{
    if (dtypeSize == 0UL) {
        return false;
    }
    if (matFormat == ge::FORMAT_ND) {
        bool innerAlign = Is256BAlign(innerSize, dtypeSize);
        // 外轴<=8192 会导致数据量增大减慢搬运, 192B为最大奇数内轴长度, 384B为最大偶数内轴长度
        bool willFitVnchwCond =
            outerSize > 8192UL && (innerSize > 1UL) &&
            (innerSize * dtypeSize <= 192UL || (innerSize * dtypeSize <= 384UL && innerSize % 2UL == 0) ||
             (innerSize * dtypeSize <= CACHELINE && innerSize % 4UL == 0UL));
        bool willInnerSizeEqualC0 = (innerSize == (32UL / dtypeSize));
        return (willFitVnchwCond && !innerAlign && !supportNd2NzOnTheWay && !willInnerSizeEqualC0);
    }
    return false;
}

ge::graphStatus Mc2MatmulV3BaseTiling::SelectNZTiling()
{
    if ((compileInfo_.supportL12BtBf16 || compileInfo_.supportL0c2out) && args_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        if (tilingSelect_ == Mc2TilingCalcSelect::ALL) {
            tilingSelect_ = Mc2TilingCalcSelect::BASE;
        } else if (tilingSelect_ != Mc2TilingCalcSelect::BASE) {
            OP_LOGE(args_.opName, "weightNz don't support this tiling select yet.");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MatmulV3BaseTiling::DoOpTiling()
{
    OP_TILING_CHECK(GetMoreArgs() != ge::GRAPH_SUCCESS, OP_LOGE(args_.opName, "invalid context."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckDimsAligned310P() != ge::GRAPH_SUCCESS, OP_LOGE(args_.opName, "invalid context."),
                    return ge::GRAPH_FAILED);

    if (InitTilingData() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetRunInfo();
    SetParamsV310();
    if (GetTilingFromRepo()) {
        DoNd2NzVectorTiling();
        SetNd2NzInfo();
        if (args_.hasBias) {
            runInfo_.baseN = std::min(256UL, runInfo_.baseN); // 有bias时 baseN 小于256
        }
        // check multi_core_split k
        if (tilingEnable_.tilingEnableSplitCore == Mc2TilingEnableSplitCore::MULTI_CORE_SPLIT_K) {
            bool isSingleRound =
                MathUtil::CeilDivision(args_.mValue, static_cast<uint64_t>(tilingData_.matmulTiling.singleCoreM)) *
                    MathUtil::CeilDivision(args_.nValue, static_cast<uint64_t>(tilingData_.matmulTiling.singleCoreN)) *
                    MathUtil::CeilDivision(args_.kValue, static_cast<uint64_t>(tilingData_.matmulTiling.singleCoreK)) <=
                compileInfo_.aicNum;
            OP_TILING_CHECK(
                args_.isATrans || !args_.isBTrans || !isSingleRound || args_.aType != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(args_.opName, "MULTI_CORE_SPLIT_K", "unsupported",
                                                      "The value of MULTI_CORE_SPLIT_K only supports fp32 to fp32, "
                                                      "transA=false and transB=true, and multi round is not permitted"),
                return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(SelectNZTiling() != ge::GRAPH_SUCCESS, OP_LOGE(args_.opName, "invalid tiling select."),
                    return ge::GRAPH_FAILED);
    DoBasicTiling();
    OptimizeBasicKernelStepK();
    SetNd2NzInfo();
    return ge::GRAPH_SUCCESS;
}

bool Mc2MatmulV3BaseTiling::IsPowerOfTwo(uint64_t x) const
{
    return x > 0UL && (x & (x - 1UL)) == 0UL;
}

void Mc2MatmulV3BaseTiling::OptimizeLoadBalanceBasicKernel()
{
    OP_LOGI(args_.opName, "Optimize LoadBalance for BasicKernel");
    // 判决门限
    // 1. 需要tiling_key==10000000000000000001UL或10000000000000000000UL，表示时BasicKernel场景
    // 2. baseM==128或256，baseN==256或128, baseK==128/dtypesize，刚好将L0 cache的利用最大化
    // 3. 要求m,n方向分合小于4轮或调小m可以不增加轮次
    // 4. 暂时适用于24核情况，aicNum == 24，其他核数目未验证
    // 5. 暂适用m为外轴
    constexpr uint64_t baseMNCheck = 32768; // 128 * 256;
    uint64_t baseKCheck = BASIC_BLOCK_SIZE_128 / aDtypeSize_;
    constexpr uint64_t aicNumCheck = 24;
    bool baseMNKFlag = runInfo_.baseM * runInfo_.baseN == baseMNCheck && runInfo_.baseK == baseKCheck;
    Mc2MixNd2NzType mixNd2nzType = GetMixNd2nzType();
    bool tilingKeyCheckFlag =
        tilingEnable_.tilingEnableFullLoad == Mc2TilingEnableFullLoad::BASE &&
        tilingEnable_.tilingEnableSplitCore == Mc2TilingEnableSplitCore::BASE &&
        tilingEnable_.tilingEnableFixOpti == Mc2TilingEnableFixOpti::BASE &&
        (mixNd2nzType == Mc2MixNd2NzType::NO_ND2NZ || mixNd2nzType == Mc2MixNd2NzType::V_HEAD_ND2NZ);
    bool aicNumCheckFlag = compileInfo_.aicNum == aicNumCheck;
    uint64_t mCore = ops::CeilDiv(args_.mValue, runInfo_.baseM);
    uint64_t nCore = ops::CeilDiv(args_.nValue, runInfo_.baseN);
    // 4为经验值，轮次更多得情况下均衡收益变小
    bool smallroundsCheckFlag = mCore * nCore < 4 * compileInfo_.aicNum;
    bool commonFlag = tilingKeyCheckFlag && aicNumCheckFlag && baseMNKFlag && !args_.isATrans;
    if (commonFlag && smallroundsCheckFlag) {
        uint64_t best_tasks = ops::CeilAlign(mCore * nCore, compileInfo_.aicNum);
        // baseN为输出内轴，所以只调整baseM
        mCore = ops::FloorDiv(best_tasks, nCore);
        runInfo_.baseM = ops::CeilAlign(ops::CeilDiv(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
        runInfo_.singleCoreM = runInfo_.baseM;
    }
    // 经验值，m再大调整意义不大
    bool smallMCheckFlag = args_.mValue <= 672 && args_.mValue > 128;
    if (commonFlag && !smallroundsCheckFlag && smallMCheckFlag) {
        // 调小m不增加mCore
        while (runInfo_.baseM > BASIC_BLOCK_SIZE_16 &&
               ops::CeilDiv(args_.mValue, runInfo_.baseM - BASIC_BLOCK_SIZE_16) == mCore) {
            runInfo_.baseM -= BASIC_BLOCK_SIZE_16;
        }
        runInfo_.singleCoreM = runInfo_.baseM;
    }
}

void Mc2MatmulV3BaseTiling::OptimizeBasicKernelStepK()
{
    // 判决门限
    // 1. 需要tiling_key==10000000000000000001UL，表示时BasicKernel场景
    // 2. baseM==128或256，baseN==256或128,baseK==64，刚好将L0 cache的利用最大化
    // 3. 要求m,n是256的倍数且大于等于768, k是256的倍数但不能是2的幂次方 或者m=[10368, 18000] && n,k=[1280, 5120]
    //    或者k=[13788, 19304] && m,n=[1280, 5120]
    //    或者m=[320000, 380000] && n=[960, 3500] && k=[1280, 3328]
    // 4. M*N大于128*256*aicNum
    // 5. 暂时适用于24核情况，aicNum == 24，其他核数目未验证
    // 6. M,N,K不能为mata值16384或32768
    // 7. 要求输入输出数据类型为Float16或BF16
    constexpr uint64_t baseMNCheck = 32768; // 128 * 256;
    constexpr uint64_t baseKCheck = 64;
    constexpr uint64_t alignCheck = 256;
    constexpr uint64_t MNCheck = 768;
    constexpr uint64_t mataCheck = 16384;
    constexpr uint64_t aicNumCheck = 24;
    bool baseMNKFlag = runInfo_.baseM * runInfo_.baseN == baseMNCheck && runInfo_.baseK == baseKCheck;
    bool alignFlag = args_.mValue % alignCheck == 0 && args_.nValue % alignCheck == 0 &&
                     args_.kValue % alignCheck == 0 && args_.mValue >= MNCheck && args_.nValue >= MNCheck &&
                     !IsPowerOfTwo(args_.kValue);
    bool middleMShapeFlag = args_.mValue >= 10368 && args_.mValue <= 18000 && args_.nValue >= 1280 &&
                            args_.nValue <= 5120 && args_.kValue >= 1280 &&
                            args_.kValue <= 5120; // "middle-m" dimension is [10368, 18000]
    bool middleKShapeFlag = args_.mValue >= 1280 && args_.mValue <= 5120 && args_.nValue >= 1280 &&
                            args_.nValue <= 5120 && args_.kValue >= 13788 &&
                            args_.kValue <= 19304; // "middle-k" dimension is [13788, 19304]
    bool BigMShapeFlag = args_.mValue >= 320000 && args_.mValue <= 380000 && args_.nValue >= 960 &&
                         args_.nValue <= 3500 && args_.kValue >= 1280 &&
                         args_.kValue <= 3328; // "big-m" dimension is [320000, 380000]
    bool globalMNFlag = args_.mValue * args_.nValue > baseMNCheck * compileInfo_.aicNum;
    bool aicNumCheckFlag = compileInfo_.aicNum == aicNumCheck;
    bool notMataFlag = args_.mValue % mataCheck != 0 && args_.nValue % mataCheck != 0;
    bool dtypeFlag = (args_.aType == ge::DT_FLOAT16 || args_.aType == ge::DT_BF16) &&
                     (args_.bType == ge::DT_FLOAT16 || args_.bType == ge::DT_BF16) &&
                     (args_.cType == ge::DT_FLOAT16 || args_.cType == ge::DT_BF16);
    uint64_t disableMixNd2nz = static_cast<uint64_t>(GetMixNd2nzType());
    tilingKey_ = GET_TPL_TILING_KEY(static_cast<uint64_t>(tilingEnable_.tilingEnableFullLoad),
                                    static_cast<uint64_t>(tilingEnable_.tilingEnableSplitCore),
                                    static_cast<uint64_t>(tilingEnable_.tilingEnableFixOpti), disableMixNd2nz);
    if (tilingEnable_.tilingEnableFullLoad == Mc2TilingEnableFullLoad::BASE &&
        tilingEnable_.tilingEnableSplitCore == Mc2TilingEnableSplitCore::BASE &&
        tilingEnable_.tilingEnableFixOpti == Mc2TilingEnableFixOpti::BASE &&
        GetMixNd2nzType() == Mc2MixNd2NzType::NO_ND2NZ && baseMNKFlag && notMataFlag && dtypeFlag &&
        (alignFlag || middleMShapeFlag || middleKShapeFlag || BigMShapeFlag) && globalMNFlag && aicNumCheckFlag) {
        OP_LOGI(args_.opName, "Fit optimization condition, tilingKey_: %lu", tilingKey_);
        OP_LOGI(args_.opName, "Fit optimization condition, M N K: %lu %lu %lu", args_.mValue, args_.nValue,
                args_.kValue);
        OP_LOGI(args_.opName, "Fit optimization condition, baseM baseN baseK: %lu %lu %lu", runInfo_.baseM,
                runInfo_.baseN, runInfo_.baseK);
        constexpr uint64_t oriStepKValue = 8;
        constexpr uint64_t optStepKValue = 4;
        if (runInfo_.stepKa == oriStepKValue) {
            runInfo_.depthA1 = runInfo_.depthA1 / runInfo_.stepKa * optStepKValue;
            runInfo_.stepKa = optStepKValue;
        }
        if (runInfo_.stepKb == oriStepKValue) {
            runInfo_.depthB1 = runInfo_.depthB1 / runInfo_.stepKb * optStepKValue;
            runInfo_.stepKb = optStepKValue;
        }
        OP_LOGI(args_.opName, "stepKa: %lu, stepKb: %lu, depthA1: %lu, depthB1: %lu", runInfo_.stepKa, runInfo_.stepKb,
                runInfo_.depthA1, runInfo_.depthB1);
    } else {
        OP_LOGI(args_.opName, "Doesn't fit optimization condition, tilingKey_: %lu", tilingKey_);
        OP_LOGI(args_.opName, "Doesn't fit optimization condition, M N K: %lu %lu %lu", args_.mValue, args_.nValue,
                args_.kValue);
        OP_LOGI(args_.opName, "Doesn't fit optimization condition, baseM baseN baseK: %lu %lu %lu", runInfo_.baseM,
                runInfo_.baseN, runInfo_.baseK);
        OP_LOGI(args_.opName, "Doesn't fit optimization condition, aicNum: %lu", compileInfo_.aicNum);
    }
}

ge::graphStatus Mc2MatmulV3BaseTiling::CheckDimsAligned310P()
{
    if (compileInfo_.supportL12BtBf16 || compileInfo_.supportL0c2out) {
        return ge::GRAPH_SUCCESS;
    }
    if (args_.outFormat == ge::FORMAT_ND) {
        if (args_.nOriValue * bDtypeSize_ % BLOCK_BYTE_SIZE != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(args_.opName, "n", std::to_string(args_.nOriValue).c_str(),
                                                  "The value of n must be 32-byte aligned.");
            return ge::GRAPH_FAILED;
        }
    }
    if (args_.aFormat == ge::FORMAT_ND) {
        if (args_.mOriValue * aDtypeSize_ % BLOCK_BYTE_SIZE != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(args_.opName, "m", std::to_string(args_.mOriValue).c_str(),
                                                  "The value of m must be 32-byte aligned.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MatmulV3BaseTiling::InitTilingData()
{
    auto aFormat = args_.aFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto bFormat = args_.bFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto cFormat = args_.outFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    mm_.SetAType(matmul_tiling::TPosition::GM, aFormat, DTYPE_MAP.at(args_.aType), args_.isATrans);
    mm_.SetBType(matmul_tiling::TPosition::GM, bFormat, DTYPE_MAP.at(args_.bType), args_.isBTrans);
    mm_.SetCType(matmul_tiling::TPosition::GM, cFormat, DTYPE_MAP.at(args_.cType));
    mm_.SetDim(compileInfo_.aicNum);
    mm_.SetShape(args_.mValue, args_.nValue, args_.kValue);
    mm_.SetOrgShape(args_.mValue, args_.nValue, args_.kValue);
    if (args_.hasBias) {
        mm_.SetBias(true);
        mm_.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, DTYPE_MAP.at(args_.biasType));
    }
    mm_.SetBufferSpace(compileInfo_.l1Size, compileInfo_.l0CSize, compileInfo_.ubSize);
    if (mm_.GetTiling(tilingData_.matmulTiling) == -1) {
        OP_LOGE(args_.opName, "Mc2MatmulV3 Get Tiling Failed!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void Mc2MatmulV3BaseTiling::SetParamsV310()
{
    if (!compileInfo_.supportL12BtBf16) {
        return;
    }
    m256Align_ = true;
    kA256Align_ = true;
    kB256Align_ = true;
    n256Align_ = true;
    calcMBasic_ = CALC_M_BASIC_L0C_256;
    calcMNBasic_ = CALC_MN_BASIC_L0C_256;
    l2TileLength_ = L2_TILE_LENGTH_L2_128;
}

void Mc2MatmulV3BaseTiling::SetNd2NzInfo()
{
    tilingData_.baseAN = static_cast<uint32_t>(runInfo_.baseAN);
    tilingData_.baseAD = static_cast<uint32_t>(runInfo_.baseAD);
    tilingData_.baseBN = static_cast<uint32_t>(runInfo_.baseBN);
    tilingData_.baseBD = static_cast<uint32_t>(runInfo_.baseBD);
}

void Mc2MatmulV3BaseTiling::SetRunInfo()
{
    tilingData_.matmulRunInfo.transA = static_cast<uint32_t>(args_.isATrans);
    tilingData_.matmulRunInfo.transB = static_cast<uint32_t>(args_.isBTrans);
    tilingData_.matmulRunInfo.nd2nzA = static_cast<uint32_t>(args_.nd2nzA);
    tilingData_.matmulRunInfo.nd2nzB = static_cast<uint32_t>(args_.nd2nzB);
    tilingData_.matmulRunInfo.isNzA = static_cast<uint32_t>(args_.isNzA);
    tilingData_.matmulRunInfo.isNzB = static_cast<uint32_t>(args_.isNzB);
    tilingData_.matmulRunInfo.isHf32 = static_cast<uint32_t>(args_.isHf32);
}

bool Mc2MatmulV3BaseTiling::GetTilingFromRepo()
{
    OP_LOGI(args_.opName, "start get tiling from repo.");
    std::shared_ptr<tuningtiling::TuningTilingDef> tuningTiling = nullptr;
    std::shared_ptr<void> inputArgs = nullptr;
    std::size_t inputArgsSize = 0;
    if (GetTilingInputArgs(inputArgs, inputArgsSize) == false) {
        return false;
    }

    auto ret = RuntimeKb::QueryBank(inputArgs.get(), inputArgsSize, "Mc2MatMulV3", compileInfo_.socVersionStr,
                                    static_cast<uint32_t>(compileInfo_.aicNum), tuningTiling);
    if (ret != 0U || tuningTiling == nullptr) {
        return false;
    }
    return TranslateAoeTiling(tuningTiling);
}

bool Mc2MatmulV3BaseTiling::GetTilingInputArgs(std::shared_ptr<void> &inputArgs, size_t &size)
{
    std::shared_ptr<tuningtiling::Mc2MatMulV3InputArgs> matmulInputArgs = nullptr;
    try {
        matmulInputArgs = std::make_shared<tuningtiling::Mc2MatMulV3InputArgs>();
    } catch (const std::bad_alloc &) {
        OP_LOGI(args_.opName, "get tiling from repo error: input args is nullptr.");
        return false;
    }

    uint64_t blockCube = 16;                                  // 16元素为32B对齐
    uint64_t reduce = (args_.aType == ge::DT_FLOAT) ? 8 : 16; // fp32元素数为8，fp16元素数16，用于32B对齐
    matmulInputArgs->trans_a_flag = args_.isATrans;
    matmulInputArgs->trans_b_flag = args_.isBTrans;
    matmulInputArgs->bias_flag = args_.hasBias;
    matmulInputArgs->a_format = args_.aFormat;
    matmulInputArgs->b_format = args_.bFormat;
    matmulInputArgs->out_format = args_.outFormat;
    matmulInputArgs->a_dtype = (args_.aType == ge::DT_BF16) ? ge::DT_FLOAT16 : args_.aType;
    matmulInputArgs->b_dtype = (args_.bType == ge::DT_BF16) ? ge::DT_FLOAT16 : args_.bType;
    matmulInputArgs->out_dtype = (args_.cType == ge::DT_BF16) ? ge::DT_FLOAT16 : args_.cType;
    matmulInputArgs->m = static_cast<int64_t>(MathUtil::Align(args_.mValue, blockCube));
    matmulInputArgs->k = static_cast<int64_t>(MathUtil::Align(args_.kValue, reduce));
    matmulInputArgs->n = static_cast<int64_t>(MathUtil::Align(args_.nValue, blockCube));
    matmulInputArgs->m_align_flag = static_cast<int64_t>(args_.mValue) == matmulInputArgs->m;
    matmulInputArgs->k_align_flag = static_cast<int64_t>(args_.kValue) == matmulInputArgs->k;
    matmulInputArgs->n_align_flag = static_cast<int64_t>(args_.nValue) == matmulInputArgs->n;
    matmulInputArgs->batch_a1 = 1;
    matmulInputArgs->batch_a2 = 1;
    matmulInputArgs->batch_a3 = 1;
    matmulInputArgs->batch_a4 = 1;
    matmulInputArgs->batch_b1 = 1;
    matmulInputArgs->batch_b2 = 1;
    matmulInputArgs->batch_b3 = 1;
    matmulInputArgs->batch_b4 = 1;
    matmulInputArgs->l1_fused_num = 0;
    matmulInputArgs->aub_double_num = 1;
    matmulInputArgs->bub_double_num = 1;
    matmulInputArgs->fused_double_operand_num = 0;
    matmulInputArgs->reserved_bool = false;
    matmulInputArgs->reserved_params1 = 0;
    matmulInputArgs->reserved_params2 = 0;
    matmulInputArgs->reserved_params3 = 0;
    matmulInputArgs->reserved_params4 = 0;
    matmulInputArgs->reserved_params5 = 0;
    matmulInputArgs->reserved_params6 = 0;

    inputArgs = matmulInputArgs;
    size = sizeof(tuningtiling::Mc2MatMulV3InputArgs);
    DebugLog(matmulInputArgs);
    return true;
}

void Mc2MatmulV3BaseTiling::DebugLog(const std::shared_ptr<tuningtiling::Mc2MatMulV3InputArgs> &inputArgs)
{
    OP_LOGD(args_.opName,
            "Binary tiling info dict, "
            "a_dtype: %d, a_format: %d, aub_double_num: %f, "
            "b_dtype: %d, b_format: %d, "
            "batch_a1: %ld, batch_a2: %ld, batch_a3: %ld, batch_a4: %ld, "
            "batch_b1: %ld, batch_b2: %ld, batch_b3: %ld, batch_b4: %ld, "
            "bias_flag: %d, bub_double_num: %f, fused_double_operand_num: %f, "
            "k: %ld, k_align_flag: %d, l1_fused_num: %f, m: %ld, m_align_flag: %d, "
            "n: %ld, n_align_flag: %d, out_dtype: %d, out_format: %d, "
            "reserved_bool: %d, reserved_params1: %lu, reserved_params2: %lu, reserved_params3: %lu, "
            "reserved_params4: %lu, reserved_params5: %lu, reserved_params6: %lu, "
            "trans_a_flag: %d, trans_b_flag: %d.",
            static_cast<int32_t>(inputArgs->a_dtype), static_cast<int32_t>(inputArgs->a_format),
            inputArgs->aub_double_num, static_cast<int32_t>(inputArgs->b_dtype),
            static_cast<int32_t>(inputArgs->b_format), inputArgs->batch_a1, inputArgs->batch_a2, inputArgs->batch_a3,
            inputArgs->batch_a4, inputArgs->batch_b1, inputArgs->batch_b2, inputArgs->batch_b3, inputArgs->batch_b4,
            static_cast<int32_t>(inputArgs->bias_flag), inputArgs->bub_double_num, inputArgs->fused_double_operand_num,
            inputArgs->k, static_cast<int32_t>(inputArgs->k_align_flag), inputArgs->l1_fused_num, inputArgs->m,
            static_cast<int32_t>(inputArgs->m_align_flag), inputArgs->n, static_cast<int32_t>(inputArgs->n_align_flag),
            static_cast<int32_t>(inputArgs->out_dtype), static_cast<int32_t>(inputArgs->out_format),
            static_cast<int32_t>(inputArgs->reserved_bool), inputArgs->reserved_params1, inputArgs->reserved_params2,
            inputArgs->reserved_params3, inputArgs->reserved_params4, inputArgs->reserved_params5,
            inputArgs->reserved_params6, static_cast<int32_t>(inputArgs->trans_a_flag),
            static_cast<int32_t>(inputArgs->trans_b_flag));
}


bool Mc2MatmulV3BaseTiling::TranslateAoeTiling(tuningtiling::TuningTilingDefPtr &tuningTiling)
{
    auto aoeTiling = std::dynamic_pointer_cast<tuningtiling::Mc2MatMulV3TunnerTiling>(tuningTiling);
    if (aoeTiling == nullptr) {
        return false;
    }
    tilingData_.matmulTiling.usedCoreNum = aoeTiling->usedCoreNum;
    tilingData_.matmulTiling.singleCoreM = aoeTiling->singleCoreM;
    tilingData_.matmulTiling.singleCoreN = aoeTiling->singleCoreN;
    tilingData_.matmulTiling.singleCoreK = aoeTiling->singleCoreK;
    tilingData_.matmulTiling.baseM = aoeTiling->baseM;
    tilingData_.matmulTiling.baseN = aoeTiling->baseN;
    tilingData_.matmulTiling.baseK = aoeTiling->baseK;
    tilingData_.matmulTiling.depthA1 = aoeTiling->depthA1;
    tilingData_.matmulTiling.depthB1 = aoeTiling->depthB1;
    tilingData_.matmulTiling.stepM = aoeTiling->stepM;
    tilingData_.matmulTiling.stepN = aoeTiling->stepN;
    tilingData_.matmulTiling.iterateOrder = aoeTiling->iterateOrder;
    tilingData_.matmulTiling.stepKa = aoeTiling->stepKa;
    tilingData_.matmulTiling.stepKb = aoeTiling->stepKb;
    tilingData_.matmulTiling.dbL0A = aoeTiling->dbL0A;
    tilingData_.matmulTiling.dbL0B = aoeTiling->dbL0B;
    tilingData_.matmulTiling.dbL0C = aoeTiling->dbL0C;
    tilingData_.matmulTiling.dbL0C = aoeTiling->dbL0C;
    if (!compileInfo_.supportL0c2out) {
        tilingData_.matmulTiling.transLength = L0C_SIZE_256_KB / NUM_HALF;
        tilingData_.matmulTiling.shareUbSize = 0;
    }
    tilingData_.tileL2cacheTiling.mTileCntL2 = aoeTiling->l2MTileCnt;
    tilingData_.tileL2cacheTiling.nTileCntL2 = aoeTiling->l2NTileCnt;
    tilingData_.tileL2cacheTiling.mTileBlock = aoeTiling->l2MTileBlock;
    tilingData_.tileL2cacheTiling.nTileBlock = aoeTiling->l2NTileBlock;
    tilingData_.tileL2cacheTiling.calOrder = aoeTiling->l2IterateOrder;
    tilingData_.matmulRunInfo.transA = args_.isATrans;
    tilingData_.matmulRunInfo.transB = args_.isBTrans;
    tilingData_.matmulRunInfo.nd2nzA = args_.nd2nzA;
    tilingData_.matmulRunInfo.nd2nzB = args_.nd2nzB;
    tilingData_.matmulRunInfo.isNzA = args_.isNzA;
    tilingData_.matmulRunInfo.isNzB = args_.isNzB;
    if (!CheckAoeTilingEnable(aoeTiling->tilingEnable, args_.opName)) {
        OP_LOGW(args_.opName, "Get tiling from repo, but the tilingEnable is invalid.");
        return false;
    }
    OP_LOGI(args_.opName, "Get tiling from repo success.");
    return true;
}

void Mc2MatmulV3BaseTiling::SetBaseBlockTiling()
{
    runInfo_.usedCoreNum = compileInfo_.aicNum;
    runInfo_.baseM = BASIC_BLOCK_SIZE_128;
    runInfo_.baseN = BASIC_BLOCK_SIZE_256;
    runInfo_.baseK = BASIC_BLOCK_K_128_BYTE / aDtypeSize_;
    if (compileInfo_.l0CSize == L0C_SIZE_256_KB) {
        runInfo_.baseM = BASIC_BLOCK_SIZE_256;
        return;
    }
    if (!args_.isATrans && !args_.isBTrans) {
        return;
    }
    if (args_.isATrans && args_.isBTrans) {
        runInfo_.baseM = BASIC_BLOCK_SIZE_256;
        runInfo_.baseN = BASIC_BLOCK_SIZE_128;
        return;
    }
    if (args_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGI(args_.opName, "ensure v3 NDNZ baseMN same with v2 NDNZ, avoid logic below.");
        return;
    }
    bool mEnableLarge = args_.mValue >= (compileInfo_.aicNum >> 1) * BASIC_BLOCK_SIZE_256;
    bool nEnableLarge = args_.nValue >= (compileInfo_.aicNum >> 1) * BASIC_BLOCK_SIZE_256;
    if (mEnableLarge && nEnableLarge) {
        uint64_t sizeMLarge = MathUtil::CeilDivision(args_.mValue, BASIC_BLOCK_SIZE_256) * args_.nValue +
                              MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_128) * args_.mValue;
        uint64_t sizeNLarge = MathUtil::CeilDivision(args_.mValue, BASIC_BLOCK_SIZE_128) * args_.nValue +
                              MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_256) * args_.mValue;
        if (sizeMLarge < sizeNLarge) {
            runInfo_.baseM = BASIC_BLOCK_SIZE_256;
            runInfo_.baseN = BASIC_BLOCK_SIZE_128;
        }
        return;
    }
    if (mEnableLarge && !nEnableLarge) {
        runInfo_.baseM = BASIC_BLOCK_SIZE_256;
        runInfo_.baseN = BASIC_BLOCK_SIZE_128;
        return;
    }
    if (!mEnableLarge && !nEnableLarge) {
        if (args_.mValue > args_.nValue) {
            runInfo_.baseM = BASIC_BLOCK_SIZE_256;
            runInfo_.baseN = BASIC_BLOCK_SIZE_128;
        }
    }
    return;
}

inline void Convert2AscendCTiling(CacheTilingData &tbeTiling, const BatchmatmulRunParas &runParams,
                                  const Mc2MatmulV3Args &args, uint64_t aicNum, Mc2MatmulV3RunInfo &runInfo)
{
    if (tbeTiling.al1_full_load && !runParams.pattern_flag) {
        tbeTiling.kal1_16 = runParams.k / tbeTiling.k_dim;
        tbeTiling.m_al1 = MathUtil::CeilDivision(runParams.m, tbeTiling.m_dim * tbeTiling.m_l0);
    }

    if (tbeTiling.bl1_full_load && !runParams.pattern_flag) {
        tbeTiling.kbl1_16 = runParams.k / tbeTiling.k_dim;
        tbeTiling.n_bl1 = MathUtil::CeilDivision(runParams.n, tbeTiling.n_dim * tbeTiling.n_l0);
    }
    uint64_t blockCube = 16; // 16元素为32B对齐
    runInfo.usedCoreNum = static_cast<uint64_t>(tbeTiling.m_dim) * tbeTiling.n_dim;
    // tbe tiling超核时，ascend使能最大核数即可，kernel侧会按base块数量分配
    if (runInfo.usedCoreNum > aicNum) {
        runInfo.usedCoreNum = aicNum;
    }
    runInfo.baseM = static_cast<uint64_t>(tbeTiling.m_l0) * blockCube;
    runInfo.baseN = static_cast<uint64_t>(tbeTiling.n_l0) * blockCube;
    runInfo.baseK = static_cast<uint64_t>(tbeTiling.k_l0) * blockCube;
    runInfo.singleCoreM = runInfo.baseM;
    runInfo.singleCoreN = runInfo.baseN;
    runInfo.singleCoreK = args.kValue;
    runInfo.stepM = static_cast<uint64_t>(tbeTiling.m_al1);
    runInfo.stepN = static_cast<uint64_t>(tbeTiling.n_bl1);
    runInfo.iterateOrder = ITER_COL_FIRST;
    runInfo.stepKa = static_cast<uint64_t>(tbeTiling.kal1_16 / tbeTiling.k_l0);
    runInfo.stepKb = static_cast<uint64_t>(tbeTiling.kbl1_16 / tbeTiling.k_l0);
    runInfo.depthA1 = runInfo.stepM * runInfo.stepKa * static_cast<uint64_t>(tbeTiling.db_al1);
    runInfo.depthB1 = runInfo.stepN * runInfo.stepKb * static_cast<uint64_t>(tbeTiling.db_bl1);
    runInfo.dbL0c = static_cast<uint64_t>(tbeTiling.db_l0c);
    runInfo.l2Info.mTile = 1;
    runInfo.l2Info.nTile = 1;
    runInfo.l2Info.mTileBlock = 1;
    runInfo.l2Info.nTileBlock = 1;
}

inline void InitRunParams(const BatchmatmulCompileParas &compileParams, const Mc2MatmulV3Args &args,
                          BatchmatmulRunParas &runParams)
{
    runParams.trans_a_flag = args.isATrans;
    runParams.trans_b_flag = args.isBTrans;
    runParams.format_a_nd = true;
    runParams.format_b_nd = true;
    runParams.format_out_nd = true;
    runParams.format_a = ge::FORMAT_ND;
    runParams.format_b = ge::FORMAT_ND;
    runParams.format_out = ge::FORMAT_ND;
    runParams.reserved_bool = 0;
    runParams.nd_flag = true;
    runParams.use_pre_ub = false;
    runParams.weight_nz_flag = false;
    runParams.batch_a1 = 1;
    runParams.batch_a2 = 1;
    runParams.batch_a3 = 1;
    runParams.batch_a4 = 1;
    runParams.batch_b1 = 1;
    runParams.batch_b2 = 1;
    runParams.batch_b3 = 1;
    runParams.batch_b4 = 1;

    runParams.b_have_batch = false;
    runParams.is_batch_matmul_mode = false;
    runParams.is_batch_matmul_op = false;
    uint64_t blockCube = 16;                                 // 16元素为32B对齐
    uint64_t reduce = (args.aType == ge::DT_FLOAT) ? 8 : 16; // fp32元素数为8，fp16元素数16，用于32B对齐
    bool alignedMKN = (args.mValue % blockCube == 0) && (args.kValue % reduce == 0) && (args.nValue % blockCube == 0);
    runParams.used_aligned_pattern = alignedMKN && runParams.nd_flag;
    runParams.bias_flag = args.hasBias;
    runParams.pattern_flag = compileParams.pattern_flag;
    runParams.unaligned_flag = !alignedMKN;
    runParams.zero_flag = compileParams.zero_flag;
    runParams.hf32_flag = 0;
    runParams.dtype_a = args.aType;
    runParams.dtype_b = args.bType;
    runParams.dtype_out = args.cType;
    runParams.dtype_bias = ge::GetSizeByDataType(args.biasType);
    runParams.m = static_cast<int64_t>(ops::CeilDiv(args.mValue, blockCube));
    runParams.k = static_cast<int64_t>(ops::CeilDiv(args.kValue, reduce));
    runParams.n = static_cast<int64_t>(ops::CeilDiv(args.nValue, blockCube));
    runParams.batch = 1;
    runParams.ori_shape_m = static_cast<int64_t>(args.mValue);
    runParams.ori_shape_k = static_cast<int64_t>(args.kValue);
    runParams.ori_shape_n = static_cast<int64_t>(args.nValue);
    runParams.bias_dtype = args.biasType;
}

void Mc2MatmulV3BaseTiling::GetV2Tiling()
{
    BatchmatmulCompileParas compileParams;
    compileParams.binary_mode_flag = true;
    compileParams.bias_flag = args_.hasBias;
    compileParams.pattern_flag = compileInfo_.supportL0c2out;
    compileParams.zero_flag = false;
    compileParams.aub_double_num = 1;
    compileParams.bub_double_num = 1;

    BatchmatmulRunParas runParams;
    InitRunParams(compileParams, args_, runParams);

    CacheTilingData tiling;
    tiling.tiling_id = std::numeric_limits<uint64_t>::max();
    GenTiling("Mc2MatMulV3", compileParams, runParams, tiling, context_);
    Convert2AscendCTiling(tiling, runParams, args_, compileInfo_.aicNum, runInfo_);
    return;
}

bool Mc2MatmulV3BaseTiling::CheckUbOverFlow(uint64_t nAligned16, uint64_t nValue, const uint64_t &baseN,
                                            const uint64_t &baseD, uint64_t dtypeSize)
{
    uint64_t nAlignedLoop = MathUtil::CeilDivision(nAligned16, baseN);
    uint64_t nValueLoop = MathUtil::CeilDivision(nValue, baseN);
    if (baseN == 0UL || dtypeSize == 0UL) {
        return false;
    }
    return ((nAlignedLoop != nValueLoop) &&
            ((nAligned16 - ((nValue / baseN) - 1) * baseN) * baseD > (compileInfo_.ubSize / NUMBER_TWO / dtypeSize)));
}

void Mc2MatmulV3BaseTiling::CalcNd2NzTiling(uint64_t dtypeSize, uint64_t nValue, uint64_t dValue, uint64_t &baseN,
                                            uint64_t &baseD)
{
    constexpr uint64_t mataD = 16384;
    constexpr uint64_t minN = 7168;
    // mata 交织场景
    if (dValue % mataD == 0 && nValue >= minN && compileInfo_.ubSize == UB_SIZE &&
        (args_.aType == ge::DT_FLOAT16 || args_.aType == ge::DT_BF16) && compileInfo_.aicNum >= 24) { // 24 核最小核数
        baseN = 96;  // baseN 经验值 96
        baseD = 512; // baseD 经验值 512
        return;
    }

    if (dtypeSize == 0) {
        return;
    }
    uint64_t vectorCoreNum = NUMBER_TWO * runInfo_.usedCoreNum;
    vectorCoreNum = std::max(vectorCoreNum, 1UL);
    uint64_t baseThres = VECTOR_D_BASE / dtypeSize;
    uint64_t c0 = BLOCK_BYTE_SIZE / dtypeSize;
    uint64_t nAligned16 = ops::CeilAlign(nValue, N_ALIGNED);
    uint64_t dAlignedC0 = ops::CeilAlign(dValue, c0);
    if (dValue <= baseThres) {
        baseD = std::max(ops::CeilAlign(dValue, c0), uint64_t(1));
        uint64_t nDim = vectorCoreNum;
        baseN = compileInfo_.ubSize / NUMBER_TWO / dtypeSize / baseD;
        uint64_t round = std::max(MathUtil::CeilDivision(MathUtil::CeilDivision(nAligned16, nDim), baseN), 1UL);
        baseN = std::max(MathUtil::CeilDivision(MathUtil::CeilDivision(nAligned16, nDim), round), NCALC_THRES);
        while (baseN > NCALC_THRES) {
            if (CheckUbOverFlow(nAligned16, nValue, baseN, baseD, dtypeSize)) {
                baseN--;
                continue;
            }
            break;
        }
        return;
    }
    uint64_t lastTail = 0;
    // bestBaseD = 4K / dtype, bestBaseN = UB / 2 / 4096
    uint64_t bestBaseN = NCALC_THRES;
    uint64_t bestBaseD = CALC_ND_BASIC.at(1) / dtypeSize;
    for (auto base : CALC_ND_BASIC) {
        baseD = std::max(std::min(dAlignedC0, base / dtypeSize), uint64_t(1));
        uint64_t dLoop = MathUtil::CeilDivision(dAlignedC0, baseD);
        uint64_t dTail = dAlignedC0 % baseD;
        if (dTail > 0 && dTail < (MIN_TAIL / dtypeSize)) {
            if (baseD * dtypeSize ==
                CALC_ND_BASIC.at(0)) { // if dtail < 0.5K && baseD == 3k, give up and return bestbase
                continue;
            }
            dLoop--;
            baseD = std::max(ops::CeilAlign(MathUtil::CeilDivision(dAlignedC0, dLoop), c0), uint64_t(1));
            ; // dtail< 0.5K, make baseD larger
        }
        baseN = std::max(compileInfo_.ubSize / NUMBER_TWO / dtypeSize / baseD, NUMBER_SIXTEEN);
        if (baseN * baseD * dtypeSize * NUMBER_TWO > compileInfo_.ubSize) { // check ub overflow
            continue;
        }
        if (CheckUbOverFlow(nAligned16, nValue, baseN, baseD, dtypeSize)) {
            continue;
        }
        uint64_t nLoop = MathUtil::CeilDivision(nAligned16, baseN);
        uint64_t tail = nLoop * dLoop % vectorCoreNum;
        while (baseN > NCALC_THRES) {
            if (CheckUbOverFlow(nAligned16, nValue, baseN, baseD, dtypeSize)) { // check ub overflow
                baseN--;
                nLoop = MathUtil::CeilDivision(nAligned16, baseN);
                tail = nLoop * dLoop % vectorCoreNum;
                continue;
            }
            if (tail == 0) {
                return;
            }
            if (tail > lastTail) {
                lastTail = tail;
                bestBaseD = baseD;
                bestBaseN = baseN;
            }
            baseN--;
            nLoop = MathUtil::CeilDivision(nAligned16, baseN);
            tail = nLoop * dLoop % vectorCoreNum;
        }
    }
    baseD = bestBaseD;
    baseN = bestBaseN;
    return;
}

void Mc2MatmulV3BaseTiling::DoNd2NzVectorTiling()
{
    UpdateNd2nzFlag();
    OP_LOGI(args_.opName, "args_.nd2nzA = %d, args_.nd2nzB = %d.", static_cast<uint32_t>(args_.nd2nzA),
            static_cast<uint32_t>(args_.nd2nzB));
    if (args_.nd2nzA) {
        uint64_t nValue = args_.isATrans ? args_.kValue : args_.mValue;
        uint64_t dValue = args_.isATrans ? args_.mValue : args_.kValue;
        CalcNd2NzTiling(aDtypeSize_, nValue, dValue, runInfo_.baseAN, runInfo_.baseAD);
        OP_LOGI(args_.opName, "baseAN is %ld baseAD is %ld", runInfo_.baseAN, runInfo_.baseAD);
    }
    if (args_.nd2nzB) {
        uint64_t nValue = args_.isBTrans ? args_.nValue : args_.kValue;
        uint64_t dValue = args_.isBTrans ? args_.kValue : args_.nValue;
        CalcNd2NzTiling(bDtypeSize_, nValue, dValue, runInfo_.baseBN, runInfo_.baseBD);
        OP_LOGI(args_.opName, "baseBN is %ld baseBD is %ld", runInfo_.baseBN, runInfo_.baseBD);
    }
}

void Mc2MatmulV3BaseTiling::DoSmallShapeTiling()
{
    OP_LOGI(args_.opName, "meet small block num or meet small shape situation, enable small shape algorithm");
    FormulaicBaseBlockTiling();
    if (!(compileInfo_.l0CSize == L0C_SIZE_256_KB)) {
        if (args_.mValue < SMALL_SHAPE_THRES && args_.mValue > SMALL_SHAPE_LOWER_THRES &&
            runInfo_.baseN > SMALL_SHAPE_LOWER_THRES) {
            runInfo_.baseN = SMALL_SHAPE_LOWER_THRES;
        } else if (args_.nValue < SMALL_SHAPE_THRES && args_.nValue > SMALL_SHAPE_LOWER_THRES &&
                   runInfo_.baseM > SMALL_SHAPE_LOWER_THRES) {
            runInfo_.baseM = SMALL_SHAPE_LOWER_THRES;
        } else if (runInfo_.baseM * runInfo_.baseN > L0C_THRES) {
            runInfo_.baseM = runInfo_.baseM > SMALL_SHAPE_LOWER_THRES && runInfo_.baseM < SMALL_SHAPE_THRES ?
                                 SMALL_SHAPE_LOWER_THRES :
                                 runInfo_.baseM;
            runInfo_.baseN = runInfo_.baseN > SMALL_SHAPE_LOWER_THRES && runInfo_.baseN < SMALL_SHAPE_THRES ?
                                 SMALL_SHAPE_LOWER_THRES :
                                 runInfo_.baseN;
        }
        if (runInfo_.baseM * runInfo_.baseN > L0C_THRES) {
            runInfo_.baseM = L0C_THRES / runInfo_.baseN;
        }
    }
}

void Mc2MatmulV3BaseTiling::DoSelectTiling()
{
    switch (tilingSelect_) {
        case Mc2TilingCalcSelect::ALL:
            DO_CACL_TILING_ENABLE(DoBL1FullloadWithFixpipeTiling())
            DO_CACL_TILING_ENABLE(DoAL1FullLoadTiling())
            DO_CACL_TILING_ENABLE(DoBL1FullLoadTiling())
            DO_CACL_TILING_ENABLE(DoL2CacheTiling())
            DO_CACL_TILING_ENABLE(DoSingleCoreSplitKTiling())
            DO_CACL_TILING_ENABLE(DoDeterministicMultiCoreSplitKTiling())
            DO_CACL_TILING_ENABLE(DoL2CacheTiling310P())
            break;
        case Mc2TilingCalcSelect::BASE:
            DO_CACL_TILING_ENABLE(DoL2CacheTiling())
            DO_CACL_TILING_ENABLE(DoL2CacheTiling310P())
            break;
        case Mc2TilingCalcSelect::SINGLE_CORE_SPLIT_K:
            DO_CACL_TILING_ENABLE(DoSingleCoreSplitKTiling())
            break;
        case Mc2TilingCalcSelect::DETERMINISTIC_SPLIT_K:
            DO_CACL_TILING_ENABLE(DoDeterministicMultiCoreSplitKTiling())
            break;
        default:
            break;
    }
}

void Mc2MatmulV3BaseTiling::DoBasicTiling()
{
    runInfo_.needUpdate = true;
    basicBlockBaseM_ = (compileInfo_.l0CSize == L0C_SIZE_256_KB) ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128;
    ResetBase(runInfo_, compileInfo_.l0CSize, aDtypeSize_);

    runInfo_.singleCoreK = args_.kValue;
    uint64_t alignedMValue = ops::CeilAlign(args_.mValue, basicBlockBaseM_);
    uint64_t alignedNValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_256);
    SetBaseBlockTiling();
    bool smallBlockNum = (alignedMValue / runInfo_.baseM) * (alignedNValue / runInfo_.baseN) < runInfo_.usedCoreNum;
    bool smallShape =
        (args_.mValue < SMALL_SHAPE_THRES) || (args_.nValue < SMALL_SHAPE_THRES); // 256为界限判断是否小shape
    if ((smallBlockNum || smallShape) && !compileInfo_.supportL12BtBf16) {
        DoSmallShapeTiling();
    }

    if (!compileInfo_.supportL0c2out) {
        runInfo_.baseM = std::min(runInfo_.baseM, args_.mValue);
        runInfo_.baseN = std::min(runInfo_.baseN, args_.nValue);
        runInfo_.baseK = std::min(runInfo_.baseK, args_.kValue);
    }

    if (compileInfo_.supportL12BtBf16) {
        const gert::Shape &shapeA = context_->GetInputShape(0)->GetStorageShape();
        const gert::Shape &shapeB = context_->GetInputShape(1)->GetStorageShape();
        const uint64_t dimsA = shapeA.GetDimNum();
        const uint64_t dimsB = shapeB.GetDimNum();
        // isolate bmm which input dims over 2
        if (smallBlockNum && (dimsA == 2 && dimsB == 2)) {
            FormulateBasicBlockDavid();
        }
        CalcTailBasicBlock();
    }

    CalL1Tiling();
    DoIncreTiling();
    OptimizeLoadBalanceBasicKernel();
    DoSelectTiling();
    // add nd2nz tiling here
    DoNd2NzVectorTiling();
    if (args_.hasBias) {
        runInfo_.baseN = std::min(256UL, runInfo_.baseN); // 有bias时 baseN 小于256
        if (tilingEnable_.tilingEnableSplitCore == Mc2TilingEnableSplitCore::BASE &&
            tilingEnable_.tilingEnableFullLoad != Mc2TilingEnableFullLoad::BL1_FULL_LOAD) {
            runInfo_.singleCoreN = runInfo_.baseN;
        }
    }
}

void Mc2MatmulV3BaseTiling::FormulateBasicBlockDavid()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    if (mCore <= nCore) {
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
        mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
        if (mCore == 0) {
            OP_LOGE(args_.opName, "mCore is zero.");
            return;
        }
        nCore = runInfo_.usedCoreNum / mCore;
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
    } else {
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
        nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
        if (nCore == 0) {
            OP_LOGE(args_.opName, "nCore is zero.");
            return;
        }
        mCore = runInfo_.usedCoreNum / nCore;
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
    }
    runInfo_.baseK = std::min(BASIC_BLOCK_SIZE_128 / aDtypeSize_, args_.kValue);
    runInfo_.baseK = ops::CeilAlign(runInfo_.baseK, BASIC_BLOCK_SIZE_16);
}

void Mc2MatmulV3BaseTiling::CalcTailBasicBlock()
{
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    uint64_t tailCnt = mCnt * nCnt % compileInfo_.aicNum;
    tailCnt = mCnt * nCnt <= compileInfo_.aicNum ? 0 : tailCnt;
    runInfo_.l2Info.mTile = 1;
    runInfo_.l2Info.nTile = 1;
    if (tailCnt != 0) {
        while ((runInfo_.l2Info.mTile + 1) * runInfo_.l2Info.nTile * tailCnt <= compileInfo_.aicNum) {
            runInfo_.l2Info.mTile += 1;
            if (runInfo_.l2Info.mTile * (runInfo_.l2Info.nTile + 1) * tailCnt <= compileInfo_.aicNum) {
                runInfo_.l2Info.nTile += 1;
            }
        }
    }
}

void Mc2MatmulV3BaseTiling::FormulaicTilingNoTrans()
{
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN); // n方向上的base块个数
    CalcBase<Mc2CalcType::M_BY_BASE_NK>(calcMBasic_);
    // 非对齐场景
    if (!kA256Align_ && !n256Align_) {
        // A,B都会做ND2NZ，baseMNK 按照负载均衡分配
        BalanceBaseBlockTiling();
    } else if (!kA256Align_) {
        // A矩阵做ND2NZ，baseN*depth保证256Byte对齐，baseM任意切分
        runInfo_.baseN = BASIC_BLOCK_SIZE_256;
        runInfo_.baseK = BASIC_BLOCK_SIZE_128 / aDtypeSize_;
        runInfo_.baseM = CalBaseSize(nCore, compileInfo_.aicNum, args_.mValue, basicBlockBaseM_);
    } else if (!n256Align_) {
        // B矩阵做ND2NZ，baseK*depth保证256BYTE对齐，baseMN任意切分
        runInfo_.baseK = BASIC_BLOCK_SIZE_128 / aDtypeSize_;
        CalBaseMBaseN(runInfo_.baseM, runInfo_.baseN);
    }
}

void Mc2MatmulV3BaseTiling::FormulaicBaseBlockTiling()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM); // m方向上的base块个数
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN); // n方向上的base块个数

    // ND场景，不需要ND2NZ操作, 说明数据内存方向的边是256B对齐的
    switch (trans_) {
        case Mc2MatmulV3Trans::NO_TRANS:
            {
                FormulaicTilingNoTrans();
                break;
            }
        case Mc2MatmulV3Trans::A_TRANS:
            {
                // [k, m], [k, n], m和n需要256B对齐，k可以任意切分
                // 如果对齐场景都为MTE2 bound，那非对齐场景的MTE2 bound会更严重，此时多核计算没有意义
                if (!m256Align_ && !n256Align_) {
                    // A,B都会做ND2NZ，baseMNK 按照负载均衡分配
                    BalanceBaseBlockTiling();
                } else if (!m256Align_) {
                    // A矩阵做ND2NZ，baseN*depth保证256BYTE对齐，baseM任意切分
                    runInfo_.baseM = CalBaseSize(nCore, compileInfo_.aicNum, args_.mValue, basicBlockBaseM_);
                } else if (!n256Align_) {
                    runInfo_.baseN = CalBaseSize(mCore, compileInfo_.aicNum, args_.nValue, BASIC_BLOCK_SIZE_256);
                }
                break;
            }
        case Mc2MatmulV3Trans::B_TRANS:
            {
                // [m, k], [n, k], k需要256B对齐，m和n可以任意切分
                // 负载均匀切分
                CalcBase<Mc2CalcType::MN_BY_BASE_K>(calcMNBasic_);
                if (!kA256Align_) {
                    // A,B都会做ND2NZ，baseMNK 按照负载均衡分配
                    BalanceBaseBlockTiling();
                }
                break;
            }
        case Mc2MatmulV3Trans::AB_TRANS:
            {
                // [k, m], [n, k], m和k需要256B对齐，n可以任意切分
                runInfo_.baseN = CalBaseSize(mCore, compileInfo_.aicNum, args_.nValue, BASIC_BLOCK_SIZE_256);
                if (!m256Align_ && kB256Align_) {
                    // A,B都会做ND2NZ，baseMNK 按照负载均衡分配
                    BalanceBaseBlockTiling();
                } else if (!m256Align_) {
                    CalBaseMBaseN(runInfo_.baseM, runInfo_.baseN);
                }
                break;
            }
        default:
            break;
    }
    // baseN>=512时, nd2nz vmuls指令dstRepeatStride会大于255, 只能设置repeat为1，指令效率不高
    if (args_.outFormat == ge::FORMAT_ND && !compileInfo_.supportL0c2out) {
        runInfo_.baseN = std::min(runInfo_.baseN, BASIC_BLOCK_SIZE_256);
    }
    return;
}

void Mc2MatmulV3BaseTiling::BalanceBaseBlockTiling()
{
    ResetBase(runInfo_, compileInfo_.l0CSize, aDtypeSize_);
    uint64_t nCore = 1;
    uint64_t mCore = 1;
    if (args_.mValue >= basicBlockBaseM_) {
        mCore = std::max(MathUtil::CeilDivision(args_.mValue, basicBlockBaseM_), 1UL);
        nCore = std::max(compileInfo_.aicNum / mCore, 1UL);
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_ALIGN_16);
        runInfo_.baseN = std::min(runInfo_.baseN, BASIC_BLOCK_SIZE_256);
        uint64_t baseKa = compileInfo_.l0ASize / DB_SIZE / aDtypeSize_ / BASIC_BLOCK_SIZE_256;
        uint64_t baseKb = ops::FloorDiv(compileInfo_.l0BSize, DB_SIZE * bDtypeSize_ * runInfo_.baseN);
        runInfo_.baseK = ops::FloorAlign(std::min(baseKa, baseKb), BASIC_ALIGN_16);
    } else {
        uint64_t maxBaseN2 = args_.mValue >= 64 ? basicBlockBaseM_ : 1024; // 64 mlimit size; 1024 max size
        runInfo_.baseM = ops::CeilAlign(args_.mValue, BASIC_ALIGN_16);
        uint64_t nTile = std::max(args_.nValue / compileInfo_.aicNum, 1UL);
        runInfo_.baseN = ops::CeilAlign(nTile, BASIC_ALIGN_16);
        runInfo_.baseN = std::max(runInfo_.baseN, maxBaseN2);
        uint64_t maxBaseN = ops::FloorDiv(compileInfo_.l0CSize, LOC_DATA_SIZE * runInfo_.baseM);
        maxBaseN = ops::FloorAlign(maxBaseN, BASIC_ALIGN_16);
        if (args_.hasBias && compileInfo_.supportL0c2out) {
            maxBaseN = std::min(maxBaseN, compileInfo_.btSize / DATA_SIZE_FP32);
        }
        runInfo_.baseN = std::min(runInfo_.baseN, maxBaseN);
        nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
        uint64_t tailCoreNum = 0;
        if (runInfo_.baseN > BASIC_BLOCK_SIZE_256) {
            tailCoreNum = MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_256) % compileInfo_.aicNum;
        }
        if (nCore % compileInfo_.aicNum < tailCoreNum) {
            runInfo_.baseN = BASIC_BLOCK_SIZE_256;
        }
        uint64_t baseKa = ops::FloorDiv(compileInfo_.l0ASize, DB_SIZE * aDtypeSize_ * runInfo_.baseM);
        uint64_t baseKb = ops::FloorDiv(compileInfo_.l0BSize, DB_SIZE * bDtypeSize_ * runInfo_.baseN);
        runInfo_.baseK = ops::FloorAlign(std::min(baseKa, baseKb), BASIC_ALIGN_16);
    }

    if (runInfo_.baseM <= 0 || runInfo_.baseN <= 0 || runInfo_.baseK <= 0 || runInfo_.baseM % BASIC_ALIGN_16 != 0 ||
        runInfo_.baseN % BASIC_ALIGN_16 != 0 || runInfo_.baseK % BASIC_ALIGN_16 != 0) {
        ResetBase(runInfo_, compileInfo_.l0CSize, aDtypeSize_);
    }
    return;
}

bool Mc2MatmulV3BaseTiling::CheckBTSize(uint64_t baseN)
{
    if (!compileInfo_.supportL0c2out && baseN * DATA_SIZE_FP32 > compileInfo_.btSize) {
        return false;
    }
    return true;
}

template <Mc2CalcType T>
void Mc2MatmulV3BaseTiling::CalcBase(const std::vector<std::vector<uint64_t>> &baseMNK)
{
    std::vector<std::vector<uint64_t>> calBaseMNK;
    size_t index = 0;
    uint64_t minLoadSize = -1;
    uint64_t nCnt = 1;
    uint64_t mCnt = 1;
    for (size_t i = 0; i < baseMNK.size(); ++i) {
        std::vector<uint64_t> tempBaseMNK(baseMNK[i].begin(), baseMNK[i].end());
        if (T == Mc2CalcType::M_BY_BASE_NK) {
            nCnt = MathUtil::CeilDivision(args_.nValue, baseMNK[i][1]);
            // baseM是除完n方向上的核数后，m方向上能分的最大base块
            tempBaseMNK[0] = CalBaseSize(nCnt, compileInfo_.aicNum, args_.mValue, baseMNK[i][0]);
            mCnt = MathUtil::CeilDivision(args_.mValue, tempBaseMNK[0]);
            calBaseMNK.push_back(tempBaseMNK);
        } else {
            CalBaseMBaseN(tempBaseMNK[0], tempBaseMNK[1], calcMNBasic_[i][0], calcMNBasic_[i][1]);
            calBaseMNK.push_back(tempBaseMNK);
            mCnt = MathUtil::CeilDivision(args_.mValue, tempBaseMNK[0]);
            nCnt = MathUtil::CeilDivision(args_.nValue, tempBaseMNK[1]);
        }
        if (args_.hasBias && !CheckBTSize(calBaseMNK[i][1])) {
            continue;
        }

        uint64_t tmpTailCore = (mCnt * nCnt) % compileInfo_.aicNum;
        uint64_t tmpLoadSize = (tempBaseMNK[0] + tempBaseMNK[1]) * (mCnt * nCnt / compileInfo_.aicNum);
        tmpLoadSize += (tmpTailCore > 0) ? (tempBaseMNK[0] + tempBaseMNK[1]) : 0;
        if (tmpLoadSize < minLoadSize) {
            index = i;
            minLoadSize = tmpLoadSize;
        }
    }
    runInfo_.baseM = calBaseMNK[index][0];               // 0: m index
    runInfo_.baseN = calBaseMNK[index][1];               // 1: n index
    runInfo_.baseK = calBaseMNK[index][2] / aDtypeSize_; // 2: k index
}

void Mc2MatmulV3BaseTiling::CalBaseMBaseN(uint64_t &baseM, uint64_t &baseN, uint64_t maxBaseM, uint64_t maxBaseN)
{
    // 根据 m方向和n方向上谁的基本块多，更新base块的大小
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, maxBaseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, maxBaseN);
    if (mCore < nCore) {
        nCore = std::max(compileInfo_.aicNum / mCore, 1UL);
        baseN = ops::CeilAlign(std::max(args_.nValue / nCore, 1UL), BASIC_ALIGN_16);
        if (baseN > maxBaseN) {
            baseN = maxBaseN;
        }
    } else {
        mCore = std::max(compileInfo_.aicNum / nCore, 1UL);
        baseM = ops::CeilAlign(std::max(args_.mValue / mCore, 1UL), BASIC_ALIGN_16);
        if (baseM > maxBaseM) {
            baseM = maxBaseM;
        }
    }

    baseM = ops::CeilAlign(std::min(args_.mValue, baseM), BASIC_ALIGN_16);
    baseN = ops::CeilAlign(std::min(args_.nValue, baseN), BASIC_ALIGN_16);
    return;
}

void Mc2MatmulV3BaseTiling::CalL1TilingV200()
{
    runInfo_.singleCoreK = args_.kValue;
    runInfo_.stepM = 1;
    runInfo_.stepN = 1;
    runInfo_.depthA1 = 16; // 16 is full use l1 space
    runInfo_.depthB1 = 16; // 16 is full use l1 space
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = runInfo_.baseN;

    if (args_.aFormat == ge::FORMAT_ND || args_.bFormat == ge::FORMAT_ND) {
        runInfo_.depthA1 = 6; // 6为经验值
        runInfo_.depthB1 = 6; // 6为经验值
    }
    runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
    runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;
    if ((args_.aFormat == ge::FORMAT_FRACTAL_NZ) && (args_.bFormat == ge::FORMAT_FRACTAL_NZ)) {
        // ka全载
        if (args_.mValue <= runInfo_.baseM) {
            runInfo_.baseM = args_.mValue;
            if (runInfo_.baseM * args_.kValue * aDtypeSize_ < (compileInfo_.l1Size / NUM_HALF)) {
                runInfo_.depthA1 = MathUtil::CeilDivision(args_.kValue, runInfo_.baseK);
                runInfo_.stepKa = runInfo_.depthA1;
            }
        }
        // kb全载
        if (args_.nValue <= runInfo_.baseN) {
            runInfo_.baseN = args_.nValue;
            if (runInfo_.baseN * args_.kValue * bDtypeSize_ < (compileInfo_.l1Size / NUM_HALF)) {
                runInfo_.depthB1 = MathUtil::CeilDivision(args_.kValue, runInfo_.baseK);
                runInfo_.stepKb = runInfo_.depthB1;
            }
        }
    }
    return;
}

void Mc2MatmulV3BaseTiling::CalL1Tiling()
{
    if (!compileInfo_.supportL0c2out) {
        CalL1TilingV200();
        return;
    }
    uint64_t totalL1Size = compileInfo_.l1Size + 256; // 256B为预留给rpc使用，单算子不涉及
    uint64_t reserveBTSize = args_.hasBias ? BIAS_TABLE_NUM * DATA_SIZE_FP32 : 0;
    runInfo_.depthA1 = totalL1Size / NUM_HALF / runInfo_.baseM / runInfo_.baseK / aDtypeSize_; // 2: half of l1
    runInfo_.depthB1 = totalL1Size / NUM_HALF / runInfo_.baseN / runInfo_.baseK / bDtypeSize_; // 2: half of l1

    uint64_t depthASize = runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * aDtypeSize_;
    uint64_t depthBSize = runInfo_.depthB1 * runInfo_.baseN * runInfo_.baseK * bDtypeSize_;
    if (depthASize + depthBSize > totalL1Size - reserveBTSize) {
        if (runInfo_.baseM <= runInfo_.baseN) {
            runInfo_.depthA1 = runInfo_.depthA1 / NUM_HALF; // 2: adjust deptch for l1 buffer
        } else {
            runInfo_.depthB1 = runInfo_.depthB1 / NUM_HALF; // 2: adjust deptch for l1 buffer
        }
    }
    runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
    runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;

    UpdateL1TilingStepK(runInfo_.stepKa);
    UpdateL1TilingStepK(runInfo_.stepKb);

    if (runInfo_.stepKa >= runInfo_.stepKb) {
        runInfo_.stepKa = runInfo_.stepKa / runInfo_.stepKb * runInfo_.stepKb;
    } else {
        runInfo_.stepKb = runInfo_.stepKb / runInfo_.stepKa * runInfo_.stepKa;
    }
    runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE; // depth % (stepKa * stepM) == 0
    runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE; // depth % (stepKb * stepN) == 0
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = runInfo_.baseN;
    return;
}

void Mc2MatmulV3BaseTiling::UpdateL1TilingStepK(uint64_t &stepK)
{
    if (stepK * runInfo_.baseK >= args_.kValue) {
        return;
    }

    if (stepK * runInfo_.baseK * aDtypeSize_ > BASIC_ALIGN_512) {
        if ((stepK * runInfo_.baseK * aDtypeSize_ % BASIC_ALIGN_512) != 0 &&
            (BASIC_ALIGN_512 % (runInfo_.baseK * aDtypeSize_)) == 0) {
            while (stepK * runInfo_.baseK * aDtypeSize_ % BASIC_ALIGN_512 != 0 && stepK > 1) {
                stepK--;
            }
        }
    } else if (stepK * runInfo_.baseK * aDtypeSize_ > BASIC_ALIGN_256) {
        if ((stepK * runInfo_.baseK * aDtypeSize_ % BASIC_ALIGN_256) != 0 &&
            (BASIC_ALIGN_256 % (runInfo_.baseK * aDtypeSize_)) == 0) {
            while (stepK * runInfo_.baseK * aDtypeSize_ % BASIC_ALIGN_256 != 0 && stepK > 1) {
                stepK--;
            }
        }
    }
}

void Mc2MatmulV3BaseTiling::InitL2SplitParams(Mc2MatmulV3L2SplitParams &l2SplitParams) const
{
    l2SplitParams.outBase = std::max(runInfo_.baseM, 1UL);
    l2SplitParams.innerBase = std::max(runInfo_.baseN, 1UL);
    l2SplitParams.outValue = args_.mValue;
    l2SplitParams.innerValue = args_.nValue;
    l2SplitParams.outDtypeSize = aDtypeSize_;
    l2SplitParams.innerDtypeSize = bDtypeSize_;
    if (runInfo_.baseN >= runInfo_.baseM) {
        l2SplitParams.outBase = runInfo_.baseN;
        l2SplitParams.innerBase = runInfo_.baseM;
        l2SplitParams.outValue = args_.nValue;
        l2SplitParams.innerValue = args_.mValue;
        l2SplitParams.outDtypeSize = bDtypeSize_;
        l2SplitParams.innerDtypeSize = aDtypeSize_;
    }

    l2SplitParams.maxConflictDim = std::min(compileInfo_.aicNum, 6UL); // 24核最多冲突6核
    l2SplitParams.minConflictDim = std::min(compileInfo_.aicNum, 3UL); // 24核内轴不亲和最多冲突3核
    if (compileInfo_.aicNum == 20) {                                   // 针对20核的场景
        l2SplitParams.maxConflictDim = 5;                              // 20核最多冲突5核
        l2SplitParams.minConflictDim = 4;                              // 20核内轴不亲和最多冲突4核
    }
}

bool Mc2MatmulV3BaseTiling::IsTailSmall(Mc2MatmulV3L2SplitParams &l2SplitParams, uint64_t outL2Split,
                                        uint64_t innerL2Split, uint64_t innerMaxConflict) const
{
    if (outL2Split == 0UL || innerL2Split == 0UL) {
        return false;
    }
    uint64_t outTailValue = ((l2SplitParams.outValue + outL2Split - 1) % outL2Split) + 1;
    uint64_t innerTailValue = ((l2SplitParams.innerValue + innerL2Split - 1) % innerL2Split) + 1;
    l2SplitParams.outTailCnt = MathUtil::CeilDivision(outTailValue, l2SplitParams.outBase);
    l2SplitParams.innerTailCnt = MathUtil::CeilDivision(innerTailValue, l2SplitParams.innerBase);
    bool isOutTailSmall = l2SplitParams.outTailCnt * l2SplitParams.maxConflictDim < compileInfo_.aicNum;
    bool isInnerTailSmall = l2SplitParams.innerTailCnt * innerMaxConflict < compileInfo_.aicNum;
    return (isOutTailSmall || isInnerTailSmall);
}

bool Mc2MatmulV3BaseTiling::CalcTile(uint64_t &outTile, uint64_t &innerTile, uint64_t &outL2Split,
                                     uint64_t &innerL2Split, const bool isInnerBad) const
{
    Mc2MatmulV3L2SplitParams l2SplitParams;
    InitL2SplitParams(l2SplitParams);
    uint64_t innerMaxConflict = isInnerBad ? l2SplitParams.minConflictDim : l2SplitParams.maxConflictDim;
    uint64_t outerMinUseDim = compileInfo_.aicNum / l2SplitParams.maxConflictDim;
    uint64_t innerMinUseDim = compileInfo_.aicNum / innerMaxConflict;
    uint64_t outOriShape = outL2Split;
    uint64_t innerOriShape = innerL2Split;
    uint64_t outConflict = 0;
    uint64_t innerConflict = 0;
    bool enableCache = false;

    for (uint64_t outerUseDim = compileInfo_.aicNum; outerUseDim >= outerMinUseDim; outerUseDim--) {
        for (uint64_t innerUseDim = compileInfo_.aicNum; innerUseDim >= innerMinUseDim; innerUseDim--) {
            uint64_t outTileTmp = std::max(outOriShape / (l2SplitParams.outBase * outerUseDim), 1UL);
            uint64_t innerTileTmp = std::max(innerOriShape / (l2SplitParams.innerBase * innerUseDim), 1UL);
            uint64_t outL2SplitTmp =
                MathUtil::Align(MathUtil::CeilDivision(l2SplitParams.outValue, outTileTmp), l2SplitParams.outBase);
            uint64_t innerL2SplitTmp = MathUtil::Align(MathUtil::CeilDivision(l2SplitParams.innerValue, innerTileTmp),
                                                       l2SplitParams.innerBase);
            uint64_t totalSize = GetTotalSize(outL2SplitTmp, args_.kValue, innerL2SplitTmp, l2SplitParams.outDtypeSize,
                                              l2SplitParams.innerDtypeSize);
            if (totalSize <= args_.l2Ratio * 100 * MB_SIZE) { // 100M为实验数据，确保不会出现L2数据置换
                if (IsTailSmall(l2SplitParams, outL2SplitTmp, innerL2SplitTmp, innerMaxConflict)) {
                    continue;
                }
                uint64_t outConflictTmp = MathUtil::CeilDivision(compileInfo_.aicNum, l2SplitParams.outTailCnt);
                uint64_t innerConflictTmp = MathUtil::CeilDivision(compileInfo_.aicNum, l2SplitParams.innerTailCnt);
                bool isUpdate = !enableCache || (outConflict >= outConflictTmp && innerConflict >= innerConflictTmp);
                if (isUpdate) {
                    enableCache = true;
                    outTile = outTileTmp;
                    innerTile = innerTileTmp;
                    outL2Split = outL2SplitTmp;
                    innerL2Split = innerL2SplitTmp;
                    outConflict = outConflictTmp;
                    innerConflict = innerConflictTmp;
                    OP_LOGI(args_.opName,
                            "Update Params! OutTile: %lu, InnerTile: %lu, OutL2Split: %lu, InnerL2Split: %lu, "
                            "outConflict: %lu, innerConflict: %lu",
                            outTile, innerTile, outL2Split, innerL2Split, outConflict, innerConflict);
                }
            }
        }
    }
    return enableCache;
}

uint64_t Mc2MatmulV3BaseTiling::GetTotalSize(uint64_t m, uint64_t k, uint64_t n, uint64_t aDtype, uint64_t bDtype) const
{
    uint64_t sizeA = m * k * aDtype;
    uint64_t sizeB = k * n * bDtype;
    uint64_t sizeC = m * n * GetSizeByDataType(args_.cType);
    return sizeA + sizeB + sizeC;
}

bool Mc2MatmulV3BaseTiling::DoBL1FullloadWithFixpipeTiling()
{
    if (!NeedSolveFixBound()) {
        return false;
    }
    runInfo_.baseN = ops::CeilAlign(args_.nValue, N_ALIGNED);
    // baseM / 2(aic: aiv = 1:2) * alignN * dtypesize < ubsize / 2(pingpong buffer)
    uint64_t baseMMax = compileInfo_.ubSize / BIAS_TABLE_NUM / cDtypeSize_;
    // baseM align to 128
    runInfo_.baseM =
        ops::FloorAlign(std::min(ops::FloorDiv(compileInfo_.l0CSize, runInfo_.baseN * LOC_DATA_SIZE), baseMMax),
                        SMALL_SHAPE_LOWER_THRES);
    uint64_t baseKa = compileInfo_.l0ASize / DB_SIZE / aDtypeSize_ / runInfo_.baseM;
    uint64_t baseKb = compileInfo_.l0BSize / DB_SIZE / bDtypeSize_ / runInfo_.baseN;
    runInfo_.baseK = ops::FloorAlign(std::min(baseKa, baseKb), N_ALIGNED);
    // B full load && Bl1 stay in L1
    runInfo_.depthB1 = ops::CeilDiv(args_.kValue, runInfo_.baseK);
    runInfo_.stepKb = runInfo_.depthB1;
    runInfo_.depthA1 =
        ops::FloorDiv((compileInfo_.l1Size / bDtypeSize_ - runInfo_.depthB1 * runInfo_.baseN * runInfo_.baseK),
                      (runInfo_.baseM * runInfo_.baseK));
    runInfo_.depthA1 = std::min(runInfo_.depthA1, runInfo_.depthB1);
    runInfo_.stepKa = runInfo_.depthA1;
    // K较大, 设置stepK为4, depthA1为8, 该tiling较优
    if (runInfo_.depthA1 >= 8) {
        runInfo_.stepKa = 4;  // K较大, 设置stepK为4, 该tiling较优
        runInfo_.depthA1 = 8; // K较大, 设置depthA1为8, 该tiling较优
    }
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = runInfo_.baseN;
    OP_TILING_CHECK(runInfo_.depthA1 == 0, OP_LOGE(context_->GetNodeName(), "Tiling calculate failed."), return false);
    tilingEnable_.tilingEnableFixOpti = Mc2TilingEnableFixOpti::BASE_ENABLE_ALIGNOUT;
    tilingEnable_.tilingEnableSplitCore = Mc2TilingEnableSplitCore::BASE;
    tilingEnable_.tilingEnableFullLoad = Mc2TilingEnableFullLoad::BL1_FULL_LOAD;
    args_.nd2nzB = false;
    // VEC_NZ2ND当前仅支持float类型
    if (args_.aType == ge::DT_FLOAT) {
        uint64_t c0 = BLOCK_BYTE_SIZE / aDtypeSize_;
        bool isSupportVecOptOut = (args_.kValue % c0 == 0) && (args_.nValue <= 192) && !args_.nd2nzA && !args_.isATrans;
        if (isSupportVecOptOut) {
            tilingEnable_.tilingEnableFixOpti = Mc2TilingEnableFixOpti::VEC_NZ2ND_UNALIGNOUT;
        }
    }
    OP_LOGI(args_.opName, "Hit NeedSolveFixBound.");
    return true;
}

bool Mc2MatmulV3BaseTiling::DoAL1FullLoadTiling()
{
    // only support fp32, and resreict transpose attrs as network cases
    if (!compileInfo_.supportL0c2out || args_.aType != ge::DT_FLOAT || args_.isATrans || !args_.isBTrans) {
        return false;
    }
    if (ShouldUseDeterministicMultiCoreSplitKwithSmallMN() && args_.mValue >= BASIC_ALIGN_8) {
        // split k will be better
        return false;
    }
    bool kAligned = args_.kValue % (BASIC_ALIGN_512 / aDtypeSize_) == 0;
    bool isValidMNK = (args_.mValue <= BASIC_ALIGN_16 && args_.nValue > BASIC_ALIGN_16 &&
                       args_.nValue <= BASIC_ALIGN_16 * compileInfo_.aicNum && args_.kValue >= 4096 &&
                       kAligned); // 4096 is k threshold for al1_full_load
    uint64_t mAlignedValue = ops::CeilAlign(args_.mValue, BASIC_ALIGN_16);
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BLOCK_BYTE_SIZE / aDtypeSize_);
    uint64_t al1Size = mAlignedValue * kAlignedValue * aDtypeSize_;
    uint64_t bl1Size = BASIC_ALIGN_16 * BASIC_ALIGN_256 * bDtypeSize_ * DB_SIZE;
    uint64_t biasSize = args_.hasBias ? BASIC_ALIGN_16 * GetSizeByDataType(args_.biasType) : 0;
    if (!isValidMNK || al1Size > (compileInfo_.l1Size - bl1Size - biasSize)) {
        return false;
    }
    runInfo_.baseM = BASIC_ALIGN_16;
    runInfo_.baseN = BASIC_ALIGN_16;
    runInfo_.baseK = BASIC_ALIGN_256;
    runInfo_.stepN = 1;
    runInfo_.stepM = 1;
    runInfo_.stepKa = MathUtil::CeilDivision(args_.kValue, runInfo_.baseK);
    runInfo_.stepKb = 1;
    runInfo_.depthA1 = runInfo_.stepKa;
    runInfo_.depthB1 = DB_SIZE;
    runInfo_.singleCoreM = args_.mValue;
    runInfo_.singleCoreN = runInfo_.baseN;
    tilingEnable_.tilingEnableFullLoad = Mc2TilingEnableFullLoad::AL1_FULL_LOAD;
    tilingEnable_.tilingEnableSplitCore = Mc2TilingEnableSplitCore::BASE;
    runInfo_.dbL0c =
        runInfo_.baseM * runInfo_.baseN * GetSizeByDataType(ge::DT_FLOAT) * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE :
                                                                                                              1;
    runInfo_.l2Info.mTileBlock = 1;
    runInfo_.l2Info.nTileBlock = MathUtil::CeilDivision(args_.nValue, runInfo_.singleCoreN);
    runInfo_.l2Info.calOrder = 1;
    return true;
}

bool Mc2MatmulV3BaseTiling::DoBL1FullLoadTiling()
{
    uint64_t c0 = BLOCK_BYTE_SIZE / aDtypeSize_;
    uint64_t innerSizeA = args_.isATrans ? args_.mValue : args_.kValue;
    uint64_t outerSizeA = args_.isATrans ? args_.kValue : args_.mValue;
    // Update cases which can used vnchw_conv. 72368 is an experiment threshold value
    uint64_t mUpperLimit = 98048; // mValue should be no more than 98048
    uint64_t mLowerLimit = 30848; // mValue should be bigger than 30848
    bool nd2nzAUsingVnchwConv =
        (args_.aType == ge::DT_FLOAT && outerSizeA >= VNCHW_UP_THRES && innerSizeA <= c0 && innerSizeA > 1);
    bool supportNd2NzOnTheFly =
        std::find(SUPPORT_ND2NZ_GM2L0.begin(), SUPPORT_ND2NZ_GM2L0.end(), args_.nValue) != SUPPORT_ND2NZ_GM2L0.end() &&
        (!args_.isBTrans || std::find(SUPPORT_ND2NZ_GM2L0.begin(), SUPPORT_ND2NZ_GM2L0.end(),
                                      args_.kValue * aDtypeSize_) != SUPPORT_ND2NZ_GM2L0.end());
    // mValue should be 16 times more than max of k/nValue, and kValue should be no more than 256
    bool validMK = args_.mValue > 16 * std::max(args_.kValue, args_.nValue) && args_.kValue <= 256;
    uint64_t biasSize = args_.hasBias ? runInfo_.baseN * GetSizeByDataType(ge::DT_FLOAT) : 0; // 默认最高精度保证BF16
    bool bl1SizeValid = (compileInfo_.l1Size / NUM_HALF - biasSize) > args_.kValue * args_.nValue * bDtypeSize_;
    bool alignedBl1FullLoad = (supportNd2NzOnTheFly && !args_.nd2nzB);
    bool bl1FullLoadCheck = !validMK || (!alignedBl1FullLoad && !(nd2nzAUsingVnchwConv && bl1SizeValid));
    uint64_t mTailBlock = (args_.mValue / BASIC_BLOCK_SIZE_128) % 12; // 当baseM为128时，M方向上分核数量为12的尾块数量
    bool mTailBlockCheck = false;
    if (mTailBlock == 0) { // 尾块数量为0时，特殊处理
        // 当baseM为128时，M方向上分核数量为12的尾块数量加127向上取整后是否为1
        mTailBlockCheck = ((args_.mValue + 127) / BASIC_BLOCK_SIZE_128) % 12 == 1;
    } else if (mTailBlock == 6) { // 尾块数量为6时，特殊处理
        // 当baseM为128时，M方向上分核数量为12的尾块数量加127向上取整后是否为7
        mTailBlockCheck = ((args_.mValue + 127) / BASIC_BLOCK_SIZE_128) % 12 != 7;
    } else if (mTailBlock >= 1 && mTailBlock < 6) { // 尾块数量在1-6之间
        mTailBlockCheck = true;
    }
    bool bL1CoreSplitFullLoadCheck =
        ((args_.aType == ge::DT_BF16 && args_.bType == ge::DT_BF16) ||
         (args_.aType == ge::DT_FLOAT16 && args_.bType == ge::DT_FLOAT16)) &&
        (args_.mValue >= mLowerLimit) && (args_.mValue <= mUpperLimit) && (args_.nValue == 512) && // N is 512
        (args_.kValue == 512) && mTailBlockCheck && !args_.isATrans && args_.isBTrans &&           // K is 512
        (args_.aFormat == ge::FORMAT_ND) && (args_.bFormat == ge::FORMAT_ND) && args_.hasBias == 0;
    if (!bl1FullLoadCheck) {
        return DoBL1FullLoadTilingBase();
    } else if (bL1CoreSplitFullLoadCheck) {
        return DoBL1CoreSplitFullLoadTiling();
    }
    return false;
}

bool Mc2MatmulV3BaseTiling::DoBL1FullLoadTilingBase()
{
    uint64_t c0 = BLOCK_BYTE_SIZE / aDtypeSize_;
    tilingEnable_.tilingEnableFullLoad = Mc2TilingEnableFullLoad::BL1_FULL_LOAD;
    tilingEnable_.tilingEnableSplitCore = Mc2TilingEnableSplitCore::BASE;
    // fine tune tiling
    runInfo_.stepM = 1;
    runInfo_.baseN = std::min(args_.nValue, runInfo_.baseN);
    // BaseN need to do Block alignment
    runInfo_.baseN = args_.isBTrans ? ops::CeilAlign(runInfo_.baseN, N_ALIGNED) : ops::CeilAlign(runInfo_.baseN, c0);
    runInfo_.stepN = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    runInfo_.stepKb = MathUtil::CeilDivision(args_.kValue, runInfo_.baseK);
    runInfo_.stepKa = runInfo_.stepKb;
    runInfo_.depthA1 = DB_SIZE * runInfo_.stepKa;
    runInfo_.depthB1 = runInfo_.stepN * runInfo_.stepKb;
    uint64_t dtypeSize = GetSizeByDataType(args_.aType);
    uint64_t loadSize = static_cast<uint64_t>(runInfo_.baseK) *
                        (runInfo_.depthA1 * runInfo_.baseM + runInfo_.depthB1 * runInfo_.baseN) * dtypeSize;
    loadSize += args_.hasBias ? runInfo_.baseN * dtypeSize : 0;
    // Check L1 load size
    while (loadSize > compileInfo_.l1Size) {
        loadSize -= runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * dtypeSize;
        runInfo_.baseM = runInfo_.baseM / NUM_HALF;
        loadSize += runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * dtypeSize;
    }
    runInfo_.singleCoreM = DB_SIZE * runInfo_.baseM;
    runInfo_.singleCoreN = args_.nValue;
    dtypeSize = GetSizeByDataType(ge::DT_FLOAT);
    runInfo_.dbL0c = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1;
    runInfo_.l2Info.mTile = 1;
    runInfo_.l2Info.nTile = 1;
    runInfo_.l2Info.mTileBlock = MathUtil::CeilDivision(args_.mValue, runInfo_.singleCoreM);
    runInfo_.l2Info.nTileBlock = 1;
    runInfo_.l2Info.calOrder = 1;
    return true;
}

bool Mc2MatmulV3BaseTiling::DoBL1CoreSplitFullLoadTiling()
{
    tilingEnable_.tilingEnableFullLoad = Mc2TilingEnableFullLoad::BL1_FULL_LOAD;
    tilingEnable_.tilingEnableSplitCore = Mc2TilingEnableSplitCore::BASE;
    runInfo_.baseN = args_.nValue / NUM_HALF;
    runInfo_.singleCoreN = runInfo_.baseN;
    runInfo_.stepM = 1;
    runInfo_.stepN = 1;
    runInfo_.stepKb = MathUtil::CeilDivision(args_.kValue, runInfo_.baseK);
    runInfo_.stepKa = runInfo_.stepKb;
    runInfo_.depthA1 = DB_SIZE * runInfo_.stepKa;
    runInfo_.depthB1 = runInfo_.stepN * runInfo_.stepKb;
    uint64_t dtypeSize = GetSizeByDataType(args_.aType);
    runInfo_.baseM = 128; // baseM为128
    uint64_t mCore = 12;  // M方向分核数量为12
    runInfo_.singleCoreM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), runInfo_.baseM);
    dtypeSize = GetSizeByDataType(ge::DT_FLOAT);
    runInfo_.dbL0c = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1;
    runInfo_.l2Info.mTile = 1;
    runInfo_.l2Info.nTile = 1;
    runInfo_.l2Info.mTileBlock = MathUtil::CeilDivision(args_.mValue, runInfo_.singleCoreM);
    runInfo_.l2Info.nTileBlock = 1;
    runInfo_.l2Info.calOrder = 1;
    return true;
}

bool Mc2MatmulV3BaseTiling::NeedSolveFixBound()
{
    if (compileInfo_.supportL12BtBf16 || !compileInfo_.supportL0c2out) {
        return false;
    }
    if (args_.nValue >= BASIC_ALIGN_256) {
        OP_LOGD(args_.opName, "N is exceeded to 256 to deal with aligning.");
        return false;
    }
    // K越小，越容易fixpipe_bound， 限制K的大小为256
    bool fixpipeBound = (args_.kValue <= BASIC_ALIGN_256 && (args_.nValue % (BASIC_ALIGN_256 / cDtypeSize_) != 0) &&
                         ((BASIC_ALIGN_256 / cDtypeSize_) % args_.nValue != 0));
    if (!fixpipeBound) {
        OP_LOGD(args_.opName, "With calculating, this shape may not be fixpipe bound.");
        return false;
    }
    if (args_.mValue < compileInfo_.aicNum * MIN_TAIL) {
        OP_LOGD(args_.opName, "M value is too small since BL1 fullload temp needs long sequence to get better perf.");
        return false;
    }
    // avoid scalar bound when Bmatrix is too small
    uint64_t c0 = BLOCK_BYTE_SIZE / aDtypeSize_;
    bool scalarBound = (args_.nValue < c0) && (args_.kValue < c0);
    if (scalarBound) {
        OP_LOGD(args_.opName, "If B matrix is too small, it will be realized as scalar bound when N aligning to 512B.");
        return false;
    }
    bool notK256Align = args_.kValue < BASIC_ALIGN_256 / aDtypeSize_;
    bool notMte2Bound = (kA256Align_ || notK256Align) && !args_.isATrans;
    if (cDtypeSize_ == 2) { // 2 means fp16 or bf16
        if (!notMte2Bound) {
            return false;
        } else {
            args_.nd2nzA = false;
            args_.nd2nzB = false;
            return true;
        }
    }
    return true;
}

bool Mc2MatmulV3BaseTiling::DoL2CacheTiling()
{
    if (compileInfo_.supportL12BtBf16 || !compileInfo_.supportL0c2out) {
        tilingEnable_.tilingEnableSplitCore = Mc2TilingEnableSplitCore::BASE;
        OP_LOGD(args_.opName, "david not support l2cache kernel, enter base kernel.");
        return false;
    }
    uint64_t totalSize = GetTotalSize(args_.mValue, args_.kValue, args_.nValue, aDtypeSize_, bDtypeSize_);
    uint64_t mL2Split = args_.mValue;
    uint64_t nL2Split = args_.nValue;
    uint64_t mTile = 1;
    uint64_t nTile = 1;

    args_.l2Ratio = 1.0 * compileInfo_.l2Size / L2_SIZE_2;
    if (totalSize > args_.l2Ratio * 100 * MB_SIZE) { // 100M为实验数据，确保不会出现L2数据置换
        enableCache_ = (runInfo_.baseN >= runInfo_.baseM) ? CalcTile(nTile, mTile, nL2Split, mL2Split, args_.isATrans) :
                                                            CalcTile(mTile, nTile, mL2Split, nL2Split, !args_.isBTrans);
    }
    uint64_t mTileBlock = MathUtil::CeilDivision(mL2Split, runInfo_.baseM);
    uint64_t nTileBlock = MathUtil::CeilDivision(nL2Split, runInfo_.baseN);
    mTile = MathUtil::CeilDivision(args_.mValue, mTileBlock * runInfo_.baseM);
    nTile = MathUtil::CeilDivision(args_.nValue, nTileBlock * runInfo_.baseN);

    if (tilingSelect_ == Mc2TilingCalcSelect::ALL && !enableCache_ && compileInfo_.supportL0c2out) {
        constexpr uint64_t mTileBlockBase = 4UL;
        mTileBlock = std::min(mTileBlockBase, MathUtil::CeilDivision(args_.mValue, runInfo_.baseM));
        nTileBlock = compileInfo_.aicNum / mTileBlockBase;
        nTileBlock = std::min(nTileBlock, MathUtil::CeilDivision(args_.nValue, runInfo_.baseN));
        mTile = MathUtil::CeilDivision(args_.mValue, mTileBlock * runInfo_.baseM);
        nTile = MathUtil::CeilDivision(args_.nValue, nTileBlock * runInfo_.baseN);
    }

    runInfo_.l2Info.mTile = mTile;
    runInfo_.l2Info.nTile = nTile;
    runInfo_.l2Info.mTileBlock = mTileBlock;
    runInfo_.l2Info.nTileBlock = nTileBlock;
    OP_LOGI(args_.opName, "Enter L2cache tile kernel.");

    if (args_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGI(args_.opName, "Do not enter split K tile kernel when weightNz.");
        return true;
    }
    return false; // 进去单核切K逻辑
}

bool Mc2MatmulV3BaseTiling::DoL2CacheTiling310P()
{
    if (compileInfo_.supportL12BtBf16 || compileInfo_.supportL0c2out) {
        return false;
    }
    // 310P use (4, 2) or (5, 2)
    uint64_t mTileBlock = compileInfo_.aicNum / 2;
    mTileBlock = std::min(mTileBlock, MathUtil::CeilDivision(args_.mValue, runInfo_.baseM));
    uint64_t nTileBlock = std::min(2UL, MathUtil::CeilDivision(args_.nValue, runInfo_.baseN));

    uint64_t mTile = MathUtil::CeilDivision(args_.mValue, mTileBlock * runInfo_.baseM);
    uint64_t nTile = MathUtil::CeilDivision(args_.nValue, nTileBlock * runInfo_.baseN);
    runInfo_.l2Info.mTile = mTile;
    runInfo_.l2Info.nTile = nTile;
    runInfo_.l2Info.mTileBlock = mTileBlock;
    runInfo_.l2Info.nTileBlock = nTileBlock;
    OP_LOGI(args_.opName, "Enter L2cache tile kernel.");
    return true;
}

void Mc2MatmulV3BaseTiling::CalTileFactor(uint64_t &nTile)
{
    // set the factor of 20 core
    vector<uint64_t> factor = {1, 2, 4, 5, 10, 20};
    // set the factor of 24 core
    if (compileInfo_.aicNum == 24) {
        factor = {1, 2, 3, 4, 6, 8, 12, 24};
    }
    // set the factor of 32 core
    if (compileInfo_.aicNum == 32) {
        factor = {1, 2, 4, 8, 16, 32};
    }
    for (uint64_t i = 0; i < factor.size(); ++i) {
        if (nTile <= factor[i]) {
            nTile = factor[i];
            break;
        }
    }
    if (nTile > compileInfo_.aicNum) {
        uint64_t tail = nTile % compileInfo_.aicNum;
        // 2 表明一半的核要计算尾块，拖尾比较严重
        if (tail > compileInfo_.aicNum / 2) {
            nTile = (nTile + compileInfo_.aicNum - 1) / compileInfo_.aicNum;
        } else {
            nTile = nTile / compileInfo_.aicNum;
        }
        nTile = nTile * compileInfo_.aicNum;
    }
    return;
}

void Mc2MatmulV3BaseTiling::SetBasicBlockOfNK33(Mc2MatmulV3RunInfo &runInfo)
{
    // 128是3*3算法的基本块
    runInfo.baseM = BASIC_BLOCK_SIZE_128;
    runInfo.baseN = BASIC_BLOCK_SIZE_128;
    // 256 means base size is 256Byte
    runInfo.baseK = BASIC_BLOCK_K_256_BYTE / aDtypeSize_;
    runInfo.usedCoreNum = compileInfo_.aicNum;
    runInfo.depthA1 = 6; // 6 = baseM * stepKa * DB_SIZE  1 * 3 * 2
    runInfo.depthB1 = 9; // 3*3算法, 9 = baseN * stepKb 3 * 3
    runInfo.stepM = 1;   // 3*3算法, stepM需要设置为1
    runInfo.stepN = 3;   // 3*3算法, stepN需要设置为3

    runInfo.stepKa = runInfo.depthA1 / runInfo.stepM / DB_SIZE;
    runInfo.stepKb = runInfo.depthB1 / runInfo.stepN;
    runInfo.iterateOrder = ITER_COL_FIRST;
}

void Mc2MatmulV3BaseTiling::SetBasicBlockOfMK33(Mc2MatmulV3RunInfo &runInfo)
{
    // 128是3*3算法的基本块
    runInfo.baseM = BASIC_BLOCK_SIZE_128;
    runInfo.baseN = BASIC_BLOCK_SIZE_128;
    // 256 means base size is 256Byte
    runInfo.baseK = BASIC_BLOCK_K_256_BYTE / aDtypeSize_;
    runInfo.usedCoreNum = compileInfo_.aicNum;
    runInfo.depthA1 = 9; // 3*3算法, 9 = baseM * stepKa
    runInfo.depthB1 = 6; // 6 = baseN * stepKb * DB_SIZE
    runInfo.stepM = 3;   // 3*3算法, stepM需要设置为3
    runInfo.stepN = 1;   // 3*3算法, stepN需要设置为1

    runInfo.stepKa = runInfo.depthA1 / runInfo.stepM;
    runInfo.stepKb = runInfo.depthB1 / runInfo.stepN / DB_SIZE;
    runInfo.iterateOrder = ITER_ROW_FIRST;
}

void Mc2MatmulV3BaseTiling::SetBasicBlockOf24(Mc2MatmulV3RunInfo &runInfo, const uint64_t &mTile,
                                              const uint64_t &nTile) const
{
    // 2*4算法
    if (mTile * nTile < compileInfo_.aicNum) {
        runInfo.depthA1 = 8; // 2*4算法, 8 = baseM * stepKa
        runInfo.depthB1 = 8; // 2*4算法, 8 = baseN * stepKb * DB_SIZE
        runInfo.stepM = 2;   // 2*4算法, stepM = 2
        runInfo.stepN = 1;
        runInfo.stepKa = runInfo.depthA1 / runInfo.stepM;
        runInfo.stepKb = runInfo.depthB1 / DB_SIZE;
    }
    // 384 = 3 * 128, M和N较小的场景
    if (args_.mValue < 384 || args_.nValue < 384) {
        runInfo.depthA1 = 8; // 2*4算法, 8 = baseM * stepKa * DB_SIZE
        runInfo.depthB1 = 8; // 2*4算法, 8 = baseN * stepKb
        runInfo.stepM = 1;
        runInfo.stepKa = 4; // M较小，放大k为4
        runInfo.stepN = 1;
        runInfo.stepKb = 4; // N较小，放大k为4
    }
}

void Mc2MatmulV3BaseTiling::DoIncreTiling()
{
    if (!compileInfo_.supportL0c2out) {
        OP_LOGI(args_.opName, "Ascend310P currently doesn't support IncreShape.");
        return;
    }
    if (args_.mValue > 128) { // 128: 增量场景要求M≤128
        OP_LOGI(args_.opName, "M > 128, doesn't belong to IncreShape.");
        return;
    }
    if (GetMixNd2nzType() != Mc2MixNd2NzType::NO_ND2NZ) { // 对应要求 N和K < 65536，K 256B对齐
        OP_LOGI(args_.opName, "IncreShape shouldn't be MixNd2nz, please check shape size.");
        return;
    }
    if (((args_.aType != ge::DT_FLOAT16) && (args_.aType != ge::DT_BF16)) ||
        ((args_.bType != ge::DT_FLOAT16) && (args_.bType != ge::DT_BF16)) ||
        ((args_.cType != ge::DT_FLOAT16) && (args_.cType != ge::DT_BF16))) {
        OP_LOGI(args_.opName, "Data type should be fp16 or bf16 in IncreShape, please check.");
        return;
    }
    if ((args_.aFormat != ge::FORMAT_ND) || (args_.bFormat != ge::FORMAT_ND) || (args_.outFormat != ge::FORMAT_ND)) {
        OP_LOGI(args_.opName, "Data format non-ND, not support in IncreShape.");
        return;
    }
    if (args_.isATrans || !args_.isBTrans) {
        OP_LOGI(args_.opName, "A is Trans or B not Trans, not support in IncreShape.");
        return;
    }
    if (args_.hasBias) {
        OP_LOGI(args_.opName, "Bias not support in IncreShape.");
        return;
    }
    tilingEnable_.tilingEnableSplitCore = Mc2TilingEnableSplitCore::BASE;
    tilingEnable_.tilingEnableFullLoad = Mc2TilingEnableFullLoad::BASE;
    tilingEnable_.tilingEnableFixOpti = Mc2TilingEnableFixOpti::BASE;
    GetV2Tiling();
    OP_LOGD(args_.opName,
            "Incre convert tbe-tiling: CoreNum(%lu) singleMNK(%lu %lu %lu) baseMNK(%lu %lu %lu) depthAB(%lu %lu) "
            "stepMNKaKb(%lu %lu %lu %lu) dbL0c(%lu)",
            runInfo_.usedCoreNum, runInfo_.singleCoreM, runInfo_.singleCoreN, runInfo_.singleCoreK, runInfo_.baseM,
            runInfo_.baseN, runInfo_.baseK, runInfo_.depthA1, runInfo_.depthB1, runInfo_.stepM, runInfo_.stepN,
            runInfo_.stepKa, runInfo_.stepKb, runInfo_.dbL0c);
}

Mc2MixNd2NzType Mc2MatmulV3BaseTiling::GetMixNd2nzType() // check different platform
{
    if (compileInfo_.supportL12BtBf16) {
        return Mc2MixNd2NzType::NO_ND2NZ; // current not support mix kernel
    }
    if (compileInfo_.supportL0c2out) {
        if (args_.nd2nzA && !args_.nd2nzB && args_.unAlignProcessType > 0 &&
            tilingEnable_.tilingEnableFullLoad == Mc2TilingEnableFullLoad::BL1_FULL_LOAD &&
            tilingEnable_.tilingEnableSplitCore == Mc2TilingEnableSplitCore::BASE &&
            tilingEnable_.tilingEnableFixOpti == Mc2TilingEnableFixOpti::BASE) {
            return Mc2MixNd2NzType::V_PARALELL_ND2NZ;
        }
        if (args_.nd2nzA || args_.nd2nzB) {
            return Mc2MixNd2NzType::V_HEAD_ND2NZ;
        }
    }
    return Mc2MixNd2NzType::NO_ND2NZ;
}

bool Mc2MatmulV3BaseTiling::IsSupportSingleCoreSplitSmallK(uint64_t xDim, uint64_t yDim) const
{
    // 条件0： A,B 矩阵数据仅支持 bf16/fp16, 矩阵格式支持 (ND,ND)
    bool isDTypeFormatSupport = args_.aFormat == ge::FORMAT_ND && args_.bFormat == ge::FORMAT_ND &&
                                (args_.aType == ge::DT_FLOAT16 || args_.aType == ge::DT_BF16) &&
                                (args_.bType == ge::DT_FLOAT16 || args_.bType == ge::DT_BF16);
    // 条件1： M,N 被128整除，K=1536 （对应DeepSeekV3 的Prefill阶段）
    bool isSmallKwithLargeMN =
        (args_.kValue == SINGLE_CORE_SPLIT_SMALL_K) && (args_.mValue % ALIGN_128 == 0 && args_.nValue % ALIGN_128 == 0);
    // 条件2： N>>M 且 M = 384 或 M>>N 且 N = 384;
    bool isLargeXSmallY = (xDim >= SINGLE_CORE_SPLIT_LARGE_MN && yDim == SINGLE_CORE_SPLIT_SMALL_MN);
    // 条件3： M * N <= 0.65 * l2Size: fixpipe 拷出矩阵占L2缓存的比例过大时，会有读写冲突，导致性能出现劣化
    constexpr double cacheRatio = 0.65;
    uint64_t l2CacheLimitation = static_cast<uint64_t>(compileInfo_.l2Size * cacheRatio) / DATA_SIZE_FP32;
    bool isL2Enough = (args_.nValue * args_.mValue <= l2CacheLimitation);
    return isDTypeFormatSupport && isSmallKwithLargeMN && isLargeXSmallY && isL2Enough;
}

bool Mc2MatmulV3BaseTiling::IsSupportSingleCoreSplitK() const
{
    bool isMKNsmallK = IsSupportSingleCoreSplitSmallK(args_.nValue, args_.mValue);
    bool isNKMsmallK = IsSupportSingleCoreSplitSmallK(args_.mValue, args_.nValue);
    if (isMKNsmallK || isNKMsmallK) {
        OP_LOGI(args_.opName, "Hit mat_mul_v3 single-core-splitk (K=1536) MKN/NKM channel.\n");
        return true;
    }
    // n非对齐为fixpipe bound, 走单核切K，由于存在串行的前vector处理，非对齐场景可能性能更差，维持原4M限制不变
    if (args_.isHf32 && !n256Align_ && args_.mValue * args_.nValue < DETER_THRES_OUT_SIZE * MB_SIZE) {
        return false;
    }
    if (args_.kValue >= SPLIT_K_THRES) {
        OP_LOGD(args_.opName, "K >= SPLIT_K_THRES, enable SingleCoreSplitK.");
        return true;
    }
    constexpr uint64_t splitKThres = 1024;                        // 1024: 切K阈值
    constexpr uint64_t splitBaseThres = BASIC_BLOCK_SIZE_128 * 3; // 3*128: 33算法 step=3
    constexpr uint64_t splitKCubeThres = 5;                       // 5: cube dound阈值
    bool isMKNLargeEnough =
        ((args_.mValue * args_.kValue >= splitKCubeThres * splitBaseThres * splitBaseThres) &&
         (args_.nValue >= splitKThres) && (args_.mValue >= splitBaseThres) && (args_.kValue >= splitBaseThres) &&
         (args_.mValue * args_.nValue >= splitKThres * splitBaseThres * compileInfo_.aicNum));
    if (!enableCache_ && isMKNLargeEnough) {
        OP_LOGD(args_.opName, "L2cache tile fail, m/k/n is enough to splitk kernel.");
        return true;
    }
    constexpr uint64_t mataThread = 8192UL;
    constexpr uint64_t tlbThread = 6144UL;
    // (k, m)(k, n) K在外轴内轴很大MTE2会有tlb问题, 内轴%8192==0,存在一定的mata问题
    bool tlbAndMataFlag = args_.isATrans && !args_.isBTrans && n256Align_ && args_.kValue >= 11000UL &&
                          ((args_.mValue % mataThread == 0 && args_.nValue >= tlbThread) ||
                           (args_.nValue % mataThread == 0 && args_.mValue >= tlbThread));
    if (tlbAndMataFlag) {
        OP_LOGI(args_.opName, "tlb mata issue, enter splitk kernel.");
        return true;
    }

    // (m, k)(n, k),k为16384的倍数，则mata冲突特别严重，走单核切K，减少MTE2搬运量
    // 限制为fp16、bf16、B2场景
    constexpr uint64_t kThread = 16384;
    constexpr uint64_t mnMinThread = 1280;
    constexpr uint64_t mnMaxThread = 8192;
    bool mataFlag = !args_.isATrans && args_.isBTrans && args_.kValue % kThread == 0 &&
                    (args_.mValue >= mnMinThread && args_.mValue <= mnMaxThread) &&
                    (args_.nValue >= mnMinThread && args_.nValue <= mnMaxThread) && args_.aFormat == ge::FORMAT_ND &&
                    args_.bFormat == ge::FORMAT_ND && (args_.aType == ge::DT_FLOAT16 || args_.aType == ge::DT_BF16) &&
                    compileInfo_.aicNum == MIX_BLOCK_NUM;
    if (mataFlag && (!args_.nd2nzB)) {
        OP_LOGD(args_.opName, "The shape is meeting the criteria for mata conflicts.");
        return true;
    }
    return false;
}

void Mc2MatmulV3BaseTiling::IsGmToL1ByShape()
{
    bool disableMixNd2nz = (GetMixNd2nzType() == Mc2MixNd2NzType::NO_ND2NZ);
    uint64_t aMatrix;
    uint64_t bMatrix;
    if (disableMixNd2nz) {
        aMatrix = args_.isATrans ? args_.mValue : args_.kValue;
        bMatrix = args_.isBTrans ? args_.kValue : args_.nValue;
    } else {
        aMatrix = args_.isATrans ? args_.kValue : args_.mValue;
        bMatrix = args_.isBTrans ? args_.nValue : args_.kValue;
    }
    if (!args_.hasBias && aMatrix <= SHAPE_LIMIT && bMatrix <= SHAPE_LIMIT) {
        tilingData_.matmulTiling.shareL1Size = static_cast<uint32_t>(0);
        if (runInfo_.stepM == STEP_NUM_3 && args_.hasBias) {
            tilingData_.matmulTiling.shareL1Size = static_cast<uint32_t>(L1_BIAS_SIZE);
        }
        tilingEnable_.tilingEnableSplitCore = Mc2TilingEnableSplitCore::SINGLE_CORE_SPLIT_K_GM_TO_L1;
        tilingData_.matmulTiling.shareL0CSize = static_cast<uint32_t>(0);
    }
}

bool Mc2MatmulV3BaseTiling::CheckSingleTilingOk(Mc2MatmulV3RunInfo &tmpRunInfo)
{
    // 判断是否走NKM模板
    bool isNKM = args_.aType == ge::DT_FLOAT && args_.nValue <= NMK_N_THERS && args_.mValue >= NMK_M_THERS &&
                 args_.kValue >= SPLIT_K_THRES && args_.kValue < ND2NZ_ON_THE_FLY_LIMIT && !args_.isATrans &&
                 args_.isBTrans;
    // 判断是否支持K=1536场景下的单核切K
    bool isMKNsmallK = IsSupportSingleCoreSplitSmallK(args_.nValue, args_.mValue);
    bool isNKMsmallK = IsSupportSingleCoreSplitSmallK(args_.mValue, args_.nValue);
    // MKN模板下，需要规避的case
    if (CheckSingleCoreSplitKEdgeCases(tmpRunInfo, isNKM, isMKNsmallK, isNKMsmallK)) {
        return false;
    }
    // singleM切分在256到384之间，使用33算法，但若时fp32的输出非对齐场景，singleK小容易fixpipe_bound
    if (isMKNsmallK || (tmpRunInfo.singleCoreM > SINGLE_CORE_M_24 && tmpRunInfo.singleCoreM <= MULTI_CORE_SINGLE_K &&
                        !(args_.aType == ge::DT_FLOAT && !n256Align_))) {
        SetBasicBlockOfMK33(tmpRunInfo);
    } else if (isNKMsmallK) {
        SetBasicBlockOfNK33(tmpRunInfo);
    }
    tmpRunInfo.singleCoreK = tmpRunInfo.stepKa * tmpRunInfo.baseK;
    tmpRunInfo.dbL0c = DB_SIZE;
    tilingEnable_.tilingEnableSplitCore = Mc2TilingEnableSplitCore::SINGLE_CORE_SPLIT_K;
    if (isNKM || isNKMsmallK) { // 走51：NKM模板
        tilingEnable_.tilingEnableSplitCore = Mc2TilingEnableSplitCore::SINGLE_CORE_NKM_SPLIT_K;
    } else if (args_.aType != ge::DT_FLOAT && args_.nValue % ALIGN_128 == 0 && !isMKNsmallK) {
        IsGmToL1ByShape(); // 走61：GM_to_L1分核模板;当前 isMKNsmallK 用例走21模板性能优于61
    }
    tilingEnable_.tilingEnableFullLoad = Mc2TilingEnableFullLoad::BASE;
    tilingEnable_.tilingEnableFixOpti = Mc2TilingEnableFixOpti::BASE;
    runInfo_ = tmpRunInfo;
    runInfo_.needUpdate = true;
    runInfo_.l2Info.calOrder = isNKMsmallK ? 1 : 0; // 0: 默认行优先； 1: 列优先
    OP_LOGI(args_.opName, "Mc2MatMulV3 tiling enable state is SingleCoreSplitK");
    return true;
}

bool Mc2MatmulV3BaseTiling::CheckSingleCoreSplitKEdgeCases(const Mc2MatmulV3RunInfo &tmpRunInfo, bool isNKM,
                                                           bool isMKNsmallK, bool isNKMsmallK)
{
    if (!isNKM && !isMKNsmallK && !isNKMsmallK) {
        // singlen过小场景, A的搬运串行占比过高, N小于512， 一般mac利用率较低
        if (tmpRunInfo.singleCoreN < 512UL) {
            OP_LOGD(args_.opName, "singleN too small.");
            return true;
        }
        // 单核切K负载不均衡
        float32_t avgRatio =
            (static_cast<float32_t>(args_.mValue * args_.nValue) /
             static_cast<float32_t>(tmpRunInfo.singleCoreN * tmpRunInfo.singleCoreM * compileInfo_.aicNum));
        if (avgRatio < 0.7f) { // 0.7是经验值
            OP_LOGD(args_.opName, "singleK avg_ratio small than 0.7.");
            return true;
        }
        // singleCoreN 小于等于640 mac利用率小于0.6，多核切K场景下N在896到2048之间mac利用率一般大于0.6
        if (n256Align_ && args_.nValue >= 896UL && args_.nValue <= 2048UL &&
            (tmpRunInfo.singleCoreN <= 640UL || avgRatio < 0.85f)) {
            OP_LOGD(args_.opName, "multiK may better than singleK");
            return true;
        }
    }
    return false;
}

bool Mc2MatmulV3BaseTiling::DoSingleCoreSplitKTiling()
{
    if (!compileInfo_.supportL0c2out || compileInfo_.supportL12BtBf16 || !IsSupportSingleCoreSplitK()) {
        OP_LOGI(args_.opName, "Mc2MatMulV3 tiling not support SingleCoreSplitK.");
        return false;
    }
    Mc2MatmulV3RunInfo tmpRunInfo = runInfo_;
    uint64_t nL2TileLength = l2TileLength_; // L2缓存为192MB,切K时N为3072保证计算恰好被L2缓存容下
    SetBasicBlockOfMK33(tmpRunInfo);
    uint64_t mTile = MathUtil::CeilDivision(args_.mValue, tmpRunInfo.stepM * tmpRunInfo.baseM);
    uint64_t nTile = MathUtil::CeilDivision(args_.nValue, nL2TileLength);
    if (!args_.hasBias) {
        SetBasicBlockOf24(tmpRunInfo, mTile, nTile);
    }
    uint64_t mAlignLength = (args_.isATrans ? ALIGN_INNER : ALIGN_OUTER) / (aDtypeSize_);
    uint64_t nAlignLength = ALIGN_INNER / aDtypeSize_;
    mTile = MathUtil::CeilDivision(args_.mValue, tmpRunInfo.stepM * tmpRunInfo.baseM);
    CalTileFactor(nTile); // nTile向上靠近aic_num的因子， 如ntile=5，aic_num=24，调整ntile为6
    if (mTile * nTile >= compileInfo_.aicNum) {
        if (nTile == 0UL) {
            mTile = 1UL;
        } else {
            mTile = std::max(compileInfo_.aicNum / nTile, 1UL);
        }
        mTile = std::max(compileInfo_.aicNum / nTile, 1UL);
        tmpRunInfo.singleCoreM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mTile), mAlignLength);
        tmpRunInfo.singleCoreM = std::min(tmpRunInfo.singleCoreM, args_.mValue);
        tmpRunInfo.singleCoreN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nTile), nAlignLength);
        tmpRunInfo.usedCoreNum = std::min(MathUtil::CeilDivision(args_.mValue, tmpRunInfo.singleCoreM) *
                                              MathUtil::CeilDivision(args_.nValue, tmpRunInfo.singleCoreN),
                                          compileInfo_.aicNum);
        if (tmpRunInfo.usedCoreNum == compileInfo_.aicNum) {
            return CheckSingleTilingOk(tmpRunInfo);
        }
    }
    if (mTile == 0) {
        nTile = 1UL;
    } else {
        nTile = std::max(compileInfo_.aicNum / mTile, 1UL);
    }
    uint64_t totalCnt = 0UL;
    uint64_t singleCoreN = args_.nValue;
    constexpr uint64_t singleCoreNThreshold = 1024UL;
    // 由于singleM, singleN做了align, 实际核数可能不是24，ntile调整和核数一致, 满核, N已经很小，停止调整mtile, ntile
    while (nTile <= compileInfo_.aicNum && totalCnt < compileInfo_.aicNum && singleCoreN >= singleCoreNThreshold) {
        CalTileFactor(nTile);
        mTile = compileInfo_.aicNum / nTile;
        singleCoreN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nTile), nAlignLength);
        uint64_t singleCoreM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mTile), mAlignLength);
        uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, singleCoreM);
        uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, singleCoreN);
        if (mCnt * nCnt > totalCnt) {
            totalCnt = mCnt * nCnt;
            tmpRunInfo.usedCoreNum = totalCnt;
            tmpRunInfo.singleCoreM = std::min(singleCoreM, args_.mValue);
            tmpRunInfo.singleCoreN = std::min(singleCoreN, args_.nValue);
        }
        nTile++;
    }
    return CheckSingleTilingOk(tmpRunInfo);
}

bool Mc2MatmulV3BaseTiling::SupportMultiSplitK() const
{
    // 判断是否为支持多核切k的芯片
    if (!compileInfo_.supportL12BtBf16 && !compileInfo_.supportL0c2out) {
        return false;
    }
    bool kIsEnoughMultiCore = args_.kValue >= compileInfo_.aicNum * MULTI_CORE_SINGLE_K;
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, BASIC_BLOCK_SIZE_128);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_128);
    bool mNIsNotEnoughCore = (mCnt * nCnt < static_cast<int64_t>(compileInfo_.aicNum) / NUMBER_TWO);
    // 判断是否时M/N不能开满多核场景，M/N轴在外轴的场景切m/n不影响MTE2搬运效率，M/N可以切小保证多核能开启，属于cube_bound场景
    bool splitKScene = kIsEnoughMultiCore && mNIsNotEnoughCore && !(!args_.isATrans && args_.isBTrans);
    if (splitKScene) {
        return true;
    }
    // 即使M/N在外轴，M/N过小时也无法分核，应当使用多核切K。
    // M N <= 64, K >= 6144 为经验值
    // M/N大时可以在M/M方向分核，可以不使能多核切K
    // K太小时，多核切K作为mix Kernel，头开销（启动开销 + tiling拷贝开销）和尾开销 （vec累加开销）相较于
    // 基础模板是劣势，不需要选择多核切K模板
    if (ShouldUseDeterministicMultiCoreSplitKwithSmallMN()) {
        OP_LOGD(args_.opName, "MultiCore splitK is supported");
        return true;
    }

    if (args_.kValue >= MIN_SPLITK_K && args_.kValue <= MAX_SPLITK_K && args_.mValue <= M_THRESHOLD &&
        args_.nValue <= N_THRESHOLD && args_.isATrans && !args_.isBTrans) {
        OP_LOGD(args_.opName, "MultiCore splitK is supported");
        return true;
    }
    // 判断k轴是否大于27392，不大于跳出
    if (args_.kValue < SPLIT_K_THRES) {
        return false;
    }
    bool innerAlignA = args_.isATrans ? m256Align_ : kA256Align_;
    bool innerAlignB = args_.isBTrans ? kB256Align_ : n256Align_;
    bool mte2BoundBaseFp32 = (!innerAlignA || !innerAlignB) && args_.aType == ge::DT_FLOAT;
    // m, n, 均大于512，且N对齐，核数足够分
    if (mCnt >= 4UL && nCnt >= 4UL && !n256Align_ && (!mte2BoundBaseFp32) &&
        static_cast<float32_t>(mCnt * nCnt) / ops::CeilAlign(mCnt * nCnt, compileInfo_.aicNum) > 0.8f) {
        OP_LOGD(args_.opName, "MultiK will fixpipebound, use L2split");
        return false;
    }
    return true;
}

bool Mc2MatmulV3BaseTiling::IsNkOrder()
{
    // M > N && N方向需要对齐32B
    aDtypeSize_ = GetSizeByDataType(args_.aType);
    bDtypeSize_ = GetSizeByDataType(args_.bType);
    if (args_.mValue <= args_.nValue || !Is32BAlign(args_.nValue, bDtypeSize_)) {
        return false;
    }
    // M限制为>=128
    constexpr uint64_t mMinValue = 128;
    if (args_.mValue < mMinValue) {
        return false;
    }
    // 在N对齐256B并且M非对齐256B的场景下，由于M非对齐NK的性能优化有限，且N对齐256B时原方案搬运性能较好
    if (Is256BAlign(args_.nValue, bDtypeSize_) && !Is256BAlign(args_.mValue, aDtypeSize_)) {
        return false;
    }
    // M非对齐256B && N*dtype <= 256B && N*dtype %32B == 0时走回MK
    if (!Is256BAlign(args_.mValue, aDtypeSize_) && args_.nValue * bDtypeSize_ <= N_THRESHOLD &&
        Is32BAlign(args_.nValue, bDtypeSize_)) {
        return false;
    }
    // N=16时，M超过2048则走回MK
    constexpr uint64_t nMinValue = 16;
    constexpr uint64_t mMaxValue = 2048;
    if (args_.mValue >= mMaxValue && args_.nValue == nMinValue) {
        return false;
    }
    return true;
}

void Mc2MatmulV3BaseTiling::GetMoreMultiCoreSplitKArgs()
{
    // 内轴较大时，串行的mixnd2nz相较随路nd2nz场景存在劣化，内轴>512B暂不走mixnd2nz(仅多核切K模板)
    uint64_t inSizeA = args_.isATrans ? args_.mValue : args_.kValue;
    uint64_t inSizeB = args_.isBTrans ? args_.kValue : args_.nValue;
    uint64_t outterSizeA = args_.isATrans ? args_.kValue : args_.mValue;
    uint64_t outterSizeB = args_.isBTrans ? args_.nValue : args_.kValue;

    // 内轴大于512B时随路方案获取更大的性能收益
    if ((inSizeA >= ND2NZ_THRES / aDtypeSize_) && (inSizeA <= ND2NZ_ON_THE_FLY_LIMIT)) {
        args_.nd2nzA = false;
    }
    if ((inSizeB >= ND2NZ_THRES / bDtypeSize_) && (inSizeB <= ND2NZ_ON_THE_FLY_LIMIT)) {
        args_.nd2nzB = false;
    }

    // 如果M、N > 128，并且非16位对齐，使用nz2ndC搬出
    if (args_.nValue > BLOCK && args_.mValue > BLOCK && !Is16Align(args_.nValue)) {
        tilingEnable_.tilingEnableFixOpti = Mc2TilingEnableFixOpti::VEC_NZ2ND_UNALIGNOUT;
    }

    uint64_t c0Size = GetSizeC0(aDtypeSize_);
    // 如果内轴 = c0Size， 外轴为16倍数，ND、NZ排布相同，可以直接当成NZ传入
    if (inSizeA == c0Size && Is16Align(outterSizeA)) {
        args_.isNzA = true;
    }

    if (inSizeB == c0Size && Is16Align(outterSizeB)) {
        args_.isNzB = true;
    }
}

bool Mc2MatmulV3BaseTiling::ShouldUseDeterministicMultiCoreSplitKwithSmallMN() const
{
    if (args_.aType != ge::DT_FLOAT || args_.isHf32) {
        return false;
    }
    return args_.mValue <= BASIC_BLOCK_SIZE_64 && args_.nValue <= BASIC_BLOCK_SIZE_64 &&
           args_.kValue >= MIN_SPLITK_MN64;
}

void Mc2MatmulV3BaseTiling::OptCoreNumsDeterministicMultiCoreSplitK()
{
    if (ShouldUseDeterministicMultiCoreSplitKwithSmallMN()) {
        // 经验值
        uint64_t kDeltaCore = ops::FloorDiv(args_.kValue - MIN_SPLITK_MN64, DELTAK_PER_CORE);
        uint64_t mDeltaCore = ops::CeilDiv(args_.mValue, BASIC_BLOCK_SIZE_16);
        uint64_t nDeltaCore = ops::CeilDiv(args_.nValue, BASIC_BLOCK_SIZE_16);
        runInfo_.usedCoreNum = std::min(compileInfo_.aicNum, MIN_CORE_SPLITK + kDeltaCore + mDeltaCore + nDeltaCore);
        OP_LOGD(args_.opName, "MultiCore splitK opt core nums to: %lu", runInfo_.usedCoreNum);
    }
}

bool Mc2MatmulV3BaseTiling::DoDeterministicMultiCoreSplitKTiling()
{
    if (compileInfo_.supportL12BtBf16 || !SupportMultiSplitK()) {
        return false;
    }
    tilingEnable_.tilingEnableSplitCore = Mc2TilingEnableSplitCore::DETERMINISTIC_SPLIT_K;
    tilingEnable_.tilingEnableFullLoad = Mc2TilingEnableFullLoad::BASE;
    tilingEnable_.tilingEnableFixOpti = Mc2TilingEnableFixOpti::BASE;
    OP_LOGI(args_.opName, "Mc2MatMulV3 tiling enable state is DeterministicMultiCoreSplitK.");

    // MK耗时：(MK + NK) / 1.6 + NK * (M / ML1 - 1) / 7.8
    // NK耗时：(MK + NK) / 1.6 + MK * (N / NL1 - 1) / 7.8
    //  MK耗时和NK耗时的比较合并为M和N的比较
    uint64_t L2_SIZE_70_pct = compileInfo_.l2Size * 7 / 10;
    if (IsNkOrder()) {
        SetBasicBlockOfNK33(runInfo_);
        // 多核切K在k完成分核，m轴不分核, n轴做外层循环进行pingpong
        runInfo_.singleCoreM = args_.mValue;
        // singleCoreN * singleCoreK表示基本块大小，一次IterateALL完成singleCoreM * singleCoreN的计算
        runInfo_.singleCoreN = runInfo_.stepN * runInfo_.baseN;
        runInfo_.singleCoreK = runInfo_.stepKb * runInfo_.baseK;
        // update L2 cache split :(runInfo_.singleCoreM * runInfo_.singleCoreK * aDtypeSize_ + runInfo_.singleCoreK *
        // runInfo_.singleCoreN * bDtypeSize_ + runInfo_.singleCoreM * runInfo_.singleCoreN * DATA_SIZE_FP32)
        // * compileInfo_.aicNum < compileInfo_.l2Size;
        uint64_t mL2Split =
            (L2_SIZE_70_pct / compileInfo_.aicNum -
             runInfo_.singleCoreK * std::min(runInfo_.singleCoreN, args_.nValue) * bDtypeSize_) /
            (runInfo_.singleCoreK * aDtypeSize_ + std::min(runInfo_.singleCoreN, args_.nValue) * DATA_SIZE_FP32);
        if (args_.mValue > mL2Split) {
            // 需要切分L2
            mL2Split = ops::CeilAlign(mL2Split, BLOCK);
            uint64_t mCount = ops::CeilDiv(args_.mValue, mL2Split);
            runInfo_.singleCoreM = ops::CeilAlign(ops::CeilDiv(args_.mValue, mCount), BLOCK);
        }
    } else {
        SetBasicBlockOfMK33(runInfo_);
        // 多核切K在k完成分核，n轴不分核, m轴做外层循环进行pingpong
        runInfo_.singleCoreN =
            args_.nValue; // Is256BAlign(args_.nValue, DATA_SIZE_FP32) ? args_.nValue : ops::CeilAlign(args_.nValue *
                          // DATA_SIZE_FP32, ALIGN_BYTE) / DATA_SIZE_FP32;
        // singleCoreM * singleCoreK表示基本块大小，一次IterateALL完成singleCoreM * singleCoreN的计算
        runInfo_.singleCoreM = runInfo_.stepM * runInfo_.baseM;
        runInfo_.singleCoreK = runInfo_.stepKa * runInfo_.baseK;
        uint64_t nL2Split =
            (L2_SIZE_70_pct / compileInfo_.aicNum -
             runInfo_.singleCoreK * std::min(runInfo_.singleCoreM, args_.mValue) * aDtypeSize_) /
            (runInfo_.singleCoreK * bDtypeSize_ + std::min(runInfo_.singleCoreM, args_.mValue) * DATA_SIZE_FP32);
        if (args_.nValue > nL2Split) {
            // 需要切分L2
            nL2Split = ops::CeilAlign(nL2Split, BLOCK);
            uint64_t nCount = ops::CeilDiv(args_.nValue, nL2Split);
            runInfo_.singleCoreN = ops::CeilAlign(ops::CeilDiv(args_.nValue, nCount), BLOCK);
        }
    }
    uint64_t kCount = MathUtil::CeilDivision(args_.kValue, runInfo_.singleCoreK);
    runInfo_.usedCoreNum = std::min(kCount, compileInfo_.aicNum);
    runInfo_.dbL0c = DB_SIZE;

    GetMoreMultiCoreSplitKArgs();
    OptCoreNumsDeterministicMultiCoreSplitK();
    return true;
}

bool Mc2MatmulV3BaseTiling::CheckMMTilingDataIsVaild()
{
    return (CheckNumberIsValid(runInfo_.usedCoreNum, args_.opName, "runInfo_.usedCoreNum") ||
            CheckNumberIsValid2(runInfo_.singleCoreM, args_.opName, "runInfo_.singleCoreM") ||
            CheckNumberIsValid2(runInfo_.singleCoreN, args_.opName, "runInfo_.singleCoreN") ||
            CheckNumberIsValid2(runInfo_.singleCoreK, args_.opName, "runInfo_.singleCoreK") ||
            CheckNumberIsValid2(runInfo_.baseM, args_.opName, "runInfo_.baseM") ||
            CheckNumberIsValid2(runInfo_.baseN, args_.opName, "runInfo_.baseN") ||
            CheckNumberIsValid2(runInfo_.baseK, args_.opName, "runInfo_.baseK") ||
            CheckNumberIsValid(runInfo_.depthA1, args_.opName, "runInfo_.depthA1") ||
            CheckNumberIsValid(runInfo_.depthB1, args_.opName, "runInfo_.depthB1") ||
            CheckNumberIsValid(runInfo_.stepM, args_.opName, "runInfo_.baseK") ||
            CheckNumberIsValid(runInfo_.stepN, args_.opName, "runInfo_.stepN") ||
            CheckNumberIsValid(runInfo_.stepKa, args_.opName, "runInfo_.baseK") ||
            CheckNumberIsValid(runInfo_.stepKb, args_.opName, "runInfo_.stepKb") ||
            CheckNumberIsValid(runInfo_.iterateOrder, args_.opName, "runInfo_.iterateOrder") ||
            CheckNumberIsValid(runInfo_.dbL0c, args_.opName, "runInfo_.dbL0c") ||
            CheckNumberIsValid(runInfo_.l2Info.mTile, args_.opName, "runInfo_.l2Info.mTile") ||
            CheckNumberIsValid(runInfo_.l2Info.nTile, args_.opName, "runInfo_.l2Info.nTile") ||
            CheckNumberIsValid(runInfo_.l2Info.mTileBlock, args_.opName, "runInfo_.l2Info.mTileBlock") ||
            CheckNumberIsValid(runInfo_.l2Info.nTileBlock, args_.opName, "runInfo_.l2Info.nTileBlock"));
}

ge::graphStatus Mc2MatmulV3BaseTiling::DoLibApiTiling()
{
    SetRunInfo();
    if (runInfo_.needUpdate) {
        if (CheckMMTilingDataIsVaild()) {
            return ge::FAILED;
        }
        tilingData_.matmulTiling.usedCoreNum = static_cast<uint32_t>(runInfo_.usedCoreNum);
        tilingData_.matmulTiling.singleCoreM = static_cast<uint32_t>(runInfo_.singleCoreM);
        tilingData_.matmulTiling.singleCoreN = static_cast<uint32_t>(runInfo_.singleCoreN);
        tilingData_.matmulTiling.singleCoreK = static_cast<uint32_t>(runInfo_.singleCoreK);
        tilingData_.matmulTiling.baseM = static_cast<uint32_t>(runInfo_.baseM);
        tilingData_.matmulTiling.baseN = static_cast<uint32_t>(runInfo_.baseN);
        tilingData_.matmulTiling.baseK = static_cast<uint32_t>(runInfo_.baseK);
        tilingData_.matmulTiling.depthA1 = static_cast<uint32_t>(runInfo_.depthA1);
        tilingData_.matmulTiling.depthB1 = static_cast<uint32_t>(runInfo_.depthB1);
        tilingData_.matmulTiling.stepM = static_cast<uint32_t>(runInfo_.stepM);
        tilingData_.matmulTiling.stepN = static_cast<uint32_t>(runInfo_.stepN);
        tilingData_.matmulTiling.stepKa = static_cast<uint32_t>(runInfo_.stepKa);
        tilingData_.matmulTiling.stepKb = static_cast<uint32_t>(runInfo_.stepKb);
        tilingData_.matmulTiling.iterateOrder = static_cast<uint32_t>(runInfo_.iterateOrder);
        tilingData_.matmulTiling.dbL0C = static_cast<uint32_t>(runInfo_.dbL0c);
        if (!compileInfo_.supportL0c2out) {
            tilingData_.matmulTiling.transLength = static_cast<uint32_t>(L0C_SIZE_256_KB / NUM_HALF);
            tilingData_.matmulTiling.shareUbSize = 0;
        }
        tilingData_.tileL2cacheTiling.mTileCntL2 = static_cast<uint32_t>(runInfo_.l2Info.mTile);
        tilingData_.tileL2cacheTiling.nTileCntL2 = static_cast<uint32_t>(runInfo_.l2Info.nTile);
        tilingData_.tileL2cacheTiling.mTileBlock = static_cast<uint32_t>(runInfo_.l2Info.mTileBlock);
        tilingData_.tileL2cacheTiling.nTileBlock = static_cast<uint32_t>(runInfo_.l2Info.nTileBlock);
        tilingData_.tileL2cacheTiling.calOrder = static_cast<uint32_t>(runInfo_.l2Info.calOrder);
        tilingData_.l2cacheUseInfo.l2CacheFlag = l2CacheFlag_;
    }
    SetNd2NzInfo();
    DoTilingKey();
    Mc2L2Cache l2Cache(args_, tilingData_);
    l2Cache.SetL2CacheFlag(tilingEnable_, compileInfo_.l2Size, l2CacheFlag_);
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2MatmulV3BaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

void Mc2MatmulV3BaseTiling::DoTilingKey()
{
    // 1: disable mix nd2nz 0: enable mix nd2nz
    uint64_t disableMixNd2nz = static_cast<uint64_t>(GetMixNd2nzType());
    tilingKey_ = GET_TPL_TILING_KEY(static_cast<uint64_t>(tilingEnable_.tilingEnableFullLoad),
                                    static_cast<uint64_t>(tilingEnable_.tilingEnableSplitCore),
                                    static_cast<uint64_t>(tilingEnable_.tilingEnableFixOpti), disableMixNd2nz);
    OP_LOGI(args_.opName, "Tiling Key is 0x%x", tilingKey_);
}

ge::graphStatus Mc2MatmulV3BaseTiling::GetWorkspaceSize()
{
    uint64_t align256Byte = 256 / aDtypeSize_; // 256B 对齐shape
    uint64_t alignedM = ops::CeilAlign(tilingData_.matmulTiling.M, 16);
    uint64_t alignedN = ops::CeilAlign(tilingData_.matmulTiling.N, 16);

    alignedM = std::max(alignedM, static_cast<uint64_t>(tilingData_.matmulTiling.singleCoreM));
    alignedN = std::max(alignedN, static_cast<uint64_t>(tilingData_.matmulTiling.singleCoreN));

    workspaceSize_ = RPC_WORKSIZE * MB_SIZE; // 20MB reserve > 16MB for rpc
    if (tilingEnable_.tilingEnableSplitCore == Mc2TilingEnableSplitCore::SINGLE_CORE_SPLIT_K ||
        tilingEnable_.tilingEnableSplitCore == Mc2TilingEnableSplitCore::SINGLE_CORE_NKM_SPLIT_K ||
        tilingEnable_.tilingEnableSplitCore == Mc2TilingEnableSplitCore::SINGLE_CORE_SPLIT_K_GM_TO_L1) {
        workspaceSize_ = args_.mValue * ops::CeilAlign(args_.nValue, align256Byte) * DATA_SIZE_FP32 +
                         RPC_WORKSIZE * MB_SIZE; // 20 means 20MB
    }
    OP_LOGI(args_.opName, "if tiling enable is deterministic splitk, workspace size is %lu", workspaceSize_);
    if (tilingEnable_.tilingEnableSplitCore == Mc2TilingEnableSplitCore::DETERMINISTIC_SPLIT_K) {
        uint64_t singleSize = alignedM * alignedN;
        workspaceSize_ =
            static_cast<uint64_t>(tilingData_.matmulTiling.usedCoreNum) * singleSize * DB_SIZE * DATA_SIZE_FP32 +
            RPC_WORKSIZE * MB_SIZE;
    }
    if (tilingEnable_.tilingEnableFixOpti == Mc2TilingEnableFixOpti::BASE_ENABLE_ALIGNOUT) {
        workspaceSize_ += ops::CeilAlign(args_.nValue, CACHELINE / cDtypeSize_) * tilingData_.matmulTiling.baseM *
                          tilingData_.matmulTiling.usedCoreNum * NUMBER_TWO * cDtypeSize_;
    }
    if (tilingEnable_.tilingEnableFixOpti == Mc2TilingEnableFixOpti::VEC_NZ2ND_UNALIGNOUT) {
        workspaceSize_ += ops::CeilAlign(args_.nValue, N_ALIGNED) * tilingData_.matmulTiling.baseM *
                          tilingData_.matmulTiling.usedCoreNum * NUMBER_TWO * cDtypeSize_;
    }
    if (!compileInfo_.supportL0c2out) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t c0Size = BLOCK_BYTE_SIZE / aDtypeSize_;
    uint64_t kALignForC0 = ops::CeilAlign(args_.kValue, c0Size);
    uint64_t kALignForN = ops::CeilAlign(args_.kValue, N_ALIGNED);
    // 非对齐场景需要预留一些workspace
    if (args_.nd2nzA) {
        if (args_.isATrans) {
            workspaceSize_ += ops::CeilAlign(args_.mValue, c0Size) * kALignForN * aDtypeSize_;
        } else {
            workspaceSize_ += ops::CeilAlign(args_.mValue, N_ALIGNED) * kALignForC0 * aDtypeSize_;
        }
    }
    if (args_.nd2nzB) {
        if (args_.isBTrans) {
            workspaceSize_ += ops::CeilAlign(args_.nValue, N_ALIGNED) * kALignForC0 * bDtypeSize_;
        } else {
            workspaceSize_ += ops::CeilAlign(args_.nValue, c0Size) * kALignForN * bDtypeSize_;
        }
    }
    OP_LOGI(args_.opName, "final workspace size is %lu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MatmulV3BaseTiling::PostTiling()
{
    size_t tilingDataSize = sizeof(Mc2MatmulV3TilingData);
    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(args_.opName, "tiling data size[%zu] is not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    context_->SetBlockDim(tilingData_.matmulTiling.usedCoreNum);
    context_->SetScheduleMode(1);
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "workspaces"),
                    return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}
} // namespace mc2_matmul_v3
} // namespace optiling
