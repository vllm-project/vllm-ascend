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
 * \file matmul_v3_basic_streamk_tiling.cpp
 * \brief
 */
#include "matmul_v3_basic_streamk_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

using Ops::NN::MathUtil;
namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;

// ------------------------------ CheckStreamKDPSKTiling -------------------------------------------//
bool CheckStreamKDPSKTilingDefault(const MatmulV3CompileInfo& /* compileInfo */, const MatMulV3Args& /* args */)
{
    return false;
}

bool CheckStreamKDPSKTilingDav3510(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args)
{
    constexpr uint64_t STREAM_K_MIN_K_THRESHOLD = 8192UL;
    // 如果k轴小于32*256/DtypeSize_ 或 mn轴不是256对齐,不走stream-k-dpsk
    if (args.mValue % BASIC_BLOCK_SIZE_256 != 0UL || args.nValue % BASIC_BLOCK_SIZE_256 != 0UL ||
        args.kValue <
            std::max(STREAM_K_MIN_K_THRESHOLD, compileInfo.aicNum * BASIC_BLOCK_K_128_BYTE) / args.aDtypeSize) {
        return false;
    }
    // 如果mn用256切分的份数小于核数 或者 取余核数为0或大于一半的核数，则不使用stream-k-dpsk
    uint64_t mCnt = MathUtil::CeilDivision(args.mValue, BASIC_BLOCK_SIZE_256);
    uint64_t nCnt = MathUtil::CeilDivision(args.nValue, BASIC_BLOCK_SIZE_256);
    uint64_t totalMNCnt = mCnt * nCnt;
    return (totalMNCnt >= compileInfo.aicNum) && (totalMNCnt % compileInfo.aicNum != 0UL) &&
           (totalMNCnt % compileInfo.aicNum <= compileInfo.aicNum / NUM_TWO);
}

using CheckStreamKDPSKTilingFunc = bool (*)(const MatmulV3CompileInfo&, const MatMulV3Args&);

const static std::map<NpuArch, CheckStreamKDPSKTilingFunc> CheckStreamKDPSKTilingFuncMap = {
    {NpuArch::DAV_3510, CheckStreamKDPSKTilingDav3510},
};

// ------------------------------ CheckStreamKSKTiling -------------------------------------------//
bool CheckStreamKSKTilingDefault(const MatmulV3CompileInfo& /* compileInfo */, const MatMulV3Args& /* args */)
{
    return false;
}

bool CheckStreamKSKTilingDav3510(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args)
{
    constexpr uint64_t STREAM_K_MIN_K_THRESHOLD = 8192UL;
    if (ops::CeilAlign(static_cast<uint64_t>(args.kValue), BASIC_BLOCK_SIZE_256) <
        std::max(STREAM_K_MIN_K_THRESHOLD, compileInfo.aicNum * BASIC_BLOCK_K_256_BYTE) / args.aDtypeSize) {
        OP_LOGD(args.opName, "MatMulV3 tiling unenable state is DoStreamK value[%lu]", args.kValue);
        return false;
    }

    uint64_t alignValue = BASIC_BLOCK_SIZE_256;
    if (args.aDtypeSize == DATA_SIZE_FP32 && !args.isHf32) {
        alignValue = BLOCK_BYTE_SIZE; // 如果是Fp32 基本块判断要用32
    }
    // 判断mn是否需要已经能切32份及以上
    uint64_t mCnt = MathUtil::CeilDivision(args.mValue, alignValue);
    uint64_t nCnt = MathUtil::CeilDivision(args.nValue, alignValue);
    if (mCnt * nCnt > compileInfo.aicNum / NUM_TWO) {
        OP_LOGD(args.opName, "MatMulV3 tiling unenable state is DoStreamK mCnt[%lu], nCnt[%lu]", mCnt, nCnt);
        return false;
    }
    OP_LOGI(args.opName, "MatMulV3 tiling enable state is DoBasicApiSplitK.");
    return true;
}

using CheckStreamKSKTilingFunc = bool (*)(const MatmulV3CompileInfo&, const MatMulV3Args&);

const static std::map<NpuArch, CheckStreamKSKTilingFunc> CheckStreamKSKTilingFuncMap = {
    {NpuArch::DAV_3510, CheckStreamKSKTilingDav3510},
};

// ------------------------------ GetL0C2OutFlag -------------------------------------------//
MatMulV3L0C2Out GetL0C2OutFlagDefault(const MatMulV3Args& /* args */) { return MatMulV3L0C2Out::ON_THE_FLY; }

MatMulV3L0C2Out GetL0C2OutFlagDav3510(const MatMulV3Args& args)
{
    if (args.nValue > BASIC_BLOCK_SIZE_64 && args.nValue % BASIC_BLOCK_SIZE_16 != 0 && args.mValue > NUM_TWO &&
        args.mValue * args.nValue >= BASIC_BLOCK_SIZE_256) {
        return MatMulV3L0C2Out::ND_FIXPIPE_1_2;
    }
    return MatMulV3L0C2Out::ON_THE_FLY;
}

using GetL0C2OutFlagFunc = MatMulV3L0C2Out (*)(const MatMulV3Args&);

const static std::map<NpuArch, GetL0C2OutFlagFunc> GetL0C2OutFlagFuncMap = {
    {NpuArch::DAV_3510, GetL0C2OutFlagDav3510},
};

} // namespace

namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3BasicStreamKTiling, DAV_3510, BASIC_STREAM_K);

bool MatMulV3BasicStreamKTiling::CheckStreamKSKTiling() const
{
    auto iter = (CheckStreamKSKTilingFuncMap.find(compileInfo_.npuArch) == CheckStreamKSKTilingFuncMap.end()) ?
                    CheckStreamKSKTilingDefault :
                    CheckStreamKSKTilingFuncMap.at(compileInfo_.npuArch);
    return iter(compileInfo_, args_);
}

bool MatMulV3BasicStreamKTiling::CheckStreamKDPSKTiling() const
{
    auto iter = (CheckStreamKDPSKTilingFuncMap.find(compileInfo_.npuArch) == CheckStreamKDPSKTilingFuncMap.end()) ?
                    CheckStreamKDPSKTilingDefault :
                    CheckStreamKDPSKTilingFuncMap.at(compileInfo_.npuArch);
    return iter(compileInfo_, args_);
}

MatMulV3L0C2Out MatMulV3BasicStreamKTiling::GetL0C2OutFlag() const
{
    auto iter = (GetL0C2OutFlagFuncMap.find(compileInfo_.npuArch) == GetL0C2OutFlagFuncMap.end()) ?
                    GetL0C2OutFlagDefault :
                    GetL0C2OutFlagFuncMap.at(compileInfo_.npuArch);
    return iter(args_);
}

bool MatMulV3BasicStreamKTiling::IsCapable()
{
    // batch一致性控制，当开关等级为2或3时，拒绝切k模板，达到强一致性和batch一致性
    OP_LOGD(args_.opName, "deterministic_level=%d", context_->GetDeterministicLevel());
    if (context_->GetDeterministicLevel() > 1) {
        return false;
    }
    if (args_.aFormat != ge::FORMAT_ND) {
        OP_LOGD(args_.opName, "ND is the only supported format for tensor_a in basic api");
        return false;
    }
    if (MatMulV3TilingHelper::IsSelfNonContiguous(context_)) {
        OP_LOGD(args_.opName, "NonContiguous self does not support StreamK");
        return false;
    }
    if (compileInfo_.aivNum != (compileInfo_.aicNum * NUM_TWO)) {
        OP_LOGD(args_.opName, "streamk only support aivNum == aicNum * 2");
        return false;
    }
    return CheckStreamKSKTiling() || CheckStreamKDPSKTiling();
}

ge::graphStatus MatMulV3BasicStreamKTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);

    mCnt_ = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    nCnt_ = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    totalMNCnt_ = mCnt_ * nCnt_;
    if (totalMNCnt_ <= compileInfo_.aicNum / NUM_TWO) {
        if (mCnt_ > compileInfo_.aicNum / NUM_THREE && mCnt_ < compileInfo_.aicNum / NUM_TWO) {
            mCnt_ = compileInfo_.aicNum / NUM_TWO;
        }
        if (nCnt_ > compileInfo_.aicNum / NUM_THREE && nCnt_ < compileInfo_.aicNum / NUM_TWO) {
            nCnt_ = compileInfo_.aicNum / NUM_TWO;
        }
        totalMNCnt_ = mCnt_ * nCnt_;
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCnt_), BASIC_BLOCK_SIZE_16);
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCnt_), BASIC_BLOCK_SIZE_16);
        runInfo_.tailInfo.kCnt = ops::FloorDiv(compileInfo_.aicNum, totalMNCnt_);
        runInfo_.singleCoreK = MathUtil::CeilDivision(args_.kValue, runInfo_.tailInfo.kCnt);
        l0C2Out_ = GetL0C2OutFlag();
    } else {
        runInfo_.tailInfo.kCnt = compileInfo_.aicNum / (totalMNCnt_ % compileInfo_.aicNum);
        uint64_t skSingleCoreK = MathUtil::CeilDivision(args_.kValue, runInfo_.tailInfo.kCnt);
        runInfo_.tailInfo.kCnt = MathUtil::CeilDivision(args_.kValue, skSingleCoreK);
        runInfo_.singleCoreK = skSingleCoreK;
    }
    if (args_.bFormat != ge::FORMAT_ND) {
        if (args_.bDtypeSize == DATA_SIZE_FP16 || (args_.bDtypeSize == DATA_SIZE_FP32 && !args_.isBTrans)) {
            runInfo_.singleCoreK = ops::CeilAlign(runInfo_.singleCoreK, BASIC_BLOCK_SIZE_16);
        } else {
            runInfo_.singleCoreK = ops::CeilAlign(runInfo_.singleCoreK, BASIC_BLOCK_SIZE_16 / NUM_TWO);
        }
    }
    uint64_t baseKAlignValue = !args_.isATrans || args_.isBTrans ? BASIC_BLOCK_SIZE_128 / args_.aDtypeSize :
                                                                   BASIC_BLOCK_SIZE_16;
    uint64_t kValueMax = ops::FloorAlign(
        L0A_SIZE_2 / DB_SIZE / args_.aDtypeSize / std::max(runInfo_.baseM, runInfo_.baseN), baseKAlignValue);
    runInfo_.baseK = std::min(runInfo_.singleCoreK, kValueMax);
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    // depthb1 is less than deptha1
    if (runInfo_.baseM == runInfo_.baseN && runInfo_.depthB1 == runInfo_.depthA1 * NUM_TWO) {
        runInfo_.depthA1 = runInfo_.depthA1 * NUM_TWO;
        runInfo_.depthB1 = runInfo_.depthB1 / NUM_TWO;
        runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;
        runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
    }
    if ((totalMNCnt_ > compileInfo_.aicNum) && args_.hasBias) {
        runInfo_.stepKb = NUM_THREE; // reserve L1 space for bias
        runInfo_.stepKa = NUM_THREE;
    }
    return ge::GRAPH_SUCCESS;
}

std::vector<size_t> MatMulV3BasicStreamKTiling::GetWorkspaceSize() const
{
    size_t workspaceSize = compileInfo_.aicNum * BASIC_BLOCK_SIZE_256 * BASIC_BLOCK_SIZE_256 * DATA_SIZE_FP32 +
                           RPC_WORKSIZE * MB_SIZE;
    OP_LOGI(args_.opName, "MatMulV3 tiling workspace size is %lu", workspaceSize);
    return {workspaceSize};
}

uint64_t MatMulV3BasicStreamKTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    bool isSplitSinglecoreK = std::string_view(context_->GetNodeType()) == "MatMulV3" &&
                              (runInfo_.singleCoreK >= FP32_SPLIT_K_THRESHOLD && args_.aDtypeSize == DATA_SIZE_FP32);
    // fusedMatMul do not checkout to tensor api
    bool basicApi = std::string_view(context_->GetNodeType()) == "FusedMatMul" || args_.isAvoidTensorApi;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel((isSplitSinglecoreK && !basicApi) ? MatMulV3Model::SK_SPLIT_K : MatMulV3Model::STREAM_K)
        .SetL0C2Out(l0C2Out_)
        .SetApiLevel(basicApi ? MatMulV3ApiLevel::BASIC_LEVEL : MatMulV3ApiLevel::TENSOR_LEVEL)
        .GetTilingKey();
}

ge::graphStatus MatMulV3BasicStreamKTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<MatMulV3BasicTilingData>(tiling);
}
} // namespace matmul_v3_advanced
} // namespace optiling
