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
 * \file flash_mla_with_kvcache_tiling.cpp
 * \brief FlashMlaWithKvcache arch35 tiling（MLA D512 noquant，基于
 *        fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_mla.cpp 移植）；
 *        metadata 输入存在时跳过切分计算直接透传（kernel 驱动 section）。
 */

#include "flash_mla_with_kvcache_tiling.h"
#include "../flash_mla_with_kvcache_tiling.h"
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../flash_mla_with_kvcache_tiling_utils.h"
#include "../../op_kernel/arch35/flash_mla_with_kvcache_template_tiling_key.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace flash_mla_with_kvcache {

// 与 fia_tiling_nonquant_mla.cpp 的 PRE_LOAD_NUM 常量定义逐字一致
constexpr uint64_t PRE_LOAD_NUM_MLA_ARCH35 = 2;

void FlashMlaWithKvcacheTilingImpl::InitTilingInfo(TilingInfo *tilingInfo)
{
    faInfo_ = static_cast<FlashMlaWithKvcacheTilingInfo *>(tilingInfo);
    // IsCapable 阶段即需判定 metadata（必传）——透传路径跳过切分计算
    metadataPassthrough_ = (faInfo_->opParamInfo.metadata.tensor != nullptr);
}

bool FlashMlaWithKvcacheTilingImpl::IsCapableBasicCheckMla()
{
    if (faInfo_ == nullptr) {
        return false;
    }
    // 不支持空Tensor
    if (faInfo_->emptyTensorFlag) {
        return false;
    }
    // 仅支持非量化
    if (faInfo_->inputQType != ge::DT_FLOAT16 && faInfo_->inputQType != ge::DT_BF16) {
        return false;
    }
    return true;
}

bool FlashMlaWithKvcacheTilingImpl::IsCapableFeatureCheckMla()
{
    if (faInfo_->kvStorageMode != KvStorageMode::PAGE_ATTENTION) { // 仅 PA 路由
        return false;
    }
    return true;
}

bool FlashMlaWithKvcacheTilingImpl::IsCapableSparseLayoutCheckMla()
{
    // 与 fia_tiling_nonquant_mla.cpp 的硬约束一致
    int64_t sparseMode = faInfo_->maskMode;
    if (sparseMode != static_cast<int64_t>(MaskMode::NO_MASK) &&
        sparseMode != static_cast<int64_t>(MaskMode::CAUSAL)) { // CAUSAL == RIGHT_DOWN (3)
        return false;
    }
    if (sparseMode == static_cast<int64_t>(MaskMode::NO_MASK) && faInfo_->attenMaskFlag) {
        return false;
    }
    // MLA D512 维度约束
    if (faInfo_->qkHeadDim != static_cast<int64_t>(arch35MLA::MLA_D_DIM_512) ||
        faInfo_->ropeHeadDim != static_cast<int64_t>(arch35MLA::MLA_ROPE_D_DIM_64) ||
        faInfo_->vHeadDim != static_cast<int64_t>(arch35MLA::MLA_D_DIM_512)) {
        return false;
    }
    // N2 = 1（单 latent KV 头）
    if (faInfo_->n2Size != 1) {
        return false;
    }
    // N1 范围 [1, 128]
    if (faInfo_->n1Size < 1 || faInfo_->n1Size > static_cast<int64_t>(arch35MLA::NUM_128)) {
        return false;
    }
    // layout 约束（宽矩阵，与 common_checker 的 SinglePara 布局路由一致）:
    // q ∈ {TND, BNSD, BSND}；out 必须与 q 同布局（不支持转置）；
    // kv 仅分页布局: PA_NZ / PA_BNBD / PA_BBND，连续 KV 永不放行
    if (faInfo_->qLayout != FlashMlaWithKvcacheLayout::TND && faInfo_->qLayout != FlashMlaWithKvcacheLayout::BNSD &&
        faInfo_->qLayout != FlashMlaWithKvcacheLayout::BSND) {
        return false;
    }
    if (faInfo_->outLayout != faInfo_->qLayout) { // out 必须与 q 一致，不支持转置
        return false;
    }
    if (faInfo_->kvLayout != FlashMlaWithKvcacheLayout::PA_NZ &&
        faInfo_->kvLayout != FlashMlaWithKvcacheLayout::PA_BNBD &&
        faInfo_->kvLayout != FlashMlaWithKvcacheLayout::PA_BBND) {
        return false;
    }
    return true;
}

bool FlashMlaWithKvcacheTilingImpl::IsCapable()
{
    if (!metadataPassthrough_) {
        // metadata 为必传（flash_attn metadata_checker 语义），缺失即不可达
        OP_LOGE(faInfo_->opName, "FlashMlaWithKvcache is not capable: metadata is required but is null.");
        return false;
    }
    if (!IsCapableBasicCheckMla()) {
        return false;
    }
    if (!IsCapableFeatureCheckMla()) {
        return false;
    }
    if (faInfo_->n1Size >= 1 && faInfo_->n1Size <= static_cast<int64_t>(arch35MLA::NUM_128) &&
        (faInfo_->n1Size & (faInfo_->n1Size - 1)) != 0) {
        return true;
    }
    if (!IsCapableSparseLayoutCheckMla()) {
        return false;
    }
    return true;
}

// FIA 最大 workspace 公式（tiling 下沉/无切分结果场景），逐字移植
void FlashMlaWithKvcacheTilingImpl::CalcMaxWorkspaceSize()
{
    constexpr uint64_t mSize = 64;
    constexpr uint64_t dVSize = arch35MLA::DSIZE_512;
    constexpr uint64_t lseSize = 8;

    workspaceSize_ = platformInfo_.defaultSysWorkspaceSize;

    const uint64_t faTmpAttenGmSize =
        static_cast<uint64_t>(platformInfo_.aicNum) * PRE_LOAD_NUM_MLA_ARCH35 * mSize * dVSize;
    const uint64_t faTmpResLseGmSize =
        static_cast<uint64_t>(platformInfo_.aicNum) * PRE_LOAD_NUM_MLA_ARCH35 * mSize * lseSize;
    workspaceSize_ += (faTmpAttenGmSize + 2 * faTmpResLseGmSize) * sizeof(float); // ResLse有2份，sum和max
}

void FlashMlaWithKvcacheTilingImpl::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(faInfo_->opName, "FlashMlaWithKvcache schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus FlashMlaWithKvcacheTilingImpl::DoOpTiling()
{
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(faInfo_->opName, "Set plat memory info fail."),
                return ge::GRAPH_FAILED);

    // 先零初始化再按路径填字段：未写字段（如 prefill 下的 flashMlaWithKvcacheWorkspaceParams、
    // leftPadding/postQuant/s1OuterSplit 等）保持确定值，避免 kernel 读到未初始化内存，
    // 同时保证 tiling data 逐字节可哈希（UT 的 FNV 断言依赖确定性）。
    memset(&tilingData_, 0, sizeof(tilingData_));

    InitImplParam();
    SplitPolicy();
    FillTiling();
    CalcScheduleMode();
    CalcWorkspaceSize();
    PrintAllTilingData();
    GenTilingKey();

    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) || (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS) ||
        (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheTilingImpl::SetPlatMemoryInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(faInfo_->opName, "The platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    platformInfo_.cvRatio = platformInfo_.aivNum / platformInfo_.aicNum;
    platformInfo_.coreNum = platformInfo_.aivNum;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);

    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGI(faInfo_->opName, "AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

// ACLNN 直调路径下 seq-lens tensor 的 GetData<int32_t>() 返回设备指针（host 侧解引用即段错误），
// 与 flash_attn 的 host tiling 同策略：只按 shape/attr 存在性推导，不读 GM 值（编译期实测）。
// 每 batch 实际长度统一取 s1Size/s2Size（= max_seqlen_q/kv attr）；needInit 仅当总 token 数为 0。
// kernel 侧的 ACTLEN_T=uint32_t 解析器仍按 cu_seqlens_q/cache_seqlens 的 GM 值工作，不受影响。
void FlashMlaWithKvcacheTilingImpl::InitImplParam()
{
    const gert::Tensor *cuSeqlensQ = faInfo_->opParamInfo.cuSeqlensQ.tensor;
    const gert::Tensor *cacheSeqlens = faInfo_->opParamInfo.cacheSeqlens.tensor;
    const gert::Tensor *seqUsedQ = faInfo_->opParamInfo.sequsedQ.tensor;
    uint32_t cuSeqLenQDims = (cuSeqlensQ != nullptr) ? static_cast<uint32_t>(cuSeqlensQ->GetShapeSize()) : 0;
    uint32_t cacheSeqlensDims = (cacheSeqlens != nullptr) ? static_cast<uint32_t>(cacheSeqlens->GetShapeSize()) : 0;
    actualSeqLenQFlag_ = (cuSeqLenQDims > 0) && (cuSeqlensQ != nullptr);
    actualSeqLenKVFlag_ = (cacheSeqlensDims > 0) && (cacheSeqlens != nullptr);

    const int64_t qLenPerBatch = (faInfo_->qTSize > 0) ? faInfo_->s1Size : 0;
    const int64_t kvLenPerBatch = (faInfo_->s2Size > 0) ? faInfo_->s2Size : 0;
    actualSeqLengthsQ_.assign(static_cast<size_t>(faInfo_->bSize), qLenPerBatch);
    actualSeqLengthsKV_.assign(static_cast<size_t>(faInfo_->bSize), kvLenPerBatch);

    // 存在性信息保留供日志/排障（seqUsedQ 存在与否不改变 shape 级推导）
    (void)seqUsedQ;

    // FIA emptyTensorParams.needInit 语义：任一 batch 的实际 Q 或 KV 长度为 0 时输出行需要初始化
    needInit_ = (faInfo_->qTSize == 0) || (faInfo_->s2Size == 0);

    OP_LOGI(faInfo_->opName, "metadataPassthrough:%u actualSeqLenQFlag:%u actualSeqLenKVFlag:%u needInit:%u",
            metadataPassthrough_ ? 1U : 0U, actualSeqLenQFlag_ ? 1U : 0U, actualSeqLenKVFlag_ ? 1U : 0U,
            needInit_ ? 1U : 0U);
}

void FlashMlaWithKvcacheTilingImpl::AdjustSinnerAndSouter()
{
    // MLA D512 固定切分大小（fia_tiling_nonquant_mla.cpp:270-276）
    sOuterFactor_ = arch35MLA::SOUTER_32;
    sInnerFactor_ = arch35MLA::SINNER_128;

    OP_LOGI(faInfo_->opName, "Souter:%u SInner:%u", sOuterFactor_, sInnerFactor_);
}

void FlashMlaWithKvcacheTilingImpl::SplitPolicy()
{
    AdjustSinnerAndSouter(); // 确定tiling切块

    if (!metadataPassthrough_) {
        // 兜底：metadata 缺失（正常应被 checker 拒绝）——无法获知分核结果，按最大 workspace 处理
        CalcMaxWorkspaceSize();
        CalcNumBlocks(platformInfo_.aicNum);
        flashDecodeFlag_ = true;
        OP_LOGW(faInfo_->opName, "metadata is null, fallback to max workspace, all cores.");
        return;
    }

    // metadata 透传——跳过 split_core_v2 切分计算（无内嵌 fiaMetaData 可写，kernel 按 GM
    // metadata 逐 section 驱动）；全核分工，FD 使能以各 section 的 FD 元数据（mLen>0）为准。
    // 路由：只要 metadata 可能存在 FD/s2-核间切分归约（长 KV 的 SectionStreamK WithFd 调度），
    // 就必须实例化 FD 能力 → 恒真（与 flash_attn tiling flashDecodeFlag_=true 一致）。曾用
    // s1Size<=1 代理，漏掉 prefill 长 KV（kvS=32768 时 producer 会产 FD 归约 section），
    // 导致 isFd=false 内核不执行归约 → LSE/out 只含单核 s2 片段。非 FD 几何下 FD section 为空
    // （mLen=0），运行时按元数据跳过，无副作用。
    flashDecodeFlag_ = true;
    CalcNumBlocks(platformInfo_.aicNum);
    OP_LOGI(faInfo_->opName, "metadata passthrough, skip split compute, flashDecodeFlag_: %u.",
            flashDecodeFlag_ ? 1U : 0U);
}

// FIA CalcNumBlocks，逐字移植；metadata 透传路径 usedAicNum = 全核
void FlashMlaWithKvcacheTilingImpl::CalcNumBlocks(uint32_t usedAicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(faInfo_->platformInfo);
    auto aivNum = usedAicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, usedAicNum, aivNum);
    OP_LOGI(faInfo_->opName, "FlashMlaWithKvcache block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum,
            usedAicNum);
}

void FlashMlaWithKvcacheTilingImpl::UpdateTilingKeyConfig()
{
    // MLA D512 唯一路径: S1=64, S2=128, D=576, DV=512 -> Config index 0
    tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512;
}

void FlashMlaWithKvcacheTilingImpl::UpdateTilingKeyLayout()
{
    // q 布局 -> InOutLayoutType（取值以 op_kernel/utils/flash_mla_with_kvcache_common_def.h
    // 为准，数值对齐 flash_attn：BSND=0 / BNSD=1 / TND=2；out 布局恒等于 q，无输出转置，
    // 不占独立键位）
    if (faInfo_->qLayout == FlashMlaWithKvcacheLayout::TND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_TND;
    } else if (faInfo_->qLayout == FlashMlaWithKvcacheLayout::BSND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BSND;
    } else if (faInfo_->qLayout == FlashMlaWithKvcacheLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD;
    }
}

void FlashMlaWithKvcacheTilingImpl::UpdateTilingKeyInfo()
{
    UpdateTilingKeyLayout();
    UpdateTilingKeyConfig();
    tilingKeyInfo_.hasAttenMask = faInfo_->attenMaskFlag;
    // kv 布局 -> KvLayoutType（取值同 flash_attn 的 UpdateTilingKeyKvLayout 映射）：
    //   PA_NZ -> KvLayoutType_PA_NZ(3)；PA_BNBD -> KvLayoutType_PA_BNBD(2)；PA_BBND -> KvLayoutType_PA_BBND(1)
    tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_NZ;
    if (faInfo_->kvLayout == FlashMlaWithKvcacheLayout::PA_BNBD) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BNBD;
    } else if (faInfo_->kvLayout == FlashMlaWithKvcacheLayout::PA_BBND) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BBND;
    }
}

void FlashMlaWithKvcacheTilingImpl::GenTilingKey()
{
    UpdateTilingKeyInfo();
    // 实参序 = ASCENDC_TPL_ARGS_DECL 声明序：InOutLayoutType, KvLayoutType, HasAttenMask, Config
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.kvLayoutType,
                                    tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.config);

    OP_LOGI(faInfo_->opName, "The tilingkey is %llu.", tilingKey_);
    OP_LOGI(faInfo_->opName,
            "The tilingkey param is inOutLayoutType: %llu, kvLayoutType: %llu, hasAttenMask: %u, config: %llu.",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.hasAttenMask,
            tilingKeyInfo_.config);
}

// FIA MLA workspace 公式，逐字移植（accumOutSize/logSumExpSize 与 FD 槽位
// mBaseSize=64/dSizeV=512 保持一致，见 flash_mla_with_kvcache_block_vec_flashdecode_mla.h taskOffset 注释）
void FlashMlaWithKvcacheTilingImpl::CalcWorkspaceSize()
{
    constexpr uint64_t mSize = 64;
    constexpr uint64_t dSize = arch35MLA::DSIZE_512;
    constexpr uint64_t lseSize = 8;

    workspaceSize_ = platformInfo_.defaultSysWorkspaceSize;

    if (flashDecodeFlag_) {
        const uint64_t faTmpAttenGmSize =
            static_cast<uint64_t>(numBlocks_) * PRE_LOAD_NUM_MLA_ARCH35 * mSize * dSize; // 每个核最多有2次写到workspace
        const uint64_t faTmpResLseGmSize =
            static_cast<uint64_t>(numBlocks_) * PRE_LOAD_NUM_MLA_ARCH35 * mSize * lseSize;
        workspaceSize_ += (faTmpAttenGmSize + 2 * faTmpResLseGmSize) * sizeof(float); // ResLse有2份，sum和max
        tilingData_.baseTiling.flashMlaWithKvcacheWorkspaceParams.accumOutSize =
            static_cast<uint32_t>(faTmpAttenGmSize);
        tilingData_.baseTiling.flashMlaWithKvcacheWorkspaceParams.logSumExpSize =
            static_cast<uint32_t>(faTmpResLseGmSize);
    }

    OP_LOGI(faInfo_->opName, "Workspaces: %lu", workspaceSize_);
}

void FlashMlaWithKvcacheTilingImpl::FillTiling()
{
    ComputeTilingData();
    SetFATilingData();
}

void FlashMlaWithKvcacheTilingImpl::ComputeTilingData()
{
    SetAttenMaskTilingData();
    SetStartIdxTilingData();
    SetPageAttentionLayoutTilingData();
}

void FlashMlaWithKvcacheTilingImpl::SetAttenMaskTilingData()
{
    if (faInfo_->attenMaskFlag) {
        uint64_t maskBatch = 1;
        uint64_t maskDimNum = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDimNum();
        uint64_t maskS1Size = 2048;
        uint64_t maskS2Size = 2048;
        if (maskDimNum != 2 || faInfo_->s1Size == 1) {
            maskBatch = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(0);
        }
        tilingData_.baseTiling.flashMlaWithKvcacheAttenMaskParams.attenMaskBatch = maskBatch;
        maskS2Size = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(maskDimNum - 1);
        maskS1Size = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(maskDimNum - 2);
        tilingData_.baseTiling.flashMlaWithKvcacheAttenMaskParams.attenMaskS1Size = maskS1Size;
        tilingData_.baseTiling.flashMlaWithKvcacheAttenMaskParams.attenMaskS2Size = maskS2Size;
    } else {
        tilingData_.baseTiling.flashMlaWithKvcacheAttenMaskParams.attenMaskS1Size = 0;
        tilingData_.baseTiling.flashMlaWithKvcacheAttenMaskParams.attenMaskS2Size = 0;
    }
    tilingData_.baseTiling.flashMlaWithKvcacheAttenMaskParams.sparseMode = static_cast<uint8_t>(faInfo_->maskMode);
    // preTokens/nextTokens 由 maskMode 推导（FIA UpdatePreNextTokenBySparseMode 语义）
    tilingData_.baseTiling.flashMlaWithKvcacheAttenMaskParams.preTokens = static_cast<int32_t>(faInfo_->preTokens);
    tilingData_.baseTiling.flashMlaWithKvcacheAttenMaskParams.nextTokens = static_cast<int32_t>(faInfo_->nextTokens);
    tilingData_.baseTiling.flashMlaWithKvcacheAttenMaskParams.isRowInvalidOpen = 0;
    tilingData_.baseTiling.flashMlaWithKvcacheAttenMaskParams.isExistRowInvalid = 0;
}

void FlashMlaWithKvcacheTilingImpl::SetStartIdxTilingData()
{
    tilingData_.baseTiling.flashMlaWithKvcachePseParams.qStartIdx = 0;
    tilingData_.baseTiling.flashMlaWithKvcachePseParams.kvStartIdx = 0;
}

void FlashMlaWithKvcacheTilingImpl::SetPageAttentionLayoutTilingData()
{
    if (faInfo_->pageAttentionFlag) {
        uint32_t keyCacheDimNum = faInfo_->opParamInfo.kCache.shape->GetStorageShape().GetDimNum();
        if (keyCacheDimNum == 3) { // 3: BBH
            tilingData_.baseTiling.flashMlaWithKvcachePageAttentionParams.paLayoutType = 1;
        } else if (keyCacheDimNum == 4) { // 4: BNBD
            tilingData_.baseTiling.flashMlaWithKvcachePageAttentionParams.paLayoutType = 0;
        } else if (keyCacheDimNum == 5) { // 5: PA NZ
            tilingData_.baseTiling.flashMlaWithKvcachePageAttentionParams.paLayoutType = 2;
        }
    }
}

void FlashMlaWithKvcacheTilingImpl::SetFATilingData()
{
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.bSize = faInfo_->bSize;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.t1Size = faInfo_->qTSize;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.t2Size = faInfo_->kTSize;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.n2Size = faInfo_->n2Size;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.gSize = faInfo_->gSize;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.s1Size = faInfo_->s1Size;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.s2Size = faInfo_->s2Size;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.dSize = faInfo_->qkHeadDim;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.dSizeV = faInfo_->vHeadDim;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.dSizeRope = faInfo_->ropeHeadDim;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.scaleValue = faInfo_->softmaxScale;
    // kernel ActualSeqLensParser<ACTLEN_T=uint32_t> 的 actualLenDims（INT32 buffer 元素数）
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.actualSeqLengthsQSize =
        static_cast<uint32_t>(faInfo_->actualLenQDims);
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.actualSeqLengthsKVSize =
        static_cast<uint32_t>(faInfo_->actualLenKvDims);
    // k_cache 单缓冲承载 key+value（PA）-> 非连续，isKvContinuous = 0
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.isKvContinuous =
        (faInfo_->kvStorageMode == KvStorageMode::BATCH_CONTINUOUS) ? 1 : 0;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.isSoftMaxLseEnable = faInfo_->softmaxLseFlag;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.coreNum = numBlocks_;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.outputLayout =
        static_cast<uint32_t>(faInfo_->kernelOutputLayout);

    tilingData_.baseTiling.flashMlaWithKvcachePageAttentionParams.blockSize = faInfo_->blockSize;
    uint32_t maxBlockNumPerBatch = 0;
    if (faInfo_->kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        maxBlockNumPerBatch = faInfo_->opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    }
    tilingData_.baseTiling.flashMlaWithKvcachePageAttentionParams.maxBlockNumPerBatch = maxBlockNumPerBatch;

    tilingData_.baseTiling.flashMlaWithKvcacheSystemPrefixParams.isActualSharedPrefixLenNull = 1;
    tilingData_.baseTiling.flashMlaWithKvcacheSystemPrefixParams.prefixSeqInnerSize = 0;

    int64_t outSize = faInfo_->opParamInfo.attnOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize = faInfo_->softmaxLseFlag ? faInfo_->opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    tilingData_.baseTiling.flashMlaWithKvcacheEmptyTensorParams.singleCoreSize = singleCoreSize;
    tilingData_.baseTiling.flashMlaWithKvcacheEmptyTensorParams.totalOutputSize = outSize;
    tilingData_.baseTiling.flashMlaWithKvcacheEmptyTensorParams.totalSoftMaxLseOutputSize = lseSize;
    tilingData_.baseTiling.flashMlaWithKvcacheEmptyTensorParams.needInit = needInit_ ? 1 : 0;

    // k==v: k_cache 单缓冲，key/value stride 同源；kRope 在合并缓冲内 offset=head_dim_v，
    // 其 stride 与 key 一致（按同 buffer 推导，FIA 有独立 k_rope 时同款赋值）
    if (faInfo_->hasViewStride) {
        tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.keyStrides.bnStride = faInfo_->keyBnStride;
        tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.keyStrides.n2Stride = faInfo_->keyN2Stride;
        tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.valueStrides.bnStride = faInfo_->valueBnStride;
        tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.valueStrides.n2Stride = faInfo_->valueN2Stride;
        tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.kRopeStrides.bnStride = faInfo_->keyBnStride;
        tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.kRopeStrides.n2Stride = faInfo_->keyN2Stride;
    }
}

ge::graphStatus FlashMlaWithKvcacheTilingImpl::SetTilingData(FlashMlaWithKvcacheTilingData &tilingData)
{
    FlashMlaWithKvcacheTilingData *tiling = context_->GetTilingData<FlashMlaWithKvcacheTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(faInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

void FlashMlaWithKvcacheTilingImpl::PrintAllTilingData()
{
    FlashMlaWithKvcacheNoQuantTilingArch35 &baseTiling = tilingData_.baseTiling;
    FlashMlaWithKvcacheBaseParams &flashMlaWithKvcacheBaseParams = baseTiling.flashMlaWithKvcacheBaseParams;
    FlashMlaWithKvcacheAttenMaskParams &flashMlaWithKvcacheAttenMaskParams =
        baseTiling.flashMlaWithKvcacheAttenMaskParams;
    FlashMlaWithKvcachePseParams &flashMlaWithKvcachePseParams = baseTiling.flashMlaWithKvcachePseParams;
    FlashMlaWithKvcacheSystemPrefixParams &flashMlaWithKvcacheSystemPrefixParams =
        baseTiling.flashMlaWithKvcacheSystemPrefixParams;
    FlashMlaWithKvcachePageAttentionParams &flashMlaWithKvcachePageAttentionParams =
        baseTiling.flashMlaWithKvcachePageAttentionParams;
    FlashMlaWithKvcacheLeftPaddingParams &flashMlaWithKvcacheLeftPaddingParams =
        baseTiling.flashMlaWithKvcacheLeftPaddingParams;
    FlashMlaWithKvcachePostQuantParams &flashMlaWithKvcachePostQuantParams =
        baseTiling.flashMlaWithKvcachePostQuantParams;
    FlashMlaWithKvcacheWorkspaceParams &flashMlaWithKvcacheWorkspaceParams =
        baseTiling.flashMlaWithKvcacheWorkspaceParams;
    FlashMlaWithKvcacheEmptyTensorParams &flashMlaWithKvcacheEmptyTensorParams =
        baseTiling.flashMlaWithKvcacheEmptyTensorParams;

    OP_LOGD(faInfo_->opName, "bSize:%d", flashMlaWithKvcacheBaseParams.bSize);
    OP_LOGD(faInfo_->opName, "t1Size:%d", flashMlaWithKvcacheBaseParams.t1Size);
    OP_LOGD(faInfo_->opName, "t2Size:%d", flashMlaWithKvcacheBaseParams.t2Size);
    OP_LOGD(faInfo_->opName, "n2Size:%d", flashMlaWithKvcacheBaseParams.n2Size);
    OP_LOGD(faInfo_->opName, "gSize:%d", flashMlaWithKvcacheBaseParams.gSize);
    OP_LOGD(faInfo_->opName, "s1Size:%d", flashMlaWithKvcacheBaseParams.s1Size);
    OP_LOGD(faInfo_->opName, "s2Size:%d", flashMlaWithKvcacheBaseParams.s2Size);
    OP_LOGD(faInfo_->opName, "dSize:%d", flashMlaWithKvcacheBaseParams.dSize);
    OP_LOGD(faInfo_->opName, "dSizeV:%d", flashMlaWithKvcacheBaseParams.dSizeV);
    OP_LOGD(faInfo_->opName, "dSizeRope:%d", flashMlaWithKvcacheBaseParams.dSizeRope);
    OP_LOGD(faInfo_->opName, "actualSeqLengthsQSize:%d", flashMlaWithKvcacheBaseParams.actualSeqLengthsQSize);
    OP_LOGD(faInfo_->opName, "actualSeqLengthsKVSize:%d", flashMlaWithKvcacheBaseParams.actualSeqLengthsKVSize);
    OP_LOGD(faInfo_->opName, "scaleValue:%f", flashMlaWithKvcacheBaseParams.scaleValue);
    OP_LOGD(faInfo_->opName, "isKvContinuous:%d", flashMlaWithKvcacheBaseParams.isKvContinuous);
    OP_LOGD(faInfo_->opName, "isSoftMaxLseEnable:%d", flashMlaWithKvcacheBaseParams.isSoftMaxLseEnable);
    OP_LOGD(faInfo_->opName, "coreNum:%d", flashMlaWithKvcacheBaseParams.coreNum);
    OP_LOGD(faInfo_->opName, "outputLayout:%d", flashMlaWithKvcacheBaseParams.outputLayout);

    OP_LOGD(faInfo_->opName, "sparseMode:%d", flashMlaWithKvcacheAttenMaskParams.sparseMode);
    OP_LOGD(faInfo_->opName, "preTokens:%d", flashMlaWithKvcacheAttenMaskParams.preTokens);
    OP_LOGD(faInfo_->opName, "nextTokens:%d", flashMlaWithKvcacheAttenMaskParams.nextTokens);
    OP_LOGD(faInfo_->opName, "attenMaskS1Size:%d", flashMlaWithKvcacheAttenMaskParams.attenMaskS1Size);
    OP_LOGD(faInfo_->opName, "attenMaskS2Size:%d", flashMlaWithKvcacheAttenMaskParams.attenMaskS2Size);
    OP_LOGD(faInfo_->opName, "isRowInvalidOpen:%d", flashMlaWithKvcacheAttenMaskParams.isRowInvalidOpen);
    OP_LOGD(faInfo_->opName, "isExistRowInvalid:%d", flashMlaWithKvcacheAttenMaskParams.isExistRowInvalid);

    OP_LOGD(faInfo_->opName, "pseS1Size:%d", flashMlaWithKvcachePseParams.pseS1Size);
    OP_LOGD(faInfo_->opName, "pseS2Size:%d", flashMlaWithKvcachePseParams.pseS2Size);
    OP_LOGD(faInfo_->opName, "qStartIdx:%d", flashMlaWithKvcachePseParams.qStartIdx);
    OP_LOGD(faInfo_->opName, "kvStartIdx:%d", flashMlaWithKvcachePseParams.kvStartIdx);

    OP_LOGD(faInfo_->opName, "isActualSharedPrefixLenNull:%d",
            flashMlaWithKvcacheSystemPrefixParams.isActualSharedPrefixLenNull);
    OP_LOGD(faInfo_->opName, "prefixSeqInnerSize:%d", flashMlaWithKvcacheSystemPrefixParams.prefixSeqInnerSize);

    if (faInfo_->pageAttentionFlag) {
        OP_LOGD(faInfo_->opName, "paLayoutType:%d", flashMlaWithKvcachePageAttentionParams.paLayoutType);
        OP_LOGD(faInfo_->opName, "blockSize:%d", flashMlaWithKvcachePageAttentionParams.blockSize);
        OP_LOGD(faInfo_->opName, "maxBlockNumPerBatch:%d", flashMlaWithKvcachePageAttentionParams.maxBlockNumPerBatch);
    }

    OP_LOGD(faInfo_->opName, "isQHasLeftPadding:%d", flashMlaWithKvcacheLeftPaddingParams.isQHasLeftPadding);
    OP_LOGD(faInfo_->opName, "isKVHasLeftPadding:%d", flashMlaWithKvcacheLeftPaddingParams.isKVHasLeftPadding);

    OP_LOGD(faInfo_->opName, "isPostQuantPerChnl:%d", flashMlaWithKvcachePostQuantParams.isPostQuantPerChnl);
    OP_LOGD(faInfo_->opName, "isPostQuantBF16:%d", flashMlaWithKvcachePostQuantParams.isPostQuantBF16);

    if (flashDecodeFlag_) {
        OP_LOGD(faInfo_->opName, "accumOutSize:%d", flashMlaWithKvcacheWorkspaceParams.accumOutSize);
        OP_LOGD(faInfo_->opName, "logSumExpSize:%d", flashMlaWithKvcacheWorkspaceParams.logSumExpSize);
    }

    OP_LOGD(faInfo_->opName, "singleCoreSize:%d", flashMlaWithKvcacheEmptyTensorParams.singleCoreSize);
    OP_LOGD(faInfo_->opName, "needInit:%d", flashMlaWithKvcacheEmptyTensorParams.needInit);
    OP_LOGD(faInfo_->opName, "totalOutputSize:%d", flashMlaWithKvcacheEmptyTensorParams.totalOutputSize);
    OP_LOGD(faInfo_->opName, "totalSoftMaxLseOutputSize:%d",
            flashMlaWithKvcacheEmptyTensorParams.totalSoftMaxLseOutputSize);

    int64_t cap = context_->GetRawTilingData()->GetCapacity();
    OP_LOGD(faInfo_->opName, "Tiling Data context_ GetCapacity: %lu.", cap);
}

} // namespace flash_mla_with_kvcache

using flash_mla_with_kvcache::FlashMlaWithKvcacheTilingImpl;

// 值越小表示优先级越高
REGISTER_TILING_TEMPLATE_FIA(FlashMlaWithKvcache, FlashMlaWithKvcacheTilingImpl,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}), 1);
} // namespace optiling
