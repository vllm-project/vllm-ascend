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
 * \file flash_mla_with_kvcache_tiling.h
 * \brief FlashMlaWithKvcache arch35 tiling impl 声明（基于 fia_tiling_nonquant_mla.h 移植）
 */
#ifndef FLASH_MLA_WITH_KVCACHE_TILING_IMPL_H_
#define FLASH_MLA_WITH_KVCACHE_TILING_IMPL_H_

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../flash_mla_with_kvcache_tiling_info.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/flash_mla_with_kvcache_tiling_data.h"
#include "../../op_kernel/arch35/flash_mla_with_kvcache_template_tiling_key.h"

namespace optiling {
namespace flash_mla_with_kvcache {

// 4 字段 tiling key —— 与 flash_mla_with_kvcache_template_tiling_key.h 的 ASCENDC_TPL_ARGS_DECL(FlashMlaWithKvcache,
// ...) 声明序一致（InOutLayoutType 8bw / KvLayoutType 8bw / HasAttenMask 1bw / Config 3bw），对齐 flash_attn。
// 键内含语义按限定路由裁剪：InOutLayoutType 恒 BSND/BNSD/TND、Config 恒 MLA(config 0)、KvLayoutType 恒 PA 三类。
// 无 isFd——FD 能力恒实例化，运行时由 metadata mLen>0 决定。
struct FlashMlaWithKvcacheTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t kvLayoutType = 0;
    bool hasAttenMask = false;
    uint64_t config = 0;
};

struct FlashMlaWithKvcachePlatFormInfo {
    uint32_t coreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t cvRatio = 0;
    uint64_t ubSize = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0aSize = 0;
    uint64_t defaultSysWorkspaceSize = 0;
};

class FlashMlaWithKvcacheTilingImpl : public FiaTilingBase {
public:
    explicit FlashMlaWithKvcacheTilingImpl(gert::TilingContext *context)
        : FiaTilingBase(context)
    {}
    ~FlashMlaWithKvcacheTilingImpl() override = default;

protected:
    void InitTilingInfo(TilingInfo *tilingInfo) override;
    bool IsCapable() override;
    bool IsCapableBasicCheckMla();
    bool IsCapableFeatureCheckMla();
    bool IsCapableSparseLayoutCheckMla();
    ge::graphStatus DoOpTiling() override;

private:
    ge::graphStatus SetPlatMemoryInfo();
    void InitImplParam();
    void SplitPolicy();
    void AdjustSinnerAndSouter();
    void ComputeTilingData();
    void SetAttenMaskTilingData();
    void SetStartIdxTilingData();
    void SetPageAttentionLayoutTilingData();
    void GenTilingKey();
    void CalcNumBlocks(uint32_t aicNum);
    void CalcMaxWorkspaceSize();
    void CalcWorkspaceSize();
    void UpdateTilingKeyConfig();
    void UpdateTilingKeyLayout();
    void UpdateTilingKeyInfo();
    void SetFATilingData();
    void FillTiling();
    void PrintAllTilingData();
    void CalcScheduleMode();
    ge::graphStatus SetTilingData(FlashMlaWithKvcacheTilingData &tilingData);

    FlashMlaWithKvcacheTilingData tilingData_;
    FlashMlaWithKvcacheTilingKeyInfo tilingKeyInfo_;
    FlashMlaWithKvcachePlatFormInfo platformInfo_;
    uint32_t sOuterFactor_ = 0;
    uint32_t sInnerFactor_ = 0;
    bool flashDecodeFlag_ = false;
    bool metadataPassthrough_ = false;
    bool actualSeqLenQFlag_ = false;
    bool actualSeqLenKVFlag_ = false;
    // host 侧推导的每 batch 实际长度（INT32 输入，无 int64 GM 物化）
    std::vector<int64_t> actualSeqLengthsQ_ = {};
    std::vector<int64_t> actualSeqLengthsKV_ = {};
    bool needInit_ = false;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::BATCH_MODE;
    uint32_t numBlocks_ = 0;

    FlashMlaWithKvcacheTilingInfo *faInfo_ = nullptr;
};

} // namespace flash_mla_with_kvcache
} // namespace optiling
#endif // FLASH_MLA_WITH_KVCACHE_TILING_IMPL_H_
