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
 * \file flash_mla_with_kvcache_common_def.h
 * \brief flash_mla_with_kvcache TilingKey 相关枚举/宏/常量（照 flash_attn 的
 *        utils/flash_attn_common_def.h 组织，host tiling 与 kernel 入口共用；
 *        template_tiling_key.h 只保留 ASCENDC_TPL_ARGS_DECL/SEL 骨架）。
 */

#ifndef FLASH_MLA_WITH_KVCACHE_COMMON_DEF_H_
#define FLASH_MLA_WITH_KVCACHE_COMMON_DEF_H_

// 3-bit 位宽宏非工具链内置（内置仅 1/2/4/8），照 flash_attn_common_def.h 同款本地定义
#define ASCENDC_TPL_3_BW 3

// q/out 布局索引（InOutLayoutType，key 位 7-0；数值对齐 flash_attn：0=BSND 1=BNSD 2=TND）
// MLA 不支持输出转置（无 BNSD_BSND），out 布局恒等于 q 布局
#define InOutLayoutType_BSND 0
#define InOutLayoutType_BNSD 1
#define InOutLayoutType_TND 2

// kv 存储格式（KvLayoutType，key 位 15-8；数值对齐 flash_attn）
#define KvLayoutType_NO_PA 0
#define KvLayoutType_PA_BBND 1
#define KvLayoutType_PA_BNBD 2
#define KvLayoutType_PA_NZ 3

// 唯一 MLA 配置（Config，key 位 19-17）：S1=64, S2=128, D=576（nope512+rope64 合并宽）, DV=512
#define Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512 0
#define Config_MLA_DEFAULT Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512

enum class inferFaLayOutTypeEnum {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
    LAYOUT_NTD_TND = 5,
    LAYOUT_NTD = 6
};

enum class inferS1TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class inferS2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned1024 = 1024,
    NotAligned,
};

enum class inferDTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned48 = 48,
    Aligned64 = 64,
    Aligned80 = 80,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    NotAligned,
};

struct ConfigParams {
    inferS1TemplateType s1;
    inferS2TemplateType s2;
    inferDTemplateType d;
    inferDTemplateType dv;
};

// config → (s1, s2, d, dv) 模板块大小（MLA 唯一行，数值对齐 fia ConfigValue[9]）
static constexpr ConfigParams ConfigValue[] = {
    {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned128, inferDTemplateType::Aligned576,
     inferDTemplateType::Aligned512} // config=0: MLA 默认（s1=64, s2=128, D=576, DV=512）
};

#endif // FLASH_MLA_WITH_KVCACHE_COMMON_DEF_H_
