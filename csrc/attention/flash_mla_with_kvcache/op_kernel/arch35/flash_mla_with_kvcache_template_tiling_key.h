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
 * \file flash_mla_with_kvcache_template_tiling_key.h
 * \brief flash_mla_with_kvcache TilingKey 定义（非量化 MLA，仅 FP16/BF16）。
 *
 * 四字段布局完全对齐 flash_attn（attention/flash_attn/op_kernel/arch35/
 * flash_attn_template_tiling_key.h）：InOutLayoutType 8bw / KvLayoutType 8bw /
 * HasAttenMask 1bw / Config 3bw，同序同位宽；枚举/表/常量见 utils/
 * flash_mla_with_kvcache_common_def.h。无 IsFd（FD 能力恒实例化，运行时由
 * metadata 各 section mLen>0 驱动，见 kernel 入口注释）。
 *
 * 实例化集合（SEL 笛卡尔积 × 2 dtype）：
 *   InOutLayoutType ∈ {BSND, BNSD, TND}                  (3, 无输出转置)
 *   KvLayoutType    ∈ {PA_BBND, PA_BNBD, PA_NZ}          (3, 仅 PA 分页)
 *   HasAttenMask    ∈ {false, true}                       (2)
 *   Config          ∈ {0: S1=64/S2=128/D=576/DV=512}      (1, MLA 唯一配置)
 *   ⇒ 18 组合/ dtype × {half, bfloat16_t}（ORIG_DTYPE_* 编译宏）= 36 个 kernel 实例。
 */

#ifndef FLASH_MLA_WITH_KVCACHE_TEMPLATE_TILING_KEY_H_
#define FLASH_MLA_WITH_KVCACHE_TEMPLATE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#include "../utils/flash_mla_with_kvcache_common_def.h"
#include "flash_mla_with_kvcache_tiling_data.h"

#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_KEY
#define ORIG_DTYPE_KEY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_ATTENTION_OUT
#define ORIG_DTYPE_ATTENTION_OUT (DT_BF16)
#endif

// 布尔 template 参数值必须带数字 token：CANN ASCENDC_TPL_PRE 预编译产出的
// `@@ASCENDC_TPL_BOOL_DECL_<name>@@ = {<values>}` 由 template_tiling.py 的
// extract_num(re.findall(r'\d+', ...)) 解析。字面 "false"/"true" 无数字 →
// "values of ASCENDC_TPL_BOOL_DECL <name> is empty!"，tbe opc 编译即失败；
// 数字 token 是工具链契约要求（历史真实损坏点，勿回退为字面 bool）。
#define FLASH_MLA_WITH_KVCACHE_TPL_BOOL_FALSE 0
#define FLASH_MLA_WITH_KVCACHE_TPL_BOOL_TRUE 1

ASCENDC_TPL_ARGS_DECL(FlashMlaWithKvcache,
                      // InOutLayoutType（位宽 8-bit，key 位 7-0）
                      //    0: InOutLayoutType_BSND （q=BSND）
                      //    1: InOutLayoutType_BNSD （q=BNSD）
                      //    2: InOutLayoutType_TND  （q=TND；无输出转置，out 布局恒等于 q）
                      ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 2),
                      // KvLayoutType（位宽 8-bit，key 位 15-8）
                      //    0: KvLayoutType_NO_PA   （不实例化：连续 KV 从未被 MLA 跑过）
                      //    1: KvLayoutType_PA_BBND
                      //    2: KvLayoutType_PA_BNBD
                      //    3: KvLayoutType_PA_NZ
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                      // HasAttenMask（位宽 1-bit，key 位 16）
                      //    0: false
                      //    1: true
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, FLASH_MLA_WITH_KVCACHE_TPL_BOOL_FALSE,
                                            FLASH_MLA_WITH_KVCACHE_TPL_BOOL_TRUE),
                      // Config（位宽 3-bit，key 位 19-17）：MLA 唯一配置
                      //    0: Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512
                      ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 0), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSND,
                                                          InOutLayoutType_BNSD, InOutLayoutType_TND),
                                     ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BBND,
                                                          KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
                                     ASCENDC_TPL_BOOL_SEL(HasAttenMask, FLASH_MLA_WITH_KVCACHE_TPL_BOOL_FALSE,
                                                          FLASH_MLA_WITH_KVCACHE_TPL_BOOL_TRUE),
                                     ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST,
                                                          Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(FlashMlaWithKvcacheTilingData)), );

#endif // FLASH_MLA_WITH_KVCACHE_TEMPLATE_TILING_KEY_H_
