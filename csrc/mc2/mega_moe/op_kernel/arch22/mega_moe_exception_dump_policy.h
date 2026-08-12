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
 * \file mega_moe_exception_dump_policy.h
 * \brief MegaMoe 算子的 ExceptionDump Policy。
 *        Policy 提供 MegaMoe 专属类型与编译期常量，注入 ExceptionDump 模板。
 *        通用方法（UpdateStage/Dump/DumpTilingData）由 ExceptionDump 实现。
 *        架构差异通过模板参数 ArchTag 区分，命名中不绑定具体架构名。
 */
#ifndef MEGA_MOE_EXCEPTION_DUMP_POLICY
#define MEGA_MOE_EXCEPTION_DUMP_POLICY

#include "../../../common/op_kernel/mc2_exception_dump.h"
#include "mega_moe_tiling_a2a3.h"

namespace MC2MegaMoeAdump {
// MegaMoe 执行阶段枚举，与 kernel 的 DispatchAndCombine 流程对应。
// 用于定位算子卡在哪个阶段，StageEnumT::END 需与 ExceptionDump 的 BlockStage 数组大小一致。
// 不同架构（A2/A3）的 stage 流程若有差异，可通过模板特化区分，此处为 arch22 通用流程。
enum class Stage : uint32_t {
    INIT = 0,
    APPLY_XACTIVE_MASK = 1,
    MOE_INIT_ROUTING = 2,
    ALLGATHER_TOKEN_PER_EXPERT = 3, // CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2
    CUMSUM_TOKEN_PER_EXPERT = 4,
    DISPATCH = 5,
    SWIGLU = 6,
    COMBINE = 7,
    RESET_TOKEN_PER_EXPERT = 8,
    CROSS_RANK_SYNC = 9,
    UNPERMUTE = 10,
    END = 11 // 哨兵，表示阶段总数，供 ExceptionDump BlockStage 数组使用
};

// MegaMoe ExceptionDump Policy：提供算子专属类型与编译期常量
// 模板参数：
//   - kIsQuantRouting: 是否为 quant routing，决定 TilingDataT 类型
//   - ArchTag: 架构标签（如 Catlass::Arch::AtlasA2），预留供架构差异扩展
template <bool kIsQuantRouting, typename ArchTag = void>
struct MegaMoeExceptionDumpPolicy {
    // 根据是否为 quant routing 选择完整 tiling 结构体
    // quant: MegaMoeTilingDataQuant（含 MoeInitRoutingQuantV2TilingData）
    // non-quant: MegaMoeTilingDataNonQuant（含 MoeInitRoutingV2TilingData）
    using TilingDataT = std::conditional_t<kIsQuantRouting, MegaMoeTilingDataQuant, MegaMoeTilingDataNonQuant>;
    using StageEnumT = Stage;
    static constexpr MC2ExceptionDump::OpType OP_TYPE = MC2ExceptionDump::OpType::OP_TYPE_MEGA_MOE;
};

// 引擎实例类型别名（供 kernel 直接使用）
// kernel 根据自身的 kRoutingIsQuant 编译期常量和 ArchTag 选择对应 Policy
template <bool kIsQuantRouting, typename ArchTag = void>
using ExceptionDumpEngine = MC2ExceptionDump::ExceptionDump<MegaMoeExceptionDumpPolicy<kIsQuantRouting, ArchTag>>;
} // namespace MC2MegaMoeAdump
#endif // MEGA_MOE_EXCEPTION_DUMP_POLICY
