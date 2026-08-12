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
 * \file mc2_gen_task_ops_utils_arch35.h
 * \brief
 */

#ifndef OPS_TRANSFORMER_DEV_MC2_COMMON_INC_MC2_ARCH35_GEN_TASK_OPS_UTILS_H
#define OPS_TRANSFORMER_DEV_MC2_COMMON_INC_MC2_ARCH35_GEN_TASK_OPS_UTILS_H

#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/kernel_launch_info.h"
#include "graph/arg_desc_info.h"
#include "mc2_common_log.h"
#include "mc2_gen_task_ops_utils.h"
#include "graph/ascend_string.h"

namespace ops {
const std::string COMM_ALG_FULLMESH_V1 = "fullmesh_v1";
const std::string COMM_ALG_FULLMESH_V2 = "fullmesh_v2";
const std::string COMM_ALG_MTE = "mte";
const std::string COMM_ALG_CCU = "ccu";

const std::string MOE_DISTRIBUTE_DISPATCH_OP_TYPE = "MoeDistributeDispatch";
const std::string MOE_DISTRIBUTE_COMBINE_OP_TYPE = "MoeDistributeCombine";
const std::string MOE_DISTRIBUTE_DISPATCH_V2_OP_TYPE = "MoeDistributeDispatchV2";
const std::string MOE_DISTRIBUTE_COMBINE_V2_OP_TYPE = "MoeDistributeCombineV2";
const std::string ALL_TO_ALLV_GROUPED_MM_OP_TYPE = "AlltoAllvGroupedMatMul";
const std::string GROUPED_MM_ALL_TO_ALLV_OP_TYPE = "GroupedMatMulAlltoAllv";
const std::string ALL_TO_ALLV_QUANT_GROUPED_MM_OP_TYPE = "AlltoAllvQuantGroupedMatMul";
const std::string QUANT_GROUPED_MM_ALL_TO_ALLV_OP_TYPE = "QuantGroupedMatMulAlltoAllv";
const std::string ALL_GATHER_MM_V2_OP_TYPE = "AllGatherMatmulV2";
const std::string MM_REDUCE_SCATTER_V2_OP_TYPE = "MatmulReduceScatterV2";
const std::string MM_ALL_REDUCE_OP_TYPE = "MatmulAllReduce";
const std::string QUANT_ALL_REDUCE_OP_TYPE = "QuantAllReduce";
const std::string QUANT_REDUCE_SCATTER_OP_TYPE = "QuantReduceScatter";
const std::string MM_ALLTO_ALL_OP_TYPE = "MatmulAlltoAll";
const std::string ALLTO_ALL_MATMUL_OP_TYPE = "AlltoAllMatmul";
const std::string ATTR_NAME_GROUP = "group";
const std::string ATTR_NAME_GROUP_EP = "group_ep";
const int32_t MAX_GROUP_CNT = 16;

struct GroupInfo {
    int32_t groupCnt = -1;                        // 算子通信域数量
    std::vector<std::string> groupAttrNames = {}; // 算子通信域属性名
};

// 当前 GetCCuTaskInfo 接口暂不支持多通信域的接口，对于双通信域的算子，暂时不增加 TP 属性名
static const std::map<const std::string, const GroupInfo> GROUP_INFO_MAP_ARCH35{
    {ALL_GATHER_MM_V2_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {MM_REDUCE_SCATTER_V2_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {MM_ALL_REDUCE_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {QUANT_ALL_REDUCE_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {QUANT_REDUCE_SCATTER_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {MM_ALLTO_ALL_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {ALLTO_ALL_MATMUL_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {ALL_TO_ALLV_GROUPED_MM_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {GROUPED_MM_ALL_TO_ALLV_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {ALL_TO_ALLV_QUANT_GROUPED_MM_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {QUANT_GROUPED_MM_ALL_TO_ALLV_OP_TYPE, {1, {ATTR_NAME_GROUP}}},
    {MOE_DISTRIBUTE_DISPATCH_OP_TYPE, {2, {ATTR_NAME_GROUP_EP}}},
    {MOE_DISTRIBUTE_COMBINE_OP_TYPE, {2, {ATTR_NAME_GROUP_EP}}},
    {MOE_DISTRIBUTE_DISPATCH_V2_OP_TYPE, {2, {ATTR_NAME_GROUP_EP}}},
    {MOE_DISTRIBUTE_COMBINE_V2_OP_TYPE, {2, {ATTR_NAME_GROUP_EP}}},
};

class Mc2Arch35GenTaskOpsUtils {
public:
    static ge::Status CreateCCUFusionTask(const gert::ExeResGenerationContext *context,
                                          std::vector<std::vector<uint8_t>> &tasks);

    static const std::string GetCommAlg(const gert::ExeResGenerationContext *context, const size_t commAlgIdx)
    {
        auto *attrs = context->GetAttrs();
        if (attrs == nullptr) {
            OPS_LOG_E(context->GetNodeName(), "Attrs pointer is null.");
            return "";
        }
        const char *commAlgPtr = attrs->GetStr(commAlgIdx);
        if (commAlgPtr == nullptr) {
            OPS_LOG_E(context->GetNodeName(), "Comm alg pointer is null.");
            return "";
        }
        const std::string commAlg = commAlgPtr;
        if (commAlg.empty()) {
            OPS_LOG_W(context->GetNodeName(), "Comm alg is empty, will use mte alg.");
            return "mte";
        }
        OPS_LOG_D(context->GetNodeName(), "Comm alg is %s.", commAlg.c_str());
        return commAlg;
    }
    static ge::Status Mc2Arch35GenTaskCallBack(const gert::ExeResGenerationContext *context,
                                               std::vector<std::vector<uint8_t>> &tasks)
    {
        if (context == nullptr) {
            OPS_LOG_E("Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack", "Failed to get context.");
            return ge::GRAPH_FAILED;
        }

        if (tasks.size() == 0U) {
            OPS_LOG_E(context->GetNodeName(), "Empty task vector when generating task.");
            return ge::GRAPH_FAILED;
        }

        if (CreateCCUFusionTask(context, tasks) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(context->GetNodeName(), "Failed to create CCU fusion task.");
            return ge::GRAPH_FAILED;
        }

        OPS_LOG_I(context->GetNodeName(), "Gen Task for CCU fusion task success.");
        return ge::GRAPH_SUCCESS;
    }

private:
    static bool GetGroupInfo(const gert::ExeResGenerationContext *context, GroupInfo &groupInfo)
    {
        const char *opType = context->GetNodeType();
        if (opType == nullptr) {
            OPS_LOG_E(context->GetNodeName(), "Op type is nullptr.");
            return false;
        }
        const std::string opTypeStr = opType;
        if (GROUP_INFO_MAP_ARCH35.find(opTypeStr) == GROUP_INFO_MAP_ARCH35.end()) {
            OPS_LOG_E(context->GetNodeName(), "Op type [%s] has not registe in group cnt map.", opType);
            return false;
        }
        groupInfo = GROUP_INFO_MAP_ARCH35.at(opTypeStr);
        const int32_t cnt = groupInfo.groupCnt;
        if ((cnt <= 0) || (cnt > MAX_GROUP_CNT)) {
            OPS_LOG_E(context->GetNodeName(), "Group cnt [%d] is invalid, it should in [1, %d].", cnt, MAX_GROUP_CNT);
            return false;
        }

        OPS_LOG_D(context->GetNodeName(), "Op [%s] get group [%d] success.", opType, cnt);
        return true;
    }
};
} // namespace ops

#endif // OPS_TRANSFORMER_DEV_MC2_COMMON_INC_MC2_ARCH35_GEN_TASK_OPS_UTILS_H
