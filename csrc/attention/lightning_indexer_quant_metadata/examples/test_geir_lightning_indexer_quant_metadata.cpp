/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>
#include <string>
#include <map>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"  // REG_OP(Data)
#include "ge_ir_build.h"

#include "nn_other.h"

#include "../op_graph/lightning_indexer_quant_metadata_proto.h"
#include "../op_kernel_aicpu/lightning_indexer_quant_metadata.h"

using namespace ge;

static const uint32_t batchSize = 4;
static const uint32_t seqSizeK = 10240;
static const uint32_t seqSizeQ = 3;
static const uint32_t numHeadsQ = 128;
static const uint32_t numHeadsK = 1;
static const uint32_t sparseMode = 3;
static const uint32_t preToken = 3;
static const uint32_t nextToken = 3;
static const uint32_t cmpRatio = 3;
static const bool isFD = false;
static const std::string layoutQuery = "BSND";
static const std::string layoutKV = "PA_ND";
static const std::string socVersion = "ascend910B";
static const uint32_t aicCoreNum = 24;
static const uint32_t aivCoreNum = 48;

static const bool enableActLenQuery = true;
static const bool enableActLenKey = true;
static const std::vector<int32_t> actualSeqLengthsQuery = {3, 6, 9, 12};
static const std::vector<int32_t> actualSeqLengthsKey = {10240, 10240, 10240, 10240};
static const std::vector<int64_t> actualSeqLengthsQueryShape = {batchSize};
static const std::vector<int64_t> actualSeqLengthsKeyShape = {batchSize};
static const std::vector<int64_t> metadataShape = {optiling::LIQ_META_SIZE};
static const std::string dumpFile = "./dump";

using namespace ge;

class GeEnv {
public:
    GeEnv() {
        std::map<AscendString, AscendString> opt = {{"ge.exec.deviceId", "0"},
                                                    {"ge.graphRunMode", "1"}};
        inited_ = GEInitialize(opt) == SUCCESS;
    }
    ~GeEnv() {
        if (inited_)
        GEFinalize();
    }

    bool Ok() { return inited_; }

private:
    bool inited_;
};

static void DumpMeta(void* data) {
  optiling::detail::LiqMetaData* metaDataPtr =
      (optiling::detail::LiqMetaData*)data;

}

int main(int argc, char **argv) {
    GeEnv geEnv;
    if (!geEnv.Ok()) {
        printf("GEInitialize fail\n");
        return -1;
    }

    Graph graph("GraphLightningIndexerQuantMetadata");

    auto metaDataOp = op::LightningIndexerQuantMetadata(
        "GraphLightningIndexerQuantMetadata-0");

    // gen graph
    auto dataOp0 = op::Data("input0").set_attr_index(0); // Data 算子
    if (enableActLenQuery) {
        TensorDesc desc(ge::Shape(actualSeqLengthsQueryShape), FORMAT_ND, DT_INT32);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        desc.SetRealDimCnt(actualSeqLengthsQueryShape.size());
        dataOp0.update_input_desc_x(desc);
        graph.AddOp(dataOp0);
        metaDataOp.set_input_actual_seq_lengths_query(dataOp0);
    }

    auto dataOp1 = op::Data("input1").set_attr_index(0); // Data 算子
    if (enableActLenKey) {
        TensorDesc desc(ge::Shape(actualSeqLengthsKeyShape), FORMAT_ND, DT_INT32);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        desc.SetRealDimCnt(actualSeqLengthsKeyShape.size());
        dataOp1.update_input_desc_x(desc);
        graph.AddOp(dataOp1);
        metaDataOp.set_input_actual_seq_lengths_key(dataOp1);
    } 

    metaDataOp.update_output_desc_metadata(TensorDesc{ge::Shape(metadataShape), FORMAT_ND, DT_INT32});
    metaDataOp.set_attr_batch_size(batchSize);
    metaDataOp.set_attr_query_head_num(numHeadsQ);
    metaDataOp.set_attr_kv_head_num(numHeadsK);
    metaDataOp.set_attr_query_seq_size(seqSizeQ);
    metaDataOp.set_attr_kv_seq_size(seqSizeK);
    metaDataOp.set_attr_layout_query(layoutQuery);
    metaDataOp.set_attr_layout_key(layoutKV);
    metaDataOp.set_attr_sparse_mode(sparseMode);
    metaDataOp.set_attr_pre_tokens(sparseMode);
    metaDataOp.set_attr_next_tokens(sparseMode);
    metaDataOp.set_attr_is_fd(isFD);
    metaDataOp.set_attr_soc_version(socVersion);
    metaDataOp.set_attr_aic_core_num(aicCoreNum);
    metaDataOp.set_attr_aiv_core_num(aivCoreNum);

    graph.AddOp(metaDataOp);

    // run Graph
    std::vector<ge::Operator> inputOps = {dataOp0, dataOp1};
    std::vector<ge::Operator> outputOps = {metaDataOp};
    graph.SetInputs(inputOps).SetOutputs(outputOps);

    aclgrphDumpGraph(graph, dumpFile.c_str(), dumpFile.length());

    std::vector<ge::Tensor> inputTensors;
    std::vector<ge::Tensor> outputTensors;

    if (enableActLenQuery) {
        inputTensors.push_back(
            Tensor{dataOp0.get_input_desc_x(),
                reinterpret_cast<const uint8_t *>(&actualSeqLengthsQuery[0]),
                actualSeqLengthsQuery.size() * sizeof(actualSeqLengthsQuery[0])});
    }

    if (enableActLenKey) {
        inputTensors.push_back(
            Tensor{dataOp1.get_input_desc_x(),
                reinterpret_cast<const uint8_t *>(&actualSeqLengthsKey[0]),
                actualSeqLengthsKey.size() * sizeof(actualSeqLengthsKey[0])});
    }

    std::map<AscendString, AscendString> build_options;
    auto session = std::make_shared<Session>(build_options);
    std::map<AscendString, AscendString> graph_options;
    uint32_t graph_id = 0;
    session->AddGraph(graph_id, graph, graph_options);
    if (session->RunGraph(graph_id, inputTensors, outputTensors)) {
        printf("RunGraph Fail\n");
        return -1;
    }

    auto tensor = outputTensors[0];
    auto data = tensor.GetData();
    auto dataSize = tensor.GetTensorDesc().GetShape().GetShapeSize();
    
    for (uint i = 0; i < 1024; i++) {
        printf("metadata[%u] = %u\n", i, data[i]);
    }

    // optiling::detail::SfaMetaData *metaDataPtr = (optiling::detail::SfaMetaData*)data;
   
    return 0;
}