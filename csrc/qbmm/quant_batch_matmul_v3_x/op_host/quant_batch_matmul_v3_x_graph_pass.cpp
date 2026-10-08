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
 * \file quant_batch_matmul_v3_x_graph_pass.cpp
 * \brief Custom graph pass: rewrite x2 Data-node metadata to FRACTAL_NZ so the kernel sees the storage layout it expects.
 * \author Feodor Pisnitchenko
 *
 * Runs at kAfterBuiltinFusionPass so the rewrite survives all
 * builtin shape/format recomputation and fusion passes.
 */
#include <string>
#include <vector>

#include "graph/graph.h"
#include "graph/gnode.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "graph/ascend_string.h"
#include "register/register_custom_pass.h"

namespace {

constexpr int64_t K_BLOCK = 32;
constexpr int64_t N_BLOCK = 16;

inline int64_t CeilDiv(int64_t v, int64_t a)
{
    return (v + a - 1) / a;
}

bool IsType(const ge::GNode &node, const char *want)
{
    ge::AscendString s;
    if (node.GetType(s) != ge::GRAPH_SUCCESS) return false;
    const char *got = s.GetString();
    return got != nullptr && std::string(got) == want;
}

bool IsTypePtr(const ge::GNodePtr &node, const char *want)
{
    if (node == nullptr) return false;
    ge::AscendString s;
    if (node->GetType(s) != ge::GRAPH_SUCCESS) return false;
    const char *got = s.GetString();
    return got != nullptr && std::string(got) == want;
}

ge::graphStatus DeriveKN(const ge::GNode &qbm, int64_t &K, int64_t &N)
{
    ge::TensorDesc x1_desc, y_desc;
    if (qbm.GetInputDesc(0, x1_desc) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (qbm.GetOutputDesc(0, y_desc) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

    auto x1_dims = x1_desc.GetShape().GetDims();
    auto y_dims = y_desc.GetShape().GetDims();
    if (x1_dims.size() < 2 || y_dims.empty()) return ge::GRAPH_FAILED;

    bool transpose_x1 = false;
    (void)qbm.GetAttr(ge::AscendString("transpose_x1"), transpose_x1);

    K = transpose_x1 ? x1_dims[x1_dims.size() - 2] : x1_dims[x1_dims.size() - 1];
    N = y_dims[y_dims.size() - 1];
    return (K > 0 && N > 0) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::Status FixX2DataNodes(ge::GraphPtr &graph, ge::CustomPassContext &)
{
    if (graph == nullptr) return ge::SUCCESS;

    int fixed = 0;
    auto all_nodes = graph->GetAllNodes();
    for (auto &node : all_nodes) {
        if (!IsType(node, "QuantBatchMatmulV3X")) continue;

        int64_t K = 0;
        int64_t N = 0;
        if (DeriveKN(node, K, N) != ge::GRAPH_SUCCESS) {
            continue;
        }

        auto peer = node.GetInDataNodesAndPortIndexs(1);
        ge::GNodePtr upstream = peer.first;
        int32_t src_port = peer.second;
        if (!IsTypePtr(upstream, "Data") && !IsTypePtr(upstream, "RefData")) {
            continue;
        }

        // Peek at origin_shape on output_desc to validate (must be 2D).
        ge::TensorDesc peek;
        if (upstream->GetOutputDesc(src_port, peek) != ge::GRAPH_SUCCESS) continue;
        auto orig_dims = peek.GetOriginShape().GetDims();
        if (orig_dims.size() != 2) {
            continue;
        }

        int64_t K1 = CeilDiv(K, K_BLOCK);
        int64_t N1 = CeilDiv(N, N_BLOCK);
        ge::Shape storage_shape(std::vector<int64_t>{K1, N1, N_BLOCK, K_BLOCK});
        ge::Shape origin_shape(std::vector<int64_t>{N, K});

        // Fetch each descriptor separately. SetShape / SetOriginShape /
        // SetSize must all be called: aclmdlGetInputSizeByIndex reads the
        // size attr (separate from shape.dim) so missing SetSize leaves the
        // runtime byte count at its default of 1 or 64.
        int64_t total_bytes = K1 * N1 * N_BLOCK * K_BLOCK;  // int8: elem==byte
        auto patch = [&](ge::TensorDesc &d) {
            d.SetShape(storage_shape);
            d.SetOriginShape(origin_shape);
            d.SetFormat(ge::FORMAT_FRACTAL_NZ);
            d.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);
            d.SetSize(total_bytes);
        };

        ge::TensorDesc in_desc;
        if (upstream->GetInputDesc(0, in_desc) == ge::GRAPH_SUCCESS) {
            patch(in_desc);
            if (upstream->UpdateInputDesc(0, in_desc) != ge::GRAPH_SUCCESS) {
                continue;
            }
        }

        ge::TensorDesc out_desc;
        if (upstream->GetOutputDesc(src_port, out_desc) == ge::GRAPH_SUCCESS) {
            patch(out_desc);
            if (upstream->UpdateOutputDesc(src_port, out_desc) != ge::GRAPH_SUCCESS) {
                continue;
            }
        }

        // Patch QBM.x2 input_desc so tiling sees the 4D storage shape.
        // Tiling reads x2.storage_shape: dim_cnt >= 4 -> FRACTAL_NZ branch
        // (N = N1*N0); else ND branch (N = transpose_x2 ? dim[-2] : dim[-1]).
        // Without this, x2.shape.dim stays [K, N] (2D) and the ND branch
        // returns the wrong N.
        ge::TensorDesc qbm_x2;
        if (node.GetInputDesc(1, qbm_x2) == ge::GRAPH_SUCCESS) {
            qbm_x2.SetShape(storage_shape);         // [K1, N1, 16, 32]
            qbm_x2.SetOriginShape(origin_shape);    // [N, K]
            (void)node.UpdateInputDesc(1, qbm_x2);
        }

        ++fixed;
    }

    (void)fixed;
    return ge::SUCCESS;
}

}  // namespace

#ifdef OP_PROTO_LIB
// kAfterBuiltinFusionPass: runs after InferShape / InferFormat and all
// builtin fusion passes, which is the earliest point at which the Data
// node's output_desc has FRACTAL_NZ metadata propagated. kAfterInferShape
// fires before InferFormat and the size attr is still default.
REGISTER_CUSTOM_PASS("QuantBatchMatmulV3XFixX2DataNode")
    .CustomPassFn(FixX2DataNodes)
    .Stage(ge::CustomPassStage::kAfterBuiltinFusionPass);
#endif
