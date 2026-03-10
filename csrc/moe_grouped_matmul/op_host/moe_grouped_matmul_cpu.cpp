/**
* This program is free software, you can redistribute it and/or modify.
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_grouped_matmul_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define OP_LOGD(nodeName, fmt, ...) printf(fmt, ##__VA_ARGS__); printf("\n")
#define OP_LOGE(nodeName, fmt, ...) printf(fmt, ##__VA_ARGS__); printf("\n")

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t GROUPLIST_INDEX = 2;
namespace optiling {
constexpr uint64_t BEST_L1_PARTA = 256UL * 1024UL;
constexpr uint64_t BEST_L1_PARTB = 128UL * 1024UL;
constexpr uint64_t L1_PARTA_SIZE = 256UL * 1024UL;
constexpr int32_t BEST_BASEN = 256;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr uint32_t FP32_DATATYPE_SIZE = 4;
constexpr int32_t MAX_BASEM = 256;

static inline uint32_t SixteenAlign(uint32_t a, bool up = false) {
    if (up) {
        a += 15U;  // 15: 16 bytes up-align
    }
    return a & ~15U;  // ~15: 16 bytes down-align
}

static inline int64_t SixteenAlign(int64_t a, bool up = false) {
    if (up) {
        a += 15;  // 15: 16 bytes up-align
    }
    return a & ~15;  // ~15: 16 bytes down-align
}

class TilingMoeGroupedMatmulFunc {
 public:
    explicit TilingMoeGroupedMatmulFunc(gert::TilingContext* tiling_context)
        : tiling_context_(tiling_context) {}

    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

 private:
    MoeGroupedMatmulTilingData tiling_data_;
    gert::TilingContext* tiling_context_ = nullptr;

    void SetTilingKey();
    void FillTilingData();
    ge::graphStatus CalMMTiling();
    ge::graphStatus GMMSetMMTiling();
    ge::graphStatus CalcStepKaKb(uint32_t& mm_step_ka, uint32_t& mm_step_kb);
    ge::graphStatus DynamicTIlingSingleN();

    void InitPlatformInfo(matmul_tiling::PlatformInfo& platformInfo);
    void GMMGetPlatformInfo();
    int64_t m_ = 0L;
    int64_t n_ = 0L;
    int64_t k_ = 0L;
    bool transpose_weight = false;
    bool weight_nz = false;
    uint32_t single_m_ = 0;
    uint32_t single_n_ = 0;
    int32_t baseM_ = 0;
    int32_t baseN_ = 0;
    int32_t baseK_ = 0;
    int32_t nz_factor_ = 1;
    uint32_t group_num_ = 0;
    uint32_t core_num_ = 0;
    uint32_t mmDataTypeSize_ = 0;
    size_t sync_workspace_size_ = 0;
    ge::DataType x_dtype;
    uint64_t l1_size, l0a_size, l0b_size, l0c_size, ub_size;
    uint32_t aic_num, aiv_num;
    // SocVersion soc_version;
};

ge::graphStatus TilingMoeGroupedMatmulFunc::CalcStepKaKb(uint32_t& mm_step_ka, uint32_t& mm_step_kb) {
    uint64_t available_l1_size = l1_size;
    if (available_l1_size < L1_PARTA_SIZE) {
      OP_LOGE(tiling_context_->GetNodeName(), "available_l1_size is less than 256k.");
      return ge::GRAPH_FAILED;
    }
    // according to double buffer, recompute the params used for data movement from GM to L1
    uint64_t l1_a_size = baseM_ > baseN_ ? L1_PARTA_SIZE : available_l1_size - L1_PARTA_SIZE;
    uint64_t l1_b_size = available_l1_size - l1_a_size;
    // 2: double buffer
    mm_step_ka = (l1_a_size / 2UL) / (static_cast<uint64_t>(baseM_) * baseK_ * mmDataTypeSize_);
    // 2: double buffer
    mm_step_kb = (l1_b_size / 2UL) / (static_cast<uint64_t>(baseN_) * baseK_ * mmDataTypeSize_);
    if (mm_step_ka == 0 || mm_step_kb == 0) {
      OP_LOGE(tiling_context_->GetNodeName(), "stepka or stepkb cannot be 0.");
      return ge::GRAPH_FAILED;
    }

    if (mm_step_ka > mm_step_kb) {
      mm_step_ka = mm_step_ka / mm_step_kb * mm_step_kb;
    } else if (mm_step_ka < mm_step_kb) {
      mm_step_kb = mm_step_kb / mm_step_ka * mm_step_ka;
    }
    return ge::GRAPH_SUCCESS;
}

void TilingMoeGroupedMatmulFunc::GMMGetPlatformInfo() {
    auto platform_info = platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
    platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
    platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a_size);
    platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b_size);
    platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);
    platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    aic_num = platform_info.GetCoreNumAic();
    aiv_num = platform_info.GetCoreNumAiv();
}

void TilingMoeGroupedMatmulFunc::InitPlatformInfo(matmul_tiling::PlatformInfo& platformInfo) {
    auto platform_info = platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
    platformInfo.socVersion = platform_info.GetSocVersion();
    platformInfo.l1Size = l1_size;
    platformInfo.l0CSize = l0c_size;
    platformInfo.ubSize = ub_size;
    platformInfo.l0ASize = l0a_size;
    platformInfo.l0BSize = l0b_size;
}

ge::graphStatus TilingMoeGroupedMatmulFunc::GMMSetMMTiling() {
    matmul_tiling::DataType matmul_dtype = static_cast<matmul_tiling::DataType>(x_dtype);
    matmul_tiling::PlatformInfo platformInfo;
    InitPlatformInfo(platformInfo);
    // matmul_tiling::MatmulApiTiling mm(platformInfo);
    matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, false);
    if (weight_nz) {
      mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_dtype, transpose_weight);
    } else {
      mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, transpose_weight);
    }
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype);
    mm.SetOrgShape(m_, n_, k_);
    mm.SetShape(m_, baseN_, k_);
    // mm.SetShape(single_m_, single_n_, k_);
    mm.SetFixSplit(baseM_, baseN_, baseK_);
    mm.SetBufferSpace(l1_size, l0c_size, ub_size);

    if (mm.GetTiling(tiling_data_.mm_tiling) == -1) {
      OP_LOGE(tiling_context_->GetNodeName(), "matmul getTiling failed.");
      return ge::GRAPH_FAILED;
    }
    uint32_t mm_step_ka = 1;
    uint32_t mm_step_kb = 1;

    auto ret = CalcStepKaKb(mm_step_ka, mm_step_kb);
    if (ret != ge::GRAPH_SUCCESS) {
      OP_LOGE(tiling_context_->GetNodeName(), "matmul calc stepka or stepkb failed.");
      return ge::GRAPH_FAILED;
    }

    constexpr uint32_t step_m = 1;  // 1: step_m set fixed value 1
    constexpr uint32_t step_n = 1;  // 1: step_n set fixed value 1
    uint32_t mm_depth_a1 = mm_step_ka * DOUBLE_BUFFER_STEPKA_STEPKB * step_m;
    uint32_t mm_depth_b1 = mm_step_kb * DOUBLE_BUFFER_STEPKA_STEPKB * step_n;
    tiling_data_.mm_tiling.set_shareMode(0);
    tiling_data_.mm_tiling.set_dbL0C(1);  // disable double buffer for LOC
    tiling_data_.mm_tiling.set_baseM(baseM_);  // set precomputed baseM
    tiling_data_.mm_tiling.set_baseN(baseN_);  // set precomputed baseN
    tiling_data_.mm_tiling.set_baseK(baseK_);  // set precomputed baseK
    tiling_data_.mm_tiling.set_stepKa(mm_step_ka);  // set precomputed mmStepKa
    tiling_data_.mm_tiling.set_depthA1(mm_depth_a1);  // set precomputed mmDepthA1
    tiling_data_.mm_tiling.set_stepKb(mm_step_kb);  // set precomputed mmStepKb
    tiling_data_.mm_tiling.set_depthB1(mm_depth_b1);  // set precomputed mmDepthB1
    tiling_data_.mm_tiling.set_stepM(step_m);  // set precomputed stepM
    tiling_data_.mm_tiling.set_stepN(step_n);  // set precomputed stepN
    OP_LOGD(context->GetNodeName(), "GMM_tiling: baseM is %d, baseK is %d, baseN is %d, transpose_weight is %d, weight_nz is %d", baseM_, baseK_, baseN_, transpose_weight, weight_nz);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMoeGroupedMatmulFunc::CalMMTiling() {
    baseN_ = BEST_BASEN;
    if (x_dtype == ge::DT_BF16 || x_dtype == ge::DT_FLOAT16) {
      mmDataTypeSize_ = 2;
    } else {
      OP_LOGE(tiling_context_->GetNodeName(), "only support bf16 or fp16.");
      return ge::GRAPH_FAILED;
    }

    baseK_ = static_cast<int32_t>((l0b_size / DOUBLE_BUFFER_L0A_L0B) / (static_cast<uint32_t>(baseN_) * mmDataTypeSize_));
    baseK_ = static_cast<int32_t>(SixteenAlign(static_cast<int64_t>(baseK_)));
    uint32_t max_base_m = static_cast<uint32_t>(l0c_size /
                                              (static_cast<uint32_t>(baseN_) * FP32_DATATYPE_SIZE));
    baseM_ = std::min<uint32_t>((l0a_size / DOUBLE_BUFFER_L0A_L0B) /
            (static_cast<uint32_t>(baseK_) * mmDataTypeSize_), max_base_m);
    baseM_ = baseM_ > m_ ? SixteenAlign(m_, true) : SixteenAlign(static_cast<uint32_t>(baseM_));

    if (baseM_ > MAX_BASEM) {
      baseM_ = MAX_BASEM;
    }

    if (baseM_ == 0 || baseK_ == 0) {
      OP_LOGE(tiling_context_->GetNodeName(), "baseM_ or baseN_ cannot be 0.");
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMoeGroupedMatmulFunc::Init() {
    GMMGetPlatformInfo();
    // only support singlex、singleweight、singley
    bool is_single_x = (tiling_context_->GetDynamicInputTensor(X_INDEX, 1) == nullptr);
    bool is_single_weight = (tiling_context_->GetDynamicInputTensor(WEIGHT_INDEX, 1) == nullptr);
    bool is_single_y = (tiling_context_->GetOutputShape(1) == nullptr);
    transpose_weight = static_cast<bool>(*(tiling_context_->GetAttrs()->GetAttrPointer<bool>(0)));
    if (!(is_single_x && is_single_weight && is_single_y)) {
      OP_LOGE(tiling_context_->GetNodeName(), "only support singlex and singleweight and singley.");
      return ge::GRAPH_FAILED;
    }

    auto x_shape = tiling_context_->GetDynamicInputShape(X_INDEX, 0)->GetOriginShape();
    auto weight_shape = tiling_context_->GetDynamicInputShape(WEIGHT_INDEX, 0)->GetOriginShape();
    auto group_list_shape = tiling_context_->GetInputShape(GROUPLIST_INDEX)->GetOriginShape();
    auto y_shape = tiling_context_->GetOutputShape(0)->GetOriginShape();
    auto weight_desc = tiling_context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    auto weight_format = static_cast<ge::Format>(ge::GetPrimaryFormat(weight_desc->GetStorageFormat()));
    x_dtype = tiling_context_->GetDynamicInputDesc(X_INDEX, 0)->GetDataType();

    // printf("weight_format %d\n", weight_format);
    weight_nz = weight_format == ge::FORMAT_FRACTAL_NZ;

    // check input shape
    if (x_shape.GetDimNum() != 2 || y_shape.GetDimNum() != 2) {
      OP_LOGE(tiling_context_->GetNodeName(), "the dimNum of input and output should be 2, but got %zu, %zu.", static_cast<size_t>(x_shape.GetDimNum()), static_cast<size_t>(y_shape.GetDimNum()));
      return ge::GRAPH_FAILED;
    }
    uint32_t weight_dim1, weight_dim2;
    n_ = transpose_weight ? weight_shape.GetDim(1) : weight_shape.GetDim(2);

    if (group_list_shape.GetDimNum() != 2) {
      OP_LOGE(tiling_context_->GetNodeName(), "only support key-value mode of groupList, the dimNum of groupList should be 2, but got %zu.", static_cast<size_t>(group_list_shape.GetDimNum()));
      return ge::GRAPH_FAILED;
    }

    m_ = x_shape.GetDim(x_shape.GetDimNum() - 2);
    k_ = x_shape.GetDim(x_shape.GetDimNum() - 1);
    group_num_ = weight_shape.GetDim(0);

    if (weight_shape.GetDim(0) != group_num_) {
      OP_LOGE(tiling_context_->GetNodeName(), "the dim0 of input weight should be equal to input groupList, but got %zu, %zu.", static_cast<size_t>(weight_shape.GetDim(0)), static_cast<size_t>(group_list_shape.GetDim(0)));
    }
    single_m_ = 128;
    single_n_ = 256;

    core_num_ = aic_num;
    auto n_task_num = (n_ + single_n_ - 1) / single_n_;
    auto task_num = m_ * n_task_num;
    if (task_num < core_num_) {
      core_num_ = task_num;
    }

    auto platform_info = platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
    sync_workspace_size_ = static_cast<size_t>(platform_info.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

void TilingMoeGroupedMatmulFunc::SetTilingKey() {
    uint64_t tiling_key = 10UL;
    if (transpose_weight) {
      tiling_key = tiling_key + 1UL;
    }
    tiling_context_->SetTilingKey(tiling_key);
}

void TilingMoeGroupedMatmulFunc::FillTilingData() {
    tiling_data_.set_m(static_cast<uint32_t>(m_));
    tiling_data_.set_n(static_cast<uint32_t>(n_));
    tiling_data_.set_k(static_cast<uint32_t>(k_));
    tiling_data_.set_single_m(single_m_);
    tiling_data_.set_single_n(single_n_);
    tiling_data_.set_group_num(group_num_);
    tiling_data_.set_core_num(core_num_);
}

ge::graphStatus TilingMoeGroupedMatmulFunc::RunKernelTiling() {
    auto ret = CalMMTiling();
    if (ret != ge::GRAPH_SUCCESS) {
      OP_LOGE(context->GetNodeName(), "cal mmtiling failed.");
      return ge::GRAPH_FAILED;
    }
    ret = GMMSetMMTiling();
    if (ret != ge::GRAPH_SUCCESS) {
      OP_LOGE(context->GetNodeName(), "gmm set mmtiling failed.");
      return ge::GRAPH_FAILED;
    }
    SetTilingKey();
    FillTilingData();
    size_t userWorkspaceSize = 0;
    size_t *currentWorkspace = tiling_context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + sync_workspace_size_;
    tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                              tiling_context_->GetRawTilingData()->GetCapacity());
    tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
    tiling_context_->SetBlockDim(core_num_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForMoeGroupedMatmulFunc(gert::TilingContext *context){
    TilingMoeGroupedMatmulFunc tilingObject(context);
    auto ret = tilingObject.Init();
    if(ret != ge::GRAPH_SUCCESS){
      OP_LOGE(context->GetNodeName(), "tiling Init failed.");
      return ge::GRAPH_FAILED;
    }
    ret = tilingObject.RunKernelTiling();
    return ret;
}

struct MatmulAllreduceAddRmsnormCompileInfo1 {};
ge::graphStatus TilingParseForMatmulAllreduceAddRmsnorm1(gert::TilingParseContext *context)
{
    // (void)context;
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(MoeGroupedMatmul)
    .Tiling(TilingForMoeGroupedMatmulFunc)
    .TilingParse<MatmulAllreduceAddRmsnormCompileInfo1>(TilingParseForMatmulAllreduceAddRmsnorm1);

} // namespace optiling

