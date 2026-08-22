/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file dequant_checker.h
 * \brief
 */

#ifndef DEQUANT_CHECKER_H
#define DEQUANT_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
class DequantChecker : public BaseChecker {
public:
    DequantChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~DequantChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

private:
    // enableNonQuant 相关校验函数
    ge::graphStatus CheckExistenceNoquant(const FiaTilingInfo &fiaInfo) const;

    // enableFullQuant 相关校验函数
    ge::graphStatus CheckDataTypeSupportFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleDtypeMLAFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleDtypeGQAPerblock(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleDtypeGQAPertensor(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleDtypeMXFP8Fullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleDtypeFP8GQAFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleDtypeFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantModeMLAFullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckDequantModeGQAPerblock(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckDequantModeGQAPertensor(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckDequantModeMXFP8Fullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckDequantModeFP8GQAFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantModeFullquant(const FiaTilingInfo &fiaInfo);

    ge::graphStatus CheckTensorExistFullquant(const FiaTilingInfo &fiaInfo, const gert::Tensor *tensor,
        const std::string &quantModeName, const std::string &inputName) const;
    ge::graphStatus CheckTensorNotExistFullquant(const FiaTilingInfo &fiaInfo, const gert::Tensor *tensor,
        const std::string &quantModeName, const std::string &inputName) const;
    ge::graphStatus CheckExistencePertensorFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckExistenceMLAFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckExistencePerblockFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckExistenceMXFP8Fullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckExistenceFP8GQAFullquant(const FiaTilingInfo &fiaInfo);

    ge::graphStatus CheckFeaturePertensorFullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeaturePerblockFullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureMLAFullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureSupportFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureMXFP8Fullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureFP8GQAFullquant(const FiaTilingInfo &fiaInfo);

    ge::graphStatus CheckDequantScaleKVMLAFullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckDequantScaleQueryMLAFullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckDequantScaleShapePertensor(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckDequantScaleShapeMXFP8(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleBnNBsDShapeMXFP8(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckDequantScaleNZShapeMXFP8(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckQuantScale1ShapePerblock(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckQuantScale1ShapeMXFP8(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckQuantScale1ShapeFP8GQA(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleShapePerblockNTD(const FiaTilingInfo &fiaInfo,
        const gert::Shape &dequantScaleQueryShape, const gert::Shape &keyAntiquantScaleShape,
        const gert::Shape &valueAntiquantScaleShape, uint32_t fp8QBlockSize, uint32_t fp8KVBlockSize) const;
    ge::graphStatus CheckDequantScaleShapePerblockNonNTD(const FiaTilingInfo &fiaInfo,
        const gert::Shape &dequantScaleQueryShape, const gert::Shape &keyAntiquantScaleShape,
        const gert::Shape &valueAntiquantScaleShape, uint32_t fp8QBlockSize, uint32_t fp8KVBlockSize) const;
    ge::graphStatus CheckDequantScaleShapePerblock(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckDequantScaleShapeFP8GQA(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleShapeFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDequantScaleShapeCrossFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputDTypeFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputLayoutPerblock(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckInputLayoutPertensor(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckInputLayoutMLAFullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckInputLayoutMXFP8Fullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckInputLayoutFP8GQAFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckStrideFP8GQAFullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckInputLayoutFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputAxisFullquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckN1SizeFullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckN2SizeFullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckGSizeFullquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckDSizeFullquant(const FiaTilingInfo &fiaInfo) const;

    // enableAntiQuant 相关校验函数
    // SinglePara
    ge::graphStatus CheckSingleParaForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckAntiquantModeForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckInputKVTypeForAntiquantPerChannel(const FiaTilingInfo &fiaInfo,
        ge::DataType inputKvType, const gert::Tensor *keyAntiquantScaleTensor);
    ge::graphStatus CheckInputKVTypeForAntiquantPerToken(const FiaTilingInfo &fiaInfo,
        ge::DataType inputKvType);
    ge::graphStatus CheckInputKVTypeForAntiquantPerTokenPA(const FiaTilingInfo &fiaInfo,
        ge::DataType inputKvType);
    ge::graphStatus CheckInputKVTypeForAntiquantPerTensorHead(const FiaTilingInfo &fiaInfo,
        ge::DataType inputKvType);
    ge::graphStatus CheckInputKVTypeForAntiquantPerTokenHead(const FiaTilingInfo &fiaInfo,
        ge::DataType inputKvType);
    ge::graphStatus CheckInputKVTypeForAntiquantPerTokenHeadPA(const FiaTilingInfo &fiaInfo,
        ge::DataType inputKvType);
    ge::graphStatus CheckInputKVTypeForAntiquantMixed(const FiaTilingInfo &fiaInfo,
        ge::DataType inputKvType);
    ge::graphStatus CheckInputKVTypeForAntiquantPerTokenGroup(const FiaTilingInfo &fiaInfo,
        ge::DataType inputKvType);
    ge::graphStatus CheckInputKVTypeForAntiquant(const FiaTilingInfo &fiaInfo);
    
    // Existence
    ge::graphStatus CheckExistenceForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckScaleExistenceForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckDescExistenceForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckOffsetExistenceForAntiquant(const FiaTilingInfo &fiaInfo) const;

    // Feature
    ge::graphStatus CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureLayoutForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureQuerySForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeaturePAForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureRopeForAntiquant(const FiaTilingInfo &fiaInfo) const;

    // MultiPara
    ge::graphStatus CheckMultiParaForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckScaleTypeForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckScaleShapeForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckOffsetTypeForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckOffsetShapeForAntiquant(const FiaTilingInfo &fiaInfo) const;

    ge::graphStatus CheckKScaleShapeForPerChannelPerTensorMode(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckKScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckKScaleShapeForPerTensorHeadMode(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckKScaleShapeForPerTokenHeadMode(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckKScaleShapeForPerTokenPAMode(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckKScaleShapeForPerTokenHeadPAMode(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckKScaleShapeForPerTokenGroupMode(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckVScaleShapeForPerTokenMode(const FiaTilingInfo &fiaInfo) const;
    int64_t GetValueScaleActualKVlens4TNDNoPa(const FiaTilingInfo &fiaInfo) const;

private:
    bool enableQKVPerblockQuant_ = false;
    bool enableQKVPertensorQuant_ = false;
    bool enableQPerTokenHeadKVPerTensor_ = false;
    bool enableQKVMxfp8FullQuant_ = false;
    bool enableQKPerTokenHeadVPerHead_ = false;
};

} // namespace optiling
#endif // DEQUANT_CHECKER_H