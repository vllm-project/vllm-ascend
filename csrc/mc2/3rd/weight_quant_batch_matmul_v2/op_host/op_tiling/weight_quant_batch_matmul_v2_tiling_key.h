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
 * \file weight_quant_batch_matmul_v2_tiling_key.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_KEY_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_KEY_H

namespace optiling {

// 对应0位 平台大类
enum class Mc2SocVersionType : uint8_t {
    RESERVERD = 0,
    SUPPORT_L0C_TO_OUT = 1,
    SUPPORT_L1_TO_BT_BF16 = 2,
    SUPPORT_MMAD_S8S4 = 3,
};

// 对应1位 平台小类
enum class Mc2SocVersionSubType {
    RESERVERD = 0,
};

// 对应2-3位 伪量化场景
enum class Mc2QuantizationScenario {
    DEFAULT = 0,
};

// 对应4位 算法大类
enum class Mc2OptimizationAlgorithmCategory {
    VECTOR_ANTIQUANT = 0,
    MULTI_SCALE_DEQUANT = 1,
    FIXPIPE_ANTIQUANT = 2,
};

// 对应5位 算法小类
enum class Mc2OptimizationAlgorithmSubCategory : uint8_t {
    VDEFAULT = 0,
    SPLIT_K = 1,
    N_FIRST_TAIL_RESPLIT = 2,
    N_FIRST_BASIC_BLOCK = 3,
    ASW = 4,
};

// 对应6-9位 fixp模板自定义组合
enum class Mc2FixpipeConfiguration : uint16_t {
    A_NORMAL_LOAD = 0,
    A_SINGLE_M_SINGLE_K_FULL_LOAD = 1,
};

enum class Mc2CustomSplitKConfiguration {
    A_NORMAL_LOAD = 0,
    A_MK_FULL_LOAD = 1,
};

// 对应14位 表示转置场景，transA/transB
enum class Mc2TransposeSituation {
    A_NOT_TRANS_B_NOT_TRANS = 0,
    A_NOT_TRANS_B_TRANS = 1,
    A_TRANS_B_NOT_TRANS = 2,
    A_TRANS_B_TRANS = 3,
};

// 对应17位 可选输入是否存在 hasAntiquantOffset/hasBias
enum class Mc2OptionInputSituation {
    ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST = 0,
    ANTIQUANT_OFFSET_NOT_EXIST_BIAS_EXIST = 1,
    ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST = 2,
    ANTIQUANT_OFFSET_EXIST_BIAS_EXIST = 3,
    ANTIQUANT_OFFSET_NOT_EXIST_BIAS_FP32_EXIST = 4,
    ANTIQUANT_OFFSET_EXIST_BIAS_FP32_EXIST = 6,
};

class Mc2TilingKeyConfigure {
public:
    // 对应0-1位 平台大类，平台小类
    uint8_t socVersionType = 0;

    // 对应2-3位 伪量化场景
    uint8_t quantizationScenario = 0;

    // 对应4-5位 算法大类、算法小类
    uint8_t algorithm = 0;

    // 对应14位 表示转置场景，transA/transB
    uint8_t transposeSituation = 0;

    // 对应15位 表示反量化的类型 perchannel/pertensor/perGroup
    uint8_t antiquantType = 0;

    // 对应16位 表示量化的类型 perchannel/perTensor/None
    uint8_t quantType = 0;

    // 对应17位 可选输入是否存在 hasAntiquantOffset/hasBias
    uint8_t optionInputSituation = 0;

    // 对应18位 weight的数据类型 weightNd/weightNz
    uint8_t weightFormat = 0;

    // 对应6-9位 模板自定义组合
    uint16_t templateCustom = 0;

    // 对应10-13位 api常量化保留位
    uint16_t apiConstexpr = 0;

public:
    void PrintTilingKeyLog() const
    {
        std::stringstream ss;
        ss << "socVersionType: " << static_cast<uint32_t>(this->socVersionType)
           << " quantizationScenario: " << static_cast<uint32_t>(this->quantizationScenario)
           << " algorithm: " << static_cast<uint32_t>(this->algorithm)
           << " transposeSituation: " << static_cast<uint32_t>(this->transposeSituation)
           << " antiquantType: " << static_cast<uint32_t>(this->antiquantType)
           << " quantType: " << static_cast<uint32_t>(this->quantType)
           << " optionInputSituation: " << static_cast<uint32_t>(this->optionInputSituation)
           << " weightFormat: " << static_cast<uint32_t>(this->weightFormat)
           << " templateCustom: " << static_cast<uint32_t>(this->templateCustom)
           << " apiConstexpr: " << static_cast<uint32_t>(this->apiConstexpr);
        OP_LOGI("Mc2WeightQuantBatchMatmulV2", "tilingKeyConfigure: %s", ss.str().c_str());
        return;
    }

    uint64_t GenTilingKey() const
    {
        PrintTilingKeyLog();
        uint64_t tilingKey = 0;
        tilingKey = this->socVersionType;
        // 伪量化场景占2位，需要乘100空余出对应的位数
        tilingKey = tilingKey * 100 + this->quantizationScenario;
        // 算法分类占2位(算法大类、算法小类)，需要乘100空余出对应的位数
        tilingKey = tilingKey * 100 + this->algorithm;
        // 模板自定义场景占4位，需要乘10000空余出对应的位数
        tilingKey = tilingKey * 10000 + this->templateCustom;
        // api常量化定义占4位，需要乘10000空余出对应的位数
        tilingKey = tilingKey * 10000 + this->apiConstexpr;
        // 输入转置情况占1位，需要乘10空余出对应的位数
        tilingKey = tilingKey * 10 + this->transposeSituation;
        // weight的量化类型占1位，需要乘10空余出对应的位数
        tilingKey = tilingKey * 10 + this->antiquantType;
        // c矩阵量化类型占1位，需要乘10空余出对应的位数
        tilingKey = tilingKey * 10 + this->quantType;
        // offset\bias的可选输入是否存在占1位，需要乘10空余出对应的位数
        tilingKey = tilingKey * 10 + this->optionInputSituation;
        // weight的format占1位，需要乘10空余出对应的位数
        tilingKey = tilingKey * 10 + this->weightFormat;
        return tilingKey;
    }
};

} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_KEY_H
