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
 * \file infer_flash_attention_comm.h
 * \brief
 */
#ifndef INFER_FLASH_ATTENTION_COMM_H
#define INFER_FLASH_ATTENTION_COMM_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "attenmask.h"
#include "pse.h"

constexpr static int64_t SPARSE_MODE_INT_DEFAULT = 2147483647;

// INPUT_T - means data type for input
// T       - means data type when calc

#define CHILD_SPEC_TEMPLATE \
    template <typename INPUT_T, typename T, ImplModeEnum implMode, LayOutTypeEnum layout, \
    S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType, \
    DTemplateType dVTemplateType, PseTypeEnum pseMode, bool hasAtten, bool hasDrop, bool hasRope, \
    typename OUTPUT_T, bool isInfer, bool isPa, bool isFd>

// 伪量化模板参数
#define CHILD_SPEC_TEMPLATE_ANTI \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, ImplModeEnum implMode,    \
    LayOutTypeEnum layout, S1TemplateType s1TemplateType, S2TemplateType s2TemplateType,    \
    DTemplateType dTemplateType, DTemplateType dVTemplateType, PseTypeEnum pseMode, AntiquantTypeEnum antiquantMode,\
    bool hasAtten, bool hasDrop, bool hasRope, bool isInfer, bool isPa, bool isFd>

#define CHILD_SPEC_TEMPLATE_ARGS \
    INPUT_T, T, implMode, layout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, \
    pseMode, hasAtten, hasDrop, hasRope, OUTPUT_T, isInfer, isPa, isFd

// 伪量化模板参数列表
#define CHILD_SPEC_TEMPLATE_ARGS_ANTI \
    Q_T, KV_T, T, OUTPUT_T, implMode, layout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, \
    pseMode, antiquantMode, hasAtten, hasDrop, hasRope, isInfer, isPa, isFd

#define TEMPLATE_INTF \
    template <typename INPUT_T, typename T, ImplModeEnum implMode, LayOutTypeEnum layout, \
    S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType, \
    DTemplateType dVTemplateType, PseTypeEnum pseMode, bool hasAtten, bool hasDrop, bool hasRope,\
    typename OUTPUT_T, bool isInfer, bool isPa, bool isFd, bool useDn>

#define TEMPLATE_INTF_ARGS \
    INPUT_T, T, implMode, layout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, \
    pseMode, hasAtten, hasDrop, hasRope, OUTPUT_T, isInfer, isPa, isFd, useDn

#define ANTIQUANT_CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(Q_T) \
    X(KV_T) \
    X(T) \
    X(OUTPUT_T)

#define ANTIQUANT_CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(implMode, ImplModeEnum, ImplModeEnum::AA_HIGH_PRECISION) \
    X(layout, LayOutTypeEnum, LayOutTypeEnum::None) \
    X(s1TemplateType, S1TemplateType, S1TemplateType::Aligned128) \
    X(s2TemplateType, S2TemplateType, S2TemplateType::Aligned128) \
    X(dTemplateType, DTemplateType, DTemplateType::Aligned128) \
    X(dVTemplateType, DTemplateType, DTemplateType::Aligned128) \
    X(pseMode, PseTypeEnum, PseTypeEnum::PSE_NONE_TYPE) \
    X(antiquantMode, AntiquantTypeEnum, AntiquantTypeEnum::PER_CHANNEL) \
    X(hasAtten, bool, false) \
    X(hasDrop, bool, false) \
    X(hasRope, bool, false) \
    X(isInfer, bool, false) \
    X(isPa, bool, false) \
    X(isFd, bool, false)

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(INPUT_T) \
    X(T) \
    X(OUTPUT_T)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(implMode, ImplModeEnum, ImplModeEnum::AA_HIGH_PRECISION) \
    X(layout, LayOutTypeEnum, LayOutTypeEnum::None) \
    X(s1TemplateType, S1TemplateType, S1TemplateType::Aligned128) \
    X(s2TemplateType, S2TemplateType, S2TemplateType::Aligned128) \
    X(dTemplateType, DTemplateType, DTemplateType::Aligned128) \
    X(dVTemplateType, DTemplateType, DTemplateType::Aligned128) \
    X(pseMode, PseTypeEnum, PseTypeEnum::PSE_NONE_TYPE) \
    X(hasAtten, bool, false) \
    X(hasDrop, bool, false) \
    X(hasRope, bool, false) \
    X(isInfer, bool, false) \
    X(isPa, bool, false) \
    X(isFd, bool, false)

/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type name = default_val,

#define TEMPLATES_DEF \
template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 2. 生成不带带默认值的模版Template */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT \
template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. 生成有默认值, 不带ChildClass的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

/*伪量化模板参数*/
#define ANTIQUANT_TEMPLATE_ARGS \
    ANTIQUANT_CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    ANTIQUANT_CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

/*Antiquant Processor */
#define ANTIQUANT_PROCESSOR_TEMPLATE_DEF \
    ANTIQUANT_CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
    ANTIQUANT_CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end

#define ANTIQUANT_PROCESSOR_ARGS \
    ANTIQUANT_CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    ANTIQUANT_CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) true

/* 4. 生成BASE的有默认值的Template, BASE带ChildClass*/
#define TEMPLATES_DEF_BASE \
template < \
    typename ChildClass, \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 5. 生成BASE的没有默认值的Template, BASE带ChildClass */
#define TEMPLATES_DEF_BASE_NO_DEFAULT \
template <\
    typename ChildClass, \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 6. 生成BASE的BaseArgs, BASE带ChildClass */
#define TEMPLATE_BASE_ARGS \
    ChildClass, \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end
#endif  // INFER_FLASH_ATTENTION_COMM_H