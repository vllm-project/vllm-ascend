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
 * \file const_args.hpp
 * \brief
 */


#ifndef CONST_ARGS_HPP
#define CONST_ARGS_HPP
constexpr static uint64_t MB_SIZE = 1024 * 1024UL;
constexpr static int32_t NUMS_PER_FLAG = 16;
constexpr static int32_t CACHE_LINE = 512;
constexpr static int32_t RESET_VAL = 0xffff;
constexpr static int32_t FLAGSTRIDE = 16;
constexpr static int32_t UB_ALIGN = 32;
constexpr static int32_t ALIGN_128 = 128;
constexpr uint16_t CROSS_CORE_FLAG_MAX_SET_COUNT = 15;
constexpr uint32_t MAX_EXPERTS_PER_RANK = 32;
constexpr static uint16_t SYNCFLAGC2V = 9;
constexpr static uint16_t SYNCFLAGV2C = 10;
constexpr static uint32_t SERVER_RANK_SIZE_A2 = 8;
constexpr static uint32_t kMaxDequantSyncGroups = 16;
#endif