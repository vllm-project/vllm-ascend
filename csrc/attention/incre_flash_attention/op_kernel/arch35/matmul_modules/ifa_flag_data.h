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
 * \file ifa_flag_data.h
 * \brief
 */

#ifndef IFA_FLAG_DATA_H
#define IFA_FLAG_DATA_H

struct IFAFlagData {
  uint64_t tscmIdx : 2;  // query tscm que idx
  uint64_t tscmReuse : 1;
  uint64_t rsvd : 61;  // 保留
};

#endif  // IFA_FLAG_DATA_H
