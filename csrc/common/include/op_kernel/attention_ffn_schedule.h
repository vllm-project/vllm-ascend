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
 * \file attention_ffn_schedule.h
 * \brief
 */
#ifndef INCLUDE_KERNEL_ATTENTION_FFN_SCHEDULE_H
#define INCLUDE_KERNEL_ATTENTION_FFN_SCHEDULE_H
#include <cstdint>
 
namespace aicpu {
#pragma pack(push, 1)
constexpr int32_t kValidFlag = 1;
constexpr int32_t kInvalidFlag = 0;

constexpr int32_t kExecuteModeLoop = 1;
constexpr int32_t kExecuteModeOnce = 0;

constexpr int32_t kScheduleModeFfn = 0;

struct FfnDataDesc {
  volatile int32_t flag;
  volatile int32_t layer_id;
  volatile int32_t expert_ids[0];
};
 
struct AttentionDataDesc {
  int32_t flag[0];
};
 
struct ScheduleContext {
  struct CommonArea {
    uint32_t session_num;  // Number of attention nodes
    uint32_t micro_batch_num;
    uint32_t micro_batch_size;
    uint32_t selected_expert_num; 
    uint32_t expert_num; // Number of experts per layer, including routing experts and shared experts.
    uint32_t attn_to_ffn_token_size;  // Each token in the Ffn window data area has a space size aligned to 512 bytes.
    uint32_t ffn_to_attn_token_size;  // Each token in the Attention window data area has a space size aligned to 512 bytes.
    int32_t schedule_mode;  // 0: Ffn only 1: Attention only
    int8_t reserve0[96];
  };
  struct ControlArea {
    int32_t run_flag;  // 0 : exited  1 : running
    int8_t reserve2[124];
  };
  struct FfnArea {
    // ffn area
    uint64_t token_info_buf;  // Points to device memory.
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;  // Points to device memory.
    uint64_t token_data_buf_size;
    uint64_t polling_index;  // For synchronous computation only: records the micro-batch ids to be processed internally by the ffn worker scheduler.
    int8_t reserve3[88];

    // ffn out area
    uint64_t layer_ids_buf;  // Points to a device memory region that stores the organized layer ids, with an array size of session_num.
    uint64_t layer_ids_buf_size;  // Total size: session_num * sizeof(int32_t)
    uint64_t session_ids_buf;  //  Points to a device memory region that stores the organized session ids. The array size is session_num.
    uint64_t session_ids_buf_size;  // Total size: session_num * sizeof(int32_t)
    uint64_t micro_batch_ids_buf;  // Points to a device memory region that stores the organized micro batch ids. The array size is session_num.
    uint64_t micro_batch_ids_buf_size;  // Total size: session_num * sizeof(int32_t)
    uint64_t expert_ids_buf;  //  Points to a device memory region that stores the organized expert ids. The tensor dimensions are [session_num, batch_size, selected_expert_num].
    uint64_t expert_ids_buf_size;  // Total memory size: session_num * batch_size * selected_expert_num * sizeof(int32_t)
    uint32_t out_num;  // Indicates the number of sessions that have been processed.
    int8_t reserve4[60];
  };

  struct AttentionArea {
    // attention area
    uint64_t token_info_buf;  // Points to device memory.
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;  // Points to device memory.
    uint64_t token_data_buf_size;
    uint32_t micro_batch_id; // Records the latest ready micro batch id.
    int8_t reserve5[92];
  };

  // common area
  CommonArea common;
  ControlArea control;
  AttentionArea attention;
  FfnArea ffn;
  // reserve area
  int8_t reserve6[384];  // Padding to 1024 bytes.
};

static_assert(sizeof(ScheduleContext) == 1024, "ScheduleContext size must be 1024 bytes");

#pragma pack(pop)
}  // namespace aicpu
#endif  // INCLUDE_KERNEL_ATTENTION_FFN_SCHEDULE_H