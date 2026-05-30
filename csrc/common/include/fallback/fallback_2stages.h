/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNNFALLBACK_OPAPI_TWOSTAGES_H_
#define ACLNNFALLBACK_OPAPI_TWOSTAGES_H_

#include <dlfcn.h>

#include <functional>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "aclnn/aclnn_base.h"
#include "fallback.h"
#include "fallback_comm.h"
#include "fallback_comm_2stages.h"
#include "log/log.h"
#include "mc2_log.h"

namespace fallback {
using namespace std;
using namespace gert;
using namespace ge;

inline void Collect(aclTensor *p, std::vector<OpApiAnyValue> &params) {
  static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
  OPS_ERR_IF(aclDestroyTensor == nullptr,
            OP_LOGE("aclnnfallback", "aclDestroyTensor is null"), return);
  params.emplace_back(OpApiAnyValue{p, [](void *param) {aclDestroyTensor(static_cast<aclTensor *>(param));}});
}

inline void Collect(aclScalar *p, std::vector<OpApiAnyValue> &params) {
  static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
  OPS_ERR_IF(aclDestroyScalar == nullptr,
            OP_LOGE("aclnnfallback", "aclDestroyScalar is null"), return);
  params.emplace_back(OpApiAnyValue{p, [](void *param) {aclDestroyScalar(static_cast<aclScalar *>(param));}});
}

inline void Collect(aclIntArray *p, std::vector<OpApiAnyValue> &params) {
  static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
  OPS_ERR_IF(aclDestroyIntArray == nullptr,
            OP_LOGE("aclnnfallback", "aclDestroyIntArray is null"), return);
  params.emplace_back(OpApiAnyValue{p, [](void *param) {aclDestroyIntArray(static_cast<aclIntArray *>(param));}});
}

inline void Collect(aclBoolArray *p, std::vector<OpApiAnyValue> &params) {
  static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
  OPS_ERR_IF(aclDestroyBoolArray == nullptr,
            OP_LOGE("aclnnfallback", "aclDestroyBoolArray is null"), return);
  params.emplace_back(OpApiAnyValue{p, [](void *param) {aclDestroyBoolArray(static_cast<aclBoolArray *>(param));}});
}

inline void Collect(aclTensorList *p, std::vector<OpApiAnyValue> &params) {
  static const auto aclDestroyTensorList = GET_OP_API_FUNC(aclDestroyTensorList);
  OPS_ERR_IF(aclDestroyTensorList == nullptr,
            OP_LOGE("aclnnfallback", "aclDestroyTensorList is null"), return);
  params.emplace_back(OpApiAnyValue{p, [](void *param) {aclDestroyTensorList(static_cast<aclTensorList *>(param));}});
}

template <typename T>
void Collect(T value, std::vector<OpApiAnyValue> &params) {
  (void)value;
  params.emplace_back(OpApiAnyValue{nullptr, nullptr});
}

template <typename Tuple, size_t... I>
void CallCollect(Tuple t, std_utils::index_sequence<I...>, std::vector<OpApiAnyValue> &params) {
  (void)std::initializer_list<int>{(Collect(std::get<I>(t), params), 0)...};
}

template <typename Tuple>
void CollectConvertedTypes(Tuple &t, std::vector<OpApiAnyValue> &params) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  CallCollect(t, std_utils::make_index_sequence<size>{}, params);
}

#define EXEC_OPAPI_PREPARE_CMD(aclnn_api, ...)                                                                       \
  ({                                                                                                                 \
    static auto ret = GRAPH_SUCCESS;                                                                                 \
    do {                                                                                                             \
      static const auto ResetCacheThreadLocalAddr = GetOpApiFuncAddr("ResetCacheThreadLocal");                       \
      static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
      static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
      if (getWorkspaceSizeFuncAddr == nullptr || opApiFuncAddr == nullptr || ResetCacheThreadLocalAddr == nullptr) { \
        OP_LOGE("aclnnfallback", "%s or %s not in  %s or %s  or ResetCacheThreadLocal not found.",                   \
                #aclnn_api "GetWorkspaceSize", #aclnn_api, GetOpApiLibName(), GetOpApiLibName());                    \
        ret = GRAPH_FAILED;                                                                                          \
        break;                                                                                                       \
      }                                                                                                              \
      auto *op_api_params = new (std::nothrow) OpApiParams();                                                        \
      auto ResetCacheThreadLocalFunc = reinterpret_cast<ResetCacheThreadLocal>(ResetCacheThreadLocalAddr);           \
      ResetCacheThreadLocalFunc();                                                                                   \
      op_api_params->op_api_func = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                       \
      uint64_t workspace_size = 0;                                                                                   \
      uint64_t* workspace_size_addr = &workspace_size;                                                               \
      aclOpExecutor** executor_addr = &op_api_params->executor;                                                      \
      auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                         \
      using TupleT = decltype(converted_params);                                                                     \
      constexpr size_t tuple_size = std::tuple_size<TupleT>::value;                                                  \
      op_api_params->converted_params.reserve(tuple_size);                                                           \
      CollectConvertedTypes(converted_params, op_api_params->converted_params);                                      \
      host_api_ctx->SetOpApiParamsWithDefaultDeleter<OpApiParams>(op_api_params);                                    \
      static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
      auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
      if (workspace_status != 0) {                                                                                   \
        OP_LOGE("aclnnfallback", "call %s failed:", #aclnn_api);                                                     \
        ret = GRAPH_FAILED;                                                                                          \
        break;                                                                                                       \
      }                                                                                                              \
      ret = host_api_ctx->SetWorkspaceSizes({workspace_size});                                                       \
    } while (false);                                                                                                 \
    (ret);                                                                                                           \
  })

}  // namespace fallback

#endif //  ACLNNFALLBACK_OPAPI_TWOSTAGES_H_