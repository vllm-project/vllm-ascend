/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fallback.h
 * \brief
 */

#ifndef ACLNNFALLBACK_OPAPI_H_
#define ACLNNFALLBACK_OPAPI_H_

#include <dlfcn.h>

#include <functional>
#include <tuple>
#include <type_traits>
#include <vector>

#include "aclnn/aclnn_base.h"
#include "fallback_comm.h"
#include "error/ops_error.h"
#include "runtime/base.h"

namespace fallback {
using namespace std;
using namespace gert;
using namespace ge;
using namespace std;

namespace std_utils {
  template <std::size_t... Is>
  struct index_sequence {};

  template <std::size_t N, std::size_t... Is>
  struct make_index_sequence_helper : make_index_sequence_helper<N - 1, N - 1, Is...> {};

  template <std::size_t... Is>
  struct make_index_sequence_helper<0, Is...> {
    using type = index_sequence<Is...>;
  };

  template <std::size_t N>
  using make_index_sequence = typename make_index_sequence_helper<N>::type;
}

using aclOpExecutor = struct aclOpExecutor;
using aclTensor = struct aclTensor;
using aclScalar = struct aclScalar;
using aclIntArray = struct aclIntArray;
using aclFloatArray = struct aclFloatArray;
using aclBoolArray = struct aclBoolArray;
using aclTensorList = struct aclTensorList;

using _aclCreateTensor = aclTensor* (*)(const int64_t* view_dims, uint64_t view_dims_num, aclDataType data_type,
                                        const int64_t* stride, int64_t offset, aclFormat format,
                                        const int64_t* storage_dims, uint64_t storage_dims_num, void* tensor_data);

using _aclCreateScalar = aclScalar* (*)(void* value, aclDataType data_type);
using _aclCreateIntArray = aclIntArray* (*)(const int64_t* value, uint64_t size);
using _aclCreateFloatArray = aclFloatArray* (*)(const float* value, uint64_t size);
using _aclCreateBoolArray = aclBoolArray* (*)(const bool* value, uint64_t size);
using _aclCreateTensorList = aclTensorList* (*)(const aclTensor* const* value, uint64_t size);

using _aclDestroyTensor = int (*)(const aclTensor* tensor);
using _aclDestroyScalar = int (*)(const aclScalar* scalar);
using _aclDestroyIntArray = int (*)(const aclIntArray* array);
using _aclDestroyFloatArray = int (*)(const aclFloatArray* array);
using _aclDestroyBoolArray = int (*)(const aclBoolArray* array);
using _aclDestroyTensorList = int (*)(const aclTensorList* array);

#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

inline const char* GetOpApiLibName(void) {
  return "libopapi.so";
}

inline const char* GetCustOpApiLibName(void) {
  return "libcust_opapi.so";
}

inline void* GetOpApiFuncAddrInLib(void* handler, const char* libName, const char* apiName) {
  auto funcAddr = dlsym(handler, apiName);
  if (funcAddr == nullptr) {
    OPS_LOG_W("aclnnfallback", "dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
  }
  return funcAddr;
}

inline void* GetOpApiLibHandler(const char* libName) {
  auto handler = dlopen(libName, RTLD_LAZY);
  if (handler == nullptr) {
    OPS_LOG_W("aclnnfallback", "dlopen %s failed, error:%s.", libName, dlerror());
  }
  return handler;
}

inline void* GetAclnnArrdByApiName(const char *apiName) {
    vector<std:: string> libs = {"libaclnn_ops_infer.so", "libaclnn_ops_train.so", "libaclnn_math.so",
                                 "libaclnn_rand.so", "libaclnn_sparse.so", "libaclnn_fft.so"};
    for (const auto &libName : libs) {
        static auto libHandler = GetOpApiLibHandler(libName.c_str());
        if (libHandler != nullptr) {
            auto funcAddr = GetOpApiFuncAddrInLib(libHandler, libName.c_str(), apiName);
            if (funcAddr != nullptr) {
                return funcAddr;
            }
        }
    }
    OPS_LOG_E("aclnnfallback", "api %s can't find in any aclnn lib.", apiName);
    return nullptr;
}

inline void* GetOpApiFuncAddr(const char* apiName) {
  static auto custOpApiHandler = GetOpApiLibHandler(GetCustOpApiLibName());
  if (custOpApiHandler != nullptr) {
    auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
    if (funcAddr != nullptr) {
      return funcAddr;
    }
  }

  static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
  if (opApiHandler != nullptr) {
      auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
      if (funcAddr != nullptr) {
          return funcAddr;
      }
  }
  OPS_LOG_D("aclnnfallback", "opapi lib is not exist,will use aclnn lib.");
  return GetAclnnArrdByApiName(apiName);
}

inline aclTensor* ConvertType(aclTensor* ge_tensor) {
  return ge_tensor;
}

inline aclIntArray* ConvertType(const std::vector<int64_t> &arr) {
  if (arr.empty()) {
    return nullptr;
  }
  static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
  auto array = aclCreateIntArray(arr.data(), arr.size());
  return array;
}

inline aclDataType GetConvertType(const gert::Tensor* ge_tensor) {
  // convert data type
  auto dataType_ge = ge_tensor->GetDataType();
  auto dataType = aclDataType::ACL_FLOAT16;
  if (dataType_ge == DT_FLOAT) {
    dataType = aclDataType::ACL_FLOAT;
  } else if (dataType_ge == DT_BF16) {
    dataType = aclDataType::ACL_BF16;
  } else if (dataType_ge == DT_BOOL) {
    dataType = aclDataType::ACL_BOOL;
  } else if (dataType_ge == DT_INT64) {
    dataType = aclDataType::ACL_INT64;
  } else if (dataType_ge == DT_INT32) {
    dataType = aclDataType::ACL_INT32;
  } else if (dataType_ge == DT_UINT64) {
    dataType = aclDataType::ACL_UINT64;
  } else if (dataType_ge == DT_UINT32) {
    dataType = aclDataType::ACL_UINT32;
  } else if (dataType_ge == DT_INT8) {
    dataType = aclDataType::ACL_INT8;
  } else if (dataType_ge == DT_UINT8) {
    dataType = aclDataType::ACL_UINT8;
  } else if (dataType_ge == DT_INT4) {
    dataType = aclDataType::ACL_INT4;
  } else {
    dataType = aclDataType::ACL_FLOAT16;
  }

  return dataType;
}

inline aclTensor* ConvertType(const gert::Tensor* ge_tensor) {
  if (ge_tensor == nullptr) {
    return nullptr;
  }

  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  OPS_ERR_IF(aclCreateTensor == nullptr, OPS_LOG_E("aclnnfallback", "aclCreateTensor nullptr"), return nullptr);

  void* device_addr = nullptr;
  auto tensor_place = ge_tensor->GetPlacement();
  device_addr = const_cast<void*>(ge_tensor->GetAddr());

  auto dataType = GetConvertType(ge_tensor);

  OPS_LOG_D("aclnnfallback", "aclCreateTensor: tensor type is %d", dataType);

  // convert shape
  auto gert_shape = ge_tensor->GetStorageShape();
  std::vector<int64_t> shape;
  for (size_t i = 0; i < gert_shape.GetDimNum(); ++i) {
    shape.push_back(gert_shape.GetDim(i));
  }

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  aclTensor* out = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                                   0, aclFormat::ACL_FORMAT_ND,
                                   shape.data(), shape.size(), device_addr);

  OPS_ERR_IF(out == nullptr,
    OPS_LOG_E("aclnnfallback", "out nullptr"), return nullptr);

  return out;
}

inline aclTensorList* ConvertType(std::vector<const gert::Tensor*>& ge_tenserList) {
  OPS_ERR_IF(ge_tenserList.size() == 0,
    OPS_LOG_E("aclnnfallback", "ge_tenserList size 0"), return nullptr);

  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  OPS_ERR_IF(aclCreateTensorList == nullptr,
    OPS_LOG_E("aclnnfallback", "ge_tenserList size 0"), return nullptr);

  std::vector<aclTensor*> tmp;
  for (size_t i = 0; i < ge_tenserList.size(); i++) {
    auto t_acl = ConvertType(ge_tenserList[i]);
    tmp.push_back(t_acl);
  }

  aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
  return tensorList;
}

template <typename T>
inline aclScalar* ConvertScalarType(T value) {
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  OPS_ERR_IF(aclCreateScalar == nullptr,
    OPS_LOG_E("aclnnfallback", "aclCreateScalar nullptr"), return nullptr);
  if (typeid(value) == typeid(float)) {
    return aclCreateScalar(&value, aclDataType::ACL_FLOAT);
  }
  return nullptr;
}

template <typename T>
T ConvertType(T value) {
  return value;
}

inline aclTensor* ConvertMmType(const gert::Tensor* ge_tensor, bool transpose, bool enable_NZ=false) {
  if (ge_tensor == nullptr) {
    return nullptr;
  }
  auto gert_shape = ge_tensor->GetStorageShape();
  if (gert_shape.GetDimNum() <= 1) {
    return ConvertType(ge_tensor);
  }

  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  OPS_ERR_IF(aclCreateTensor == nullptr, OPS_LOG_E("aclnnfallback", "aclCreateTensor nullptr"), return nullptr);

  void* device_addr = const_cast<void*>(ge_tensor->GetAddr());
  // convert data type
  auto dataType_ge = ge_tensor->GetDataType();
  auto dataType = ToAclDataType(dataType_ge);
  // convert shape
  std::vector<int64_t> shape;
  for (size_t i = 0; i < gert_shape.GetDimNum(); ++i) {
    shape.push_back(gert_shape.GetDim(i));
  }
  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  auto viewShape = shape;
  // 对于transpose后的tensor对后两维度进行strides, viewShape转换
  if (transpose) {
    // dimM 为倒数第二维， dimN 为倒数第一维度
    auto dimM = shape.size() - 2;
    auto dimN = shape.size() - 1;
    auto swap =  strides[dimN];
    strides[dimN] = strides[dimM];
    strides[dimM] = swap;
    // 修改viewShape
    viewShape[dimN] = shape[dimM];
    viewShape[dimM] = shape[dimN];
  }
  auto acl_format = aclFormat::ACL_FORMAT_ND;
  if (enable_NZ && GetPrimaryFormat(ge_tensor->GetStorageFormat()) == ge::Format::FORMAT_FRACTAL_NZ) {
    acl_format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
  }
  aclTensor* out = aclCreateTensor(viewShape.data(), shape.size(), dataType, strides.data(),
                                   0, acl_format, shape.data(), shape.size(), device_addr);
  OPS_ERR_IF(out == nullptr, OPS_LOG_E("aclnnfallback", "out nullptr"), return nullptr);

  return out;
}

inline void Release(aclTensor* p) {
  static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
  OPS_ERR_IF(aclDestroyTensor == nullptr,
    OPS_LOG_E("aclnnfallback", "aclDestroyTensor is null"), return);
  aclDestroyTensor(p);
}

inline void Release(aclScalar* p) {
  static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
  OPS_ERR_IF(aclDestroyScalar == nullptr,
    OPS_LOG_E("aclnnfallback", "aclDestroyScalar is null"), return);
  aclDestroyScalar(p);
}

inline void Release(aclIntArray* p) {
  static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
  OPS_ERR_IF(aclDestroyIntArray == nullptr,
    OPS_LOG_E("aclnnfallback", "aclDestroyIntArray is null"), return);
  aclDestroyIntArray(p);
}

inline void Release(aclBoolArray* p) {
  static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
  OPS_ERR_IF(aclDestroyBoolArray == nullptr,
    OPS_LOG_E("aclnnfallback", "aclDestroyBoolArray is null"), return);
  aclDestroyBoolArray(p);
}

inline void Release(aclTensorList* p) {
  static const auto aclDestroyTensorList = GET_OP_API_FUNC(aclDestroyTensorList);
  OPS_ERR_IF(aclDestroyTensorList == nullptr,
    OPS_LOG_E("aclnnfallback", "aclDestroyTensorList is null"), return);
  aclDestroyTensorList(p);
}

template <typename T>
void Release(T value) {
  (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std_utils::index_sequence<I...>) {
  (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple& t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  CallRelease(t, std_utils::make_index_sequence<size>{});
}

template <typename... Ts>
auto ConvertTypes(Ts&... args) -> decltype(std::make_tuple(ConvertType(args)...)) {
    auto tp = std::make_tuple(ConvertType(args)...);
    return tp;
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std_utils::index_sequence<I...>)  -> int {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t) -> int {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std_utils::make_index_sequence<size>{});
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr, std_utils::index_sequence<I...>)
    -> int (*)(typename std::decay<decltype(std::get<I>(params))>::type...) {
    using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr)
    -> typename std::enable_if<std::tuple_size<Tuple>::value != 0,
    decltype(ConvertToOpApiFunc(params, opApiAddr, std_utils::make_index_sequence<std::tuple_size<Tuple>::value>{}))>::type {
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std_utils::make_index_sequence<size>{});
}

template <typename Tuple>
class ConvertedParams {
 public:
  ConvertedParams(Tuple&& convertedParams) : convertedParams_(std::move(convertedParams)){};
  ConvertedParams(ConvertedParams&& other) : convertedParams_(std::move(other.convertedParams_)) {
    other.validParams_ = false;
  };
  ConvertedParams& operator=(ConvertedParams&& other) {
    if (this == &other) {
      return *this;
    }

    convertedParams_ = std::move(other.convertedParams_);
    validParams_ = true;
    other.validParams_ = false;
    return *this;
  }

  ConvertedParams() = delete;
  ConvertedParams(const ConvertedParams& other) = delete;
  ConvertedParams& operator=(const ConvertedParams& other) = delete;

  ~ConvertedParams() {
    if (validParams_) {
      ReleaseConvertTypes(convertedParams_);
    }
  }

  const Tuple& GetConvertedParams() const {
    return convertedParams_;
  }

 private:
  Tuple convertedParams_;
  bool validParams_{true};
};

using InitHugeMemThreadLocal = int (*)(void*, bool);
using UnInitHugeMemThreadLocal = void (*)(void*, bool);
using ReleaseHugeMem = void (*)(void*, bool);
using PTAGetExecCache = aclOpExecutor* (*)(uint64_t, uint64_t*);
using InitPTACacheThreadLocal = void (*)();
using SetPTAHashKey = void (*)(uint64_t);
using CanUsePTACache = bool (*)(const char*);

using ResetCacheThreadLocal = void (*)();

#define EXEC_OPAPI_CMD(aclnn_api, ...)                                                                               \
  ({                                                                                                                 \
    static auto ret = GRAPH_SUCCESS;                                                                                 \
    do {                                                                                                             \
      static const auto ResetCacheThreadLocalAddr = GetOpApiFuncAddr("ResetCacheThreadLocal");                       \
      static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
      static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
      if (getWorkspaceSizeFuncAddr == nullptr || opApiFuncAddr == nullptr || ResetCacheThreadLocalAddr == nullptr) { \
        OPS_LOG_E("aclnnfallback", "%s or %s not in  %s or %s  or ResetCacheThreadLocal not found.",                   \
                #aclnn_api "GetWorkspaceSize", #aclnn_api, GetOpApiLibName(), GetOpApiLibName());                    \
        ret = GRAPH_FAILED;                                                                                          \
        break;                                                                                                       \
      }                                                                                                              \
      auto ResetCacheThreadLocalFunc = reinterpret_cast<ResetCacheThreadLocal>(ResetCacheThreadLocalAddr);           \
      ResetCacheThreadLocalFunc();                                                                                   \
      uint64_t workspace_size = 0;                                                                                   \
      uint64_t* workspace_size_addr = &workspace_size;                                                               \
      aclOpExecutor* executor = nullptr;                                                                             \
      aclOpExecutor** executor_addr = &executor;                                                                     \
      auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                         \
      static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
      auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
      if (workspace_status != 0) {                                                                                   \
        OPS_LOG_E("aclnnfallback", "call %s failed:", #aclnn_api);                                                     \
        ret = GRAPH_FAILED;                                                                                          \
        break;                                                                                                       \
      }                                                                                                              \
      void* workspace_addr = nullptr;                                                                                \
      if (workspace_size > 0) {                                                                                      \
        workspace_addr = host_api_ctx->MallocWorkspace(workspace_size);                                              \
        if (workspace_addr == nullptr) {                                                                             \
          OPS_LOG_E("aclnnfallback", "call %s allocate workspace failed", #aclnn_api);                                 \
          ret = GRAPH_FAILED;                                                                                        \
          break;                                                                                                     \
        }                                                                                                            \
      }                                                                                                              \
      auto acl_stream = host_api_ctx->GetStream();                                                                   \
      auto acl_call = [converted_params, workspace_addr, workspace_size, host_api_ctx, acl_stream,                   \
                       executor]() -> int {                                                                          \
        using OpApiFunc = int (*)(void*, uint64_t, aclOpExecutor*, const aclrtStream);                               \
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                            \
        auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                              \
        ReleaseConvertTypes(converted_params);                                                                       \
        host_api_ctx->FreeWorkspace();                                                                               \
        if (api_ret != 0) {                                                                                          \
          OPS_LOG_E("aclnnfallback", "call %s allocate workspace failed api_ret: %d", #aclnn_api, api_ret);            \
          return GRAPH_FAILED;                                                                                       \
        }                                                                                                            \
        return api_ret;                                                                                              \
      };                                                                                                             \
                                                                                                                     \
      ret = acl_call();                                                                                              \
    } while (false);                                                                                                 \
    (ret);                                                                                                           \
  })

}  // namespace fallback

#endif  //  ACLNNFALLBACK_OPAPI_H_
