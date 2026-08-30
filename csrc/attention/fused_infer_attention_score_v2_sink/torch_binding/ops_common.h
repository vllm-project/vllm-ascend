/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_
#define TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_

#include <fstream>
#include <cstring>
#include <sys/stat.h>
#include <dlfcn.h>
#include <vector>
#include <functional>
#include <type_traits>
#include <ATen/Tensor.h>
#include <ATen/NamedTensorUtils.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/flopcount/FlopCount.h"
#include "torch_npu/csrc/flopcount/FlopCounter.h"
#include "torch_npu/csrc/custom_dtype/Init.h"


typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalarList aclScalarList;

typedef aclTensor *(*_aclCreateTensor)(const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
                                       const int64_t *stride, int64_t offset, aclFormat format,
                                       const int64_t *storage_dims, uint64_t storage_dims_num, void *tensor_data);
typedef aclScalar *(*_aclCreateScalar)(void *value, aclDataType data_type);
typedef aclIntArray *(*_aclCreateIntArray)(const int64_t *value, uint64_t size);
typedef aclFloatArray *(*_aclCreateFloatArray)(const float *value, uint64_t size);
typedef aclBoolArray *(*_aclCreateBoolArray)(const bool *value, uint64_t size);
typedef aclTensorList *(*_aclCreateTensorList)(const aclTensor *const *value, uint64_t size);
typedef aclScalarList *(*_aclCreateScalarList)(const aclScalar *const *value, uint64_t size);

typedef int (*_aclDestroyTensor)(const aclTensor *tensor);
typedef int (*_aclDestroyScalar)(const aclScalar *scalar);
typedef int (*_aclDestroyIntArray)(const aclIntArray *array);
typedef int (*_aclDestroyFloatArray)(const aclFloatArray *array);
typedef int (*_aclDestroyBoolArray)(const aclBoolArray *array);
typedef int (*_aclDestroyTensorList)(const aclTensorList *array);
typedef int (*_aclDestroyScalarList)(const aclScalarList *array);

using OpApiFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

constexpr int BLOCKSIZE = 16;
constexpr int BLOCKBYTES = 32;
constexpr int MAX_FORMAT_SHAPE_SIZE = 8;
using FormatShape = c10::SmallVector<int64_t, MAX_FORMAT_SHAPE_SIZE>;
using shapeInfer = std::function<FormatShape(c10::IntArrayRef dims, size_t itemsize)>;

inline FormatShape InferShapeLessTo4(c10::IntArrayRef arrayDims, size_t itemsize);
inline FormatShape InferShape4To5(c10::IntArrayRef dims, size_t itemsize);
inline FormatShape InferShape5To4(c10::IntArrayRef dims, size_t itemsize);
inline FormatShape InferShapeNDToNZ(c10::IntArrayRef dims, size_t itemsize);
inline FormatShape InferShapeNDToZ(c10::IntArrayRef dims, size_t itemsize);
inline FormatShape InferShapeofNCHW(c10::IntArrayRef dims, size_t itemsize);
inline FormatShape InferShapeofND(c10::IntArrayRef dims, size_t itemsize);

// converter between base format
inline FormatShape InferShapeNCHWToND(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims, size_t itemsize);
inline FormatShape InferShapeNCDHWToND(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims, size_t itemsize);
inline FormatShape InferShapeNDToNCHW(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims, size_t itemsize);
inline FormatShape InferShapeNDToNCDHW(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims, size_t itemsize);

// base format is NCDHW
inline FormatShape InferShapeOfNDHWC(c10::IntArrayRef arrayDims, size_t itemsize);
inline FormatShape InferShapeOfNCDHW(c10::IntArrayRef arrayDims, size_t itemsize);
inline FormatShape InferShapeOfNDC1HWC0(c10::IntArrayRef arrayDims, size_t itemsize);
inline FormatShape InferShapeOfFZ3D(c10::IntArrayRef arrayDims, size_t itemsize);

inline FormatShape InferShapeofNHWC(c10::IntArrayRef dims, size_t itemsize);

inline FormatShape InferShapeOfNDHWC(c10::IntArrayRef arrayDims, size_t itemsize)
{
    if (arrayDims.size() < 5) {
        AT_ERROR("dim (", arrayDims, ") cannot convert to NDHWC");
    }
    FormatShape formatShape;
    formatShape.resize(5);
    formatShape[0] = arrayDims[0];
    formatShape[1] = arrayDims[2];
    formatShape[2] = arrayDims[3];
    formatShape[3] = arrayDims[4];
    formatShape[4] = arrayDims[1];
    return formatShape;
}

inline FormatShape InferShapeOfNCDHW(c10::IntArrayRef arrayDims, size_t itemsize)
{
    if (arrayDims.size() < 5) {
        AT_ERROR("dim (", arrayDims, ") cannot convert to NCDHW");
    }
    FormatShape formatShape;
    formatShape.resize(5);
    formatShape[0] = arrayDims[0];
    formatShape[1] = arrayDims[1];
    formatShape[2] = arrayDims[2];
    formatShape[3] = arrayDims[3];
    formatShape[4] = arrayDims[4];
    return formatShape;
}

inline FormatShape InferShapeOfNDC1HWC0(c10::IntArrayRef arrayDims, size_t itemsize)
{
    if (arrayDims.size() < 5) {
        AT_ERROR("dim (", arrayDims, ") cannot convert to NDC1HWC0");
    }
    FormatShape formatShape;
    formatShape.resize(6);
    formatShape[0] = arrayDims[0];
    formatShape[1] = arrayDims[2];
    formatShape[2] = (arrayDims[1] + BLOCKSIZE - 1) / BLOCKSIZE;
    formatShape[3] = arrayDims[3];
    formatShape[4] = arrayDims[4];
    formatShape[5] = BLOCKSIZE;
    return formatShape;
}

inline FormatShape InferShapeOfFZ3D(c10::IntArrayRef arrayDims, size_t itemsize)
{
    if (arrayDims.size() < 5) {
        AT_ERROR("dim (", arrayDims, ") cannot convert to FZ_3D");
    }

    int64_t dim1 = arrayDims[2];
    int64_t dim2 = (arrayDims[1] + BLOCKSIZE - 1) / BLOCKSIZE;
    int64_t dim3 = arrayDims[3];
    int64_t dim4 = arrayDims[4];
    int64_t dim5 = (arrayDims[0] + BLOCKSIZE - 1) / BLOCKSIZE;
    int64_t dim6 = BLOCKSIZE;
    int64_t dim7 = BLOCKSIZE;

    FormatShape formatShape;
    formatShape.resize(4);
    formatShape[0] = dim1 * dim2 * dim3 * dim4;
    formatShape[1] = dim5;
    formatShape[2] = dim6;
    formatShape[3] = dim7;
    return formatShape;
}

inline FormatShape InferShapeofNCHW(c10::IntArrayRef dims, size_t itemsize)
{
    if (dims.size() < 5) {
        return InferShapeLessTo4(dims, itemsize);
    } else {
        return InferShapeofND(dims, itemsize);
    }
}

inline FormatShape InferShapeofND(c10::IntArrayRef dims, size_t itemsize)
{
    FormatShape res;
    res.resize(dims.size());
    for (size_t j = 0; j < dims.size(); j++) {
        res[j] = dims[j];
    }
    return res;
}

inline FormatShape InferShape4To5(c10::IntArrayRef dims, size_t itemsize)
{
    FormatShape res;
    res.resize(5);
    if (dims.size() < 4) {
        ASCEND_LOGD("infershape4to5 but input dim < 4");
        return InferShape4To5(InferShapeLessTo4(dims, itemsize), itemsize);
    } else if (dims.size() > 4) {
        ASCEND_LOGE("infershape4to5 but input dim > 4");
    }
    res[0] = dims[0];
    res[1] = (dims[1] + 15) / 16;
    res[2] = dims[2];
    res[3] = dims[3];
    res[4] = BLOCKSIZE;
    return res;
}

inline FormatShape InferShapeofNHWC(c10::IntArrayRef dims, size_t itemsize)
{
    AT_ASSERT(dims.size() == 4, "input dim should be equal to 4 when InferShapeofNHWC");
    return FormatShape(dims.begin(), dims.end());
}

inline FormatShape InferShapeNDToNZ(c10::IntArrayRef dims, size_t itemsize)
{
    FormatShape res;
    // sum(keepdim = false) may make tensor dim = 0
    FormatShape dim;
    for (size_t i = 0; i < dims.size(); i++) {
        dim.emplace_back(dims[i]);
    }

    // this action will move to GuessStorageSizeWhenConvertFormat
    if (dim.size() == 0) {
        dim.emplace_back(1);
    }
    if (dim.size() == 1) {
        dim.emplace_back(1);
    }

    size_t i = 0;
    for (; i < dim.size() - 2; i++) {
        res.emplace_back(dim[i]);
    }

    AT_ASSERT(itemsize != 0, "dtype itemsize should not be 0");

    // float32 will cast to float16
    auto itemsize_ = (itemsize > 2) ? 2 : itemsize;
    auto lastSize = BLOCKBYTES / itemsize_;
    res.emplace_back((dim[i + 1] + lastSize - 1) / lastSize);
    res.emplace_back((dim[i] + BLOCKSIZE - 1) / BLOCKSIZE);
    res.emplace_back(BLOCKSIZE);
    res.emplace_back(lastSize);

    return res;
}

inline FormatShape InferShapeNDToZ(c10::IntArrayRef dims, size_t itemsize)
{
    FormatShape res;
    if (dims.size() < 4) {
        return InferShapeNDToZ(InferShapeLessTo4(dims, itemsize), itemsize);
    }

    res.emplace_back((dims[1] + 15) / BLOCKSIZE * dims[2] * dims[3]);
    res.emplace_back((dims[0] + 15) / BLOCKSIZE);
    res.emplace_back(BLOCKSIZE);
    res.emplace_back(BLOCKSIZE);

    return res;
}

inline FormatShape InferShapeLessTo4(c10::IntArrayRef arrayDims, size_t itemsize)
{
    FormatShape formatShape;
    formatShape.resize(4);
    AT_ASSERT(arrayDims.size() <= 4, "input dim > 4 when InferShapeLessTo4");
    switch (arrayDims.size()) {
        case 0:
            formatShape[0] = 1;
            formatShape[1] = 1;
            formatShape[2] = 1;
            formatShape[3] = 1;
            break;
        case 1:  // RESHAPE_TYPE_C
            formatShape[0] = 1;
            formatShape[1] = arrayDims[0];
            formatShape[2] = 1;
            formatShape[3] = 1;
            break;
        case 2:  // RESHAPE_TYPE_CH
            formatShape[0] = 1;
            formatShape[1] = arrayDims[0];
            formatShape[2] = arrayDims[1];
            formatShape[3] = 1;
            break;
        case 3:  // RESHAPE_TYPE_CHW
            formatShape[0] = 1;
            formatShape[1] = arrayDims[0];
            formatShape[2] = arrayDims[1];
            formatShape[3] = arrayDims[2];
            break;
        case 4:
            formatShape[0] = arrayDims[0];
            formatShape[1] = arrayDims[1];
            formatShape[2] = arrayDims[2];
            formatShape[3] = arrayDims[3];
            break;
        default:
            AT_ERROR("dims of NCHW shape should not be greater than 4, which is ", arrayDims.size());
    }
    return formatShape;
}

typedef struct AclFormatInfoShape_ {
    aclFormat formatInfo = ACL_FORMAT_ND;
    aclFormat baseFormat = ACL_FORMAT_ND;
    shapeInfer shapeFunc = nullptr;
    char aclFormatName[30] = {0};
    bool isPad = false;
} AclFormatInfoShape;

static std::unordered_map<aclFormat, AclFormatInfoShape> InitializeInfo()
{
    return {
        {ACL_FORMAT_NC1HWC0,
         (AclFormatInfoShape){ACL_FORMAT_NC1HWC0, ACL_FORMAT_NCHW, InferShape4To5, "NC1HWC0", true}},
        {ACL_FORMAT_ND, (AclFormatInfoShape){ACL_FORMAT_ND, ACL_FORMAT_ND, InferShapeofND, "ND", false}},
        {ACL_FORMAT_NCHW, (AclFormatInfoShape){ACL_FORMAT_NCHW, ACL_FORMAT_NCHW, InferShapeofNCHW, "NCHW", false}},
        {ACL_FORMAT_NHWC, (AclFormatInfoShape){ACL_FORMAT_NHWC, ACL_FORMAT_NHWC, InferShapeofNHWC, "NHWC", false}},
        {ACL_FORMAT_FRACTAL_NZ,
         (AclFormatInfoShape){ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_ND, InferShapeNDToNZ, "FRACTAL_NZ", true}},
        {ACL_FORMAT_FRACTAL_Z,
         (AclFormatInfoShape){ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_NCHW, InferShapeNDToZ, "FRACTAL_Z", true}},
        {ACL_FORMAT_NDHWC, (AclFormatInfoShape){ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW, InferShapeOfNDHWC, "NDHWC", false}},
        {ACL_FORMAT_NCDHW, (AclFormatInfoShape){ACL_FORMAT_NCDHW, ACL_FORMAT_NCDHW, InferShapeOfNCDHW, "NCDHW", false}},
        {ACL_FORMAT_NDC1HWC0,
         (AclFormatInfoShape){ACL_FORMAT_NDC1HWC0, ACL_FORMAT_NCDHW, InferShapeOfNDC1HWC0, "NDC1HWC0", true}},
        {ACL_FRACTAL_Z_3D,
         (AclFormatInfoShape){ACL_FRACTAL_Z_3D, ACL_FORMAT_NCDHW, InferShapeOfFZ3D, "FRACTAL_Z_3D", true}},
        {ACL_FORMAT_FRACTAL_NZ_C0_16,
            (AclFormatInfoShape){ACL_FORMAT_FRACTAL_NZ_C0_16, ACL_FORMAT_ND, nullptr, "FRACTAL_NZ_C0_16", true}},
        {ACL_FORMAT_FRACTAL_NZ_C0_32,
            (AclFormatInfoShape){ACL_FORMAT_FRACTAL_NZ_C0_32, ACL_FORMAT_ND, nullptr, "FRACTAL_NZ_C0_32", true}},
    };
};


struct NPUStorageDesc {
public:
  struct use_byte_size_t {};

  c10::SmallVector<int64_t, 5> base_sizes_;
  c10::SmallVector<int64_t, 5> base_strides_;
  c10::SmallVector<int64_t, 5> storage_sizes_;
  int64_t base_offset_ = 0;
  use_byte_size_t base_dtype_ = {};
  aclFormat origin_format_ = ACL_FORMAT_UNDEFINED;
  aclFormat npu_format_ = ACL_FORMAT_ND;
  caffe2::TypeMeta data_type_ = caffe2::TypeMeta::Make<uint8_t>();
};

struct NPUStorageImpl : public c10::StorageImpl {
  explicit NPUStorageImpl(
    use_byte_size_t use_byte_size,
    size_t size_bytes,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizable);
  ~NPUStorageImpl() override = default;
  void release_resources() override;

  // not private
  NPUStorageDesc npu_desc_;

  NPUStorageDesc get_npu_desc() const
  {
    return npu_desc_;
  }

  uint64_t unique_id_{0};

  uint64_t get_unique_id()
  {
    return unique_id_;
  }

  std::mutex unique_id_mutex_;
};

const int N = 32;
// npu tensor max size
const int SIZE = 8;
const int INT4_NUMS_IN_INT32_SPACE = 8;
const int NPU_NSA_COMPRESS_INPUT_DIM_SECOND = 1;
const int NPU_NSA_COMPRESS_INPUT_DIM_THIRD = 2;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

namespace {
  constexpr int64_t MAX_DIM_NUM = 5;
  constexpr int64_t NCL_DIM_NUM = 3;
  constexpr int64_t NCHW_DIM_NUM = 4;
  constexpr int64_t NCDHW_DIM_NUM = 5;
}

constexpr int g_hash_buf_size = 8192;
constexpr int g_hash_buf_max_size = g_hash_buf_size + 1024;
extern thread_local char g_hash_buf[g_hash_buf_size];
extern thread_local int g_hash_offset;

constexpr int kHashBufSize = 8192;
constexpr int kHashBufMaxSize = kHashBufSize + 1024;
extern thread_local char g_hashBuf[kHashBufSize];
extern thread_local int g_hashOffset;

// dtype convert map
#ifdef SUPPORT_ACL_FLOAT8
#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)                                                                    \
    _(at::ScalarType::Byte, ACL_UINT8)                                                                                 \
    _(at::ScalarType::Char, ACL_INT8)                                                                                  \
    _(at::ScalarType::Short, ACL_INT16)                                                                                \
    _(at::ScalarType::Int, ACL_INT32)                                                                                  \
    _(at::ScalarType::Long, ACL_INT64)                                                                                 \
    _(at::ScalarType::Half, ACL_FLOAT16)                                                                               \
    _(at::ScalarType::Float, ACL_FLOAT)                                                                                \
    _(at::ScalarType::Double, ACL_DOUBLE)                                                                              \
    _(at::ScalarType::ComplexHalf, ACL_COMPLEX32)                                                                      \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)                                                                     \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)                                                                   \
    _(at::ScalarType::Bool, ACL_BOOL)                                                                                  \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::BFloat16, ACL_BF16)                                                                              \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)                                                                      \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)                                                                      \
    _(at::ScalarType::Bits1x8, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits2x4, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits4x2, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits8, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::Bits16, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::Float8_e5m2, ACL_DT_UNDEFINED)                                                                   \
    _(at::ScalarType::Float8_e4m3fn, ACL_FLOAT8_E4M3FN)                                                                \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)                                                                     \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)
#else
#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_) \
  _(at::ScalarType::Byte, ACL_UINT8)                \
  _(at::ScalarType::Char, ACL_INT8)                 \
  _(at::ScalarType::Short, ACL_INT16)               \
  _(at::ScalarType::Int, ACL_INT32)                 \
  _(at::ScalarType::Long, ACL_INT64)                \
  _(at::ScalarType::Half, ACL_FLOAT16)              \
  _(at::ScalarType::Float, ACL_FLOAT)               \
  _(at::ScalarType::Double, ACL_DOUBLE)             \
  _(at::ScalarType::ComplexHalf, ACL_COMPLEX32)  \
  _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)    \
  _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)  \
  _(at::ScalarType::Bool, ACL_BOOL)                 \
  _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)       \
  _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)       \
  _(at::ScalarType::BFloat16, ACL_BF16)             \
  _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)     \
  _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)     \
  _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)    \
  _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)
#endif

constexpr aclDataType kATenScalarTypeToAclDataTypeTable
    [static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
        AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// load aclnn api so
static std::vector<std::string> split_str(std::string s, const std::string &del)
{
    int end = s.find(del);
    std::vector<std::string> path_list;
    while (end != -1) {
        path_list.push_back(s.substr(0, end));
        s.erase(s.begin(), s.begin() + end + 1);
        end = s.find(del);
    }
    path_list.push_back(s);
    return path_list;
}

static bool is_file_exist(const std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        return false;
    }
    return (access(path.c_str(), F_OK) == 0) ? true : false;
}

inline  std::string real_path(const std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        return "";
    }
    char realPath[PATH_MAX] = {0};
    if (realpath(path.c_str(), realPath) == nullptr) {
        return "";
    }
    return std::string(realPath);
}

inline std::vector<std::string> get_custom_lib_path()
{
    char *ascend_custom_opppath = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    std::vector<std::string> custom_lib_path_list;

    if (ascend_custom_opppath == nullptr) {
        ASCEND_LOGW("ASCEND_CUSTOM_OPP_PATH is not exists");
        return std::vector<std::string>();
    }

    std::string ascend_custom_opppath_str(ascend_custom_opppath);
    // split string with ":"
    custom_lib_path_list = split_str(ascend_custom_opppath_str, ":");
    if (custom_lib_path_list.empty()) {
        return std::vector<std::string>();
    }
    for (auto &it : custom_lib_path_list) {
        it = it + "/op_api/lib/";
    }

    return custom_lib_path_list;
}

inline std::vector<std::string> get_default_custom_lib_path()
{
    char *ascend_opp_path = std::getenv("ASCEND_OPP_PATH");
    std::vector<std::string> default_vendors_list;

    if (ascend_opp_path == nullptr) {
        ASCEND_LOGW("ASCEND_OPP_PATH is not exists");
        return std::vector<std::string>();
    }

    std::string vendors_path(ascend_opp_path);
    vendors_path = vendors_path + "/vendors";
    std::string vendors_config_file = real_path(vendors_path + "/config.ini");
    if (vendors_config_file.empty()) {
        ASCEND_LOGW("config.ini is not exists");
        return std::vector<std::string>();
    }

    if (!is_file_exist(vendors_config_file)) {
        ASCEND_LOGW("config.ini is not exists or the path length is more than %d", PATH_MAX);
        return std::vector<std::string>();
    }

    std::ifstream ifs(vendors_config_file);
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find("load_priority=") == 0) {
            break;
        }
    }
    std::string head = "load_priority=";
    line.erase(0, head.length());

    // split string with ","
    default_vendors_list = split_str(line, ",");
    if (default_vendors_list.empty()) {
        return std::vector<std::string>();
    }
    for (auto &it : default_vendors_list) {
        it = real_path(vendors_path + "/" + it + "/op_api/lib/");
    }

    return default_vendors_list;
}

extern const std::vector<std::string> g_custom_lib_path;
extern const std::vector<std::string> g_default_custom_lib_path;
void *GetOpApiFuncAddrFromFeatureLib(const char *api_name);


inline const char *GetOpApiLibName(void)
{
    return "libopapi.so";
}

inline const char *GetCustOpApiLibName(void)
{
    return "libcust_opapi.so";
}

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName, const char *apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void *GetOpApiLibHandler(const char *libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}


// get aclnn api from loaded so
#define GET_OP_API_FUNC_FROM_FEATURE_LIB(lib_handler, lib_name, api_name)      \
  do {                                                                         \
    static auto lib_handler = GetOpApiLibHandler(lib_name);                    \
    if ((lib_handler) != nullptr) {                                            \
      auto funcAddr = GetOpApiFuncAddrInLib(lib_handler, lib_name, api_name);  \
      if (funcAddr != nullptr) {                                               \
        return funcAddr;                                                       \
      }                                                                        \
    }                                                                          \
  } while (0)

#define GET_OP_API_FUNC(apiName) \
  reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

inline void *GetOpApiFuncAddr(const char *apiName)
{
    if (!g_custom_lib_path.empty()) {
        for (auto &it : g_custom_lib_path) {
            auto cust_opapi_lib = real_path(it + "/" + GetCustOpApiLibName());
            if (cust_opapi_lib.empty()) {
                continue;
            }
            auto custOpApiHandler = GetOpApiLibHandler(cust_opapi_lib.c_str());
            if (custOpApiHandler != nullptr) {
                auto funcAddr =
                    GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
                if (funcAddr != nullptr) {
                    ASCEND_LOGI("%s is found in %s.", apiName, cust_opapi_lib.c_str());
                    return funcAddr;
                }
            }
        }
        ASCEND_LOGI("%s is not in custom lib.", apiName);
    }

    if (!g_default_custom_lib_path.empty()) {
        for (auto &it : g_default_custom_lib_path) {
            auto default_cust_opapi_lib = real_path(it + "/" + GetCustOpApiLibName());
            if (default_cust_opapi_lib.empty()) {
                continue;
            }
            auto custOpApiHandler = GetOpApiLibHandler(default_cust_opapi_lib.c_str());
            if (custOpApiHandler != nullptr) {
                auto funcAddr =
                    GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
                if (funcAddr != nullptr) {
                    ASCEND_LOGI("%s is found in %s.", apiName, default_cust_opapi_lib.c_str());
                    return funcAddr;
                }
            }
        }
        ASCEND_LOGI("%s is not in default custom lib.", apiName);
    }

    GET_OP_API_FUNC_FROM_FEATURE_LIB(opapiMathHandler, "libopapi_math.so", apiName);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(opapiNnHandler, "libopapi_nn.so", apiName);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(opapiCvHandler, "libopapi_cv.so", apiName);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(opapiTransformerHandler, "libopapi_transformer.so", apiName);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(opapiLegacyHandler, "libopapi_legacy.so", apiName);

    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    return GetOpApiFuncAddrFromFeatureLib(apiName);
}


// convert args
inline c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor)
{
  c10::Scalar expScalar;
  const at::Tensor *aclInput = &tensor;
  if (aclInput->scalar_type() == at::ScalarType::Double) {
    double value = *(double *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Long) {
    int64_t value = *(int64_t *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Float) {
    float value = *(float *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Int) {
    int value = *(int *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Half) {
    c10::Half value = *(c10::Half *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Bool) {
    int8_t value = *(int8_t *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::ComplexDouble) {
    c10::complex<double> value = *(c10::complex<double> *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::ComplexFloat) {
    c10::complex<float> value = *(c10::complex<float> *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::BFloat16) {
    c10::BFloat16 value = *(c10::BFloat16 *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  }
  return expScalar;
}

inline at::Tensor CopyTensorHostToDevice(const at::Tensor &cpu_tensor)
{
  at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
  int deviceIndex = 0;
  return cpuPinMemTensor.to(c10::Device(torch_npu::utils::get_npu_device_type(), deviceIndex),
                            cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type)
{
  return CopyTensorHostToDevice(
      scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

static bool IsOpInputBaseFormatCommon(const at::Tensor &at_tensor)
{
  if (!torch_npu::utils::is_npu(at_tensor)) {
    return true;
  }
  const auto format = static_cast<NPUStorageImpl *>(at_tensor.storage().unsafeGetStorageImpl())->npu_desc_.npu_format_;
  return (format == ACL_FORMAT_ND) || (format == ACL_FORMAT_NCHW) || (format == ACL_FORMAT_NHWC) ||
      (format == ACL_FORMAT_NCDHW);
}

static bool IsBaseFormatTypeCommon(aclFormat format)
{
    std::unordered_map<aclFormat, AclFormatInfoShape> info = InitializeInfo();
    const auto &itr = info.find(format);
    if (itr == info.end()) {
        return ACL_FORMAT_ND;
    }
    return itr->second.baseFormat == format;
}

inline aclTensor *ConvertType(const at::Tensor &at_tensor)
{
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }

  if (!at_tensor.defined()) {
    return nullptr;
  }
  at::ScalarType scalar_data_type = at_tensor.scalar_type();
  aclDataType acl_data_type =
      kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
  TORCH_CHECK(
      acl_data_type != ACL_DT_UNDEFINED,
      std::string(c10::toString(scalar_data_type)) + " has not been supported")
  c10::SmallVector<int64_t, SIZE> storageDims;

  const auto dimNum = at_tensor.sizes().size();
  aclFormat format = ACL_FORMAT_ND;
  if (!IsOpInputBaseFormatCommon(at_tensor)) {
    format = static_cast<NPUStorageImpl *>(at_tensor.storage().unsafeGetStorageImpl())->npu_desc_.npu_format_;
    if (acl_data_type != ACL_STRING) {
        TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
        storageDims = static_cast<NPUStorageImpl *>(
                          at_tensor.storage().unsafeGetStorageImpl())
                          ->npu_desc_.storage_sizes_;
    }
  } else {
    switch (dimNum) {
        case 3:
            format = ACL_FORMAT_NCL;
            break;
        case 4:
            format = ACL_FORMAT_NCHW;
            break;
        case 5:
            format = ACL_FORMAT_NCDHW;
            break;
        default:
            format = ACL_FORMAT_ND;
    }
    if (acl_data_type != ACL_STRING) {
        TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
        storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
    }
  }

  if (at_tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    // no need this ConvertTensorToScalar
    c10::Scalar expScalar = at_tensor.item();
    at::Tensor aclInput = CopyScalarToDevice(expScalar, scalar_data_type);
    return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(),
                           acl_data_type, aclInput.strides().data(),
                           aclInput.storage_offset(), format,
                           storageDims.data(), storageDims.size(),
                           const_cast<void *>(aclInput.storage().data()));
  }

  auto acl_tensor = aclCreateTensor(
      at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type,
      at_tensor.strides().data(), at_tensor.storage_offset(), format,
      storageDims.data(), storageDims.size(),
      const_cast<void *>(at_tensor.storage().data()));
  return acl_tensor;
}

inline aclScalar *ConvertType(const at::Scalar &at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type =
        kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
                std::string(c10::toString(scalar_data_type)) + " has not been supported")
    aclScalar *acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double: {
            double value = at_scalar.toDouble();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::Long: {
            int64_t value = at_scalar.toLong();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::Bool: {
            bool value = at_scalar.toBool();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::ComplexDouble: {
            auto value = at_scalar.toComplexDouble();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        default:
            acl_scalar = nullptr;
            break;
        }
    return acl_scalar;
}

inline aclIntArray *ConvertType(const at::IntArrayRef &at_array)
{
  static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
  if (aclCreateIntArray == nullptr) {
    return nullptr;
  }
  auto array = aclCreateIntArray(at_array.data(), at_array.size());
  return array;
}

template <std::size_t N>
inline aclBoolArray *ConvertType(const std::array<bool, N> &value)
{
  static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
  if (aclCreateBoolArray == nullptr) {
    return nullptr;
  }

  auto array = aclCreateBoolArray(value.data(), value.size());
  return array;
}

inline aclBoolArray *ConvertType(const at::ArrayRef<bool> &value)
{
  static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
  if (aclCreateBoolArray == nullptr) {
    return nullptr;
  }

  auto array = aclCreateBoolArray(value.data(), value.size());
  return array;
}

inline aclIntArray *ConvertType(const at::ArrayRef<c10::SymInt> &at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto int_array = c10::asIntArrayRefUnchecked(at_array);
    auto array = aclCreateIntArray(int_array.data(), int_array.size());
    return array;
}

inline aclTensorList *ConvertType(const at::TensorList &at_tensor_list)
{
  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  if (aclCreateTensorList == nullptr) {
    return nullptr;
  }

  std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
  for (size_t i = 0; i < at_tensor_list.size(); i++) {
    tensor_list[i] = ConvertType(at_tensor_list[i]);
  }
  auto acl_tensor_list =
      aclCreateTensorList(tensor_list.data(), tensor_list.size());
  return acl_tensor_list;
}

inline aclScalarList *ConvertType(const at::ArrayRef<at::Scalar> &at_scalar_list)
{
    static const auto aclCreateScalarList = GET_OP_API_FUNC(aclCreateScalarList);
    if (aclCreateScalarList == nullptr) {
        return nullptr;
    }

    std::vector<const aclScalar *> scalar_list(at_scalar_list.size());
    for (size_t i = 0; i < at_scalar_list.size(); i++) {
        scalar_list[i] = ConvertType(at_scalar_list[i]);
    }
    auto acl_scalar_list = aclCreateScalarList(scalar_list.data(), scalar_list.size());
    return acl_scalar_list;
}

inline aclTensor *ConvertType(const c10::optional<at::Tensor> &opt_tensor)
{
  if (opt_tensor.has_value() && opt_tensor.value().defined()) {
    return ConvertType(opt_tensor.value());
  }

  return nullptr;
}

inline aclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &opt_array)
{
  if (opt_array.has_value()) {
    return ConvertType(opt_array.value());
  }
  return nullptr;
}

inline aclScalar *ConvertType(const c10::optional<at::Scalar> &opt_scalar)
{
  if (opt_scalar.has_value()) {
    return ConvertType(opt_scalar.value());
  }
  return nullptr;
}

inline aclIntArray *ConvertType(const c10::OptionalIntArrayRef &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }

    return nullptr;
}

inline aclIntArray *ConvertType(const c10::OptionalArrayRef<c10::SymInt> &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }

    return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalarType)
{
  return kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalarType)];
}


// add declare from other hpp
typedef struct {
    const at::Tensor& tensor_;
    aclDataType dtype;
} TensorWrapper;

typedef struct {
    const at::TensorList& tensor_list_;
    aclDataType dtype;
} TensorListWrapper;


c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape);
// add declare from other hpp

inline aclTensor *ConvertType(const TensorWrapper &tensor_r)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    const at::Tensor &at_tensor = tensor_r.tensor_;

    if (!at_tensor.defined()) {
        return nullptr;
    }
    TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
        "Expected all tensors to be on the same device. "
        "Expected NPU tensor, please check whether the input tensor device is correct.");

    aclDataType acl_data_type = tensor_r.dtype;
    c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = array_to_small_vector(at_tensor.strides());
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = array_to_small_vector(at_tensor.sizes());

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    if (!IsOpInputBaseFormatCommon(at_tensor)) {
        format = static_cast<NPUStorageImpl *>(at_tensor.storage().unsafeGetStorageImpl())->npu_desc_.npu_format_;
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
            storageDims = static_cast<NPUStorageImpl *>(
                              at_tensor.storage().unsafeGetStorageImpl())
                              ->npu_desc_.storage_sizes_;
        }
    } else {
        switch (dimNum) {
            case NCL_DIM_NUM:
                format = ACL_FORMAT_NCL;
                break;
            case NCHW_DIM_NUM:
                format = ACL_FORMAT_NCHW;
                break;
            case NCDHW_DIM_NUM:
                format = ACL_FORMAT_NCDHW;
                break;
            default:
                format = ACL_FORMAT_ND;
        }
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
            storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
        }
    }

    auto acl_tensor =
        aclCreateTensor(wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, wrapperStride.data(),
                        at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

inline aclTensorList *ConvertType(const TensorListWrapper &tensor_list_wrapper)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(tensor_list_wrapper.tensor_list_.size());
    for (size_t i = 0; i < tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(TensorWrapper{
            tensor_list_wrapper.tensor_list_[i], tensor_list_wrapper.dtype});
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}


template <typename T>
T ConvertType(T value)
{
  return value;
}


template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr, std::index_sequence<I...>)
{
  typedef int (*OpApiFunc)(
      typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
  return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr)
{
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return ConvertToOpApiFunc(params, opApiAddr,
                            std::make_index_sequence<size>{});
}


// add release for all type
inline void Release(aclTensor *p)
{
  static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
  if (aclDestroyTensor == nullptr) {
    return;
  }
  aclDestroyTensor(p);
}

inline void Release(aclScalar *p)
{
  static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
  if (aclDestroyScalar == nullptr) {
    return;
  }
  aclDestroyScalar(p);
}

inline void Release(aclIntArray *p)
{
  static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
  if (aclDestroyIntArray == nullptr) {
    return;
  }

  aclDestroyIntArray(p);
}

inline void Release(aclBoolArray *p)
{
  static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
  if (aclDestroyBoolArray == nullptr) {
    return;
  }

  aclDestroyBoolArray(p);
}

inline void Release(aclTensorList *p)
{
  static const auto aclDestroyTensorList =
      GET_OP_API_FUNC(aclDestroyTensorList);
  if (aclDestroyTensorList == nullptr) {
    return;
  }

  aclDestroyTensorList(p);
}

inline void Release(aclScalarList *p)
{
    static const auto aclDestroyScalarList = GET_OP_API_FUNC(aclDestroyScalarList);
    if (aclDestroyScalarList == nullptr) {
        return;
    }

    aclDestroyScalarList(p);
}

template <typename T>
void Release(T value)
{
  (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>)
{
  (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple &t)
{
  static constexpr auto size = std::tuple_size<Tuple>::value;
  CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts &... args)
{
  return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>)
{
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t)
{
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}


// AddParamToBuf
#define MEMCPY_TO_BUF(data_expression, size_expression)                        \
  do {                                                                         \
    if (g_hashOffset + (size_expression) > kHashBufSize) {                     \
      g_hashOffset = kHashBufMaxSize;                                          \
      return;                                                                  \
    }                                                                          \
    memcpy(g_hashBuf + g_hashOffset, data_expression, size_expression);        \
    g_hashOffset += size_expression;                                           \
  } while (0)


template <std::size_t N>
void AddParamToBuf(const std::array<bool, N> &value)
{
  MEMCPY_TO_BUF(value.data(), value.size() * sizeof(bool));
}

template <typename T>
void AddParamToBuf(const T &value)
{
  MEMCPY_TO_BUF(&value, sizeof(T));
}

void AddParamToBuf(const at::Tensor &);
void AddParamToBuf(const at::Scalar &);
void AddParamToBuf(const at::IntArrayRef &);
void AddParamToBuf(const at::ArrayRef<bool> &);
void AddParamToBuf(const at::TensorList &);
void AddParamToBuf(const c10::optional<at::Tensor> &);
void AddParamToBuf(const c10::optional<at::IntArrayRef> &);
void AddParamToBuf(const c10::optional<at::Scalar> &);
void AddParamToBuf(const at::ScalarType);
void AddParamToBuf(const string &);
void AddParamToBuf();

template <typename T, typename... Args>
void AddParamToBuf(const T &arg, Args &... args)
{
  AddParamToBuf(arg);
  AddParamToBuf(args...);
}


// for cache
uint64_t CalcHashId();

typedef int (*InitHugeMemThreadLocal)(void *, bool);
typedef void (*UnInitHugeMemThreadLocal)(void *, bool);
typedef void (*ReleaseHugeMem)(void *, bool);
typedef aclOpExecutor *(*PTAGetExecCache)(uint64_t, uint64_t *);
typedef aclOpExecutor *(*PTAFindExecCache)(uint8_t *, size_t, uint64_t *);
typedef void (*InitPTACacheThreadLocal)();
typedef void (*SetPTAHashKey)(uint64_t);
typedef void (*SetPTACacheHashKey)(uint8_t *, size_t);
typedef bool (*CanUsePTACache)(const char *);
typedef void (*UnInitPTACacheThreadLocal)();


inline void UnInitCacheThreadLocal()
{
    static const auto unInitPTACacheThreadLocalAddr = GetOpApiFuncAddr("UnInitPTACacheThreadLocal");
    UnInitPTACacheThreadLocal unInitPTACacheThreadLocalFunc =
        reinterpret_cast<UnInitPTACacheThreadLocal>(unInitPTACacheThreadLocalAddr);
    if (unInitPTACacheThreadLocalFunc) {
        unInitPTACacheThreadLocalFunc();
    }
}


// ===================== PTA executor cache 快路径（对齐 torch_npu op_api_common.h hit_cache）=====================
// 目的: 同 shape 同地址重复调用跳过 GetWorkspaceSize/ConvertTypes（host 每调用省数百 us）。
// 安全性: 哈希覆盖 aclnn 名 + 全部入参（张量经 libtorch_npu 的 add_param_to_buf 序列化, 含地址/shape/dtype）
//         + stream 指针; miss 时保留 key, 由 GetWorkspaceSize 将新 executor 入缓存。
// 哈希缓冲为本 so 私有 thread_local（libtorch_npu 的同名符号为 hidden 不可跨 so 链接）。
constexpr int FIA_HASH_BUF_SIZE = 8192;
static thread_local char g_fiaHashBuf[FIA_HASH_BUF_SIZE];
static thread_local int g_fiaHashOffset = 0;

inline void FiaAddRaw(const void *data, size_t size)
{
    if (g_fiaHashOffset < 0 || g_fiaHashOffset >= FIA_HASH_BUF_SIZE) {
        g_fiaHashOffset = FIA_HASH_BUF_SIZE + 1024;  // overflow 标记: 禁用本次缓存
        return;
    }
    if (static_cast<size_t>(g_fiaHashOffset) + size > static_cast<size_t>(FIA_HASH_BUF_SIZE)) {
        g_fiaHashOffset = FIA_HASH_BUF_SIZE + 1024;
        return;
    }
    memcpy(g_fiaHashBuf + g_fiaHashOffset, data, size);
    g_fiaHashOffset += static_cast<int>(size);
}

// 张量序列化（保守含存储地址）: 地址变化/shape 变化/dtype 变化 → 不同 key, 杜绝错误复用
inline void FiaHashOne(const at::Tensor &t)
{
    uint8_t defined = t.defined() ? 1 : 0;
    FiaAddRaw(&defined, sizeof(defined));
    if (!t.defined()) {
        return;
    }
    uintptr_t ptr = reinterpret_cast<uintptr_t>(t.storage().data());
    int64_t off = t.storage_offset();
    int64_t dt = static_cast<int64_t>(t.scalar_type());
    int64_t dim = t.dim();
    FiaAddRaw(&ptr, sizeof(ptr));
    FiaAddRaw(&off, sizeof(off));
    FiaAddRaw(&dt, sizeof(dt));
    FiaAddRaw(&dim, sizeof(dim));
    for (int64_t i = 0; i < dim; ++i) {
        int64_t sz = t.sym_size(i).guard_int(__FILE__, __LINE__);
        int64_t st = t.sym_stride(i).guard_int(__FILE__, __LINE__);
        FiaAddRaw(&sz, sizeof(sz));
        FiaAddRaw(&st, sizeof(st));
    }
}
inline void FiaHashOne(const at::TensorList &ts)
{
    int64_t n = static_cast<int64_t>(ts.size());
    FiaAddRaw(&n, sizeof(n));
    for (const auto &t : ts) {
        FiaHashOne(t);
    }
}
inline void FiaHashOne(const c10::optional<at::Tensor> &t)
{
    if (t.has_value() && t.value().defined()) {
        FiaHashOne(t.value());
    } else {
        uint8_t none = 2;
        FiaAddRaw(&none, sizeof(none));
    }
}
inline void FiaHashOne(const std::string &v) { FiaAddRaw(v.c_str(), v.size() + 1); }
inline void FiaHashOne(const char *v) { FiaAddRaw(v, strlen(v) + 1); }
inline void FiaHashOne(char *v) { FiaAddRaw(v, strlen(v) + 1); }
template <typename T> inline void FiaHashOne(const T &v) { FiaAddRaw(&v, sizeof(T)); }

inline void FiaHashAll() {}
template <typename T, typename... Rest> inline void FiaHashAll(const T &first, const Rest &...rest)
{
    FiaHashOne(first);
    FiaHashAll(rest...);
}

inline uint64_t FiaCalcHash64()
{
    if (g_fiaHashOffset > FIA_HASH_BUF_SIZE) {
        return 0;  // overflow → 调用方禁用缓存
    }
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < g_fiaHashOffset; ++i) {
        h ^= static_cast<unsigned char>(g_fiaHashBuf[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

// ===================== 绑定层 aclnn 执行器缓存 ====================
// 等价官方 PTAGetExecCache 模式（该机制对自定义算子不插入, 故在绑定层自建）:
//   key   = aclnn 名 + 全部入参(张量地址/shape/stride/dtype + 标量) + stream 指针
//   value = {executor, workspaceSize}（慢路径 GetWorkspaceSize 产出后存入; 命中即以新 workspace 重发射）
// Repeated calls with the same addresses and shapes may skip
// ConvertTypes+GetWorkspaceSize. Device-side qlen, kvlen and metadata values
// are read again for every execution; the cache path never reads them on host.
#include <mutex>
#include <unordered_map>
struct FiaExecEntry {
    aclOpExecutor *executor = nullptr;
    uint64_t wsSize = 0;
};
static std::mutex g_fiaCacheMutex;
static std::unordered_map<uint64_t, FiaExecEntry> g_fiaCache;
static thread_local uint64_t g_fiaPendingKey = 0;   // >0: 本次慢路径结束后入缓存
static thread_local bool g_fiaSkipRelease = false;  // 慢路径入缓存时不释放 converted_params(executor 可能引用)
constexpr size_t FIA_EXEC_CACHE_MAX = 128;

template <typename... Args>
inline bool FiaTryExecCache(aclrtStream acl_stream, const char *aclnn_api, void *opApiFuncAddr,
                            uint64_t *wsSizeAddr, bool *cacheDisabled, Args &&...args)
{
    *cacheDisabled = true;
    g_fiaPendingKey = 0;
    g_fiaSkipRelease = false;
    // AICPU 算子（Metadata）executor 不支持重复发射（aclnn 返回失败, 实测）
    if (strstr(aclnn_api, "Metadata") != nullptr) {
        return false;
    }
    // AICore 主算子 executor 在当前 runtime 不支持安全重发射。
    constexpr bool enableOwnCache = false;
    if (!enableOwnCache) {
        return false;
    }
    g_fiaHashOffset = 0;
    FiaHashAll(std::string(aclnn_api), args..., reinterpret_cast<uintptr_t>(acl_stream));
    uint64_t key = FiaCalcHash64();
    if (key == 0) {
        return false;  // 哈希溢出等异常: 走慢路径且不入缓存
    }
    {
        std::lock_guard<std::mutex> lk(g_fiaCacheMutex);
        auto it = g_fiaCache.find(key);
        if (it != g_fiaCache.end()) {
            // 命中: 新 workspace + 重发射
            void *workspace_addr = nullptr;
            at::Tensor workspace_tensor;
            if (it->second.wsSize != 0) {
                auto options = at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kByte);
                workspace_tensor = at::empty({static_cast<int64_t>(it->second.wsSize)}, options);
                workspace_addr = const_cast<void *>(workspace_tensor.storage().data());
            }
            auto apiRet = reinterpret_cast<OpApiFunc>(opApiFuncAddr)(workspace_addr, it->second.wsSize,
                                                                     it->second.executor, acl_stream);
            TORCH_CHECK(apiRet == 0, "cached call ", aclnn_api, " failed");
            *cacheDisabled = false;
            return true;
        }
    }
    // miss: 登记 pending key, 慢路径完成后存入
    g_fiaPendingKey = key;
    g_fiaSkipRelease = true;
    *cacheDisabled = false;
    return false;
}

inline void FiaStoreExecCacheKeyed(uint64_t key, aclOpExecutor *executor, uint64_t wsSize)
{
    if (key == 0 || executor == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_fiaCacheMutex);
    if (g_fiaCache.size() >= FIA_EXEC_CACHE_MAX) {
        g_fiaCache.clear();  // 简单上限保护（正常负载 key 数极少）
    }
    g_fiaCache[key] = FiaExecEntry{executor, wsSize};
}

#define EXEC_NPU_CMD_V1(aclnn_api, ...)                                                                                \
    do {                                                                                                               \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");                   \
        static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");                                       \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",               \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    "not found.");                                                                                     \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        uint64_t workspace_size = 0;                                                                                   \
        uint64_t *workspace_size_addr = &workspace_size;                                                               \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
        InitPTACacheThreadLocal initPTACacheThreadLocalFunc =                                                          \
            reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);                                    \
        SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);                          \
        uint64_t fiaKeyThisCall = 0;                                                                                   \
        bool fiaKeepConverted = false;                                                                                 \
        {  /* 绑定层执行器缓存: 命中即 launch 并跳出; 未命中登记 key 由慢路径存入 */                                  \
            uint64_t cacheWsSize = 0;                                                                                  \
            bool ptaDisabled = true;                                                                                   \
            if (FiaTryExecCache(acl_stream, #aclnn_api, opApiFuncAddr, &cacheWsSize, &ptaDisabled, __VA_ARGS__)) {     \
                break;                                                                                                 \
            }                                                                                                          \
            fiaKeyThisCall = g_fiaPendingKey;                                                                          \
            fiaKeepConverted = g_fiaSkipRelease;                                                                       \
            g_fiaPendingKey = 0;                                                                                       \
            g_fiaSkipRelease = false;                                                                                  \
            if (ptaDisabled && initPTACacheThreadLocalFunc && setPTAHashKeyFunc) {                                     \
                initPTACacheThreadLocalFunc();                                                                         \
                setPTAHashKeyFunc(0);                                                                                  \
            }                                                                                                          \
        }                                                                                                              \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                         \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        TORCH_CHECK(workspace_status== 0, "call " #aclnn_api " failed");                                               \
        void *workspace_addr = nullptr;                                                                                \
        at::Tensor workspace_tensor;                                                                                   \
        if (workspace_size != 0) {                                                                                     \
            at::TensorOptions options =                                                                                \
                at::TensorOptions(torch_npu::utils::get_npu_device_type());                                            \
            auto workspace_tensor =                                                                                    \
                at::empty({static_cast<int64_t>(workspace_size)}, options.dtype(at::kByte));                           \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor,                      \
                         fiaKeyThisCall, fiaKeepConverted]() -> int {                                                  \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                          \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                            \
            TORCH_CHECK(api_ret==0, "call " #aclnn_api " failed");                                                     \
            FiaStoreExecCacheKeyed(fiaKeyThisCall, executor, workspace_size);                                          \
            if (!fiaKeepConverted) {                                                                                   \
                ReleaseConvertTypes(converted_params);                                                                 \
                ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                      \
                if (releaseMemFunc) {                                                                                                      releaseMemFunc(nullptr, false);                                                                    \
                }                                                                                                      \
            }                                                                                                          \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call);                                                   \
        if (unInitMemFunc) {                                                                                           \
            unInitMemFunc(nullptr, false);                                                                             \
        }                                                                                                              \
        UnInitCacheThreadLocal();                                                                                      \
    } while (false)


#define EXEC_NPU_CMD_v0(aclnn_api, ...)                                          \
  do {                                                                        \
    static const auto getWorkspaceSizeFuncAddr =                              \
        GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                      \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);           \
    static const auto initMemAddr =                                           \
        GetOpApiFuncAddr("InitHugeMemThreadLocal");                           \
    static const auto unInitMemAddr =                                         \
        GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                         \
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");    \
    TORCH_CHECK(                                                              \
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,      \
        #aclnn_api, " or ", #aclnn_api "GetWorkspaceSize", " not in ",        \
        GetOpApiLibName(), ", or ", GetOpApiLibName(), "not found.");         \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);           \
    uint64_t workspace_size = 0;                                              \
    uint64_t *workspace_size_addr = &workspace_size;                          \
    aclOpExecutor *executor = nullptr;                                        \
    aclOpExecutor **executor_addr = &executor;                                \
    InitHugeMemThreadLocal initMemFunc =                                      \
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                \
    UnInitHugeMemThreadLocal unInitMemFunc =                                  \
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
    if (initMemFunc) {                                                        \
      initMemFunc(nullptr, false);                                            \
    }                                                                         \
    auto converted_params =                                                   \
        ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);        \
    static auto getWorkspaceSizeFunc =                                        \
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);       \
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);     \
    TORCH_CHECK(workspace_status == 0,                                        \
                "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg()); \
    void *workspace_addr = nullptr;                                           \
    if (workspace_size != 0) {                                                \
      at::TensorOptions options =                                             \
          at::TensorOptions(torch_npu::utils::get_npu_device_type());         \
      auto workspace_tensor =                                                 \
          at::empty({workspace_size}, options.dtype(at::kByte));                  \
      workspace_addr = const_cast<void *>(workspace_tensor.storage().data()); \
    }                                                                         \
    auto acl_call = [converted_params, workspace_addr, workspace_size,        \
                     acl_stream, executor]() -> int {                         \
      typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *,             \
                               const aclrtStream);                            \
      OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);       \
      auto api_ret =                                                          \
          opApiFunc(workspace_addr, workspace_size, executor, acl_stream);    \
      TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:",        \
                  aclGetRecentErrMsg());                                      \
      ReleaseConvertTypes(converted_params);                                  \
      ReleaseHugeMem releaseMemFunc =                                         \
          reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                   \
      if (releaseMemFunc) {                                                   \
        releaseMemFunc(nullptr, false);                                       \
      }                                                                       \
      return api_ret;                                                         \
    };                                                                        \
    at_npu::native::OpCommand cmd;                                            \
    cmd.Name(#aclnn_api);                                                     \
    cmd.SetCustomHandler(acl_call);                                           \
    cmd.Run();                                                                \
    if (unInitMemFunc) {                                                      \
      unInitMemFunc(nullptr, false);                                          \
    }                                                                         \
  } while (false)

struct TensorStruct {
    void *data_ptr = nullptr;       // at_tensor.storage().data()
    aclDataType acl_type;           // aclDataType of at_tensor
    aclFormat acl_format;
    size_t nbytes;                  // at_tensor.storage().nbytes()
    size_t itemsize;                // at_tensor.itemsize()
    int64_t storage_offset;         // at_tensor.storage_offset()
    std::vector<int64_t> sizes;     // at_tensor.sizes()
    std::vector<int64_t> strides;   // at_tensor.strides()
    std::vector<int64_t> storage_sizes;

    TensorStruct(
        void *data_ptr_, aclDataType acl_type_, aclFormat acl_format_,
        size_t nbytes_, size_t itemsize_, int64_t storage_offset_,
        at::IntArrayRef sizes_, at::IntArrayRef strides_, at::IntArrayRef storage_sizes_
    ) : data_ptr(data_ptr_), acl_type(acl_type_), acl_format(acl_format_),
        nbytes(nbytes_), itemsize(itemsize_), storage_offset(storage_offset_),
        sizes(sizes_.vec()), strides(strides_.vec()), storage_sizes(storage_sizes_.vec())
    {
    }
};
using TensorStructPtr = std::shared_ptr<TensorStruct>;

inline aclTensor *ConvertTypeV2(TensorStructPtr at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    if (at_tensor == nullptr) {
        return nullptr;
    }
    aclDataType acl_data_type = (*at_tensor).acl_type;
    c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;

    const auto dimNum = (*at_tensor).sizes.size();
    aclFormat format = ACL_FORMAT_ND;
    if (!IsBaseFormatTypeCommon((*at_tensor).acl_format)) {
        format = (*at_tensor).acl_format;
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK((*at_tensor).itemsize > 0, "the itemsize of tensor must be greater than 0.");
            storageDims = (*at_tensor).storage_sizes;
        }
    } else {
        switch (dimNum) {
            case NCL_DIM_NUM:
                format = ACL_FORMAT_NCL;
                break;
            case NCHW_DIM_NUM:
                format = ACL_FORMAT_NCHW;
                break;
            case NCDHW_DIM_NUM:
                format = ACL_FORMAT_NCDHW;
                break;
            default:
                format = ACL_FORMAT_ND;
        }
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK((*at_tensor).itemsize > 0, "the itemsize of tensor must be greater than 0.");
            storageDims.push_back((*at_tensor).nbytes / (*at_tensor).itemsize);
        }
    }

    auto acl_tensor = aclCreateTensor(
        (*at_tensor).sizes.data(), (*at_tensor).sizes.size(), acl_data_type, (*at_tensor).strides.data(),
        (*at_tensor).storage_offset, format, storageDims.data(), storageDims.size(), (*at_tensor).data_ptr);
    return acl_tensor;
}

inline aclScalar *ConvertTypeV2(const at::Scalar &at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
    aclScalar *acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double:
            {
                double value = at_scalar.toDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Long:
            {
                int64_t value = at_scalar.toLong();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Bool:
            {
                bool value = at_scalar.toBool();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::ComplexDouble:
            {
                auto value = at_scalar.toComplexDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        default:
            acl_scalar = nullptr;
            break;
    }

    return acl_scalar;
}

inline aclIntArray *ConvertTypeV2(const std::vector<int64_t> &int_list)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto array = aclCreateIntArray(int_list.data(), int_list.size());
    return array;
}

template <std::size_t N> inline aclBoolArray *ConvertTypeV2(const std::array<bool, N> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclBoolArray *ConvertTypeV2(const std::vector<bool> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    void *temp_value_ptr = malloc(value.size() * sizeof(bool));
    if (temp_value_ptr == nullptr) {
        return nullptr;
    }

    bool *value_ptr = reinterpret_cast<bool *>(temp_value_ptr);
    for (size_t i = 0; i < value.size(); i++) {
        value_ptr[i] = value[i];
    }
    auto array = aclCreateBoolArray(value_ptr, value.size());
    free(value_ptr);
    return array;
}

inline aclTensorList *ConvertTypeV2(const std::vector<TensorStructPtr> &at_tensor_list)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = ConvertTypeV2(at_tensor_list[i]);
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

inline aclScalarList *ConvertTypeV2(const std::vector<at::Scalar> &at_scalar_list)
{
    static const auto aclCreateScalarList = GET_OP_API_FUNC(aclCreateScalarList);
    if (aclCreateScalarList == nullptr) {
        return nullptr;
    }

    std::vector<const aclScalar *> scalar_list(at_scalar_list.size());
    for (size_t i = 0; i < at_scalar_list.size(); i++) {
        scalar_list[i] = ConvertTypeV2(at_scalar_list[i]);
    }
    auto acl_scalar_list = aclCreateScalarList(scalar_list.data(), scalar_list.size());
    return acl_scalar_list;
}

inline aclIntArray *ConvertTypeV2(const c10::optional<std::vector<int64_t>> &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertTypeV2(opt_array.value());
    }

    return nullptr;
}

inline aclScalar *ConvertTypeV2(const c10::optional<at::Scalar> &opt_scalar)
{
    if (opt_scalar.has_value()) {
        return ConvertTypeV2(opt_scalar.value());
    }

    return nullptr;
}

inline aclDataType ConvertTypeV2(const at::ScalarType scalarType)
{
    return kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalarType)];
}

inline char* ConvertTypeV2(const std::string &str)
{
    char* string_ptr = const_cast<char *>(str.c_str());
    return string_ptr;
}

template <typename T> T ConvertTypeV2(T value)
{
    return value;
}

template <typename Tuple, std::size_t... I>
auto convert_types_impl_v2(const Tuple &t, std::index_sequence<I...>)
{
    return std::make_tuple(ConvertTypeV2(std::get<I>(t))...);
}

template <typename... Ts> constexpr auto ConvertTypesV2(
    const std::tuple<Ts...> &args,
    uint64_t *workspace_size_addr, aclOpExecutor **executor_addr)
{
    auto convert_args = convert_types_impl_v2(args, std::make_index_sequence<sizeof...(Ts)>{});
    auto appends = std::make_tuple(workspace_size_addr, executor_addr);
    return std::tuple_cat(convert_args, appends);
}

inline TensorStructPtr CopyTypeV2(const at::Tensor &at_tensor)
{
    if (!at_tensor.defined()) {
        return nullptr;
    }
    aclDataType acl_data_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at_tensor.scalar_type())];
    return std::make_shared<TensorStruct>(
        const_cast<void *>(at_tensor.storage().data()),
        acl_data_type,
        static_cast<NPUStorageImpl *>(at_tensor.storage().unsafeGetStorageImpl())->npu_desc_.npu_format_,
        at_tensor.storage().nbytes(),
        at_tensor.itemsize(),
        at_tensor.storage_offset(),
        at_tensor.sizes(),
        at_tensor.strides(),
        static_cast<NPUStorageImpl *>(at_tensor.storage().unsafeGetStorageImpl())->npu_desc_.storage_sizes_);
}

inline TensorStructPtr CopyTypeV2(const TensorWrapper &tensor_r)
{
    const at::Tensor &at_tensor = tensor_r.tensor_;
    if (!at_tensor.defined()) {
        return nullptr;
    }
    TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
        "Expected all tensors to be on the same device. "
        "Expected NPU tensor, please check whether the input tensor device is correct.");
    return std::make_shared<TensorStruct>(
        const_cast<void *>(at_tensor.storage().data()),
        tensor_r.dtype,
        static_cast<NPUStorageImpl *>(at_tensor.storage().unsafeGetStorageImpl())->npu_desc_.npu_format_,
        at_tensor.storage().nbytes(),
        at_tensor.itemsize(),
        at_tensor.storage_offset(),
        at_tensor.sizes(),
        at_tensor.strides(),
        static_cast<NPUStorageImpl *>(at_tensor.storage().unsafeGetStorageImpl())->npu_desc_.storage_sizes_);
}

inline std::vector<int64_t> CopyTypeV2(const at::IntArrayRef &at_array)
{
    return at_array.vec();
}

inline std::vector<int64_t> CopyTypeV2(const at::ArrayRef<c10::SymInt> &at_array)
{
    auto int_array = c10::asIntArrayRefUnchecked(at_array);
    return int_array.vec();
}

template <std::size_t N> inline std::array<bool, N> CopyTypeV2(const std::array<bool, N> &value)
{
    return value;
}

inline std::vector<bool> CopyTypeV2(const at::ArrayRef<bool> &value)
{
    return value.vec();
}

inline std::vector<TensorStructPtr> CopyTypeV2(const at::TensorList &at_tensor_list)
{
    std::vector<TensorStructPtr> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = CopyTypeV2(at_tensor_list[i]);
    }
    return tensor_list;
}

inline std::vector<TensorStructPtr> CopyTypeV2(const TensorListWrapper &tensor_list_wrapper)
{
    std::vector<TensorStructPtr> tensor_list(tensor_list_wrapper.tensor_list_.size());
    for (size_t i = 0; i < tensor_list.size(); i++) {
        tensor_list[i] = CopyTypeV2(TensorWrapper{
            tensor_list_wrapper.tensor_list_[i], tensor_list_wrapper.dtype});
    }
    return tensor_list;
}

inline std::vector<at::Scalar> CopyTypeV2(const at::ArrayRef<at::Scalar> &at_scalar_list)
{
    return at_scalar_list.vec();
}

inline TensorStructPtr CopyTypeV2(const c10::optional<at::Tensor> &opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return CopyTypeV2(opt_tensor.value());
    }

    return nullptr;
}

inline c10::optional<std::vector<int64_t>> CopyTypeV2(const c10::optional<at::IntArrayRef> &opt_array)
{
    if (opt_array.has_value()) {
        return CopyTypeV2(opt_array.value());
    }

    return c10::nullopt;
}

inline c10::optional<std::vector<int64_t>> CopyTypeV2(const c10::OptionalArrayRef<c10::SymInt> &opt_array)
{
    if (opt_array.has_value()) {
        return CopyTypeV2(opt_array.value());
    }

    return c10::nullopt;
}

inline c10::optional<std::vector<int64_t>> CopyTypeV2(const c10::OptionalIntArrayRef &opt_array)
{
    if (opt_array.has_value()) {
        return CopyTypeV2(opt_array.value());
    }

    return c10::nullopt;
}

inline std::string CopyTypeV2(char* str)
{
    std::string result = str;
    return result;
}

template <typename T> T CopyTypeV2(T value)
{
    return value;
}

template <typename... Ts>
constexpr auto CopyTypesV2(Ts &...args)
{
    return std::make_tuple(CopyTypeV2(args)...);
}

#define EXEC_UPDATE_NPU_CMD_V1(aclnn_api, workspace_addr, workspace_size, ...)                         \
    do {                                                                                                               \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");                   \
        static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");                                       \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",               \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    "not found.");                                                                                     \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
        InitPTACacheThreadLocal initPTACacheThreadLocalFunc =                                                          \
            reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);                                    \
        SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);                          \
        if (initPTACacheThreadLocalFunc && setPTAHashKeyFunc) {                                                        \
            initPTACacheThreadLocalFunc();                                                                             \
            setPTAHashKeyFunc(0);                                                                                      \
        }                                                                                                              \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto copied_params = CopyTypesV2(__VA_ARGS__);                                                                 \
        uint64_t fake_workspace_size = 0;                                                                              \
        uint64_t *workspace_size_addr = &fake_workspace_size;                                                          \
        auto converted_params = ConvertTypesV2(copied_params, workspace_size_addr, executor_addr);                     \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        TORCH_CHECK(workspace_status== 0, "call " #aclnn_api " failed");                                               \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]()->int {              \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                          \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                            \
            TORCH_CHECK(api_ret==0, "call " #aclnn_api " failed");                                                     \
            ReleaseConvertTypes(converted_params);                                                                     \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                          \
            if (releaseMemFunc) {                                                                                      \
                releaseMemFunc(nullptr, false);                                                                        \
            }                                                                                                          \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call);                                                   \
        if (unInitMemFunc) {                                                                                           \
            unInitMemFunc(nullptr, false);                                                                             \
        }                                                                                                              \
        UnInitCacheThreadLocal();                                                                                      \
    } while (false)

#define EXEC_GET_MAX_WORKSPACE_CMD(aclnn_api, ...)                                                                     \
    [](const char *apiName, auto &...args)->auto {                                                                     \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetMaxWorkspaceSize");               \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");                   \
        static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");                                       \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr, #aclnn_api "GetMaxWorkspaceSize", " not in ",                 \
                    GetOpApiLibName(), ", or ", GetOpApiLibName(), "not found.");             \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        uint64_t workspace_size = 0;                                                                                   \
        uint64_t *workspace_size_addr = &workspace_size;                                                               \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        auto converted_params = ConvertTypes(args..., workspace_size_addr, executor_addr);                             \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed");                                              \
        ReleaseConvertTypes(converted_params);                                                                         \
        return workspace_size;                                                                                         \
    }(#aclnn_api, __VA_ARGS__)

#endif  // TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_
