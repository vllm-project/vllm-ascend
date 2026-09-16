/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _OP_API_UT_COMMON_TUPLE_UTILS_H_
#define _OP_API_UT_COMMON_TUPLE_UTILS_H_

#include <tuple>
#include <assert.h>
#include "nlohmann/json.hpp"

#include "opdev/common_types.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/inner/rts_interface.h"

using namespace std;
using namespace nlohmann;

////////// transfer Acl***Desc to acl*** /////////

template <typename T>
inline T DescToAclContainer(const T &t)
{
    return t;
}

template <typename T>
constexpr auto ConvertDescToAclTypes(T t)
{
    constexpr auto size = tuple_size<T>::value;
    return ConvertDescToAclTypes(t, make_index_sequence<size>{});
}

template <typename T, size_t... I>
constexpr auto ConvertDescToAclTypes(T t, index_sequence<I...>)
{
    return make_tuple(DescToAclContainer(get<I>(t))...);
}

/////////////////////// release acl*** ////////////

template <typename T>
void Release(T p)
{
    (void)p;
    return;
}

template <typename T, typename... Rest>
struct IsAny : false_type {};

template <typename T, typename First>
struct IsAny<T, First> : is_same<T, First> {};

template <typename T, typename First, typename... Rest>
struct IsAny<T, First, Rest...> : integral_constant<bool, is_same_v<T, First> || IsAny<T, Rest...>::value> {};

template <typename... T>
void ReleaseAclTypes(const tuple<T...> &t)
{
    apply(
        [](const auto &...args) {
            ((IsAny<remove_const_t<remove_pointer_t<remove_reference_t<decltype(args)>>>, aclTensor, aclScalar,
                    aclTensorList, aclIntArray, aclFloatArray, aclBoolArray>::value ?
                  Release(args) :
                  void()),
             ...);
        },
        t);
}

////////////  infer acl*** from Acl***Desc  ////////////

template <typename T>
inline T InferAclType(const T &t)
{
    return t;
}

template <typename T>
inline constexpr auto InferAclTypes(T t)
{
    constexpr auto size = tuple_size<T>::value;
    return InferAclTypes(t, make_index_sequence<size>{});
}

template <typename T, size_t... I>
inline constexpr auto InferAclTypes(T t, index_sequence<I...>)
{
    return make_tuple(InferAclType(get<I>(t))...);
}

////////////  build json ////////

template <typename T>
inline void DescToJson(json &root, const T &t, [[maybe_unused]] bool is_input = true)
{
    json x;
    stringstream ss;
    if constexpr (is_pointer_v<T>) {
        if (t == nullptr) {
            ss << "None";
        } else if constexpr (std::is_same_v<std::decay_t<T>, char *> || std::is_same_v<std::decay_t<T>, const char *>) {
            ss << t;
        } else {
            assert(false && "Not support yet !!");
        }
    } else if (typeid(T) == typeid(uint8_t)) {
        ss << static_cast<uint>(t);
    } else if (typeid(T) == typeid(int8_t)) {
        ss << static_cast<int>(t);
    } else {
        ss << t;
    }
    x["value"] = ss.str();
    x["dtype"] = typeid(t).name();
    root.push_back(x);
}

template <typename... T>
inline void ConvertDescToJson(json &root, const tuple<T...> &t, bool is_input = true)
{
    apply([&root, &is_input](const auto &...args) { (DescToJson(root, args, is_input), ...); }, t);
}

////////////  invoke acl***GetWorkspaceSize function ////////

template <typename Function, typename Tuple, typename ExtraT, size_t... I, size_t... EI>
inline aclnnStatus InvokeGetWorkspaceSizeFunc(Function f, Tuple t, [[maybe_unused]] ExtraT et, index_sequence<I...>,
                                              index_sequence<EI...>, uint64_t *workspace_size, aclOpExecutor **executor)
{
    return f(get<I>(t)..., get<EI>(et)..., workspace_size, executor);
}

template <typename Function, typename Tuple, typename ExtraT>
inline aclnnStatus InvokeGetWorkspaceSizeFunc(Function f, Tuple t, ExtraT et, uint64_t *workspace_size,
                                              aclOpExecutor **executor)
{
    constexpr auto s = tuple_size<Tuple>::value;
    constexpr auto es = tuple_size<ExtraT>::value;
    return InvokeGetWorkspaceSizeFunc(f, t, et, make_index_sequence<s>{}, make_index_sequence<es>{}, workspace_size,
                                      executor);
}

#endif
