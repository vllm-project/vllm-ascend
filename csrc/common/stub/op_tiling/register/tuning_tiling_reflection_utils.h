/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_REGISTER_TUNING_TILING_REFLECTION_UTILS_HEADER__
#define __INC_REGISTER_TUNING_TILING_REFLECTION_UTILS_HEADER__
#include <string>
#include <type_traits>
#include <tuple>
#include <nlohmann/json.hpp>

namespace tuningtiling {
// implement for std c++11
template <class T>
using decay_t = typename std::decay<T>::type;

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename T, T... Ints>
struct integer_sequence {
    using value_type = T;
    static constexpr std::size_t size()
    {
        return sizeof...(Ints);
    }
};

template <std::size_t... Ints>
using index_sequence = integer_sequence<std::size_t, Ints...>;

template <typename T, std::size_t N, T... Is>
struct make_integer_sequence : make_integer_sequence<T, N - 1U, N - 1U, Is...> {
};

template <typename T, T... Is>
struct make_integer_sequence<T, 0, Is...> : integer_sequence<T, Is...> {
};

template <std::size_t N>
using make_index_sequence = make_integer_sequence<std::size_t, N>;

template <typename T>
struct StructInfo {
    static std::tuple<> Info()
    {
        return std::make_tuple();
    }
};

#define DECLARE_SCHEMA(Struct, ...)                          \
    template <>                                              \
    struct StructInfo<Struct> {                              \
        static decltype(std::make_tuple(__VA_ARGS__)) Info() \
        {                                                    \
            return std::make_tuple(__VA_ARGS__);             \
        }                                                    \
    };

#define FIELD(class, FieldName) std::make_tuple(#FieldName, &class ::FieldName)

template <typename Fn, typename Tuple, typename Field, std::size_t... Is>
void ForEachTuple(Tuple&& tuple, Field&& fields, Fn&& fn, index_sequence<Is...>)
{
    (void)std::initializer_list<size_t>{
        (fn(std::get<0>(std::get<Is>(fields)), tuple.*std::get<1>(std::get<Is>(fields))), Is)...};
}

template <typename Fn, typename Tuple>
void ForEachTuple(Tuple&& tuple, Fn&& fn)
{
    const auto fields = StructInfo<decay_t<Tuple>>::Info();
    ForEachTuple(
        std::forward<Tuple>(tuple), fields, std::forward<Fn>(fn),
        make_index_sequence<std::tuple_size<decltype(fields)>::value>{});
}

template <typename T>
struct is_optional : std::false_type {
};

template <typename T>
struct is_optional<std::unique_ptr<T>> : std::true_type {
};

template <typename T>
bool is_optional_v()
{
    return is_optional<decay_t<T>>::value;
}

template <typename T>
decltype(std::begin(T()), std::true_type{}) containable(size_t);

template <typename T>
std::false_type containable(...);

template <typename T>
using is_containable = decltype(containable<T>(0U));

template <typename T>
constexpr bool IsSerializeType()
{
    return ((!std::is_class<decay_t<T>>::value) || is_containable<decay_t<T>>());
}

template <typename T, typename Fn>
void ForEachField(T&& value, Fn&& fn)
{
    ForEachTuple(std::forward<T>(value), std::forward<Fn>(fn));
}

template <typename Fn>
struct DumpFunctor;

template <typename T, typename Js, enable_if_t<!IsSerializeType<T>()>* = nullptr>
void DumpObj(T&& obj, const std::string& field_name, Js& j)
{
    if (field_name.empty()) {
        ForEachField(std::forward<T>(obj), DumpFunctor<Js>(j));
        return;
    }
    ForEachField(std::forward<T>(obj), DumpFunctor<Js>(j[field_name]));
}

template <typename T, typename Js, enable_if_t<IsSerializeType<T>()>* = nullptr>
void DumpObj(T&& obj, const std::string& field_name, Js& j)
{
    if (field_name.empty()) {
        return;
    }
    j[field_name] = std::forward<T>(obj);
}

template <typename T>
struct DumpFunctor {
    explicit DumpFunctor(T& j) : js(j)
    {}
    template <typename Name, typename Field>
    void operator()(Name&& name, Field&& field) const
    {
        DumpObj(std::forward<Field>(field), std::forward<Name>(name), js);
    }
    T& js;
};

template <typename Fn>
struct FromJsonFunctor;

template <typename T, typename Js, enable_if_t<!IsSerializeType<T>()>* = nullptr>
void FromJsonImpl(T&& obj, const std::string& field_name, const Js& j)
{
    if (field_name.empty()) {
        ForEachField(std::forward<T>(obj), FromJsonFunctor<Js>(j));
        return;
    }
    if (j.find(field_name) == j.cend()) {
        return;
    }
    ForEachField(std::forward<T>(obj), FromJsonFunctor<Js>(j[field_name]));
}

template <typename T, typename Js, enable_if_t<IsSerializeType<T>()>* = nullptr>
void FromJsonImpl(T&& obj, const std::string& field_name, const Js& j)
{
    // ignore missing field of optional
    if ((tuningtiling::is_optional_v<decltype(obj)>()) || (j.find(field_name) == j.cend())) {
        return;
    }
    j.at(field_name).get_to(std::forward<T>(obj));
}

template <typename Js>
struct FromJsonFunctor {
    explicit FromJsonFunctor(const Js& j) : js(j)
    {}
    template <typename Name, typename Field>
    void operator()(Name&& name, Field&& field) const
    {
        FromJsonImpl(std::forward<Field>(field), std::forward<Name>(name), js);
    }
    const Js& js;
};
} // namespace tuningtiling
#endif