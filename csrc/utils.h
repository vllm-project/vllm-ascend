#pragma once

template <typename scalar_t>
struct acc_type;

template<>
struct acc_type<bfloat16_t> {
    using type = float;
};


template <>
struct acc_type<half> {
    using type = half;
};

template <>
struct acc_type<float> {
    using type = float;
};

template<>
struct acc_type<int8_t> {
  using type = int;
};
