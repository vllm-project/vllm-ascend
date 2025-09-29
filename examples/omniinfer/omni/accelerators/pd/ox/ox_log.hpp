// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once
#include <iostream>

#ifdef NDEBUG

#define LOG_DEBUG null_stream()
#define LOG_INFO null_stream()
#define LOG_WARNING null_stream()
#define LOG_ERROR null_stream()

class null_stream
{
public:
    template <typename T>
    null_stream &operator<<(const T &) { return *this; }

    null_stream &operator<<(std::ostream &(*)(std::ostream &)) { return *this; }
};

#else

#define LOG_DEBUG std::cout << "[DEBUG] "
#define LOG_INFO std::cout << "[INFO] "
#define LOG_WARNING std::cout << "[WARNING] "
#define LOG_ERROR std::cerr << "[ERROR] "
#endif