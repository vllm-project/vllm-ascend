// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "acl/acl.h"
#include "gtest/gtest.h"
#include <iostream>

int main(int argc, char **argv) {
    std::cout << "Running main() from custom test_main.cpp" << std::endl;

    // Initialize Google Test.
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests.
    return RUN_ALL_TESTS();
}