/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

using namespace std;

class OpApiUtEnvironment : public testing::Environment {
public:
    OpApiUtEnvironment()
    {
        cout << "OpApiUtEnvironment new" << endl;
    }
    ~OpApiUtEnvironment()
    {
        cout << "OpApiUtEnvironment delete" << endl;
    }
    virtual void SetUp()
    {
        cout << "Global Environment SetpUp." << endl;
    }

    virtual void TearDown()
    {
        cout << "Global Environment TearDown" << endl;
    }
};

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new OpApiUtEnvironment());

    return RUN_ALL_TESTS();
}
