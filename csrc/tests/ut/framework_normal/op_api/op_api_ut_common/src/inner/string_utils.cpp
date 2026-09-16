/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "op_api_ut_common/inner/string_utils.h"

using namespace std;

vector<string> Split(const string &s, const string &spliter)
{
    vector<string> strings;
    size_t i = 0;
    while (i < s.size()) {
        auto pos = s.find_first_of(spliter, i);
        if (pos == string::npos) {
            strings.emplace_back(s, i);
            break;
        } else {
            strings.emplace_back(s, i, pos - i);
            i = pos + spliter.size();
        }
    }

    return strings;
}

string &Ltrim(string &s)
{
    (void)s.erase(s.begin(), find_if(s.begin(), s.end(), [](const char c) {
                      return static_cast<bool>(isspace(static_cast<unsigned char>(c)) == 0);
                  }));
    return s;
}

string &Rtrim(string &s)
{
    (void)s.erase(find_if(s.rbegin(), s.rend(),
                          [](const char c) { return static_cast<bool>(isspace(static_cast<unsigned char>(c)) == 0); })
                      .base(),
                  s.end());
    return s;
}

string Trim(string &s)
{
    return Ltrim(Rtrim(s));
}
