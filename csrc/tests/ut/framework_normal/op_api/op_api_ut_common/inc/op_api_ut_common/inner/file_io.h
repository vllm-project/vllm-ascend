/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _OP_API_UT_COMMON_FILE_IO_H_
#define _OP_API_UT_COMMON_FILE_IO_H_

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

using namespace std;

void *ReadBinFile(const string &file_name, size_t &size);
int WriteBinFile(const void *host_mem, const string &file_name, size_t size);
bool FileExists(const string &file_name);
void DeleteUtTmpFiles(const string &file_prefix);
void DeleteCwdFilesEndsWith(const string &file_suffix);
string RealPath(const string &path);
void SetUtTmpFileSwitch();
bool GetUtTmpFileSwitch();
bool IsDir(const string &path);
void GetFilesWithSuffix(const string &path, const string &suffix, vector<string> &files);
unique_ptr<char[]> GetBinFromFile(const string &path, uint32_t &data_len);

#endif
