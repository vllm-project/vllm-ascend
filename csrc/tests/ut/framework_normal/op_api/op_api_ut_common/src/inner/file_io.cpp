/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dirent.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>
#include "op_api_ut_common/inner/file_io.h"

using namespace std;

static bool ut_tmp_file_switch = false;

void SetUtTmpFileSwitch()
{
    constexpr const char *ENV_KEEP_UT_TMP_FILE = "KEEP_UT_TMP_FILE";
    const char *p = getenv(ENV_KEEP_UT_TMP_FILE);
    if (p == nullptr) {
        ut_tmp_file_switch = false;
        return;
    }
    auto pv = string(p);
    if (pv == "true") {
        ut_tmp_file_switch = true;
    } else {
        ut_tmp_file_switch = false;
    }
}

bool GetUtTmpFileSwitch()
{
    return ut_tmp_file_switch;
}

void *ReadBinFile(const string &file_name, size_t &size)
{
    size = 0;
    ifstream file(file_name, ios::in | ios::binary);
    if (!file) {
        cout << "Failed to open file for reading: " << file_name << endl;
        return nullptr;
    }

    file.seekg(0, ios_base::end);
    streampos file_size = file.tellg();
    file.seekg(0, ios_base::beg);

    char *p = new (nothrow) char[file_size];
    if (p == nullptr) {
        file.close();
        cout << "Malloc buffer to read binary file failed. size: " << file_size << endl;
        return nullptr;
    }

    if (!file.read(p, file_size)) {
        cout << "Failed to read binary file: " << file_name << endl;
        delete[] p;
        p = nullptr;
    } else {
        size = file_size;
    }

    file.close();
    return p;
}

int WriteBinFile(const void *host_mem, const string &file_name, size_t size)
{
    ofstream file(file_name, ios::out | ios::binary);
    if (!file) {
        cout << "Failed to open file for writing: " << file_name << endl;
        return 1;
    }

    if (!file.write(static_cast<const char *>(host_mem), size)) {
        cout << "Failed to write binary file: " << file_name << endl;
        file.close();
        return 1;
    }
    file.close();
    return 0;
}

bool FileExists(const string &file_name)
{
    struct stat buf;
    return stat(file_name.c_str(), &buf) == 0;
}

void DeleteUtTmpFiles(const string &file_prefix)
{
    if (GetUtTmpFileSwitch()) {
        return;
    }

    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        return;
    }

    DIR *dir = opendir(cwd);
    if (dir == NULL) {
        return;
    }

    struct dirent *entry;
    string json_file = file_prefix + ".json";
    string input_file_prefix = file_prefix + "_input_";
    string output_file_prefix = file_prefix + "_output_";
    string golden_file_prefix = file_prefix + "_golden_";
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, input_file_prefix.c_str(), input_file_prefix.size()) == 0 ||
            strncmp(entry->d_name, output_file_prefix.c_str(), output_file_prefix.size()) == 0 ||
            strncmp(entry->d_name, golden_file_prefix.c_str(), golden_file_prefix.size()) == 0 ||
            strncmp(entry->d_name, json_file.c_str(), json_file.size()) == 0) {
            (void)remove(entry->d_name);
        }
    }

    closedir(dir);
    return;
}

void DeleteCwdFilesEndsWith(const string &file_suffix)
{
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        return;
    }

    DIR *dir = opendir(cwd);
    if (dir == NULL) {
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strlen(entry->d_name) < file_suffix.size())
            continue;
        if (strncmp(entry->d_name + strlen(entry->d_name) - file_suffix.size(), file_suffix.c_str(),
                    file_suffix.size()) == 0) {
            (void)remove(entry->d_name);
        }
    }

    closedir(dir);
    return;
}

string RealPath(const string &path)
{
    // Nullptr is returned when the path does not exist or there is no permission
    // Return absolute path when path is accessible
    string res;
    char resolved_path[PATH_MAX] = {};
    const char *p = realpath(path.c_str(), resolved_path);
    if (p == nullptr) {
        res.assign("");
    } else {
        res.assign(resolved_path);
    }
    return res;
}

bool IsDir(const string &path)
{
    DIR *p = opendir(path.c_str());
    if (p != nullptr) {
        (void)closedir(p);
        return true;
    }
    return false;
}

void GetFilesWithSuffix(const string &path, const string &suffix, vector<string> &files)
{
    if (path.empty()) {
        return;
    }

    string real_path = RealPath(path);
    if (real_path.empty()) {
        return;
    }

    if (!IsDir(real_path)) {
        return;
    }

    struct dirent **entries = nullptr;
    const auto file_num = scandir(real_path.c_str(), &entries, nullptr, nullptr);
    if (entries == nullptr) {
        return;
    }

    if (file_num < 0) {
        free(entries);
        return;
    }

    for (int i = 0; i < file_num; ++i) {
        const dirent *const dir_ent = entries[i];
        const string name = string(dir_ent->d_name);
        if ((strcmp(name.c_str(), ".") == 0) || (strcmp(name.c_str(), "..") == 0)) {
            continue;
        }

        if (dir_ent->d_type == DT_DIR) {
            continue;
        }

        if (name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }

        const string full_name = path + "/" + name;
        files.push_back(full_name);
    }

    for (int i = 0; i < file_num; i++) {
        free(entries[i]);
    }
    free(entries);
}

unique_ptr<char[]> GetBinFromFile(const string &path, uint32_t &data_len)
{
    ifstream ifs(path, ios::in | ios::binary);
    if (!ifs.is_open()) {
        return nullptr;
    }

    ifs.seekg(0, ios_base::end);
    streampos file_size = ifs.tellg();
    ifs.seekg(0, ios_base::beg);

    auto p = unique_ptr<char[]>(new (nothrow) char[file_size]);
    if (p == nullptr) {
        ifs.close();
        return nullptr;
    }

    if (!ifs.read(reinterpret_cast<char *>(p.get()), file_size)) {
        p = nullptr;
    } else {
        data_len = static_cast<uint32_t>(file_size);
    }

    ifs.close();
    return p;
}
