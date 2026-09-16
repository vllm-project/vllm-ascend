/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <wait.h>
#include <unistd.h>
#include <linux/limits.h>

using namespace std;

#include "op_api_ut_common/inner/c_shell.h"

#define MAX_BUFFER_LEN 1024

int ExecuteShellCmd(const string &cmd, stringstream &ss)
{
    FILE *fp = NULL;
    char buffer[MAX_BUFFER_LEN] = {0};
    fp = popen(cmd.c_str(), "r");
    if (fp == nullptr) {
        ss << "popen fail.";
        return 1;
    }

    while (fgets(buffer, MAX_BUFFER_LEN, fp))
        ss << buffer;
    pid_t status = pclose(fp);
    if (status == -1) {
        ss << "system error.";
        return 1;
    }
    if (WIFEXITED(status)) {
        if (0 == WEXITSTATUS(status)) {
            return 0;
        } else {
            ss << "run shell command exit non-zero code: " << WEXITSTATUS(status);
            return 1;
        }
    } else {
        if (WIFSIGNALED(status)) {
            ss << "shell running thread is killed (signal " << WTERMSIG(status) << ")";
        } else if (WIFSTOPPED(status)) {
            ss << "shell running thread is stopped (signal " << WSTOPSIG(status) << ")";
        } else {
            ss << "Unexpected exit status!";
        }
        return 1;
    }
}

static string op_ut_src_path;
static string exe_path;

void SetOpUtSrcPath(const char *p)
{
    op_ut_src_path = p;
}

const string &GetOpUtSrcPath()
{
    return op_ut_src_path;
}

const string &GetExePath()
{
    if (exe_path.empty()) {
        char path[PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", path, sizeof(path));
        if (n > 0) {
            path[n] = '\0';
            exe_path.assign(path);
        } else {
            exe_path.assign(realpath(".", NULL));
        }
    }
    return exe_path;
}

int GetEnv(const string &name, string &value)
{
    if (name.length() == 0) {
        cout << "Invalid input parameter." << endl;
        return 1;
    }

    const char *p = getenv(name.c_str());
    if (p == nullptr) {
        cout << "env parameter " << name << " is not set." << endl;
        return 0;
    }

    value.assign(p);
    return 0;
}
