/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <cerrno>
#include <cstddef>
#include <fcntl.h>
#include <fstream>
#include <limits.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "common_utils.h"

struct ExampleIoPaths {
    std::string baseDir;
    std::string inputDir;
    std::string outputDir;
};

inline ExampleIoPaths GetExampleIoPaths()
{
    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    std::string baseDir = ".";
    if (len > 0) {
        exePath[len] = '\0';
        baseDir = exePath;
        size_t lastSlash = baseDir.find_last_of('/');
        if (lastSlash != std::string::npos) {
            baseDir.resize(lastSlash);
        }
    }
    return {baseDir, baseDir + "/input", baseDir + "/output"};
}

inline bool ReadFile(const std::string& filePath, size_t& fileSize, void* buffer, size_t bufferSize)
{
    if (buffer == nullptr) {
        ERROR_LOG("Read file failed. buffer is nullptr");
        return false;
    }

    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf* buf = file.rdbuf();
    std::streampos fileEnd = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (fileEnd == std::streampos(-1)) {
        ERROR_LOG("Failed to query file size. path = %s", filePath.c_str());
        file.close();
        return false;
    }
    std::streamoff sizeOffset = fileEnd - std::streampos(0);
    if (sizeOffset < 0) {
        ERROR_LOG("File size is invalid. path = %s", filePath.c_str());
        file.close();
        return false;
    }
    size_t size = static_cast<size_t>(sizeOffset);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    if (size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        ERROR_LOG("file size exceeds supported stream size");
        file.close();
        return false;
    }
    if (buf->pubseekpos(0, std::ios::in) == std::streampos(-1)) {
        ERROR_LOG("Failed to reset file position. path = %s", filePath.c_str());
        file.close();
        return false;
    }
    std::streamsize readSize = buf->sgetn(static_cast<char*>(buffer), static_cast<std::streamsize>(size));
    if (readSize != static_cast<std::streamsize>(size)) {
        ERROR_LOG("Read file failed.");
        file.close();
        return false;
    }
    fileSize = size;
    file.close();
    return true;
}

inline bool ReadExactFile(const std::string& filePath, void* buffer, size_t expectedSize)
{
    size_t fileSize = expectedSize;
    if (!ReadFile(filePath, fileSize, buffer, expectedSize)) {
        return false;
    }
    if (fileSize != expectedSize) {
        ERROR_LOG("%s size does not match the expected tensor size", filePath.c_str());
        return false;
    }
    return true;
}

inline bool WriteFile(const std::string& filePath, const void* buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    const char* data = static_cast<const char*>(buffer);
    size_t writeSize = 0;
    while (writeSize < size) {
        ssize_t currentWrite = write(fd, data + writeSize, size - writeSize);
        if (currentWrite < 0 && errno == EINTR) {
            continue;
        }
        if (currentWrite <= 0) {
            (void)close(fd);
            ERROR_LOG("Write file failed. path = %s", filePath.c_str());
            return false;
        }
        writeSize += static_cast<size_t>(currentWrite);
    }
    (void)close(fd);

    return true;
}

#endif
