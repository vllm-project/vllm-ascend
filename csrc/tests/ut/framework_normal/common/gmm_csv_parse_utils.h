/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_UT_FRAMEWORK_NORMAL_COMMON_GMM_CSV_PARSE_UTILS_H
#define TESTS_UT_FRAMEWORK_NORMAL_COMMON_GMM_CSV_PARSE_UTILS_H

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ops::ut {

inline void SplitStr2Vec(const std::string &input, const std::string &delimiter, std::vector<std::string> &output)
{
    const auto delimiterLen = delimiter.size();
    std::string::size_type currPos = 0;
    std::string::size_type nextPos = input.find(delimiter, currPos);
    while (nextPos != std::string::npos) {
        output.emplace_back(input.substr(currPos, nextPos - currPos));
        currPos = nextPos + delimiterLen;
        nextPos = input.find(delimiter, currPos);
    }
    if (currPos <= input.size()) {
        output.emplace_back(input.substr(currPos));
    }
}

inline std::string Trim(std::string value)
{
    auto isNotSpace = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), isNotSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), isNotSpace).base(), value.end());
    return value;
}

inline std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline bool ParseBool(const std::string &value)
{
    const std::string lower = ToLower(Trim(value));
    return lower == "true" || lower == "1" || lower == "yes";
}

inline int64_t ParseScalarToken(const std::string &value, const std::map<std::string, int64_t> &symbols = {})
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty()) {
        return 0;
    }

    const auto it = symbols.find(trimmed);
    if (it != symbols.end()) {
        return it->second;
    }

    const auto divPos = trimmed.find('/');
    if (divPos != std::string::npos) {
        const int64_t lhs = ParseScalarToken(trimmed.substr(0, divPos), symbols);
        const int64_t rhs = ParseScalarToken(trimmed.substr(divPos + 1), symbols);
        return rhs == 0 ? 0 : lhs / rhs;
    }

    return std::stoll(trimmed);
}

inline std::vector<int64_t> ParseDims(const std::string &value, const std::map<std::string, int64_t> &symbols = {},
                                      bool noneAsZero = false)
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return noneAsZero ? std::vector<int64_t>{0} : std::vector<int64_t>{};
    }
    if (trimmed == "ZERO") {
        return {0};
    }

    std::vector<std::string> dimTokens;
    SplitStr2Vec(trimmed, ":", dimTokens);

    std::vector<int64_t> dims;
    dims.reserve(dimTokens.size());
    for (const auto &token : dimTokens) {
        dims.emplace_back(ParseScalarToken(token, symbols));
    }
    return dims;
}

inline std::vector<int64_t> ParseI64List(const std::string &value, const std::string &sep = "|",
                                         const std::map<std::string, int64_t> &symbols = {})
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }

    std::vector<std::string> tokens;
    SplitStr2Vec(trimmed, sep, tokens);

    std::vector<int64_t> out;
    out.reserve(tokens.size());
    for (const auto &token : tokens) {
        out.emplace_back(ParseScalarToken(token, symbols));
    }
    return out;
}

inline std::vector<std::vector<int64_t>> ParseDimsList(const std::string &value)
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }

    std::vector<std::string> items;
    SplitStr2Vec(trimmed, ";", items);

    std::vector<std::vector<int64_t>> out;
    out.reserve(items.size());
    for (const auto &item : items) {
        out.emplace_back(ParseDims(item));
    }
    return out;
}

inline std::string NormalizePath(std::string path)
{
    std::replace(path.begin(), path.end(), '\\', '/');
    return path;
}

inline std::string JoinPath(const std::string &lhs, const std::string &rhs)
{
    if (lhs.empty()) {
        return rhs;
    }
    if (rhs.empty()) {
        return lhs;
    }
    if (lhs.back() == '/' || lhs.back() == '\\') {
        return lhs + rhs;
    }
    return lhs + "/" + rhs;
}

inline bool FileExists(const std::string &path)
{
    std::ifstream file(path, std::ios::in);
    return file.good();
}

inline std::string GetDirName(std::string path)
{
    path = NormalizePath(path);
    const auto pos = path.find_last_of('/');
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

inline std::string GetExeDirPath()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    const auto len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    return GetDirName(std::string(path, len));
#else
    char path[4096] = {0};
    const ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len <= 0) {
        return ".";
    }
    path[len] = '\0';
    return GetDirName(std::string(path));
#endif
}

inline std::string GetCurrentFileDir(const char *file)
{
    return GetDirName(file);
}

inline std::string GetCwd()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    const auto len = GetCurrentDirectoryA(MAX_PATH, path);
    return len == 0 ? std::string(".") : std::string(path, len);
#else
    char path[4096] = {0};
    if (getcwd(path, sizeof(path)) == nullptr) {
        return ".";
    }
    return std::string(path);
#endif
}

inline std::string ResolveCsvPath(const std::string &csv_name, const std::string &repo_relative_dir,
                                  const char *current_file)
{
    const std::string exe_dir = GetExeDirPath();
    const std::string cwd = GetCwd();
    const std::string current_file_dir = GetCurrentFileDir(current_file);

    std::vector<std::string> candidates;
    candidates.emplace_back(JoinPath(current_file_dir, csv_name));
    candidates.emplace_back(JoinPath(exe_dir, csv_name));
    candidates.emplace_back(JoinPath(cwd, csv_name));

    const char *code_path = std::getenv("CODE_PATH");
    if (code_path != nullptr) {
        candidates.emplace_back(JoinPath(code_path, JoinPath(repo_relative_dir, csv_name)));
    }

    candidates.emplace_back(JoinPath(".", JoinPath(repo_relative_dir, csv_name)));
    candidates.emplace_back(JoinPath(exe_dir, JoinPath("../../../../../", JoinPath(repo_relative_dir, csv_name))));
    candidates.emplace_back(JoinPath(cwd, JoinPath("../../../../../", JoinPath(repo_relative_dir, csv_name))));
    candidates.emplace_back(csv_name);
    candidates.emplace_back(JoinPath(".", csv_name));

    for (const auto &candidate : candidates) {
        if (FileExists(candidate)) {
            return candidate;
        }
    }
    return candidates.front();
}

inline std::string MakeSafeParamName(std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char ch) { return std::isalnum(ch) ? static_cast<char>(ch) : '_'; });
    return name;
}

inline std::string BuildCsvParseErrorMessage(const std::string &csv_path, size_t line_no, const std::string &case_name,
                                             const std::exception &error)
{
    std::ostringstream oss;
    oss << "Failed to parse CSV case. file=" << csv_path << ", line=" << line_no;
    if (!case_name.empty()) {
        oss << ", case=" << case_name;
    }
    oss << ", error=" << error.what();
    return oss.str();
}

} // namespace ops::ut

#endif // TESTS_UT_FRAMEWORK_NORMAL_COMMON_GMM_CSV_PARSE_UTILS_H
