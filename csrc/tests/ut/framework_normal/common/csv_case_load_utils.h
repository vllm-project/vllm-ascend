/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSV_CASE_LOAD_UTILS_H
#define CSV_CASE_LOAD_UTILS_H

#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <filesystem>

using csv_map = std::unordered_map<std::string, std::string>;

template <typename Map>
inline typename Map::mapped_type ReadMap(const Map &m, const typename Map::key_type &key,
                                         const typename Map::mapped_type &defaultValue = typename Map::mapped_type{})
{
    auto it = m.find(key);
    return it != m.end() ? it->second : defaultValue;
}

inline std::vector<int64_t> GetShapeArr(const std::string &shapeArrStr)
{
    std::vector<int64_t> shapeArr;
    std::istringstream iss(shapeArrStr);
    int64_t num;
    while (iss >> num) {
        shapeArr.emplace_back(num);
    }
    return shapeArr;
}

inline std::string ReplaceFileExtension2Csv(const char *file)
{
    return std::filesystem::path(file).replace_extension("csv").string();
}

template <typename T> // T 需要支持 T(const csv_map&) 构造函数
std::vector<T> GetCasesFromCsv(const std::string &csvPath)
{
    std::vector<T> cases;
    std::ifstream csvFile(csvPath, std::ios::in);
    if (!csvFile.is_open()) {
        std::cout << "[ERROR] Cannot open case file \"" << csvPath << "\"!" << std::endl;
        return cases;
    }

    std::vector<std::string> keys;
    std::string line;
    std::getline(csvFile, line);
    std::istringstream stream(line);
    for (std::string data; std::getline(stream, data, ',');) {
        keys.emplace_back(data);
    }
    // lineNum = 1 是表头
    for (int lineNum = 2; std::getline(csvFile, line); ++lineNum) {
        if (line.empty()) {
            std::cout << "[ERROR] " << csvPath << ":" << lineNum << " Row data is empty!" << std::endl;
            return std::vector<T>();
        }
        std::vector<std::string> values;
        std::istringstream stream(line);
        for (std::string data; std::getline(stream, data, ',');) {
            values.emplace_back(data);
        }
        if (line.back() == ',') {
            values.emplace_back("");
        }
        if (keys.size() != values.size()) {
            std::cout << "[ERROR] " << csvPath << ":" << lineNum
                      << " Row data does not match CSV header columns. Row has " << values.size()
                      << " columns, but header has " << keys.size() << " columns." << std::endl;
            return std::vector<T>();
        }
        csv_map csvMap;
        for (size_t i = 0; i < keys.size(); ++i) {
            if (!values[i].empty()) {
                csvMap[keys[i]] = values[i];
            }
        }
        cases.emplace_back(csvMap);
    }
    return cases;
}

template <typename T>
inline std::string PrintCaseInfoString(const testing::TestParamInfo<T> &info)
{
    return info.param.case_name;
}

#endif // CSV_CASE_LOAD_UTILS_H
