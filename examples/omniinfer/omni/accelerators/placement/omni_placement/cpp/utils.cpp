// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "utils.h"

void CSVExporter::dump_expert_mapping_to_file(
    const std::string& filename,
    void* expert_mapping,
    const std::vector<int64_t>& expert_mapping_shape,
    int layer_id,
    bool append) {

    int64_t num_rows = expert_mapping_shape[0];
    int64_t num_cols = expert_mapping_shape[1];
    int32_t* expert_data = static_cast<int32_t*>(expert_mapping);

    // 分配主机内存
    std::vector<int32_t> host_data(num_rows * num_cols);

    // 从NPU复制到主机
    aclError ret = aclrtMemcpy(
        host_data.data(),
        host_data.size() * sizeof(int32_t),
        expert_data,
        host_data.size() * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST  // NPU设备到主机
    );

    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("Failed to copy data from NPU to host: " + std::to_string(ret));
    }

    // 打开文件，根据append参数决定是追加还是覆盖
    std::ofstream outfile;
    if (append) {
        outfile.open(filename, std::ios::app);
    } else {
        outfile.open(filename);
    }

    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // 自动生成标签，使用layer_id
    std::string label = "Layer " + std::to_string(layer_id);

    // 添加CSV格式的分隔注释行
    outfile << "# New Tensor Data," << label << "," << num_rows << "x" << num_cols << "\n";

    // 写入数据矩阵
    for (int64_t i = 0; i < num_rows; i++) {
        for (int64_t j = 0; j < num_cols; j++) {
            outfile << host_data[i * num_cols + j];
            if (j < num_cols - 1) outfile << ",";
        }
        outfile << "\n";
    }

    // 添加一个空行作为视觉分隔符
    outfile << "\n";

    outfile.close();

    std::cout << "Expert mapping data " << (append ? "appended" : "dumped")
        << " to file: " << filename
        << " for layer " << layer_id << std::endl;
}

void CSVExporter::dump_variables_to_csv(
    const std::string& filename,
    int E_non,
    int E_red_pos,
    int E_red,
    int frozen_map_value,
    bool append) {

    // 根据append参数决定打开模式
    std::ofstream outfile;
    if (append) {
        outfile.open(filename, std::ios::app);
    } else {
        outfile.open(filename);
    }

    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // 如果是新文件（非追加模式），写入表头
    if (!append) {
        outfile << "variable,value" << std::endl;
    }

    // 写入数据
    outfile << "E_non," << E_non << std::endl;
    outfile << "E_red_pos," << E_red_pos << std::endl;
    outfile << "E_red," << E_red << std::endl;
    outfile << "frozen_map_value," << frozen_map_value << std::endl;

    // 添加空行作为分隔
    outfile << std::endl;

    outfile.close();
    std::cout << "Variables " << (append ? "appended" : "dumped") << " to CSV file: " << filename << std::endl;
}

bool CSVExporter::append_table_to_csv(
    const std::vector<std::vector<int>>& freq_table,
    const std::string& filename,
    char delimiter,
    const std::string& matrix_separator,
    const std::string& header) {

    std::ofstream outfile(filename, std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for appending." << std::endl;
        return false;
    }

    // 添加可选的表头或分隔标识
    if (!header.empty()) {
        outfile << header << std::endl;
    }

    // 写入矩阵数据
    for (const auto& row : freq_table) {
        for (size_t i = 0; i < row.size(); ++i) {
            outfile << row[i];
            if (i < row.size() - 1) {
                outfile << delimiter;
            }
        }
        outfile << std::endl;
    }

    // 添加矩阵分隔符
    outfile << matrix_separator;
    outfile.close();
    return true;
}