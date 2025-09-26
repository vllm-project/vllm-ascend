// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include "acl/acl.h"

class CSVExporter {
public:
    CSVExporter() = default;
    ~CSVExporter() = default;

    /**
     * 将专家映射数据导出到CSV文件
     *
     * @param filename 文件名
     * @param expert_mapping 专家映射数据指针
     * @param expert_mapping_shape 数据形状
     * @param layer_id 层ID
     * @param append 是否追加到文件
     */
    void dump_expert_mapping_to_file(
        const std::string& filename,
        void* expert_mapping,
        const std::vector<int64_t>& expert_mapping_shape,
        int layer_id,
        bool append = false
    );

    /**
     * 将变量导出到CSV文件
     *
     * @param filename 文件名
     * @param E_non 新冗余专家编号
     * @param E_red_pos 冗余专家位置
     * @param E_red 原冗余专家编号
     * @param frozen_map_value 冻结映射值
     * @param append 是否追加到文件
     */
    void dump_variables_to_csv(
        const std::string& filename,
        int E_non,
        int E_red_pos,
        int E_red,
        int frozen_map_value,
        bool append = false
    );

    /**
     * 将表格数据追加到CSV文件
     *
     * @param freq_table 频率表数据
     * @param filename 文件名
     * @param delimiter 分隔符
     * @param matrix_separator 矩阵分隔符
     * @param header 表头
     * @return 是否成功
     */
    bool append_table_to_csv(
        const std::vector<std::vector<int>>& freq_table,
        const std::string& filename,
        char delimiter = ',',
        const std::string& matrix_separator = "\n",
        const std::string& header = ""
    );
};

// 其他工具函数或类可以添加在这里

#endif // UTILS_H