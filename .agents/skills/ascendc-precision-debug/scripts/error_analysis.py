#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
精度调试 - 误差分析工具

分析算子输出与期望值之间的误差，提供详细的误差统计报告。
"""

import numpy as np
import sys


def analyze_error(pred_file, truth_file, rtol=1e-5, atol=1e-6):
    """
    分析预测值与真值之间的误差

    Args:
        pred_file: 预测结果文件路径 (.npy)
        truth_file: 真值文件路径 (.npy)
        rtol: 相对误差容差
        atol: 绝对误差容差

    Returns:
        bool: 是否通过验证 (通过率 >= 99%)
    """
    try:
        pred = np.load(pred_file)
        truth = np.load(truth_file)
    except Exception as e:
        print(f"错误: 无法加载文件 - {e}")
        return False

    # 检查形状
    if pred.shape != truth.shape:
        print(f"错误: 形状不匹配 - pred={pred.shape}, truth={truth.shape}")
        return False

    # 计算误差
    abs_error = np.abs(pred - truth)
    rel_error = abs_error / (np.abs(truth) + atol)

    print("=" * 60)
    print("误差分析报告")
    print("=" * 60)
    print(f"预测文件: {pred_file}")
    print(f"真值文件: {truth_file}")
    print(f"数据形状: {pred.shape}")
    print()

    # 绝对误差统计
    print("【绝对误差统计】")
    print(f"  最大值: {abs_error.max():.6e}")
    print(f"  平均值: {abs_error.mean():.6e}")
    print(f"  中位数: {np.median(abs_error):.6e}")
    print(f"  标准差: {abs_error.std():.6e}")
    print()

    # 相对误差统计
    print("【相对误差统计】")
    print(f"  最大值: {rel_error.max():.6e}")
    print(f"  平均值: {rel_error.mean():.6e}")
    print(f"  中位数: {np.median(rel_error):.6e}")
    print(f"  95分位: {np.percentile(rel_error, 95):.6e}")
    print(f"  99分位: {np.percentile(rel_error, 99):.6e}")
    print()

    # 通过率
    pass_mask = np.logical_or(abs_error < atol, rel_error < rtol)
    pass_count = pass_mask.sum()
    total_count = pass_mask.size
    pass_rate = pass_count / total_count * 100

    print(f"【通过率】")
    print(f"  通过: {pass_count}/{total_count}")
    print(f"  通过率: {pass_rate:.2f}%")
    print(f"  容差: rtol={rtol:.0e}, atol={atol:.0e}")
    print()

    # 误差分布
    print("【误差分布】")
    for threshold in [1e-3, 1e-4, 1e-5, 1e-6]:
        count = (abs_error > threshold).sum()
        rate = count / abs_error.size * 100
        print(f"  误差 > {threshold:.0e}: {count:6d} ({rate:5.2f}%)")
    print()

    # 最差样本
    worst_idx = abs_error.argmax()
    worst_pos = np.unravel_index(worst_idx, pred.shape)
    print(f"【最差样本】")
    print(f"  位置: {worst_pos}")
    print(f"  预测值: {pred[worst_pos]:.6f}")
    print(f"  真值: {truth[worst_pos]:.6f}")
    print(f"  绝对误差: {abs_error[worst_pos]:.6e}")
    print(f"  相对误差: {rel_error[worst_pos]:.6e}")
    print()

    # 判断结果
    if pass_rate >= 99.0:
        print("✓ 验证: PASS")
        return True
    else:
        print("✗ 验证: FAIL")

        # 打印失败样本（前10个）
        fail_indices = np.where(~pass_mask)
        fail_count = min(10, len(fail_indices[0]))
        if fail_count > 0:
            print()
            print("【失败样本（前10个）】")
            for i in range(fail_count):
                idx = tuple(dim[i] for dim in fail_indices)
                print(f"  @{idx}:")
                print(f"    预测={pred[idx]:.6f}, 期望={truth[idx]:.6f}, "
                      f"abs_err={abs_error[idx]:.2e}, rel_err={rel_error[idx]:.2e}")
        return False


def main():
    if len(sys.argv) < 3:
        print("用法: python3 error_analysis.py <output.npy> <expected.npy> [rtol] [atol]")
        print()
        print("示例:")
        print("  python3 error_analysis.py output.npy expected.npy")
        print("  python3 error_analysis.py output.npy expected.npy 1e-3 1e-4  # FP16")
        print("  python3 error_analysis.py output.npy expected.npy 1e-5 1e-6  # FP32")
        sys.exit(1)

    pred_file = sys.argv[1]
    truth_file = sys.argv[2]
    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    atol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-6

    success = analyze_error(pred_file, truth_file, rtol, atol)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
