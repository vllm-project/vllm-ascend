# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# PyTorch 通路测试脚本
# ============================================================================
#
# [MODIFY] 创建新算子时修改:
# 1. SO_NAME: .so 文件名（需与 CMakeLists.txt 中的 target 名一致）
# 2. OP_NAME: 算子名称（需与 register.cpp 中的注册名一致）
# 3. compute_golden(): 替换为你的算子的正确计算逻辑
# 4. 测试用例: 根据算子语义调整输入和预期输出
# ============================================================================

import sys
import os

import torch
import torch_npu

from golden import compute_golden

# [MODIFY] 算子配置
SO_NAME = "libadd_custom_ops.so"    # .so 文件名，需与 CMakeLists.txt target 一致
OP_NAME = "add_custom"              # 算子名称，需与 register.cpp 注册名一致
DTYPE = torch.float32               # 数据类型
ATOL = 1e-6                         # 绝对容差
RTOL = 1e-6                         # 相对容差


def run_test(name, x1, x2):
    """运行单个测试用例，返回 (name, passed, max_diff)"""
    op_fn = getattr(torch.ops.npu, OP_NAME)
    y = op_fn(x1.npu(), x2.npu())
    golden = compute_golden(x1, x2).npu()
    max_diff = torch.max(torch.abs(y - golden)).item()
    passed = torch.allclose(y.cpu(), golden.cpu(), atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    # 加载算子库
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    # [MODIFY] 测试用例（编号需与 PLAN.md §3.3 的 T*/P* 对齐）
    results = []

    # P1: 随机数据
    x1 = torch.randn(8, 16, dtype=DTYPE)
    x2 = torch.randn(8, 16, dtype=DTYPE)
    results.append(run_test("P1 random", x1, x2))

    # P2: 零值
    x1 = torch.zeros(8, 16, dtype=DTYPE)
    x2 = torch.zeros(8, 16, dtype=DTYPE)
    results.append(run_test("P2 zeros", x1, x2))

    # P3: 正负混合
    x1 = torch.randn(8, 16, dtype=DTYPE)
    x2 = -x1
    results.append(run_test("P3 pos_neg", x1, x2))

    # 汇总
    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*50}")
    for name, ok, diff in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} (Max diff={diff})")
    print(f"{'='*50}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
