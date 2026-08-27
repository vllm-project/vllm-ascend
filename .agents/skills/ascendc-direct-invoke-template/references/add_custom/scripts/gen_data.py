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
# 测试数据生成脚本
# ============================================================================
#
# [MODIFY] 创建新算子时修改:
# 1. total_length: 总元素数（需与 add.asc 中的保持一致）
# 2. dtype: 数据类型（需与 add.asc 中的保持一致）
# 3. 输入数据: 根据你的算子生成对应的输入
# 4. golden 计算: 修改 golden.py 中的 compute_golden()
# ============================================================================

import numpy as np
import os

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# [MODIFY] 数据规格参数
total_length = 4096 * 8    # 总元素数，需与 add.asc 中 main 函数保持一致
dtype = np.float32         # 数据类型，需与 add.asc 中模板参数保持一致

# [MODIFY] 生成输入数据（根据你的算子调整）
x = np.random.randn(total_length).astype(dtype)
y = np.random.randn(total_length).astype(dtype)

x.tofile("input/input_x.bin")
y.tofile("input/input_y.bin")

# [MODIFY] 计算 golden（逻辑定义在 golden.py 中，修改那里即可）
golden = compute_golden(x, y)
golden.tofile("output/golden.bin")

print(f"Generated test data: {total_length} elements, dtype={dtype}")
print(f"  input/input_x.bin: {x.shape}, {x.dtype}")
print(f"  input/input_y.bin: {y.shape}, {y.dtype}")
print(f"  output/golden.bin: {golden.shape}, {golden.dtype}")
