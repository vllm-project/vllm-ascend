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
# 结果验证脚本
# ============================================================================
#
# [MODIFY] 创建新算子时修改:
# 1. dtype: 数据类型（需与 gen_data.py 和 add.asc 保持一致）
# 2. rtol/atol: 容差参数（根据算子精度要求调整）
# ============================================================================

import numpy as np
import sys

# [MODIFY] 验证参数
dtype = np.float32   # 数据类型，需与 gen_data.py 和 add.asc 保持一致
rtol = 1e-5          # 相对容差
atol = 1e-8          # 绝对容差

def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)
    
    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False
    
    if np.allclose(output, golden, rtol=rtol, atol=atol):
        print(f"Verification PASSED! Shape: {output.shape}")
        print(f"Max diff: {np.max(np.abs(output - golden))}")
        return True
    else:
        diff = np.abs(output - golden)
        print(f"Verification FAILED!")
        print(f"Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden))[0]
        print(f"Mismatch count: {len(mismatches)} / {len(golden)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)
    
    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
