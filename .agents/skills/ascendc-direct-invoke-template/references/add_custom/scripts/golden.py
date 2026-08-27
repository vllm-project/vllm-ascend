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
# Golden 计算（双通路共用）
# ============================================================================
#
# [MODIFY] 创建新算子时修改:
# 1. compute_golden(): 替换为你的算子的正确计算逻辑
#
# 本文件被 gen_data.py 和 test_torch.py 共同引用，
# 修改此处即同时更新两条通路的 golden 基准。
# ============================================================================

import numpy as np


# [MODIFY] 替换为你的算子的正确计算逻辑
def compute_golden(x1, x2):
    """计算算子的参考输出。

    Args:
        x1: numpy array 或 torch.Tensor
        x2: numpy array 或 torch.Tensor

    Returns:
        与输入同类型的参考输出
    """
    return x1 + x2
