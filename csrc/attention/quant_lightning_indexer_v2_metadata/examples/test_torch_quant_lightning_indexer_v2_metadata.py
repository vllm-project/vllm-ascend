# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=torch.tensor([3, 3, 3, 3, 3, 3, 3, 3], dtype=torch.int32).npu(),
    batch_size=8,
    max_seqlen_q=10,
    max_seqlen_k=10,
    num_heads_q=64,
    num_heads_k=1,
    head_dim=128,
    topk=2048,
    quant_mode=1,
    mask_mode=3,
    layout_q="BSND",
    layout_k="BSND",
    cmp_ratio=128,
)
