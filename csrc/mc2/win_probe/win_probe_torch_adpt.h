/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef VLLM_ASCEND_WIN_PROBE_TORCH_ADPT_H
#define VLLM_ASCEND_WIN_PROBE_TORCH_ADPT_H

namespace vllm_ascend {
at::Tensor mc2_win_probe(const at::Tensor& context)
{
    TORCH_CHECK(context.scalar_type() == at::kInt, "context must be int32");
    at::Tensor addrs = at::empty({2}, context.options().dtype(at::kLong));
    EXEC_NPU_CMD(aclnnWinProbe, context, addrs);
    return addrs;
}
}  // namespace vllm_ascend
#endif
