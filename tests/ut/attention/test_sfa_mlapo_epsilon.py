# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


@pytest.mark.parametrize("q_epsilon,kv_epsilon", [(1e-5, 1e-4), (1e-6, 1e-6)])
def test_mlapo_preserves_each_layernorm_epsilon(q_epsilon, kv_epsilon):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.q_a_layernorm = SimpleNamespace(variance_epsilon=q_epsilon)
    impl.kv_a_layernorm = SimpleNamespace(variance_epsilon=kv_epsilon)
    impl.W_UK_T = torch.empty(2, 4, 8)
    impl.q_lora_rank = 16
    for name in (
        "wd_qkv",
        "deq_scale_qkv",
        "gamma1",
        "beta1",
        "wu_q",
        "qb_deq_scl",
        "gamma2",
        "quant_scale0",
        "quant_offset0",
        "quant_bias_qkv",
        "quant_scale1",
        "quant_offset1",
        "qb_qt_bias",
        "ctkv_scale",
        "q_nope_scale",
    ):
        setattr(impl, name, torch.empty(1))
    hidden = torch.empty(1, 32)
    cache = (torch.empty(1, 128, 8), torch.empty(1, 128, 4))
    rope = torch.empty(1, 4)
    slots = torch.zeros(1, dtype=torch.int32)

    with patch("vllm_ascend.attention.sfa_v1.torch.ops._C_ascend.mla_preprocess", create=True) as operator:
        result = impl._sfa_preprocess_mlapo(hidden, cache, rope, rope, slots, num_input_tokens=1)

    assert operator.call_count == 1
    assert operator.call_args.kwargs["q_epsilon"] == q_epsilon
    assert operator.call_args.kwargs["kv_epsilon"] == kv_epsilon
    assert result[0] is hidden
    inner = result[3]
    assert isinstance(inner, torch.Tensor)
    assert inner.shape == (1, impl.q_lora_rank)


@pytest.mark.parametrize("missing", ["q_a_layernorm", "kv_a_layernorm"])
def test_mlapo_requires_both_layernorms(missing):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.q_a_layernorm = SimpleNamespace(variance_epsilon=1e-5)
    impl.kv_a_layernorm = SimpleNamespace(variance_epsilon=1e-5)
    setattr(impl, missing, None)
    tensor = torch.empty(1)
    with pytest.raises(AssertionError, match="MLAPO requires Q and KV RMSNorm layers"):
        impl._sfa_preprocess_mlapo(tensor, (tensor, tensor), tensor, tensor, tensor)
