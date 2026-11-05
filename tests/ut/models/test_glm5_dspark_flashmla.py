# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM DSpark context cache views and writer routing."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.models import glm5_dspark


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("use_prolog", [False, True])
def test_context_writer_preserves_fused_cache_backing(fused, use_prolog):
    backing = torch.zeros(2, 128, 1, 576)
    cache = backing if fused else (backing[..., :512].contiguous(), backing[..., 512:].contiguous())
    writer = MagicMock()
    prolog = MagicMock()
    projection = MagicMock(side_effect=lambda states: (torch.zeros(states.shape[0], 580), None))
    attn = SimpleNamespace(
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.ones(8, 64)),
        fused_qkv_a_proj=projection,
        q_lora_rank=4,
        kv_lora_rank=512,
        attn=SimpleNamespace(
            kv_cache=cache,
            impl=SimpleNamespace(
                exec_kv_prefill=writer,
                exec_context_kv_prolog_v3=prolog,
                _cache_requires_prolog_v3_writer=lambda _: use_prolog,
            ),
        ),
    )
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
    slots = torch.tensor([0, 128], dtype=torch.int32)
    glm5_dspark.Glm5DSparkModel.precompute_and_store_context_kv(model, torch.zeros(2, 8), torch.tensor([0, 1]), slots)
    if use_prolog:
        prolog.assert_called_once()
        writer.assert_not_called()
        projection.assert_not_called()
        components = prolog.call_args.args[3]
    else:
        writer.assert_called_once()
        prolog.assert_not_called()
        projection.assert_called_once()
        components = writer.call_args.args[3]
    assert isinstance(components, tuple)
    assert components[0].shape == (2, 128, 1, 512)
    assert components[1].shape == (2, 128, 1, 64)
    if fused:
        assert components[0].untyped_storage() is backing.untyped_storage()
        assert components[1].untyped_storage() is backing.untyped_storage()
        assert components[1].storage_offset() - components[0].storage_offset() == 512
    else:
        assert components is cache
    assert attn.attn.kv_cache is cache
