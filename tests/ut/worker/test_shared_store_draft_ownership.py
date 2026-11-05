# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec
from vllm_ascend.worker.v2 import attn_utils


@pytest.mark.parametrize("kind", ["attention", "state", "ring"])
@pytest.mark.parametrize("draft", [False, True])
def test_actual_ascend_spec_builder_preserves_module_ownership(monkeypatch, kind, draft):
    if kind == "state":
        source = MambaSpec(
            block_size=640, shapes=((4, 8), (2, 4, 4)), dtypes=(torch.bfloat16, torch.float32), mamba_cache_mode="align"
        )
    elif kind == "ring":
        source = AscendIndexerKPoolTailSpec(
            block_size=5, num_kv_heads=1, head_size=256, dtype=torch.float32, sliding_window=4, compress_ratio=4
        )
    else:
        source = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    if not hasattr(source, "is_draft"):
        pytest.skip("This vLLM runtime has no draft spec metadata API")
    layer = SimpleNamespace(
        is_draft_layer=draft,
        get_kv_cache_spec=lambda _: source,
        get_attn_backend=lambda: SimpleNamespace(),
        align_kv_cache_with_mamba=True,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=False, dtype=torch.bfloat16, get_total_num_kv_heads=lambda: 1),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        attention_config=SimpleNamespace(indexer_kv_dtype="auto"),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=128),
    )
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args, **_kwargs: {"owner": layer})
    monkeypatch.setattr(attn_utils, "enable_sfa_dcp_replicated_indexer", lambda _: False)
    actual = attn_utils.get_kv_cache_spec(config)["owner"]
    assert actual.is_draft is draft
    assert not source.is_draft  # Do not mutate the model's reusable source spec.
    assert actual.page_size_bytes == source.page_size_bytes
