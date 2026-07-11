from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.23.0"):
    pytest.skip(
        "v2 model runner patches are not supported on vLLM 0.23.0",
        allow_module_level=True,
    )

from vllm_ascend.worker.v2 import attn_utils  # noqa: E402


def test_build_attn_metadata_resolves_causal_per_kv_cache_group(monkeypatch):
    group_causals = []

    class FakeCommonAttentionMetadata:
        def __init__(self, **kwargs):
            group_causals.append(kwargs["causal"])

    monkeypatch.setattr(
        attn_utils,
        "AscendCommonAttentionMetadata",
        FakeCommonAttentionMetadata,
    )

    attn_groups = []
    for index in range(2):
        builder = MagicMock()
        builder.build.return_value = f"metadata-{index}"
        group = SimpleNamespace(
            layer_names=[f"layer-{index}"],
            get_metadata_builder=lambda _index, builder=builder: builder,
        )
        attn_groups.append([group])

    metadata = attn_utils.build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1]),
        query_start_loc_cpu=torch.tensor([0, 1]),
        max_query_len=1,
        seq_lens=torch.tensor([1]),
        max_seq_len=1,
        block_tables=[torch.zeros((1, 1), dtype=torch.int32) for _ in range(2)],
        slot_mappings=torch.zeros((2, 1), dtype=torch.int64),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object(), object()]),
        seq_lens_np=np.array([1], dtype=np.int32),
        causal={0: False},
    )

    assert group_causals == [False, True]
    assert metadata == {"layer-0": "metadata-0", "layer-1": "metadata-1"}
