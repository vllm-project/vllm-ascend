# SPDX-License-Identifier: Apache-2.0
"""Behavioral regressions for Ascend fused draft metadata updates."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADCPMetadataBuilder,
    AscendSFADSACPMetadataBuilder,
    AscendSFADSADCPMetadataBuilder,
    AscendSFAPCPDCPMetadataBuilder,
)
from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder


@pytest.mark.parametrize(
    "builder_cls",
    [
        AscendSFAMetadataBuilder,
        AscendSFADCPMetadataBuilder,
        AscendSFADSACPMetadataBuilder,
        AscendSFADSADCPMetadataBuilder,
        AscendSFAPCPDCPMetadataBuilder,
    ],
)
def test_sfa_cp_does_not_inherit_native_fused_capability(builder_cls):
    # Exercise base initialization on each real subclass: its derived layout
    # must opt in separately instead of inheriting the native RoPE hook.
    builder = builder_cls.__new__(builder_cls)
    builder.device = torch.device("cpu")
    config = SimpleNamespace(
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={"layer": SimpleNamespace(qk_rope_head_dim=64)}),
    )
    with (
        patch("vllm_ascend.attention.sfa_v1.MLACommonMetadataBuilder.__init__", return_value=None),
        patch("vllm_ascend.attention.sfa_v1.select_common_block_size", return_value=128),
        patch("vllm_ascend.attention.sfa_v1.AttentionMaskBuilder"),
    ):
        AscendSFAMetadataBuilder.__init__(builder, SimpleNamespace(block_size=128), ["layer"], config, builder.device)
    assert builder.supports_draft_decode_metadata_update is (builder_cls is AscendSFAMetadataBuilder)


@pytest.mark.parametrize("builder_cls", [AscendSFAMetadataBuilder, AscendSFAIndexerMetadataBuilder])
@pytest.mark.parametrize("c8", [False, True])
def test_sfa_fused_rope_preserves_buffers(builder_cls, c8):
    builder = builder_cls.__new__(builder_cls)
    builder.nope = False
    positions = torch.tensor([127, 255], dtype=torch.int64)
    metadata = SimpleNamespace(
        positions=positions,
        num_input_tokens=2,
        cos=torch.zeros(2, 1, 1, 2),
        sin=torch.zeros(2, 1, 1, 2),
        slot_mapping=torch.tensor([127, 255], dtype=torch.int32),
        group_len=torch.zeros(2, dtype=torch.int32),
        group_key_idx=torch.zeros(2, dtype=torch.int32),
        group_key_cache_idx=torch.zeros(2, dtype=torch.int32),
        block_size=128,
    )
    cos, sin = metadata.cos, metadata.sin

    def lookup(p, use_cache):
        assert use_cache
        values = p.float().view(2, 1, 1, 1).expand(2, 1, 1, 2)
        return values, -values

    with (
        patch(f"{builder_cls.__module__}.get_cos_and_sin_mla", side_effect=lookup),
        patch(f"{builder_cls.__module__}.get_ascend_config", return_value=SimpleNamespace(c8_reshape_optim_enabled=c8)),
        patch("torch.ops._C_ascend.store_kv_block_metadata", create=True) as store,
    ):
        for _ in range(3):
            positions.add_(1)
            metadata.slot_mapping.add_(1)
            builder.update_draft_decode_metadata(metadata)
            assert metadata.cos is cos
            assert metadata.sin is sin
            torch.testing.assert_close(cos[:, 0, 0, 0], positions.float())
            torch.testing.assert_close(sin[:, 0, 0, 0], -positions.float())
        assert store.call_count == (3 if c8 else 0)
        if c8:
            assert store.call_args.args[0] is metadata.slot_mapping
