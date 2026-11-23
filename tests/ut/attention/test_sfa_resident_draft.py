# SPDX-License-Identifier: Apache-2.0
"""Resident draft KV must not enter the target's Host-offload path."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ascend_config import draft_config_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_kv_offload import (
    AscendSFAKVOffloadImpl,
    AscendSFAKVOffloadMetadataBuilder,
    AscendSFAOffloadMetadata,
)
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder


@pytest.mark.parametrize("host_backend", ["memfabric", "mooncake"])
@pytest.mark.parametrize("offload_enabled", [False, True])
@pytest.mark.parametrize("is_draft", [False, True])
def test_draft_implementation_uses_hbm_without_disabling_target_offload(host_backend, offload_enabled, is_draft):
    config = SimpleNamespace(attention_config=SimpleNamespace(hisparse_config=None))
    ascend = SimpleNamespace(
        sparse_kv_offload_config=SimpleNamespace(enabled=offload_enabled, host_backend=host_backend)
    )
    with (
        patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.attention.sfa_v1.get_ascend_config", return_value=ascend),
        patch("vllm_ascend.attention.sfa_v1.enable_sfa_dcp_replicated_indexer", return_value=False),
        draft_config_context() if is_draft else nullcontext(),
    ):
        expected = AscendSFAKVOffloadImpl if offload_enabled and not is_draft else AscendSFAImpl
        assert AscendSFABackend.get_impl_cls() is expected

    # Leaving draft reconstruction must restore target selection.
    with (
        patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.attention.sfa_v1.get_ascend_config", return_value=ascend),
        patch("vllm_ascend.attention.sfa_v1.enable_sfa_dcp_replicated_indexer", return_value=False),
    ):
        expected = AscendSFAKVOffloadImpl if offload_enabled else AscendSFAImpl
        assert AscendSFABackend.get_impl_cls() is expected


@pytest.mark.parametrize("store_on_host", [False, True])
@pytest.mark.parametrize("fused_copy", [False, True])
@pytest.mark.parametrize("dummy", [False, True])
@pytest.mark.parametrize("draft_step", [None, 0, 2])
def test_metadata_honors_placement_for_real_dummy_and_draft_steps(store_on_host, fused_copy, dummy, draft_step):
    cfg = SimpleNamespace(use_fused_copy_sfa=fused_copy)
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True, is_kv_producer=False)
    )
    metadata = SimpleNamespace(attn_state=AscendAttentionState.ChunkedPrefill)
    common = SimpleNamespace(offload_dummy=dummy)
    populated = object()
    with (
        patch.object(AscendSFAMetadataBuilder, "__init__", return_value=None) as init,
        patch.object(AscendSFAKVOffloadMetadataBuilder, "_init_copy_sfa_metadata_buffers") as allocate,
        patch.object(AscendSFAMetadataBuilder, "build", return_value=metadata),
        patch.object(AscendSFAMetadataBuilder, "build_for_drafting", return_value=metadata),
        patch("vllm_ascend.attention.sfa_kv_offload._validate_fused_copy_sfa_config") as validate,
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_ascend_config",
            return_value=SimpleNamespace(sparse_kv_offload_config=cfg),
        ),
    ):
        builder = AscendSFAKVOffloadMetadataBuilder(
            kv_cache_spec=SimpleNamespace(store_on_host=store_on_host, is_draft=not store_on_host),
            layer_names=["model.layers.78.self_attn.attn"],
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )
        builder._populate_offload_metadata = MagicMock(return_value=populated)
        result = (
            builder.build(0, common)
            if draft_step is None
            else builder.build_for_drafting(common, draft_step)
        )

    expected_cls = AscendSFAOffloadMetadata if store_on_host else AscendSFAMetadata
    assert init.call_args.args[4] is expected_cls
    assert builder.use_fused_copy_sfa is (store_on_host and fused_copy)
    if store_on_host:
        validate.assert_called_once_with(vllm_config, cfg)
        builder._populate_offload_metadata.assert_called_once()
        assert result is populated
    else:
        validate.assert_not_called()
        builder._populate_offload_metadata.assert_not_called()
        assert result is metadata
        assert metadata.attn_state is AscendAttentionState.ChunkedPrefill
        assert not hasattr(result, "fused_copy_sfa_enabled")
    assert allocate.call_count == int(store_on_host and fused_copy)


@pytest.mark.parametrize("host_backend", ["memfabric", "mooncake"])
@pytest.mark.parametrize("defer_wait", [False, True])
def test_resident_mtp_exec_kv_never_queries_host_manager(host_backend, defer_wait):
    config = SimpleNamespace(attention_config=SimpleNamespace(hisparse_config=None))
    ascend = SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=True, host_backend=host_backend))
    with (
        patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.attention.sfa_v1.get_ascend_config", return_value=ascend),
        patch("vllm_ascend.attention.sfa_v1.enable_sfa_dcp_replicated_indexer", return_value=False),
        draft_config_context(),
    ):
        impl_cls = AscendSFABackend.get_impl_cls()
    assert impl_cls is AscendSFAImpl
    impl = impl_cls.__new__(impl_cls)
    impl.pcp_enabled = False
    impl.qk_rope_head_dim = 2
    impl.kv_lora_rank = 4
    impl.num_kv_heads = 1
    impl.enable_sparse_sfa_c8 = False
    impl.enable_sparse_sfa_turboquant = False
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    kv_cache = (torch.empty(1, 2, 1, 4), torch.empty(1, 2, 1, 2))
    kv_no_split = torch.ones(2, 6)
    cos, sin = torch.ones(2, 2), torch.zeros(2, 2)
    slots = torch.arange(2, dtype=torch.int64)
    with (
        patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_kv_rmsnorm_rope_cache", create=True) as write_kv,
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
            side_effect=AssertionError("resident MTP queried the Host offload manager"),
        ),
    ):
        result = impl.exec_kv(kv_no_split, cos, sin, kv_cache, slots, SimpleNamespace(), defer_wait=defer_wait)
        if defer_wait:
            write_kv.assert_not_called()
            result = result()
        assert result == (None, None)
    write_kv.assert_called_once()
    assert write_kv.call_args.args[5] is kv_cache[1]
    assert write_kv.call_args.args[6] is kv_cache[0]
