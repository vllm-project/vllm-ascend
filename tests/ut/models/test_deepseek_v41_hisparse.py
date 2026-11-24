# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.models.deepseek_v41.hisparse import (
    finish_source_write,
    offloaded_sources,
    prepare_cache_views,
    resident_write_slots,
)


def _vllm_config_with_sources(sources=None):
    extra = {} if sources is None else {"dsv41_sources": sources}
    return SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra))


def test_offloaded_sources_default_and_validation():
    assert offloaded_sources(_vllm_config_with_sources()) == {20}
    assert offloaded_sources(_vllm_config_with_sources([2, 20])) == {2, 20}
    with pytest.raises(ValueError, match="dsv41_sources"):
        offloaded_sources(_vllm_config_with_sources([99]))
    with pytest.raises(ValueError, match="dsv41_sources"):
        offloaded_sources(_vllm_config_with_sources([]))


def test_resident_write_slots_c1_and_c2():
    slots = torch.tensor([0, 1, 2, 3, -1, 5], dtype=torch.int32)
    positions = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)

    c1_meta = SimpleNamespace(
        num_input_tokens=6,
        compress_ratio=1,
        flat_slot_mapping=torch.tensor([0, 1, -1, 3, 4, 5], dtype=torch.int64),
        query_start_loc=query_start_loc,
        num_actual_reqs=1,
        num_actual_tokens=4,
        positions=positions,
    )
    assert resident_write_slots(slots, c1_meta).tolist() == [0, 1, -1, 3, -1, -1]

    c2_meta = SimpleNamespace(
        num_input_tokens=6,
        compress_ratio=2,
        flat_slot_mapping=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64),
        query_start_loc=query_start_loc,
        num_actual_reqs=1,
        num_actual_tokens=4,
        positions=positions,
    )
    assert resident_write_slots(slots, c2_meta).tolist() == [-1, 0, -1, 1, -1, -1]


def test_finish_source_write_guards():
    handle = MagicMock()
    handle.runtime.request_state_indices = torch.tensor([0], dtype=torch.int32)

    cap_meta = SimpleNamespace(
        hisparse_for_capture=True,
        flat_slot_mapping=torch.tensor([0], dtype=torch.int64),
        req_id_per_token=torch.tensor([0], dtype=torch.int32),
    )
    finish_source_write(handle, cap_meta)
    handle.finish_kv_update.assert_not_called()

    real_meta = SimpleNamespace(
        hisparse_for_capture=False,
        flat_slot_mapping=torch.tensor([0], dtype=torch.int64),
        req_id_per_token=torch.tensor([0], dtype=torch.int32),
    )
    with patch("vllm_ascend.models.deepseek_v41.hisparse.has_kv_transfer_group", return_value=False):
        finish_source_write(handle, real_meta)
    handle.finish_kv_update.assert_not_called()

    no_meta_group = MagicMock()
    no_meta_group.has_connector_metadata.return_value = False
    with (
        patch("vllm_ascend.models.deepseek_v41.hisparse.has_kv_transfer_group", return_value=True),
        patch("vllm_ascend.models.deepseek_v41.hisparse.get_kv_transfer_group", return_value=no_meta_group),
    ):
        finish_source_write(handle, real_meta)
    handle.finish_kv_update.assert_not_called()

    active_group = MagicMock()
    active_group.has_connector_metadata.return_value = True
    with (
        patch("vllm_ascend.models.deepseek_v41.hisparse.has_kv_transfer_group", return_value=True),
        patch("vllm_ascend.models.deepseek_v41.hisparse.get_kv_transfer_group", return_value=active_group),
    ):
        finish_source_write(handle, real_meta)
    handle.runtime.invalidate_written_slots.assert_called_once()
    handle.finish_kv_update.assert_called_once()


def test_prepare_cache_views_pa_bbnd_and_mxfp4():
    layer_name = "model.layers.20.self_attn.long_kv_cache"
    kv_spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=320,
        dtype=torch.uint8,
        model_version="deepseek_v41",
        tokens_per_state=2,
    )
    kv_raw = torch.zeros((4, 1, 64, 320), dtype=torch.uint8)
    kv_caches = {layer_name: kv_raw}
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=kv_spec, layer_names=[layer_name])],
    )
    prepare_cache_views(kv_cache_config, kv_caches)
    assert kv_caches[layer_name].shape == (4, 64, 1, 320)

    idx_name = "model.layers.20.self_attn.indexer.k_cache"
    idx_spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        model_version="deepseek_v41",
        scale_dtype=torch.uint8,
        scale_dim=4,
        tokens_per_state=1,
    )
    idx_raw = torch.zeros((4, 128, 132), dtype=torch.uint8)
    kv_caches = {idx_name: idx_raw}
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=idx_spec, layer_names=[idx_name])],
    )
    prepare_cache_views(kv_cache_config, kv_caches)
    key, scale = kv_caches[idx_name]
    assert key.shape == (4, 128, 1, 128)
    assert scale.shape == (4, 128, 1, 4)
