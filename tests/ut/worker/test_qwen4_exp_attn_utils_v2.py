#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Six-region QSA hybrid cache consumption on the V2 runner (Qwen3.8-Flash-Next).

Covers the three V2-side additions around the Qwen4Exp integration:

- ``get_kv_cache_spec`` pads descriptor-native attention owners up to the
  Mamba page while Mamba keeps its own page size;
- ``_allocate_kv_cache`` maps every non-hidden owner onto one contiguous
  six-region backing, and ``_reshape_kv_cache_v2`` rebuilds the role views
  from that deterministic layout;
- ``build_attn_metadata`` clamps the QSA builder's CPU query boundaries to
  the live requests under FULL graph padding.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.core.six_region_kv_cache_layout import (
    HIDDEN,
    QSA_RAW,
    build_six_region_kv_cache_layout,
)
from vllm_ascend.models.qwen4_exp.qsa import AscendQSABackend
from vllm_ascend.worker.v2 import attn_utils

# The production six-region geometry shared with the V1 layout tests.
SOURCES = (
    "model.layers.11.self_attn",
    "model.layers.47.self_attn",
    "model.mtp.layers.48.self_attn",
)
MAIN_LAYER = f"{SOURCES[0]}.attn"
RAW_LAYER = f"{SOURCES[0]}.indexer.raw_key_cache"
COMPRESSED_LAYER = f"{SOURCES[0]}.indexer.compressed_key_cache"
GDN_LAYERS = ("model.layers.0.linear_attn", "model.layers.1.linear_attn")
PLE_LAYER = "model.ple"
HIDDEN_LAYER = "model.cache_only_layers.0"


def _six_region_groups() -> list[KVCacheGroupSpec]:
    main = FullAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )
    raw = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=32,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    compressed = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    gdn = MambaSpec(
        block_size=256,
        shapes=((64, 4), (128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
    )
    ple = MambaSpec(
        block_size=256,
        shapes=((128, 5),),
        dtypes=(torch.bfloat16,),
        tp_replicated=True,
    )
    hidden = HiddenStateCacheSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )
    raw_sources = [SOURCES[2], SOURCES[0], SOURCES[1]]
    compressed_sources = [SOURCES[1], SOURCES[2], SOURCES[0]]
    return [
        KVCacheGroupSpec([f"{source}.attn" for source in SOURCES], main),
        KVCacheGroupSpec([f"{source}.indexer.raw_key_cache" for source in raw_sources], raw),
        KVCacheGroupSpec(
            [f"{source}.indexer.compressed_key_cache" for source in compressed_sources],
            compressed,
        ),
        KVCacheGroupSpec(list(GDN_LAYERS), gdn),
        KVCacheGroupSpec([PLE_LAYER], ple),
        KVCacheGroupSpec([HIDDEN_LAYER], hidden),
    ]


def _six_region_config(num_blocks: int = 3) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=_six_region_groups(),
    )


def _v2_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={},
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen4_exp_text")),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        quant_config=None,
        attention_config=SimpleNamespace(indexer_kv_dtype="auto"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )


@pytest.fixture(autouse=True)
def _cpu_v2_runner(monkeypatch):
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: _v2_vllm_config())
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(attn_utils, "can_share_full_and_sliding_kv_cache", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda _cfg: False)


def _non_hidden_attention_groups(kv_cache_config: KVCacheConfig) -> list[AttentionGroup]:
    """Attention groups for every owner that shares the six-region backing.

    The hidden-state owner is allocated outside the shared backing and does
    not flow through this reshape path.
    """
    groups = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        if group_id == len(kv_cache_config.kv_cache_groups) - 1:
            break  # hidden group
        groups.append(
            AttentionGroup(
                backend=AscendQSABackend if group_id < 3 else AscendAttentionBackend,
                layer_names=list(group.layer_names),
                kv_cache_spec=group.kv_cache_spec,
                kv_cache_group_id=group_id,
            )
        )
    return groups


def test_six_region_allocation_maps_owners_to_one_backing():
    num_blocks = 3
    kv_cache_config = _six_region_config(num_blocks)
    layout = build_six_region_kv_cache_layout(kv_cache_config.kv_cache_groups, num_blocks)
    assert layout is not None

    raw = attn_utils._allocate_kv_cache(kv_cache_config, shared_layers={}, device=torch.device("cpu"))

    non_hidden = [owner for owner in layout.owners if owner.role != HIDDEN]
    hidden = [owner for owner in layout.owners if owner.role == HIDDEN]
    assert non_hidden and hidden
    backing = raw[non_hidden[0].layer_name]
    assert isinstance(backing, torch.Tensor)
    assert backing.dtype == torch.int8
    for owner in non_hidden:
        assert raw[owner.layer_name] is backing
    for owner in hidden:
        assert owner.layer_name not in raw
    assert backing.numel() == layout.slot_count * layout.slot_backing_size


def test_six_region_reshape_builds_role_views():
    num_blocks = 3
    kv_cache_config = _six_region_config(num_blocks)
    raw = attn_utils._allocate_kv_cache(kv_cache_config, shared_layers={}, device=torch.device("cpu"))
    layout = build_six_region_kv_cache_layout(kv_cache_config.kv_cache_groups, num_blocks)
    backing = raw[MAIN_LAYER]

    kv_caches = attn_utils._reshape_kv_cache_v2(
        attn_groups=_non_hidden_attention_groups(kv_cache_config),
        kv_cache_raw_tensors=raw,
        cache_dtype="auto",
        kernel_block_sizes=[group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups],
        shared_kv_cache_layers={},
        kv_cache_config=kv_cache_config,
    )

    expected_layers = {owner.layer_name for owner in layout.owners if owner.role != HIDDEN}
    assert set(kv_caches) == expected_layers

    k_cache, v_cache = kv_caches[MAIN_LAYER]
    assert k_cache.shape == (num_blocks, 256, 1, 64)
    assert v_cache.shape == (num_blocks, 256, 1, 64)
    assert k_cache.dtype == v_cache.dtype == torch.bfloat16
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    # Both views address the same contiguous backing at disjoint offsets.
    assert k_cache.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    assert v_cache.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    slot = layout.owner(MAIN_LAYER).slot
    assert k_cache.storage_offset() == (slot * layout.slot_backing_size + layout.region("r2").offset) // 2
    assert v_cache.storage_offset() == (slot * layout.slot_backing_size + layout.region("r3").offset) // 2

    raw_cache = kv_caches[RAW_LAYER]
    assert isinstance(raw_cache, torch.Tensor)
    assert raw_cache.shape == (num_blocks, 1, 8, 32)
    assert raw_cache.dtype == torch.bfloat16

    compressed_cache = kv_caches[COMPRESSED_LAYER]
    assert isinstance(compressed_cache, torch.Tensor)
    # The compressed cache stores block_size/ratio states per page.
    assert compressed_cache.shape == (num_blocks, 1, 64, 32)

    gdn_caches = kv_caches[GDN_LAYERS[0]]
    assert isinstance(gdn_caches, list)
    assert [cache.shape for cache in gdn_caches] == [(num_blocks, 64, 4), (num_blocks, 128, 128)]

    ple_caches = kv_caches[PLE_LAYER]
    assert isinstance(ple_caches, list)
    assert [cache.shape for cache in ple_caches] == [(num_blocks, 128, 5)]


def test_six_region_reshape_rejects_page_mismatch(monkeypatch):
    raw_layer = RAW_LAYER
    raw_spec = _six_region_groups()[1].kv_cache_spec
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([raw_layer], raw_spec)],
    )
    tampered = SimpleNamespace(
        slot_count=1,
        slot_backing_size=1 << 20,
        owner=lambda _layer_name: SimpleNamespace(slot=0, role=QSA_RAW),
        region=lambda _name: SimpleNamespace(offset=0, page_size_bytes=3),
    )
    monkeypatch.setattr(attn_utils, "build_six_region_kv_cache_layout", lambda *_args: tampered)

    with pytest.raises(ValueError, match="six-region page mismatch"):
        attn_utils._reshape_kv_cache_v2(
            attn_groups=[
                AttentionGroup(
                    backend=AscendQSABackend,
                    layer_names=[raw_layer],
                    kv_cache_spec=raw_spec,
                    kv_cache_group_id=0,
                )
            ],
            kv_cache_raw_tensors={raw_layer: torch.zeros(1 << 20, dtype=torch.int8)},
            cache_dtype="auto",
            kernel_block_sizes=[raw_spec.block_size],
            shared_kv_cache_layers={},
            kv_cache_config=kv_cache_config,
        )


class _HybridLayers:
    """One attention owner plus two Mamba owners with distinct page sizes."""

    ATTENTION = "model.layers.0.self_attn.attn"
    BIG_MAMBA = "model.layers.0.linear_attn"
    SMALL_MAMBA = "model.layers.1.linear_attn"

    def __init__(self):
        self.attention_spec = FullAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=64,
            head_size_v=64,
            dtype=torch.bfloat16,
        )
        self.big_mamba_spec = MambaSpec(
            block_size=256,
            shapes=((64, 4), (256, 256)),
            dtypes=(torch.bfloat16, torch.bfloat16),
        )
        self.small_mamba_spec = MambaSpec(
            block_size=256,
            shapes=((64, 4), (128, 128)),
            dtypes=(torch.bfloat16, torch.bfloat16),
        )
        self.layers = {
            self.ATTENTION: SimpleNamespace(
                get_kv_cache_spec=lambda _cfg: self.attention_spec,
                get_attn_backend=lambda: self.attention_backend,
            ),
            self.BIG_MAMBA: SimpleNamespace(get_kv_cache_spec=lambda _cfg: self.big_mamba_spec),
            self.SMALL_MAMBA: SimpleNamespace(get_kv_cache_spec=lambda _cfg: self.small_mamba_spec),
        }

    attention_backend = AscendQSABackend


def _run_hybrid_get_kv_cache_spec(monkeypatch, layers: _HybridLayers) -> dict:
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda _cfg, _type: layers.layers)
    monkeypatch.setattr(attn_utils, "enable_sfa_dcp_replicated_indexer", lambda _cfg: False)
    monkeypatch.setattr(attn_utils, "kv_cache_dtype_str_to_dtype", lambda _s, _m: torch.bfloat16)
    return attn_utils.get_kv_cache_spec(_v2_vllm_config())


def test_descriptor_native_hybrid_pads_attention_to_mamba_page(monkeypatch):
    layers = _HybridLayers()
    specs = _run_hybrid_get_kv_cache_spec(monkeypatch, layers)

    attention = specs[_HybridLayers.ATTENTION]
    big_mamba = specs[_HybridLayers.BIG_MAMBA]
    small_mamba = specs[_HybridLayers.SMALL_MAMBA]
    # Attention page: 256 * (64 + 64) * 2 = 65536 bytes; big Mamba page:
    # (64 * 4 + 256 * 256) * 2 = 131584 bytes.
    assert attention.real_page_size_bytes == 65536
    assert attention.page_size_padded == 131584
    assert attention.page_size_bytes == 131584
    # Mamba keeps its own page geometry on the descriptor-native path.
    assert big_mamba.page_size_bytes == 131584
    assert big_mamba.page_size_padded is None
    assert small_mamba.page_size_bytes == 33280
    assert small_mamba.page_size_padded is None


def test_plain_hybrid_still_pads_all_layers_to_common_page(monkeypatch):
    layers = _HybridLayers()
    layers.attention_backend = AscendAttentionBackend
    specs = _run_hybrid_get_kv_cache_spec(monkeypatch, layers)

    attention = specs[_HybridLayers.ATTENTION]
    small_mamba = specs[_HybridLayers.SMALL_MAMBA]
    # The legacy branch pads every layer up to the common maximum.
    assert attention.page_size_padded == 131584
    assert small_mamba.page_size_padded == 131584


def _run_qsa_build_attn_metadata(num_reqs: int, num_actual_reqs: int):
    builder = MagicMock(spec=QSAMetadataBuilder)
    attn_group = SimpleNamespace(
        layer_names=[MAIN_LAYER],
        get_metadata_builder=lambda _: builder,
    )
    kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())])
    boundaries = [0, 2, 4, 5, 6, 7]

    metadata = attn_utils.build_attn_metadata(
        attn_groups=[[attn_group]],
        num_reqs=num_reqs,
        num_actual_reqs=num_actual_reqs,
        num_tokens=5,
        query_start_loc_gpu=torch.tensor(boundaries[: num_reqs + 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(boundaries[: num_reqs + 1], dtype=torch.int32),
        max_query_len=2,
        seq_lens=torch.tensor([5] * num_reqs, dtype=torch.int32),
        max_seq_len=5,
        block_tables=(torch.zeros((num_reqs, 1), dtype=torch.int32),),
        slot_mappings=(torch.zeros(num_reqs, dtype=torch.int64),),
        kv_cache_config=kv_cache_config,
        is_prefilling=torch.tensor([False] * num_reqs),
        seq_lens_np=np.full(num_reqs, 5, dtype=np.int32),
        positions=torch.zeros(5, dtype=torch.int64),
    )
    return metadata, builder


def test_qsa_builder_receives_live_boundaries_under_graph_padding():
    """FULL graph padding appends dummy request boundaries; the QSA builder
    reads the CPU tail as its mapped-token count and must only see the live
    requests."""
    metadata, builder = _run_qsa_build_attn_metadata(num_reqs=4, num_actual_reqs=3)

    common = builder.build.call_args.kwargs["common_attn_metadata"]
    assert common.query_start_loc_cpu.tolist() == [0, 2, 4, 5]
    assert metadata[MAIN_LAYER] is builder.build.return_value


def test_qsa_builder_keeps_full_boundaries_without_padding():
    metadata, builder = _run_qsa_build_attn_metadata(num_reqs=4, num_actual_reqs=4)

    common = builder.build.call_args.kwargs["common_attn_metadata"]
    assert common.query_start_loc_cpu.tolist() == [0, 2, 4, 5, 6]
    assert metadata[MAIN_LAYER] is builder.build.return_value
