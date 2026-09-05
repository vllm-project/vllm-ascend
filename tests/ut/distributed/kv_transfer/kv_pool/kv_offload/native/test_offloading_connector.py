# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheTensor, MambaSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_offload.base import CanonicalKVCaches

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native import offloading_connector as module
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector import (
    AscendOffloadingConnector,
    AscendOffloadingConnectorWorker,
    _canonicalize_split_attention_cache,
    _make_int8_block_view,
)


@pytest.mark.parametrize(("shape", "blocks"), [((), 1), ((1, 4), 2)])
def test_block_view_rejects_insufficient_physical_blocks(shape, blocks):
    with pytest.raises(ValueError, match="too few physical blocks"):
        _make_int8_block_view(torch.empty(shape, dtype=torch.int8), blocks, 1, 4)


@pytest.mark.parametrize(("factor", "stride", "message"), [(1, 2, "overlap"), (2, 8, "Cannot coalesce")])
def test_block_view_rejects_overlapping_or_padded_coalesced_storage(factor, stride, message):
    tensor = torch.empty(64, dtype=torch.int8).as_strided((4, 4), (stride, 1))
    with pytest.raises(ValueError, match=message):
        _make_int8_block_view(tensor, 2, factor, 4)


@pytest.mark.parametrize("parts", [[], [None]])
def test_split_cache_requires_nonempty_tensor_sequence(parts):
    with pytest.raises(TypeError, match="one or more tensors"):
        _canonicalize_split_attention_cache(parts, 2, 8)


def test_split_cache_rejects_empty_payload_and_incomplete_page():
    with pytest.raises(ValueError, match="non-empty"):
        _canonicalize_split_attention_cache([torch.empty(2, 0)], 2, 8)
    with pytest.raises(ValueError, match="do not cover one logical page"):
        _canonicalize_split_attention_cache([torch.empty(2, 3, dtype=torch.int8)], 2, 8)


def make_adapter(groups, descriptors=()):
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = SimpleNamespace(num_blocks=2, kv_cache_groups=groups, kv_cache_tensors=descriptors)
    worker._init_worker = MagicMock()
    return worker


@pytest.mark.parametrize("packed", [False, True])
def test_mixed_tensor_and_split_layout_deduplicates_shared_references(packed):
    spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=2, dtype=torch.int8)
    tensor = torch.arange(48, dtype=torch.int8).as_strided((2, 8), (16, 1), 4)
    part = torch.arange(16, dtype=torch.int8).reshape(4, 4)
    uniform = UniformTypeKVCacheSpecs(block_size=2, kv_cache_specs={"plain": spec, "split": spec, "alias": spec})
    groups = [
        SimpleNamespace(layer_names=["plain", "split", "alias"], kv_cache_spec=uniform),
        SimpleNamespace(layer_names=["split"], kv_cache_spec=spec),
    ]
    descriptor = KVCacheTensor(size=48, shared_by=["plain"], block_stride=16 if packed else 0)
    worker = make_adapter(groups, [descriptor])
    worker.register_kv_caches({"plain": tensor, "split": (part, part), "alias": (part, part)})
    result = worker._init_worker.call_args.args[0]
    assert len(result.tensors) == 2
    assert result.tensors[0].tensor.data_ptr() == tensor.data_ptr()
    assert result.tensors[0].tensor.stride(0) == (16 if packed else 8)
    assert [ref.tensor_idx for ref in result.group_data_refs[0]] == [0, 1, 1]
    assert [ref.tensor_idx for ref in result.group_data_refs[1]] == [1]


def test_compatible_groups_delegate_to_upstream_without_adaptation(monkeypatch):
    spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=2, dtype=torch.int8)
    groups = [
        SimpleNamespace(layer_names=["a", "b"], kv_cache_spec=spec),
        SimpleNamespace(layer_names=[], kv_cache_spec=spec),
    ]
    worker = make_adapter(groups)
    upstream = MagicMock()
    monkeypatch.setattr(module.OffloadingConnectorWorker, "register_kv_caches", upstream)
    caches = {"a": torch.empty(2, 8), "b": torch.empty(2, 8)}
    worker.register_kv_caches(caches)
    upstream.assert_called_once_with(caches)
    worker._init_worker.assert_not_called()


@pytest.mark.parametrize("case", ["attention_type", "empty_mamba", "tuple_mamba", "mamba_size", "unsupported"])
def test_adapter_rejects_invalid_layout_before_initializing_worker(case):
    attention = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=2, dtype=torch.int8)
    mamba = MambaSpec(block_size=2, shapes=((2,),), dtypes=(torch.int8,))
    spec, value, error = {
        "attention_type": (attention, object(), TypeError),
        "empty_mamba": (mamba, [], TypeError),
        "tuple_mamba": (mamba, (torch.empty(2, 2),), TypeError),
        "mamba_size": (mamba, [torch.empty(2, 3, dtype=torch.int8)], ValueError),
        "unsupported": (SimpleNamespace(), torch.empty(2, 2), NotImplementedError),
    }[case]
    worker = make_adapter(
        [
            SimpleNamespace(layer_names=["trigger"], kv_cache_spec=attention),
            SimpleNamespace(layer_names=["invalid"], kv_cache_spec=spec),
        ]
    )
    with pytest.raises(error):
        worker.register_kv_caches({"trigger": [torch.empty(2, 8, dtype=torch.int8)], "invalid": value})
    worker._init_worker.assert_not_called()


def test_scheduler_connector_keeps_absent_worker(monkeypatch):
    monkeypatch.setattr(OffloadingConnector, "__init__", lambda self, *args: setattr(self, "connector_worker", None))
    connector = AscendOffloadingConnector(object(), KVConnectorRole.SCHEDULER, object())
    assert connector.connector_worker is None


def test_ascend_connector_replaces_worker_with_current_vllm_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_config = object()
    kv_cache_config = object()
    spec = SimpleNamespace(
        replicated_layout=False,
        config=SimpleNamespace(parallel=SimpleNamespace(rank=0)),
    )

    def fake_upstream_init(
        self,
        init_vllm_config,
        role,
        init_kv_cache_config,
    ) -> None:
        assert init_vllm_config is vllm_config
        assert role == KVConnectorRole.WORKER
        assert init_kv_cache_config is kv_cache_config
        self.connector_worker = SimpleNamespace(spec=spec)

    monkeypatch.setattr(OffloadingConnector, "__init__", fake_upstream_init)

    connector = AscendOffloadingConnector(
        vllm_config,
        KVConnectorRole.WORKER,
        kv_cache_config,
    )

    assert isinstance(connector.connector_worker, AscendOffloadingConnectorWorker)
    assert connector.connector_worker.spec is spec
    assert connector.connector_worker.vllm_config is vllm_config
    assert connector.connector_worker.kv_cache_config is kv_cache_config


def test_split_kv_cache_is_canonicalized_without_copy() -> None:
    key = torch.empty((4, 2, 3), dtype=torch.bfloat16)
    value = torch.empty((4, 2, 3), dtype=torch.bfloat16)

    views = _canonicalize_split_attention_cache(
        (key, value),
        num_blocks=4,
        unpadded_page_size_bytes=24,
    )

    assert len(views) == 2
    assert [view.shape for view, _ in views] == [(4, 12), (4, 12)]
    assert [copy_size for _, copy_size in views] == [12, 12]
    assert views[0][0].data_ptr() == key.data_ptr()
    assert views[1][0].data_ptr() == value.data_ptr()


def test_split_kv_cache_coalesces_kernel_blocks() -> None:
    key = torch.empty((8, 2), dtype=torch.int8)
    value = torch.empty((8, 2), dtype=torch.int8)

    views = _canonicalize_split_attention_cache(
        (key, value),
        num_blocks=4,
        unpadded_page_size_bytes=8,
    )

    assert [view.shape for view, _ in views] == [(4, 4), (4, 4)]
    assert [copy_size for _, copy_size in views] == [4, 4]


def test_extra_physical_blocks_do_not_hide_separate_value_cache() -> None:
    key = torch.empty((10, 2), dtype=torch.int8)
    value = torch.empty((10, 2), dtype=torch.int8)

    views = _canonicalize_split_attention_cache(
        (key, value),
        num_blocks=4,
        unpadded_page_size_bytes=4,
    )

    assert len(views) == 2
    assert [view.shape for view, _ in views] == [(4, 2), (4, 2)]
    assert views[0][0].data_ptr() == key.data_ptr()
    assert views[1][0].data_ptr() == value.data_ptr()


def test_split_kv_cache_prefers_complete_overlapping_view() -> None:
    full = torch.empty((4, 8), dtype=torch.int8)
    key = full[:, :4]
    scale = full[:, 4:]

    views = _canonicalize_split_attention_cache(
        (key, scale, full),
        num_blocks=4,
        unpadded_page_size_bytes=8,
    )

    assert len(views) == 1
    assert views[0][0].data_ptr() == full.data_ptr()
    assert views[0][0].shape == (4, 8)
    assert views[0][1] == 8


def test_split_kv_cache_rejects_noncontiguous_block_payload() -> None:
    key = torch.empty((4, 2, 3), dtype=torch.int8).transpose(1, 2)
    value = torch.empty((4, 3, 2), dtype=torch.int8)

    with pytest.raises(ValueError, match="block payload is non-contiguous"):
        _canonicalize_split_attention_cache(
            (key, value),
            num_blocks=4,
            unpadded_page_size_bytes=12,
        )


def test_ascend_connector_worker_accepts_separate_kv_tensors() -> None:
    layer_name = "model.layers.0.self_attn"
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.bfloat16,
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=4,
        kv_cache_groups=[SimpleNamespace(layer_names=[layer_name], kv_cache_spec=spec)],
        kv_cache_tensors=[],
    )
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = kv_cache_config
    captured: list[CanonicalKVCaches] = []
    worker._init_worker = captured.append

    worker.register_kv_caches(
        {
            layer_name: (
                torch.empty((4, 2, 1, 3), dtype=torch.bfloat16),
                torch.empty((4, 2, 1, 3), dtype=torch.bfloat16),
            )
        }
    )

    assert len(captured) == 1
    canonical = captured[0]
    assert len(canonical.tensors) == 2
    assert [tensor.tensor.shape for tensor in canonical.tensors] == [
        (4, 12),
        (4, 12),
    ]
    assert [ref.page_size_bytes for ref in canonical.group_data_refs[0]] == [
        12,
        12,
    ]


def test_ascend_connector_worker_accepts_aligned_mamba_states() -> None:
    layer_name = "model.layers.0.mixer"
    spec = MambaSpec(
        block_size=1,
        shapes=((2,), (3,)),
        dtypes=(torch.int8, torch.int8),
        page_size_padded=8,
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=4,
        kv_cache_groups=[SimpleNamespace(layer_names=[layer_name], kv_cache_spec=spec)],
        kv_cache_tensors=[],
    )
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = kv_cache_config
    captured: list[CanonicalKVCaches] = []
    worker._init_worker = captured.append

    raw = torch.empty(1 + 4 * 2 + 4 * 3, dtype=torch.int8)
    first_state = raw[1:9].view(4, 2)
    second_state = raw[9:21].view(4, 3)
    worker.register_kv_caches({layer_name: [first_state, second_state]})

    assert len(captured) == 1
    canonical = captured[0]
    assert [tensor.tensor.shape for tensor in canonical.tensors] == [
        (4, 2),
        (4, 3),
    ]
    assert [ref.page_size_bytes for ref in canonical.group_data_refs[0]] == [
        2,
        3,
    ]
    assert canonical.tensors[0].tensor.data_ptr() == first_state.data_ptr()
    assert canonical.tensors[1].tensor.data_ptr() == second_state.data_ptr()


def test_offloading_connector_is_registered_with_ascend_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.distributed.kv_transfer import register_connector

    registrations: dict[str, tuple[str, str]] = {}

    def capture_registration(
        cls,
        name: str,
        module_path: str,
        class_name: str,
    ) -> None:
        registrations[name] = (module_path, class_name)

    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    monkeypatch.setattr(
        KVConnectorFactory,
        "register_connector",
        classmethod(capture_registration),
    )
    register_connector()

    assert registrations["OffloadingConnector"] == (
        "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector",
        "AscendOffloadingConnector",
    )
