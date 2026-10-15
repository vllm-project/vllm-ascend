# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU page-registration/transfer contracts; no Mooncake or NPU execution."""

import ast
import ctypes
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MOONCAKE = "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/"


class MLASpec(SimpleNamespace):
    pass


class IndexerSpec(MLASpec):
    pass


class TransferMetadata(SimpleNamespace):
    pass


def _load(file, name, scope, owner=None):
    nodes = ast.parse((ROOT / file).read_text(encoding="utf-8")).body
    if owner:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), file, "exec"), scope)
    return scope[name]


@pytest.fixture
def contract():
    scope = {
        "torch": torch,
        "np": np,
        "math": math,
        "MLAAttentionSpec": MLASpec,
        "SlidingWindowMLASpec": type("SlidingMLA", (), {}),
        "AscendSFAIndexerCacheSpec": IndexerSpec,
        "MooncakeTransferMetadata": TransferMetadata,
        "MooncakePPTransferMetadata": SimpleNamespace,
        "MooncakeTPTransferMetadata": SimpleNamespace,
        "MooncakePCPTransferMetadata": SimpleNamespace,
        "MambaSpec": type("Mamba", (), {}),
        "is_circular_kv_cache_spec": lambda _: False,
        "logger": Mock(),
        "global_te": Mock(),
        "get_kv_cache_tensor_layers": lambda tensor: tensor.layers,
        "collect_configured_register_regions": lambda *_: SimpleNamespace(ptrs=[], lengths=[]),
        "validate_register_region_count": Mock(),
        "tensor_storage_key": lambda tensor: tensor.untyped_storage().data_ptr(),
    }
    _load(MOONCAKE + "utils.py", "as_kv_cache_tensors", scope)
    _load(MOONCAKE + "utils.py", "group_concurrent_contiguous", scope)
    _load(MOONCAKE + "base_worker.py", "_is_component_major_mla_binding", scope)
    for name in ("_get_shared_page_metadata", "register_kv_caches"):
        _load(MOONCAKE + "base_worker.py", name, scope, "MooncakeBaseConnectorWorker")
    _load(MOONCAKE + "pull_scheduler.py", "_merge_metadata_by_pp_rank", scope, "MooncakeSchedulerSendingThread")
    for name in (
        "_build_remote_transfer_layout",
        "_append_spec_transfer_addresses",
        "_append_block_transfer_addresses",
    ):
        _load(MOONCAKE + "pull_worker.py", name, scope, "MooncakePullRecvingThread")
    return scope


def _register(scope, layout, manager_block_size=128, padding=0):
    num_blocks = 12
    slot_elems = 128 * 576 + padding
    raw = torch.full((num_blocks * slot_elems,), -17, dtype=torch.bfloat16)
    fused = raw.as_strided((num_blocks, 128, 1, 576), (slot_elems, 576, 576, 1))
    if layout == "fused":
        cache = fused
    else:
        cache = (
            raw.as_strided((num_blocks, 128, 1, 512), (slot_elems, 512, 512, 1)),
            raw.as_strided((num_blocks, 128, 1, 64), (slot_elems, 64, 64, 1), 128 * 512),
        )
    spec = MLASpec(block_size=manager_block_size)
    worker = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=num_blocks * 128 // manager_block_size,
            kv_cache_tensors=[SimpleNamespace(layers=["mla"])],
        ),
        ascend_config=SimpleNamespace(kvpp_config=SimpleNamespace(size=1)),
        pcp_rank=0,
        tp_rank=0,
        tp_size=1,
        engine_id="producer",
        te_rpc_port=9000,
        block_size=manager_block_size,
        side_channel_host="127.0.0.1",
        handshake_port=5000,
        layer_name_to_group_index={"mla": 0},
        layer_name_to_spec_index={"mla": 0},
        kv_cache_specs=[spec],
        _build_kv_cache_spec_mappings=lambda: None,
    )
    worker._get_shared_page_metadata = MethodType(scope["_get_shared_page_metadata"], worker)
    scope["register_kv_caches"](worker, {"mla": cache})
    return worker.transfer_metadata, fused, raw


def _merge(scope, metadata):
    scheduler = SimpleNamespace(engine_id="producer", pp_size=1, pcp_size=1, tp_size=1, use_kv_pp=False)
    pp = scope["_merge_metadata_by_pp_rank"](scheduler, {(0, 0, 0): metadata})
    return SimpleNamespace(pcp_size=1, tp_size=1, dcp_size=1, use_kv_pp=False, metadata_by_pp_rank=pp)


def _consumer(scope, metadata):
    worker = SimpleNamespace(
        layer_names=metadata.layer_names,
        spec_indices=[0],
        kv_cache_specs=[MLASpec()],
        block_shapes=metadata.block_shapes,
        block_strides=metadata.block_strides,
        block_lens=metadata.block_lens,
        kv_caches_base_addr=metadata.kv_caches_base_addr,
        _get_layer_remote_tp_rank_groups=Mock(return_value=[[0]]),
        _append_block_transfer_addresses=scope["_append_block_transfer_addresses"],
    )
    return worker


@pytest.mark.parametrize("padding", [0, 256])
@pytest.mark.parametrize("manager_block_size", [128, 768])
def test_fused_pd_registration_and_page_copy_preserve_tokens(contract, padding, manager_block_size):
    producer, p_cache, p_raw = _register(contract, "fused", padding=padding)
    consumer, d_cache, d_raw = _register(contract, "fused", manager_block_size, padding=padding)
    assert producer.block_shapes == consumer.block_shapes == [[(128, 1, 576)]]
    assert consumer.block_size_scales == [[manager_block_size // 128]]
    remote = _merge(contract, producer)
    worker = _consumer(contract, consumer)
    assert contract["_build_remote_transfer_layout"](worker, remote)[1] == {0: [(0, 0)]}
    values = torch.arange(128 * 576).reshape(128, 1, 576).to(torch.bfloat16)
    p_cache[1].copy_(values)
    p_cache[2].copy_(values + 100)
    before = d_raw.clone()
    src: list[int] = []
    dst: list[int] = []
    lengths: list[int] = []
    contract["_append_spec_transfer_addresses"](
        worker,
        0,
        0,
        0,
        1,
        1,
        1,
        {(0, 0): [("request", [3, 4], [1, 2])]},
        remote.metadata_by_pp_rank[0],
        src,
        dst,
        lengths,
    )
    # Mooncake names the local read destination src and the remote source dst.
    # Simulate only the planned CPU byte copies, using the actual production plan.
    for local_addr, remote_addr, length in zip(src, dst, lengths):
        ctypes.memmove(local_addr, remote_addr, length)
    torch.testing.assert_close(d_cache[3], p_cache[1])
    torch.testing.assert_close(d_cache[4], p_cache[2])
    span = 128 * 576 + padding
    torch.testing.assert_close(d_raw[: 3 * span], before[: 3 * span])
    torch.testing.assert_close(d_raw[5 * span :], before[5 * span :])
    assert p_raw.numel() == d_raw.numel()


@pytest.mark.parametrize("local_layout,remote_layout", [("fused", "component"), ("component", "fused")])
def test_equal_length_pd_pages_with_different_representations_are_rejected(contract, local_layout, remote_layout):
    local, _, _ = _register(contract, local_layout)
    remote, _, _ = _register(contract, remote_layout)
    assert local.block_lens == remote.block_lens
    assert local.block_strides == remote.block_strides
    assert local.block_shapes != remote.block_shapes
    worker = _consumer(contract, local)
    with pytest.raises(ValueError, match="MLA page representation mismatch"):
        contract["_build_remote_transfer_layout"](worker, _merge(contract, remote))
    worker._get_layer_remote_tp_rank_groups.assert_not_called()


def test_component_major_pd_retains_its_flat_page_contract(contract):
    local, _, _ = _register(contract, "component")
    assert local.block_shapes == [[(128 * 576,)]]
    worker = _consumer(contract, local)
    assert contract["_build_remote_transfer_layout"](worker, _merge(contract, local))[1] == {0: [(0, 0)]}


def test_every_mla_layer_is_checked_even_when_the_rank_group_spec_is_cached(contract):
    local, _, _ = _register(contract, "fused")
    remote = _merge(contract, local)
    pp = remote.metadata_by_pp_rank[0]
    pp.layer_names = ["first", "second"]
    pp.block_shapes = [[(128, 1, 576)], [(128 * 576,)]]
    worker = _consumer(contract, local)
    worker.layer_names = ["first", "second"]
    worker.spec_indices = [0, 0]
    worker.block_shapes = [[(128, 1, 576)], [(128, 1, 576)]]
    with pytest.raises(ValueError, match="mismatch for layer 'second'"):
        contract["_build_remote_transfer_layout"](worker, remote)
    assert worker._get_layer_remote_tp_rank_groups.call_count == 1


def test_indexer_representation_is_not_checked_as_flashmla(contract):
    local, _, _ = _register(contract, "fused")
    remote = _merge(contract, local)
    remote.metadata_by_pp_rank[0].block_shapes = [[(128 * 576,)]]
    worker = _consumer(contract, local)
    worker.kv_cache_specs = [IndexerSpec()]
    assert contract["_build_remote_transfer_layout"](worker, remote)[1] == {0: [(0, 0)]}


@pytest.mark.parametrize("compatible", [False, True])
def test_sliding_mla_pages_also_require_matching_representations(contract, compatible):
    local, _, _ = _register(contract, "fused")
    remote = _merge(contract, local)
    if not compatible:
        remote.metadata_by_pp_rank[0].block_shapes = [[(128 * 576,)]]
    worker = _consumer(contract, local)
    worker.kv_cache_specs = [contract["SlidingWindowMLASpec"]()]
    if compatible:
        assert contract["_build_remote_transfer_layout"](worker, remote)[1] == {0: [(0, 0)]}
    else:
        with pytest.raises(ValueError, match="MLA page representation mismatch"):
            contract["_build_remote_transfer_layout"](worker, remote)
