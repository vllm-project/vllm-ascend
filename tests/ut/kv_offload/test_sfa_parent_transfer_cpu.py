# SPDX-License-Identifier: Apache-2.0
"""CPU transfer contract; run with --confcutdir=tests/ut/kv_offload.

Loads complete production modules while replacing only vLLM logging/network
imports and the unrelated metadata base. The copy engine uses real CPU memory;
this does not validate memfabric or NPU kernels.
"""

import ctypes
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture
def reader(monkeypatch):
    root = Path(__file__).parents[3] / "vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h"
    stubs = {
        "vllm.logger": {"logger": logging.getLogger("sfa_cpu")},
        "vllm.utils.network_utils": {"get_ip": lambda: "127.0.0.1"},
        "vllm.distributed.kv_transfer.kv_connector.v1.base": {"KVConnectorMetadata": object},
    }
    for name, attrs in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol", "protocol.py")
    return load("sfa_cpu_read_thread", "read_thread.py")


def make_transfer(reader, tp_rank=0, tp_size=1):
    source = torch.arange(6 * 4 * 12, dtype=torch.float32).to(torch.bfloat16).view(6, 4, 1, 12)
    target = torch.full_like(source, -1)
    state = reader.ConsumerReadState(
        num_blocks=6,
        tp_size=tp_size,
        layer_metadata={},
        main_name_to_idx={"layer": 0},
        indexer_tensors=[None],
        indexer_scale_tensors=[None],
        dest_blocks_by_req={"req": ([4, 0, 3], [])},
        get_offload_layer_id=lambda _: 0,
        cpu_parent_caches=[target],
        main_parent_gva_bases=[target.data_ptr()],
        main_parent_block_lens=[96],
        main_cache_dtype="bfloat16",
        main_cache_num_heads=1,
        main_cache_nope_dim=8,
        main_cache_rope_dim=4,
    )
    thread = reader.MembPullReadThread.__new__(reader.MembPullReadThread)
    thread.tp_rank = tp_rank
    thread._state = state
    meta = {
        "layer": {
            "base_addrs": [source.data_ptr()],
            "block_len": [48],
            "block_size_scale": [2],
            "main_tensor_count": 1,
            "main_cache_layout": "token_concat",
            "main_cache_dtype": "bfloat16",
            "main_cache_num_heads": 1,
            "main_cache_nope_dim": 8,
            "main_cache_rope_dim": 4,
            "has_indexer": False,
        }
    }
    return thread, source, target, meta


def test_full_parent_transfer_updates_both_views_without_copy(reader):
    thread, source, target, meta = make_transfer(reader)
    nope, rope = target[..., :8], target[..., 8:]
    layer = thread._resolve_read_layer("layer", meta)
    local, peer, lengths, info = thread._build_req_descriptors(layer, "req", [2, 5, 1], [], True)
    assert lengths == [96, 96, 96]
    assert info["atomic_transfers"] == 3
    for dst, src, size in zip(local, peer, lengths):
        ctypes.memmove(dst, src, size)
    for dst_block, src_block in zip([4, 0, 3], [2, 5, 1]):
        torch.testing.assert_close(nope[dst_block], source[src_block, ..., :8])
        torch.testing.assert_close(rope[dst_block], source[src_block, ..., 8:])
    assert torch.all(target[1] == -1)
    assert nope.data_ptr() == target.data_ptr()
    assert rope.data_ptr() == target.data_ptr() + 16


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_tp_partition_covers_parent_pages_once(reader, tp_size):
    thread, source, target, meta = make_transfer(reader, tp_size=tp_size)
    layer = thread._resolve_read_layer("layer", meta)
    destinations = []
    for rank in range(tp_size):
        thread.tp_rank = rank
        local, _, lengths, _ = thread._build_req_descriptors(layer, "req", [2, 5, 1], [], True)
        assert all(length == 96 for length in lengths)
        destinations.extend(local)
    assert sorted(destinations) == sorted(target.data_ptr() + block * 96 for block in [4, 0, 3])


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"main_cache_layout": "separate_nope_rope"}, "token_concat main cache layout"),
        ({"main_tensor_count": 2}, "one parent main tensor"),
        ({"main_cache_dtype": "float16"}, "main cache dtype mismatch"),
        ({"main_cache_dtype": None}, "main cache dtype mismatch"),
        ({"main_cache_dtype": "float32"}, "main cache dtype mismatch"),
        ({"main_cache_nope_dim": 7}, "head geometry mismatch"),
        ({"block_len": []}, "array lengths mismatch"),
        ({"block_len": [0]}, "invalid page geometry"),
        ({"block_size_scale": [1]}, "main parent page bytes mismatch"),
        ({"base_addrs": [1, 2], "block_len": [48, 48], "block_size_scale": [2, 2]}, "unexpected tensor count"),
    ],
)
def test_invalid_metadata_rejected_before_transfer(reader, overrides, message):
    thread, source, target, meta = make_transfer(reader)
    meta["layer"].update(overrides)
    with pytest.raises(RuntimeError, match=message):
        thread._resolve_read_layer("layer", meta)
    assert torch.all(target == -1)
