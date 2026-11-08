# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Independent CPU oracle for native Qwen K/V and full GDN state transfer.

Geometry matches the Qwen3.8-27B text config, without weights or NPU calls.
Expected payloads use logical tensor indices, never Store segment templates.
"""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

native_connector_module = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector")
if not hasattr(native_connector_module.MooncakeStoreConnector, "_create_worker"):
    pytest.skip("requires vLLM native Store device hooks (#66294)", allow_module_level=True)

# isort: split
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout import (
    HeadMajorStoreLayout,
    KeyMetadata,
    TokenMajorStoreLayout,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator, is_conv_state_dim_first
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaAttentionBackendEnum,
    MambaSpec,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.native_worker import AscendNativeStoreWorker

BLOCK = 256
CHUNK = 128
BLOCKS = 3
STORE_TP = 4
KV_HEADS = 4
HEAD_DIM = 256
KEY_HEADS = 16
VALUE_HEADS = 48
STATE_DIM = 128
PROJECTIONS = (KEY_HEADS * STATE_DIM, KEY_HEADS * STATE_DIM, VALUE_HEADS * STATE_DIM)
DRAFTS = 3
CONV_ROWS = 3 + DRAFTS


def make_qwen_registration_inputs(tp, physical_layout):
    shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp, KEY_HEADS, VALUE_HEADS, STATE_DIM, STATE_DIM, 4, DRAFTS
    )
    attention = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=KV_HEADS // tp, head_size=HEAD_DIM, dtype=torch.bfloat16
    )
    state = MambaSpec(
        block_size=BLOCK,
        shapes=shapes,
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
        num_speculative_blocks=DRAFTS,
    )
    groups = [KVCacheGroupSpec(["attention"], attention), KVCacheGroupSpec(["gdn"], state)]
    cache_config = KVCacheConfig(num_blocks=BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups)
    # K and V intentionally share one allocation, as real planar caches can.
    shape = (
        (BLOCKS, BLOCK, KV_HEADS // tp, HEAD_DIM)
        if physical_layout == "NHD"
        else (BLOCKS, KV_HEADS // tp, BLOCK, HEAD_DIM)
    )
    kv = torch.full((2, *shape), -1, dtype=torch.bfloat16)
    conv, recurrent = [torch.full((BLOCKS, *shape), -1, dtype=dtype) for shape, dtype in zip(shapes, state.dtypes)]
    caches = {"attention": (kv[0], kv[1]), "gdn": (conv, recurrent)}
    config = SimpleNamespace(
        num_hash_lookahead_tokens=0,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5_text")),
        compilation_config=SimpleNamespace(static_forward_context={"gdn": SimpleNamespace(kv_cache=(conv, recurrent))}),
    )
    return config, cache_config, caches


def make_worker(tp, rank, physical_layout):
    config, cache_config, caches = make_qwen_registration_inputs(tp, physical_layout)
    groups = cache_config.kv_cache_groups
    kv = caches["attention"][0]
    conv, recurrent = caches["gdn"]
    worker = AscendNativeStoreWorker.__new__(AscendNativeStoreWorker)
    worker._capacity_only = False
    worker._ascend_vllm_config = config
    worker._kv_cache_config = cache_config
    worker._kv_cache_groups = groups
    worker._group_tp_replication_factors = (1, 1)
    worker.tp_size, worker.tp_rank = tp, rank
    worker.pp_size = worker.pcp_size = worker.dcp_size = 1
    worker.pp_rank = worker.pcp_rank = worker.dcp_rank = 0
    worker.hash_block_size = CHUNK
    worker.requested_store_chunk_size = CHUNK
    worker.store_tp_size = STORE_TP
    worker._device_regions = {}
    worker.store = SimpleNamespace(register_buffer=Mock(return_value=0), raise_if_failed=Mock())
    worker._start_transfer_threads = Mock()
    layout_cls = TokenMajorStoreLayout if physical_layout == "NHD" else HeadMajorStoreLayout
    plans = tuple(worker._make_store_group_plan(group, STORE_TP, layout_cls, 1) for group in groups)
    assert all(plan is not None for plan in plans)
    worker.token_dbs = worker._build_token_databases(KeyMetadata("qwen-cpu", rank, 0, 0, 0), plans)
    worker.register_kv_caches(caches)  # Exercise Qwen worker dispatch as well as registration.
    worker._start_transfer_threads.assert_called_once_with()
    registrations = [call.args for call in worker.store.register_buffer.call_args_list]
    expected_regions = {
        (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes()) for tensor in (kv, conv, recurrent)
    }
    assert len(registrations) == 3  # K/V must not register their shared storage twice.
    assert set(registrations) == expected_regions
    assert worker._device_regions == dict(expected_regions)
    assert worker.num_blocks == BLOCKS
    return worker, caches


def logical_views(caches, physical_layout):
    k, v = caches["attention"]
    if physical_layout == "HND":
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    conv, recurrent = caches["gdn"]
    if is_conv_state_dim_first():
        conv = conv.transpose(1, 2)
    return k, v, conv, recurrent  # B/token/head/dim, B/history/channel, B/head/value/key


def seed_source(views):
    # Different bytes for plane, block, token/history, global head/channel and
    # dimension. Integers in [-125, 125] are exact in BF16.
    for member, tensor in enumerate(views):
        indices = torch.arange(tensor.numel(), dtype=torch.int64).reshape(tensor.shape)
        tensor.copy_(((indices * 17 + member * 43) % 251 - 125).to(tensor.dtype))


def as_bytes(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def expected_payload(views, group, block, shard, start):
    k, v, conv, recurrent = views
    if group == 0:
        # Four global KV heads and four Store shards: one head per shard.
        return b"".join(as_bytes(plane[block, start : start + CHUNK, shard, :]) for plane in (k, v))
    pieces = []
    offset = 0
    for width in PROJECTIONS:
        pieces.append(conv[block, :, offset + shard * (width // STORE_TP) : offset + (shard + 1) * (width // STORE_TP)])
        offset += width
    if is_conv_state_dim_first():
        conv_bytes = b"".join(as_bytes(piece.transpose(0, 1)) for piece in pieces)
    else:
        conv_bytes = as_bytes(torch.cat(pieces, dim=1))
    heads = VALUE_HEADS // STORE_TP
    return conv_bytes + as_bytes(recurrent[block, shard * heads : (shard + 1) * heads])


def expected_reader(source, rank, tp):
    k, v, conv, recurrent = source
    heads = KV_HEADS // tp
    channels = []
    offset = 0
    for width in PROJECTIONS:
        channels.append(conv[:, :, offset + rank * (width // tp) : offset + (rank + 1) * (width // tp)])
        offset += width
    return (
        k[:, :, rank * heads : (rank + 1) * heads],
        v[:, :, rank * heads : (rank + 1) * heads],
        torch.cat(channels, dim=2),
        recurrent[:, rank * (VALUE_HEADS // tp) : (rank + 1) * (VALUE_HEADS // tp)],
    )


@pytest.mark.parametrize("reader_tp", [2, 4])
@pytest.mark.parametrize("physical_layout", ["NHD", "HND"])
def test_tp1_payload_to_all_reader_ranks(reader_tp, physical_layout, monkeypatch):
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", physical_layout)
    source, caches = make_worker(1, 0, physical_layout)
    source_views = logical_views(caches, physical_layout)
    seed_source(source_views)
    # Two logical pages, with different physical mappings at each endpoint.
    source_ids, reader_ids = [2, 0], [0, 1]
    chunks = [(0, CHUNK), (CHUNK, BLOCK), (BLOCK, BLOCK + CHUNK)]
    for rank in range(reader_tp):
        reader, reader_caches = make_worker(reader_tp, rank, physical_layout)
        actual = logical_views(reader_caches, physical_layout)
        expected = [tensor.clone() for tensor in actual]
        global_slice = expected_reader(source_views, rank, reader_tp)
        for start, end in chunks:
            page, token = divmod(start, BLOCK)
            for index in range(2):
                expected[index][reader_ids[page], token : token + CHUNK] = global_slice[index][
                    source_ids[page], token : token + CHUNK
                ]
            for index in range(2, 4):
                expected[index][reader_ids[page]] = global_slice[index][source_ids[page]]
        for group, (src_db, dst_db) in enumerate(zip(source.token_dbs, reader.token_dbs)):
            src, dst = src_db.store_layout, dst_db.store_layout
            for shard in dst.local_shard_ids:
                assert src.key_for(shard, b"prefix") == dst.key_for(shard, b"prefix")
                for chunk in chunks:
                    start, end = chunk
                    page, token = divmod(start, BLOCK)
                    # State checkpoints own a whole page regardless of partial tail.
                    selected_chunk = chunk if group == 0 else (page * BLOCK, end)
                    sa, ss, sb = src.prepare_values([selected_chunk], source_ids, [shard])
                    da, ds, db = dst.prepare_values([selected_chunk], reader_ids, [shard])
                    assert sb == [source_ids[page]] and db == [reader_ids[page]]
                    payload = b"".join(ctypes.string_at(address, size) for address, size in zip(sa[0], ss[0]))
                    assert payload == expected_payload(source_views, group, source_ids[page], shard, token)
                    assert sum(ds[0]) == len(payload)
                    cursor = 0
                    for address, size in zip(da[0], ds[0]):
                        ctypes.memmove(address, payload[cursor : cursor + size], size)
                        cursor += size
        for tensor, oracle in zip(actual, expected):
            assert torch.equal(tensor, oracle)  # Also checks untouched tokens/blocks and speculative rows.
