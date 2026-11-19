# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Mooncake pull layout selection, without the NPU runtime."""

import ast
import logging
import math
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake"


class FullAttentionSpec(SimpleNamespace):
    pass


class MLAAttentionSpec(FullAttentionSpec):
    pass


class SlidingWindowSpec(SimpleNamespace):
    pass


class SlidingWindowMLASpec(SlidingWindowSpec):
    pass


class AscendSFAIndexerCacheSpec(MLAAttentionSpec):
    pass


class MambaSpec(SimpleNamespace):
    pass


def load_class(filename, name, methods, dependencies=None):
    tree = ast.parse((SOURCE / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.bases = []
    cls.body = [node for node in cls.body if not isinstance(node, ast.FunctionDef) or node.name in methods]
    scope = dict(
        globals(),
        math=math,
        prod=math.prod,
        dataclass=dataclass,
        is_circular_kv_cache_spec=lambda spec: False,
        cdiv=lambda a, b: -(-a // b),
    )
    scope.update(dependencies or {})
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / filename), "exec"), scope)
    return scope[name]


def test_handshake_merge_preserves_layout_by_layer_and_rejects_rank_mismatch():
    metadata_types = {
        name: load_class("metadata.py", name, {"__post_init__"})
        for name in (
            "MooncakeTransferMetadata",
            "MooncakePPTransferMetadata",
            "MooncakePCPTransferMetadata",
            "MooncakeTPTransferMetadata",
        )
    }
    cls = load_class("base_scheduler.py", "MooncakeMetadataThread", {"_merge_metadata_by_pp_rank"}, metadata_types)
    scheduler = cls()
    scheduler.engine_id = "p"
    scheduler.pp_size = scheduler.pcp_size = 1
    scheduler.tp_size = 2
    scheduler.use_kv_pp = False
    first = metadata_types["MooncakeTransferMetadata"](
        engine_id="p",
        te_rpc_port=1,
        block_size=16,
        num_blocks=4,
        layer_names=["z", "a"],
        layer_block_sizes=[16, 16],
        group_indices=[0, 1],
        kv_caches_base_addr=[[1000], [2000]],
        block_strides=[[128], [128]],
        block_lens=[[128], [128]],
        block_shapes=[[(1, 16, 4)], [(1, 16, 4)]],
        block_size_scales=[[1], [1]],
        cp_kv_cache_interleave_size=16,
        indexer_replication_sizes=[0, 0],
        dcp_sharded=[False, True],
        quant_flashmla_layers=("z",),
    )
    merged = scheduler._merge_metadata_by_pp_rank({0: first, 1: first})[0]
    assert merged.layer_names == ["a", "z"]
    assert merged.dcp_sharded == [True, False]
    assert merged.quant_flashmla_layers == ("z",)
    with pytest.raises(ValueError, match="mismatch in.*dcp_sharded"):
        scheduler._merge_metadata_by_pp_rank({0: first, 1: replace(first, dcp_sharded=[True, True])})

    with pytest.raises(ValueError, match="mismatch in.*quant_flashmla_layout"):
        scheduler._merge_metadata_by_pp_rank({0: first, 1: replace(first, quant_flashmla_layers=())})


@pytest.mark.parametrize("producer_quant,consumer_quant", [(False, False), (True, True), (True, False), (False, True)])
def test_pull_rejects_mismatched_flashmla_layout_before_building_transfer(producer_quant, consumer_quant):
    cls = load_class("pull_worker.py", "MooncakePullRecvingThread", {"_build_remote_transfer_layout"})
    receiver = cls()
    receiver.layer_names = ["draft"]
    receiver.local_metadata = SimpleNamespace(quant_flashmla_layers=("draft",) if consumer_quant else ())
    receiver.spec_indices = [0]
    receiver.kv_cache_specs = [MLAAttentionSpec()]
    calls = []

    def layer_ranks(*args):
        calls.append(args)
        return [[0]]

    receiver._get_layer_remote_tp_rank_groups = layer_ranks
    producer = SimpleNamespace(
        layer_names=["draft"],
        quant_flashmla_layers=("draft",) if producer_quant else (),
        dcp_sharded=[False],
        block_size_scales=[[1]],
    )
    metadata = SimpleNamespace(
        use_kv_pp=False,
        dcp_size=1,
        pcp_size=1,
        tp_size=1,
        metadata_by_pp_rank={0: producer},
    )
    if producer_quant != consumer_quant:
        with pytest.raises(ValueError, match="QuantFlashMLA page layout mismatch.*draft"):
            receiver._build_remote_transfer_layout(metadata)
        assert calls == []
    else:
        assert receiver._build_remote_transfer_layout(metadata) == ({0: {(0, 0): [[0]]}}, {0: [(0, 0)]})
        assert len(calls) == 1


@pytest.mark.parametrize("producer_quant,consumer_capable", [(False, False), (True, False), (True, True)])
def test_pull_handshake_does_not_serve_quant_pages_to_an_old_consumer(producer_quant, consumer_capable):
    metadata_source = ast.parse((SOURCE / "metadata.py").read_text())
    capability = next(
        ast.literal_eval(node.value)
        for node in metadata_source.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "QUANT_FLASHMLA_LAYOUT_CAPABILITY"
    )
    metadata_request = b"get_meta_msg"
    cls = load_class(
        "pull_scheduler.py",
        "MooncakeSchedulerSendingThread",
        {"_run_busy_loop"},
        {
            "msgspec": msgspec,
            "zmq": SimpleNamespace(Again=TimeoutError),
            "logger": logging.getLogger(__name__),
            "GET_META_MSG": metadata_request,
            "QUANT_FLASHMLA_LAYOUT_CAPABILITY": capability,
        },
    )
    producer = cls()
    producer.has_quant_flashmla_layout = producer_quant
    producer.encoded_metadata = b"metadata"
    producer.stop_event = threading.Event()
    message = (metadata_request, capability) if consumer_capable else (metadata_request,)

    def receive():
        producer.stop_event.set()
        return [b"peer", b"", msgspec.msgpack.encode(message)]

    replies: list[tuple[bytes, bytes, bytes]] = []
    producer._run_busy_loop(SimpleNamespace(recv_multipart=receive, send_multipart=replies.append))
    expected = b"" if producer_quant and not consumer_capable else producer.encoded_metadata
    assert replies == [(b"peer", b"", expected)]


@pytest.fixture
def worker():
    cls = load_class(
        "pull_worker.py",
        "MooncakePullRecvingThread",
        {
            "_get_attention_dcp_sizes",
            "_get_layer_remote_tp_rank_groups",
            "_infer_total_num_kv_heads",
            "_get_attention_remote_tp_rank_groups",
            "_get_head_interval",
            "_compute_group_block_ids",
            "_expand_block_ids",
            "_select_remote_tp_rank",
        },
    )
    worker = cls()
    worker.tp_size = 4
    worker.tp_rank = 0
    worker.pcp_size = 1
    worker.pcp_rank = 0
    worker.dcp_size = 2
    worker.dcp_rank = 0
    worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=16))
    return worker


@pytest.mark.parametrize("spec_type", [FullAttentionSpec, MLAAttentionSpec, SlidingWindowSpec, SlidingWindowMLASpec])
@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("local_sharded", [False, True])
@pytest.mark.parametrize("remote_sharded", [False, True])
def test_layout_uses_independent_endpoint_flags(worker, spec_type, heads, local_sharded, remote_sharded):
    spec = spec_type(dcp_sharded=local_sharded)
    worker.block_shapes = [[(heads, 16, 4)]]
    remote = SimpleNamespace(dcp_sharded=[remote_sharded], block_shapes=[[(heads, 16, 4)]])
    assert worker._get_attention_dcp_sizes(spec, 0, remote, 4) == (
        2 if local_sharded else 1,
        4 if remote_sharded else 1,
    )


@pytest.mark.parametrize(
    "local_sharded,remote_sharded,local_heads,remote_heads,expected",
    [
        (False, False, 1, 1, [[0]]),
        (True, True, 2, 2, [[0, 1]]),
        (True, False, 2, 1, [[0], [1]]),
        (False, True, 1, 2, [[0, 1]]),
    ],
)
def test_routing_and_block_mapping_follow_layout(
    worker, local_sharded, remote_sharded, local_heads, remote_heads, expected
):
    spec = FullAttentionSpec(dcp_sharded=local_sharded)
    worker.block_shapes = [[(local_heads, 16, 4)]]
    remote = SimpleNamespace(dcp_sharded=[remote_sharded], block_shapes=[[(remote_heads, 16, 4)]])
    groups = worker._get_layer_remote_tp_rank_groups(0, 0, spec, remote, 1, 4, 2)
    assert groups == expected
    local_size, remote_size = worker._get_attention_dcp_sizes(spec, 0, remote, 2)
    result = worker._compute_group_block_ids(
        "request",
        groups,
        remote_size,
        0,
        16,
        16,
        [10, 11],
        [10, 11],
        [20, 21, 22, 23],
        65 if local_sharded else 33,
        64 if local_sharded else 32,
        0,
        1,
        1,
        spec,
        0,
        remote_interleave_size=16,
        local_dcp_size=local_size,
    )
    if local_sharded and not remote_sharded:
        assert result == [(0, [10, 11], [20, 22]), (1, [10, 11], [20, 22])]
    elif remote_sharded and not local_sharded:
        assert result == [(0, [10], [20]), (1, [11], [20])]
    else:
        assert result == [(0, [10, 11], [20, 21])]


@pytest.mark.parametrize("sharded,expected", [(False, [0, 1, 2, 3]), (True, [0, 1])])
def test_scheduler_prompt_blocks_follow_layout(sharded, expected):
    cls = load_class("base_scheduler.py", "MooncakeBaseConnectorScheduler", {"_get_transfer_block_ids"})
    scheduler = cls()
    scheduler.dcp_size = 2
    scheduler.group_unique_specs = [[FullAttentionSpec(dcp_sharded=sharded)]]
    scheduler.group_block_size = [16]
    assert scheduler._get_transfer_block_ids(([0, 1, 2, 3, 4],), 64) == (expected,)


def test_handshake_rejects_missing_layer_flags():
    cls = load_class("metadata.py", "MooncakeTransferMetadata", {"__post_init__"})
    with pytest.raises(ValueError, match="dcp_sharded"):
        cls(
            engine_id="p",
            te_rpc_port=1,
            block_size=16,
            num_blocks=4,
            layer_names=["layer"],
            layer_block_sizes=[16],
            group_indices=[0],
            kv_caches_base_addr=[[1000]],
            block_strides=[[128]],
            block_lens=[[128]],
            block_shapes=[[(1, 16, 4)]],
            block_size_scales=[[1]],
            cp_kv_cache_interleave_size=16,
            indexer_replication_sizes=[0],
            dcp_sharded=[],
        )


@pytest.mark.parametrize("local_replicas", [1, 2])
@pytest.mark.parametrize("remote_replicas", [1, 2])
@pytest.mark.parametrize("remote_scale", [1, 4])
def test_indexer_replication_uses_explicit_metadata(worker, local_replicas, remote_replicas, remote_scale):
    spec = AscendSFAIndexerCacheSpec(dcp_sharded=True, sfa_dcp_replicated_indexer_size=local_replicas)
    remote = SimpleNamespace(
        dcp_sharded=[True],
        indexer_replication_sizes=[remote_replicas],
        block_size_scales=[[remote_scale]],
    )
    assert worker._get_attention_dcp_sizes(spec, 0, remote, 4) == (
        1 if local_replicas > 1 else 2,
        1 if remote_replicas > 1 else 4,
    )
