# SPDX-License-Identifier: Apache-2.0
"""Cross-layer KV sharing through MooncakeConnectorV2.

A KV-sharing layer reads another layer's cache and owns no blocks: it has no
``kv_cache_tensors`` entry, and the runner registers it bound to its owner's
cache. These tests use a plain GQA model whose trailing layers read earlier
layers' KV, and run registration on both peers and the consumer's real layer
matching and address generation. Only the transfer engine is replaced, by one
that copies the byte ranges it is given, so the assertions are on the bytes
that land in the consumer's caches.
"""

import ctypes
import threading
from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from tests.ut.kv_offload.mooncake_v2.helpers import make_kv_cache_tensor
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker, pull_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import MooncakeSchedulerSendingThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import as_kv_cache_tensors

BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 8
NUM_LAYERS = 6
PROMPT_TOKENS = 40  # Two full blocks and a partial third.
PRODUCER_NUM_BLOCKS = 8
CONSUMER_NUM_BLOCKS = 6
SENTINEL = -1


def layer_name(index: int) -> str:
    return f"model.layers.{index}.self_attn.attn"


# The trailing layers read the KV of earlier layers, as in models that share
# KV across layers to shrink the cache.
SHARING_TARGETS = {layer_name(4): layer_name(1), layer_name(5): layer_name(3)}
ALL_LAYERS = [layer_name(index) for index in range(NUM_LAYERS)]


def owners_of(sharing_targets: Mapping[str, str]) -> list[str]:
    return [name for name in ALL_LAYERS if name not in sharing_targets]


def gqa_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        head_size_v=HEAD_SIZE,
        dtype=torch.float16,
    )


def make_gqa_caches(owners: list[str], num_blocks: int, *, fill: bool) -> dict[str, list[torch.Tensor]]:
    """Allocate one peer's K/V caches in HND layout, one pair per owner."""
    shape = (num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)

    def tensor() -> torch.Tensor:
        if not fill:
            return torch.full(shape, SENTINEL, dtype=torch.float16)
        # Small integers are exact in fp16, so equality checks are exact.
        return torch.randint(0, 64, shape).to(torch.float16)

    return {name: [tensor(), tensor()] for name in owners}


def make_kv_cache_config(num_blocks: int, spec, caches: Mapping[str, object]) -> KVCacheConfig:
    """The scheduler's view: owners only, KV-sharing layers appear nowhere."""
    tensors = []
    for name, cache in caches.items():
        size = sum(view.nbytes for view in as_kv_cache_tensors(cache))
        tensors.append(
            make_kv_cache_tensor(size=size, layers=[name], layer_stride=size, block_stride=size // num_blocks)
        )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=[KVCacheGroupSpec(layer_names=list(caches), kv_cache_spec=spec)],
    )


def make_vllm_config(sharing_targets: Mapping[str, str]) -> SimpleNamespace:
    """A config whose static forward context declares the given KV sharing."""
    forward_context = {
        name: SimpleNamespace(kv_sharing_target_layer_name=sharing_targets.get(name)) for name in ALL_LAYERS
    }
    return SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=forward_context),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        model_config=SimpleNamespace(
            is_deepseek_mla=False,
            hf_text_config=SimpleNamespace(num_key_value_heads=NUM_KV_HEADS),
        ),
        speculative_config=None,
    )


def bind_kv_sharing_layers(caches: Mapping[str, object], sharing_targets: Mapping[str, str]) -> dict[str, object]:
    """What the runner registers: ``kv_caches[layer] = kv_caches[owner]``."""
    bound = dict(caches)
    for name, target in sharing_targets.items():
        bound[name] = bound[target]
    return bound


def make_worker(
    engine_id: str,
    num_blocks: int,
    spec,
    caches: Mapping[str, object],
    sharing_targets: Mapping[str, str],
) -> MooncakeBaseConnectorWorker:
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config = make_vllm_config(sharing_targets)
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.pcp_rank, worker.tp_rank, worker.tp_size = 0, 0, 1
    worker.kv_cache_config = make_kv_cache_config(num_blocks, spec, caches)
    worker.engine_id, worker.te_rpc_port = engine_id, 9000
    worker.block_size, worker.side_channel_host, worker.handshake_port = BLOCK_SIZE, "127.0.0.1", 5000
    worker.xfer_handshake_metadata = None
    return worker


@pytest.fixture
def transfer_engine(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    engine = MagicMock()
    monkeypatch.setattr(base_worker, "global_te", engine)
    monkeypatch.setattr(base_worker, "validate_register_region_count", MagicMock())
    return engine


def register_peer(
    engine_id: str,
    num_blocks: int,
    sharing_targets: Mapping[str, str],
    *,
    fill: bool,
) -> tuple[MooncakeBaseConnectorWorker, dict[str, list[torch.Tensor]]]:
    caches = make_gqa_caches(owners_of(sharing_targets), num_blocks, fill=fill)
    worker = make_worker(engine_id, num_blocks, gqa_spec(), caches, sharing_targets)
    worker.register_kv_caches(bind_kv_sharing_layers(caches, sharing_targets))
    return worker, caches


class CopyingTransferEngine:
    """Stands in for a Mooncake READ: copy each remote range into its local one."""

    def __init__(self) -> None:
        self.num_bytes = 0
        self._lock = threading.Lock()

    def batch_transfer_sync_read(
        self, session_id: str, local_addrs: list[int], remote_addrs: list[int], lengths: list[int]
    ) -> int:
        for local_addr, remote_addr, length in zip(local_addrs, remote_addrs, lengths):
            ctypes.memmove(local_addr, remote_addr, length)
        with self._lock:
            self.num_bytes += sum(lengths)
        return 0


def make_consumer_thread(worker: MooncakeBaseConnectorWorker, engine: object) -> MooncakePullRecvingThread:
    # The arguments MooncakePullConnectorWorker.register_kv_caches passes.
    return MooncakePullRecvingThread(
        engine=engine,
        vllm_config=worker.vllm_config,
        kv_cache_config=worker.kv_cache_config,
        kv_cache_specs=worker.kv_cache_specs,
        layer_name_to_group_index=worker.layer_name_to_group_index,
        layer_name_to_spec_index=worker.layer_name_to_spec_index,
        local_metadata=worker.xfer_handshake_metadata,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_rank=0,
        dp_size=1,
        pcp_rank=0,
        pcp_size=1,
        dcp_rank=0,
        dcp_size=1,
        device=0,
        ready_event=threading.Event(),
    )


def serve_producer_metadata(monkeypatch: pytest.MonkeyPatch, producer: MooncakeBaseConnectorWorker) -> None:
    """Answer the consumer's metadata request with what the producer publishes."""
    sending = MooncakeSchedulerSendingThread(
        host="127.0.0.1",
        port=6000,
        engine_id="engine-p",
        metadata={0: producer.xfer_handshake_metadata},
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        use_kv_pp=False,
        ready_event=threading.Event(),
    )
    context = MagicMock()
    context.__enter__.return_value = MagicMock()
    monkeypatch.setattr(pull_worker, "zmq_ctx", MagicMock(return_value=context))
    monkeypatch.setattr(pull_worker, "ensure_zmq_send", MagicMock())
    monkeypatch.setattr(pull_worker, "ensure_zmq_recv", MagicMock(return_value=sending.encoded_metadata))


PRODUCER_BLOCKS = [5, 2, 6]
CONSUMER_BLOCKS = [1, 4, 3]


def pull_one_request(
    monkeypatch: pytest.MonkeyPatch,
    producer: MooncakeBaseConnectorWorker,
    consumer: MooncakeBaseConnectorWorker,
) -> CopyingTransferEngine:
    serve_producer_metadata(monkeypatch, producer)
    engine = CopyingTransferEngine()
    thread = make_consumer_thread(consumer, engine)
    request = ReqMeta(
        local_block_ids=(CONSUMER_BLOCKS,),
        local_num_prompt_tokens=PROMPT_TOKENS,
        num_external_tokens=PROMPT_TOKENS,
        num_computed_tokens=0,
        remote_block_ids=(PRODUCER_BLOCKS,),
        remote_host="127.0.0.1",
        remote_port=6000,
        remote_engine_id="engine-p",
        remote_request_id="request-p",
        remote_num_prompt_tokens=PROMPT_TOKENS,
        local_full_block_ids=(CONSUMER_BLOCKS,),
    )
    try:
        failed = thread._handle_requests("engine-p", "127.0.0.1", 6000, {"request-d": request})
    finally:
        thread.executor.shutdown(wait=True)
    assert failed == set()
    return engine


def assert_prompt_blocks_equal(local: list[torch.Tensor], remote: list[torch.Tensor], name: str) -> None:
    for local_cache, remote_cache in zip(local, remote, strict=True):
        for consumer_block, producer_block in zip(CONSUMER_BLOCKS, PRODUCER_BLOCKS):
            assert torch.equal(local_cache[consumer_block], remote_cache[producer_block]), (name, consumer_block)


def test_pull_transfers_each_owner_once_and_borrowers_read_it(
    monkeypatch: pytest.MonkeyPatch, transfer_engine: MagicMock
) -> None:
    torch.manual_seed(0)
    producer, producer_caches = register_peer("engine-p", PRODUCER_NUM_BLOCKS, SHARING_TARGETS, fill=True)
    consumer, consumer_caches = register_peer("engine-d", CONSUMER_NUM_BLOCKS, SHARING_TARGETS, fill=False)

    engine = pull_one_request(monkeypatch, producer, consumer)

    expected_bytes = 0
    for name in owners_of(SHARING_TARGETS):
        assert_prompt_blocks_equal(consumer_caches[name], producer_caches[name], name)
        for cache in consumer_caches[name]:
            untouched = [block for block in range(CONSUMER_NUM_BLOCKS) if block not in CONSUMER_BLOCKS]
            assert torch.all(cache[untouched] == SENTINEL), name
            expected_bytes += len(CONSUMER_BLOCKS) * cache[0].nbytes
    # A borrower adds no reads of its own.
    assert engine.num_bytes == expected_bytes
    # Read through each borrower's own binding: it is its owner's cache, so it
    # attends to the producer's owner KV.
    for borrower, owner in SHARING_TARGETS.items():
        assert_prompt_blocks_equal(consumer.kv_caches[borrower], producer_caches[owner], borrower)


def test_pull_accepts_a_layer_that_only_the_consumer_shares(
    monkeypatch: pytest.MonkeyPatch, transfer_engine: MagicMock
) -> None:
    """The consumer reads its owner's KV; the producer's own copy of the layer is not pulled."""
    torch.manual_seed(0)
    consumer_only = layer_name(5)
    producer_targets = {name: target for name, target in SHARING_TARGETS.items() if name != consumer_only}
    producer, producer_caches = register_peer("engine-p", PRODUCER_NUM_BLOCKS, producer_targets, fill=True)
    consumer, consumer_caches = register_peer("engine-d", CONSUMER_NUM_BLOCKS, SHARING_TARGETS, fill=False)
    assert consumer_only in producer.xfer_handshake_metadata.layer_names

    engine = pull_one_request(monkeypatch, producer, consumer)

    block_bytes = consumer_caches[layer_name(0)][0][0].nbytes
    assert engine.num_bytes == len(owners_of(SHARING_TARGETS)) * 2 * len(CONSUMER_BLOCKS) * block_bytes
    owner = SHARING_TARGETS[consumer_only]
    assert_prompt_blocks_equal(consumer.kv_caches[consumer_only], producer_caches[owner], consumer_only)


def test_pull_rejects_a_layer_that_only_the_producer_shares(
    monkeypatch: pytest.MonkeyPatch, transfer_engine: MagicMock
) -> None:
    """The consumer owns a cache for the layer, but the producer publishes no KV for it."""
    producer_only = layer_name(5)
    consumer_targets = {name: target for name, target in SHARING_TARGETS.items() if name != producer_only}
    producer, _ = register_peer("engine-p", PRODUCER_NUM_BLOCKS, SHARING_TARGETS, fill=True)
    consumer, _ = register_peer("engine-d", CONSUMER_NUM_BLOCKS, consumer_targets, fill=False)
    serve_producer_metadata(monkeypatch, producer)
    thread = make_consumer_thread(consumer, CopyingTransferEngine())

    try:
        with pytest.raises(ValueError, match="missing layers required by this worker") as error:
            thread._get_remote_metadata("engine-p", "127.0.0.1", 6000)
    finally:
        thread.executor.shutdown(wait=True)
    assert producer_only in str(error.value)
