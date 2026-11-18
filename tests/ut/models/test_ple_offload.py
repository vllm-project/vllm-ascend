# SPDX-License-Identifier: Apache-2.0
"""PLE host ownership and TP loading regressions without allocating an NPU table."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.qwen4_exp.ple_embedding import AscendPLEHostEmbedding


def embedding():
    obj = object.__new__(AscendPLEHostEmbedding)
    torch.nn.Module.__init__(obj)
    obj.weight = torch.nn.Parameter(torch.zeros(4, 160, dtype=torch.bfloat16), requires_grad=False)
    obj._shared_group = None
    obj._host_scale = None
    obj._mxfp8 = False
    obj._quantized = False
    obj.etp_data_parallel_size = 1
    obj.shard_indices = SimpleNamespace(org_vocab_start_index=10, org_vocab_end_index=14)
    return obj


def test_close_is_idempotent_and_invalidates_parameter():
    obj = embedding()
    owner = Mock()
    obj._host_table = owner
    obj.close_host_offload()
    obj.close_host_offload()
    owner.close.assert_called_once()
    assert obj.weight is None and obj._host_table is None
    with pytest.raises(RuntimeError, match="closed"):
        obj.forward(torch.zeros((1, 16), dtype=torch.int64))


def test_failed_close_preserves_owner_for_retry():
    obj = embedding()
    owner = Mock()
    owner.close.side_effect = [RuntimeError("unregister failed"), None]
    obj._host_table = owner
    with pytest.raises(RuntimeError, match="unregister"):
        obj.close_host_offload()
    assert obj.weight is None and obj._host_table is owner
    obj.close_host_offload()
    assert obj.weight is None and owner.close.call_count == 2


@pytest.mark.parametrize("reduce_results", [True, False])
@pytest.mark.parametrize("format_", ["bf16", "int8", "mxfp8"])
def test_backend_reduces_only_when_outer_sp_does_not(monkeypatch, reduce_results, format_):
    from vllm_ascend.ops.triton.qwen4_exp import ple_lookup, ple_lookup_quantized

    obj = embedding()
    obj._host_table = SimpleNamespace(ptrs=None)
    obj._mxfp8 = format_ == "mxfp8"
    obj._quantized = format_ != "bf16"
    if obj._quantized:
        obj._host_scale = SimpleNamespace(ptrs=None)
        obj.weight_scale = torch.nn.Parameter(torch.ones(4, 5), requires_grad=False)
    obj.embedding_dim = 160
    obj.tp_size = 4
    obj.reduce_results = reduce_results
    obj.parallel_group = SimpleNamespace(all_reduce=Mock(side_effect=lambda x: x + 1))
    monkeypatch.setattr(ple_lookup, "lookup", lambda ptrs, ids, out, *args: out.fill_(2))
    monkeypatch.setattr(
        ple_lookup_quantized, "lookup_quantized", lambda ptrs, scales, ids, out, *args, **kwargs: out.fill_(2)
    )
    output = obj(torch.zeros((3, 16), dtype=torch.int64))
    assert torch.all(output == (3 if reduce_results else 2))
    assert obj.parallel_group.all_reduce.call_count == int(reduce_results)


@pytest.mark.parametrize("failure", ["unregister", "free"])
def test_host_buffer_release_failure_is_retryable(failure):
    import ctypes

    from vllm_ascend.models.deepseek_v41.engram.npu import HostUvaBuffer

    buffer = object.__new__(HostUvaBuffer)
    buffer.pointer = ctypes.c_void_p(65536)
    buffer._allocation_pointer = ctypes.c_void_p(64000)
    buffer._registered = True
    buffer.ptrs = torch.zeros(1, dtype=torch.int64)
    buffer.tensor = torch.zeros(1)
    buffer.buffer = object()
    buffer.lib = SimpleNamespace(
        aclrtHostUnregister=Mock(side_effect=[1, 0] if failure == "unregister" else [0]),
        aclrtFreeHost=Mock(side_effect=[1, 0] if failure == "free" else [0]),
    )
    with pytest.raises(RuntimeError, match="failed"):
        buffer.close()
    assert buffer.tensor is not None and buffer.pointer.value
    buffer.close()
    buffer.close()
    assert buffer.tensor is None and not buffer.pointer.value
    assert buffer.lib.aclrtHostUnregister.call_count == (2 if failure == "unregister" else 1)
    assert buffer.lib.aclrtFreeHost.call_count == (2 if failure == "free" else 1)


@pytest.mark.parametrize("rank", [0, 1])
def test_shared_loader_has_one_writer(monkeypatch, rank):
    from vllm_ascend.models.qwen4_exp import ple_embedding

    obj = embedding()
    obj._shared_group = SimpleNamespace(rank_in_group=rank, world_size=4, cpu_group=object())
    source = torch.arange(6 * 160).reshape(6, 160).bfloat16()
    synchronized = []
    monkeypatch.setattr(
        ple_embedding.dist,
        "all_gather_object",
        lambda errors, error, group: synchronized.append(error),
    )
    obj.weight_loader(obj.weight, source, checkpoint_start=8)
    torch.testing.assert_close(obj.weight, source[2:] if rank == 0 else torch.zeros_like(obj.weight))
    assert synchronized == [None]


def test_shared_loader_propagates_peer_failure(monkeypatch):
    from vllm_ascend.models.qwen4_exp import ple_embedding

    obj = embedding()
    obj._shared_group = SimpleNamespace(rank_in_group=1, world_size=4, cpu_group=object())

    def gather(errors, error, group):
        errors[0] = "checkpoint failed"

    monkeypatch.setattr(ple_embedding.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="checkpoint failed"):
        obj.weight_loader(obj.weight, torch.zeros_like(obj.weight), checkpoint_start=10)


def test_oot_constructs_backend_without_rebinding_upstream(monkeypatch):
    from vllm.models.qwen4_exp.amd import ple_layer
    from vllm.models.qwen4_exp.common.ngram_embedding import (
        Qwen4ExpPLEDeviceEmbedding,
        Qwen4ExpPLEEmbedding,
        Qwen4ExpPLEPinnedHostEmbedding,
    )

    from vllm_ascend.models.qwen4_exp import ple_embedding

    monkeypatch.setattr(ple_embedding, "get_current_vllm_config", lambda: SimpleNamespace(engram_config=None))
    initialized = []

    def initialize(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        initialized.append(self)

    monkeypatch.setattr(Qwen4ExpPLEEmbedding, "__init__", initialize)
    assert ple_layer.Qwen4ExpPLEPinnedHostEmbedding is Qwen4ExpPLEPinnedHostEmbedding
    obj = ple_layer.Qwen4ExpPLEPinnedHostEmbedding()
    assert type(obj) is AscendPLEHostEmbedding
    assert initialized == [obj]
    assert obj.supports_prefetch is False
    device = Qwen4ExpPLEDeviceEmbedding()
    assert type(device) is Qwen4ExpPLEDeviceEmbedding


@pytest.mark.parametrize("buffer_constructor_failed", [False, True])
def test_constructor_cleanup_failure_retains_host_owner(monkeypatch, buffer_constructor_failed):
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding

    from vllm_ascend.models.deepseek_v41.engram.npu import HostUvaBufferCleanupError
    from vllm_ascend.models.qwen4_exp import ple_embedding

    monkeypatch.setattr(ple_embedding, "get_current_vllm_config", lambda: SimpleNamespace(engram_config=None))
    owner = Mock()
    owner.close.side_effect = [RuntimeError("unregister failed"), None]
    instances = []

    def initialize(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        instances.append(self)
        if buffer_constructor_failed:
            raise HostUvaBufferCleanupError(owner, ValueError("allocation failed"), RuntimeError("unregister failed"))
        self._host_table = owner
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        raise ValueError("construction failed")

    monkeypatch.setattr(Qwen4ExpPLEEmbedding, "__init__", initialize)
    with pytest.raises(HostUvaBufferCleanupError) as caught:
        AscendPLEHostEmbedding()
    assert caught.value.host_buffer is owner
    assert instances[0]._host_table is owner
    if buffer_constructor_failed:
        owner.close.assert_not_called()
        owner.close.side_effect = [None]
    else:
        owner.close.assert_called_once()
    instances[0].close_host_offload()
    assert instances[0]._host_table is None


def test_load_scope_retains_successful_and_failed_constructors(monkeypatch):
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding

    from vllm_ascend.models.host_offload import track_host_offload_owners
    from vllm_ascend.models.qwen4_exp import ple_embedding

    monkeypatch.setattr(ple_embedding, "get_current_vllm_config", lambda: SimpleNamespace(engram_config=None))
    buffers = [Mock(), Mock()]

    def initialize(self, fail=False):
        torch.nn.Module.__init__(self)
        self._host_table = buffers[int(fail)]
        if fail:
            raise ValueError("later layer failed")

    monkeypatch.setattr(Qwen4ExpPLEEmbedding, "__init__", initialize)
    with track_host_offload_owners() as owners:
        first = AscendPLEHostEmbedding()
        with pytest.raises(ValueError, match="later layer"):
            AscendPLEHostEmbedding(fail=True)
        assert len(owners) == 2 and owners[0] is first
        for owner in owners:
            owner.close_host_offload()
    for buffer in buffers:
        buffer.close.assert_called_once()


def test_lookup_accepts_keyword_ids_and_reuses_output(monkeypatch):
    from vllm_ascend.ops.triton.qwen4_exp import ple_lookup

    obj = embedding()
    obj.embedding_dim = 160
    obj._host_table = SimpleNamespace(ptrs=None)
    ids = torch.zeros((3, 16), dtype=torch.int64)
    output = torch.empty((3, 16, 160), dtype=torch.bfloat16)
    lookup = Mock(side_effect=lambda ptrs, ids, out, *args: out.fill_(2))
    monkeypatch.setattr(ple_lookup, "lookup", lookup)
    result = obj._lookup(input_ids=ids, output=output)
    assert result.shape == (3, 16, 160) and result.dtype == torch.bfloat16
    assert torch.all(result == 2)
    assert result is output
    lookup.assert_called_once()


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "device"])
def test_lookup_rejects_incompatible_output(monkeypatch, mismatch):
    from vllm_ascend.ops.triton.qwen4_exp import ple_lookup

    obj = embedding()
    obj.embedding_dim = 160
    obj._host_table = SimpleNamespace(ptrs=None)
    ids = torch.zeros((3, 16), dtype=torch.int64)
    output = torch.empty(
        (3, 16, 159) if mismatch == "shape" else (3, 16, 160),
        dtype=torch.float32 if mismatch == "dtype" else torch.bfloat16,
        device="meta" if mismatch == "device" else "cpu",
    )
    lookup = Mock()
    monkeypatch.setattr(ple_lookup, "lookup", lookup)
    with pytest.raises(ValueError, match="input shape, lookup dtype, and input device"):
        obj._lookup(input_ids=ids, output=output)
    lookup.assert_not_called()
