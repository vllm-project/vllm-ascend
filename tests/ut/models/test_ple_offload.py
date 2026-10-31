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


def test_failed_close_preserves_owner_and_alias_for_retry():
    obj = embedding()
    param = obj.weight
    owner = Mock()
    owner.close.side_effect = [RuntimeError("unregister failed"), None]
    obj._host_table = owner
    with pytest.raises(RuntimeError, match="unregister"):
        obj.close_host_offload()
    assert obj.weight is param and obj._host_table is owner
    obj.close_host_offload()
    assert obj.weight is None and owner.close.call_count == 2


@pytest.mark.skipif(
    not hasattr(AscendPLEHostEmbedding, "_copy_embedding_shard_"),
    reason="_copy_embedding_shard_ not available in this vLLM version",
)
def test_checkpoint_overlap_keeps_cpu_target_and_row_order():
    obj = embedding()
    source = torch.arange(6 * 160).reshape(6, 160).bfloat16()
    copied = obj._copy_embedding_shard_(obj.weight, source, checkpoint_start=8)
    assert copied == 4 and obj.weight.device.type == "cpu"
    torch.testing.assert_close(obj.weight, source[2:], rtol=0, atol=0)
    assert obj._copy_embedding_shard_(obj.weight, source, checkpoint_start=20) == 0


@pytest.mark.parametrize("reduce_results", [True, False])
def test_backend_reduces_only_when_outer_sp_does_not(monkeypatch, reduce_results):
    from vllm_ascend.ops.triton.qwen4_exp import ple_lookup

    obj = embedding()
    obj._host_table = SimpleNamespace(ptrs=None)
    obj.etp_data_parallel_size = 1
    obj.embedding_dim = 160
    obj.tp_size = 4
    obj.reduce_results = reduce_results
    obj.parallel_group = SimpleNamespace(all_reduce=Mock(side_effect=lambda x: x + 1))
    monkeypatch.setattr(ple_lookup, "lookup", lambda ptrs, ids, out, *args: out.fill_(2))
    output = obj(torch.zeros((3, 16), dtype=torch.int64))
    assert torch.all(output == (3 if reduce_results else 2))
    assert obj.parallel_group.all_reduce.call_count == int(reduce_results)


@pytest.mark.parametrize("enabled", [None, False, True])
def test_qwen_offload_remains_opt_in(monkeypatch, enabled):
    from vllm.config import EngramConfig

    from vllm_ascend.patch.platform import patch_engram_config as patch

    config = SimpleNamespace(
        model_config=SimpleNamespace(architecture="Qwen4ExpForConditionalGeneration"),
        speculative_config=None,
        engram_config=None if enabled is None else EngramConfig(cpu_offload=enabled),
    )
    original = Mock()
    monkeypatch.setattr(patch, "_original_resolve_engram", original)
    patch.resolve_engram_config(config)
    assert config.engram_config.cpu_offload is (enabled is True)
    original.assert_called_once_with(config)


def test_constructor_failure_releases_allocated_table(monkeypatch):
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding

    from vllm_ascend.models.qwen4_exp import ple_embedding

    monkeypatch.setattr(ple_embedding, "get_current_vllm_config", lambda: SimpleNamespace(engram_config=None))
    owner = Mock()

    def fail(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self._host_table = owner
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        raise ValueError("construction failed")

    monkeypatch.setattr(Qwen4ExpPLEEmbedding, "__init__", fail)
    with pytest.raises(ValueError, match="construction failed"):
        AscendPLEHostEmbedding()
    owner.close.assert_called_once()


@pytest.mark.parametrize(
    "field",
    [
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "dp_shared_memory",
        "use_thp",
        "enable_elastic_ep",
        "use_ubatching",
    ],
)
def test_offload_rejects_unsupported_topology(field):
    from vllm.config import EngramConfig

    from vllm_ascend.platform import _validate_engram_config

    parallel = SimpleNamespace(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        use_ubatching=False,
        enable_elastic_ep=False,
    )
    engram_kwargs = dict(cpu_offload=True, dp_shared_memory=False)
    # dp_shared_memory and use_thp are engram_config attributes, not parallel
    if field in ("dp_shared_memory", "use_thp", "embedding_across_dp"):
        engram_kwargs[field] = True
    else:
        setattr(parallel, field, 2 if field.endswith("size") else True)
    config = SimpleNamespace(
        engram_config=EngramConfig(**engram_kwargs),
        model_config=SimpleNamespace(architecture="Qwen4ExpForConditionalGeneration"),
        speculative_config=None,
        parallel_config=parallel,
        use_v2_model_runner=True,
    )
    with pytest.raises(ValueError, match="Qwen PLE offload requires"):
        _validate_engram_config(config)


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


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
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
