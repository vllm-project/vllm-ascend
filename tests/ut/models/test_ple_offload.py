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


@pytest.mark.parametrize("sp", [False, True])
def test_mtp_hc_sp_shards_decoder_and_restores_both_outputs(monkeypatch, sp):
    from vllm_ascend.models.qwen4_exp import mtp

    calls = []

    def layer(**kwargs):
        hidden = kwargs["hidden_states"]
        calls.append(hidden.shape[0])
        return hidden, None, None

    model = SimpleNamespace(
        hc_count=2,
        hidden_size=4,
        num_mtp_layers=1,
        use_hc_sequence_parallel=sp,
        layers=[layer],
        embed_input_ids=lambda ids: torch.ones(ids.numel(), 4),
        pre_fc_norm_embedding=torch.nn.Identity(),
        fc_embedding=torch.nn.Identity(),
        pre_fc_norm_hidden=torch.nn.Identity(),
        fc_hidden=torch.nn.Identity(),
        hyper_connection_mixer=SimpleNamespace(combine_and_mix=lambda h, b, i: (h, h[:, :4], None)),
    )
    monkeypatch.setattr(mtp, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(mtp, "sp_shard", lambda h: h[:2])
    monkeypatch.setattr(mtp, "sp_all_gather", lambda h: torch.cat([h, h]))
    sample, multi = mtp.AscendQwen4ExpMultiTokenPredictor.forward(
        model, torch.ones(4, dtype=torch.int64), torch.arange(4), torch.ones(4, 8)
    )
    assert calls == [2 if sp else 4]
    assert sample.shape == (4, 4) and multi.shape == (4, 8)
    assert sample.is_contiguous() and multi.is_contiguous() if sp else True
    torch.testing.assert_close(sample, torch.full((4, 4), 2.0))
    torch.testing.assert_close(multi, torch.full((4, 8), 2.0))


@pytest.mark.parametrize(
    "field",
    [
        "data_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "use_ubatching",
        "enable_elastic_ep",
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
    setattr(parallel, field, 2 if field.endswith("size") else True)
    config = SimpleNamespace(
        engram_config=EngramConfig(cpu_offload=True, dp_shared_memory=False),
        model_config=SimpleNamespace(architecture="Qwen4ExpForConditionalGeneration"),
        speculative_config=None,
        parallel_config=parallel,
        use_v2_model_runner=True,
    )
    with pytest.raises(ValueError, match="Qwen PLE offload requires"):
        _validate_engram_config(config)


@pytest.mark.parametrize("architecture", ["Qwen4ExpForConditionalGeneration", "Qwen4ExpMTP", "OtherModel"])
def test_hc_sp_validation_preserves_caller_config(monkeypatch, architecture):
    from vllm_ascend.patch.platform import patch_engram_config as patch

    config = SimpleNamespace(
        architecture=architecture,
        architectures=[architecture],
        registry=SimpleNamespace(is_hc_sp_supported_model=Mock(return_value=True)),
    )
    parallel = SimpleNamespace(enable_hc_sp=True, pipeline_parallel_size=1)
    original = Mock()
    monkeypatch.setattr(patch, "_original_verify_parallel", original)
    patch.verify_qwen_hc_parallel(config, parallel)
    checked = original.call_args.args[1]
    assert parallel.enable_hc_sp
    assert checked.enable_hc_sp is (architecture == "OtherModel")
    assert (checked is parallel) is (architecture == "OtherModel")


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


@pytest.mark.parametrize("target", ["mtp_shard", "ple_shard", "ple_reduce_scatter"])
def test_compiled_sp_handles_padded_and_unpadded_batches(monkeypatch, target):
    from vllm.models.common.ops import sequence_parallel as common_sp
    from vllm.models.qwen4_exp.amd import ple_layer
    from vllm.models.qwen4_exp.amd.ops import sequence_parallel as model_sp

    from vllm_ascend.models.qwen4_exp import mtp

    monkeypatch.setattr(common_sp, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(common_sp, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(model_sp, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(common_sp, "_custom_collective", lambda *args: None)
    monkeypatch.setattr(common_sp, "tensor_model_parallel_reduce_scatter", lambda x, dim: x.chunk(4)[3].clone())
    operation = {
        "mtp_shard": mtp.sp_shard,
        "ple_shard": ple_layer.sp_shard,
        "ple_reduce_scatter": ple_layer.sp_reduce_scatter,
    }[target]
    op_name = "sp_reduce_scatter" if target == "ple_reduce_scatter" else "sp_shard"
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    library = torch.library.Library("vllm", "IMPL", "CPU")
    try:
        library.impl("sp_shard", model_sp._sp_shard_op)
        library.impl("sp_reduce_scatter", common_sp.sp_reduce_scatter)
        compiled = torch.compile(operation, backend=backend, dynamic=True, fullgraph=True)
        for rows in (8, 2, 5, 12):
            tensor = torch.arange(rows * 3).reshape(rows, 3)
            expected = torch.nn.functional.pad(tensor, (0, 0, 0, (-rows) % 4))
            expected = expected.chunk(4)[3]
            torch.testing.assert_close(compiled(tensor), expected)
        # Padding must execute inside the opaque op, not specialize during tracing.
        assert graphs and all(f"vllm.{op_name}" in graph.code for graph in graphs)
    finally:
        library._destroy()


@pytest.mark.parametrize("moe_sp", [False, True])
def test_hc_does_not_reduce_complete_ascend_moe_output_twice(monkeypatch, moe_sp):
    from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpSparseMoeBlock

    def initialize(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.is_sequence_parallel = moe_sp
        self.experts = SimpleNamespace(moe_config=SimpleNamespace(skip_final_all_reduce=True))

    monkeypatch.setattr(AscendQwen4ExpSparseMoeBlock.__bases__[0], "__init__", initialize)
    layer = AscendQwen4ExpSparseMoeBlock()
    assert layer.experts.moe_config.skip_final_all_reduce is moe_sp
