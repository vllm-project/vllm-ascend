# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.qwen4_exp import ple_embedding
from vllm_ascend.models.qwen4_exp.ple_embedding import AscendPLEHostEmbedding


def embedding():
    obj = object.__new__(AscendPLEHostEmbedding)
    torch.nn.Module.__init__(obj)
    obj.weight = torch.nn.Parameter(torch.zeros(4, 128, dtype=torch.bfloat16), requires_grad=False)
    obj._shared_group = None
    obj._context_key = "test.ple"
    obj.shard_indices = SimpleNamespace(org_vocab_start_index=10, org_vocab_end_index=14)
    return obj


def test_close_detaches_parameter_and_retains_owner_for_retry():
    obj = embedding()
    owner = Mock()
    owner.close.side_effect = [RuntimeError("free failed"), None]
    obj._host_table = owner
    with pytest.raises(RuntimeError, match="free"):
        obj.close_host_offload()
    assert obj.weight is None and obj._host_table is owner
    obj.close_host_offload()
    obj.close_host_offload()
    assert owner.close.call_count == 2 and obj._host_table is None


@pytest.mark.parametrize("reduce_results", [False, True])
def test_outer_sequence_parallel_controls_reduce(monkeypatch, reduce_results):
    obj = embedding()
    obj._host_table = Mock()
    obj.etp_data_parallel_size = 1
    obj.embedding_dim = 128
    obj.tp_size = 4
    obj.reduce_results = reduce_results
    obj.parallel_group = SimpleNamespace(all_reduce=Mock(side_effect=lambda x: x + 1))
    monkeypatch.setattr(torch.ops.vllm, "ple_gva_lookup", lambda ids, name, width: torch.full((*ids.shape, width), 2))
    result = obj(torch.zeros((3, 2), dtype=torch.int64))
    assert torch.all(result == (3 if reduce_results else 2))
    assert obj.parallel_group.all_reduce.call_count == int(reduce_results)


def test_custom_op_preserves_vocab_range(monkeypatch):
    obj = embedding()
    obj._host_table = Mock()
    monkeypatch.setattr(
        ple_embedding, "get_forward_context", lambda: SimpleNamespace(no_compile_layers={"test.ple": obj})
    )
    ids = torch.tensor([9, 10, 13, 14])
    result = ple_embedding.ple_gva_lookup(ids, "test.ple", 128)
    obj._host_table.lookup.assert_called_once_with(ids, 10, 14)
    assert result is obj._host_table.lookup.return_value


@pytest.mark.skipif(not hasattr(AscendPLEHostEmbedding, "_copy_embedding_shard_"), reason="older upstream loader")
def test_checkpoint_overlap_preserves_row_order():
    obj = embedding()
    source = torch.arange(6 * 128).reshape(6, 128).bfloat16()
    assert obj._copy_embedding_shard_(obj.weight, source, checkpoint_start=8) == 4
    torch.testing.assert_close(obj.weight, source[2:], rtol=0, atol=0)
    assert obj.weight.device.type == "cpu"


@pytest.mark.parametrize("rank", [0, 1])
def test_only_shared_owner_loads_weights(monkeypatch, rank):
    obj = embedding()
    obj._shared_group = SimpleNamespace(rank_in_group=rank, world_size=2, cpu_group=object())
    loader = Mock()
    monkeypatch.setattr(ple_embedding.Qwen4ExpPLEEmbedding, "weight_loader", loader)
    monkeypatch.setattr(ple_embedding.dist, "all_gather_object", lambda *args, **kwargs: None)
    obj.weight_loader(obj.weight, torch.ones_like(obj.weight), 0)
    assert loader.call_count == int(rank == 0)


def test_shared_load_error_propagates(monkeypatch):
    obj = embedding()
    obj._shared_group = SimpleNamespace(rank_in_group=1, world_size=2, cpu_group=object())

    def gather(errors, *args, **kwargs):
        errors[:] = ["bad checkpoint", None]

    monkeypatch.setattr(ple_embedding.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        obj.weight_loader(obj.weight, torch.ones_like(obj.weight))


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("has_layers", [False, True])
@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("draft", [False, True])
def test_qwen_offload_uses_upstream_defaults(enabled, has_layers, dp_size, draft):
    from vllm.config import EngramConfig, VllmConfig

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="Qwen4ExpForConditionalGeneration",
            hf_text_config=SimpleNamespace(ple_layer_ids=[1] if has_layers else []),
        ),
        speculative_config=None,
        engram_config=None if enabled is None else EngramConfig(cpu_offload=enabled),
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            prefill_context_parallel_size=1,
            enable_elastic_ep=False,
        ),
    )
    if draft:
        target = config.model_config
        config.model_config = SimpleNamespace(architecture="Qwen4ExpMTP")
        config.speculative_config = SimpleNamespace(target_model_config=target, draft_model_config=config.model_config)
    if not has_layers and enabled is not None:
        with pytest.raises(ValueError, match="non-empty"):
            VllmConfig._resolve_and_verify_engram_config(config)
    else:
        VllmConfig._resolve_and_verify_engram_config(config)
        if has_layers:
            assert config.engram_config.cpu_offload is (enabled is not False)
            assert config.engram_config.dp_shared_memory is (enabled is not False and dp_size > 1)
        else:
            assert config.engram_config is None
