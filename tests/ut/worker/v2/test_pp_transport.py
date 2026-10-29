# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.sequence import IntermediateTensors

from vllm_ascend.worker.v2 import pp_transport
from vllm_ascend.worker.v2.pp_transport import resolve_spec_pp_support


def test_token_shard_send_restores_only_extra_tp_padding():
    tensors = {
        "hidden_states": torch.full((2, 4), 7.0),
        "residual": torch.full((2, 3, 4), 9.0),
        "pp_transport_aux_hidden_states_0": torch.full((3, 4), 11.0),
    }
    pp_transport.zero_pp_shard_padding(tensors, ("hidden_states", "residual"), 3, 2, 1)
    assert tensors["hidden_states"][1].count_nonzero() == 0
    assert tensors["residual"][1].count_nonzero() == 0
    assert tensors["hidden_states"][0].eq(7).all()
    assert tensors["pp_transport_aux_hidden_states_0"].eq(11).all()


def test_token_shard_send_rejects_full_payload():
    with pytest.raises(ValueError, match="expected 2"):
        pp_transport.zero_pp_shard_padding({"hidden_states": torch.empty(3, 4)}, ("hidden_states",), 3, 2, 0)


@pytest.mark.parametrize(
    "indexer_type,use_index_cache,pattern,expected",
    [
        ("full", True, "FFSF", True),
        ("full", True, "FFFF", False),
        ("full", False, "FFSF", False),
        ("shared", False, "FFFF", True),
        ("shared", True, "FFFF", True),
        ("shared", True, "FFSF", True),
    ],
)
def test_topk_boundary_combines_indexshare_and_indexcache(indexer_type, use_index_cache, pattern, expected):
    config = SimpleNamespace(
        num_hidden_layers=4,
        indexer_types=["full", "shared", indexer_type, "shared"],
        use_index_cache=use_index_cache,
        index_topk_pattern=pattern,
    )
    assert pp_transport.pp_stage_requires_topk_indices(config, 2) is expected
    assert not pp_transport.pp_stage_requires_topk_indices(config, 0)
    assert not pp_transport.pp_stage_requires_topk_indices(config, config.num_hidden_layers)


@pytest.mark.parametrize(
    "method,architecture,needs_aux",
    [
        ("mtp", "Qwen3_5ForConditionalGeneration", False),
        ("eagle3", "MiniMaxM3SparseForCausalLM", True),
        ("eagle3", "MiniMaxM3SparseForConditionalGeneration", True),
        ("dspark", "DeepseekV4ForCausalLM", True),
        ("dspark", "GlmMoeDsaForCausalLM", True),
    ],
)
@pytest.mark.parametrize("pp_size", [1, 2])
def test_resolve_spec_pp_support(method, architecture, needs_aux, pp_size):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        model_config=SimpleNamespace(architecture=architecture),
        speculative_config=SimpleNamespace(method=method),
    )
    support = resolve_spec_pp_support(config)
    if pp_size == 1:
        assert support is None
    else:
        assert support.needs_aux_hidden_states is needs_aux


@pytest.mark.parametrize("method", [None, "ngram", "eagle3"])
def test_unregistered_spec_pp_support(method):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        model_config=SimpleNamespace(architecture="OtherModel"),
        speculative_config=SimpleNamespace(method=method) if method else None,
    )
    assert resolve_spec_pp_support(config) is None


def test_aux_buffer_includes_stage_boundary():
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=4), start_layer=2, aux_hidden_state_layers=(0, 2, 4))
    factory = pp_transport.make_empty_intermediate_tensors(
        model, lambda batch, dtype, device: IntermediateTensors({"hidden_states": torch.zeros(batch, 4)})
    )
    tensors = factory(3, torch.float32, torch.device("cpu"))
    aux = pp_transport.get_pp_transport_tensors(tensors, pp_transport.PPTransportDataType.AUX_HIDDEN_STATES)
    assert len(aux) == 2
    assert all(t.shape == (3, 4) for t in aux)


def test_token_shard_buffer_preserves_storage_across_batch_sizes():
    tensors = {
        "hidden_states": torch.arange(8).reshape(4, 2),
        "residual": torch.arange(24).reshape(4, 3, 2),
        "pp_transport_aux_hidden_states_0": torch.arange(16).reshape(8, 2),
        "pp_transport_topk_indices_0": torch.arange(32).reshape(8, 4),
    }
    buffer = pp_transport.PPTokenShardBuffer(IntermediateTensors(dict(tensors)), ("hidden_states", "residual"), 2)

    for num_tokens in (8, 1, 3, 8):
        buffer.activate(num_tokens)
        expected_rows = (num_tokens + 1) // 2
        assert buffer.active["hidden_states"].shape == (expected_rows, 2)
        assert buffer.active["residual"].shape == (expected_rows, 3, 2)
        assert buffer.active["pp_transport_aux_hidden_states_0"].shape == (num_tokens, 2)
        assert buffer.active["pp_transport_topk_indices_0"].shape == (num_tokens, 4)
        for key, tensor in buffer.active.tensors.items():
            assert tensor.data_ptr() == tensors[key].data_ptr()
            torch.testing.assert_close(tensor, tensors[key][: tensor.shape[0]])


def test_token_shard_buffer_fork_shares_storage_without_changing_parent_views():
    buffer = pp_transport.PPTokenShardBuffer(
        IntermediateTensors({"hidden_states": torch.zeros(4, 2), "aux": torch.zeros(8, 2)}),
        ("hidden_states",),
        2,
    )
    buffer.activate(1)
    fork = buffer.fork()
    fork.activate(8)

    assert fork.active is not buffer.active
    assert buffer.active["hidden_states"].shape == (1, 2)
    assert buffer.active["aux"].shape == (1, 2)
    fork.active["hidden_states"].fill_(7)
    fork.active["aux"].fill_(9)
    torch.testing.assert_close(buffer.active["hidden_states"], torch.full((1, 2), 7.0))
    torch.testing.assert_close(buffer.active["aux"], torch.full((1, 2), 9.0))


@pytest.mark.parametrize("limited_key", ["hidden_states", "aux"])
def test_token_shard_buffer_rejects_capacity_overflow_without_replacing_views(limited_key):
    tensors = {"hidden_states": torch.zeros(4, 2), "aux": torch.zeros(8, 2)}
    tensors[limited_key] = tensors[limited_key][:1]
    buffer = pp_transport.PPTokenShardBuffer(IntermediateTensors(tensors), ("hidden_states",), 2)
    buffer.activate(1)
    previous_views = buffer.active.tensors

    with pytest.raises(ValueError, match=f"PP buffer {limited_key} needs"):
        buffer.activate(3)

    assert buffer.active.tensors is previous_views


@pytest.mark.parametrize("tp_size", [0, -1])
def test_token_shard_buffer_rejects_invalid_tp_size(tp_size):
    with pytest.raises(ValueError, match="TP size must be positive"):
        pp_transport.PPTokenShardBuffer(IntermediateTensors({"hidden_states": torch.zeros(1, 2)}), (), tp_size)


def _make_config(
    architecture: str,
    *,
    method: str = "dspark",
    pipeline_parallel_size: int = 2,
):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method=method),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
        ),
        model_config=SimpleNamespace(architecture=architecture),
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "KimiLinearForCausalLM",
        "KimiK3ForCausalLM",
        "KimiK3ForConditionalGeneration",
    ],
)
def test_kimi_k3_dspark_pp_supports_all_target_aliases(architecture):
    support = resolve_spec_pp_support(_make_config(architecture))

    assert support is not None
    assert support.needs_aux_hidden_states


@pytest.mark.parametrize(
    "config",
    [
        _make_config("KimiLinearForCausalLM", pipeline_parallel_size=1),
        _make_config("UnsupportedForPP"),
        _make_config("KimiLinearForCausalLM", method="eagle"),
    ],
)
def test_kimi_k3_dspark_pp_support_stays_scoped(config):
    assert resolve_spec_pp_support(config) is None
