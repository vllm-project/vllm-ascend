# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.sequence import IntermediateTensors

import vllm_ascend.patch.worker.patch_deepseek_v2 as patch_deepseek_v2
import vllm_ascend.worker.v2.pp_transport as pp_transport
from vllm_ascend.patch.worker.patch_deepseek_v2 import (
    _patched_deepseek_v2_model_init,
    _patched_forward,
    _should_skip_indexer_init,
)
from vllm_ascend.worker.v2.pp_transport import (
    PPTransportDataType,
    add_pp_transport_tensors,
    configure_pp_topk_transport,
    get_pp_transport_tensors,
    make_empty_intermediate_tensors,
)


def _config(**overrides) -> SimpleNamespace:
    values = {"num_hidden_layers": 80}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_glm51_skip_topk_keeps_per_layer_indexer():
    assert not _should_skip_indexer_init(
        _config(),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_glm52_shared_layer_skips_indexer_init():
    assert _should_skip_indexer_init(
        _config(indexer_types=["full", "full", "shared"]),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_mtp_layer_keeps_indexer():
    indexer_types = ["full"] * 80 + ["shared"]
    assert not _should_skip_indexer_init(
        _config(indexer_types=indexer_types),
        "model.layers.80.self_attn",
        skip_topk=True,
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("boundaries", [(0, 78), (0, 42, 78), (0, 20, 40, 59, 78)])
def test_aux_relay_matches_unpartitioned_forward(monkeypatch, native, boundaries):
    if native and not hasattr(DeepseekV2Model, "pack_local_aux_hidden_states"):
        pytest.skip("The installed vLLM release has no native aux relay")
    aux_layers = (0, 2, 20, 39, 58, 75, 78)
    ids = torch.zeros(4, dtype=torch.long)

    def layer(positions, hidden, residual, scaling):
        residual = torch.zeros_like(hidden) if residual is None else residual
        return hidden + 1, residual + 1

    def run(split):
        incoming = None
        for start, end in zip(split, split[1:]):
            first, last = start == 0, end == 78
            group = SimpleNamespace(is_first_rank=first, is_last_rank=last, world_size=len(split) - 1)
            monkeypatch.setattr(patch_deepseek_v2, "get_pp_group", lambda group=group: group)
            model = DeepseekV2Model.__new__(DeepseekV2Model)
            torch.nn.Module.__init__(model)
            model.config = SimpleNamespace(hidden_size=2)
            model.hidden_size = 2
            model.start_layer, model.end_layer = start, end
            model.layers = [layer] * 78
            model.aux_hidden_state_layers = aux_layers
            model._use_upstream_aux_relay = native
            model.embed_input_ids = lambda ids: torch.zeros(len(ids), 2)
            model.norm = lambda hidden, residual: (hidden + residual, None)
            if native:
                import vllm.distributed.parallel_state as parallel_state
                from vllm.v1.worker.gpu.pp_utils import PPHandler

                monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
                monkeypatch.setattr(parallel_state, "get_pp_group", lambda group=group: group)
                model._set_aux_hidden_state_layers(aux_layers)
            output = patch_deepseek_v2._patched_forward(model, ids if first else None, ids, incoming)
            if not last:
                if native:
                    # Use the actual upstream relay, without creating streams or process groups.
                    handler = SimpleNamespace(
                        aux_hidden_state_relay_keys=[
                            f"aux_hidden_states_{i}" for i in range(model._aux_slot_base_cached)
                        ]
                    )
                    output = PPHandler.relay_aux_hidden_states(handler, incoming, output)
                prefix = "aux_hidden_states_" if native else "pp_transport_aux_hidden_states_"
                expected_count = sum(idx <= end for idx in aux_layers)
                assert set(output.tensors) == {"hidden_states", "residual"} | {
                    f"{prefix}{idx}" for idx in range(expected_count)
                }
            incoming = output
        return output

    expected_hidden, expected_aux = run((0, 78))
    actual_hidden, actual_aux = run(boundaries)
    torch.testing.assert_close(actual_hidden, expected_hidden)
    assert len(actual_aux) == len(expected_aux) == len(aux_layers)
    for actual, expected in zip(actual_aux, expected_aux):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("legacy,v2", [(True, True), (False, True), (False, False)])
def test_aux_buffer_factory_uses_one_protocol(monkeypatch, legacy, v2):
    factory = object()
    model = SimpleNamespace(layers=[], make_empty_intermediate_tensors=factory)
    monkeypatch.setattr(patch_deepseek_v2, "_original_deepseek_v2_model_init", lambda *args, **kwargs: None)
    monkeypatch.setattr(patch_deepseek_v2.pp_transport, "use_legacy_spec_pp", lambda: legacy)
    wrapped = object()
    wrap_factory = Mock(return_value=wrapped)
    monkeypatch.setattr(patch_deepseek_v2.pp_transport, "make_empty_intermediate_tensors", wrap_factory)
    patch_deepseek_v2._patched_deepseek_v2_model_init(model, vllm_config=SimpleNamespace(use_v2_model_runner=v2))
    assert model._use_upstream_aux_relay is (v2 and not legacy)
    expected_data_types: tuple[PPTransportDataType, ...] = (PPTransportDataType.TOPK_INDICES,)
    if not model._use_upstream_aux_relay:
        expected_data_types = (PPTransportDataType.AUX_HIDDEN_STATES, *expected_data_types)
    wrap_factory.assert_called_once_with(model, factory, expected_data_types)
    assert model.make_empty_intermediate_tensors is wrapped


@pytest.mark.parametrize("native", [False, True])
def test_aux_setter_preserves_v1_behavior(native):
    if not hasattr(patch_deepseek_v2, "_set_aux_hidden_state_layers"):
        pytest.skip("The installed vLLM release uses its original setter")
    model = SimpleNamespace(_use_upstream_aux_relay=native, _set_aux_hidden_state_layers=Mock())
    layers = (2, 20, 39, 58, 75)
    patch_deepseek_v2._set_aux_hidden_state_layers(SimpleNamespace(model=model), layers)
    if native:
        model._set_aux_hidden_state_layers.assert_called_once_with(layers)
    else:
        model._set_aux_hidden_state_layers.assert_not_called()
        assert model.aux_hidden_state_layers == layers


def test_model_init_adds_pp_topk_receive_buffer(monkeypatch):
    topk_indices_buffer = torch.zeros((8, 2), dtype=torch.int32)

    def original_init(model, *, vllm_config, prefix):
        model.layers = [SimpleNamespace(self_attn=SimpleNamespace(topk_indices_buffer=topk_indices_buffer))]
        model.config = _config(
            num_hidden_layers=8,
            indexer_types=["full", "shared", "shared", "shared"] * 2,
        )
        model.start_layer = 2
        model.end_layer = 4
        model.make_empty_intermediate_tensors = lambda batch_size, dtype, device: IntermediateTensors(
            {"hidden_states": torch.zeros((batch_size, 4), dtype=dtype, device=device)}
        )

    monkeypatch.setattr(
        patch_deepseek_v2,
        "_original_deepseek_v2_model_init",
        original_init,
    )
    monkeypatch.setattr(patch_deepseek_v2.pp_transport, "use_legacy_spec_pp", lambda: False)
    model = SimpleNamespace()

    _patched_deepseek_v2_model_init(
        model,
        vllm_config=SimpleNamespace(use_v2_model_runner=True),
    )
    intermediate_tensors = model.make_empty_intermediate_tensors(
        3,
        torch.bfloat16,
        torch.device("cpu"),
    )

    receive_buffers = get_pp_transport_tensors(
        intermediate_tensors,
        PPTransportDataType.TOPK_INDICES,
    )
    assert len(receive_buffers) == 1
    assert receive_buffers[0].shape == (3, 2)
    assert receive_buffers[0].dtype == torch.int32
    assert receive_buffers[0].data_ptr() == topk_indices_buffer.data_ptr()


def test_pp_forward_propagates_aliased_topk_indices(monkeypatch):
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    monkeypatch.setattr(patch_deepseek_v2, "get_pp_group", lambda: pp_group)
    topk_indices_buffer = torch.zeros((4, 2), dtype=torch.int32)
    received_topk_indices = topk_indices_buffer[:2]
    received_topk_indices.copy_(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
    model = SimpleNamespace(
        _use_upstream_aux_relay=False,
        receive_pp_topk_indices=True,
        send_pp_topk_indices=True,
        topk_indices_buffer=topk_indices_buffer,
        config=SimpleNamespace(llama_4_scaling=None),
        layers=[lambda positions, hidden_states, residual, scaling: (hidden_states, residual)],
        start_layer=0,
        end_layer=1,
        aux_hidden_state_layers=(),
    )
    intermediate_tensors = add_pp_transport_tensors(
        IntermediateTensors(
            {
                "hidden_states": torch.ones((2, 4)),
                "residual": torch.zeros((2, 4)),
            }
        ),
        PPTransportDataType.TOPK_INDICES,
        [received_topk_indices],
    )

    output = _patched_forward(
        model,
        input_ids=None,
        positions=torch.arange(2),
        intermediate_tensors=intermediate_tensors,
    )

    assert isinstance(output, IntermediateTensors)
    torch.testing.assert_close(
        model.topk_indices_buffer[:2],
        received_topk_indices,
    )
    transported = get_pp_transport_tensors(
        output,
        PPTransportDataType.TOPK_INDICES,
    )
    assert len(transported) == 1
    torch.testing.assert_close(transported[0], received_topk_indices)


def _empty_tensor_factory(batch_size, dtype, device):
    return IntermediateTensors({"hidden_states": torch.zeros((batch_size, 4), dtype=dtype, device=device)})


def test_make_empty_intermediate_tensors_aux_only():
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=4),
        start_layer=2,
        aux_hidden_state_layers=(1, 2, 5),
    )
    factory = make_empty_intermediate_tensors(model, _empty_tensor_factory)
    result = factory(3, torch.bfloat16, torch.device("cpu"))
    buffers = get_pp_transport_tensors(result, PPTransportDataType.AUX_HIDDEN_STATES)
    assert len(buffers) == 2
    assert all(buffer.shape == (3, 4) for buffer in buffers)


def test_make_empty_intermediate_tensors_indexcache_only(monkeypatch):
    monkeypatch.setattr(pp_transport, "should_reuse_topk", lambda config, layer_idx: layer_idx in {2, 4})
    topk_buffer = torch.zeros((8, 2), dtype=torch.int32)
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=8, use_index_cache=True),
        start_layer=2,
        end_layer=4,
        topk_indices_buffer=topk_buffer,
    )
    transport_data_types = (PPTransportDataType.TOPK_INDICES,)
    configure_pp_topk_transport(model, transport_data_types)
    factory = make_empty_intermediate_tensors(model, _empty_tensor_factory, transport_data_types)
    result = factory(3, torch.bfloat16, torch.device("cpu"))
    assert model.receive_pp_topk_indices and model.send_pp_topk_indices
    buffers = get_pp_transport_tensors(result, PPTransportDataType.TOPK_INDICES)
    assert len(buffers) == 1
    assert buffers[0].data_ptr() == topk_buffer.data_ptr()


@pytest.mark.parametrize("topk_mode", ["indexshare", "indexcache"])
def test_make_empty_intermediate_tensors_aux_and_topk(monkeypatch, topk_mode):
    topk_buffer = torch.zeros((8, 2), dtype=torch.int32)
    config = SimpleNamespace(num_hidden_layers=8, hidden_size=4)
    if topk_mode == "indexshare":
        config.indexer_types = ["full", "full", "shared", "full", "shared", "full", "full", "full"]
    else:
        config.use_index_cache = True
        monkeypatch.setattr(pp_transport, "should_reuse_topk", lambda config, layer_idx: layer_idx in {2, 4})
    model = SimpleNamespace(
        config=config,
        start_layer=2,
        end_layer=4,
        aux_hidden_state_layers=(1, 2, 5),
        topk_indices_buffer=topk_buffer,
    )
    transport_data_types = (
        PPTransportDataType.AUX_HIDDEN_STATES,
        PPTransportDataType.TOPK_INDICES,
    )
    configure_pp_topk_transport(model, transport_data_types)
    factory = make_empty_intermediate_tensors(
        model,
        _empty_tensor_factory,
        transport_data_types,
    )
    result = factory(3, torch.bfloat16, torch.device("cpu"))
    assert model.receive_pp_topk_indices and model.send_pp_topk_indices
    aux = get_pp_transport_tensors(result, PPTransportDataType.AUX_HIDDEN_STATES)
    topk = get_pp_transport_tensors(result, PPTransportDataType.TOPK_INDICES)
    assert len(aux) == 2 and len(topk) == 1
    assert topk[0].data_ptr() == topk_buffer.data_ptr()
