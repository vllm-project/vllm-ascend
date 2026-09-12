# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.sequence import IntermediateTensors

import vllm_ascend.patch.worker.patch_deepseek_v2 as patch_deepseek_v2
import vllm_ascend.worker.v2.pp_utils as pp_utils
from vllm_ascend.patch.worker.patch_deepseek_v2 import (
    _deepseek_v2_model_init_with_pp_topk_transport,
    _patched_forward,
    _should_skip_indexer_init,
)
from vllm_ascend.worker.v2.pp_utils import (
    PPTransportDataType,
    add_pp_transport_tensors,
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
    model = SimpleNamespace()

    _deepseek_v2_model_init_with_pp_topk_transport(
        model,
        vllm_config=SimpleNamespace(),
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
    return IntermediateTensors(
        {"hidden_states": torch.zeros((batch_size, 4), dtype=dtype, device=device)}
    )


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
    monkeypatch.setattr(
        pp_utils, "should_reuse_topk", lambda config, layer_idx: layer_idx in {2, 4}
    )
    topk_buffer = torch.zeros((8, 2), dtype=torch.int32)
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=8, use_index_cache=True),
        start_layer=2, end_layer=4, topk_indices_buffer=topk_buffer,
    )
    factory = make_empty_intermediate_tensors(
        model, _empty_tensor_factory, (PPTransportDataType.TOPK_INDICES,)
    )
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
        config.indexer_types = [
            "full", "full", "shared", "full", "shared", "full", "full", "full"
        ]
    else:
        config.use_index_cache = True
        monkeypatch.setattr(
            pp_utils, "should_reuse_topk", lambda config, layer_idx: layer_idx in {2, 4}
        )
    model = SimpleNamespace(
        config=config, start_layer=2, end_layer=4,
        aux_hidden_state_layers=(1, 2, 5), topk_indices_buffer=topk_buffer,
    )
    factory = make_empty_intermediate_tensors(
        model, _empty_tensor_factory,
        (PPTransportDataType.AUX_HIDDEN_STATES, PPTransportDataType.TOPK_INDICES),
    )
    result = factory(3, torch.bfloat16, torch.device("cpu"))
    assert model.receive_pp_topk_indices and model.send_pp_topk_indices
    aux = get_pp_transport_tensors(result, PPTransportDataType.AUX_HIDDEN_STATES)
    topk = get_pp_transport_tensors(result, PPTransportDataType.TOPK_INDICES)
    assert len(aux) == 2 and len(topk) == 1
    assert topk[0].data_ptr() == topk_buffer.data_ptr()
