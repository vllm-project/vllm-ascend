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
    _deepseek_v2_decoder_layer_forward,
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


@pytest.mark.parametrize("non_causal", [None, False, True])
def test_mla_constructor_preserves_decode_mask_flag(monkeypatch, non_causal):
    for name in ("DeepSeekV2FusedQkvAProjLinear", "ColumnParallelLinear", "RowParallelLinear", "RMSNorm"):
        monkeypatch.setattr(patch_deepseek_v2, name, lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(patch_deepseek_v2, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(patch_deepseek_v2, "get_rope", lambda *args, **kwargs: torch.nn.Identity())
    wrapper = Mock(return_value=torch.nn.Identity())
    monkeypatch.setattr(patch_deepseek_v2, "MultiHeadLatentAttentionWrapper", wrapper)
    options = {} if non_causal is None else {"non_causal_multi_token_decode": non_causal}
    attention = torch.nn.Module()
    patch_deepseek_v2._deepseek_v2_mla_attention_init(
        attention,
        vllm_config=SimpleNamespace(),
        config=_config(rms_norm_eps=1e-5, rope_parameters={"rope_type": "default"}),
        hidden_size=32,
        num_heads=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        q_lora_rank=16,
        kv_lora_rank=8,
        prefix="model.layers.0.self_attn",
        **options,
    )
    assert wrapper.call_args.kwargs["non_causal_multi_token_decode"] is (non_causal is True)


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


def test_single_token_sp_decoder_gathers_rank_local_state(monkeypatch):
    gather_calls = []

    def all_gather(hidden_states, dim):
        gather_calls.append(tuple(hidden_states.shape))
        return torch.cat([hidden_states, hidden_states + 1], dim=dim)

    monkeypatch.setattr(patch_deepseek_v2, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(patch_deepseek_v2, "tensor_model_parallel_all_gather", all_gather)
    monkeypatch.setattr(patch_deepseek_v2, "tensor_model_parallel_reduce_scatter", lambda value, dim: value[:1])

    def norm(hidden_states, residual=None):
        return hidden_states, hidden_states if residual is None else residual

    def attention(_positions, hidden_states, _scaling=None):
        return hidden_states

    layer = SimpleNamespace(
        use_sequence_parallel_moe=True,
        use_mha=False,
        input_layernorm=norm,
        post_attention_layernorm=norm,
        self_attn=attention,
        mlp=lambda hidden_states, **_kwargs: hidden_states,
        routed_scaling_factor=1.0,
        layer_idx=1,
    )
    hidden_states = torch.tensor([[2.0]])
    residual = torch.tensor([[1.0]])

    output, output_residual = _deepseek_v2_decoder_layer_forward(
        layer,
        positions=torch.tensor([0]),
        hidden_states=hidden_states,
        residual=residual,
        input_is_sequence_parallel=True,
    )

    assert gather_calls == [(1, 1)]
    torch.testing.assert_close(output, hidden_states)
    torch.testing.assert_close(output_residual, residual)


def test_forward_gathers_explicit_sp_state_when_local_rows_equal_full(monkeypatch):
    gather_calls = []
    group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(patch_deepseek_v2, "get_pp_group", lambda: group)

    def all_gather(hidden_states, dim):
        gather_calls.append(tuple(hidden_states.shape))
        return torch.cat([hidden_states, hidden_states + 1], dim=dim)

    monkeypatch.setattr(patch_deepseek_v2, "tensor_model_parallel_all_gather", all_gather)

    def layer(_positions, hidden_states, residual, _scaling):
        return hidden_states + 1, torch.zeros_like(hidden_states) if residual is None else residual

    layer.use_sequence_parallel_moe = True
    model = SimpleNamespace(
        config=SimpleNamespace(llama_4_scaling=None),
        hidden_size=2,
        layers=[layer],
        start_layer=0,
        end_layer=1,
        aux_hidden_state_layers=(),
        _use_upstream_aux_relay=False,
        send_pp_topk_indices=False,
        embed_input_ids=lambda ids: torch.ones((len(ids), 2)),
        norm=lambda hidden_states, residual: (hidden_states + residual, None),
    )

    output = _patched_forward(model, torch.tensor([1]), torch.tensor([0]), None)

    assert gather_calls == [(1, 4)]
    torch.testing.assert_close(output, torch.full((1, 2), 2.0))


@pytest.mark.parametrize("num_tokens", [1, 3, 4])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("input_is_sp", [False, True])
def test_sp_decoder_keeps_residual_aligned(monkeypatch, num_tokens, rank, input_is_sp):
    import vllm.model_executor.models.utils as model_utils

    monkeypatch.setattr(patch_deepseek_v2, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(model_utils, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(model_utils, "get_tensor_model_parallel_rank", lambda: rank)
    # Exercise upstream's padding/chunk implementation, including the padded
    # rank when one token is distributed over two ranks.
    monkeypatch.setattr(patch_deepseek_v2, "sequence_parallel_chunk", model_utils.sequence_parallel_chunk_impl)
    full = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4)[:, ::2]
    local = model_utils.sequence_parallel_chunk_impl(full)
    gather_calls = []

    def gather(value, dim):
        gather_calls.append(tuple(value.shape))
        return torch.nn.functional.pad(full, (0, 0, 0, num_tokens % 2))

    def norm(value, residual=None):
        if residual is not None:
            assert value.shape == residual.shape
        return value, value if residual is None else residual

    layer = SimpleNamespace(
        use_sequence_parallel_moe=True,
        use_mha=False,
        input_layernorm=lambda value, residual=None: value if residual is None else (value, residual),
        post_attention_layernorm=norm,
        self_attn=lambda positions, value, scaling: value,
        mlp=lambda value, **kwargs: value,
        routed_scaling_factor=1.0,
        layer_idx=1,
    )
    monkeypatch.setattr(patch_deepseek_v2, "tensor_model_parallel_all_gather", gather)

    def reduce_scatter(value, dim):
        assert value.is_contiguous()
        return value.chunk(2, dim)[rank]

    monkeypatch.setattr(patch_deepseek_v2, "tensor_model_parallel_reduce_scatter", reduce_scatter)
    output, residual = _deepseek_v2_decoder_layer_forward(
        layer,
        torch.arange(num_tokens),
        local if input_is_sp else full,
        local if input_is_sp else None,
        input_is_sequence_parallel=input_is_sp,
    )
    assert len(gather_calls) == int(input_is_sp)
    torch.testing.assert_close(output, local)
    torch.testing.assert_close(residual, local)


@pytest.mark.parametrize("sp_layouts", [(False, True, True), (True, False), (True, True)])
def test_forward_passes_layout_across_decoder_boundaries(monkeypatch, sp_layouts):
    group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(patch_deepseek_v2, "get_pp_group", lambda: group)
    monkeypatch.setattr(patch_deepseek_v2, "tensor_model_parallel_all_gather", lambda value, dim: value.repeat(2, 1))
    seen = []

    def decoder(layer, positions, hidden, residual, scaling, *, input_is_sequence_parallel):
        seen.append(input_is_sequence_parallel)
        assert hidden.shape[0] == (2 if input_is_sequence_parallel else 4)
        if layer.use_sequence_parallel_moe:
            hidden = hidden[:2]
        return hidden, torch.zeros_like(hidden)

    monkeypatch.setattr(patch_deepseek_v2, "_deepseek_v2_decoder_layer_forward", decoder)
    layers = []
    for enabled in sp_layouts:
        layer = patch_deepseek_v2.DeepseekV2DecoderLayer.__new__(patch_deepseek_v2.DeepseekV2DecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.use_sequence_parallel_moe = enabled
        layers.append(layer)
    model = SimpleNamespace(
        config=SimpleNamespace(llama_4_scaling=None),
        hidden_size=2,
        layers=layers,
        start_layer=0,
        end_layer=len(layers),
        aux_hidden_state_layers=tuple(range(len(layers) + 1)),
        _use_upstream_aux_relay=False,
        send_pp_topk_indices=False,
        embed_input_ids=lambda ids: torch.ones((len(ids), 2)),
        norm=lambda hidden, residual: (hidden + residual, None),
    )
    output, auxiliary = _patched_forward(model, torch.arange(4), torch.arange(4), None)
    assert seen == [False] + [previous and current for previous, current in zip(sp_layouts, sp_layouts[1:])]
    torch.testing.assert_close(output, torch.ones((4, 2)))
    assert all(value.shape == (4, 2) for value in auxiliary)


@pytest.mark.parametrize("input_is_sp", [False, True])
def test_sp_decoder_compiles_with_explicit_layout(monkeypatch, input_is_sp):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(patch_deepseek_v2, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(patch_deepseek_v2, "tensor_model_parallel_all_gather", lambda value, dim: value.repeat(2, 1))
    monkeypatch.setattr(patch_deepseek_v2, "tensor_model_parallel_reduce_scatter", lambda value, dim: value[:1])
    monkeypatch.setattr(patch_deepseek_v2, "sequence_parallel_chunk", lambda value: value[:1])
    layer = SimpleNamespace(
        use_sequence_parallel_moe=True,
        use_mha=False,
        input_layernorm=lambda value, residual: (value + residual, residual),
        post_attention_layernorm=lambda value, residual: (value, residual),
        self_attn=lambda positions, value, scaling: value,
        mlp=lambda value, **kwargs: value,
        routed_scaling_factor=1.0,
        layer_idx=1,
    )
    args = (layer, torch.tensor([0]), torch.ones((1, 2)), torch.ones((1, 2)))
    torch._dynamo.reset()
    try:
        with torch.inference_mode():
            expected = _deepseek_v2_decoder_layer_forward(*args, input_is_sequence_parallel=input_is_sp)
            actual = torch.compile(_deepseek_v2_decoder_layer_forward, backend="eager", fullgraph=True)(
                *args, input_is_sequence_parallel=input_is_sp
            )
        torch.testing.assert_close(actual, expected)
    finally:
        torch._dynamo.reset()


def test_sp_decoder_export_preserves_padding_for_odd_runtime_batch(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(patch_deepseek_v2, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        patch_deepseek_v2, "tensor_model_parallel_reduce_scatter", lambda value, dim: value.reshape(2, -1, 2)[0]
    )
    monkeypatch.setattr(
        patch_deepseek_v2,
        "sequence_parallel_chunk",
        lambda value: torch.nn.functional.pad(value, (0, 0, 0, (-value.shape[0]) % 2)).reshape(2, -1, 2)[0],
    )
    layer = SimpleNamespace(
        use_sequence_parallel_moe=True,
        use_mha=False,
        input_layernorm=lambda value, residual: (value, residual),
        post_attention_layernorm=lambda value, residual: (value, residual),
        self_attn=lambda positions, value, scaling: value,
        mlp=lambda value, **kwargs: value,
        routed_scaling_factor=1.0,
        layer_idx=1,
    )

    def forward(positions, hidden):
        return _deepseek_v2_decoder_layer_forward(layer, positions, hidden, hidden)

    torch._dynamo.reset()
    try:
        # npugraph_ex can reuse the graph built from an even profiling batch
        # for an odd prefill batch. Run that graph directly to exercise the
        # symbolic padding, independently of Dynamo's Python recompile guards.
        exported = torch._dynamo.export(forward, aten_graph=True)(torch.arange(4), torch.ones((4, 2)))
        output, residual = exported.graph_module(torch.arange(3), torch.ones((3, 2)))
        expected = torch.ones((2, 2))
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(residual, expected)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("last_rank", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_forward_compilation_does_not_trigger_buffer_mutation_guard(monkeypatch, last_rank, native):
    # CPU UTs mock the NPU backend, but Dynamo's stream tracking calls the
    # real accelerator API. This test only compiles CPU tensor operations.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    group = SimpleNamespace(is_first_rank=True, is_last_rank=last_rank)
    monkeypatch.setattr(patch_deepseek_v2, "get_pp_group", lambda: group)
    model = SimpleNamespace(
        config=SimpleNamespace(llama_4_scaling=None),
        layers=[lambda positions, hidden, residual, scaling: (hidden + 1, hidden)],
        start_layer=0,
        end_layer=1,
        aux_hidden_state_layers=(0,),
        _use_upstream_aux_relay=native,
        send_pp_topk_indices=False,
        embed_input_ids=lambda ids: ids.float().unsqueeze(-1),
        norm=lambda hidden, residual: (hidden + residual, None),
        pack_local_aux_hidden_states=lambda states: {"aux_hidden_states_0": states[0]},
    )
    ids = torch.arange(4)
    compiled_codes = []

    def capture_bytecode(old_code, new_code):
        if old_code is _patched_forward.__code__:
            compiled_codes.append(new_code)

    torch._dynamo.reset()
    handle = torch._dynamo.convert_frame.register_bytecode_hook(capture_bytecode)
    try:
        with torch.inference_mode():
            expected = _patched_forward(model, ids, ids, None)
            actual = torch.compile(_patched_forward, backend="eager", fullgraph=True)(model, ids, ids, None)
        assert compiled_codes
        # vLLM checks co_names, including names retained from traced-out branches.
        assert all("update" not in code.co_names for code in compiled_codes)
        if last_rank:
            torch.testing.assert_close(actual, expected)
        else:
            torch.testing.assert_close(actual.tensors, expected.tensors)
    finally:
        handle.remove()
        torch._dynamo.reset()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("boundaries", [(0, 78), (0, 38, 78), (0, 42, 78), (0, 20, 40, 59, 78)])
@pytest.mark.parametrize("aux_layers", [(0, 2, 20, 39, 58, 75, 78), (2, 22, 38, 58, 74)])
def test_aux_relay_matches_unpartitioned_forward(monkeypatch, native, boundaries, aux_layers):
    if native and not hasattr(DeepseekV2Model, "pack_local_aux_hidden_states"):
        pytest.skip("The installed vLLM release has no native aux relay")
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
            model.send_pp_topk_indices = False
            model.embed_input_ids = lambda ids: torch.zeros(len(ids), 2)
            model.norm = lambda hidden, residual: (hidden + residual, None)
            if native:
                import vllm.distributed.parallel_state as parallel_state
                from vllm.v1.worker.gpu.pp_utils import PPHandler

                monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
                monkeypatch.setattr(parallel_state, "get_pp_group", lambda group=group: group)
                model._set_aux_hidden_state_layers(aux_layers)
                # IDs are boundaries after a target layer. The state at a
                # PP cut belongs to the upstream stage, including boundary38.
                expected_incoming = 0 if first else sum(idx <= start for idx in aux_layers)
                assert model._aux_slot_base_cached == expected_incoming
                assert model._aux_upstream_total_cached == (expected_incoming if last else 0)
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


@pytest.mark.parametrize("v2", [False, True])
def test_aux_buffer_factory_uses_one_protocol(monkeypatch, v2):
    factory = object()
    model = SimpleNamespace(layers=[], make_empty_intermediate_tensors=factory)
    monkeypatch.setattr(patch_deepseek_v2, "_original_deepseek_v2_model_init", lambda *args, **kwargs: None)
    wrapped = object()
    wrap_factory = Mock(return_value=wrapped)
    monkeypatch.setattr(patch_deepseek_v2.pp_transport, "make_empty_intermediate_tensors", wrap_factory)
    patch_deepseek_v2._patched_deepseek_v2_model_init(model, vllm_config=SimpleNamespace(use_v2_model_runner=v2))
    assert model._use_upstream_aux_relay is v2
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
    model = SimpleNamespace()

    _patched_deepseek_v2_model_init(
        model,
        vllm_config=SimpleNamespace(use_v2_model_runner=True),
        prefix="",
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


@pytest.mark.parametrize("topk_mode", ["indexshare", "indexcache", "indexshare_with_indexcache"])
def test_make_empty_intermediate_tensors_aux_and_topk(monkeypatch, topk_mode):
    topk_buffer = torch.zeros((8, 2), dtype=torch.int32)
    config = SimpleNamespace(num_hidden_layers=8, hidden_size=4)
    if topk_mode == "indexshare":
        config.indexer_types = ["full", "full", "shared", "full", "shared", "full", "full", "full"]
    elif topk_mode == "indexcache":
        config.use_index_cache = True
        monkeypatch.setattr(pp_transport, "should_reuse_topk", lambda config, layer_idx: layer_idx in {2, 4})
    else:
        config.indexer_types = ["full", "shared", "full", "shared", "full", "shared", "full", "shared"]
        config.use_index_cache = True
        config.index_topk_pattern = "FFSFSFSF"
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
