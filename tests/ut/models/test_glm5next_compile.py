# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.compilation import decorators
from vllm.compilation.backends import split_graph
from vllm.config.compilation import CUDAGraphMode, DynamicShapesType
from vllm.models.common.ops import sequence_parallel
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.glm5next import kda, mtp
from vllm_ascend.models.glm5next import model as glm_model
from vllm_ascend.models.glm5next.model import Glm5NextModel
from vllm_ascend.models.glm5next.mtp import Glm5NextMTP


@pytest.mark.parametrize("input_kind", ["ids", "embeddings", "intermediate"])
@pytest.mark.parametrize("model_class", [Glm5NextModel, Glm5NextMTP])
def test_compile_marks_token_dimensions_dynamic(monkeypatch, input_kind, model_class):
    # This checks the decorator's marking contract before compilation. Dynamic
    # execution across token counts is exercised by the compiled tests below.
    num_tokens = 32
    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.do_not_compile = False
    model.compiled = False
    model.compilation_config = SimpleNamespace(dynamic_shapes_config=SimpleNamespace(type=DynamicShapesType.BACKED))
    monkeypatch.setattr(decorators.envs, "VLLM_USE_AOT_COMPILE", False)
    monkeypatch.setattr(decorators, "is_forward_context_available", lambda: False)

    class ReadyToCompile(Exception):
        pass

    def stop_before_compilation():
        raise ReadyToCompile

    monkeypatch.setattr(model, "original_code_object", stop_before_compilation)
    positions = torch.arange(num_tokens)
    ids = torch.zeros(num_tokens, dtype=torch.long) if input_kind == "ids" else None
    embeddings = torch.zeros(num_tokens, 8) if input_kind == "embeddings" else None
    intermediate = None
    tensors = [positions]
    if input_kind == "intermediate":
        intermediate = IntermediateTensors(
            {
                "hidden_states": torch.zeros(num_tokens, 8),
                "residual": torch.zeros(num_tokens, 4, 8),
                "post": torch.zeros(num_tokens, 4, 1),
                "comb": torch.zeros(num_tokens, 4, 4),
            }
        )
        tensors.extend(intermediate.tensors.values())
    tensors.extend(t for t in (ids, embeddings) if t is not None)
    kwargs = dict(input_ids=ids, positions=positions, intermediate_tensors=intermediate, inputs_embeds=embeddings)
    if model_class is Glm5NextMTP:
        kwargs["hidden_states"] = torch.zeros(num_tokens, 8)
        tensors.append(kwargs["hidden_states"])
    with pytest.raises(ReadyToCompile):
        model(**kwargs)
    for tensor in tensors:
        assert tensor._dynamo_dynamic_indices == {0}


def test_compiled_kda_reads_runtime_context_and_writes_output(monkeypatch):
    # CPU tracing must not query streams from the shared UT NPU mocks.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    calls = []
    context = SimpleNamespace(no_compile_layers={})

    def dispatch(qkv, g1, beta, output):
        calls.append(context.value)
        output.copy_(qkv + g1 + beta + context.value)

    def norm(output, gate, *, out):
        assert not torch.compiler.is_compiling()
        assert out is output
        out.copy_(output * gate + context.value)
        return out

    context.no_compile_layers["kda"] = SimpleNamespace(_forward=dispatch, o_norm=SimpleNamespace(forward_oot=norm))
    monkeypatch.setattr(kda, "get_forward_context", lambda: context)

    def forward(qkv, g1, beta):
        output = torch.empty_like(qkv)
        result = torch.ops.vllm.glm5next_kda_forward(qkv, g1, beta, g1, output, "kda")
        assert result is None
        return output * 2

    # The production kernel dispatches to NPU. Register the same entry point on
    # CPU only for this test, so Dynamo and AOTAutograd run without device mocks.
    with torch.library._scoped_library("vllm", "IMPL", "CPU") as library:
        library.impl("glm5next_kda_forward", kda.glm5next_kda_forward)
        compiled = torch.compile(forward, backend="aot_eager", fullgraph=True, dynamic=True)
        for num_tokens, value in [(32, 0), (129, 3), (1, 7), (32, 11)]:
            context.value = value
            inputs = [torch.ones(num_tokens, 8) for _ in range(3)]
            torch.testing.assert_close(compiled(*inputs), torch.full_like(inputs[0], (3 + 2 * value) * 2))
    assert calls == [0, 3, 7, 11]


def test_compiled_mtp_norm_calls_runtime_kernel_for_dynamic_inputs(monkeypatch):
    # CPU tracing must not query streams from the shared UT NPU mocks.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    calls = []

    def kernel(positions, embeddings, previous, enorm, hnorm, eps):
        assert not torch.compiler.is_compiling()
        calls.append(positions.shape[0])
        return torch.cat((embeddings * enorm + positions[:, None], previous * hnorm), dim=-1)

    monkeypatch.setattr(mtp, "fused_eh_norm", kernel)

    def forward(positions, embeddings, previous, enorm, hnorm):
        return torch.ops.vllm.glm5next_mtp_eh_norm(positions, embeddings, previous, enorm, hnorm, 1e-6)

    with torch.library._scoped_library("vllm", "IMPL", "CPU") as library:
        library.impl("glm5next_mtp_eh_norm", mtp.glm5next_mtp_eh_norm)
        compiled = torch.compile(forward, backend="aot_eager", fullgraph=True, dynamic=True)
        for count in (32, 129, 1, 513):
            positions = torch.arange(count)
            inputs = torch.ones(count, 4)
            output = compiled(positions, inputs, inputs * 2, torch.ones(4), torch.full((4,), 3.0))
            torch.testing.assert_close(output[:, :4], inputs + positions[:, None])
            torch.testing.assert_close(output[:, 4:], torch.full_like(inputs, 6.0))
    assert calls == [32, 129, 1, 513]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mtp_norm_fake_matches_runtime_metadata(monkeypatch, dtype):
    # The CPU kernel preserves the real operator's output shape, dtype and
    # allocation contract; numerical normalization is covered on NPU.
    def kernel(positions, embeddings, previous, enorm, hnorm, eps):
        return torch.cat((embeddings, previous), dim=-1)

    monkeypatch.setattr(mtp, "fused_eh_norm", kernel)
    args = (
        torch.arange(3),
        torch.ones(3, 4, dtype=dtype),
        torch.full((3, 4), 2.0, dtype=dtype),
        torch.ones(4, dtype=dtype),
        torch.ones(4, dtype=dtype),
        1e-6,
    )
    with torch.library._scoped_library("vllm", "IMPL", "CPU") as library:
        library.impl("glm5next_mtp_eh_norm", mtp.glm5next_mtp_eh_norm)
        torch.library.opcheck(
            torch.ops.vllm.glm5next_mtp_eh_norm.default,
            args,
            test_utils=("test_schema", "test_faketensor"),
        )


@pytest.fixture
def make_kda_layer(monkeypatch):
    def base_init(self, config, vllm_config, prefix):
        torch.nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.tp_size = 1
        self.quant_config = None

    class Linear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(128, 4))

    monkeypatch.setattr(kda.GatedDeltaNetAttention, "__init__", base_init)
    for name in ("_Glm5NextMergedColumnParallelLinear", "ColumnParallelLinear", "RowParallelLinear"):
        monkeypatch.setattr(kda, name, Linear)
    # Import the device-dependent constructor dependencies only after CPU UT
    # mocks are installed; retain the real model's compilation registration.
    from vllm.third_party.flash_linear_attention.ops import kda as upstream_kda

    from vllm_ascend import utils

    monkeypatch.setattr(utils, "register_ascend_fla_customop", lambda: None)
    monkeypatch.setattr(upstream_kda, "FusedRMSNormGated", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(kda, "is_conv_state_dim_first", lambda: False)
    config = SimpleNamespace(
        hidden_size=16, linear_head_dim=128, linear_num_heads=1, linear_conv_kernel_dim=4, linear_lower_bound=-4.0
    )

    def construct(compilation_config, prefix):
        vllm_config = SimpleNamespace(
            compilation_config=compilation_config,
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            quant_config=None,
            speculative_config=SimpleNamespace(num_speculative_tokens=5),
        )
        monkeypatch.setattr(kda, "get_current_vllm_config", lambda: vllm_config)
        return kda.Glm5NextLinearAttention(config, vllm_config, prefix)

    return construct


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY])
def test_kda_preserves_full_graph_compilation(make_kda_layer, mode):
    compilation_config = SimpleNamespace(cudagraph_mode=mode, splitting_ops=[], static_forward_context={})
    make_kda_layer(compilation_config, "layer")
    assert compilation_config.splitting_ops == []


@pytest.mark.parametrize("mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE])
def test_piecewise_kda_is_split_and_reads_runtime_metadata(make_kda_layer, monkeypatch, mode):
    compilation_config = SimpleNamespace(
        cudagraph_mode=mode, splitting_ops=["vllm::mla_forward"], static_forward_context={}
    )
    make_kda_layer(compilation_config, "layer")
    make_kda_layer(compilation_config, "another_layer")

    graph = torch.fx.Graph()
    inputs = graph.placeholder("inputs")
    projection = graph.call_function(torch.add, (inputs, 1))
    output = graph.call_function(torch.empty_like, (inputs,))
    graph.call_function(
        torch.ops.vllm.glm5next_kda_forward.default, (projection, inputs, inputs, inputs, output, "layer")
    )
    result = graph.call_function(torch.mul, (output, 2))
    graph.output(result)
    split, partitions = split_graph(torch.fx.GraphModule(torch.nn.Module(), graph), compilation_config.splitting_ops)
    kda_partitions = [
        partition
        for partition in partitions
        if any(
            node.op == "call_function" and node.target == torch.ops.vllm.glm5next_kda_forward.default
            for node in partition.graph.graph.nodes
        )
    ]
    # The backend captures only non-splitting graphs. KDA must remain outside
    # capture even when the initial profile has no attention metadata.
    assert len(kda_partitions) == 1
    assert kda_partitions[0].is_splitting_graph
    assert any(not partition.is_splitting_graph for partition in partitions)
    assert compilation_config.splitting_ops == ["vllm::mla_forward", "vllm::glm5next_kda_forward"]

    class StaticOutputGraph(torch.nn.Module):
        """Model the persistent output allocations of a captured producer."""

        def __init__(self, module):
            super().__init__()
            self.module = module
            self.outputs = None

        def forward(self, *args):
            outputs = self.module(*args)
            if self.outputs is None:
                self.outputs = outputs
            else:
                for static, current in zip(self.outputs, outputs):
                    static.copy_(current)
            return self.outputs

    class FixedInputGraph(torch.nn.Module):
        """Replay reads the captured addresses, without rebinding new inputs."""

        def __init__(self, module):
            super().__init__()
            self.module = module
            self.inputs = None

        def forward(self, *args):
            if self.inputs is None:
                self.inputs = args
            return self.module(*self.inputs)

    producer, _, consumer = partitions
    assert not producer.is_splitting_graph and not consumer.is_splitting_graph
    split.add_module(producer.submod_name, StaticOutputGraph(producer.graph))
    split.add_module(consumer.submod_name, FixedInputGraph(consumer.graph))
    context = SimpleNamespace(metadata=None, no_compile_layers={})
    calls = []
    norm_output_addresses = []

    def dispatch(qkv, g1, beta, output):
        calls.append(context.metadata)
        if context.metadata is None:
            output.zero_()
        else:
            output.copy_(qkv + g1 + beta + context.metadata)

    def norm(output, gate, *, out):
        # A fresh return value here would not update the buffer bound into the
        # downstream replay. The real custom op must provide its static output.
        assert out is output
        norm_output_addresses.append(out.data_ptr())
        out.copy_(output * gate)
        return out

    context.no_compile_layers["layer"] = SimpleNamespace(_forward=dispatch, o_norm=SimpleNamespace(forward_oot=norm))
    monkeypatch.setattr(kda, "get_forward_context", lambda: context)
    with torch.library._scoped_library("vllm", "IMPL", "CPU") as library:
        library.impl("glm5next_kda_forward", kda.glm5next_kda_forward)
        inputs = torch.ones(32, 8)
        torch.testing.assert_close(split(inputs), torch.zeros_like(inputs))
        for input_value, metadata in [(2, 3), (4, 7)]:
            inputs.fill_(input_value)
            context.metadata = metadata
            torch.testing.assert_close(
                split(inputs), torch.full_like(inputs, (3 * input_value + 1 + metadata) * input_value * 2)
            )
    assert calls == [None, 3, 7]
    assert len(set(norm_output_addresses)) == 1


@pytest.mark.parametrize("operation", ["shard", "reduce_scatter"])
@pytest.mark.parametrize("rank", [0, 7])
def test_sequence_parallel_padding_stays_dynamic(monkeypatch, operation, rank):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(glm_model, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(sequence_parallel, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(sequence_parallel, "get_tensor_model_parallel_rank", lambda: rank)
    group = SimpleNamespace(device_communicator=None)
    monkeypatch.setattr(sequence_parallel, "get_tp_group", lambda: group)
    # Simulate this rank's collective result, retaining the real SP padding
    # and split code so CPU tracing exercises the same dynamic shape branches.
    monkeypatch.setattr(
        sequence_parallel,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim: x[rank * (x.shape[0] // 8) : (rank + 1) * (x.shape[0] // 8)],
    )

    def forward(x):
        padded = glm_model._pad_sequence_parallel_tokens(x)
        if operation == "shard":
            return sequence_parallel.sp_shard(padded)
        return sequence_parallel.sp_reduce_scatter(padded)

    torch._dynamo.reset_code(forward.__code__)
    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True, dynamic=True)
    with torch._dynamo.config.patch(error_on_recompile=True):
        for count in [64, 27, 129, 31, 64]:
            inputs = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4)
            torch._dynamo.mark_dynamic(inputs, 0)
            shard_size = (count + 7) // 8
            expected = torch.zeros(shard_size * 8, 4)
            expected[:count] = inputs
            expected = expected[rank * shard_size : (rank + 1) * shard_size]
            torch.testing.assert_close(compiled(inputs), expected)


@pytest.mark.parametrize("caller", ["model", "decoder"])
def test_model_sequence_parallel_collectives_receive_padded_tokens(monkeypatch, caller):
    tp_size = 8
    num_tokens = 27
    inputs = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4)
    positions = torch.arange(num_tokens)
    padded_tokens = ((num_tokens + tp_size - 1) // tp_size) * tp_size
    expected = torch.zeros(padded_tokens, inputs.shape[1])
    expected[:num_tokens] = inputs
    monkeypatch.setattr(glm_model, "get_tensor_model_parallel_world_size", lambda: tp_size)

    class CollectiveReached(Exception):
        pass

    def check_collective_input(value):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        raise CollectiveReached

    if caller == "model":
        model = Glm5NextModel.__new__(Glm5NextModel)
        torch.nn.Module.__init__(model)
        model.is_sequence_parallel = True
        monkeypatch.setattr(glm_model, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
        monkeypatch.setattr(glm_model, "sp_shard", check_collective_input)
        with pytest.raises(CollectiveReached):
            model.forward(None, positions, None, inputs_embeds=inputs)
    else:
        layer = glm_model.Glm5NextDecoderLayer.__new__(glm_model.Glm5NextDecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.mhc = layer.is_sequence_parallel = True
        layer.is_mtp_layer = False
        layer.hc_attn_fn = layer.hc_attn_scale = layer.hc_attn_base = None
        layer.input_layernorm = SimpleNamespace(weight=torch.ones(inputs.shape[1]), variance_epsilon=1e-6)
        # Retain the real gather, attention and scatter control flow while
        # replacing the device-dependent mHC and attention computations.
        layer.hc_post_pre = lambda value, *args, **kwargs: (None, None, None, value)
        layer.self_attn = lambda hidden_states, positions: hidden_states
        monkeypatch.setattr(glm_model, "sp_all_gather", lambda value: expected)
        monkeypatch.setattr(glm_model, "sp_reduce_scatter", check_collective_input)
        with pytest.raises(CollectiveReached):
            layer.forward(positions, inputs[: padded_tokens // tp_size], post=torch.ones(1))
