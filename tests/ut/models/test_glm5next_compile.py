# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.compilation import decorators
from vllm.config.compilation import DynamicShapesType
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.glm5next import kda, mtp
from vllm_ascend.models.glm5next.model import Glm5NextModel
from vllm_ascend.models.glm5next.mtp import Glm5NextMTP


@pytest.mark.parametrize("num_tokens", [1, 32, 129, 513])
@pytest.mark.parametrize("input_kind", ["ids", "embeddings", "intermediate"])
@pytest.mark.parametrize("model_class", [Glm5NextModel, Glm5NextMTP])
def test_compile_marks_token_dimensions_dynamic(monkeypatch, num_tokens, input_kind, model_class):
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
    calls = []
    context = SimpleNamespace(no_compile_layers={})

    def dispatch(qkv, g1, beta, output):
        calls.append(context.value)
        output.copy_(qkv + g1 + beta + context.value)

    def norm(output, gate):
        assert not torch.compiler.is_compiling()
        return output * gate + context.value

    context.no_compile_layers["kda"] = SimpleNamespace(_forward=dispatch, o_norm=norm)
    monkeypatch.setattr(kda, "get_forward_context", lambda: context)

    def forward(qkv, g1, beta):
        output = torch.empty_like(qkv)
        output = torch.ops.vllm.glm5next_kda_forward(qkv, g1, beta, g1, output, "kda")
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
