# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, local
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn


class _World:
    def __init__(self, size):
        self.size = size
        self.state = local()
        self.barrier = Barrier(size, timeout=15)
        self.inputs = [None] * size

    def chunk(self, tensor):
        padded = torch.nn.functional.pad(tensor, (0, 0, 0, -len(tensor) % self.size))
        return padded.chunk(self.size)[self.state.rank]

    def collective(self, tensor, operation):
        self.inputs[self.state.rank] = tensor
        self.barrier.wait()
        if operation == "gather":
            result = torch.cat(self.inputs)
        else:
            result = torch.stack(self.inputs).sum(0)
            if operation == "scatter":
                result = result.chunk(self.size)[self.state.rank]
        self.barrier.wait()
        return result

    def all_gather(self, tensor, dim):
        assert dim == 0
        return self.collective(tensor, "gather")

    def reduce_scatter(self, tensor, dim):
        assert dim == 0
        return self.collective(tensor, "scatter")


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_counts = []

    def forward(self, hidden_states, residual=None):
        self.token_counts.append(len(hidden_states))
        if residual is not None:
            assert hidden_states.shape == residual.shape
            hidden_states = hidden_states + residual
        output = hidden_states * torch.rsqrt(hidden_states.square().mean(-1, keepdim=True) + 1e-6)
        return output if residual is None else (output, hidden_states)


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.world = config.world
        self.o_proj = SimpleNamespace(reduce_results=True, custom_op=config.custom_op)
        self.token_counts = []

    def forward(self, positions, hidden_states):
        assert len(hidden_states) == len(positions)
        self.token_counts.append(len(hidden_states))
        partial = (self.world.state.rank + 1) * (hidden_states + hidden_states.mean(0) / 4)
        return self.world.collective(partial, "reduce") if self.o_proj.reduce_results else partial


class _LegacyMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.world = config.world

    def experts(self, hidden_states, router_logits):
        assert router_logits is hidden_states
        return hidden_states.tanh()

    def forward(self, hidden_states):
        num_tokens = len(hidden_states)
        hidden_states = self.world.chunk(hidden_states)
        return self.world.all_gather(self.experts(hidden_states, hidden_states), 0)[:num_tokens]


class _LegacyDecoder(nn.Module):
    def __init__(self, vllm_config, prefix="", is_fused_checkpoint_transposed=False):
        super().__init__()
        self.input_layernorm = _Norm()
        self.post_attention_layernorm = _Norm()
        self.self_attn = _Attention(vllm_config)
        self.mlp = vllm_config.moe_type(vllm_config)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class _LegacyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.start_layer, self.end_layer = 0, 2
        self.layers = nn.ModuleList(config.decoder_type(config) for _ in range(2))
        self.norm = _Norm()
        self.collect_aux = config.collect_aux
        self.embed_input_ids = nn.Identity()

    def _maybe_add_hidden_state(self, states, layer_idx, hidden_states, residual):
        if self.collect_aux:
            states.append(hidden_states if residual is None else hidden_states + residual)
        return states

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None
        auxiliary = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for index, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(auxiliary, index + 1, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return (hidden_states, auxiliary) if auxiliary else hidden_states


@pytest.fixture
def load_patch(monkeypatch):
    """Import the real patch against isolated pre-backport vLLM interfaces."""

    def load(world, native=False):
        qwen = ModuleType("vllm.model_executor.models.qwen3_moe")
        qwen.Qwen3MoeDecoderLayer = type("Decoder", (_LegacyDecoder,), {})
        qwen.Qwen3MoeSparseMoeBlock = type("MoE", (_LegacyMoE,), {})
        qwen.Qwen3MoeModel = type("Model", (_LegacyModel,), {})
        if native:
            qwen.Qwen3MoeModel.use_sequence_parallel = property(lambda self: True)
        distributed = ModuleType("vllm.distributed")
        distributed.get_tensor_model_parallel_world_size = lambda: world.size
        distributed.tensor_model_parallel_all_gather = world.all_gather
        distributed.tensor_model_parallel_reduce_scatter = world.reduce_scatter
        config_module = ModuleType("vllm.config")
        config_module.VllmConfig = SimpleNamespace
        utils = ModuleType("vllm.model_executor.models.utils")
        utils.sequence_parallel_chunk = world.chunk
        for module in (qwen, distributed, config_module, utils):
            monkeypatch.setitem(sys.modules, module.__name__, module)
        path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_qwen3_moe.py"
        spec = importlib.util.spec_from_file_location("isolated_qwen3_moe_patch", path)
        patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch)
        return patch, qwen

    return load


def _config(world, qwen, collect_aux=False):
    return SimpleNamespace(
        world=world,
        decoder_type=qwen.Qwen3MoeDecoderLayer,
        moe_type=qwen.Qwen3MoeSparseMoeBlock,
        collect_aux=collect_aux,
        custom_op=None,
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            hf_text_config=SimpleNamespace(num_experts=8, mlp_only_layers=[], decoder_sparse_step=1),
        ),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=True, pipeline_parallel_size=1),
    )


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("num_tokens", [1, 3, 16])
@pytest.mark.parametrize("collect_aux", [False, True])
def test_patch_shards_norms_and_preserves_multirank_outputs(load_patch, tp_size, num_tokens, collect_aux):
    world = _World(tp_size)
    _, qwen = load_patch(world)
    inputs = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4) / 7

    def run(rank, enabled):
        world.state.rank = rank
        config = _config(world, qwen, collect_aux)
        config.parallel_config.use_sequence_parallel_moe = enabled
        instance = qwen.Qwen3MoeModel(config)
        # Exercise both the embedding and inputs_embeds entry paths.
        result = instance(inputs, torch.arange(num_tokens), inputs_embeds=inputs if collect_aux else None)
        expected = (num_tokens + tp_size - 1) // tp_size if enabled else num_tokens
        for layer in instance.layers:
            assert layer.input_layernorm.token_counts == [expected]
            assert layer.post_attention_layernorm.token_counts == [expected]
            assert layer.self_attn.token_counts == [num_tokens]
        assert instance.norm.token_counts == [expected]
        return result

    with ThreadPoolExecutor(max_workers=tp_size) as pool:
        baseline = list(pool.map(lambda rank: run(rank, False), range(tp_size)))
        actual = list(pool.map(lambda rank: run(rank, True), range(tp_size)))
    for output, expected in zip(actual, baseline):
        torch.testing.assert_close(output, expected)
        if collect_aux:
            output, auxiliary = output
            assert all(tensor.shape == inputs.shape for tensor in auxiliary)
        assert output.shape == inputs.shape


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("parallel_config", "use_sequence_parallel_moe", False),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("model_config", "is_multimodal_model", True),
        ("hf_text_config", "num_experts", 0),
        ("hf_text_config", "mlp_only_layers", [1]),
        ("hf_text_config", "decoder_sparse_step", 2),
        (None, "custom_op", object()),
    ],
)
def test_ineligible_models_preserve_original_projection_and_layout(load_patch, section, key, value):
    world = _World(2)
    _, qwen = load_patch(world)
    config = _config(world, qwen)
    if section == "hf_text_config":
        target = config.model_config.hf_text_config
    else:
        target = getattr(config, section) if section else config
    setattr(target, key, value)
    instance = qwen.Qwen3MoeModel(config)
    assert not instance.use_sequence_parallel
    assert all(layer.self_attn.o_proj.reduce_results for layer in instance.layers)


def test_native_upstream_sp_is_not_overridden(load_patch):
    _, qwen = load_patch(_World(2), native=True)
    assert qwen.Qwen3MoeDecoderLayer.__init__ is _LegacyDecoder.__init__
    assert qwen.Qwen3MoeDecoderLayer.forward is _LegacyDecoder.forward
    assert qwen.Qwen3MoeSparseMoeBlock.forward is _LegacyMoE.forward
    assert qwen.Qwen3MoeModel.forward is _LegacyModel.forward


def test_patch_is_idempotent_and_preserves_attention_forward(load_patch):
    patch, qwen = load_patch(_World(2))
    forwards = (qwen.Qwen3MoeDecoderLayer.forward, qwen.Qwen3MoeSparseMoeBlock.forward, qwen.Qwen3MoeModel.forward)
    patch._apply_patch()
    assert forwards == (
        qwen.Qwen3MoeDecoderLayer.forward,
        qwen.Qwen3MoeSparseMoeBlock.forward,
        qwen.Qwen3MoeModel.forward,
    )
    instance = qwen.Qwen3MoeModel(_config(_World(2), qwen))
    assert instance.layers[0].self_attn.forward.__func__ is _Attention.forward


def test_subclass_without_base_init_stays_replicated(load_patch):
    _, qwen = load_patch(_World(2))

    class Decoder(qwen.Qwen3MoeDecoderLayer):
        def __init__(self):
            nn.Module.__init__(self)

    model = qwen.Qwen3MoeModel.__new__(qwen.Qwen3MoeModel)
    nn.Module.__init__(model)
    model.start_layer = 0
    model.layers = nn.ModuleList([Decoder()])
    assert not model.use_sequence_parallel
