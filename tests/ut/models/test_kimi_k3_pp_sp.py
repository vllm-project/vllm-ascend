# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU test that the production PP+SP sharding path matches the unsplit model."""

import __future__

import ast
import threading
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


def load_definitions(path, names, namespace, *, bases=None, methods=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names), path
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            node.bases = [ast.Name(id=bases[node.name], ctx=ast.Load())]
            node.decorator_list = []
            node.body = [item for item in node.body if getattr(item, "name", None) in methods[node.name]]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec", flags=__future__.annotations.compiler_flag), namespace)


class IntermediateTensors:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        if isinstance(key, slice):
            return IntermediateTensors({name: tensor[key] for name, tensor in self.tensors.items()})
        return self.tensors[key]

    def __setitem__(self, key, value):
        self.tensors[key] = value

    def items(self):
        return self.tensors.items()

    @staticmethod
    def empty_like(tensors):
        return IntermediateTensors({name: torch.empty_like(tensor) for name, tensor in tensors.tensors.items()})


def config(tp=2, pp=2, dp=1, ep=True, architecture="KimiLinearForCausalLM", runner_v2=False):
    return SimpleNamespace(
        use_v2_model_runner=runner_v2,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            enable_expert_parallel=ep,
            use_sequence_parallel_moe=dp > 1 and tp > 1 and ep,
        ),
    )


class BaseModel(nn.Module):
    def _cache_aux_pp_layout(self):
        # Upstream EagleModelMixin caches post-layer aux slots with <= start.
        self._aux_slot_base_cached = bisect_right(self.aux_hidden_state_layers, self.start_layer)
        self._aux_upstream_total_cached = self._aux_slot_base_cached

    def _maybe_add_hidden_state(self, states, layer_idx, hidden, residual):
        if layer_idx in self.aux_hidden_state_layers:
            if self.config.attn_res_block_size is None and residual is not None:
                hidden = hidden + residual
            states.append(hidden)
        return states

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        hidden_size = self.config.hidden_size
        residual_shape: tuple[int, ...] = (batch_size, hidden_size)
        block_size = self.config.attn_res_block_size
        if block_size is not None:
            residual_shape = (batch_size, (self.start_layer + block_size - 1) // block_size, hidden_size)
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
                "residual": torch.zeros(residual_shape, dtype=dtype, device=device),
            }
        )


class Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.ones(3))
        self.variance_epsilon = 1e-5

    def forward(self, hidden, residual=None):
        return hidden if residual is None else (hidden + residual, hidden + residual)


def attn_res_fwd_cpu(
    prefix,
    addend,
    blocks,
    projection,
    gamma,
    epsilon,
    valid,
    output_gamma=None,
    output_epsilon=1e-5,
    block_write_idx=-1,
    return_materialized=False,
    mix=True,
    optimize_prefill=False,
):
    """Reference the fused op's add/mix/norm/write contract without NPU dispatch."""
    raw = prefix if addend is None else prefix + addend
    materialized = raw
    if mix and valid:
        values = torch.cat((blocks[:, :valid], raw.unsqueeze(1)), dim=1).float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
        logits = (normalized * gamma.float() * projection.float()).sum(-1)
        materialized = (logits.softmax(-1).unsqueeze(-1) * values).sum(1).to(raw.dtype)
    output = materialized
    if output_gamma is not None:
        value = materialized.float()
        output = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + output_epsilon) * output_gamma).to(
            raw.dtype
        )
    if block_write_idx >= 0:
        blocks[:, block_write_idx].copy_(raw)
    return output, raw, materialized if return_materialized else output


class BaseMLP(nn.Module):
    def forward(self, hidden):
        return (hidden * 0.25 + 0.1) * self.weight_fraction


class BaseDecoder(nn.Module):
    def forward(self, positions, hidden_states, residual, **kwargs):
        # Mirror upstream KimiDecoderLayer.forward's dispatch.
        if getattr(self, "use_attn_residuals", False):
            return self.forward_attn_residual(positions, hidden_states, residual)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self._run_self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Collectives:
    """Blocking TP collectives with distinct rank data, also for empty shards."""

    def __init__(self, tp, context):
        self.tp = tp
        self.context = context
        self.barrier = threading.Barrier(tp, timeout=20)
        self.values = [None] * tp
        self.calls = [0] * tp

    def exchange(self, tensor, reduce=False):
        rank = self.context.rank
        self.values[rank] = tensor
        self.barrier.wait()
        if reduce:
            full = torch.stack(self.values).sum(0)
            padding = (-full.shape[0]) % self.tp
            full = torch.nn.functional.pad(full, (0, 0, 0, padding))
            size = full.shape[0] // self.tp
            result = full[rank * size : (rank + 1) * size]
        else:
            result = torch.cat(self.values, dim=0)
        self.calls[rank] += 1
        self.barrier.wait()
        return result


@pytest.fixture
def runtime(monkeypatch):
    from collections.abc import Sequence
    from enum import Enum

    context = threading.local()
    context.rank = 0
    context.pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    namespace = {
        "torch": torch,
        "F": torch.nn.functional,
        "nn": nn,
        "BaseModel": BaseModel,
        "BaseDecoder": BaseDecoder,
        "BaseMLP": BaseMLP,
        "bisect_left": bisect_left,
        "model_parallel_is_initialized": lambda: True,
        "IntermediateTensors": IntermediateTensors,
        "Enum": Enum,
        "Sequence": Sequence,
        "_PP_TRANSPORT_PREFIX": "pp_transport",
        "cdiv": lambda a, b: (a + b - 1) // b,
        "_use_attn_res_prefill_cache": lambda: False,
        "envs": SimpleNamespace(VLLM_MOE_SKIP_PADDING=True),
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: context.forward,
        "get_pp_group": lambda: context.pp,
        "get_tensor_model_parallel_world_size": lambda: context.tp,
        "get_tensor_model_parallel_rank": lambda: context.rank,
    }
    # Transport and sharding use the repository implementations, including
    # multi-dimensional residuals and zero-length token tensors.
    transport = ast.parse((ROOT / "vllm_ascend/worker/v2/pp_transport.py").read_text())
    enum_node = next(node for node in transport.body if getattr(node, "name", None) == "PPTransportDataType")
    exec(compile(ast.Module(body=[enum_node], type_ignores=[]), "pp_transport.py", "exec"), namespace)
    load_definitions(
        "vllm_ascend/worker/v2/pp_transport.py",
        {
            "_get_transport_key_prefix",
            "get_pp_transport_tensors",
            "add_pp_transport_tensors",
            "add_pp_transport_buffers",
            "make_empty_intermediate_tensors",
            "_add_aux_hidden_state_buffers",
            "_add_topk_indices_buffer",
        },
        namespace,
    )
    namespace["MappingProxyType"] = MappingProxyType
    factories_node = next(
        node
        for node in transport.body
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "_PP_TRANSPORT_BUFFER_FACTORIES"
    )
    exec(
        compile(
            ast.Module(body=[factories_node], type_ignores=[]),
            "pp_transport.py",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    namespace["make_pp_empty_intermediate_tensors"] = namespace["make_empty_intermediate_tensors"]
    load_definitions(
        "vllm_ascend/models/common/ops/sequence_parallel.py",
        {"sp_shard", "sp_padding_mask", "_ascend_sp_shard_impl", "_ascend_sp_padding_mask_impl"},
        namespace,
    )
    # Exercise the real kernels on CPU without PrivateUse1 dispatch or NPU initialization.
    for name in ("ascend_sp_shard_impl", "ascend_sp_padding_mask_impl"):
        monkeypatch.setattr(torch.ops.vllm, name, namespace[f"_{name}"], raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "attn_res_fwd", attn_res_fwd_cpu, raising=False)
    load_definitions(
        "vllm_ascend/models/kimi_k3.py",
        {"AscendKimiLinearModel", "AscendKimiDecoderLayer", "AscendKimiMLP"},
        namespace,
        bases={
            "AscendKimiLinearModel": "BaseModel",
            "AscendKimiDecoderLayer": "BaseDecoder",
            "AscendKimiMLP": "BaseMLP",
        },
        methods={
            "AscendKimiLinearModel": {
                "__init__",
                "forward",
                "make_empty_intermediate_tensors",
                "_cache_aux_pp_layout",
            },
            "AscendKimiDecoderLayer": {"forward", "prepare_attn_residual", "forward_attn_residual", "_run_self_attn"},
            "AscendKimiMLP": {"forward"},
        },
    )
    return namespace, context


def make_model(namespace, context, start, end, block_size, sp, materialized):
    class Attention(nn.Module):
        def forward(self, *, hidden_states, positions):
            assert hidden_states.shape[0] == positions.shape[0]
            return hidden_states / context.tp if sp else hidden_states

    model = namespace["AscendKimiLinearModel"].__new__(namespace["AscendKimiLinearModel"])
    nn.Module.__init__(model)
    model.config = SimpleNamespace(attn_res_block_size=block_size, hidden_size=3)
    model.start_layer, model.end_layer = start, end
    model.use_sequence_parallel = sp
    model.dspark_aux_capture_materialized = materialized
    model.aux_hidden_state_layers = (0, 1, 2, 3, 4, 5)
    model.layers = nn.ModuleList()
    for idx in range(end):
        layer = namespace["AscendKimiDecoderLayer"]()
        layer.use_sequence_parallel = sp
        layer.fuse_o_proj_mm_reduce_scatter = False
        layer.use_attn_residuals = block_size is not None
        layer.is_moe_layer = idx % 2 == 1
        layer.input_layernorm = Norm()
        layer.post_attention_layernorm = Norm()
        layer.self_attn = Attention()
        layer.mlp = BaseMLP() if layer.is_moe_layer else namespace["AscendKimiMLP"]()
        layer.mlp.use_sequence_parallel = sp
        # Distinct TP partial results exercise the real inner MLP AG/RS.
        layer.mlp.weight_fraction = (
            (context.rank + 1) / (context.tp * (context.tp + 1) / 2) if sp and not layer.is_moe_layer else 1.0
        )
        layer.prev_valid_blocks = (idx + block_size - 1) // block_size if block_size else 0
        layer.is_block_write_layer = block_size is not None and idx % block_size == 0
        layer.block_write_idx = idx // block_size if block_size else 0
        for name in ("self_attention_res", "mlp_res"):
            setattr(layer, f"{name}_proj", SimpleNamespace(weight=torch.ones(1, 3) * 0.1))
            setattr(layer, f"{name}_norm", SimpleNamespace(weight=torch.ones(3), variance_epsilon=1e-5))
        model.layers.append(layer)
    model.output_attn_res_proj = SimpleNamespace(weight=torch.ones(1, 3) * 0.1)
    model.output_attn_res_norm = SimpleNamespace(weight=torch.ones(3), variance_epsilon=1e-5)
    return model


@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("tp,num_tokens", [(2, 8), (2, 3), (16, 8)])
@pytest.mark.parametrize("cuts", [(0, 1, 3, 5), (0, 0, 3, 5), (0, 2, 2, 5), (0, 2, 5, 5)])
def test_pipeline_shards_match_unsplit_model(runtime, materialized, sp, tp, num_tokens, cuts):
    block_size = 2
    namespace, context = runtime
    dtype = torch.float32
    inputs = torch.arange(num_tokens * 3, dtype=dtype).reshape(num_tokens, 3) / 10
    positions = torch.arange(num_tokens)
    context.tp = 1
    reference = make_model(namespace, context, 0, 5, block_size, False, materialized)
    expected, expected_aux = reference(None, positions, None, inputs_embeds=inputs)

    collectives = Collectives(tp, context)
    namespace["sp_all_gather"] = collectives.exchange
    namespace["sp_reduce_scatter"] = lambda value: collectives.exchange(value, reduce=True)
    shard = namespace["sp_shard"]
    shards_per_rank = [0] * tp

    def count_shards(value):
        shards_per_rank[context.rank] += 1
        return shard(value)

    namespace["sp_shard"] = count_shards

    def run_rank(rank):
        context.tp, context.rank = tp, rank
        intermediate = None
        for stage, (start, end) in enumerate(zip(cuts, cuts[1:])):
            context.pp = SimpleNamespace(is_first_rank=stage == 0, is_last_rank=stage == len(cuts) - 2)
            context.forward = SimpleNamespace(is_padding=torch.zeros(num_tokens, dtype=torch.bool))
            model = make_model(namespace, context, start, end, block_size, sp, materialized)
            if intermediate is not None:
                model._cache_aux_pp_layout()
                capacity = max(12, num_tokens)
                buffers = model.make_empty_intermediate_tensors(capacity, dtype, "cpu")
                assert buffers.tensors.keys() == intermediate.tensors.keys()
                assert model._aux_slot_base_cached == len(intermediate.tensors) - 2
                if context.pp.is_last_rank:
                    assert model._aux_upstream_total_cached == len(intermediate.tensors) - 2
                # Boundary tensors are full-sequence replicas under the
                # stage-boundary gather contract.
                for tensor in intermediate.tensors.values():
                    assert tensor.shape[0] == num_tokens
                # Upstream copies full-token tensors into persistent buffers.
                for name, tensor in buffers.tensors.items():
                    tensor.fill_(999)
                    assert tensor[:num_tokens].shape == intermediate[name].shape
                    tensor[:num_tokens].copy_(intermediate[name])
                intermediate = IntermediateTensors(
                    {name: tensor[:num_tokens] for name, tensor in buffers.tensors.items()}
                )
            output = model(None, positions, intermediate, inputs_embeds=inputs if stage == 0 else None)
            if sp:
                expected_mask = torch.arange((num_tokens + tp - 1) // tp) + rank * ((num_tokens + tp - 1) // tp)
                torch.testing.assert_close(context.forward.is_padding, expected_mask >= num_tokens)
            else:
                assert not context.forward.is_padding.any()
            if context.pp.is_last_rank:
                return output
            intermediate = output

    with ThreadPoolExecutor(max_workers=tp) as pool:
        outputs = list(pool.map(run_rank, range(tp)))
    assert all((count > 0) == sp for count in shards_per_rank)
    assert len(set(collectives.calls)) == 1 and (collectives.calls[0] > 0) == sp
    for output, aux in outputs:
        torch.testing.assert_close(output, expected)
        assert len(aux) == len(expected_aux)
        for actual, reference_aux in zip(aux, expected_aux):
            torch.testing.assert_close(actual, reference_aux)


@pytest.mark.parametrize("stage", range(4))
@pytest.mark.parametrize("runner_v2", [False, True])
@pytest.mark.parametrize(
    "draft_type,draft_arch,materialized",
    [
        ("qwen3", "DSparkDraftModel", True),
        ("qwen3", "Qwen3DSparkModel", True),
        ("kimi_k3", "K3DSparkModel", False),
        (None, None, False),
    ],
)
def test_aux_contract_selected_before_draft_load(runtime, stage, runner_v2, draft_type, draft_arch, materialized):
    namespace, context = runtime
    context.tp = 2
    context.pp = SimpleNamespace(is_first_rank=stage == 0, is_last_rank=stage == 3)
    cfg = config(tp=2, pp=4, runner_v2=runner_v2)
    cfg.model_config.hf_text_config = SimpleNamespace(
        vocab_size=16,
        hidden_size=3,
        num_hidden_layers=8,
        num_attention_heads=2,
        attn_res_block_size=2,
        rms_norm_eps=1e-5,
    )
    cfg.speculative_config = (
        SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=draft_type, architectures=[draft_arch])
            ),
        )
        if draft_type is not None
        else None
    )
    namespace.update(
        VocabParallelEmbedding=lambda *args, **kwargs: nn.Identity(),
        PPMissingLayer=nn.Identity,
        RMSNorm=lambda *args, **kwargs: nn.Identity(),
        ReplicatedLinear=lambda *args, **kwargs: nn.Identity(),
        make_layers=lambda *args, **kwargs: (stage * 2, stage * 2 + 2, nn.ModuleList()),
    )
    # AST extraction retains methods only, so supply the production class default.
    namespace["AscendKimiLinearModel"].dspark_aux_capture_materialized = False
    model = namespace["AscendKimiLinearModel"](vllm_config=cfg)
    assert model.dspark_aux_capture_materialized is materialized
    assert model.use_sequence_parallel is runner_v2


@pytest.mark.parametrize("owns_embed", [False, True])
def test_draft_mapper_reuses_upstream_rules(owns_embed):
    @dataclass
    class Mapper:
        orig_to_new_substr: dict
        orig_to_new_prefix: dict
        orig_to_new_stacked: dict

    mapper = Mapper(
        {"confidence_head": None, "embed_tokens": None, "lm_head": None, "future_rule": "preserved"},
        {"": "model."},
        {".gate_proj": (".gate_up_proj", 0)},
    )
    captured = []
    namespace = {
        "BaseModel": nn.Module,
        "replace": replace,
        "AutoWeightsLoader": lambda model: SimpleNamespace(
            load_weights=lambda weights, mapper: captured.append(mapper)
        ),
    }
    load_definitions(
        "vllm_ascend/models/kimi_k3_dspark.py",
        {"AscendK3DSparkForCausalLM"},
        namespace,
        bases={"AscendK3DSparkForCausalLM": "BaseModel"},
        methods={"AscendK3DSparkForCausalLM": {"load_weights"}},
    )
    model = namespace["AscendK3DSparkForCausalLM"]()
    model.hf_to_vllm_mapper = mapper
    model._owns_embed_tokens = owns_embed
    model.load_weights([])
    selected = captured[0]
    expected = dict(mapper.orig_to_new_substr)
    if owns_embed:
        del expected["embed_tokens"]
    assert selected.orig_to_new_substr == expected
    assert selected.orig_to_new_prefix is mapper.orig_to_new_prefix
    assert selected.orig_to_new_stacked is mapper.orig_to_new_stacked
    assert "embed_tokens" in mapper.orig_to_new_substr
    assert (selected is mapper) is not owns_embed


def initialize_pp_cpu_count_sync(runner):
    tree = ast.parse((ROOT / "vllm_ascend/worker/v2/model_runner.py").read_text())
    runner_cls = next(node for node in tree.body if getattr(node, "name", None) == "NPUModelRunner")
    assert isinstance(runner_cls, ast.ClassDef)
    initializer = next(node for node in runner_cls.body if getattr(node, "name", None) == "__init__")
    assert isinstance(initializer, ast.FunctionDef)
    assignment = next(
        node
        for node in initializer.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "sync_spec_pp_cpu_counts" for target in node.targets
        )
    )
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "model_runner.py", "exec"), {"self": runner})


@pytest.mark.parametrize(
    "architecture",
    [
        "KimiLinearForCausalLM",
        "KimiK3ForCausalLM",
        "KimiK3ForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "DeepseekV4ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "MiniMaxM3SparseForCausalLM",
        "Qwen3ForCausalLM",
    ],
)
@pytest.mark.parametrize("use_pp,num_speculative_steps", [(False, 0), (False, 7), (True, 0), (True, 7)])
def test_pp_cpu_count_sync_is_scoped(architecture, use_pp, num_speculative_steps):
    runner = SimpleNamespace(
        use_pp=use_pp,
        num_speculative_steps=num_speculative_steps,
        model_config=SimpleNamespace(architecture=architecture),
    )
    initialize_pp_cpu_count_sync(runner)
    enabled = architecture.startswith("Kimi") or architecture == "Qwen3_5ForConditionalGeneration"
    assert runner.sync_spec_pp_cpu_counts is (enabled and use_pp and num_speculative_steps > 0)


@pytest.mark.parametrize("architecture", ["KimiK3ForCausalLM", "DeepseekV4ForCausalLM"])
@pytest.mark.parametrize("legacy_transport", [False, True])
@pytest.mark.parametrize(
    "use_pp,num_speculative_steps,owns_speculator,prefill_chunk",
    [
        (True, 7, False, False),
        (True, 7, True, False),
        (False, 7, True, False),
        (False, 0, False, False),
        (True, 0, False, False),
        (True, 7, False, True),
        (True, 7, True, True),
        (True, 0, False, True),
    ],
)
def test_host_positions_after_rejection_or_chunk(
    architecture, legacy_transport, use_pp, num_speculative_steps, owns_speculator, prefill_chunk
):
    events = []

    class BaseStateRunner(SimpleNamespace):
        def postprocess_sampled(self, *args):
            events.append("reject")
            self.req_states.num_computed_tokens.gpu[0] = 11

        def postprocess_num_computed_tokens(self, input_batch):
            events.append("advance")
            self.req_states.num_computed_tokens.gpu[0] = 27

        def _copy_num_computed_tokens_to_cpu(self):
            events.append("copy")
            self.num_computed_tokens_cpu.copy_(self.req_states.num_computed_tokens.gpu)

    namespace = {"BaseStateRunner": BaseStateRunner, "MambaHybridModelState": type("MambaHybridModelState", (), {})}
    load_definitions(
        "vllm_ascend/worker/v2/model_runner.py",
        {"NPUModelRunner"},
        namespace,
        bases={"NPUModelRunner": "BaseStateRunner"},
        methods={"NPUModelRunner": {"postprocess_sampled", "postprocess_num_computed_tokens", "_update_seq_lens_cpu"}},
    )
    runner = namespace["NPUModelRunner"]()
    runner.speculator = object() if owns_speculator else None
    runner.use_spec_pp = use_pp and num_speculative_steps > 0 and legacy_transport
    runner.use_pp = use_pp
    runner.is_last_pp_rank = not use_pp
    runner.model_state = object()
    runner.num_speculative_steps = num_speculative_steps
    runner.model_config = SimpleNamespace(architecture=architecture)
    initialize_pp_cpu_count_sync(runner)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"r": 0},
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([19])),
        num_computed_tokens_cpu=torch.tensor([35]),
    )
    runner.num_computed_tokens_cpu = torch.tensor([35])
    runner.num_computed_tokens_event = SimpleNamespace(synchronize=lambda: events.append("wait"))
    runner.input_buffers = SimpleNamespace(seq_lens_cpu=torch.zeros(1, dtype=torch.int64))
    if prefill_chunk:
        runner.postprocess_num_computed_tokens(SimpleNamespace())
        expected_position = 27
    else:
        runner.postprocess_sampled(None, None, None, None)
        expected_position = 11
    scheduler = SimpleNamespace(num_scheduled_tokens={"r": 4}, scheduled_cached_reqs=SimpleNamespace(req_ids=["r"]))
    runner._update_seq_lens_cpu(scheduler, ["r"])
    needs_sync = owns_speculator or runner.sync_spec_pp_cpu_counts
    copies_count = runner.sync_spec_pp_cpu_counts if prefill_chunk else needs_sync
    expected_events = ["advance" if prefill_chunk else "reject"]
    if copies_count:
        expected_events.append("copy")
    if needs_sync:
        expected_events.append("wait")
    assert events == expected_events
    expected_cpu_position = expected_position if copies_count else 35
    assert runner.input_buffers.seq_lens_cpu[0] == expected_cpu_position + 4
    assert runner.req_states.num_computed_tokens.gpu[0] == expected_position
