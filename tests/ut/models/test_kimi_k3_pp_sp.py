# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU test that the production PP+SP sharding path matches the unsplit model."""

import __future__

import ast
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


def load_definitions(path, names, namespace, *, bases=None, methods=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    # Also pick up module-level simple assignments referenced by the loaded defs.
    nodes += [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
    ]
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
    def forward(self, hidden, residual=None):
        return hidden if residual is None else (hidden + residual, hidden + residual)


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


class BaseV2Runner:
    intermediate_tensors: IntermediateTensors
    sync_and_slice_intermediate_tensors: Callable[[int], None]

    def execute_model(self, scheduler_output, intermediate_tensors, **kwargs):
        n = scheduler_output.num_tokens
        self.sync_and_slice_intermediate_tensors(n)
        # Mirror the pinned upstream copy boundary, not the SP implementation:
        # upstream uses the full token count for both source and destination.
        views = {}
        for name, tensor in self.intermediate_tensors.tensors.items():
            views[name] = tensor[:n]
            assert views[name].shape == intermediate_tensors[name][:n].shape
            views[name].copy_(intermediate_tensors[name][:n])
        return IntermediateTensors(views)


def make_v2_runner(namespace, buffers, *, tp=2, sp=True):
    runner = namespace["V2Runner"]()
    runner.vllm_config = config(tp=tp, ep=sp, runner_v2=True)
    runner.is_first_pp_rank = False
    runner.intermediate_tensors = buffers
    runner.model_state = SimpleNamespace()
    runner.ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(profiling_chunk_config=None))
    runner.kvpp = SimpleNamespace(complete_forward=lambda: None)
    runner.pcp_manager = None
    return runner


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
def runtime():
    from collections.abc import Sequence
    from enum import Enum

    context = threading.local()
    context.rank = 0
    context.pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    namespace = {
        "torch": torch,
        "nn": nn,
        "BaseModel": BaseModel,
        "BaseDecoder": BaseDecoder,
        "IntermediateTensors": IntermediateTensors,
        "Enum": Enum,
        "Sequence": Sequence,
        "_PP_TRANSPORT_PREFIX": "pp_transport",
        "cdiv": lambda a, b: (a + b - 1) // b,
        "apply_attn_res": None,
        "envs": SimpleNamespace(VLLM_MOE_SKIP_PADDING=True),
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: context.forward,
        "get_pp_group": lambda: context.pp,
        "get_tensor_model_parallel_world_size": lambda: context.tp,
        "get_tensor_model_parallel_rank": lambda: context.rank,
    }
    # Transport and sharding use the repository implementations, including
    # multi-dimensional residuals and zero-length token tensors.
    transport = ast.parse((ROOT / "vllm_ascend/worker/v2/pp_utils.py").read_text())
    enum_node = next(node for node in transport.body if getattr(node, "name", None) == "PPTransportDataType")
    exec(compile(ast.Module(body=[enum_node], type_ignores=[]), "pp_utils.py", "exec"), namespace)
    load_definitions(
        "vllm_ascend/worker/v2/pp_utils.py",
        {
            "_get_transport_key_prefix",
            "get_pp_transport_tensors",
            "add_pp_transport_tensors",
            "add_pp_transport_buffers",
            "make_empty_intermediate_tensors",
        },
        namespace,
    )
    namespace["make_pp_empty_intermediate_tensors"] = namespace["make_empty_intermediate_tensors"]
    load_definitions("vllm_ascend/models/common/ops/sequence_parallel.py", {"sp_shard", "sp_padding_mask"}, namespace)
    load_definitions(
        "vllm_ascend/models/kimi_k3.py",
        {"_apply_ascend_attn_res", "AscendKimiLinearModel", "AscendKimiDecoderLayer"},
        namespace,
        bases={"AscendKimiLinearModel": "BaseModel", "AscendKimiDecoderLayer": "BaseDecoder"},
        methods={
            "AscendKimiLinearModel": {"forward", "make_empty_intermediate_tensors", "_maybe_add_hidden_state"},
            "AscendKimiDecoderLayer": {"forward_attn_residual", "_run_self_attn", "_run_mlp"},
        },
    )
    load_definitions(
        "vllm_ascend/utils.py",
        {"_SP_ACROSS_PP_ARCHITECTURES", "_uses_sp_across_pp", "enable_sp_across_pp", "enable_sp"},
        namespace,
    )
    v2_namespace = {
        **namespace,
        "BaseV2Runner": BaseV2Runner,
        "vllm_version_is": lambda _: False,
        "_start_profiling_chunk_timing": lambda *_: None,
        "_finish_profiling_chunk_timing": lambda *_: None,
        "has_kv_transfer_group": lambda: False,
        "get_kv_transfer_group": lambda: None,
        "pcp_dispatch_context": nullcontext,
    }
    load_definitions(
        "vllm_ascend/worker/v2/model_runner.py",
        {"NPUModelRunner"},
        v2_namespace,
        bases={"NPUModelRunner": "BaseV2Runner"},
        methods={"NPUModelRunner": {"execute_model", "sync_and_slice_intermediate_tensors"}},
    )
    namespace["V2Runner"] = v2_namespace["NPUModelRunner"]
    return namespace, context


def make_model(namespace, context, start, end, block_size, sp, materialized):
    class Attention(nn.Module):
        def forward(self, *, hidden_states, positions):
            assert hidden_states.shape[0] == positions.shape[0]
            return hidden_states / context.tp if sp else hidden_states

    class MLP(nn.Module):
        def __init__(self, dense):
            super().__init__()
            self.dense = dense

        def forward(self, hidden):
            result = hidden * 0.25 + 0.1
            return result / context.tp if sp and self.dense else result

    model = namespace["AscendKimiLinearModel"]()
    model.config = SimpleNamespace(attn_res_block_size=block_size, hidden_size=3)
    model.start_layer, model.end_layer = start, end
    model.use_sequence_parallel = sp
    model.dspark_aux_capture_materialized = materialized
    model.aux_hidden_state_layers = (0, 1, 3, 4)
    model.layers = nn.ModuleList()
    for idx in range(end):
        layer = namespace["AscendKimiDecoderLayer"]()
        layer.use_sequence_parallel = sp
        layer.use_attn_residuals = block_size is not None
        layer.is_moe_layer = idx % 2 == 1
        layer.input_layernorm = Norm()
        layer.post_attention_layernorm = Norm()
        layer.self_attn = Attention()
        layer.mlp = MLP(not layer.is_moe_layer)
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


def test_pipeline_shards_match_unsplit_model(runtime):
    block_size = 2
    namespace, context = runtime
    num_tokens, dtype, materialized, cuts = 8, torch.float32, False, (0, 1, 3, 5)
    inputs = torch.arange(num_tokens * 3, dtype=dtype).reshape(num_tokens, 3) / 10
    positions = torch.arange(num_tokens)
    context.tp = 1
    reference = make_model(namespace, context, 0, 5, block_size, False, materialized)
    expected, expected_aux = reference(None, positions, None, inputs_embeds=inputs)

    tp = 2
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
            model = make_model(namespace, context, start, end, block_size, True, materialized)
            if intermediate is not None:
                local_tokens = (num_tokens + tp - 1) // tp
                capacity = max(12, num_tokens)
                buffers = model.make_empty_intermediate_tensors(capacity, dtype, "cpu")
                assert buffers.tensors.keys() == intermediate.tensors.keys()
                for tensor in intermediate.tensors.values():
                    assert tensor.shape[0] == local_tokens
                for tensor in buffers.tensors.values():
                    tensor.fill_(999)
                runner = make_v2_runner(namespace, buffers, tp=tp)
                intermediate = runner.execute_model(SimpleNamespace(num_tokens=num_tokens), intermediate)
                assert runner.intermediate_tensors is buffers
            output = model(None, positions, intermediate, inputs_embeds=inputs if stage == 0 else None)
            expected_mask = torch.arange((num_tokens + tp - 1) // tp) + rank * ((num_tokens + tp - 1) // tp)
            torch.testing.assert_close(context.forward.is_padding, expected_mask >= num_tokens)
            if context.pp.is_last_rank:
                return output
            intermediate = output

    with ThreadPoolExecutor(max_workers=tp) as pool:
        outputs = list(pool.map(run_rank, range(tp)))
    assert shards_per_rank == [1] * tp
    assert collectives.calls[0] == collectives.calls[1] > 0
    for output, aux in outputs:
        torch.testing.assert_close(output, expected)
        assert len(aux) == len(expected_aux)
        for actual, reference_aux in zip(aux, expected_aux):
            torch.testing.assert_close(actual, reference_aux)


