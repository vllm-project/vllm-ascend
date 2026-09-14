# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of production PP/SP methods, without NPU-only model imports."""

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
    assert len(nodes) == len(names)
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
    _slice_kimi_sp_intermediate_tensors: Callable[[int], None]

    def execute_model(
        self, scheduler_output, intermediate_tensors, dummy_run=False, skip_attn_for_dummy_run=False, **kwargs
    ):
        n = scheduler_output.num_tokens
        if not dummy_run:
            self._slice_kimi_sp_intermediate_tensors(n)
        elif not skip_attn_for_dummy_run:
            self.prepare_dummy_attn(SimpleNamespace(num_tokens_after_padding=n))
        # Mirror the pinned upstream copy boundary, not the SP implementation:
        # upstream uses the full token count for both source and destination.
        views = {}
        for name, tensor in self.intermediate_tensors.tensors.items():
            views[name] = tensor[:n]
            if not dummy_run:
                assert views[name].shape == intermediate_tensors[name][:n].shape
                views[name].copy_(intermediate_tensors[name][:n])
        if getattr(self, "fail", False):
            raise RuntimeError("model failure")
        return IntermediateTensors(views)

    def prepare_dummy_attn(self, input_batch, **kwargs):
        return (), torch.empty(0)


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
        "BaseRunner": object,
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
        },
        namespace,
    )
    load_definitions("vllm_ascend/models/common/ops/sequence_parallel.py", {"sp_shard", "sp_padding_mask"}, namespace)
    load_definitions(
        "vllm_ascend/models/kimi_k3.py",
        {"_apply_ascend_attn_res", "AscendKimiLinearModel", "AscendKimiDecoderLayer"},
        namespace,
        bases={"AscendKimiLinearModel": "BaseModel", "AscendKimiDecoderLayer": "BaseDecoder"},
        methods={
            "AscendKimiLinearModel": {"forward", "make_empty_intermediate_tensors", "_maybe_add_hidden_state"},
            "AscendKimiDecoderLayer": {"forward", "forward_attn_residual", "_run_self_attn", "_run_mlp"},
        },
    )
    load_definitions("vllm_ascend/utils.py", {"_is_kimi_k3_target", "enable_kimi_k3_sp", "enable_sp"}, namespace)
    load_definitions(
        "vllm_ascend/worker/model_runner_v1.py",
        {"NPUModelRunner"},
        namespace,
        bases={"NPUModelRunner": "BaseRunner"},
        methods={
            "NPUModelRunner": {
                "_get_pp_input_num_tokens",
                "sync_and_slice_intermediate_tensors",
                "sync_and_gather_intermediate_tensors",
            }
        },
    )
    v2_namespace = {
        **namespace,
        "BaseV2Runner": BaseV2Runner,
        "vllm_version_is": lambda _: False,
        "_start_profiling_chunk_timing": lambda *_: None,
        "_finish_profiling_chunk_timing": lambda *_: None,
    }
    load_definitions(
        "vllm_ascend/worker/v2/model_runner.py",
        {"NPUModelRunner"},
        v2_namespace,
        bases={"NPUModelRunner": "BaseV2Runner"},
        methods={"NPUModelRunner": {"execute_model", "_slice_kimi_sp_intermediate_tensors", "prepare_dummy_attn"}},
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


@pytest.mark.parametrize("num_tokens", [0, 1, 3, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block_size", [None, 2, 3])
@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("cuts", [(0, 1, 3, 5), (0, 0, 3, 5)])
@pytest.mark.parametrize("runner_version", ["mrv1", "mrv2"])
def test_pipeline_shards_match_unsplit_model(
    runtime, num_tokens, block_size, materialized, cuts, dtype, runner_version
):
    namespace, context = runtime
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
                if runner_version == "mrv1":
                    runner = namespace["NPUModelRunner"]()
                    runner.vllm_config = config(tp=tp)
                    capacity = runner._get_pp_input_num_tokens(capacity)
                buffers = model.make_empty_intermediate_tensors(capacity, dtype, "cpu")
                assert buffers.tensors.keys() == intermediate.tensors.keys()
                for tensor in intermediate.tensors.values():
                    assert tensor.shape[0] == local_tokens
                for tensor in buffers.tensors.values():
                    tensor.fill_(999)
                if runner_version == "mrv1":
                    runner.intermediate_tensors = buffers
                    intermediate = runner.sync_and_gather_intermediate_tensors(num_tokens, intermediate, True)
                else:
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


@pytest.mark.parametrize(
    "architecture", ["KimiLinearForCausalLM", "KimiK3ForCausalLM", "KimiK3ForConditionalGeneration"]
)
@pytest.mark.parametrize("pp", [1, 2, 4])
@pytest.mark.parametrize("dp", [1, 2])
def test_kimi_sp_switch_includes_dp1_and_pp(architecture, pp, dp):
    namespace: dict[str, Any] = {}
    load_definitions("vllm_ascend/utils.py", {"_is_kimi_k3_target", "enable_kimi_k3_sp", "enable_sp"}, namespace)
    assert namespace["enable_sp"](config(pp=pp, dp=dp, architecture=architecture))
    assert not namespace["enable_sp"](config(pp=pp, dp=dp, ep=False))
    assert not namespace["enable_sp"](config(pp=pp, dp=dp, tp=1))
    assert not namespace["enable_sp"](config(architecture="Qwen3MoeForCausalLM"))


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("method", ["sync_and_slice_intermediate_tensors", "sync_and_gather_intermediate_tensors"])
def test_mrv1_pp_receive_preserves_capacity_and_auxiliary_shapes(runtime, tp, sp, sync, method):
    namespace, _ = runtime
    runner = namespace["NPUModelRunner"]()
    runner.vllm_config = config(tp=tp, ep=sp)
    capacity = runner._get_pp_input_num_tokens(12)
    buffers = IntermediateTensors({"hidden_states": torch.zeros(capacity, 3), "residual": torch.zeros(capacity, 2, 3)})
    runner.intermediate_tensors = buffers
    for num_tokens in (3, 9, 1, 0):
        local_tokens = (num_tokens + tp - 1) // tp if sp else num_tokens
        incoming = IntermediateTensors(
            {name: torch.ones_like(tensor[:local_tokens]) for name, tensor in buffers.tensors.items()}
        )
        incoming["pp_transport_aux_hidden_states_0"] = torch.full((local_tokens, 3), 7.0)
        output = getattr(runner, method)(num_tokens, incoming if sync else None, sync)
        assert output["hidden_states"].shape == (local_tokens, 3)
        assert output["residual"].shape == (local_tokens, 2, 3)
        if sync:
            torch.testing.assert_close(output["hidden_states"], incoming["hidden_states"])
            torch.testing.assert_close(output["residual"], incoming["residual"])
            torch.testing.assert_close(
                output["pp_transport_aux_hidden_states_0"], incoming["pp_transport_aux_hidden_states_0"]
            )
            assert buffers["pp_transport_aux_hidden_states_0"].shape == (capacity, 3)
        assert output["hidden_states"].data_ptr() == buffers["hidden_states"].data_ptr() or local_tokens == 0
        assert runner.intermediate_tensors is buffers


@pytest.mark.parametrize("dp", [1, 2])
def test_mrv2_enables_kimi_pp_sp(dp):
    namespace: dict[str, Any] = {}
    load_definitions("vllm_ascend/utils.py", {"_is_kimi_k3_target", "enable_kimi_k3_sp", "enable_sp"}, namespace)
    vc = config(dp=dp)
    vc.use_v2_model_runner = True
    assert namespace["enable_kimi_k3_sp"](vc)
    assert namespace["enable_sp"](vc)
    vc.parallel_config.pipeline_parallel_size = 1
    assert namespace["enable_kimi_k3_sp"](vc)


@pytest.mark.parametrize("dummy_run,skip_attn", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("tp,sp", [(1, True), (2, True), (4, True), (2, False)])
def test_mrv2_receive_restores_capacity(runtime, dummy_run, skip_attn, fail, tp, sp):
    namespace, _ = runtime
    buffers = IntermediateTensors(
        {
            "hidden_states": torch.zeros(16, 3),
            "residual": torch.zeros(16, 2, 3),
            "pp_transport_aux_hidden_states_0": torch.zeros(16, 3),
        }
    )
    runner = make_v2_runner(namespace, buffers, tp=tp, sp=sp)
    runner.fail = fail
    for num_tokens in (3, 9, 1, 0):
        local_tokens = (num_tokens + tp - 1) // tp if sp else num_tokens
        incoming = IntermediateTensors(
            {name: torch.ones_like(tensor[:local_tokens]) for name, tensor in buffers.items()}
        )
        scheduler_output = SimpleNamespace(num_tokens=num_tokens, total_num_scheduled_tokens=num_tokens)
        if fail:
            with pytest.raises(RuntimeError, match="model failure"):
                runner.execute_model(scheduler_output, incoming, dummy_run=dummy_run, skip_attn_for_dummy_run=skip_attn)
        else:
            output = runner.execute_model(
                scheduler_output, incoming, dummy_run=dummy_run, skip_attn_for_dummy_run=skip_attn
            )
            for name, tensor in output.items():
                assert tensor.shape == incoming[name].shape
                assert tensor.data_ptr() == buffers[name].data_ptr() or local_tokens == 0
                if not dummy_run:
                    torch.testing.assert_close(tensor, incoming[name])
        assert runner.intermediate_tensors is buffers


@pytest.mark.parametrize("residual_kind", ["none", "ordinary", "attn_res_bank"])
def test_mla_aux_capture_keeps_raw_prefix_sum(runtime, residual_kind):
    namespace, _ = runtime
    model = namespace["AscendKimiLinearModel"]()
    model.aux_hidden_state_layers = (3,)
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    residual = None
    expected = hidden
    if residual_kind == "ordinary":
        residual = torch.full_like(hidden, 2)
        expected = hidden + residual
    elif residual_kind == "attn_res_bank":
        residual = torch.full((4, 2, 3), 99.0)
    states = model._maybe_add_hidden_state([], 3, hidden, residual)
    assert len(states) == 1
    torch.testing.assert_close(states[0], expected)
    assert model._maybe_add_hidden_state([], 2, hidden, residual) == []


@pytest.mark.parametrize("owns_speculator,use_spec_pp", [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize("prefill_chunk", [False, True])
def test_mrv2_pp_refreshes_host_positions(owns_speculator, use_spec_pp, prefill_chunk):
    events = []

    class BaseStateRunner:
        def postprocess_sampled(self, *args):
            events.append("reject")
            self.req_states.num_computed_tokens.gpu[0] = 11

        def postprocess_num_computed_tokens(self, input_batch):
            events.append("advance")
            self.req_states.num_computed_tokens.gpu[0] = 27

        def _copy_num_computed_tokens_to_cpu(self):
            events.append("copy")
            self.num_computed_tokens_cpu.copy_(self.req_states.num_computed_tokens.gpu)

    namespace = {"BaseStateRunner": BaseStateRunner}
    load_definitions(
        "vllm_ascend/worker/v2/model_runner.py",
        {"NPUModelRunner"},
        namespace,
        bases={"NPUModelRunner": "BaseStateRunner"},
        methods={"NPUModelRunner": {"postprocess_sampled", "postprocess_num_computed_tokens", "_update_seq_lens_cpu"}},
    )
    runner = namespace["NPUModelRunner"]()
    runner.speculator = object() if owns_speculator else None
    runner.use_spec_pp = use_spec_pp
    runner.req_states = SimpleNamespace(
        req_id_to_index={"r": 0},
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([19])),
        num_computed_tokens_cpu=torch.tensor([19]),
    )
    runner.num_computed_tokens_cpu = torch.tensor([-1])
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
    if use_spec_pp:
        assert events == ["advance" if prefill_chunk else "reject", "copy", "wait"]
        assert runner.input_buffers.seq_lens_cpu[0] == expected_position + 4
    else:
        assert events == ["advance" if prefill_chunk else "reject"]
        assert runner.input_buffers.seq_lens_cpu[0] == 23


def graph_classes():
    class BaseGraphManager:
        def __init__(self):
            self.replays: dict[int, Callable[[], None]] = {}
            self.intermediate_tensors: IntermediateTensors | None = None

        def run_fullgraph(self, desc):
            replay = self.replays.get(desc.num_tokens)
            if replay is not None:
                replay()
            assert self.intermediate_tensors is not None
            return self.intermediate_tensors[: desc.num_tokens]

    namespace = {
        "torch": SimpleNamespace(
            full=torch.full,
            npu=SimpleNamespace(current_stream=lambda: None, is_current_stream_capturing=lambda: False),
        ),
        "nn": nn,
        "BaseWrapper": nn.Module,
        "BaseGraphManager": BaseGraphManager,
        "IntermediateTensors": IntermediateTensors,
        "contextmanager": __import__("contextlib").contextmanager,
        "CUDAGraphMode": SimpleNamespace(PIECEWISE="piecewise"),
        "get_forward_context": lambda: SimpleNamespace(cudagraph_runtime_mode="full"),
        "set_current_vllm_config": lambda *_: nullcontext(),
        "set_forward_context": lambda *_, **__: nullcontext(),
        "_EXTRA_CTX": SimpleNamespace(),
        "_get_graph_update_backend": lambda _: None,
        "update_full_graph_params": lambda *_: None,
        "logger": SimpleNamespace(info_once=lambda *_: None),
    }
    load_definitions("vllm_ascend/utils.py", {"_is_kimi_k3_target", "enable_kimi_k3_sp"}, namespace)
    load_definitions(
        "vllm_ascend/worker/v2/aclgraph_utils.py",
        {"ModelWithContext", "ModelAclGraphManager"},
        namespace,
        bases={"ModelWithContext": "BaseWrapper", "ModelAclGraphManager": "BaseGraphManager"},
        methods={
            "ModelWithContext": {"__init__", "forward"},
            "ModelAclGraphManager": {"_prepare_pp_sp_output", "_pp_sp_capture_buffers", "run_fullgraph"},
        },
    )
    return namespace


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mrv2_graph_buffers_across_gears(tp, dtype):
    graph_namespace = graph_classes()
    manager = graph_namespace["ModelAclGraphManager"]()
    manager.vllm_config = config(tp=tp, runner_v2=True)
    manager._pp_sp_output_buffers = None
    manager.update_stream = SimpleNamespace(wait_stream=lambda _: None)
    manager.model_runner = SimpleNamespace(
        dp_size=1, model_state=SimpleNamespace(attn_metadata=None), attn_groups=[], speculative_config=None
    )
    inputs = IntermediateTensors(
        {
            "hidden_states": torch.zeros(8, 3, dtype=dtype),
            "residual": torch.zeros(8, 2, 3, dtype=dtype),
            "pp_transport_aux_hidden_states_0": torch.zeros(8, 3, dtype=dtype),
        }
    )

    class Stage(nn.Module):
        def forward(self, *, positions, intermediate_tensors):
            local_tokens = (positions.shape[0] + tp - 1) // tp
            assert all(tensor.shape[0] == local_tokens for tensor in intermediate_tensors.tensors.values())
            return IntermediateTensors({name: tensor * 2 for name, tensor in intermediate_tensors.items()})

    wrapped = graph_namespace["ModelWithContext"](
        Stage(), pp_sp_tp_size=tp, prepare_pp_output=manager._prepare_pp_sp_output
    )
    pointers = None
    with manager._pp_sp_capture_buffers():
        for num_tokens in (8, 3, 1):
            positions = torch.arange(num_tokens)

            def capture_step(positions=positions, num_tokens=num_tokens):
                output = wrapped(positions=positions, intermediate_tensors=inputs[:num_tokens])
                # These are the pinned upstream graph-output copy semantics.
                for name, tensor in output.items():
                    assert manager.intermediate_tensors[name][:num_tokens].shape == tensor.shape
                    manager.intermediate_tensors[name][:num_tokens].copy_(tensor)

            capture_step()
            assert manager.intermediate_tensors is not None
            current_pointers = {name: tensor.data_ptr() for name, tensor in manager.intermediate_tensors.items()}
            if pointers is None:
                pointers = current_pointers
            assert current_pointers == pointers
            manager.replays[num_tokens] = capture_step
    assert manager.intermediate_tensors is not None
    assert manager.intermediate_tensors["hidden_states"].shape[0] == 8 // tp
    backing = manager.intermediate_tensors
    for num_tokens in (1, 8, 3):
        for tensor in inputs.tensors.values():
            tensor.fill_(num_tokens + 1)
        output = manager.run_fullgraph(SimpleNamespace(num_tokens=num_tokens, cg_mode="full"))
        for name, tensor in output.items():
            assert tensor.shape[0] == (num_tokens + tp - 1) // tp
            torch.testing.assert_close(tensor, torch.full_like(tensor, 2 * (num_tokens + 1)))
        # CPU replay also runs the Python wrapper; restore the persistent
        # backing as real graph replay does not re-enter that Python code.
        manager.intermediate_tensors = backing


@pytest.mark.parametrize(
    "graph_mode,use_mla,dcp_size,sp,expected",
    [
        ("none", False, 1, True, True),
        ("none", False, 1, False, False),
        ("none", True, 1, True, False),
        ("none", False, 2, True, False),
        ("full", True, 2, False, True),
    ],
)
def test_mrv1_attention_padding_uses_runner_config(graph_mode, use_mla, dcp_size, sp, expected):
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    execute = next(
        node for node in runner_class.body if isinstance(node, ast.FunctionDef) and node.name == "execute_model"
    )
    predicate = next(
        node.test
        for node in ast.walk(execute)
        if isinstance(node, ast.If) and "enable_sp(" in ast.unparse(node.test) and "use_mla" in ast.unparse(node.test)
    )
    runner = SimpleNamespace(
        vllm_config=config(ep=sp), model_config=SimpleNamespace(use_mla=use_mla), dcp_size=dcp_size
    )

    def check_sp(runner_config):
        # Real PP input preparation has no global model-initialization context.
        assert runner_config is runner.vllm_config
        return sp

    namespace = {
        "self": runner,
        "cudagraph_mode": graph_mode,
        "CUDAGraphMode": SimpleNamespace(FULL="full"),
        "enable_sp": check_sp,
    }
    assert eval(compile(ast.Expression(predicate), str(path), "eval"), namespace) is expected
