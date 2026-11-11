# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real projection executors with CPU collective references, not NPU timing.

The small stateful core below is an oracle for owner isolation, not an
implementation or accuracy test of the recurrent KDA kernel.
"""

import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from vllm.config import CUDAGraphMode

from vllm_ascend.ascend_config import FinegrainedTPConfig
from vllm_ascend.ops import finegrained_tp as fine
from vllm_ascend.ops import linear_op

FIELDS = ("embedding", "lmhead", "mlp", "qkv_iproj", "gproj", "oproj")
CONFIGURATIONS = list(itertools.product((1, 8), repeat=6))


class Stream:
    def record_event(self):
        return object()

    def wait_event(self, event):
        pass


class Collectives:
    def __init__(self, size):
        self.size = size
        self.condition = threading.Condition()
        self.sequence = [0] * size
        self.calls = {}

    def launch(self, kind, rank, source, output):
        with self.condition:
            index = self.sequence[rank]
            self.sequence[rank] += 1
            call = self.calls.setdefault(index, {"kind": kind, "inputs": {}, "outputs": {}, "ready": False})
            assert call["kind"] == kind, "collective order differs across owners"
            call["inputs"][rank] = source.detach().clone()
            call["outputs"][rank] = output
            if len(call["inputs"]) == self.size:
                tensors = [call["inputs"][i] for i in range(self.size)]
                assert len({(tuple(t.shape), t.dtype) for t in tensors}) == 1
                for destination in range(self.size):
                    if kind == "gather":
                        result = torch.cat(tensors, dim=0)
                    elif kind == "scatter":
                        result = torch.stack(tensors).sum(0).chunk(self.size)[destination]
                    else:
                        result = torch.cat([t.chunk(self.size)[destination] for t in tensors], dim=0)
                    call["outputs"][destination].copy_(result)
                call["ready"] = True
                self.condition.notify_all()

        def wait():
            with self.condition:
                assert self.condition.wait_for(lambda: call["ready"], timeout=20), (kind, index, rank)

        return SimpleNamespace(wait=wait)


def runtime(monkeypatch, world):
    groups, local = {}, threading.local()

    def group(size, rank):
        key = (size, rank // size)
        if key not in groups:
            hub = Collectives(size)
            groups[key] = [
                SimpleNamespace(world_size=size, rank_in_group=i, device_group=SimpleNamespace(hub=hub, rank=i))
                for i in range(size)
            ]
        result = groups[key][rank % size]

        def scatter(tensor, dim=0):
            assert dim == 0
            output = tensor.new_empty((tensor.shape[0] // size, *tensor.shape[1:]))
            result.device_group.hub.launch("scatter", result.rank_in_group, tensor, output).wait()
            return output

        result.reduce_scatter = scatter
        return result

    # Build groups before starting threads: no dictionary creation races.
    for size in (2, 4, 8):
        for rank in range(world):
            group(size, rank)

    def operation(kind):
        def launch(output, source, group, async_op=False):
            work = group.hub.launch(kind, group.rank, source, output)
            if async_op:
                return work
            work.wait()

        return launch

    monkeypatch.setattr(torch.npu, "current_stream", lambda: Stream())
    monkeypatch.setattr(torch.npu, "Stream", Stream)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(fine, "npu_stream_switch", lambda stream: nullcontext())
    monkeypatch.setattr(fine, "projection_capacity", lambda hidden: 3)
    for kind, name in (
        ("gather", "all_gather_into_tensor"),
        ("scatter", "reduce_scatter_tensor"),
        ("exchange", "all_to_all_single"),
    ):
        monkeypatch.setattr(fine.dist, name, operation(kind))
    monkeypatch.setattr(linear_op, "get_otp_group", lambda: local.output_group)
    return group, local


def matrix(rows, columns, offset):
    return torch.sin(torch.arange(rows * columns).reshape(rows, columns).float() + offset) / 8


class Linear:
    def __init__(self, weight):
        self.weight, self.bias = weight, None
        self.quant_method = SimpleNamespace(apply=lambda layer, x, bias=None: F.linear(x, layer.weight, bias))

    def __call__(self, x):
        return F.linear(x, self.weight), None


def core(qkv, forget, beta, state, live):
    q, k, v = qkv.chunk(3, dim=-1)
    output = torch.zeros_like(q)
    current = (q[:live].sigmoid() * k[:live] + v[:live] + forget[:live]) * beta[:live].repeat_interleave(
        2, -1
    ).sigmoid()
    output[:live] = current + state
    state.add_(current.sum(0))
    return output


@pytest.mark.parametrize(
    "sizes", CONFIGURATIONS + [(1, 1, 1, q, g, o) for q, g, o in itertools.product((2, 4, 8), repeat=3)]
)
def test_dp32_all_six_switch_combinations(monkeypatch, sizes):
    torch.set_num_threads(1)
    world, heads, dim, hidden_size, vocab_size, intermediate = 32, 8, 2, 4, 64, 16
    width = heads * dim
    group, local = runtime(monkeypatch, world)
    e_size, h_size, m_size, q_size, g_size, o_size = sizes
    weights = {
        "embed": matrix(vocab_size, hidden_size, 0),
        "head": matrix(vocab_size, hidden_size, 1),
        "qkv": matrix(3 * width, hidden_size, 2),
        "beta": matrix(heads, hidden_size, 3),
        "fa": matrix(dim, hidden_size, 4),
        "fb": matrix(width, dim, 5),
        "g": matrix(width, hidden_size, 6),
        "o": matrix(hidden_size, width, 7),
        "up": matrix(2 * intermediate, hidden_size, 8),
        "down": matrix(hidden_size, intermediate, 9),
    }

    @torch.inference_mode()
    def worker(rank):
        q_group = group(q_size, rank) if q_size > 1 else None
        g_group = group(g_size, rank) if g_size > 1 else None
        projection = fine.KDAProjectionTP(q_group, g_group, heads, dim) if q_size > 1 or g_size > 1 else None
        q_parts = [w.chunk(q_size)[rank % q_size] for w in weights["qkv"].chunk(3)]
        layer = SimpleNamespace(
            b_proj=Linear(weights["beta"].chunk(q_size)[rank % q_size]),
            f_a_proj=Linear(weights["fa"]),
            f_b_proj=Linear(weights["fb"]),
            g_proj=Linear(weights["g"].chunk(g_size)[rank % g_size]),
            _quantize_fused_qkv=lambda x: x,
            _matmul_fused_qkv=lambda x: F.linear(x, torch.cat(q_parts)),
        )
        o_layer = SimpleNamespace(
            weight=weights["o"].chunk(o_size, dim=1)[rank % o_size],
            bias=None,
            input_is_parallel=True,
            input_size_per_partition=width // o_size,
            output_size=hidden_size,
            reduce_results=True,
            return_bias=True,
            skip_bias_add=False,
            prefix="model.layers.1.self_attn.o_proj",
        )
        o_layer.quant_method = SimpleNamespace(apply=lambda layer, x, bias=None: F.linear(x, layer.weight, bias))
        if o_size > 1:
            local.output_group = group(o_size, rank)
            o_op = linear_op.OProjRowParallelOp(o_layer)
            o_op.update_attrs()
        up = torch.cat([w.chunk(m_size)[rank % m_size] for w in weights["up"].chunk(2)])
        mlp = SimpleNamespace(
            gate_up_proj=Linear(up),
            down_proj=Linear(weights["down"].chunk(m_size, 1)[rank % m_size]),
            _prepare_down_input=lambda x: x.chunk(2, -1)[0] * x.chunk(2, -1)[1],
        )
        mlp_tp = fine.ChunkedMLPTP(group(m_size, rank)) if m_size > 1 else None
        head = Linear(weights["head"].chunk(h_size)[rank % h_size])
        processor = SimpleNamespace(
            org_vocab_size=vocab_size, _apply_head=lambda layer, x, bias: F.linear(x, layer.weight, bias)
        )
        head_tp = fine.ChunkedLMHeadTP(group(h_size, rank)) if h_size > 1 else None
        state, reference_state = torch.zeros(width), torch.zeros(width)
        for step in range(2):
            live = (rank + step) % 4
            ids = (torch.arange(3) + rank * 3 + step) % vocab_size
            ids[live:] = 0
            if e_size > 1:
                e_group = group(e_size, rank)
                communication = fine.TPCommunication()
                all_ids = communication.gather(ids, e_group).wait()
                lo, hi = rank % e_size * vocab_size // e_size, (rank % e_size + 1) * vocab_size // e_size
                values = weights["embed"][all_ids] * ((all_ids >= lo) & (all_ids < hi)).unsqueeze(-1)
                hidden = communication.scatter(values, e_group).wait()
            else:
                hidden = weights["embed"][ids]
            q_ref = F.linear(hidden, weights["qkv"])
            b_ref = F.linear(hidden, weights["beta"])
            f_ref = F.linear(F.linear(hidden, weights["fa"]), weights["fb"])
            g_ref = F.linear(hidden, weights["g"])
            if projection is not None:
                qkv, forget, beta, gate = projection.project(layer, hidden)
                actual_core = core(qkv, forget[0].flatten(1), beta[0], state, live)
                actual_g = gate.wait().flatten(1)
            else:
                actual_core = core(q_ref, f_ref, b_ref, state, live)
                actual_g = g_ref
            expected_core = core(q_ref, f_ref, b_ref, reference_state, live)
            torch.testing.assert_close(state, reference_state)
            gated = actual_core * actual_g.sigmoid()
            expected = F.linear(expected_core * g_ref.sigmoid(), weights["o"])
            projected = o_op.apply(gated)[0] if o_size > 1 else F.linear(gated, weights["o"])
            torch.testing.assert_close(projected, expected)
            actual = (
                mlp_tp.forward(mlp, projected)
                if mlp_tp is not None
                else F.linear(mlp._prepare_down_input(F.linear(projected, weights["up"])), weights["down"])
            )
            expected = F.linear(mlp._prepare_down_input(F.linear(expected, weights["up"])), weights["down"])
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-4)
            logits = (
                head_tp.forward(processor, actual, head, None)
                if head_tp is not None
                else F.linear(actual, weights["head"])
            )
            torch.testing.assert_close(logits, F.linear(expected, weights["head"]), atol=1e-7, rtol=1e-4)

    with ThreadPoolExecutor(max_workers=world) as executor:
        list(executor.map(worker, range(world)))


def deployment():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_moe=True,
            hf_text_config=SimpleNamespace(
                model_type="kimi_linear",
                num_attention_heads=96,
                linear_attn_config={"num_heads": 96, "use_full_rank_gate": True},
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=32,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=384, max_num_seqs=384),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL, max_cudagraph_capture_size=384, cudagraph_capture_sizes=[384]
        ),
    )


@pytest.mark.parametrize("sizes", CONFIGURATIONS)
def test_dp32_configuration_matrix(sizes):
    cfg = FinegrainedTPConfig(**{f"{name}_tensor_parallel_size": size for name, size in zip(FIELDS, sizes)})
    vc = deployment()
    cfg._validate_preconditions(vc)
    assert [getattr(cfg, f"{name}_tensor_parallel_size") for name in FIELDS] == list(sizes)


@pytest.mark.parametrize("size", (2, 4, 8))
@pytest.mark.parametrize("loader_name", ("weight_loader", "weight_loader_v2"))
def test_real_merged_loader_slices_q_k_v_separately(monkeypatch, size, loader_name):
    from unittest.mock import MagicMock

    from vllm.config.vllm import set_current_vllm_config

    config = MagicMock()
    config.compilation_config.custom_ops = ["all"]
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    with set_current_vllm_config(config):
        for rank in range(size):
            group = SimpleNamespace(world_size=size, rank_in_group=rank)
            layer = fine.ProjectionMergedColumnParallelLinear(
                4, [16] * 3, group=group, params_dtype=torch.float32, prefix="attn.in_proj_qkv"
            )
            weights = [matrix(16, 4, index) for index in range(3)]
            for index, weight in enumerate(weights):
                getattr(layer, loader_name)(layer.weight, weight, index)
            torch.testing.assert_close(layer.weight, torch.cat([w.chunk(size)[rank] for w in weights]))
            assert layer.tp_size == size
            assert layer.weight.tp_rank == rank
            assert layer.weight.tp_size == size


@pytest.mark.parametrize("disabled", (0, 1))
def test_disabled_sizes_do_not_restrict_dense_native_tp(disabled):
    fields = {f"{name}_tensor_parallel_size": disabled for name in FIELDS}
    config = SimpleNamespace(
        model_config=SimpleNamespace(is_moe=False),
        parallel_config=SimpleNamespace(data_parallel_size=1, tensor_parallel_size=8),
    )
    FinegrainedTPConfig(**fields)._validate_preconditions(config)


@pytest.mark.parametrize("size", (2, 4, 8))
def test_mxfp8_weight_and_block32_scale_follow_each_qkv_slice(monkeypatch, size):
    from vllm_ascend.quantization.method_adapters import AscendLinearMethod
    from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod

    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    scheme = object.__new__(AscendW8A8MXFP8DynamicLinearMethod)
    scheme.group_size = 32
    config = SimpleNamespace(
        get_quant_method=lambda layer, prefix: AscendLinearMethod(scheme), online_quantization_config=None
    )
    weights = [((torch.arange(32 * 64).reshape(32, 64) + i) % 9).to(torch.float8_e4m3fn) for i in range(3)]
    scales = [((torch.arange(32 * 2).reshape(32, 2) + i) % 255).to(torch.uint8) for i in range(3)]
    for rank in range(size):
        layer = fine.ProjectionMergedColumnParallelLinear(
            64,
            [32] * 3,
            group=SimpleNamespace(world_size=size, rank_in_group=rank),
            quant_config=config,
            params_dtype=torch.bfloat16,
            prefix="attn.in_proj_qkv",
        )
        for index, (weight, scale) in enumerate(zip(weights, scales)):
            layer.weight_loader(layer.weight, weight, index)
            layer.weight_loader(layer.weight_scale, scale, index)
        torch.testing.assert_close(layer.weight.float(), torch.cat([w.chunk(size)[rank].float() for w in weights]))
        torch.testing.assert_close(layer.weight_scale, torch.cat([s.chunk(size)[rank] for s in scales]))
        assert layer.weight_scale.shape == (3 * 32 // size, 2)


def test_dsa_profile_joins_gate_collectives(monkeypatch):
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

    events = []
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.enable_sparse_sfa_turboquant = False
    impl.g_proj = object()
    impl.gproj_tp = SimpleNamespace(
        gather=lambda hidden: (events.append("gate_ag"), object())[1],
        project=lambda value, layer: (
            events.append("gate_project_return"),
            SimpleNamespace(wait=lambda: events.append("gate_join")),
        )[1],
    )
    output = torch.empty(3, 4)
    result = impl._forward_impl("layer", torch.zeros(3, 4), (), None, output=output)
    assert events == ["gate_ag", "gate_project_return", "gate_join"]
    torch.testing.assert_close(result, torch.zeros_like(output))


def test_core_is_enqueued_before_output_gate_join(monkeypatch):
    from vllm_ascend.ops import kimi_kda

    events = []
    qkv, forget, beta = torch.zeros(2, 12), torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2)
    gate = SimpleNamespace(wait=lambda: (events.append("gate_wait"), torch.zeros(2, 2, 2))[1])
    layer = SimpleNamespace(
        local_num_heads=2,
        head_dim=2,
        projection_tp=SimpleNamespace(project=lambda *args: (qkv, forget, beta, gate)),
        _forward_core=lambda *args: (events.append("core"), torch.zeros(1, 2, 2, 2))[1],
        o_norm=SimpleNamespace(forward_oot=lambda x, g, out: out.copy_(x)),
        o_proj=lambda x: (x, None),
    )
    monkeypatch.setattr(kimi_kda, "get_forward_context", lambda: SimpleNamespace(no_compile_layers={"layer": layer}))
    monkeypatch.setattr(kimi_kda, "projection_capacity", lambda x: 2)
    monkeypatch.setattr(kimi_kda, "maybe_save_kv_layer_to_connector", lambda *args: None)
    kimi_kda.kimi_kda_finegrained_forward(torch.zeros(2, 4), torch.empty(2, 4), "layer")
    assert events == ["core", "gate_wait"]


@pytest.mark.parametrize("field", ("qkv_iproj", "gproj"))
@pytest.mark.parametrize("condition", ("TP", "PCP", "DCP", "PP", "producer", "eager", "capture", "heads", "model"))
def test_projection_startup_rejects_unsupported_conditions(field, condition):
    cfg = FinegrainedTPConfig(**{f"{field}_tensor_parallel_size": 8})
    vc = deployment()
    dimensions = {
        "TP": "tensor_parallel_size",
        "PCP": "prefill_context_parallel_size",
        "DCP": "decode_context_parallel_size",
        "PP": "pipeline_parallel_size",
    }
    if condition in dimensions:
        setattr(vc.parallel_config, dimensions[condition], 2)
    elif condition == "producer":
        vc.kv_transfer_config.is_kv_consumer = False
    elif condition == "eager":
        vc.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    elif condition == "capture":
        vc.compilation_config.max_cudagraph_capture_size = 128
    elif condition == "heads":
        vc.model_config.hf_text_config.linear_attn_config["num_heads"] = 7
    else:
        vc.model_config.hf_text_config.model_type = "other"
    with pytest.raises(ValueError):
        cfg._validate_preconditions(vc)


def test_shared_projection_group_is_destroyed_once(monkeypatch):
    from unittest.mock import MagicMock

    from vllm_ascend.distributed import parallel_state

    group = MagicMock()
    names = ("_MLP_TP", "_LMTP", "_EMBED_TP", "_OTP", "_QKV_IPROJ_TP", "_GPROJ_TP")
    for name in names:
        monkeypatch.setattr(parallel_state, name, group)
    for name in ("_MC2", "_KVPP", "_P_TP", "_DYNAMIC_EPLB"):
        monkeypatch.setattr(parallel_state, name, None)
    parallel_state.destroy_ascend_model_parallel()
    group.destroy.assert_called_once()
    assert all(getattr(parallel_state, name) is None for name in names)


@pytest.mark.parametrize("q_size,g_size", ((1, 8), (8, 1), (4, 8)))
def test_kda_constructor_keeps_state_heads_and_does_not_retain_full_bfg(monkeypatch, q_size, g_size):
    from unittest.mock import MagicMock

    from vllm.config.vllm import set_current_vllm_config
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_ascend.ops import kimi_kda

    quant = SimpleNamespace(
        uses_kimi_k3_mixed_kda_projection=lambda prefix: True,
        get_quant_method=lambda layer, prefix: UnquantizedLinearMethod(),
        online_quantization_config=None,
    )
    config = SimpleNamespace(rms_norm_eps=1e-6)
    vc = SimpleNamespace(
        quant_config=quant,
        additional_config={
            "finegrained_tp_config": {"qkv_iproj_tensor_parallel_size": q_size, "gproj_tensor_parallel_size": g_size}
        },
    )

    def init(layer, config, vllm_config, prefix):
        nn.Module.__init__(layer)
        layer.hidden_size, layer.num_heads, layer.head_dim = 4, 8, 2
        layer.tp_size, layer.local_num_heads = 1, 8
        layer.projection_size = layer.local_projection_size = 16
        layer.conv_size, layer.model_config = 4, SimpleNamespace(dtype=torch.bfloat16)
        layer.o_norm = SimpleNamespace(eps=1e-5)
        layer.conv1d = nn.Module()
        layer.conv1d.weight = nn.Parameter(torch.empty(48, 1, 4))
        layer.conv1d.quant_method = SimpleNamespace(process_weights_after_loading=lambda: None)
        layer.f_b_proj = nn.Linear(2, 16, bias=False)

    monkeypatch.setattr(kimi_kda.KimiK3DeltaAttention, "__init__", init)
    monkeypatch.setattr(kimi_kda, "get_qkv_iproj_tp_group", lambda: SimpleNamespace(world_size=q_size, rank_in_group=0))
    monkeypatch.setattr(kimi_kda, "get_gproj_tp_group", lambda: SimpleNamespace(world_size=g_size, rank_in_group=0))
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    context = MagicMock()
    context.compilation_config.custom_ops = ["all"]
    with set_current_vllm_config(context):
        layer = kimi_kda.AscendKimiK3DeltaAttention(config, vc, "layer")
    assert (layer.tp_size, layer.local_num_heads) == (1, 8)
    assert layer.in_proj_qkvgfab.weight.shape == (48 // q_size, 4)
    assert layer.b_proj.weight.shape == (8 // q_size, 4)
    assert layer.g_proj.weight.shape == (16 // g_size, 4)
    assert layer.f_a_proj.weight.shape == (2, 4)
    assert layer.f_b_proj.weight.shape == (16, 2)
    assert not any("fused_bfg" in name for name, _ in layer.named_parameters())


def test_custom_ops_have_fake_inplace_output_contracts():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_ascend.models.kimi_k3 import kimi_k3  # noqa: F401
    from vllm_ascend.ops import kimi_kda  # noqa: F401

    with FakeTensorMode():
        hidden, output = torch.empty(3, 4), torch.empty(3, 4)
        torch.ops.vllm.kimi_kda_finegrained_forward(hidden, output, "layer")
        torch.ops.vllm.kimi_mlp_finegrained_forward(hidden, output, "mlp")
        assert output.shape == hidden.shape


@pytest.mark.parametrize("gate_tp_enabled", (False, True))
def test_dsa_active_path_defers_gate_join_through_indexer_and_uv(monkeypatch, gate_tp_enabled):
    from vllm_ascend.attention import sfa_v1

    events = []
    hidden, output = torch.randn(2, 4), torch.empty(2, 4)

    def trace(name, result):
        def call(*args, **kwargs):
            events.append(name)
            return result

        return call

    impl = SimpleNamespace(
        enable_sparse_sfa_turboquant=False,
        qk_rope_head_dim=2,
        pcp_enabled=False,
        _start_o_proj_weight_gather=lambda: None,
        g_proj=trace("owner_g", (torch.zeros(2, 4), None)),
        gproj_tp=SimpleNamespace(
            gather=trace("gate_ag", object()),
            project=trace("gate_project_return", SimpleNamespace(wait=trace("gate_join", torch.zeros(2, 2, 2)))),
        )
        if gate_tp_enabled
        else None,
        _compose_sfa_kv_cache=lambda cache: cache,
        _get_sfa_kv_slot_mapping=lambda metadata: metadata.slot_mapping,
        _get_indexer_attn_metadata=lambda: object(),
        _get_parallel_forward_context=lambda *args: SimpleNamespace(
            actual_seq_lengths_query=[2],
            actual_seq_lengths_key=[2],
            gather_full_o_proj=False,
            kv_slot_mapping=torch.arange(2),
            topk_num_tokens=2,
        ),
        preprocess_type=sfa_v1.PreprocessType.PROLOG_V3,
        runtime_has_indexer=True,
        layerwise_kv_cache_hook=None,
        _sfa_preprocess_prolog_v3=trace("prolog", (hidden, torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2))),
        _record_query_gather_context=lambda *args: None,
        _on_query_ready=lambda *args: None,
        _on_kv_cache_written=lambda *args: None,
        _prepare_indexer_metadata=lambda *args: None,
        indexer=trace("indexer", torch.zeros(2, 1, dtype=torch.int32)),
        skip_topk=False,
        use_index_cache=False,
        _get_indexer_attn_q_gather_handle=lambda *args: None,
        layer_name="layer",
        _execute_sparse_flash_attention_process=trace("sfa", torch.ones(2, 4)),
        _v_up_proj=trace("uv", torch.ones(2, 4)),
        _finalize_o_proj=trace("o", output),
    )
    for name in ("wait_for_kv_layer_from_connector", "notify_kv_cache_written", "maybe_save_kv_layer_to_connector"):
        monkeypatch.setattr(sfa_v1, name, lambda *args: None)
    monkeypatch.setattr(sfa_v1, "attention_transfer_window", nullcontext)
    monkeypatch.setattr(sfa_v1, "flash_attention_gate", trace("gate", None))
    metadata = SimpleNamespace(
        num_input_tokens=2,
        num_actual_tokens=2,
        cum_query_lens=[2],
        seq_lens=[2],
        cos=torch.ones(2, 2),
        sin=torch.ones(2, 2),
        slot_mapping=torch.arange(2),
    )
    sfa_v1.AscendSFAImpl._forward_impl(impl, "layer", hidden, (torch.zeros(1),), metadata, output=output)
    expected = ["gate_ag", "prolog", "gate_project_return", "indexer", "sfa", "uv", "gate_join", "gate", "o"]
    if not gate_tp_enabled:
        expected = ["prolog", "indexer", "sfa", "uv", "owner_g", "gate", "o"]
    assert events == expected


def test_real_compiler_partitions_kda_host_hooks_per_step(monkeypatch):
    from vllm.compilation.backends import split_graph

    from vllm_ascend.ops import kimi_kda

    calls, partitions = [], []
    step = [0]
    layer = SimpleNamespace(local_num_heads=2, head_dim=2)
    gate = SimpleNamespace(wait=lambda: torch.ones(2, 2, 2))
    layer.projection_tp = SimpleNamespace(
        project=lambda _, hidden: (hidden.repeat(1, 3), hidden.view(1, 2, 2, 2), torch.ones(1, 2, 2), gate)
    )

    def core(qkv, forget, beta, output):
        calls.append(("wait", step[0]))
        return qkv[:, :4].view(1, 2, 2, 2)

    layer._forward_core = core
    layer.o_norm = SimpleNamespace(forward_oot=lambda x, g, out: out.copy_(x))
    layer.o_proj = lambda x: (x, None)
    monkeypatch.setattr(kimi_kda, "get_forward_context", lambda: SimpleNamespace(no_compile_layers={"layer": layer}))
    monkeypatch.setattr(kimi_kda, "projection_capacity", lambda x: 2)
    monkeypatch.setattr(kimi_kda, "maybe_save_kv_layer_to_connector", lambda *args: calls.append(("save", step[0])))

    def backend(graph, example_inputs):
        split, items = split_graph(graph, ["vllm::kimi_kda_finegrained_forward"])
        partitions.extend(items)
        return split.forward

    def forward(hidden):
        hidden = hidden + 1
        output = torch.empty_like(hidden)
        torch.ops.vllm.kimi_kda_finegrained_forward(hidden, output, "layer")
        return output + 2

    # The production op dispatches on NPU; supply its same Python body for
    # this CPU compiler-partition test, scoped to the lifetime of this test.
    library = torch.library.Library("vllm", "IMPL", "CPU")
    try:
        library.impl("kimi_kda_finegrained_forward", kimi_kda.kimi_kda_finegrained_forward)
        compiled = torch.compile(forward, backend=backend, fullgraph=True)
        for index in range(2):
            step[0] = index
            torch.testing.assert_close(compiled(torch.full((2, 4), float(index))), torch.full((2, 4), float(index + 3)))
        boundaries = [item for item in partitions if item.is_splitting_graph]
        assert len(boundaries) == 1
        assert calls == [("wait", 0), ("save", 0), ("wait", 1), ("save", 1)]
    finally:
        library._destroy()


@pytest.mark.parametrize("rows", [0, 1, 3])
def test_actual_lmhead_processor_restores_rows_outside_config_scope(monkeypatch, rows):
    from vllm.config.vllm import set_current_vllm_config

    from vllm_ascend.ops import vocab_parallel_embedding as vocab

    groups, local = runtime(monkeypatch, 2)
    monkeypatch.setattr(vocab, "get_lmhead_tp_group", lambda: local.group)

    def embedding_init(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.parallel_mode = vocab.VocabParallelMode.FINE_GRAINED

    monkeypatch.setattr(vocab.AscendVocabParallelEmbedding, "__init__", embedding_init)
    weight = matrix(8, 4, 7)
    heads = []
    config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="kimi_linear")))
    with set_current_vllm_config(config):
        for rank in range(2):
            local.group = groups(2, rank)
            head = vocab.AscendParallelLMHead(8, 4)
            head.weight = nn.Parameter(weight.chunk(2)[rank], requires_grad=False)
            heads.append(head)

    def no_config():
        raise AssertionError("Sampling has no current VllmConfig scope")

    monkeypatch.setattr(vocab, "get_current_vllm_config", no_config)

    def owner(rank):
        processor = vocab.AscendLogitsProcessor.__new__(vocab.AscendLogitsProcessor)
        processor.org_vocab_size = 7
        processor._apply_head = lambda head, hidden, bias: F.linear(hidden, head.weight, bias)
        hidden = matrix(rows, 4, rank)
        actual = processor._get_logits_lmheadtp(hidden, heads[rank], None)
        torch.testing.assert_close(actual, F.linear(hidden, weight)[:, :7])
        assert actual.shape == (rows, 7)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(owner, range(2)))


@pytest.mark.parametrize("live_rows", [(1, 2), (0, 2)])
@pytest.mark.parametrize("gate_enabled", [False, True])
def test_nope_piecewise_real_sfa_and_o_keep_uniform_collectives(monkeypatch, live_rows, gate_enabled):
    from vllm_ascend.attention import sfa_v1

    groups, local = runtime(monkeypatch, 2)
    monkeypatch.setattr(sfa_v1, "projection_capacity", lambda hidden: 5)
    monkeypatch.setattr(
        sfa_v1, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE)
    )
    for name in ("wait_for_kv_layer_from_connector", "notify_kv_cache_written", "maybe_save_kv_layer_to_connector"):
        monkeypatch.setattr(sfa_v1, name, lambda *args: None)
    monkeypatch.setattr(sfa_v1, "attention_transfer_window", nullcontext)
    monkeypatch.setattr(sfa_v1, "flash_attention_gate", lambda output, gate: output.mul_(gate.sigmoid()))
    weight, gate_weight = matrix(4, 4, 1), matrix(4, 4, 2)

    def owner(rank):
        rows = live_rows[rank]
        local.output_group = groups(2, rank)
        layer = SimpleNamespace(
            weight=weight.chunk(2, dim=1)[rank],
            bias=None,
            input_is_parallel=True,
            input_size_per_partition=2,
            output_size=4,
            reduce_results=True,
            return_bias=True,
            skip_bias_add=False,
            prefix="self_attn.o_proj",
        )
        layer.quant_method = SimpleNamespace(apply=lambda layer, x, bias=None: F.linear(x, layer.weight, bias))
        op = linear_op.OProjRowParallelOp(layer)
        op.update_attrs()
        impl = sfa_v1.AscendSFAImpl.__new__(sfa_v1.AscendSFAImpl)
        impl.enable_sparse_sfa_turboquant = False
        impl.qk_rope_head_dim = 0
        impl.local_num_heads, impl.v_head_dim = 2, 2
        impl.pcp_enabled = False
        impl.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1))
        impl._parallel_o_proj = None
        impl.finegrained_o_tp = True
        impl.o_proj = op.apply_impl
        impl.g_proj = Linear(gate_weight.chunk(2)[rank]) if gate_enabled else None
        impl.gproj_tp = fine.FullRankGateTP(groups(2, rank), 2, 2) if gate_enabled else None
        impl.layerwise_kv_cache_hook = None
        impl.layer_name = "layer"
        impl._get_indexer_attn_metadata = lambda: None
        impl.preprocess_type = sfa_v1.PreprocessType.PROLOG_V3
        impl.has_indexer = False
        impl.indexer, impl.skip_topk = None, True
        impl._sfa_preprocess_prolog_v3 = lambda hidden_states, **kwargs: (hidden_states,) * 4
        impl._get_indexcache_topk_indices = lambda count: torch.zeros(count, 1, dtype=torch.int32)
        attention = matrix(rows, 4, rank + 3)
        impl._execute_sparse_flash_attention_process = lambda *args: attention.clone()
        impl._v_up_proj = lambda value: value
        hidden, output = matrix(3, 4, rank + 4), torch.zeros(3, 4)
        metadata = SimpleNamespace(
            num_actual_tokens=rows,
            num_input_tokens=3,
            slot_mapping=torch.arange(3),
            cum_query_lens=[rows],
            seq_lens=[rows],
            cos=None,
            sin=None,
        )
        impl._forward_impl("layer", hidden, (torch.zeros(1, 4),), metadata, output)
        expected = attention
        if gate_enabled:
            expected = expected * F.linear(hidden[:rows], gate_weight).sigmoid()
        torch.testing.assert_close(output[:rows], F.linear(expected, weight))
        if rows == 0:
            assert not output.any()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(owner, range(2)))
    calls = groups(2, 0).device_group.hub.calls
    # All collectives compare every peer's shape; O must include the idle owner.
    assert len(calls) == (4 if gate_enabled else 2)
