# SPDX-License-Identifier: Apache-2.0
"""A5 hardware regressions; CPU UTs maintain configuration and lifecycle boundaries."""

import gc
import json
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypedDict

import pytest
import torch
import torch_npu
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager, OffloadState
from vllm_ascend.expert_offload.graph import GraphExpertStaging
from vllm_ascend.expert_offload.host_store import HostExpertStore
from vllm_ascend.expert_offload.quantization import MXFP4ExpertAdapter
from vllm_ascend.expert_offload.residency import ExpertResidency
from vllm_ascend.expert_offload.transfer import TorchExpertTransfer
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDSDynamicFusedMoEMethod,
    maybe_normalize_mxfp_scale_layout,
)
from vllm_ascend.utils import is_950

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

pytestmark = pytest.mark.skipif(not is_950(), reason="A5 MXFP4 expert offload requires Ascend 950")
NUM_EXPERTS = 6
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 256
GROUP_SIZE = 32


@pytest.fixture
def expert_bundle(monkeypatch):
    generator = torch.Generator().manual_seed(0)
    tensors = {
        "w13_weight": torch.randint(
            256, (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 2), dtype=torch.uint8, generator=generator
        ),
        "w2_weight": torch.randint(
            256, (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2), dtype=torch.uint8, generator=generator
        ),
        "w13_weight_scale": torch.full(
            (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // GROUP_SIZE), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
        "w2_weight_scale": torch.full(
            (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // GROUP_SIZE), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
    }
    layer = SimpleNamespace(
        local_num_experts=NUM_EXPERTS,
        layer_name="test.experts",
        weight_loader=lambda *args, **kwargs: True,
        **{name: torch.nn.Parameter(tensor, requires_grad=False) for name, tensor in tensors.items()},
    )
    store = HostExpertStore(layer, MXFP4ExpertAdapter.tensor_names)
    for name in store.names:
        shards = ("w1", "w3") if name.startswith("w13") else ("w2",)
        store.loaded[name] = {(eid, shard) for eid in range(NUM_EXPERTS) for shard in shards}
    store.finalize(layer)
    # These standalone grouped GEMMs do not initialize the distributed MegaMoE runtime.
    monkeypatch.setattr("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.use_cann_megamoe", lambda _config: False)
    # Exercise the real post-processing without constructing a complete model.
    scheme = object.__new__(AscendW4A8MXFPDSDynamicFusedMoEMethod)
    scheme.group_size = GROUP_SIZE
    with set_current_vllm_config(VllmConfig()):
        yield MXFP4ExpertAdapter(SimpleNamespace(activation="silu"), scheme), store


def _gemm(x, weight, scale, counts):
    quantized, token_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
    token_scale = maybe_normalize_mxfp_scale_layout(token_scale)
    return torch_npu.npu_grouped_matmul(
        x=[quantized],
        weight=[weight],
        scale=None,
        antiquant_scale=[scale],
        per_token_scale=[token_scale],
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        split_item=2,
        group_type=0,
        group_list=torch.tensor(counts, dtype=torch.int64, device=x.device),
        group_list_type=0,
        x_dtype=torch.float8_e4m3fn,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        output_dtype=torch.bfloat16,
    )[0]


@torch.inference_mode()
def test_acl_graph_replay_updates_nz_slots_and_rejects_callback_failure(expert_bundle, monkeypatch):
    adapter, store = expert_bundle
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    manager = ExpertOffloadManager()
    manager.state = OffloadState.READY
    manager.graph_load_stream = torch.npu.Stream(device=device)
    resident = adapter.materialize(store, [0, 1], device)
    binding = SimpleNamespace(
        manager=manager,
        adapter=adapter,
        host_store=store,
        resident=resident,
        residency=ExpertResidency(NUM_EXPERTS, 2),
        num_experts=NUM_EXPERTS,
        device=device,
        close=lambda: None,
    )
    binding.residency.commit(binding.residency.plan([0, 1]))
    manager.bindings["test"] = binding
    staging = GraphExpertStaging(binding, (1, 1), torch.int32)
    ids = torch.zeros((1, 1), dtype=torch.int32, device=device)
    x = torch.randn(1, HIDDEN_SIZE, generator=torch.Generator().manual_seed(7)).to(device, dtype=torch.bfloat16)
    full = adapter.materialize(store, range(NUM_EXPERTS), device)
    store.prepare_native(adapter, resident, device)

    def forbidden_conversion(*args):
        raise AssertionError("graph replay must not format-cast")

    monkeypatch.setattr(adapter.scheme, "process_weights_after_loading", forbidden_conversion)
    pointers = {name: tensor.data_ptr() for name, tensor in resident.items()}
    capture_stream = torch.npu.Stream(device=device)
    manager.subscribe_graph_stream(capture_stream)

    def forward(weights, scale, hidden, counts):
        quantized, token_scale = torch_npu.npu_dynamic_mx_quant(hidden, dst_type=torch.float8_e4m3fn)
        return torch_npu.npu_grouped_matmul(
            x=[quantized],
            weight=[weights],
            scale=None,
            antiquant_scale=[scale],
            per_token_scale=[maybe_normalize_mxfp_scale_layout(token_scale)],
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=2,
            group_type=0,
            group_list=counts,
            group_list_type=0,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            output_dtype=torch.bfloat16,
        )[0]

    try:
        # Initialize kernel/workspace paths before capture.
        _gemm(x, resident["w13_weight"], resident["w13_weight_scale"], [1, 1])
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with manager.graph_capture(), torch.npu.graph(graph, stream=capture_stream):
            physical = staging.prepare(ids)
            counts = (torch.arange(2, device=device) >= physical.reshape(-1)[0]).to(torch.int64)
            gate, up = forward(resident["w13_weight"], resident["w13_weight_scale"], x, counts).chunk(2, -1)
            hidden = torch.nn.functional.silu(gate) * up
            output = forward(resident["w2_weight"], resident["w2_weight_scale"], hidden, counts)
        for expert_id in (5, 3, 1, 4, 5):
            manager.begin_execution()
            try:
                ids.fill_(expert_id)
                graph.replay()  # Replay on the default stream, not capture_stream.
                torch.npu.current_stream().synchronize()
                manager.check_graph_errors()
                expected_counts = [int(i >= expert_id) for i in range(NUM_EXPERTS)]
                gate, up = _gemm(x, full["w13_weight"], full["w13_weight_scale"], expected_counts).chunk(2, -1)
                expected = _gemm(
                    torch.nn.functional.silu(gate) * up, full["w2_weight"], full["w2_weight_scale"], expected_counts
                )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                assert {name: tensor.data_ptr() for name, tensor in resident.items()} == pointers
            finally:
                manager.finish_execution(failed=manager.graph_error is not None)
        manager.begin_execution()
        before = binding.residency.logical_to_physical()

        def fail_copy(*_args, **_kwargs):
            raise RuntimeError("injected graph transfer failure")

        monkeypatch.setattr(TorchExpertTransfer, "copy_experts", fail_copy)
        ids.fill_(0)
        graph.replay()
        torch.npu.current_stream().synchronize()
        with pytest.raises(RuntimeError, match="rejecting execution output"):
            manager.check_graph_errors()
        assert binding.residency.logical_to_physical() == before
        assert manager.state is OffloadState.FAILED
    finally:
        manager.close()


class _ExpertOffloadAudit(TypedDict):
    steps: list[dict[str, bool | int | str]]
    replays: int


class ExpertOffloadAuditWorker:
    # vLLM installs extension methods on its worker instance.
    model_runner: "NPUModelRunner"

    def expert_offload_test_audit(self, action):
        """Inspect a real worker, and observe dispatch without changing its decisions."""
        runner = self.model_runner
        manager = runner.expert_offload_manager
        assert manager is not None
        if action == "install":
            assert manager.state.name == "READY"
            audit: _ExpertOffloadAudit = {"steps": [], "replays": 0}
            runner._expert_offload_test_audit = audit
            prepare = runner._prepare_inputs_impl

            def observed_prepare(scheduler_output, batch_req_state, batch_desc):
                audit["steps"].append(
                    {
                        "prefill": bool(batch_req_state.has_prefill),
                        "tokens": int(batch_desc.num_tokens),
                        "mode": batch_desc.cg_mode.name,
                    }
                )
                return prepare(scheduler_output, batch_req_state, batch_desc)

            runner._prepare_inputs_impl = observed_prepare
            replay = runner.cudagraph_manager.run_fullgraph

            def observed_replay(*args, **kwargs):
                audit["replays"] += 1
                return replay(*args, **kwargs)

            runner.cudagraph_manager.run_fullgraph = observed_replay
        audit = runner._expert_offload_test_audit
        bindings = tuple(manager.bindings.values())
        snapshot = {
            "state": manager.state.name,
            "steps": list(audit["steps"]),
            "replays": audit["replays"],
            "capture_sizes": list(runner.cudagraph_manager.capture_sizes),
            "layer_names": [binding.layer_name for binding in bindings],
            "loaded": all(binding.host_store.finalized and binding.host_store.prepared for binding in bindings),
            "addresses": {
                binding.layer_name: {name: tensor.data_ptr() for name, tensor in binding.resident.items()}
                for binding in bindings
            },
        }
        if action == "close":
            runner.shutdown()
            snapshot["closed_state"] = manager.state.name
            snapshot["released"] = (
                not manager.bindings
                and not manager.prefill_pool.tensors
                and all(not binding.resident and not binding.graph_staging for binding in bindings)
            )
        return snapshot


@pytest.fixture(autouse=True)
def use_mrv2(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """Write ordinary HF shards with real packed MXFP weights and E8M0 scales."""
    root = tmp_path_factory.mktemp("mxfp4-qwen3-moe")
    hidden, intermediate, experts, layers, vocab = 256, 256, 8, 2, 256
    config = Qwen3MoeConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=512,
        moe_intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=experts,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        bos_token_id=0,
        eos_token_id=1,
        architectures=["Qwen3MoeForCausalLM"],
        quantization_config={"quant_method": "ascend", "model_quant_type": "W4A8_MXFP"},
    )
    config.save_pretrained(root)
    generator = torch.Generator().manual_seed(17)
    weights = {}

    def floating(name, shape, *, norm=False):
        weights[name] = (
            torch.ones(shape, dtype=torch.bfloat16)
            if norm
            else (torch.randn(shape, generator=generator) * 0.02).to(torch.bfloat16)
        )

    floating("model.embed_tokens.weight", (vocab, hidden))
    floating("model.norm.weight", (hidden,), norm=True)
    floating("lm_head.weight", (vocab, hidden))
    description = {"group_size": 32, "model_quant_type": "W4A8_MXFP"}
    for index in range(layers):
        prefix = f"model.layers.{index}"
        for name in ("input_layernorm", "post_attention_layernorm"):
            floating(f"{prefix}.{name}.weight", (hidden,), norm=True)
        for name, output in (("q_proj", hidden), ("k_proj", 128), ("v_proj", 128), ("o_proj", hidden)):
            floating(f"{prefix}.self_attn.{name}.weight", (output, hidden))
        for name in ("q_norm", "k_norm"):
            floating(f"{prefix}.self_attn.{name}.weight", (64,), norm=True)
        floating(f"{prefix}.mlp.gate.weight", (experts, hidden))
        for expert in range(experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                output, inputs = (hidden, intermediate) if projection == "down_proj" else (intermediate, hidden)
                name = f"{prefix}.mlp.experts.{expert}.{projection}"
                weights[name + ".weight"] = torch.randint(
                    0, 256, (output, inputs // 2), dtype=torch.uint8, generator=generator
                )
                weights[name + ".weight_scale"] = torch.full((output, inputs // 32), 123, dtype=torch.uint8)
                description[name + ".weight"] = "W4A8_MXFP"
    for name in weights:
        if name.endswith(".weight"):
            description.setdefault(name, "FLOAT")
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    (root / "quant_model_description.json").write_text(json.dumps(description), encoding="utf-8")
    return str(root)


def _stop(llm):
    llm.llm_engine.engine_core.shutdown()
    gc.collect()
    torch.npu.empty_cache()


def _generate_batches(llm, batches):
    sampling = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=1)
    result = []
    for batch in batches:
        outputs = llm.generate([{"prompt_token_ids": tokens} for tokens in batch], sampling, use_tqdm=False)
        for output in outputs:
            completion = output.outputs[0]
            logprobs = [entry[token].logprob for token, entry in zip(completion.token_ids, completion.logprobs)]
            assert all(math.isfinite(value) for value in logprobs)
            result.append((list(completion.token_ids), logprobs))
    return result


def test_checkpoint_prefill_graph_decode_capacity_fallback_and_cleanup(tiny_checkpoint):
    """A different model name must pass the same loader/layout contract end to end."""
    prompts = [[2 + (offset + index) % 200 for index in range(40)] for offset in range(3)]
    batches = [[prompts[0]], prompts, [prompts[1]]]
    common = dict(
        model=tiny_checkpoint,
        skip_tokenizer_init=True,
        worker_extension_cls=f"{__name__}.ExpertOffloadAuditWorker",
        dtype="bfloat16",
        quantization="ascend",
        async_scheduling=False,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        kv_cache_memory_bytes=32 * 1024 * 1024,
        enable_prefix_caching=False,
        seed=17,
    )
    llm = LLM(
        **common,
        additional_config={"expert_offload_config": {"enabled": True, "num_device_experts": 4}},
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]},
    )
    try:
        before = llm.collective_rpc("expert_offload_test_audit", args=("install",))[0]
        actual = _generate_batches(llm, batches)
        after = llm.collective_rpc("expert_offload_test_audit", args=("snapshot",))[0]
        assert before["loaded"] and len(before["layer_names"]) == 2
        assert set(before["capture_sizes"]) == {1, 2}
        assert after["addresses"] == before["addresses"]
        assert after["state"] == "READY" and after["replays"] > 0
        assert any(step["prefill"] for step in after["steps"])
        assert any(not step["prefill"] and step["mode"] == "FULL" for step in after["steps"])
        assert any(not step["prefill"] and step["mode"] == "NONE" and step["tokens"] > 2 for step in after["steps"])
        closed = llm.collective_rpc("expert_offload_test_audit", args=("close",))[0]
        assert closed["closed_state"] == "CLOSED" and closed["released"]
    finally:
        _stop(llm)
    reference = LLM(**common, enforce_eager=True)
    try:
        expected = _generate_batches(reference, batches)
    finally:
        _stop(reference)
    assert len(actual) == len(expected)
    for (actual_tokens, actual_logprobs), (expected_tokens, expected_logprobs) in zip(actual, expected):
        assert actual_tokens == expected_tokens
        torch.testing.assert_close(torch.tensor(actual_logprobs), torch.tensor(expected_logprobs), rtol=1e-3, atol=1e-3)
