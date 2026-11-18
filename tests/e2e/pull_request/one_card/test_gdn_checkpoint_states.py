# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def _init_triton_device_properties() -> None:
    # Standalone kernel tests bypass the worker's device initialization.
    init_device_properties_triton()


@pytest.fixture(scope="module")
def _init_checkpoint_prefill_ops() -> None:
    # Registration must happen in this pytest process, as in worker startup.
    if not enable_custom_op():
        pytest.fail("GDN checkpoint prefill tests require the vllm-ascend custom ops extension")
    assert hasattr(torch.ops._C_ascend, "chunk_fwd_o_vllm"), (
        "The installed vllm-ascend extension does not register chunk_fwd_o_vllm"
    )


@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_internal_state_matches_cache_roundtrip(cache_dtype):
    torch.manual_seed(17)
    device = "npu"
    k = torch.randn(1, 192, 2, 128, device=device, dtype=torch.bfloat16) * 0.01
    w = torch.randn_like(k) * 0.01
    u = torch.randn_like(k) * 0.01
    g = torch.zeros(1, 2, 192, device=device, dtype=torch.float32)
    initial = torch.randn(1, 2, 128, 128, device=device, dtype=torch.float32) * 0.01
    states = torch.empty_like(initial)
    _, _, full_final = chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g,
        initial,
        True,
        checkpoint_output=states,
        checkpoint_offsets=torch.tensor([128], dtype=torch.int32, device=device),
        checkpoint_state_dtype=cache_dtype,
    )
    _, _, truncated = chunk_gated_delta_rule_fwd_h(
        k[:, :128].contiguous(),
        w[:, :128].contiguous(),
        u[:, :128].contiguous(),
        g[:, :, :128].contiguous(),
        initial,
        True,
    )
    assert full_final is not None and truncated is not None
    assert states.dtype == torch.float32
    torch.testing.assert_close(states.transpose(-1, -2), truncated.to(cache_dtype).float(), rtol=0, atol=0)
    _, _, resumed_final = chunk_gated_delta_rule_fwd_h(
        k[:, 128:].contiguous(),
        w[:, 128:].contiguous(),
        u[:, 128:].contiguous(),
        g[:, :, 128:].contiguous(),
        states.to(cache_dtype).transpose(-1, -2).contiguous(),
        True,
    )
    torch.testing.assert_close(resumed_final, full_final, rtol=1e-5, atol=1e-6)


def test_shared_exporter_preserves_ascend_conv_layout_and_mtp_slots():
    from vllm.model_executor.layers.mamba.checkpoint import (
        MambaPrefillCheckpointExporter,
        MambaPrefillCheckpointMetadata,
    )

    device = "npu"
    raw = torch.arange(352 * 16, device=device, dtype=torch.float32).view(352, 16)
    conv = torch.full((5, 6, 16), -42.0, device=device)
    recurrent = torch.full((5, 2, 128, 128), -42.0, device=device)
    states = torch.randn(3, 2, 128, 128, device=device)
    checkpoint = MambaPrefillCheckpointMetadata(
        checkpoint_offsets=torch.tensor([64, 0, 128], dtype=torch.int32, device=device),
        state_indices=torch.tensor([2, 0, 4], dtype=torch.int64, device=device),
    )
    MambaPrefillCheckpointExporter(state_len=3).export(
        checkpoint,
        raw_qkv=raw,
        conv_state=conv.transpose(-1, -2),
        recurrent_checkpoint=states,
        recurrent_state=recurrent,
        cu_seqlens=torch.tensor([0, 96, 160, 352], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(conv[2, :3], raw[61:64], rtol=0, atol=0)
    torch.testing.assert_close(conv[4, :3], raw[285:288], rtol=0, atol=0)
    torch.testing.assert_close(recurrent[2], states[0], rtol=0, atol=0)
    torch.testing.assert_close(recurrent[4], states[2], rtol=0, atol=0)
    assert (conv[:, 3:] == -42).all()
    assert (conv[0] == -42).all()
    assert (recurrent[0] == -42).all()


@pytest.mark.usefixtures("_init_checkpoint_prefill_ops")
@pytest.mark.parametrize("offset,length", [(1408, 1464), (1280, 1336), (256, 315)])
def test_ascendc_segments_preserve_checkpoint_and_final_state(offset, length):
    from vllm_ascend.ops.gdn_ascendc_checkpoint import (
        ascendc_checkpoint_recurrence,
        build_ascendc_checkpoint_plan,
    )

    torch.manual_seed(31)
    device = torch.device("npu")
    k = torch.randn(1, 2, length, 128, device=device, dtype=torch.bfloat16) * 0.01
    w, u = torch.randn_like(k) * 0.01, torch.randn_like(k) * 0.01
    g = torch.zeros(1, 2, length, device=device, dtype=torch.float32)
    initial = torch.randn(1, 2, 128, 128, device=device, dtype=torch.float32) * 0.01
    states = torch.empty_like(initial)
    plan = build_ascendc_checkpoint_plan((0, length), [offset], device)
    _, _, actual_final = ascendc_checkpoint_recurrence(k, w, u, g, initial, states, plan)

    def reference(start, end, state):
        return torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h_vllm(
            k[:, :, start:end].contiguous(),
            w[:, :, start:end].contiguous(),
            u[:, :, start:end].contiguous(),
            g=g[:, :, start:end].contiguous(),
            initial_state=state,
            output_final_state=True,
            chunk_size=64,
            cu_seqlens=(0, end - start),
            chunk_indices=tuple(value for i in range((end - start + 63) // 64) for value in (0, i)),
        )[2]

    prefix = reference(0, offset, initial)
    final = reference(offset, length, prefix)
    torch.testing.assert_close(states.transpose(-1, -2), prefix, rtol=0, atol=0)
    torch.testing.assert_close(actual_final, final, rtol=0, atol=0)


@pytest.mark.parametrize("offset,length", [(1408, 1464), (1280, 1336), (256, 315)])
def test_fused_segments_preserve_checkpoint_and_restored_tail(offset, length):
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
    from vllm_ascend.ops.gdn_ascendc_checkpoint import build_ascendc_checkpoint_plan
    from vllm_ascend.ops.gdn_fused_checkpoint import fused_gdn_checkpoint_prefill

    if not get_current_hardware_profile().supports(HardwareCapability.FLA_GDN_PREFILL):
        pytest.skip("The fused GDN prefill kernel requires A5")
    chunk_gated_delta_rule_fwd = getattr(pytest.importorskip("fla_npu.ops.ascendc"), "chunk_gated_delta_rule_fwd", None)
    if chunk_gated_delta_rule_fwd is None:
        pytest.skip("The installed FLA NPU package lacks fused GDN prefill")
    torch.manual_seed(41)
    device = torch.device("npu")
    q, k, v = [torch.randn(1, length, 4, 128, device=device, dtype=torch.bfloat16) * 0.1 for _ in range(3)]
    g = -torch.rand(1, length, 4, device=device) * 0.01
    beta = torch.rand(1, length, 4, device=device, dtype=torch.bfloat16)
    initial = torch.randn(1, 4, 128, 128, device=device, dtype=torch.float32) * 0.01
    snapshots = torch.empty_like(initial)
    plan = build_ascendc_checkpoint_plan((0, length), [offset], device)
    output, final = fused_gdn_checkpoint_prefill(q, k, v, g, beta, initial, snapshots, plan, chunk_gated_delta_rule_fwd)

    def reference(start, end, state):
        metadata = SimpleNamespace(
            cu_seqlens_host=(0, end - start),
            chunk_indices_chunk64_host=tuple(value for i in range((end - start + 63) // 64) for value in (0, i)),
            keep_meta=None,
        )
        return DeviceOperator.fla_gdn_prefill(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            state,
            128**-0.5,
            metadata,
            chunk_gated_delta_rule_fwd,
        )

    prefix_output, prefix = reference(0, offset, initial)
    tail_output, expected_final = reference(offset, length, prefix)
    torch.testing.assert_close(snapshots, prefix, rtol=0, atol=0)
    torch.testing.assert_close(final, expected_final, rtol=0, atol=0)
    torch.testing.assert_close(output, torch.cat((prefix_output, tail_output), dim=1), rtol=0, atol=0)


def test_cache_hit_prefill_exports_tail_in_one_model_step(monkeypatch, model_name="Qwen/Qwen3.5-35B-A3B"):
    """Exercise private checkpoint pages through the real two-layer engine."""
    from vllm import LLM, SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    steps = []
    original_schedule = Scheduler.schedule

    def record_schedule(self, *args, **kwargs):
        output = original_schedule(self, *args, **kwargs)
        steps.append(dict(output.num_scheduled_tokens))
        return output

    def two_layer_config(config):
        text = getattr(config, "text_config", config)
        text.num_hidden_layers = 2
        text.layer_types = ["linear_attention", "full_attention"]
        text.num_experts = 4
        text.num_experts_per_tok = 2
        return config

    monkeypatch.setattr(Scheduler, "schedule", record_schedule)
    engine = LLM(
        model=model_name,
        load_format="dummy",
        hf_overrides=two_layer_config,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        prefix_match_unit=128,
        block_size=2048,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=4,
        gpu_memory_utilization=0.2,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={"gdn_internal_prefill_checkpoints": True},
    )
    params = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0)
    for hit, end in [(2048, 3512), (2176, 3512), (896, 1211)]:
        assert engine.reset_prefix_cache()
        seed = [i % 1000 + 100 for i in range(hit + 1)]
        engine.generate([{"prompt_token_ids": seed}], params, use_tqdm=False)
        prompt = seed[:hit] + [2000 + i % 1000 for i in range(end - hit)]
        steps.clear()
        output = engine.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0]
        assert output.num_cached_tokens == hit
        assert [sum(row.values()) for row in steps if row] == [end - hit]
        steps.clear()
        replay = engine.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0]
        boundary = (end - 1) // 128 * 128
        assert replay.num_cached_tokens == boundary
        assert [sum(row.values()) for row in steps if row] == [end - boundary]


@pytest.mark.usefixtures("_init_checkpoint_prefill_ops")
@pytest.mark.parametrize("layout", ["head_first", "token_slice"])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_checkpoint_prefill_matches_dense_inputs_and_restored_tail(layout, cache_dtype):
    from vllm_ascend.ops.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata
    from vllm_ascend.ops.triton.fla import chunk

    torch.manual_seed(23)
    lengths = [145, 65, 209]
    total, heads, dim = sum(lengths), 4, 128

    def strided_input():
        if layout == "head_first":
            return torch.randn(heads, total, dim, device="npu", dtype=torch.bfloat16).transpose(0, 1).unsqueeze(0)
        return torch.randn(1, total * 2, heads, dim, device="npu", dtype=torch.bfloat16)[:, ::2]

    q, k, v = (strided_input() for _ in range(3))
    assert not v.is_contiguous()
    g = -torch.rand(1, total * 2, heads, device="npu")[:, ::2] * 0.01
    beta = torch.rand(1, total * 2, heads, device="npu", dtype=torch.bfloat16)[:, ::2]
    initial = torch.randn(3, heads, dim, dim, device="npu", dtype=torch.float32) * 0.01
    cu_cpu = torch.tensor([0, 145, 210, total], dtype=torch.int32)
    cu = cu_cpu.to("npu")
    builder = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(linear_num_value_heads=heads)),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
    )
    metadata = _build_non_spec_chunked_prefill_metadata(builder, cu_cpu, torch.device("npu"))
    offsets = torch.tensor([128, 0, 192], device="npu", dtype=torch.int32)
    snapshots = torch.full_like(initial, float("nan"))
    dense_snapshots = torch.full_like(initial, float("nan"))
    with (
        patch.object(chunk, "get_forward_context", return_value=SimpleNamespace(attn_metadata=None)),
        patch.object(chunk, "get_pcp_group", return_value=SimpleNamespace(world_size=1)),
    ):
        actual = chunk.chunk_gated_delta_rule_checkpoint(
            q,
            k,
            v,
            g,
            beta,
            initial,
            cu,
            metadata,
            snapshots,
            offsets,
            checkpoint_state_dtype=cache_dtype,
        )
        expected = chunk.chunk_gated_delta_rule_checkpoint(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            initial,
            cu,
            metadata,
            dense_snapshots,
            offsets,
            checkpoint_state_dtype=cache_dtype,
        )
        # A real cache hit restores the cache dtype and computes only the tail;
        # no new checkpoint is created, but the recurrence must stay the same.
        tail_cu_cpu = torch.tensor([0, 17], dtype=torch.int32)
        tail_metadata = _build_non_spec_chunked_prefill_metadata(builder, tail_cu_cpu, torch.device("npu"))
        restored = chunk.chunk_gated_delta_rule_checkpoint(
            q[:, 128:145],
            k[:, 128:145],
            v[:, 128:145],
            g[:, 128:145],
            beta[:, 128:145],
            snapshots[:1].to(cache_dtype).transpose(-1, -2).contiguous(),
            tail_cu_cpu.to("npu"),
            tail_metadata,
            None,
            None,
            checkpoint_state_dtype=cache_dtype,
        )
    torch.testing.assert_close(restored[0], actual[0][:, 128:145], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(restored[1], actual[1][:1], rtol=1e-4, atol=1e-5)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(snapshots[[0, 2]], dense_snapshots[[0, 2]], rtol=1e-5, atol=1e-6)
    assert torch.isfinite(snapshots[[0, 2]]).all()
    assert torch.isnan(snapshots[1]).all()
