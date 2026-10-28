# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

"""MiniMax Flash draft attention precision and device-length regressions."""

import math
from types import SimpleNamespace as NS

import pytest
import torch
import torch_npu  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.fa_v1 import (
    AscendFlashAttentionImpl as MiniMaxDSparkFlashImpl,
)
from vllm_ascend.attention.fa_v1 import (
    AscendFlashAttentionMetadataBuilder as MiniMaxDSparkFlashMetadataBuilder,
)
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

pytestmark = pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.DEVICE_LENGTH_FLASH_ATTN),
    reason="Requires the A5 CANN FlashAttn stack",
)


class NoDeviceHostRead(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.cpu_scalar_reads = 0
        self.npu_scalar_reads = 0
        self.device_to_host_copies = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        device_inputs = [
            x for x in tree_leaves((args, kwargs)) if isinstance(x, torch.Tensor) and x.device.type == "npu"
        ]
        if device_inputs:
            if "_local_scalar_dense" in str(func):
                self.npu_scalar_reads += 1
            assert "_local_scalar_dense" not in str(func), f"NPU scalar synchronization: {func}"
            if "_to_copy" in str(func) or "copy_" in str(func):
                destination = kwargs.get("device")
                if destination is not None:
                    if torch.device(destination).type == "cpu":
                        self.device_to_host_copies += 1
                    assert torch.device(destination).type != "cpu", f"NPU to CPU: {func}"
                if "copy_" in str(func) and isinstance(args[0], torch.Tensor):
                    if args[0].device.type == "cpu":
                        self.device_to_host_copies += 1
                    assert args[0].device.type != "cpu", f"NPU to CPU copy: {func}"
        elif "_local_scalar_dense" in str(func):
            self.cpu_scalar_reads += 1
        return func(*args, **kwargs)


def test_full_decode_graph_replay_refreshes_rejected_device_lengths():
    torch.manual_seed(18717)
    q = torch.randn(16, 16, 128, dtype=torch.bfloat16)
    k = torch.randn(4, 128, 1, 128, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    query, key, value = q.npu(), k.npu(), v.npu()
    output = torch.empty_like(query)
    impl = object.__new__(MiniMaxDSparkFlashImpl)
    impl.num_heads, impl.num_kv_heads, impl.sinks = 16, 1, None
    impl.scale, impl.sliding_window, impl.use_bnsd_kv_cache = 1 / math.sqrt(128), 4096, False
    impl.key_cache, impl.value_cache = key, value
    common = make_common([8, 8], [129, 63])
    builder = make_builder()
    for _ in range(3):
        impl.forward_impl(query, None, None, (key, value), builder.build(0, common), output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    metadata = builder.build(0, common)
    with torch.npu.graph(graph):
        impl.forward_impl(query, None, None, (key, value), metadata, output)
    graph.replay()
    expected = reference(q, k, v, [8, 8], [129, 63], True, 4096)
    torch.testing.assert_close(output.cpu().float(), expected, atol=0.025, rtol=0.025)
    common.seq_lens.copy_(torch.tensor([73, 197], dtype=torch.int32, device="npu"))
    with NoDeviceHostRead():
        graph.replay()
    expected = reference(q, k, v, [8, 8], [73, 197], True, 4096)
    torch.testing.assert_close(output.cpu().float(), expected, atol=0.025, rtol=0.025)


def make_common(q_lengths, kv_lengths, device="npu"):
    qsl = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32)
    n = len(q_lengths)
    return NS(
        num_actual_tokens=sum(q_lengths),
        num_reqs=n,
        max_query_len=max(q_lengths),
        max_seq_len=256,
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        # Deliberately wrong CPU lengths must never reach an attention kernel.
        seq_lens_cpu=torch.full((n,), 256, dtype=torch.int32),
        seq_lens=torch.tensor(kv_lengths, dtype=torch.int32, device=device),
        block_table_tensor=torch.arange(n * 2, dtype=torch.int32, device=device).view(n, 2),
        slot_mapping=torch.arange(sum(q_lengths), dtype=torch.int64, device=device),
        causal=True,
        attn_state=AscendAttentionState.SpecDecoding,
        context_parallel_metadata=None,
    )


def make_builder():
    builder = object.__new__(MiniMaxDSparkFlashMetadataBuilder)
    builder.model_config = NS(runner_type="generate")
    builder.attn_mask_builder = AttentionMaskBuilder(torch.device("npu"))
    builder.attn_mask_builder.get_splitfuse_attn_mask()
    return builder


def reference(query, key, value, q_lengths, kv_lengths, causal, window):
    outputs = []
    q_offset = 0
    for i, (ql, kl) in enumerate(zip(q_lengths, kv_lengths)):
        if ql == 0:
            continue
        q = query[q_offset : q_offset + ql].float().transpose(0, 1)
        k = (
            key[i * 2 : i * 2 + 2]
            .reshape(256, 1, 128)[:kl]
            .float()
            .repeat_interleave(query.shape[1], dim=1)
            .transpose(0, 1)
        )
        v = (
            value[i * 2 : i * 2 + 2]
            .reshape(256, 1, 128)[:kl]
            .float()
            .repeat_interleave(query.shape[1], dim=1)
            .transpose(0, 1)
        )
        scores = q @ k.transpose(1, 2) / math.sqrt(128)
        qpos = torch.arange(kl - ql, kl)[:, None]
        kpos = torch.arange(kl)[None, :]
        keep = torch.ones((ql, kl), dtype=torch.bool)
        if causal:
            keep &= kpos <= qpos
        if window is not None:
            keep &= kpos >= qpos - (window - 1)
            if not causal:
                keep &= kpos <= qpos + (window - 1)
        scores.masked_fill_(~keep[None], float("-inf"))
        outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
        q_offset += ql
    return torch.cat(outputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal,window", [(True, None), (False, None), (True, 16), (False, 16), (False, 4096)])
@pytest.mark.parametrize("layout", ["PA_BBND", "PA_BNBD"])
@pytest.mark.parametrize("width", [7, 8])
def test_flash_attention_matches_reference_after_rejection_and_padding(dtype, causal, window, layout, width):
    torch.manual_seed(18717)
    q_lengths, kv_lengths = [width, width, 0], [129, 63, 0]
    q_cpu = torch.randn(width * 2, 16, 128, dtype=dtype)
    k_cpu = torch.randn(6, 128, 1, 128, dtype=dtype)
    v_cpu = torch.randn_like(k_cpu)
    query, key, value = q_cpu.npu(), k_cpu.npu(), v_cpu.npu()
    impl = object.__new__(MiniMaxDSparkFlashImpl)
    impl.num_heads, impl.num_kv_heads = 16, 1
    impl.scale, impl.sliding_window, impl.sinks = 1 / math.sqrt(128), window, None
    impl.use_bnsd_kv_cache = layout == "PA_BNBD"
    if impl.use_bnsd_kv_cache:
        key, value = key.permute(0, 2, 1, 3).contiguous(), value.permute(0, 2, 1, 3).contiguous()
    impl.key_cache, impl.value_cache = key, value
    common = make_common(q_lengths, kv_lengths)
    common.causal = causal
    builder = make_builder()
    metadata = builder.build(0, common)
    output = torch.empty_like(query)
    with NoDeviceHostRead():
        impl.forward_impl(query, None, None, (key, value), metadata, output)
        first_plan = next(iter(metadata.flash_attn_plans.values()))[0]
        impl.forward_impl(query, None, None, (key, value), metadata, output)
        assert first_plan is next(iter(metadata.flash_attn_plans.values()))[0]
    expected = reference(q_cpu, k_cpu, v_cpu, q_lengths, kv_lengths, causal, window)
    torch.testing.assert_close(output.cpu().float(), expected, atol=0.025, rtol=0.025)


def test_133k_context_swa_preserves_device_lengths():
    torch.manual_seed(18717)
    length, block_size, window = 133000, 128, 4096
    blocks = (length + block_size - 1) // block_size
    q = torch.randn(8, 16, 128, dtype=torch.bfloat16)
    k = torch.randn(blocks, block_size, 1, 128, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    query, key, value = q.npu(), k.npu(), v.npu()
    impl = object.__new__(MiniMaxDSparkFlashImpl)
    impl.num_heads, impl.num_kv_heads, impl.sinks = 16, 1, None
    impl.scale, impl.sliding_window, impl.use_bnsd_kv_cache = 1 / math.sqrt(128), window, False
    impl.key_cache, impl.value_cache = key, value
    common = make_common([8], [length])
    common.max_seq_len = length
    common.block_table_tensor = torch.arange(blocks, dtype=torch.int32, device="npu").view(1, -1)
    builder = make_builder()
    output = torch.empty_like(query)
    with NoDeviceHostRead():
        impl.forward_impl(query, None, None, (key, value), builder.build(0, common), output)
    # The earliest query sees 4096 positions; later queries slide forward.
    start = length - 8 - window + 1
    keys = k.view(-1, 1, 128)[start:length].float().repeat_interleave(16, dim=1).transpose(0, 1)
    values = v.view(-1, 1, 128)[start:length].float().repeat_interleave(16, dim=1).transpose(0, 1)
    scores = q.float().transpose(0, 1) @ keys.transpose(1, 2) / math.sqrt(128)
    qpos = torch.arange(length - 8, length)[:, None]
    kpos = torch.arange(start, length)[None, :]
    keep = (kpos <= qpos) & (kpos >= qpos - window + 1)
    scores.masked_fill_(~keep[None], float("-inf"))
    expected = (scores.softmax(-1) @ values).transpose(0, 1)
    torch.testing.assert_close(output.cpu().float(), expected, atol=0.025, rtol=0.025)


def test_attention_forward_writes_current_draft_kv_before_reading():
    from vllm.v1.attention.backend import AttentionType

    torch.manual_seed(18717)
    q_lengths, kv_lengths = [7, 7], [129, 63]
    q = torch.randn(14, 16, 128, dtype=torch.bfloat16)
    k = torch.randn(4, 128, 1, 128, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    new_k = torch.randn(14, 1, 128, dtype=torch.bfloat16)
    new_v = torch.randn_like(new_k)
    slots = []
    offset = 0
    for index, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths)):
        k[index * 2 : index * 2 + 2].view(256, 1, 128)[kv_len - q_len : kv_len] = new_k[offset : offset + q_len]
        v[index * 2 : index * 2 + 2].view(256, 1, 128)[kv_len - q_len : kv_len] = new_v[offset : offset + q_len]
        slots.extend(range(index * 256 + kv_len - q_len, index * 256 + kv_len))
        offset += q_len
    cache_k, cache_v = torch.zeros_like(k).npu(), torch.zeros_like(v).npu()
    # Seed only historical context; the attention entry point must insert Q's KV.
    context_k, context_v = k.clone(), v.clone()
    for slot in slots:
        context_k.view(-1, 1, 128)[slot] = 0
        context_v.view(-1, 1, 128)[slot] = 0
    cache_k.copy_(context_k.npu())
    cache_v.copy_(context_v.npu())
    impl = object.__new__(MiniMaxDSparkFlashImpl)
    impl.num_heads, impl.num_kv_heads, impl.sinks = 16, 1, None
    impl.scale, impl.sliding_window, impl.use_bnsd_kv_cache = 1 / math.sqrt(128), 4096, False
    impl.key_cache, impl.value_cache = cache_k, cache_v
    impl.pcp_enabled, impl.is_kv_producer = False, False
    impl.kv_sharing_target_layer_name, impl.attn_type = None, AttentionType.DECODER
    common = make_common(q_lengths, kv_lengths)
    common.causal = False
    common.slot_mapping = torch.tensor(slots, dtype=torch.int64, device="npu")
    metadata = make_builder().build(0, common)
    layer = NS(layer_name="draft.layers.0.self_attn.attn", _k_scale_float=1.0, _v_scale_float=1.0)
    query, keys, values = q.npu(), new_k.npu(), new_v.npu()
    output = torch.empty_like(query)
    with NoDeviceHostRead():
        impl.forward(layer, query, keys, values, (cache_k, cache_v), metadata, output)
    expected = reference(q, k, v, q_lengths, kv_lengths, False, 4096)
    torch.testing.assert_close(output.cpu().float(), expected, atol=0.025, rtol=0.025)
