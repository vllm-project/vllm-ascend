# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi TP8 BF16 MLA prolog, learned query tail and DCP owner slots on 950."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile


@torch.inference_mode()
def test_kimi_bf16_sfa_prolog_owner_slots_and_graph():
    if get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED:
        pytest.skip("requires Ascend 950")
    if not callable(getattr(pytest.importorskip("cann_ops_transformer"), "mla_prolog", None)):
        pytest.skip("requires cann_ops_transformer.mla_prolog")

    def channel_selector(rows, columns):
        weight = torch.zeros(rows, columns, dtype=torch.bfloat16)
        row = torch.arange(rows)
        weight[row, row % columns] = 1
        return weight.npu()

    def norm(size):
        return SimpleNamespace(weight=torch.ones(size, dtype=torch.bfloat16, device="npu"), variance_epsilon=1e-5)

    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl._quant_type = None
    impl.use_rope = impl.enable_sparse_sfa_c8 = False
    impl.rl_weight_update_enabled = impl.has_indexer = True
    impl.q_lora_rank, impl.kv_lora_rank, impl.qk_rope_head_dim, impl.local_num_heads = 1536, 512, 64, 12
    impl.q_a_layernorm, impl.kv_a_layernorm = norm(1536), norm(512)
    impl.fused_qkv_a_proj = SimpleNamespace(weight=channel_selector(2112, 7168))
    impl.q_proj = SimpleNamespace(weight=channel_selector(12 * 192, 1536))
    impl.W_UK_T = channel_selector(512, 128).T.expand(12, -1, -1).contiguous()
    impl._process_weights_for_fused_prolog_v3()

    hidden_cpu = torch.randn(3, 7168, generator=torch.Generator().manual_seed(73)).to(torch.bfloat16)
    hidden = hidden_cpu.npu()
    slots = torch.tensor([127, -1, 128], device="npu", dtype=torch.int64)
    # Both caches are first-axis-strided views, as in the paired allocator.
    backing = torch.full((2, 128 * 576), 17, dtype=torch.bfloat16, device="npu")
    cache = (backing[:, : 128 * 512].view(2, 128, 1, 512), backing[:, 128 * 512 :].view(2, 128, 1, 64))

    def run():
        return impl._sfa_preprocess_prolog_v3(hidden, cache, None, None, slots)

    def check(result, source, destinations):
        def rms(x):
            return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), eps=1e-5).bfloat16()

        q_c = rms(source[:, :1536])
        q = q_c[:, torch.arange(12 * 192) % 1536].reshape(3, 12, 192)
        torch.testing.assert_close(result[1].cpu(), q[..., torch.arange(512) % 128], rtol=0.01, atol=0.01)
        torch.testing.assert_close(result[2].cpu(), q[..., 128:], rtol=0.01, atol=0.01)
        torch.testing.assert_close(result[3].cpu(), q_c, rtol=0.01, atol=0.01)
        expected_nope = torch.full((256, 512), 17, dtype=torch.bfloat16)
        expected_tail = torch.full((256, 64), 17, dtype=torch.bfloat16)
        for row, dest in enumerate(destinations):
            if dest >= 0:
                expected_nope[dest] = rms(source[row, 1536:2048])
                expected_tail[dest] = source[row, 2048:2112]
        torch.testing.assert_close(cache[0].cpu().reshape(256, 512), expected_nope, rtol=0.01, atol=0.01)
        torch.testing.assert_close(cache[1].cpu().reshape(256, 64), expected_tail, rtol=0, atol=0)

    check(run(), hidden_cpu, [127, -1, 128])
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = run()
    backing.fill_(17)
    hidden.copy_(-hidden_cpu)
    slots.copy_(torch.tensor([-1, 129, 2]))
    graph.replay()
    torch.npu.synchronize()
    check(result, -hidden_cpu, [-1, 129, 2])
