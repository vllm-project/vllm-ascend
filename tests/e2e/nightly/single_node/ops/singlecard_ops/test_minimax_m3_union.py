# SPDX-License-Identifier: Apache-2.0
"""Optional custom-Union correctness, request boundaries and NPUGraph replay."""

import pytest
import torch

from vllm_ascend.models.minimax_m3.ops.msa_m3_union import UnionSparseAttention
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("capture", [False, True])
def test_union_matches_exact_token_visibility(dtype, capture):
    if not hasattr(torch, "npu") or not torch.npu.is_available() or get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("Custom Union requires Ascend950")
    pytest.importorskip("cann_ops_nn_custom")
    pytest.importorskip("cann_ops_transformer_custom")
    torch.manual_seed(0)
    device = "npu"
    # Both requests end in partial groups. Their physical block tables differ
    # and their final selected blocks are only partly visible (causal mask).
    q_lens, kv_lens = [9, 17], [300, 400]
    q = torch.randn(sum(q_lens), 16, 128, device=device, dtype=dtype)
    key = torch.randn(8, 128, 1, 128, device=device, dtype=dtype)
    value = torch.randn_like(key)
    table_cpu = torch.tensor([[3, 0, 5, 2], [6, 1, 7, 4]], dtype=torch.int32)
    table = table_cpu.to(device)
    indices_cpu = torch.tensor([[[2, 0, -1] if t % 2 else [1, 0, -1] for t in range(sum(q_lens))]])
    indices = indices_cpu.to(device=device, dtype=torch.int32)
    cu_q = torch.tensor([0, 9, 26], dtype=torch.int32, device=device)
    cu_kv = torch.tensor([0, 300, 700], dtype=torch.int32, device=device)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    output = torch.empty_like(q)
    attention = UnionSparseAttention()

    def run():
        attention(q, (key, value), indices, table, cu_q, cu_kv, seq_lens, 5, 4, 128**-0.5, output)

    graph = None
    if capture:
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                run()
        torch.npu.current_stream().wait_stream(stream)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()

    for _ in range(2):
        q.copy_(torch.randn_like(q))
        if graph is None:
            run()
        else:
            graph.replay()
        torch.npu.synchronize()
        query_cpu, key_cpu, value_cpu = q.cpu().float(), key.cpu().float(), value.cpu().float()
        expected = torch.empty_like(query_cpu)
        start = 0
        for request, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
            logical_k = key_cpu[table_cpu[request].long()].reshape(-1, 128)
            logical_v = value_cpu[table_cpu[request].long()].reshape(-1, 128)
            for offset in range(q_len):
                token = start + offset
                visible = []
                for block in indices_cpu[0, token].tolist():
                    if block >= 0:
                        visible.extend(range(block * 128, min((block + 1) * 128, kv_len - q_len + offset + 1)))
                selected = torch.tensor(visible, dtype=torch.long)
                scores = query_cpu[token] @ logical_k[selected].T * 128**-0.5
                expected[token] = torch.softmax(scores, dim=-1) @ logical_v[selected]
            start += q_len
        tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
        torch.testing.assert_close(output.cpu().float(), expected, atol=tolerance, rtol=tolerance)
