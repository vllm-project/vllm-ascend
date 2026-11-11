# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-weight HCCL/MXFP8/graph checks; no model download or service launch.

Run directly with torchrun --standalone --nproc-per-node=2 FILE --worker,
or through pytest. The stateless oracle exercises projection/layout contracts,
not the accuracy of recurrent KDA or the complete Kimi model.
"""

import itertools
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch_npu
from torch.nn import functional as F

from vllm_ascend.ops import finegrained_tp as fine
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod
from vllm_ascend.utils import is_950


def _linear(weight):
    class Linear:
        bias = None

        def __init__(self):
            self.weight = weight
            self.quant_method = SimpleNamespace(apply=lambda layer, x, bias=None: F.linear(x, layer.weight, bias))

        def __call__(self, hidden):
            return F.linear(hidden, weight), None

    return Linear()


def _mx_weight(weight):
    quantized, scale = torch_npu.npu_dynamic_mx_quant(weight, dst_type=torch.float8_e4m3fn, scale_alg=0)
    # 950DT uses ND transpose views; weight and MX scales must agree on transpose.
    scale = scale.reshape(weight.shape[0], -1, 2).transpose(0, 1)
    return quantized.t(), scale


def _mx_apply(hidden, weight):
    quantized, scale = hidden
    return torch_npu.npu_quant_matmul(
        quantized,
        weight[0],
        weight[1],
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale=scale,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, 32],
    )


@torch.inference_mode()
def _worker():
    local_rank, rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    assert world == 2
    group = SimpleNamespace(world_size=world, rank_in_group=rank, device_group=dist.group.WORLD)
    device = torch.device("npu", local_rank)
    torch.manual_seed(17)
    capacity, hidden_size, heads, dim, intermediate, vocab = 5, 128, 8, 16, 128, 64
    width = heads * dim

    def matrix(rows, columns):
        return (torch.randn(rows, columns) / columns**0.5).to(device=device, dtype=torch.bfloat16)

    weights = {
        "embed": matrix(vocab, hidden_size),
        "head": matrix(vocab, hidden_size),
        "qkv": matrix(3 * width, hidden_size),
        "beta": matrix(heads, hidden_size),
        "fa": matrix(dim, hidden_size),
        "fb": matrix(width, dim),
        "g": matrix(width, hidden_size),
        "o": matrix(hidden_size, width),
        "up": matrix(2 * intermediate, hidden_size),
        "down": matrix(hidden_size, intermediate),
    }
    # Exercise the production merged loader and post-load format as well as
    # the small ND references used by the communication matrix below.
    from vllm_ascend.quantization.method_adapters import AscendLinearMethod

    scheme = object.__new__(AscendW8A8MXFP8DynamicLinearMethod)
    scheme.group_size = 32
    scheme.dynamic_mx_quant_scale_alg = 0
    quant_config = SimpleNamespace(
        get_quant_method=lambda layer, prefix: AscendLinearMethod(scheme), online_quantization_config=None
    )
    loaded = fine.ProjectionMergedColumnParallelLinear(
        hidden_size,
        [width] * 3,
        group=group,
        quant_config=quant_config,
        params_dtype=torch.bfloat16,
        prefix="in_proj_qkv",
    ).to(device)
    for shard, weight in enumerate(weights["qkv"].chunk(3)):
        quantized, scale = torch_npu.npu_dynamic_mx_quant(weight, dst_type=torch.float8_e4m3fn, scale_alg=0)
        loaded.weight_loader(loaded.weight, quantized, shard)
        loaded.weight_loader(loaded.weight_scale, scale, shard)
    scheme.process_weights_after_loading(loaded)
    load_input = matrix(capacity, hidden_size)
    expected_weight = torch.cat([weight.chunk(world)[rank] for weight in weights["qkv"].chunk(3)])
    torch.testing.assert_close(loaded(load_input)[0], F.linear(load_input, expected_weight), atol=0.12, rtol=0.12)

    full_qkv = _mx_weight(weights["qkv"])
    quantize = lambda x: torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, scale_alg=0)
    static_input = matrix(capacity, hidden_size)

    for sizes in itertools.product((1, 2), repeat=6):
        e_size, h_size, m_size, q_size, g_size, o_size = sizes
        qkv_parts = [w.chunk(q_size)[rank % q_size] for w in weights["qkv"].chunk(3)]
        quant_weight = _mx_weight(torch.cat(qkv_parts))
        projection = fine.KDAProjectionTP(group if q_size > 1 else None, group if g_size > 1 else None, heads, dim)
        layer = SimpleNamespace(
            b_proj=_linear(weights["beta"].chunk(q_size)[rank % q_size]),
            f_a_proj=_linear(weights["fa"]),
            f_b_proj=_linear(weights["fb"]),
            g_proj=_linear(weights["g"].chunk(g_size)[rank % g_size]),
            _quantize_fused_qkv=quantize,
            _matmul_fused_qkv=lambda x, weight=quant_weight: _mx_apply(x, weight),
        )
        up = torch.cat([w.chunk(m_size)[rank % m_size] for w in weights["up"].chunk(2)])
        mlp = SimpleNamespace(
            gate_up_proj=_linear(up),
            down_proj=_linear(weights["down"].chunk(m_size, 1)[rank % m_size]),
            _prepare_down_input=lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
        )
        mlp_tp, head_tp = fine.ChunkedMLPTP(group), fine.ChunkedLMHeadTP(group)
        head = _linear(weights["head"].chunk(h_size)[rank % h_size])
        processor = SimpleNamespace(
            org_vocab_size=vocab, _apply_head=lambda layer, x, bias: F.linear(x, layer.weight, bias)
        )
        communication = fine.TPCommunication()

        def reference(hidden, mlp=mlp):
            qkv = _mx_apply(quantize(hidden), full_qkv)
            forget = F.linear(F.linear(hidden, weights["fa"]), weights["fb"])
            beta = F.linear(hidden, weights["beta"]).float().sigmoid().to(hidden.dtype)
            value = (qkv.chunk(3, -1)[2] + forget) * beta.repeat_interleave(dim, -1)
            value = value * F.linear(hidden, weights["g"]).sigmoid()
            value = F.linear(value, weights["o"])
            value = F.linear(mlp._prepare_down_input(F.linear(value, weights["up"])), weights["down"])
            return F.linear(value, weights["head"])

        def execute(
            hidden,
            q_size=q_size,
            g_size=g_size,
            projection=projection,
            layer=layer,
            o_size=o_size,
            communication=communication,
            mlp_tp=mlp_tp,
            mlp=mlp,
            m_size=m_size,
            head_tp=head_tp,
            processor=processor,
            head=head,
            h_size=h_size,
        ):
            if q_size > 1 or g_size > 1:
                qkv, forget, beta, gate = projection.project(layer, hidden)
                value = (qkv.chunk(3, -1)[2] + forget[0].flatten(1)) * beta[0].float().sigmoid().to(
                    hidden.dtype
                ).repeat_interleave(dim, -1)
                value = value * gate.wait().flatten(1).sigmoid()
            else:
                qkv = _mx_apply(quantize(hidden), full_qkv)
                forget = F.linear(F.linear(hidden, weights["fa"]), weights["fb"])
                beta = F.linear(hidden, weights["beta"]).float().sigmoid().to(hidden.dtype)
                value = (qkv.chunk(3, -1)[2] + forget) * beta.repeat_interleave(dim, -1)
                value = value * F.linear(hidden, weights["g"]).sigmoid()
            if o_size > 1:
                source = value.reshape(capacity, world, width // world).transpose(0, 1).contiguous()
                partial_heads = communication.exchange(source, group).wait().reshape(world * capacity, width // world)
                partial = F.linear(partial_heads, weights["o"].chunk(world, 1)[rank])
                value = communication.scatter(partial, group).wait()
            else:
                value = F.linear(value, weights["o"])
            value = (
                mlp_tp.forward(mlp, value)
                if m_size > 1
                else F.linear(mlp._prepare_down_input(F.linear(value, weights["up"])), weights["down"])
            )
            return head_tp.forward(processor, value, head, None) if h_size > 1 else F.linear(value, weights["head"])

        for step in range(2):
            ids = (torch.arange(capacity, device=device) + rank * 7 + step) % vocab
            if e_size > 1:
                gathered_ids = communication.gather(ids, group).wait()
                lo, hi = rank * vocab // world, (rank + 1) * vocab // world
                lookup = weights["embed"][gathered_ids] * ((gathered_ids >= lo) & (gathered_ids < hi)).unsqueeze(-1)
                hidden = communication.scatter(lookup, group).wait()
            else:
                hidden = weights["embed"][ids]
            if rank == 0 and step == 0:
                hidden.zero_()  # An idle owner still serves the peer's weight shard.
            torch.testing.assert_close(execute(hidden).float(), reference(hidden).float(), atol=2e-5, rtol=0.06)

        if sizes == (2,) * 6:
            # Every stream and process group is warm before capture.
            for _ in range(3):
                execute(static_input)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                graph_output = execute(static_input)
            for step in range(2):
                static_input.copy_(weights["embed"][torch.arange(capacity, device=device) + rank + step])
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(graph_output.float(), reference(static_input).float(), atol=2e-5, rtol=0.06)

    # DSA uses the same full-rank gate transport independently of KDA QKV.
    gate_tp = fine.FullRankGateTP(group, heads, dim)
    ticket = gate_tp.gather(static_input)
    gate = gate_tp.project(ticket, _linear(weights["g"].chunk(world)[rank])).wait()
    torch.testing.assert_close(gate.flatten(1), F.linear(static_input, weights["g"]), atol=1e-3, rtol=0.02)
    torch.npu.synchronize()
    dist.destroy_process_group()
    print(f"rank={rank}: 64 combinations, MXFP8 QKV, independent G, and graph replay passed", flush=True)


@pytest.mark.skipif(
    not torch.npu.is_available() or torch.npu.device_count() < 2 or not is_950(), reason="requires two Ascend 950 NPUs"
)
def test_kimi_localtp_exchanges_and_graph():
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", __file__, "--worker"],
        check=True,
        timeout=300,
    )


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
