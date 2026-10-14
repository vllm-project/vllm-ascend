# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ychy7001
"""验证 310P GDN 二维状态表、变长请求和 Graph 的输出及全部状态槽。"""

import json
from functools import partial

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend._310p.ops.fla.gdn_310 import npu_recurrent_gated_delta_rule_310


def reference(q, k, v, beta, g, initial, lengths, table, accepted):
    state = initial.float().clone()
    output = torch.zeros_like(v)
    offset = 0
    ratio = v.shape[1] // q.shape[1]
    for row, length in enumerate(lengths):
        if not length:
            continue
        source = table[row, accepted[row] - 1] if accepted else table[row, 0]
        current = initial[source].float().clone()
        for token in range(offset, offset + length):
            query = q[token].float().repeat_interleave(ratio, dim=0) * q.shape[-1] ** -0.5
            key = k[token].float().repeat_interleave(ratio, dim=0)
            if g is not None:
                current *= g[token].float().exp()[:, None, None]
            delta = (v[token].float() - (current * key[:, None, :]).sum(-1)) * beta[token].float()[:, None]
            current += delta[:, :, None] * key[:, None, :]
            output[token] = (current * query[:, None, :]).sum(-1).half()
            state[table[row, token - offset]] = current
        offset += length
    return output, state.half()


def call_kernel(op, wrapper, q, k, v, beta, state, seq, indices, g, accepted, scale):
    if wrapper:
        cumulative = torch.cat((seq.new_zeros(1), seq.cumsum(0).to(torch.int32)))
        return wrapper(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            None if g is None else g.unsqueeze(0),
            beta.unsqueeze(0),
            state,
            cumulative,
            indices,
            accepted,
            use_qk_l2norm_in_kernel=False,
        ).squeeze(0)
    return op(q, k, v, beta, state, seq, indices, g, None, accepted, scale)


@pytest.mark.parametrize("graph_mode,use_wrapper", [(False, False), (True, False), (True, True)])
def test_recurrent_state_rows(graph_mode, use_wrapper):
    # Exercise dynamic layouts on the same buffers, including graph replay.
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(17856)
    op = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310
    wrapper = npu_recurrent_gated_delta_rule_310 if use_wrapper else None
    passed = 0
    configurations = [
        (2, False, True, 4),
        (3, False, True, 4),
        (2, True, False, 4),
        (2, False, True, 3),
        (2, False, False, 2),
        (1, True, False, 2),
    ]
    for batch, legacy, use_g, width in configurations:
        tokens, nk, nv, dim = (6 if width == 4 else width + 1), 4, 8, 128
        q = torch.nn.functional.normalize(torch.randn(tokens, nk, dim), dim=-1).half()
        k = torch.nn.functional.normalize(torch.randn(tokens, nk, dim), dim=-1).half()
        v = (torch.randn(tokens, nv, dim) * 0.1).half()
        beta = torch.rand(tokens, nv).half()
        g = -torch.rand(tokens, nv) if use_g else None
        # Unique shuffled slots and different random values expose request mixing.
        table = torch.randperm(batch * width).reshape(batch, width).int()
        initial = (torch.randn(batch * width, nv, dim, dim) * 0.1).half()
        state = initial.npu()
        qd, kd, vd, bd = [t.npu() for t in (q, k, v, beta)]
        gd = None if g is None else g.npu()
        seq = torch.zeros(batch, dtype=torch.int32).npu()
        accepted_dev = torch.zeros_like(seq)
        if legacy:
            if batch == 1:
                layouts = [([1], [1])]
                indices = table[0, :1].npu()
            else:
                layouts = [([3, 3], [2, 1])]
                indices = torch.cat([table[0, :3], table[1, :3]]).npu()
        elif batch == 2:
            if width == 4:
                layouts = [
                    ([2, 4], [4, 1]),
                    ([4, 2], [1, 4]),
                    ([3, 3], [4, 4]),
                    ([1, 3], [4, 4]),
                ]
            elif width == 3:
                layouts = [([1, 3], [3, 1]), ([3, 1], [1, 3]), ([2, 2], [3, 3])]
            else:
                layouts = [([1, 2], [2, 1]), ([2, 1], [1, 2])]
            indices = table.npu()
        else:
            layouts = [
                ([0, 2, 4], [0, 4, 1]),
                ([2, 0, 4], [4, 0, 2]),
                ([3, 3, 0], [4, 4, 0]),
            ]
            indices = table.npu()

        call = partial(
            call_kernel,
            op,
            wrapper,
            qd,
            kd,
            vd,
            bd,
            state,
            seq,
            indices,
            gd,
            accepted_dev,
            dim**-0.5,
        )

        seq.copy_(torch.tensor(layouts[0][0], dtype=torch.int32))
        accepted_dev.copy_(torch.tensor(layouts[0][1], dtype=torch.int32))
        call()
        torch.npu.synchronize()
        graph = None
        if graph_mode:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                graph_output = call()
        for repeat in range(3):
            for lengths, accepted in layouts:
                current_table = table.clone()
                if not legacy:
                    for row, length in enumerate(lengths):
                        if not length:
                            current_table[row].fill_(-1)
                    indices.copy_(current_table)
                expected, expected_state = reference(q, k, v, beta, g, initial, lengths, current_table, accepted)
                seq.copy_(torch.tensor(lengths, dtype=torch.int32))
                accepted_dev.copy_(torch.tensor(accepted, dtype=torch.int32))
                state.copy_(initial)
                if graph is None:
                    output = call()
                else:
                    graph_output.fill_(float("nan"))
                    graph.replay()
                    output = graph_output
                torch.npu.synchronize()
                valid_tokens = sum(lengths)
                actual, actual_state = output.cpu()[:valid_tokens], state.cpu()
                expected = expected[:valid_tokens]
                torch.testing.assert_close(actual, expected, rtol=3e-3, atol=2e-3)
                torch.testing.assert_close(actual_state, expected_state, rtol=3e-3, atol=2e-3)
                passed += 1
                print(
                    json.dumps(
                        {
                            "status": "PASS",
                            "graph": graph_mode,
                            "legacy": legacy,
                            "wrapper": bool(wrapper),
                            "state_row_width": width,
                            "valid_tokens": valid_tokens,
                            "allocated_tokens": tokens,
                            "lengths": lengths,
                            "accepted": accepted,
                            "repeat": repeat,
                            "output_max_abs": (actual - expected).abs().max().item(),
                            "state_max_abs": (actual_state - expected_state).abs().max().item(),
                        }
                    ),
                    flush=True,
                )
    print(json.dumps({"passed": passed, "graph": graph_mode}), flush=True)
