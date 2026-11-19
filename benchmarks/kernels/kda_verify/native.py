# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from fla_npu.ops.ascendc import causal_conv1d_update
from vllm_ascend.ops.triton.kda.kda import rms_norm_gated
from vllm_ascend.ops.triton.kda.recoverssm import kda_recoverssm_verify


class Native:
    def __init__(self, c, inputs):
        self.c, self.i = c, inputs
        self.conv, self.verify, self.norm = causal_conv1d_update, kda_recoverssm_verify, rms_norm_gated
        b, l, h, d = c.batch, c.tokens, c.heads, c.dim
        self.x = inputs["x"].reshape(b * l, c.hidden)
        self.cw = inputs["conv_w"].permute(3, 0, 1, 2).reshape(4, 3 * h * d).contiguous()
        self.cs = torch.zeros((b + 1, 3 + l - 1, 3 * h * d), device="npu", dtype=torch.bfloat16)
        self.cs[1:, :3] = inputs["history"].permute(0, 4, 1, 2, 3).reshape(b, 3, 3 * h * d)
        self.cs_base = self.cs.clone()
        self.state = torch.zeros((b + 1, h, d, d), device="npu", dtype=torch.float32)
        self.state[1:] = inputs["state"]
        self.correction = torch.empty((b + 1, h, l, d), device="npu", dtype=torch.float32)
        self.kg = torch.empty((b + 1, h, l, 2 * d), device="npu", dtype=torch.bfloat16)
        self.starts = torch.arange(b + 1, device="npu", dtype=torch.int32) * l
        self.indices = torch.arange(1, b + 1, device="npu", dtype=torch.int32)
        self.accepted = torch.ones((b,), device="npu", dtype=torch.int32)
        self.conv_out = torch.empty((b * l, 3 * h * d), device="npu", dtype=torch.bfloat16)
        self.core = torch.empty((1, b * l, h, d), device="npu", dtype=torch.bfloat16)
        self.normalized = torch.empty_like(self.core)

    def reset(self):
        self.cs.copy_(self.cs_base)

    def project(self):
        return F.linear(self.x, self.i["w_in"])

    def __call__(self):
        c, i = self.c, self.i
        b, l, h, d = c.batch, c.tokens, c.heads, c.dim
        projected = self.project()
        qkv, beta, fa, ga = projected.split([3 * h * d, h, d, d], -1)
        gate = F.linear(fa, i["w_f"]).reshape(1, b * l, h, d)
        g2 = F.linear(ga, i["w_g"]).reshape(b * l, h, d)
        convolved = self.conv(
            qkv,
            self.cs,
            self.cw,
            bias=None,
            activation="silu",
            conv_state_indices=self.indices,
            num_accepted_tokens=self.accepted,
            query_start_loc=self.starts,
            null_block_id=0,
            out=self.conv_out,
        )
        q, k, v = [t.reshape(1, b * l, h, d) for t in convolved.split(h * d, -1)]
        core = self.verify(
            q,
            k,
            v,
            gate,
            beta.unsqueeze(0),
            i["a_log"],
            i["dt_bias"],
            c.lower_bound,
            self.state,
            self.correction,
            self.kg,
            self.starts,
            self.indices,
            l,
            out=self.core,
        )
        norm = self.norm(core, g2, i["norm_w"], None, activation="sigmoid", eps=c.eps, out=self.normalized)
        output = F.linear(norm.reshape(b * l, h * d), i["w_o"])
        return output
